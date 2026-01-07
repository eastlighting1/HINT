import torch
import h5py
import json
import numpy as np
from pathlib import Path
from typing import Optional, List
from torch.utils.data import DataLoader
from ..common.base import BaseDomainService
from .trainer import ICDTrainer
from .evaluator import ICDEvaluator
from ....domain.entities import ICDModelEntity
from ....domain.vo import ICDConfig
from ....foundation.interfaces import TelemetryObserver, Registry, StreamingSource
from ....infrastructure.datasource import HDF5StreamingSource, collate_tensor_batch
from ....infrastructure.networks import get_network_class

class ICDService(BaseDomainService):
    """Service that trains and evaluates ICD models.

    Attributes:
        cfg (ICDConfig): ICD configuration.
        registry (Registry): Artifact registry.
        device (torch.device): Target device.
        ignored_indices (List[int]): Label indices to ignore.
    """
    def __init__(self, config: ICDConfig, registry: Registry, observer: TelemetryObserver, **kwargs):
        """Initialize the ICD service.

        Args:
            config (ICDConfig): ICD configuration.
            registry (Registry): Artifact registry.
            observer (TelemetryObserver): Logging observer.
            **kwargs (Any): Additional dependencies.
        """
        super().__init__(observer)
        self.cfg = config
        self.registry = registry
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ignored_indices = []

    def execute(self) -> None:
        """Run the ICD training and evaluation workflow."""
        cache_dir = Path(self.cfg.data.data_cache_dir)
        stats_path = cache_dir / "stats.json"
        
        if not stats_path.exists():
            self.observer.log("ERROR", f"Stats file not found at {stats_path}")
            return

        self.observer.log("INFO", "ICDService: Stage 1/4 loading class statistics from cache")
        with open(stats_path, "r") as f: 
            stats = json.load(f)
        
        num_classes = len(stats["icd_classes"])
        self.ignored_indices = [i for i, label in enumerate(stats["icd_classes"]) if label in ["__MISSING__", "__OTHER__"]]
        self.observer.log("INFO", f"ICDService: Loaded class metadata classes={num_classes}")
        
        prefix = self.cfg.data.input_h5_prefix
        
        self.observer.log("INFO", "ICDService: Stage 2/4 initializing HDF5 streaming sources")
        try:
            train_source = HDF5StreamingSource(cache_dir / f"{prefix}_train.h5", label_key="y")
            val_source = HDF5StreamingSource(cache_dir / f"{prefix}_val.h5", label_key="y")
            test_source = HDF5StreamingSource(cache_dir / f"{prefix}_test.h5", label_key="y")
        except Exception as e:
            self.observer.log("ERROR", f"Failed to initialize HDF5 sources: {e}. Check if ETL pipeline produced '{prefix}_*.h5' files.")
            return
        
        self.observer.log("INFO", "ICDService: Stage 3/4 computing class frequency statistics")
        try:
            with h5py.File(train_source.h5_path, "r") as f:
                shape = f["X_num"].shape
                num_feats, seq_len = shape[1], shape[2]
                
                if "y" in f:
                    y_data = f["y"][:]
                    if y_data.ndim == 1:
                        if y_data.dtype.kind == 'f':
                            y_data = y_data.astype(np.int64)
                        target_counts = np.bincount(y_data, minlength=num_classes)
                    else:
                        target_counts = np.sum(y_data, axis=0)
                else:
                    target_counts = np.ones(num_classes)
        except Exception as e:
            self.observer.log("ERROR", f"Failed to compute class stats: {e}")
            return

        self.observer.log("INFO", f"ICDService: Computed input shape num_feats={num_feats} seq_len={seq_len}")
        self.observer.log("INFO", "ICDService: Stage 4/4 building dataloaders")
        
        dl_tr = DataLoader(
            train_source, 
            batch_size=self.cfg.batch_size, 
            collate_fn=collate_tensor_batch, 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True,
            drop_last=True 
        )
        
        dl_val = DataLoader(val_source, batch_size=self.cfg.batch_size, collate_fn=collate_tensor_batch, num_workers=4, pin_memory=True)
        dl_test = DataLoader(test_source, batch_size=self.cfg.batch_size, collate_fn=collate_tensor_batch, num_workers=4, pin_memory=True)

        models_to_run = self.cfg.models_to_run if self.cfg.models_to_run else ["MedBERT"]

        for model_name in models_to_run:
            self.observer.log("INFO", f"========================================")
            self.observer.log("INFO", f"Starting workflow for model: {model_name}")
            self.observer.log("INFO", f"========================================")

            try:
                NetworkClass = get_network_class(model_name)
                network = NetworkClass(
                    num_classes=num_classes, 
                    input_dim=num_feats, 
                    seq_len=seq_len, 
                    dropout=self.cfg.dropout
                )
                
                entity_name = f"{self.cfg.artifacts.model_name}_{model_name}"
                model_entity = ICDModelEntity(network)
                model_entity.name = entity_name

                trainer = ICDTrainer(
                    config=self.cfg, 
                    entity=model_entity, 
                    registry=self.registry, 
                    observer=self.observer, 
                    device=self.device, 
                    class_freq=target_counts, 
                    ignored_indices=self.ignored_indices
                )
                
                evaluator = ICDEvaluator(
                    config=self.cfg, 
                    entity=model_entity, 
                    registry=self.registry, 
                    observer=self.observer, 
                    device=self.device, 
                    ignored_indices=self.ignored_indices
                )
                
                self.observer.log("INFO", f"[{model_name}] Entering training phase")
                trainer.train(dl_tr, dl_val, evaluator)
                
                self.observer.log("INFO", f"[{model_name}] Preparing best checkpoint for testing phase")
                
                try:
                    best_state = self.registry.load_model(entity_name, tag="best", device=str(self.device))
                    
                    if best_state is not None:
                        state_dict = best_state["model"] if "model" in best_state else best_state
                        model_entity.model.load_state_dict(state_dict)
                        model_entity.to(self.device)
                        self.observer.log("INFO", f"[{model_name}] Best checkpoint loaded.")
                    else:
                        self.observer.log("WARNING", f"[{model_name}] No best checkpoint found. Using current state.")
                        
                except RuntimeError as e:
                    if "size mismatch" in str(e):
                        if self.cfg.epochs > 0:
                            self.observer.log("WARNING", f"[{model_name}] Checkpoint size mismatch. Ignoring old checkpoint and using newly trained weights.")
                        else:
                            self.observer.log("ERROR", f"[{model_name}] Checkpoint size mismatch with 'epochs=0'. Cannot proceed with inference. Please set epochs > 0 to retrain.")
                            continue
                    else:
                        self.observer.log("ERROR", f"[{model_name}] Error loading checkpoint: {e}")
                except Exception as e:
                    self.observer.log("ERROR", f"[{model_name}] Unexpected error loading checkpoint: {e}")

                self.observer.log("INFO", f"[{model_name}] Entering test phase")
                test_metrics = evaluator.evaluate(dl_test)
                
                self.observer.log("INFO", f"[{model_name}] FINAL TEST RESULTS: {json.dumps(test_metrics, indent=2)}")
                
                self.observer.track_metric(f"{model_name}_test_acc", test_metrics.get("accuracy", 0.0), step=-1)
                self.observer.track_metric(f"{model_name}_test_loss", test_metrics.get("loss", 0.0), step=-1)

                self.observer.log("INFO", f"Successfully finished workflow for {model_name}")

            except Exception as e:
                self.observer.log("ERROR", f"Failed to run model {model_name}: {str(e)}")
                import traceback
                self.observer.log("ERROR", traceback.format_exc())
                continue

    def generate_intervention_dataset(self, cnn_config) -> None:
        """Generate intervention datasets using an ICD model.

        Args:
            cnn_config (Any): CNN configuration for output settings.
        """
        self.observer.log("INFO", "ICDService: Starting intervention dataset generation.")
        
        models_to_run = self.cfg.models_to_run if self.cfg.models_to_run else ["MedBERT"]
        if not models_to_run:
            self.observer.log("ERROR", "No models configured, cannot generate dataset.")
            return
        
        model_name = models_to_run[0]
        self.observer.log("INFO", f"ICDService: Selected model for generation model={model_name}")
        
        cache_dir = Path(self.cfg.data.data_cache_dir)
        stats_path = cache_dir / "stats.json"
        
        with open(stats_path, "r") as f: 
            stats = json.load(f)
        num_classes = len(stats["icd_classes"])
        
        icd_prefix = self.cfg.data.input_h5_prefix
        cnn_prefix = cnn_config.data.input_h5_prefix
        
        sample_file = cache_dir / f"{icd_prefix}_train.h5"
        if not sample_file.exists():
            self.observer.log("ERROR", f"Source file {sample_file} not found.")
            return
            
        with h5py.File(sample_file, "r") as f:
            shape = f["X_num"].shape
            num_feats, seq_len = shape[1], shape[2]

        try:
            NetworkClass = get_network_class(model_name)
            network = NetworkClass(
                num_classes=num_classes, 
                input_dim=num_feats, 
                seq_len=seq_len, 
                dropout=self.cfg.dropout
            )
            
            entity_name = f"{self.cfg.artifacts.model_name}_{model_name}"
            best_state = self.registry.load_model(entity_name, tag="best", device=str(self.device))
            
            if best_state:
                state = best_state["model"] if "model" in best_state else best_state
                network.load_state_dict(state)
            
            network.to(self.device)
            network.eval()
            
        except Exception as e:
            self.observer.log("ERROR", f"Failed to initialize model for generation: {e}")
            return

        splits = ["train", "val", "test"]
        
        for split in splits:
            src_path = cache_dir / f"{icd_prefix}_{split}.h5"
            tgt_path = cache_dir / f"{cnn_prefix}_{split}.h5"
            
            if not src_path.exists(): continue
                
            self.observer.log("INFO", f"Generating {tgt_path} from {src_path}...")
            source = HDF5StreamingSource(src_path, label_key="y")
            dl = DataLoader(source, batch_size=self.cfg.batch_size, collate_fn=collate_tensor_batch, num_workers=4)
            
            all_preds = []
            
            try:
                with torch.no_grad():
                    for batch in dl:
                        x_num = batch.x_num.to(self.device)
                        logits = network(x_num)
                        probs = torch.sigmoid(logits)
                        all_preds.append(probs.cpu().numpy())
                
                if not all_preds: continue
                full_preds = np.concatenate(all_preds, axis=0)
                
                with h5py.File(src_path, "r") as f_src, h5py.File(tgt_path, "w") as f_tgt:
                    for key in f_src.keys():
                        f_src.copy(key, f_tgt)
                    
                    if "X_icd" in f_tgt: del f_tgt["X_icd"]
                    f_tgt.create_dataset("X_icd", data=full_preds)
                    
                self.observer.log("INFO", f"Successfully created {tgt_path} with shape {full_preds.shape}")
                
            except Exception as e:
                self.observer.log("ERROR", f"Error processing split {split}: {e}")
                continue

        self.observer.log("INFO", "ICDService: Intervention dataset generation complete.")
