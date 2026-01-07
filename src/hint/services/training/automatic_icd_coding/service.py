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
    """Coordinate ICD training, evaluation, and downstream inference tasks.

    Manages dataset preparation, model lifecycle, and evaluation outputs for
    ICD coding workflows under a unified service interface.

    Attributes:
        cfg (ICDConfig): Training and data configuration for ICD workflows.
        registry (Registry): Artifact registry for checkpoints and metadata.
        device (torch.device): Device used for training and inference.
        ignored_indices (List[int]): Label indices excluded from training targets.
    """

    def __init__(self, config: ICDConfig, registry: Registry, observer: TelemetryObserver, **kwargs):
        """Initialize the ICD service with configuration and dependencies.

        Args:
            config (ICDConfig): ICD configuration payload.
            registry (Registry): Artifact registry for loading and saving assets.
            observer (TelemetryObserver): Telemetry adapter for logs and metrics.
            **kwargs: Additional dependencies reserved for future use.
        """
        super().__init__(observer)
        self.cfg = config
        self.registry = registry
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ignored_indices = []

    def execute(self) -> None:
        """Execute the ICD training workflow for all configured models.

        Loads cached statistics, prepares HDF5 sources, computes class weights,
        and runs training plus evaluation across each selected model.

        Returns:
            None: This method does not return a value.
        """
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
        self.observer.log("INFO", f"ICDService: Loaded class metadata classes={num_classes} ignored={len(self.ignored_indices)}")

        prefix = self.cfg.data.input_h5_prefix

        self.observer.log("INFO", "ICDService: Stage 2/4 initializing HDF5 streaming sources")
        train_source = HDF5StreamingSource(cache_dir / f"{prefix}_train.h5", label_key="y")
        val_source = HDF5StreamingSource(cache_dir / f"{prefix}_val.h5", label_key="y")
        test_source = HDF5StreamingSource(cache_dir / f"{prefix}_test.h5", label_key="y")

        self.observer.log("INFO", "ICDService: Stage 3/4 computing class frequency statistics from training data")
        try:
            with h5py.File(train_source.h5_path, "r") as f:
                shape = f["X_num"].shape
                num_feats, seq_len = shape[1], shape[2]
                
                if "y" in f:
                    y_data = f["y"][:]
                    target_counts = np.bincount(y_data, minlength=num_classes)
                else:
                    target_counts = np.ones(num_classes)
        except Exception as e:
            self.observer.log("ERROR", f"Failed to compute class stats: {e}")
            return

        self.observer.log("INFO", f"ICDService: Computed input shape num_feats={num_feats} seq_len={seq_len}")
        self.observer.log("INFO", "ICDService: Stage 4/4 building dataloaders and executing model loop")
        dl_tr = DataLoader(train_source, batch_size=self.cfg.batch_size, collate_fn=collate_tensor_batch, shuffle=True, num_workers=4, pin_memory=True)
        dl_val = DataLoader(val_source, batch_size=self.cfg.batch_size, collate_fn=collate_tensor_batch, num_workers=4, pin_memory=True)
        dl_test = DataLoader(test_source, batch_size=self.cfg.batch_size, collate_fn=collate_tensor_batch, num_workers=4, pin_memory=True)

        models_to_run = self.cfg.models_to_run if self.cfg.models_to_run else ["MedBERT"]

        self.observer.log("INFO", f"ICDService: Model selection resolved models={models_to_run}")
        for model_name in models_to_run:
            self.observer.log("INFO", f"========================================")
            self.observer.log("INFO", f"Starting workflow for model: {model_name}")
            self.observer.log("INFO", f"========================================")

            try:
                self.observer.log("INFO", f"[{model_name}] Resolving network class definition")
                NetworkClass = get_network_class(model_name)
                self.observer.log("INFO", f"[{model_name}] Initializing model with input_dim={num_feats} seq_len={seq_len}")
                network = NetworkClass(
                    num_classes=num_classes,
                    input_dim=num_feats,
                    seq_len=seq_len,
                    dropout=self.cfg.dropout
                )

                entity_name = f"{self.cfg.artifacts.model_name}_{model_name}"
                model_entity = ICDModelEntity(network)
                model_entity.name = entity_name

                self.observer.log("INFO", f"[{model_name}] Building trainer and evaluator")
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
                                                                                    
                        if "model" in best_state:
                            state_dict = best_state["model"]
                        else:
                            state_dict = best_state
                            
                        model_entity.model.load_state_dict(state_dict)
                        model_entity.to(self.device)
                        self.observer.log("INFO", f"[{model_name}] Best checkpoint loaded and applied")
                    else:
                        self.observer.log("WARNING", f"[{model_name}] No best checkpoint returned (None). Using current state.")

                except FileNotFoundError:
                    self.observer.log("WARNING", f"[{model_name}] Best checkpoint file not found. Using current state (last epoch).")
                except Exception as e:
                    self.observer.log("ERROR", f"[{model_name}] Error loading best checkpoint: {e}. Using current state.")

                self.observer.log("INFO", f"[{model_name}] Entering test phase")
                test_metrics = evaluator.evaluate(dl_test)

                self.observer.log("INFO", f"[{model_name}] Final test metrics summary")
                self.observer.log("INFO", json.dumps(test_metrics, indent=2))

                self.observer.track_metric(f"{model_name}_test_acc", test_metrics.get("accuracy", 0.0), step=-1)
                self.observer.track_metric(f"{model_name}_test_prec", test_metrics.get("precision", 0.0), step=-1)
                self.observer.track_metric(f"{model_name}_test_rec", test_metrics.get("recall", 0.0), step=-1)
                self.observer.track_metric(f"{model_name}_test_f1", test_metrics.get("f1_macro", 0.0), step=-1)
                self.observer.track_metric(f"{model_name}_test_loss", test_metrics.get("loss", 0.0), step=-1)

                self.observer.log("INFO", f"Successfully finished workflow for {model_name}")

            except Exception as e:
                self.observer.log("ERROR", f"Failed to run model {model_name}: {str(e)}")
                import traceback
                self.observer.log("ERROR", traceback.format_exc())
                continue

    def generate_intervention_dataset(self, cnn_config) -> None:
        """Generate intervention datasets by appending ICD predictions.

        Runs inference on each ICD HDF5 split and writes predictions into the
        corresponding CNN HDF5 files under the X_icd dataset.

        Args:
            cnn_config: Configuration for the downstream CNN stage.

        Returns:
            None: This method does not return a value.
        """
        self.observer.log("INFO", "ICDService: Starting intervention dataset generation")

        models_to_run = self.cfg.models_to_run if self.cfg.models_to_run else ["MedBERT"]
        if not models_to_run:
            self.observer.log("ERROR", "No models configured for ICD service, cannot generate dataset.")
            return

        model_name = models_to_run[0]
        self.observer.log("INFO", f"ICDService: Selected model for generation model={model_name}")

        cache_dir = Path(self.cfg.data.data_cache_dir)
        stats_path = cache_dir / "stats.json"

        if not stats_path.exists():
            self.observer.log("ERROR", f"Stats file not found at {stats_path}")
            return

        self.observer.log("INFO", "ICDService: Loading class metadata for inference")
        with open(stats_path, "r") as f:
            stats = json.load(f)
        num_classes = len(stats["icd_classes"])

        icd_prefix = self.cfg.data.input_h5_prefix
        cnn_prefix = cnn_config.data.input_h5_prefix

        sample_file = cache_dir / f"{icd_prefix}_train.h5"
        if not sample_file.exists():
            self.observer.log("ERROR", f"Source file {sample_file} not found.")
            return

        self.observer.log("INFO", f"ICDService: Inspecting sample file for input shape path={sample_file}")
        with h5py.File(sample_file, "r") as f:
            shape = f["X_num"].shape
            num_feats, seq_len = shape[1], shape[2]

        try:
            self.observer.log("INFO", f"ICDService: Initializing model for inference input_dim={num_feats} seq_len={seq_len}")
            NetworkClass = get_network_class(model_name)
            network = NetworkClass(
                num_classes=num_classes,
                input_dim=num_feats,
                seq_len=seq_len,
                dropout=self.cfg.dropout
            )

            entity_name = f"{self.cfg.artifacts.model_name}_{model_name}"
            best_state = self.registry.load_model(entity_name, tag="best", device=str(self.device))

            if best_state is None:
                self.observer.log("ERROR", f"Could not load best model checkpoint for {model_name}. Aborting generation.")
                return

            if "model" in best_state:
                network.load_state_dict(best_state["model"])
            else:
                network.load_state_dict(best_state)
            
            network.to(self.device)
            network.eval()

        except Exception as e:
            self.observer.log("ERROR", f"Failed to initialize model for generation: {e}")
            return

        splits = ["train", "val", "test"]

        for split in splits:
            src_path = cache_dir / f"{icd_prefix}_{split}.h5"
            tgt_path = cache_dir / f"{cnn_prefix}_{split}.h5"

            if not src_path.exists():
                self.observer.log("WARNING", f"Source file {src_path} does not exist. Skipping.")
                continue

            self.observer.log("INFO", f"ICDService: Generating split={split} src={src_path} tgt={tgt_path}")
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
                
                if not all_preds:
                    self.observer.log("WARNING", f"No data processed for {split}.")
                    continue

                full_preds = np.concatenate(all_preds, axis=0)

                with h5py.File(src_path, "r") as f_src, h5py.File(tgt_path, "w") as f_tgt:
                    for key in f_src.keys():
                        f_src.copy(key, f_tgt)

                    if "X_icd" in f_tgt:
                        del f_tgt["X_icd"]

                    f_tgt.create_dataset("X_icd", data=full_preds)
                    
                self.observer.log("INFO", f"Successfully created {tgt_path} with shape {full_preds.shape}")
                
            except Exception as e:
                self.observer.log("ERROR", f"Error processing split {split}: {e}")
                continue

        self.observer.log("INFO", "ICDService: Intervention dataset generation complete.")
