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
    """Service orchestrating ICD model training, validation, and testing."""
    def __init__(self, config: ICDConfig, registry: Registry, observer: TelemetryObserver, **kwargs):
        super().__init__(observer)
        self.cfg = config
        self.registry = registry
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ignored_indices = []

    def execute(self) -> None:
        """Run the ICD training workflow across configured models."""
        cache_dir = Path(self.cfg.data.data_cache_dir)
        stats_path = cache_dir / "stats.json"
        
        if not stats_path.exists():
            self.observer.log("ERROR", f"Stats file not found at {stats_path}")
            return

        self.observer.log("INFO", "ICDService: Stage 1/4 loading class statistics")
        with open(stats_path, "r") as f: 
            stats = json.load(f)
        
        num_classes = len(stats["icd_classes"])
        self.ignored_indices = [i for i, label in enumerate(stats["icd_classes"]) if label in ["__MISSING__", "__OTHER__"]]
        
        prefix = self.cfg.data.input_h5_prefix
        
        self.observer.log("INFO", "ICDService: Stage 2/4 initializing HDF5 sources")
        train_source = HDF5StreamingSource(cache_dir / f"{prefix}_train.h5", label_key="y")
        val_source = HDF5StreamingSource(cache_dir / f"{prefix}_val.h5", label_key="y")
        test_source = HDF5StreamingSource(cache_dir / f"{prefix}_test.h5", label_key="y")
        
        self.observer.log("INFO", "ICDService: Stage 3/4 computing class frequencies")
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

        self.observer.log("INFO", "ICDService: Stage 4/4 building dataloaders and executing model loop")
        dl_tr = DataLoader(train_source, batch_size=self.cfg.batch_size, collate_fn=collate_tensor_batch, shuffle=True, num_workers=4, pin_memory=True)
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
                
                                                       
                self.observer.log("INFO", f"[{model_name}] Starting Training Phase...")
                trainer.train(dl_tr, dl_val, evaluator)
                
                                                
                self.observer.log("INFO", f"[{model_name}] Loading Best Checkpoint for Testing...")
                
                try:
                    best_state = self.registry.load_model(entity_name, tag="best", device=str(self.device))
                    
                    if best_state is not None:
                                                                                    
                        if "model" in best_state:
                            state_dict = best_state["model"]
                        else:
                            state_dict = best_state
                            
                        model_entity.model.load_state_dict(state_dict)
                        model_entity.to(self.device)
                        self.observer.log("INFO", f"[{model_name}] Successfully loaded best checkpoint.")
                    else:
                        self.observer.log("WARNING", f"[{model_name}] No best checkpoint returned (None). Using current state.")
                        
                except FileNotFoundError:
                    self.observer.log("WARNING", f"[{model_name}] Best checkpoint file not found. Using current state (last epoch).")
                except Exception as e:
                    self.observer.log("ERROR", f"[{model_name}] Error loading best checkpoint: {e}. Using current state.")

                                
                self.observer.log("INFO", f"[{model_name}] Starting Test Phase...")
                test_metrics = evaluator.evaluate(dl_test)
                
                                
                self.observer.log("INFO", f"[{model_name}] FINAL TEST RESULTS:")
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
