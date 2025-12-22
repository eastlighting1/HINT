import torch
import h5py
import json
import numpy as np
from pathlib import Path
from typing import Optional, List
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from sklearn.preprocessing import LabelEncoder

from ..common.base import BaseDomainService
from .trainer import ICDTrainer
from .evaluator import ICDEvaluator
from ....domain.entities import ICDModelEntity
from ....domain.vo import ICDConfig, CNNConfig
from ....foundation.interfaces import TelemetryObserver, Registry, StreamingSource
from ....infrastructure.datasource import HDF5StreamingSource, collate_tensor_batch
from ....infrastructure.networks import get_network_class

class ICDService(BaseDomainService):
    def __init__(
        self, 
        config: ICDConfig,
        registry: Registry, 
        observer: TelemetryObserver,
        train_source: Optional[StreamingSource] = None,
        val_source: Optional[StreamingSource] = None,
        test_source: Optional[StreamingSource] = None
    ):
        super().__init__(observer)
        self.cfg = config
        self.registry = registry
        self.train_source = train_source 
        self.val_source = val_source
        self.test_source = test_source
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.entity: Optional[ICDModelEntity] = None
        self.le = None
        self.num_feats = 0
        self.ts_seq_len = 0 
        self.feats = []
        self.ignored_indices: List[int] = []

    def _prepare_data(self):
        self.observer.log("INFO", "ICD Service: Loading metadata from cache.")
        cache_dir = Path(self.cfg.data.data_cache_dir)
        stats_path = cache_dir / "stats.json"
        
        if not stats_path.exists():
            raise FileNotFoundError(f"Stats file not found at {stats_path}. Run ETL TensorConverter first.")

        with open(stats_path, "r") as f:
            stats = json.load(f)
            
        if "icd_classes" in stats:
            self.le = LabelEncoder()
            self.le.classes_ = np.array(stats["icd_classes"])
            self.observer.log("INFO", f"ICD Service: Loaded {len(self.le.classes_)} classes from metadata.")
            
            excluded_labels = ["__MISSING__", "__OTHER__"]
            self.ignored_indices = [
                i for i, label in enumerate(self.le.classes_) 
                if label in excluded_labels
            ]
            if self.ignored_indices:
                self.observer.log("INFO", f"ICD Service: Ignoring classes {excluded_labels} at indices {self.ignored_indices}")
        else:
            raise ValueError("icd_classes not found in stats.json")

        self.num_feats = 0
        self.ts_seq_len = 0
        
        if isinstance(self.train_source, HDF5StreamingSource):
             _ = self.train_source[0] 
             if "X_num" in self.train_source.h5_file:
                shape = self.train_source.h5_file["X_num"].shape
                self.num_feats = shape[1]
                self.ts_seq_len = shape[2] 
                self.feats = [f"feat_{i}" for i in range(self.num_feats)]
                self.observer.log("INFO", f"ICD Service: Inferred Input Dim={self.num_feats}, Seq Len={self.ts_seq_len} from HDF5.")

        return self.feats, "y", self.le

    def execute(self) -> None:
        feats, label_col, le = self._prepare_data()
        
        if not isinstance(self.train_source, HDF5StreamingSource):
             prefix = self.cfg.data.input_h5_prefix
             cache_dir = Path(self.cfg.data.data_cache_dir)
             self.train_source = HDF5StreamingSource(cache_dir / f"{prefix}_train.h5", label_key="y")
             self.val_source = HDF5StreamingSource(cache_dir / f"{prefix}_val.h5", label_key="y")
             
             if self.num_feats == 0:
                 with h5py.File(self.train_source.h5_path, "r") as f:
                     shape = f["X_num"].shape
                     self.num_feats = shape[1]
                     self.ts_seq_len = shape[2]

        num_feats = self.num_feats
        seq_len_for_model = self.ts_seq_len if self.ts_seq_len > 0 else self.cfg.max_length
        num_classes = len(le.classes_)
        
        subset_ratio = self.cfg.execution.subset_ratio if hasattr(self.cfg, "execution") else 1.0
        self.observer.log("INFO", f"ICD Service: Subsampling ratio set to {subset_ratio * 100:.1f}%.")

        self.observer.log("INFO", "ICD Service: Calculating sampler weights...")
        with h5py.File(self.train_source.h5_path, "r") as f:
             y_all = f["y"][:]
        
        target_counts = np.bincount(y_all)
        class_weights_map = {}
        for cls, count in enumerate(target_counts):
            if cls in self.ignored_indices:
                class_weights_map[cls] = 0.0
            elif count > 0:
                class_weights_map[cls] = 1.0 / (count ** self.cfg.sampler_alpha)
            else:
                class_weights_map[cls] = 1.0

        sample_weights = torch.tensor([class_weights_map.get(x, 1.0) for x in y_all], dtype=torch.double)
        
        num_train_samples = int(len(sample_weights) * subset_ratio)
        sampler = WeightedRandomSampler(
            weights=sample_weights, 
            num_samples=num_train_samples, 
            replacement=True
        )
        collate_fn = collate_tensor_batch

        dl_tr = DataLoader(
            self.train_source, 
            batch_size=self.cfg.batch_size, 
            sampler=sampler, 
            collate_fn=collate_fn, 
            num_workers=self.cfg.num_workers
        )

        if subset_ratio < 1.0:
            val_indices = np.random.choice(
                len(self.val_source), 
                int(len(self.val_source) * subset_ratio), 
                replace=False
            )
            val_indices.sort()
            val_subset = Subset(self.val_source, val_indices)
            dl_val = DataLoader(val_subset, batch_size=self.cfg.batch_size, collate_fn=collate_fn, num_workers=self.cfg.num_workers)
        else:
            dl_val = DataLoader(self.val_source, batch_size=self.cfg.batch_size, collate_fn=collate_fn, num_workers=self.cfg.num_workers)

        models_to_run = getattr(self.cfg, "models_to_run", ["MedBERT"])
        self.observer.log("INFO", f"ICD Service: Models scheduled for execution: {models_to_run}")

        for model_name in models_to_run:
            self.observer.log("INFO", f"=== Starting Training Loop for Model: {model_name} ===")
            
            specific_cfg = self.cfg.model_configs.get(model_name, {}) if hasattr(self.cfg, "model_configs") else {}
            
            network_args = {
                "num_classes": num_classes,
                "input_dim": num_feats,
                "seq_len": seq_len_for_model, 
                "dropout": self.cfg.dropout,
                **specific_cfg
            }
            
            try:
                NetworkClass = get_network_class(model_name)
                network = NetworkClass(**network_args)
                
                # [Removed] Stacker initialization
                self.entity = ICDModelEntity(network)
                self.entity.name = f"{self.cfg.artifacts.model_name}_{model_name}"

                trainer = ICDTrainer(
                    self.cfg, self.entity, self.registry, self.observer, self.device, 
                    target_counts, ignored_indices=self.ignored_indices
                )
                evaluator = ICDEvaluator(
                    self.cfg, self.entity, self.registry, self.observer, self.device,
                    ignored_indices=self.ignored_indices
                )
                
                trainer.train(dl_tr, dl_val, evaluator)
                self.observer.log("INFO", f"=== Finished Training Model: {model_name} ===")
                
            except Exception as e:
                self.observer.log("ERROR", f"Failed to train model {model_name}: {e}")
                import traceback
                self.observer.log("ERROR", traceback.format_exc())

    def _ensure_entity_loaded(self) -> bool:
        return self.entity is not None

    def generate_intervention_dataset(self, cnn_config: CNNConfig) -> None:
        self.observer.log("INFO", "ICD Service: generate_intervention_dataset called but skipped (as per strategy).")
        return