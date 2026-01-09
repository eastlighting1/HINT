import torch
import h5py
import numpy as np
import json
from typing import Optional
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from ..common.base import BaseDomainService

# [수정] 순환 참조 방지를 위해 Top-level import 제거
# from .trainer import InterventionTrainer
# from .evaluator import InterventionEvaluator

from ....domain.entities import InterventionModelEntity
from ....domain.vo import CNNConfig
from ....foundation.interfaces import TelemetryObserver, Registry
from ....infrastructure.datasource import collate_tensor_batch, HDF5StreamingSource
from ....infrastructure.networks import get_network_class

class InterventionService(BaseDomainService):
    """Service that trains the HINT model for intervention prediction."""
    
    def __init__(self, config: CNNConfig, registry: Registry, observer: TelemetryObserver, entity: InterventionModelEntity, train_dataset: Dataset, val_dataset: Dataset, test_dataset: Optional[Dataset] = None):
        super().__init__(observer)
        self.cfg = config
        self.registry = registry
        self.entity = entity
        self.train_ds = train_dataset
        self.val_ds = val_dataset
        self.test_ds = test_dataset
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def execute(self) -> None:
        """Execute the full intervention training workflow."""
        # [수정] 메서드 내부에서 임포트 (Lazy Import)
        from .trainer import InterventionTrainer
        from .evaluator import InterventionEvaluator

        self.observer.log("INFO", "InterventionService: Stage 1/4 building dataloaders and class weights.")

        class_weights = None
        try:
            if isinstance(self.train_ds, HDF5StreamingSource):
                with h5py.File(self.train_ds.h5_path, 'r') as f:
                    label_key = self.train_ds.label_key
                    if label_key in f:
                        y_train = f[label_key][:]
                        if y_train.ndim > 1:
                            y_train = y_train.flatten()
                        
                        valid_mask = y_train != -100
                        y_valid = y_train[valid_mask].astype(np.int64)
                        
                        if len(y_valid) > 0:
                            classes = np.unique(y_valid)
                            weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_valid)
                            weight_map = {c: w for c, w in zip(classes, weights)}
                            final_weights = [weight_map.get(i, 1.0) for i in range(4)]
                            class_weights = torch.FloatTensor(final_weights).to(self.device)
                            class_weights = torch.clamp(class_weights, max=20.0)
                            self.observer.log("INFO", f"Computed Class Weights: {class_weights.tolist()}")
        except Exception as e:
            self.observer.log("WARNING", f"Failed to compute class weights: {e}")

        dl_tr = DataLoader(self.train_ds, batch_size=self.cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_tensor_batch)
        dl_val = DataLoader(self.val_ds, batch_size=self.cfg.batch_size, shuffle=False, num_workers=4, collate_fn=collate_tensor_batch)
        dl_test = None
        if self.test_ds:
            dl_test = DataLoader(self.test_ds, batch_size=self.cfg.batch_size, shuffle=False, num_workers=4, collate_fn=collate_tensor_batch)

        self.observer.log("INFO", "InterventionService: Stage 2/4 initializing HINT model and input dimensions.")

        with h5py.File(self.train_ds.h5_path, 'r') as f:
            # [수정] X_val 대신 X_num 사용
            num_feats = f["X_num"].shape[1]
            
            icd_dim = 0
            if "X_icd" in f:
                icd_dim = f["X_icd"].shape[1]
                self.observer.log("INFO", f"Found ICD context with dim={icd_dim}")
            else:
                self.observer.log("WARNING", "X_icd not found in dataset. Gating disabled.")

        # [수정] "HINT" 대신 "TCN" 사용
        NetworkClass = get_network_class("TCN")
        
        # [수정] TCNClassifier 인자 호환성 보장
        vocab_sizes = []
        if hasattr(self.train_ds, "get_real_vocab_sizes"):
             vocab_sizes = self.train_ds.get_real_vocab_sizes()

        network = NetworkClass(
            in_chs=num_feats,
            n_cls=4,
            vocab_sizes=vocab_sizes,
            icd_dim=icd_dim,
            embed_dim=self.cfg.embed_dim if hasattr(self.cfg, "embed_dim") else 128,
            cat_embed_dim=self.cfg.cat_embed_dim if hasattr(self.cfg, "cat_embed_dim") else 32,
            head_drop=self.cfg.dropout,
            tcn_drop=self.cfg.tcn_dropout if hasattr(self.cfg, "tcn_dropout") else 0.2,
            kernel=self.cfg.tcn_kernel_size if hasattr(self.cfg, "tcn_kernel_size") else 5,
            layers=self.cfg.tcn_layers if hasattr(self.cfg, "tcn_layers") else 5
        )
        
        entity_name = self.cfg.artifacts.model_name
        self.entity = InterventionModelEntity(network)
        self.entity.name = entity_name
        
        trainer = InterventionTrainer(self.cfg, self.entity, self.registry, self.observer, self.device, class_weights=class_weights)
        evaluator = InterventionEvaluator(self.cfg, self.entity, self.registry, self.observer, self.device)
        
        self.observer.log("INFO", "InterventionService: Stage 3/4 entering training loop.")
        trainer.train(dl_tr, dl_val, evaluator)

        self.observer.log("INFO", "InterventionService: Stage 4/4 entering test phase.")
        if dl_test:
            best_state = self.registry.load_model(entity_name, tag="best", device=str(self.device))
            if best_state:
                self.entity.load_state_dict(best_state)
                self.entity.to(self.device)

            metrics = evaluator.evaluate(dl_test)
            self.observer.log("INFO", f"Final Test Results: {json.dumps(metrics, indent=2)}")
        else:
            self.observer.log("WARNING", "InterventionService: No test dataset available, skipping test evaluation.")