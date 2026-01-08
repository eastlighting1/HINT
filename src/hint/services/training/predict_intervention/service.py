import torch
import h5py
import numpy as np
import json
from typing import Optional
from torch.utils.data import DataLoader, Dataset
from ..common.base import BaseDomainService
from .trainer import InterventionTrainer
from .evaluator import InterventionEvaluator
from ....domain.entities import InterventionModelEntity
from ....domain.vo import CNNConfig
from ....foundation.interfaces import TelemetryObserver, Registry
from ....infrastructure.datasource import collate_tensor_batch, HDF5StreamingSource

class InterventionService(BaseDomainService):
    """Service that trains the intervention prediction model.

    This service builds data loaders, initializes trainers, and executes
    the training loop for intervention targets.

    Attributes:
        cfg (CNNConfig): Intervention configuration.
        registry (Registry): Artifact registry.
        entity (InterventionModelEntity): Model entity wrapper.
        train_ds (Dataset): Training dataset.
        val_ds (Dataset): Validation dataset.
        test_ds (Optional[Dataset]): Test dataset.
        device (str): Target device identifier.
    """
    def __init__(
        self,
        config: CNNConfig,
        registry: Registry,
        observer: TelemetryObserver,
        entity: InterventionModelEntity,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Optional[Dataset] = None
    ):
        """Initialize the service with datasets and dependencies.

        Args:
            config (CNNConfig): Intervention configuration.
            registry (Registry): Artifact registry.
            observer (TelemetryObserver): Logging observer.
            entity (InterventionModelEntity): Model entity wrapper.
            train_dataset (Dataset): Training dataset.
            val_dataset (Dataset): Validation dataset.
            test_dataset (Optional[Dataset]): Test dataset.
        """
        super().__init__(observer)
        self.cfg = config
        self.registry = registry
        self.entity = entity
        self.train_ds = train_dataset
        self.val_ds = val_dataset
        self.test_ds = test_dataset
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def execute(self) -> None:
        """Run the training workflow for intervention prediction."""
        self.observer.log("INFO", "InterventionService: Stage 1/4 building dataloaders.")
        
        # [NEW] Calculate Class Weights for Imbalanced Learning
        class_weights = None
        try:
            if isinstance(self.train_ds, HDF5StreamingSource):
                with h5py.File(self.train_ds.h5_path, 'r') as f:
                    # ICDService에서 y를 덮어썼으므로 'y'를 읽어야 함
                    label_key = self.train_ds.label_key
                    if label_key in f:
                        y_train = f[label_key][:]
                        if y_train.ndim > 1:
                            y_train = y_train.flatten()
                        
                        # Count frequencies
                        counts = np.bincount(y_train.astype(np.int64), minlength=4)
                        n_classes = len(counts)
                        total = np.sum(counts)
                        
                        # Inverse frequency weights
                        # weight = total / (n_classes * count)
                        weights = total / (n_classes * counts + 1e-6)
                        class_weights = torch.FloatTensor(weights).to(self.device)
                        
                        self.observer.log("INFO", f"Computed Class Weights: {weights}")
        except Exception as e:
            self.observer.log("WARNING", f"Failed to compute class weights: {e}")

        dl_tr = DataLoader(
            self.train_ds, 
            batch_size=self.cfg.batch_size, 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True,
            collate_fn=collate_tensor_batch 
        )
        dl_val = DataLoader(
            self.val_ds, 
            batch_size=self.cfg.batch_size, 
            shuffle=False, 
            num_workers=4,
            collate_fn=collate_tensor_batch
        )
        
        dl_test = None
        if self.test_ds:
            dl_test = DataLoader(
                self.test_ds,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=4,
                collate_fn=collate_tensor_batch
            )

        self.observer.log("INFO", "InterventionService: Stage 2/4 initializing trainer and evaluator.")
        
        # [MODIFIED] Pass class_weights to Trainer
        trainer = InterventionTrainer(
            self.cfg, 
            self.entity, 
            self.registry, 
            self.observer, 
            self.device,
            class_weights=class_weights
        )
        evaluator = InterventionEvaluator(self.cfg, self.entity, self.registry, self.observer, self.device)
        
        self.observer.log("INFO", "InterventionService: Stage 3/4 entering training loop.")
        trainer.train(dl_tr, dl_val, evaluator)

        self.observer.log("INFO", "InterventionService: Stage 4/4 entering test phase.")
        if dl_test:
            try:
                best_state = self.registry.load_model(self.cfg.artifacts.model_name, tag="best", device=str(self.device))
                
                if best_state is not None:
                    # entity.state_dict() is saved directly in trainer
                    self.entity.load_state_dict(best_state)
                    self.entity.to(self.device)
                    self.observer.log("INFO", "InterventionService: Best checkpoint loaded for testing.")
                else:
                    self.observer.log("WARNING", "InterventionService: No best checkpoint found. Using current state.")

                test_metrics = evaluator.evaluate(dl_test)
                self.observer.log("INFO", f"InterventionService: FINAL TEST RESULTS: {json.dumps(test_metrics, indent=2)}")
                
                for k, v in test_metrics.items():
                    self.observer.track_metric(f"cnn_test_{k}", v, step=-1)

            except Exception as e:
                self.observer.log("ERROR", f"InterventionService: Error during test phase: {e}")
        else:
            self.observer.log("WARNING", "InterventionService: No test dataset available. Skipping test phase.")