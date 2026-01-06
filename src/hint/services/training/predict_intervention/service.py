import torch
from torch.utils.data import DataLoader, Dataset
from ..common.base import BaseDomainService
from .trainer import InterventionTrainer
from .evaluator import InterventionEvaluator
from ....domain.entities import InterventionModelEntity
from ....domain.vo import CNNConfig
from ....foundation.interfaces import TelemetryObserver, Registry
from ....infrastructure.datasource import collate_tensor_batch

class InterventionService(BaseDomainService):
    def __init__(
        self,
        config: CNNConfig,
        registry: Registry,
        observer: TelemetryObserver,
        entity: InterventionModelEntity,
        train_dataset: Dataset,
        val_dataset: Dataset
    ):
        super().__init__(observer)
        self.cfg = config
        self.registry = registry
        self.entity = entity
        self.train_ds = train_dataset
        self.val_ds = val_dataset
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def execute(self) -> None:
        """Execute the intervention prediction training workflow."""
        self.observer.log("INFO", "InterventionService: Stage 1/3 building dataloaders.")
        
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

        self.observer.log("INFO", "InterventionService: Stage 2/3 initializing trainer and evaluator.")
        trainer = InterventionTrainer(self.cfg, self.entity, self.registry, self.observer, self.device)
        evaluator = InterventionEvaluator(self.cfg, self.entity, self.registry, self.observer, self.device)
        
        self.observer.log("INFO", "InterventionService: Stage 3/3 entering training loop.")
        trainer.train(dl_tr, dl_val, evaluator)
