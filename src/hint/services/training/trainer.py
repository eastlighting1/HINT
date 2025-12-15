import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Optional

from ...foundation.interfaces import TelemetryObserver, Registry
from ...domain.entities import InterventionModelEntity
from ...domain.vo import CNNConfig
from ...infrastructure.components import FocalLoss
from ...infrastructure.datasource import collate_tensor_batch

class TrainingService:
    """
    Domain service for training the Intervention Prediction CNN using Pre-joined Datasets.

    Args:
        config: CNN configuration object.
        registry: Persistence registry.
        observer: Telemetry observer for logging and metrics.
        entity: Model entity wrapping network and EMA utilities.
        device: Target device string.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
    """
    def __init__(
        self,
        config: CNNConfig,
        registry: Registry,
        observer: TelemetryObserver,
        entity: InterventionModelEntity,
        device: str,
        train_dataset: Dataset,
        val_dataset: Dataset
    ):
        self.cfg = config
        self.registry = registry
        self.observer = observer
        self.entity = entity
        self.device = device
        self.loss_fn = FocalLoss(gamma=self.cfg.focal_gamma, label_smoothing=self.cfg.label_smoothing).to(device)
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train_model(self) -> None:
        self.entity.to(self.device)
        self.entity.network.train()
        
        optimizer = torch.optim.AdamW(self.entity.network.parameters(), lr=self.cfg.lr, weight_decay=1e-5)

        self.observer.log("INFO", f"TrainingService: Starting training for {self.cfg.epochs} epochs on {self.device}.")
        
        dl_tr = DataLoader(
            self.train_dataset, 
            batch_size=self.cfg.batch_size, 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True,
            collate_fn=collate_tensor_batch 
        )
        dl_val = DataLoader(
            self.val_dataset, 
            batch_size=self.cfg.batch_size, 
            shuffle=False, 
            num_workers=4,
            collate_fn=collate_tensor_batch
        )
        
        no_improve = 0
        
        for epoch in range(1, self.cfg.epochs + 1):
            self.entity.epoch = epoch
            self.observer.log("INFO", f"TrainingService: Epoch {epoch} training loop starting.")
            self._train_epoch(epoch, dl_tr, optimizer)
            
            val_acc = self._validate(dl_val)
            self.observer.track_metric("cnn_val_acc", val_acc, step=epoch)
            self.observer.log("INFO", f"TrainingService: Epoch {epoch} validation accuracy {val_acc:.4f}.")
            
            if val_acc > self.entity.best_metric + 1e-6:
                self.entity.best_metric = val_acc
                no_improve = 0
                self.entity.update_ema()
                self.registry.save_model(self.entity.state_dict(), "cnn_model", "best")
                self.observer.log("INFO", f"CNN Service: Saved best model acc={val_acc:.4f} at epoch {epoch}")
            else:
                no_improve += 1
                if no_improve >= self.cfg.patience:
                    self.observer.log("WARNING", "CNN Service: Early stopping triggered.")
                    break
        self.observer.log("INFO", f"TrainingService: Training complete with best validation accuracy {self.entity.best_metric:.4f}.")

    def _train_epoch(self, epoch: int, loader: DataLoader, optimizer: torch.optim.Optimizer) -> None:
        self.entity.network.train()
        total_loss = 0.0
        steps = 0
        
        with self.observer.create_progress(f"Epoch {epoch} CNN Train", total=len(loader)) as progress:
            task = progress.add_task("Training", total=len(loader))
            for batch in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()

                logits = self.entity.network(batch.x_num, batch.x_cat, batch.x_icd)
                
                loss = self.loss_fn(logits, batch.y)
                loss.backward()
                optimizer.step()
                self.entity.update_ema()
                
                total_loss += loss.item()
                steps += 1
                progress.advance(task)
        
        avg_loss = total_loss / max(1, steps)
        self.observer.track_metric("cnn_train_loss", avg_loss, step=epoch)

    def _validate(self, loader: DataLoader) -> float:
        self.entity.network.eval()
        correct = 0
        total = 0
        
        with self.entity.ema.average_parameters():
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(self.device)
                    
                    logits = self.entity.network(batch.x_num, batch.x_cat, batch.x_icd)
                    
                    preds = logits.argmax(dim=1)
                    
                    correct += (preds == batch.y).sum().item()
                    total += batch.y.size(0)
                
        return correct / max(1, total)
