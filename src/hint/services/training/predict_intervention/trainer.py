import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ..common.base import BaseTrainer, BaseEvaluator
from ....domain.entities import InterventionModelEntity
from ....domain.vo import CNNConfig

class InterventionTrainer(BaseTrainer):
    """Trainer for intervention prediction models.

    Attributes:
        cfg (CNNConfig): Training configuration.
        entity (InterventionModelEntity): Model entity wrapper.
        loss_fn (nn.Module): Loss function.
    """
    def __init__(self, config: CNNConfig, entity: InterventionModelEntity, registry, observer, device, class_weights=None):
        """Initialize the intervention trainer.

        Args:
            config (CNNConfig): Training configuration.
            entity (InterventionModelEntity): Model entity wrapper.
            registry (Any): Artifact registry.
            observer (Any): Logging observer.
            device (Any): Target device.
            class_weights (torch.Tensor, optional): Pre-computed class weights for handling imbalance.
        """
        super().__init__(registry, observer, device)
        self.cfg = config
        self.entity = entity
        
        # [MODIFIED] Use Weighted CrossEntropyLoss to handle imbalance effectively
        if class_weights is not None:
            self.observer.log("INFO", "InterventionTrainer: Using Weighted CrossEntropyLoss.")
            self.loss_fn = nn.CrossEntropyLoss(
                weight=class_weights, 
                label_smoothing=self.cfg.label_smoothing
            ).to(device)
        else:
            self.loss_fn = nn.CrossEntropyLoss(
                label_smoothing=self.cfg.label_smoothing
            ).to(device)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, evaluator: BaseEvaluator) -> None:
        """Run the training loop with periodic evaluation.

        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            evaluator (BaseEvaluator): Evaluator instance.
        """
        self.entity.to(self.device)
        self.entity.network.train()
        
        optimizer = torch.optim.AdamW(self.entity.network.parameters(), lr=self.cfg.lr, weight_decay=1e-5)
        
        no_improve = 0
        for epoch in range(1, self.cfg.epochs + 1):
            self.entity.epoch = epoch
            self.observer.log("INFO", f"InterventionTrainer: Epoch {epoch} training loop starting.")
            self._train_epoch(epoch, train_loader, optimizer)
            
            metrics = evaluator.evaluate(val_loader)
            val_acc = metrics["accuracy"]
            # It's often better to track F1 or loss when handling imbalanced data
            val_loss = metrics.get("loss", 0.0) 
            
            self.observer.track_metric("cnn_val_acc", val_acc, step=epoch)
            self.observer.log("INFO", f"InterventionTrainer: Epoch {epoch} val_acc={val_acc:.4f}")
            
            if val_acc > self.entity.best_metric + 1e-6:
                self.entity.best_metric = val_acc
                no_improve = 0
                self.entity.update_ema()
                self.registry.save_model(self.entity.state_dict(), self.cfg.artifacts.model_name, "best")
                self.observer.log("INFO", f"InterventionTrainer: Saved best model acc={val_acc:.4f} at epoch {epoch}")
            else:
                no_improve += 1
                if no_improve >= self.cfg.patience:
                    self.observer.log("WARNING", "InterventionTrainer: Early stopping triggered.")
                    break
        self.observer.log("INFO", f"InterventionTrainer: Training complete with best validation accuracy {self.entity.best_metric:.4f}.")

    def _train_epoch(self, epoch: int, loader: DataLoader, optimizer: torch.optim.Optimizer) -> None:
        """Train for a single epoch.

        Args:
            epoch (int): Epoch index.
            loader (DataLoader): Training data loader.
            optimizer (torch.optim.Optimizer): Optimizer instance.
        """
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