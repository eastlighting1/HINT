# src/hint/services/training/predict_intervention/trainer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ..common.base import BaseTrainer, BaseEvaluator
from ....domain.entities import InterventionModelEntity
from ....domain.vo import CNNConfig

class InterventionTrainer(BaseTrainer):
    """Trainer for HINT model intervention prediction.

    This trainer handles optimization, focal loss computation, and
    early stopping based on validation metrics.

    Attributes:
        cfg (CNNConfig): Training configuration.
        entity (InterventionModelEntity): Trainable model entity.
        class_weights (Optional[torch.Tensor]): Optional class weighting tensor.
        gamma (float): Focal loss focusing parameter.
    """
    
    def __init__(self, config: CNNConfig, entity: InterventionModelEntity, registry, observer, device, class_weights=None):
        """Initialize the intervention trainer.

        Args:
            config (CNNConfig): Training configuration.
            entity (InterventionModelEntity): Wrapped model entity.
            registry (Any): Artifact registry for checkpointing.
            observer (Any): Telemetry observer for logging.
            device (str): Target training device.
            class_weights (Optional[torch.Tensor]): Optional class weights.
        """
        super().__init__(registry, observer, device)
        self.cfg = config
        self.entity = entity
        self.class_weights = class_weights
        self.gamma = 2.0

    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute class-weighted focal loss.

        Args:
            logits (torch.Tensor): Raw model logits.
            targets (torch.Tensor): Target class indices.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=-100)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.class_weights is not None:
            valid_mask = targets != -100
            weights = torch.ones_like(targets, dtype=torch.float)
            
            safe_targets = targets.clone()
            safe_targets[~valid_mask] = 0
            
            weights[valid_mask] = self.class_weights[safe_targets[valid_mask]]
            focal_loss = focal_loss * weights
            
        return focal_loss.mean()

    def _prepare_inputs(self, batch) -> dict:
        """Prepare model inputs from a TensorBatch.

        Args:
            batch (Any): Batch with x_num and optional x_icd.

        Returns:
            dict: Model input dictionary.
        """
        x_num = batch.x_num.to(self.device).float()
        
        inputs = {"x_num": x_num}
        
        # [수정] x_cat이 존재하면 inputs 딕셔너리에 추가
        if batch.x_cat is not None:
            inputs["x_cat"] = batch.x_cat.to(self.device).long()
        
        if batch.x_icd is not None:
            inputs["x_icd"] = batch.x_icd.to(self.device).float()
        else:
            inputs["x_icd"] = None
            
        return inputs

    def train(self, train_loader: DataLoader, val_loader: DataLoader, evaluator: BaseEvaluator) -> None:
        """Run the training loop with validation and early stopping.

        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            evaluator (BaseEvaluator): Evaluator for validation metrics.
        """
        self.entity.to(self.device)
        self.entity.network.train()
        
        optimizer = torch.optim.AdamW(self.entity.network.parameters(), lr=self.cfg.lr, weight_decay=1e-4)
        
        no_improve = 0
        self.observer.log("INFO", "InterventionTrainer: Starting training loop.")
        for epoch in range(1, self.cfg.epochs + 1):
            self.entity.epoch = epoch
            self.observer.log("INFO", f"InterventionTrainer: Epoch {epoch} start.")
            self._train_epoch(epoch, train_loader, optimizer)
            
            self.observer.log("INFO", f"InterventionTrainer: Epoch {epoch} validation start.")
            metrics = evaluator.evaluate(val_loader)
            val_f1 = metrics["f1_score"]
            val_loss = metrics.get("loss", 0.0)
            
            self.observer.log("INFO", f"Epoch {epoch} | Loss={val_loss:.4f} | F1={val_f1:.4f} | Acc={metrics['accuracy']:.4f}")
            
            if val_f1 > self.entity.best_metric:
                self.entity.best_metric = val_f1
                self.entity.update_ema()
                self.registry.save_model(self.entity.state_dict(), self.cfg.artifacts.model_name, "best")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.cfg.patience:
                    self.observer.log("WARNING", "Early stopping.")
                    break

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
        
        with self.observer.create_progress(f"Epoch {epoch} Train", total=len(loader)) as progress:
            task = progress.add_task("Training", total=len(loader))
            for batch in loader:
                optimizer.zero_grad()
                
                inputs = self._prepare_inputs(batch)
                y = batch.y.to(self.device)
                
                # [수정] datasource.py 수정으로 차원이 유지되므로, 이제 안전하게 인덱싱 가능
                target = y[:, -1] 
                
                logits = self.entity.network(**inputs)
                
                loss = self.focal_loss(logits, target)
                loss.backward()
                optimizer.step()
                self.entity.update_ema()
                
                total_loss += loss.item()
                steps += 1
                progress.advance(task)
        
        avg_loss = total_loss / max(1, steps)
        self.observer.track_metric("train_loss", avg_loss, step=epoch)