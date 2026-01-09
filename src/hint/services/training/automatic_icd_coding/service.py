# src/hint/services/training/automatic_icd_coding/trainer.py

import torch
import torch.nn as nn
import numpy as np
import random
from typing import List, Optional, Dict
from torch.utils.data import DataLoader
from ..common.base import BaseTrainer, BaseEvaluator
from ....domain.entities import ICDModelEntity
from ....domain.vo import ICDConfig
from ....foundation.dtos import TensorBatch

class ICDTrainer(BaseTrainer):
    """Trainer for ICD classification with partial label learning.

    This trainer uses adaptive softmax with candidate sampling for
    partial label learning and tracks validation accuracy.

    Attributes:
        cfg (ICDConfig): Training configuration.
        entity (ICDModelEntity): Wrapped model entity.
        class_freq (Optional[np.ndarray]): Optional class frequency vector.
        scaler (Optional[torch.amp.GradScaler]): AMP gradient scaler.
        adaptive_loss_fn (Any): Adaptive softmax loss function.
    """
    
    def __init__(self, config: ICDConfig, entity: ICDModelEntity, registry, observer, device, class_freq: np.ndarray = None, ignored_indices: Optional[List[int]] = None):
        """Initialize the ICD trainer.

        Args:
            config (ICDConfig): Training configuration.
            entity (ICDModelEntity): Wrapped model entity.
            registry (Any): Artifact registry instance.
            observer (Any): Telemetry observer.
            device (str): Training device.
            class_freq (Optional[np.ndarray]): Optional class frequency vector.
            ignored_indices (Optional[List[int]]): Unused indices reserved for filtering.
        """
        super().__init__(registry, observer, device)
        self.cfg = config
        self.entity = entity
        self.class_freq = class_freq
        self.scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        
        self.adaptive_loss_fn = self.entity.model.adaptive_head

    def _prepare_inputs(self, batch: TensorBatch) -> Dict[str, torch.Tensor]:
        """Prepare inputs from a TensorBatch.

        Args:
            batch (TensorBatch): Batch containing x_num (and optionally x_cat).

        Returns:
            Dict[str, torch.Tensor]: Dictionary with model inputs.
        """
        inputs = {}
        
        # FIX: Use x_num directly as it is now pre-concatenated by the loader
        if hasattr(batch, 'x_num') and batch.x_num is not None:
            inputs["x_num"] = batch.x_num.to(self.device).float()
        else:
            # Fallback logic only if x_num is missing (should not happen with correct loader)
            x_val = batch.x_val.to(self.device).float()
            x_msk = batch.x_msk.to(self.device).float()
            x_delta = batch.x_delta.to(self.device).float()
            inputs["x_num"] = torch.cat([x_val, x_msk, x_delta], dim=1)
        
        # Support categorical features if present
        if batch.x_cat is not None:
            inputs["x_cat"] = batch.x_cat.to(self.device).long()
        
        return inputs

    def _sample_target_from_candidates(self, candidates: torch.Tensor) -> torch.Tensor:
        """Sample one valid label from each candidate set.

        Args:
            candidates (torch.Tensor): Candidate label indices.

        Returns:
            torch.Tensor: Sampled target labels.
        """
        targets = []
        candidates_np = candidates.cpu().numpy()
        for row in candidates_np:
            valid_cands = row[row >= 0]
            if len(valid_cands) > 0:
                targets.append(np.random.choice(valid_cands))
            else:
                targets.append(0)
        return torch.tensor(targets, device=self.device, dtype=torch.long)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, evaluator: BaseEvaluator) -> None:
        """Run the training loop with validation.

        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            evaluator (BaseEvaluator): Evaluator instance.
        """
        self.entity.to(self.device)
        self.entity.model.train()
        params = list(self.entity.model.parameters())
        optimizer = torch.optim.Adam(params, lr=self.cfg.lr)
        
        self.observer.log("INFO", "ICDTrainer: Starting training loop.")
        for epoch in range(1, self.cfg.epochs + 1):
            self.entity.epoch = epoch
            epoch_loss = 0.0
            steps = 0
            
            with self.observer.create_progress(f"Epoch {epoch} ICD Train", total=len(train_loader)) as progress:
                task = progress.add_task("Training", total=len(train_loader))
                
                for batch in train_loader:
                    optimizer.zero_grad()
                    inputs = self._prepare_inputs(batch)
                    
                    candidates = batch.candidates
                    target = self._sample_target_from_candidates(candidates)
                    
                    with torch.amp.autocast('cuda', enabled=self.scaler is not None):
                        embeddings = self.entity.model(**inputs, return_embeddings=True)
                        _, loss = self.adaptive_loss_fn(embeddings, target)
                    
                    if self.scaler:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                        
                    epoch_loss += loss.item()
                    steps += 1
                    progress.advance(task)
            
            avg_loss = epoch_loss / max(1, steps)
            self.observer.track_metric("icd_train_loss", avg_loss, step=epoch)
            
            self.observer.log("INFO", f"ICDTrainer: Epoch {epoch} validation start.")
            metrics = evaluator.evaluate(val_loader)
            val_acc = metrics.get("accuracy", 0.0)
            self.observer.log("INFO", f"Epoch {epoch} | Loss={avg_loss:.4f} | Val Acc={val_acc:.4f}")
            
            if val_acc > self.entity.best_metric:
                self.entity.best_metric = val_acc
                self.registry.save_model(self.entity.model.state_dict(), self.entity.name, "best")