import torch
import numpy as np
from typing import List, Optional, Dict
from torch.utils.data import DataLoader
from ..common.base import BaseTrainer, BaseEvaluator
from ....domain.entities import ICDModelEntity
from ....domain.vo import ICDConfig
from ....infrastructure.components import CBFocalLoss, CLPLLoss
from ....foundation.dtos import TensorBatch

class ICDTrainer(BaseTrainer):
    def __init__(self, config: ICDConfig, entity: ICDModelEntity, registry, observer, device, class_freq: np.ndarray = None, ignored_indices: Optional[List[int]] = None):
        super().__init__(registry, observer, device)
        self.cfg = config
        self.entity = entity
        self.class_freq = class_freq if class_freq is not None else np.ones(1)
        self.ignored_indices = ignored_indices if ignored_indices else []
        
        # [FIX] Use new torch.amp API
        self.scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    def _prepare_inputs(self, batch: TensorBatch) -> Dict[str, torch.Tensor]:
        inputs = {}
        inputs["x_num"] = batch.x_num.to(self.device).float()
        
        if batch.mask is not None:
            inputs["mask"] = batch.mask.to(self.device).float()
        
        if hasattr(batch, "delta") and batch.delta is not None:
            inputs["delta"] = batch.delta.to(self.device).float()

        if getattr(batch, "input_ids", None) is not None:
            inputs["input_ids"] = batch.input_ids.to(self.device).long()
        
        if getattr(batch, "attention_mask", None) is not None:
            inputs["attention_mask"] = batch.attention_mask.to(self.device).long()
            
        if batch.x_cat is not None:
            inputs["x_cat"] = batch.x_cat.to(self.device).long()
            
        if batch.x_icd is not None:
            inputs["x_icd"] = batch.x_icd.to(self.device).float()

        return inputs

    def train(self, train_loader: DataLoader, val_loader: DataLoader, evaluator: BaseEvaluator) -> None:
        self.entity.to(self.device)
        self.observer.log("INFO", f"ICDTrainer: Start training for {self.cfg.epochs} epochs (AMP + PLL Mode: {self.cfg.loss_type}).")
        
        optimizer = torch.optim.Adam(self.entity.model.parameters(), lr=self.cfg.lr, weight_decay=1e-5)
        
        # [FIX] Removed 'verbose' argument
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=3
        )
        
        # Initialize Loss Function based on Config
        if self.cfg.loss_type == "clpl":
            loss_fn = CLPLLoss().to(self.device)
        else:
            loss_fn = CBFocalLoss(class_counts=self.class_freq, beta=self.cfg.cb_beta, gamma=self.cfg.focal_gamma, device=str(self.device))
        
        no_improve = 0
        
        for epoch in range(1, self.cfg.epochs + 1):
            self.entity.epoch = epoch
            self.entity.model.train()
            
            if hasattr(self.entity.model, "set_backbone_grad"):
                self.entity.model.set_backbone_grad(not (epoch <= self.cfg.freeze_bert_epochs))
            
            epoch_loss = 0.0
            batch_count = 0
            
            with self.observer.create_progress(f"Epoch {epoch} Train", total=len(train_loader)) as progress:
                task = progress.add_task("Training", total=len(train_loader))
                
                for batch in train_loader:
                    target = batch.y.to(self.device)
                    
                    valid_mask = torch.ones(target.shape[0], dtype=torch.bool, device=self.device)
                    if self.ignored_indices:
                        for idx in self.ignored_indices: valid_mask &= (target != idx)
                    if not valid_mask.any():
                        progress.advance(task)
                        continue
                    
                    target = target[valid_mask]
                    batch_inputs = self._prepare_inputs(batch)
                    filtered_inputs = {k: v[valid_mask] for k, v in batch_inputs.items()}

                    optimizer.zero_grad()
                    
                    # [FIX] Use new torch.amp API
                    with torch.amp.autocast('cuda', enabled=self.scaler is not None):
                        logits = self.entity.model(**filtered_inputs)
                        
                        # CLPL Mode Logic
                        if self.cfg.loss_type == "clpl" and getattr(batch, "candidates", None) is not None:
                            cands = batch.candidates.to(self.device).long()
                            cands = cands[valid_mask]
                            
                            # Create Multi-hot Candidate Mask (1 for candidates, 0 for others)
                            cand_mask = torch.zeros_like(logits, dtype=torch.float32)
                            valid_cands_idx = (cands >= 0) & (cands < logits.size(1))
                            rows = torch.arange(cands.size(0), device=self.device).unsqueeze(1).expand_as(cands)
                            
                            cand_mask[rows[valid_cands_idx], cands[valid_cands_idx]] = 1.0
                            
                            # Ensure ground truth is included in candidates (Safety check)
                            cand_mask.scatter_(1, target.unsqueeze(1), 1.0)
                            
                            if self.ignored_indices:
                                logits[:, self.ignored_indices] = -1e4

                            # Pass logits and mask to CLPL Loss (No forced masking of logits)
                            loss = loss_fn(logits, cand_mask)

                        else:
                            # Standard / Original PLL Logic (Masking Non-Candidates)
                            if getattr(batch, "candidates", None) is not None:
                                cands = batch.candidates.to(self.device).long()
                                cands = cands[valid_mask]
                                
                                cand_mask = torch.full_like(logits, fill_value=False, dtype=torch.bool)
                                
                                valid_cands_idx = (cands >= 0) & (cands < logits.size(1))
                                rows = torch.arange(cands.size(0), device=self.device).unsqueeze(1).expand_as(cands)
                                
                                cand_mask[rows[valid_cands_idx], cands[valid_cands_idx]] = True
                                cand_mask.scatter_(1, target.unsqueeze(1), True)
                                
                                # [FIX] Use -1e4 to prevent float16 overflow
                                logits = logits.masked_fill(~cand_mask, -1e4)

                            if self.ignored_indices: 
                                logits[:, self.ignored_indices] = -1e4
                            
                            loss = loss_fn(logits, target)

                    if self.scaler:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.entity.model.parameters(), max_norm=1.0)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.entity.model.parameters(), max_norm=1.0)
                        optimizer.step()
                    
                    epoch_loss += loss.item()
                    batch_count += 1
                    progress.advance(task)

            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
            self.observer.track_metric("train_loss", avg_loss, step=epoch)

            # Evaluation
            metrics = evaluator.evaluate(val_loader)
            val_acc = metrics.get("accuracy", 0.0)
            val_f1 = metrics.get("f1_macro", 0.0)
            
            self.observer.track_metric("icd_val_acc", val_acc, step=epoch)
            self.observer.log("INFO", f"Epoch {epoch} | TrLoss={avg_loss:.4f} | Acc={val_acc:.4f} | F1={val_f1:.4f}")

            scheduler.step(val_acc)

            if val_acc > self.entity.best_metric:
                self.entity.best_metric = val_acc
                self.registry.save_model(self.entity.state_dict(), self.entity.name, "best")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.cfg.patience: break
                    
        self.observer.log("INFO", "ICDTrainer: Training complete.")