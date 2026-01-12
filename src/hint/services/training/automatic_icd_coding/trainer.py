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

class CLPLLoss(nn.Module):
    """Convex Loss for Partial Labels (CLPL) for ICD coding."""
    def __init__(self, loss_type: str = "exponential"):
        super().__init__()
        self.loss_type = loss_type

    def psi(self, u: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "exponential":
            return torch.exp(-u)
        if self.loss_type == "logistic":
            return torch.log1p(torch.exp(-u))
        if self.loss_type == "hinge":
            return torch.clamp(1 - u, min=0)
        raise ValueError(f"Unknown loss type: {self.loss_type}")

    def forward(self, logits: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        if candidates is None:
            raise ValueError("CLPLLoss requires candidate sets.")

        batch_size, num_classes = logits.shape
        candidates = candidates.to(logits.device).long()
        valid = candidates >= 0

        y_mask = torch.zeros((batch_size, num_classes), device=logits.device, dtype=logits.dtype)
        if valid.any():
            row_ids = torch.arange(batch_size, device=logits.device).unsqueeze(1).expand_as(candidates)
            y_mask[row_ids[valid], candidates[valid]] = 1.0

        y_cardinality = torch.clamp(y_mask.sum(dim=1), min=1.0)
        avg_candidate_score = (logits * y_mask).sum(dim=1) / y_cardinality
        term1 = self.psi(avg_candidate_score)

        non_candidate_mask = 1.0 - y_mask
        term2 = (self.psi(-logits) * non_candidate_mask).sum(dim=1)

        return (term1 + term2).mean()

class AdaptiveCLPLLoss(nn.Module):
    """Adaptive CLPL that approximates tail negatives via sampling."""
    def __init__(
        self,
        head_size: int = 2000,
        tail_sample_size: int = 100,
        loss_type: str = "logistic",
        logit_clip: float = 20.0,
    ):
        super().__init__()
        self.head_size = head_size
        self.tail_sample_size = tail_sample_size
        self.loss_type = loss_type
        self.logit_clip = logit_clip

    def psi(self, u: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "exponential":
            return torch.exp(-u)
        if self.loss_type == "logistic":
            return torch.log1p(torch.exp(-u))
        if self.loss_type == "hinge":
            return torch.clamp(1 - u, min=0)
        raise ValueError(f"Unknown loss type: {self.loss_type}")

    def forward(self, logits: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        if candidates is None:
            raise ValueError("AdaptiveCLPLLoss requires candidate sets.")

        if self.logit_clip is not None:
            logits = logits.clamp(-self.logit_clip, self.logit_clip)

        batch_size, num_classes = logits.shape
        candidates = candidates.to(logits.device).long()
        valid = candidates >= 0

        # Term 1: candidate positives over full logits
        y_mask = torch.zeros((batch_size, num_classes), device=logits.device, dtype=logits.dtype)
        if valid.any():
            row_ids = torch.arange(batch_size, device=logits.device).unsqueeze(1).expand_as(candidates)
            y_mask[row_ids[valid], candidates[valid]] = 1.0
        y_cardinality = torch.clamp(y_mask.sum(dim=1), min=1.0)
        avg_candidate_score = (logits * y_mask).sum(dim=1) / y_cardinality
        term1 = self.psi(avg_candidate_score)

        # Split head/tail
        head_size = min(self.head_size, num_classes)
        tail_size = max(0, num_classes - head_size)
        head_logits = logits[:, :head_size]
        tail_logits = logits[:, head_size:] if tail_size > 0 else None

        # Term 2: head negatives (full)
        head_mask = torch.zeros((batch_size, head_size), device=logits.device, dtype=logits.dtype)
        if valid.any() and head_size > 0:
            head_valid = valid & (candidates < head_size)
            if head_valid.any():
                row_ids = torch.arange(batch_size, device=logits.device).unsqueeze(1).expand_as(candidates)
                head_mask[row_ids[head_valid], candidates[head_valid]] = 1.0
        term2 = (self.psi(-head_logits) * (1.0 - head_mask)).sum(dim=1) if head_size > 0 else 0.0

        # Term 3: tail negatives (sampled)
        term3 = 0.0
        if tail_size > 0 and tail_logits is not None:
            sample_size = min(self.tail_sample_size, tail_size)
            sampled_indices = torch.randint(0, tail_size, (sample_size,), device=logits.device)
            sampled_tail_logits = tail_logits[:, sampled_indices]

            tail_candidate_mask = torch.zeros((batch_size, tail_size), device=logits.device, dtype=torch.bool)
            if valid.any():
                tail_valid = valid & (candidates >= head_size)
                if tail_valid.any():
                    row_ids = torch.arange(batch_size, device=logits.device).unsqueeze(1).expand_as(candidates)
                    tail_candidate_mask[row_ids[tail_valid], candidates[tail_valid] - head_size] = True
            sampled_is_candidate = tail_candidate_mask[:, sampled_indices]
            tail_scale = float(tail_size) / float(sample_size)
            term3 = (self.psi(-sampled_tail_logits) * (~sampled_is_candidate).float()).sum(dim=1) * tail_scale

        return (term1 + term2 + term3).mean()

class ICDTrainer(BaseTrainer):
    """Trainer for ICD classification."""
    
    def __init__(
        self,
        config: ICDConfig,
        entity: ICDModelEntity,
        registry,
        observer,
        device,
        class_freq: np.ndarray = None,
        ignored_indices: Optional[List[int]] = None,
        use_amp: Optional[bool] = None,
        lr_override: Optional[float] = None,
    ):
        super().__init__(registry, observer, device)
        self.cfg = config
        self.entity = entity
        self.class_freq = class_freq
        self.use_amp = self.cfg.use_amp if use_amp is None else use_amp
        self.lr_override = lr_override
        self.scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() and self.use_amp else None
        self.loss_type = self.cfg.loss_type
        if self.loss_type in ("adaptive_softmax", "adaptive_clpl"):
            self.loss_fn = AdaptiveCLPLLoss(loss_type="logistic")
        else:
            self.loss_fn = CLPLLoss(loss_type=self.loss_type)

    def _prepare_inputs(self, batch: TensorBatch) -> Dict[str, torch.Tensor]:
        """Prepare inputs from a TensorBatch."""
        inputs = {}
        if hasattr(batch, 'x_num') and batch.x_num is not None:
            inputs["x_num"] = batch.x_num.to(self.device).float()
        else:
            x_val = batch.x_val.to(self.device).float()
            x_msk = batch.x_msk.to(self.device).float()
            x_delta = batch.x_delta.to(self.device).float()
            inputs["x_num"] = torch.cat([x_val, x_msk, x_delta], dim=1)
        
        if batch.x_cat is not None:
            inputs["x_cat"] = batch.x_cat.to(self.device).long()
        
        return inputs

    def _sample_target_from_candidates(self, candidates: torch.Tensor) -> torch.Tensor:
        """Sample one valid label from each candidate set."""
        valid_mask = candidates >= 0
        has_valid = valid_mask.any(dim=1)
        weights = valid_mask.float()
        if (~has_valid).any():
            weights[~has_valid, 0] = 1.0
        idx = torch.multinomial(weights, 1).squeeze(1)
        targets = candidates.gather(1, idx.unsqueeze(1)).squeeze(1)
        targets = torch.where(has_valid, targets, torch.zeros_like(targets))
        return targets.to(self.device).long()

    def train(self, train_loader: DataLoader, val_loader: DataLoader, evaluator: BaseEvaluator) -> None:
        """Run the training loop with validation."""
        self.entity.to(self.device)
        self.entity.model.train()
        params = list(self.entity.model.parameters())
        lr = self.cfg.lr if self.lr_override is None else self.lr_override
        optimizer = torch.optim.Adam(params, lr=lr)
        
        # [수정] Early Stopping을 위한 카운터 초기화
        no_improve = 0
        
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
                    
                    with torch.amp.autocast('cuda', enabled=self.scaler is not None):
                        if hasattr(self.entity.model, "adaptive_head"):
                            embeddings = self.entity.model(**inputs, return_embeddings=True)
                            logits = self.entity.model.adaptive_head.log_prob(embeddings)
                        else:
                            logits = self.entity.model(**inputs, return_embeddings=False)
                            if isinstance(logits, tuple):
                                logits = logits[0]
                        loss = self.loss_fn(logits, candidates)

                    if not torch.isfinite(loss):
                        self.observer.log("WARNING", "ICDTrainer: Non-finite loss detected; skipping step.")
                        continue

                    if self.scaler:
                        self.scaler.scale(loss).backward()
                        if self.cfg.grad_clip_norm > 0:
                            self.scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip_norm)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        if self.cfg.grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip_norm)
                        optimizer.step()
                        
                    epoch_loss += loss.item()
                    steps += 1
                    progress.advance(task)
            
            avg_loss = epoch_loss / max(1, steps)
            self.observer.track_metric("icd_train_loss", avg_loss, step=epoch)
            
            self.observer.log("INFO", f"ICDTrainer: Epoch {epoch} validation start.")
            metrics = evaluator.evaluate(val_loader)
            
            cand_acc = metrics.get("candidate_accuracy", 0.0)
            self.observer.log("INFO", f"Epoch {epoch} | Loss={avg_loss:.4f} | Cand Acc={cand_acc:.4f}")
            
            # [수정] Early Stopping 로직 적용
            if cand_acc > self.entity.best_metric:
                self.entity.best_metric = cand_acc
                self.registry.save_model(self.entity.model.state_dict(), self.entity.name, "best")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.cfg.patience:
                    self.observer.log("WARNING", f"Early stopping triggered after {epoch} epochs.")
                    break
