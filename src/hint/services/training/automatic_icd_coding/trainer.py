"""Summary of the trainer module.

Longer description of the module purpose and usage.
"""

import torch

import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import random

from typing import List, Optional, Dict

from torch.utils.data import DataLoader

from ..common.base import BaseTrainer, BaseEvaluator

from ....domain.entities import ICDModelEntity
from ....infrastructure.models.tabnet import TabNetICD

from ....domain.vo import ICDConfig

from ....foundation.dtos import TensorBatch




class CLPLLoss(nn.Module):

    """Summary of CLPLLoss purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    loss_type (Any): Description of loss_type.
    """

    def __init__(self, loss_type: str = "exponential", class_weights: Optional[torch.Tensor] = None):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
        loss_type (Any): Description of loss_type.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        super().__init__()

        self.loss_type = loss_type
        self.class_weights = class_weights



    def psi(self, u: torch.Tensor) -> torch.Tensor:

        """Summary of psi.
        
        Longer description of the psi behavior and usage.
        
        Args:
        u (Any): Description of u.
        
        Returns:
        torch.Tensor: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        if self.loss_type == "exponential":

            return torch.exp(-u)

        if self.loss_type == "logistic":

            return torch.log1p(torch.exp(-u))

        if self.loss_type == "hinge":

            return torch.clamp(1 - u, min=0)

        raise ValueError(f"Unknown loss type: {self.loss_type}")



    def forward(self, logits: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:

        """Summary of forward.
        
        Longer description of the forward behavior and usage.
        
        Args:
        logits (Any): Description of logits.
        candidates (Any): Description of candidates.
        
        Returns:
        torch.Tensor: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        if candidates is None:

            raise ValueError("CLPLLoss requires candidate sets.")



        batch_size, num_classes = logits.shape

        candidates = candidates.to(logits.device).long()

        valid = candidates >= 0



        y_mask = torch.zeros((batch_size, num_classes), device=logits.device, dtype=logits.dtype)

        if valid.any():

            row_ids = torch.arange(batch_size, device=logits.device).unsqueeze(1).expand_as(candidates)

            y_mask[row_ids[valid], candidates[valid]] = 1.0



        y_counts = y_mask.sum(dim=1)
        y_cardinality = torch.clamp(y_counts, min=1.0)

        avg_candidate_score = (logits * y_mask).sum(dim=1) / y_cardinality

        term1 = self.psi(avg_candidate_score)



        non_candidate_mask = 1.0 - y_mask

        term2 = (self.psi(-logits) * non_candidate_mask).sum(dim=1)



        loss = term1 + term2

        if self.class_weights is not None:
            class_weights = self.class_weights.to(logits.device)
            cand_weights = (y_mask * class_weights).sum(dim=1) / y_cardinality
            cand_weights = torch.where(y_counts > 0, cand_weights, torch.ones_like(cand_weights))
            loss = loss * cand_weights

        return loss.mean()



class AdaptiveCLPLLoss(nn.Module):

    """Summary of AdaptiveCLPLLoss purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    head_size (Any): Description of head_size.
    logit_clip (Any): Description of logit_clip.
    loss_type (Any): Description of loss_type.
    tail_sample_size (Any): Description of tail_sample_size.
    """

    def __init__(

        self,

        head_size: int = 2000,

        tail_sample_size: int = 100,

        loss_type: str = "logistic",

        logit_clip: float = 20.0,
        class_weights: Optional[torch.Tensor] = None,

    ):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
        head_size (Any): Description of head_size.
        tail_sample_size (Any): Description of tail_sample_size.
        loss_type (Any): Description of loss_type.
        logit_clip (Any): Description of logit_clip.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        super().__init__()

        self.head_size = head_size

        self.tail_sample_size = tail_sample_size

        self.loss_type = loss_type

        self.logit_clip = logit_clip
        self.class_weights = class_weights



    def psi(self, u: torch.Tensor) -> torch.Tensor:

        """Summary of psi.
        
        Longer description of the psi behavior and usage.
        
        Args:
        u (Any): Description of u.
        
        Returns:
        torch.Tensor: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        if self.loss_type == "exponential":

            return torch.exp(-u)

        if self.loss_type == "logistic":

            return torch.log1p(torch.exp(-u))

        if self.loss_type == "hinge":

            return torch.clamp(1 - u, min=0)

        raise ValueError(f"Unknown loss type: {self.loss_type}")



    def forward(self, logits: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:

        """Summary of forward.
        
        Longer description of the forward behavior and usage.
        
        Args:
        logits (Any): Description of logits.
        candidates (Any): Description of candidates.
        
        Returns:
        torch.Tensor: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        if candidates is None:

            raise ValueError("AdaptiveCLPLLoss requires candidate sets.")



        if self.logit_clip is not None:

            logits = logits.clamp(-self.logit_clip, self.logit_clip)



        batch_size, num_classes = logits.shape

        candidates = candidates.to(logits.device).long()

        valid = candidates >= 0





        y_mask = torch.zeros((batch_size, num_classes), device=logits.device, dtype=logits.dtype)

        if valid.any():

            row_ids = torch.arange(batch_size, device=logits.device).unsqueeze(1).expand_as(candidates)

            y_mask[row_ids[valid], candidates[valid]] = 1.0

        y_counts = y_mask.sum(dim=1)
        y_cardinality = torch.clamp(y_counts, min=1.0)

        avg_candidate_score = (logits * y_mask).sum(dim=1) / y_cardinality

        term1 = self.psi(avg_candidate_score)





        head_size = min(self.head_size, num_classes)

        tail_size = max(0, num_classes - head_size)

        head_logits = logits[:, :head_size]

        tail_logits = logits[:, head_size:] if tail_size > 0 else None





        head_mask = torch.zeros((batch_size, head_size), device=logits.device, dtype=logits.dtype)

        if valid.any() and head_size > 0:

            head_valid = valid & (candidates < head_size)

            if head_valid.any():

                row_ids = torch.arange(batch_size, device=logits.device).unsqueeze(1).expand_as(candidates)

                head_mask[row_ids[head_valid], candidates[head_valid]] = 1.0

        term2 = (self.psi(-head_logits) * (1.0 - head_mask)).sum(dim=1) if head_size > 0 else 0.0





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



        loss = term1 + term2 + term3

        if self.class_weights is not None:
            class_weights = self.class_weights.to(logits.device)
            cand_weights = (y_mask * class_weights).sum(dim=1) / y_cardinality
            cand_weights = torch.where(y_counts > 0, cand_weights, torch.ones_like(cand_weights))
            loss = loss * cand_weights

        return loss.mean()



class ICDTrainer(BaseTrainer):

    """Summary of ICDTrainer purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    cfg (Any): Description of cfg.
    class_freq (Any): Description of class_freq.
    entity (Any): Description of entity.
    loss_fn (Any): Description of loss_fn.
    loss_type (Any): Description of loss_type.
    lr_override (Any): Description of lr_override.
    scaler (Any): Description of scaler.
    use_amp (Any): Description of use_amp.
    """



    def __init__(

        self,

        config: ICDConfig,

        entity: ICDModelEntity,

        registry,

        observer,

        device,

        class_freq: np.ndarray = None,

        ignored_indices: Optional[List[int]] = None,

        num_classes: int = 0,

        use_amp: Optional[bool] = None,

        lr_override: Optional[float] = None,

    ):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
        config (Any): Description of config.
        entity (Any): Description of entity.
        registry (Any): Description of registry.
        observer (Any): Description of observer.
        device (Any): Description of device.
        class_freq (Any): Description of class_freq.
        ignored_indices (Any): Description of ignored_indices.
        use_amp (Any): Description of use_amp.
        lr_override (Any): Description of lr_override.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        super().__init__(registry, observer, device)

        self.cfg = config

        self.entity = entity

        self.class_freq = class_freq
        self.ignored_indices = ignored_indices if ignored_indices else []

        self.use_amp = self.cfg.use_amp if use_amp is None else use_amp

        self.lr_override = lr_override

        self.num_classes = num_classes

        self.scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() and self.use_amp else None

        self.loss_type = self.cfg.loss_type
        self.class_weights = None
        if self.class_freq is not None:
            freq = np.asarray(self.class_freq, dtype=np.float32)
            freq = np.where(freq <= 0, 1.0, freq)
            power = float(getattr(self.cfg, "class_weight_power", 0.5))
            weights = 1.0 / np.power(freq, power)
            weights = weights / np.mean(weights)
            clip = float(getattr(self.cfg, "class_weight_clip", 0.0))
            if clip > 0:
                weights = np.clip(weights, 0.0, clip)
            self.class_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        if self.loss_type in ("adaptive_softmax", "adaptive_clpl"):
            head_size = min(getattr(self.cfg, "adaptive_clpl_head_size", 2000), self.num_classes or 0)
            if head_size <= 0:
                head_size = 2000
            self.loss_fn = AdaptiveCLPLLoss(
                head_size=head_size,
                tail_sample_size=getattr(self.cfg, "adaptive_clpl_tail_sample_size", 100),
                loss_type="logistic",
                logit_clip=getattr(self.cfg, "adaptive_clpl_logit_clip", 20.0),
                class_weights=self.class_weights,
            )

        else:

            self.loss_fn = CLPLLoss(loss_type=self.loss_type, class_weights=self.class_weights)

    def _prepare_inputs(self, batch: TensorBatch) -> Dict[str, torch.Tensor]:

        """Summary of _prepare_inputs.
        
        Longer description of the _prepare_inputs behavior and usage.
        
        Args:
        batch (Any): Description of batch.
        
        Returns:
        Dict[str, torch.Tensor]: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

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

        """Summary of _sample_target_from_candidates.
        
        Longer description of the _sample_target_from_candidates behavior and usage.
        
        Args:
        candidates (Any): Description of candidates.
        
        Returns:
        torch.Tensor: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

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

        """Summary of train.
        
        Longer description of the train behavior and usage.
        
        Args:
        train_loader (Any): Description of train_loader.
        val_loader (Any): Description of val_loader.
        evaluator (Any): Description of evaluator.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        self.entity.to(self.device)

        self.entity.model.train()

        params = list(self.entity.model.parameters())

        lr = self.cfg.lr if self.lr_override is None else self.lr_override

        optimizer = torch.optim.Adam(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=getattr(self.cfg, "lr_plateau_factor", 0.5),
            patience=getattr(self.cfg, "lr_plateau_patience", 2),
            min_lr=getattr(self.cfg, "lr_plateau_min_lr", 1e-6),
        )
        self.observer.log(
            "INFO",
            f"ICDTrainer: Stage 1/4 setup device={self.device} loss_type={self.loss_type} use_amp={self.use_amp} lr={lr}.",
        )





        no_improve = 0



        self.observer.log("INFO", "ICDTrainer: Stage 2/4 starting training loop.")

        for epoch in range(1, self.cfg.epochs + 1):

            self.entity.epoch = epoch

            self.entity.model.train()

            epoch_loss = 0.0

            steps = 0



            with self.observer.create_progress(f"Epoch {epoch} ICD Train", total=len(train_loader)) as progress:

                task = progress.add_task("Training", total=len(train_loader))



                for batch in train_loader:

                    optimizer.zero_grad()

                    inputs = self._prepare_inputs(batch)



                    candidates = batch.candidates



                    with torch.amp.autocast('cuda', enabled=self.scaler is not None):

                        use_adaptive = self.loss_type in ("adaptive_softmax", "adaptive_clpl") and hasattr(
                            self.entity.model, "adaptive_head"
                        )
                        sparse_loss = None
                        is_tabnet = isinstance(self.entity.model, TabNetICD)
                        if use_adaptive:
                            if is_tabnet:
                                embeddings, sparse_loss = self.entity.model(
                                    **inputs, return_embeddings=True, return_sparse_loss=True
                                )
                            else:
                                embeddings = self.entity.model(**inputs, return_embeddings=True)
                            logits = self.entity.model.adaptive_head.log_prob(embeddings)
                        else:
                            if is_tabnet:
                                logits, sparse_loss = self.entity.model(
                                    **inputs, return_embeddings=False, return_sparse_loss=True
                                )
                            else:
                                logits = self.entity.model(**inputs, return_embeddings=False)
                                if isinstance(logits, tuple):
                                    logits = logits[0]

                        loss = self.loss_fn(logits, candidates)
                        if sparse_loss is not None:
                            loss = loss + (self.cfg.lambda_sparse * sparse_loss)

                        pseudo_weight = float(getattr(self.cfg, "pseudo_label_ce_weight", 0.0))
                        if pseudo_weight > 0.0 and candidates is not None:
                            cand = candidates.to(logits.device).long()
                            valid = cand >= 0
                            cand_mask = torch.zeros(
                                logits.shape[0], logits.shape[1], device=logits.device, dtype=torch.bool
                            )
                            if valid.any():
                                row_ids = torch.arange(logits.shape[0], device=logits.device).unsqueeze(1).expand_as(cand)
                                cand_mask[row_ids[valid], cand[valid]] = True
                                if self.ignored_indices:
                                    cand_mask[:, self.ignored_indices] = False
                                row_valid = cand_mask.any(dim=1)
                                if row_valid.any():
                                    masked_logits = logits.masked_fill(~cand_mask, float("-inf"))
                                    margin = float(getattr(self.cfg, "pseudo_label_margin", 0.0))
                                    if margin > 0.0:
                                        top2 = torch.topk(masked_logits[row_valid], k=2, dim=1).values
                                        confident = (top2[:, 0] - top2[:, 1]) >= margin
                                    else:
                                        confident = torch.ones(int(row_valid.sum().item()), device=logits.device, dtype=torch.bool)
                                    if confident.any():
                                        pseudo_targets = masked_logits[row_valid].argmax(dim=1)
                                        loss_inputs = masked_logits[row_valid][confident]
                                        if use_adaptive:
                                            pl_loss = F.nll_loss(loss_inputs, pseudo_targets[confident])
                                        else:
                                            pl_loss = F.cross_entropy(loss_inputs, pseudo_targets[confident])
                                        loss = loss + (pseudo_weight * pl_loss)



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

            self.observer.log("INFO", f"ICDTrainer: Stage 2/4 epoch={epoch} train_loss={avg_loss:.4f}.")



            self.observer.log("INFO", f"ICDTrainer: Stage 3/4 epoch={epoch} validation start.")

            metrics = evaluator.evaluate(val_loader)



            val_loss = metrics.get("loss", 0.0)
            cand_acc = metrics.get("candidate_accuracy", 0.0)
            hit3 = metrics.get("cand_hit@3", 0.0)
            hit5 = metrics.get("cand_hit@5", 0.0)
            hit10 = metrics.get("cand_hit@10", 0.0)
            self.observer.log(
                "INFO",
                (
                    f"ICDTrainer: Stage 3/4 epoch={epoch} Loss={val_loss:.4f} "
                    f"Candidate Accuracy={cand_acc:.4f} Hit@3={hit3:.4f} "
                    f"Hit@5={hit5:.4f} Hit@10={hit10:.4f}."
                ),
            )
            scheduler.step(val_loss)

            if val_loss < self.entity.best_metric:

                self.entity.best_metric = val_loss

                self.registry.save_model(self.entity.model.state_dict(), self.entity.name, "best")

                no_improve = 0

            else:

                no_improve += 1

                if no_improve >= self.cfg.patience:

                    self.observer.log("WARNING", f"ICDTrainer: Stage 4/4 early stopping epoch={epoch}.")

                    break
