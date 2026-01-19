"""Summary of the evaluator module.

Longer description of the module purpose and usage.
"""

import torch
import torch.nn as nn
import numpy as np

from typing import List, Dict, Optional, Union

from torch.utils.data import DataLoader

from ..common.base import BaseEvaluator
from .trainer import CLPLLoss, AdaptiveCLPLLoss
from ....domain.entities import ICDModelEntity
from ....domain.vo import ICDConfig
from ....foundation.dtos import TensorBatch


class ICDEvaluator(BaseEvaluator):

    """Summary of ICDEvaluator purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    cfg (Any): Description of cfg.
    entity (Any): Description of entity.
    ignored_indices (Any): Description of ignored_indices.
    loss_fn (Any): Description of loss_fn.
    """

    def __init__(
        self,
        config: ICDConfig,
        entity: ICDModelEntity,
        registry,
        observer,
        device,
        ignored_indices: Optional[List[int]] = None
    ):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
        config (Any): Description of config.
        entity (Any): Description of entity.
        registry (Any): Description of registry.
        observer (Any): Description of observer.
        device (Any): Description of device.
        ignored_indices (Any): Description of ignored_indices.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        super().__init__(registry, observer, device)

        self.cfg = config

        self.entity = entity

        self.ignored_indices = ignored_indices if ignored_indices else []

        self.temperature = 1.0

        if self.cfg.loss_type in ("adaptive_softmax", "adaptive_clpl"):
            head_size = min(getattr(self.cfg, "adaptive_clpl_head_size", 2000), self.entity.num_classes or 0)
            if head_size <= 0:
                head_size = 2000
            self.loss_fn = AdaptiveCLPLLoss(
                head_size=head_size,
                tail_sample_size=getattr(self.cfg, "adaptive_clpl_tail_sample_size", 100),
                loss_type="logistic",
                logit_clip=getattr(self.cfg, "adaptive_clpl_logit_clip", 20.0),
            )

        else:

            self.loss_fn = CLPLLoss(loss_type=self.cfg.loss_type)

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

    def evaluate(self, dataloader: DataLoader, temperature: Optional[float] = None, log_metrics: bool = True) -> Dict[str, float]:

        """Summary of evaluate.
        
        Longer description of the evaluate behavior and usage.
        
        Args:
        dataloader (Any): Description of dataloader.
        temperature (Any): Description of temperature.
        log_metrics (Any): Description of log_metrics.
        
        Returns:
        Dict[str, float]: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        self.entity.model.eval()

        if temperature is None:
            temperature = self.temperature

        if log_metrics:
            self.observer.log(
                "INFO",
                f"ICDEvaluator: Stage 1/3 starting evaluation batches={len(dataloader)}.",
            )

        all_candidate_hits = []

        total_loss = 0.0

        pred_counts = None

        total_preds = 0

        total_cand_sizes = 0

        total_cand_rows = 0

        total_cpm = 0.0

        total_cmg = 0.0

        cand_seen = None

        with torch.no_grad():

            for batch in dataloader:

                batch_inputs = self._prepare_inputs(batch)

                filtered_inputs = batch_inputs

                use_adaptive = self.cfg.loss_type in ("adaptive_softmax", "adaptive_clpl") and hasattr(
                    self.entity.model, "adaptive_head"
                )
                if use_adaptive:
                    embeddings = self.entity.model(**filtered_inputs, return_embeddings=True)
                    logits = self.entity.model.adaptive_head.log_prob(embeddings)
                else:
                    logits = self.entity.model(**filtered_inputs)
                    if isinstance(logits, tuple):
                        logits = logits[0]


                candidates = getattr(batch, "candidates", None)

                if candidates is not None:

                    cands = candidates.to(self.device).long()

                    if self.ignored_indices:

                        ignored = torch.tensor(self.ignored_indices, device=self.device)

                        cands = torch.where(torch.isin(cands, ignored), torch.tensor(-1, device=self.device), cands)


                    scaled_logits = logits / float(max(temperature, 1e-6))
                    scaled_preds = torch.argmax(scaled_logits, dim=1)

                    if pred_counts is None:

                        pred_counts = torch.zeros(logits.shape[1], dtype=torch.long)

                    pred_counts += torch.bincount(scaled_preds.detach().cpu(), minlength=pred_counts.numel())

                    total_preds += scaled_preds.numel()

                    cand_sizes = (cands >= 0).sum(dim=1)

                    valid_rows = cand_sizes > 0

                    if valid_rows.any():

                        total_cand_sizes += cand_sizes[valid_rows].sum().item()

                        total_cand_rows += int(valid_rows.sum().item())


                        hits = (cands == scaled_preds.unsqueeze(1)).any(dim=1)

                        all_candidate_hits.extend(hits[valid_rows].cpu().numpy())

                        probs = torch.softmax(scaled_logits, dim=1)

                        cand_indices = cands.clone()
                        cand_valid = cand_indices >= 0
                        cand_indices = torch.where(cand_valid, cand_indices, torch.zeros_like(cand_indices))
                        cand_probs = probs.gather(1, cand_indices)
                        cand_probs = cand_probs * cand_valid.float()
                        total_cpm += cand_probs.sum(dim=1)[valid_rows].sum().item()

                        cand_mask = torch.zeros_like(probs, dtype=torch.bool)
                        row_ids = torch.arange(cands.shape[0], device=cands.device).unsqueeze(1).expand_as(cands)
                        cand_mask[row_ids[cand_valid], cand_indices[cand_valid]] = True
                        cand_max = probs.masked_fill(~cand_mask, float("-inf")).max(dim=1).values
                        non_cand_max = probs.masked_fill(cand_mask, float("-inf")).max(dim=1).values
                        margins = cand_max - non_cand_max
                        total_cmg += margins[valid_rows].sum().item()

                        if cand_seen is None:
                            cand_seen = torch.zeros(logits.shape[1], dtype=torch.bool)
                        cand_flat = cands[cand_valid].detach().cpu()
                        cand_seen[cand_flat] = True

                    loss = self.loss_fn(logits, cands)

                    total_loss += loss.item()

                else:

                    target = getattr(batch, "y", None)

                    if target is not None:

                        target = target.to(self.device)

                        if self.ignored_indices:

                            logits[:, self.ignored_indices] = -1e9

                        if target.dim() > 1 and target.shape[1] > 1:

                            target_indices = target.argmax(dim=1)

                        else:

                            target_indices = target

                        loss = torch.nn.CrossEntropyLoss()(logits, target_indices)

                        total_loss += loss.item()

        cand_acc = np.mean(all_candidate_hits) if all_candidate_hits else 0.0

        cpm = float(total_cpm / total_cand_rows) if total_cand_rows > 0 else 0.0

        cmg = float(total_cmg / total_cand_rows) if total_cand_rows > 0 else 0.0

        ndi = 0.0

        epr = 0.0

        if pred_counts is not None and total_preds > 0:

            unique_preds = int((pred_counts > 0).sum().item())

            top_idx = int(torch.argmax(pred_counts).item())

            top_ratio = float(pred_counts[top_idx].item() / total_preds)

            avg_cand = float(total_cand_sizes / total_cand_rows) if total_cand_rows > 0 else 0.0

            k_cand = int(cand_seen.sum().item()) if cand_seen is not None else 0

            if k_cand > 0:
                cand_pred_counts = pred_counts[cand_seen] if cand_seen is not None else pred_counts
                di = float(cand_pred_counts.max().item() / total_preds)
                if k_cand > 1:
                    ndi = float((di - 1.0 / k_cand) / (1.0 - 1.0 / k_cand))

                q = pred_counts.float() / float(total_preds)
                q_nonzero = q[q > 0]
                entropy = float(-(q_nonzero * torch.log(q_nonzero)).sum().item())
                k_eff = float(torch.exp(torch.tensor(entropy)).item())
                epr = float(k_eff / k_cand) if k_cand > 0 else 0.0

            if log_metrics:
                self.observer.log(
                    "INFO",
                    f"ICDEvaluator: Stage 2/3 pred_unique={unique_preds} top_pred={top_idx} top_ratio={top_ratio:.4f} avg_cand={avg_cand:.2f} CPM={cpm:.4f} CMG={cmg:.4f} NDI={ndi:.4f} EPR={epr:.4f}.",
                )


        if log_metrics:
            self.observer.log(
                "INFO",
                f"ICDEvaluator: Stage 3/3 complete Candidate Accuracy={cand_acc:.4f} Loss={total_loss / len(dataloader) if len(dataloader) > 0 else 0.0:.4f}.",
            )

        return {

            "candidate_accuracy": cand_acc,

            "loss": total_loss / len(dataloader) if len(dataloader) > 0 else 0.0,
            "cpm": cpm,
            "cmg": cmg,
            "ndi": ndi,
            "epr": epr,

        }
