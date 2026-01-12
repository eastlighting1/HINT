import torch
import numpy as np
from typing import List, Dict, Optional, Union
from torch.utils.data import DataLoader

from ..common.base import BaseEvaluator
from .trainer import CLPLLoss, AdaptiveCLPLLoss
from ....domain.entities import ICDModelEntity
from ....domain.vo import ICDConfig
from ....foundation.dtos import TensorBatch

class ICDEvaluator(BaseEvaluator):
    """Evaluator for ICD model validation and testing.

    Attributes:
        cfg (ICDConfig): ICD configuration.
        entity (ICDModelEntity): Model entity wrapper.
        ignored_indices (List[int]): Label indices to exclude from scoring.
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
        super().__init__(registry, observer, device)
        self.cfg = config
        self.entity = entity
        self.ignored_indices = ignored_indices if ignored_indices else []
        if self.cfg.loss_type in ("adaptive_softmax", "adaptive_clpl"):
            self.loss_fn = AdaptiveCLPLLoss(loss_type="logistic")
        else:
            self.loss_fn = CLPLLoss(loss_type=self.cfg.loss_type)

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

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run evaluation on a dataloader and return metrics."""
        self.entity.model.eval()
        self.observer.log("INFO", "ICDEvaluator: Starting validation/test pass.")
        
        all_candidate_hits = []
        total_loss = 0.0
        pred_counts = None
        total_preds = 0
        total_cand_sizes = 0
        total_cand_rows = 0

        with torch.no_grad():
            for batch in dataloader:
                batch_inputs = self._prepare_inputs(batch)
                filtered_inputs = batch_inputs

                if hasattr(self.entity.model, "adaptive_head"):
                    embeddings = self.entity.model(**filtered_inputs, return_embeddings=True)
                    logits = self.entity.model.adaptive_head.log_prob(embeddings)
                else:
                    logits = self.entity.model(**filtered_inputs)

                if isinstance(logits, tuple):
                    logits = logits[0]

                # --- Candidate Set Accuracy Logic ---
                candidates = getattr(batch, "candidates", None)
                if candidates is not None:
                    cands = candidates.to(self.device).long()
                    if self.ignored_indices:
                        ignored = torch.tensor(self.ignored_indices, device=self.device)
                        cands = torch.where(torch.isin(cands, ignored), torch.tensor(-1, device=self.device), cands)

                    # Raw Prediction check (Before masking)
                    raw_preds = torch.argmax(logits, dim=1) # [Batch]
                    if pred_counts is None:
                        pred_counts = torch.zeros(logits.shape[1], dtype=torch.long)
                    pred_counts += torch.bincount(raw_preds.detach().cpu(), minlength=pred_counts.numel())
                    total_preds += raw_preds.numel()

                    cand_sizes = (cands >= 0).sum(dim=1)
                    total_cand_sizes += cand_sizes.sum().item()
                    total_cand_rows += cand_sizes.numel()
                    
                    # Check if raw_preds is in cands. cands is [Batch, K]
                    hits = (cands == raw_preds.unsqueeze(1)).any(dim=1)
                    all_candidate_hits.extend(hits.cpu().numpy())

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
        if pred_counts is not None and total_preds > 0:
            unique_preds = int((pred_counts > 0).sum().item())
            top_idx = int(torch.argmax(pred_counts).item())
            top_ratio = float(pred_counts[top_idx].item() / total_preds)
            avg_cand = float(total_cand_sizes / total_cand_rows) if total_cand_rows > 0 else 0.0
            self.observer.log(
                "INFO",
                f"ICDEvaluator: pred_unique={unique_preds} top_pred={top_idx} top_ratio={top_ratio:.4f} avg_cand={avg_cand:.2f}",
            )
        
        # [수정] 오직 loss와 candidate_accuracy만 반환
        return {
            "candidate_accuracy": cand_acc,
            "loss": total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        }
