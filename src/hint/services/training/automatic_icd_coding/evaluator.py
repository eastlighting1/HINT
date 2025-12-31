import torch
import numpy as np
from typing import List, Dict, Optional, Union
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from ..common.base import BaseEvaluator
from ....domain.entities import ICDModelEntity
from ....domain.vo import ICDConfig
from ....foundation.dtos import TensorBatch

class ICDEvaluator(BaseEvaluator):
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

    def _prepare_inputs(self, batch: TensorBatch) -> Dict[str, torch.Tensor]:
        """
        Duplicate of _prepare_inputs from Trainer to ensure standalone functionality.
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

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.entity.model.eval()
        self.observer.log("INFO", "ICDEvaluator: Starting validation pass.")
        
        all_preds = []
        all_targets = []
        total_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in dataloader:
                target = batch.y.to(self.device)

                if self.ignored_indices:
                    valid_mask = torch.ones(target.shape[0], dtype=torch.bool, device=self.device)
                    for idx in self.ignored_indices:
                        valid_mask &= (target != idx)
                    
                    if not valid_mask.any():
                        continue
                        
                    target = target[valid_mask]
                    batch_inputs = self._prepare_inputs(batch)
                    filtered_inputs = {k: v[valid_mask] for k, v in batch_inputs.items()}
                else:
                    filtered_inputs = self._prepare_inputs(batch)

                # [FIXED] Pass inputs as dictionary kwargs
                logits = self.entity.model(**filtered_inputs)
                
                # Inference Time Disambiguation:
                # If candidates exist, restrict prediction to the candidate set
                if getattr(batch, "candidates", None) is not None:
                    cands = batch.candidates.to(self.device).long()
                    
                    if self.ignored_indices and 'valid_mask' in locals():
                         cands = cands[valid_mask]

                    # Mask non-candidates with -inf for argmax
                    inf_mask = torch.full_like(logits, float('-inf'))
                    
                    valid_cands_idx = (cands >= 0) & (cands < logits.size(1))
                    rows = torch.arange(cands.size(0), device=self.device).unsqueeze(1).expand_as(cands)
                    
                    # Set candidate positions to 0.0 (keeping original logits)
                    inf_mask[rows[valid_cands_idx], cands[valid_cands_idx]] = 0.0
                    
                    preds = torch.argmax(logits + inf_mask, dim=1)
                else:
                    preds = torch.argmax(logits, dim=1)
                
                if self.ignored_indices:
                    logits[:, self.ignored_indices] = -1e9

                loss = criterion(logits, target)
                total_loss += loss.item()

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        if not all_targets:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_macro": 0.0, "loss": 0.0}

        acc = accuracy_score(all_targets, all_preds)
        p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
        
        return {
            "accuracy": acc,
            "precision": p,
            "recall": r,
            "f1_macro": f1,
            "loss": total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        }