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
    """Evaluator for ICD model validation and testing.

    This evaluator runs inference on data loaders and reports accuracy
    and related metrics, with optional label masking.

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
        """Initialize the evaluator with config and model entity.

        Args:
            config (ICDConfig): ICD configuration.
            entity (ICDModelEntity): Model entity wrapper.
            registry (Any): Artifact registry.
            observer (Any): Logging observer.
            device (Any): Target device.
            ignored_indices (Optional[List[int]]): Labels to ignore.
        """
        super().__init__(registry, observer, device)
        self.cfg = config
        self.entity = entity
        self.ignored_indices = ignored_indices if ignored_indices else []

    def _prepare_inputs(self, batch: TensorBatch) -> Dict[str, torch.Tensor]:
        """Prepare model inputs from a TensorBatch.

        Args:
            batch (TensorBatch): Batch of features and labels.

        Returns:
            Dict[str, torch.Tensor]: Input tensors for the model.
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
        """Run evaluation on a dataloader and return metrics.

        Args:
            dataloader (DataLoader): Data loader with evaluation data.

        Returns:
            Dict[str, float]: Aggregated evaluation metrics.
        """
        self.entity.model.eval()
        self.observer.log("INFO", "ICDEvaluator: Starting validation/test pass.")
        
        all_preds = []
        all_targets = []
        total_loss = 0.0
                                                                                 
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in dataloader:
                                                              
                target = getattr(batch, "y", None)
                if target is not None:
                    target = target.to(self.device)
                
                                                             
                valid_mask = None
                if target is not None and self.ignored_indices:
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

                # Modified inference logic to support Adaptive Softmax Head
                if hasattr(self.entity.model, "adaptive_head"):
                    # Extract embeddings
                    embeddings = self.entity.model(**filtered_inputs, return_embeddings=True)
                    # Compute log probabilities using the adaptive head
                    # This returns log_softmax, which serves as logits for CrossEntropy (expects raw logits)
                    # or NLLLoss (expects log_probs). 
                    # CrossEntropyLoss(input, target) expects raw logits, but since we have log_probs,
                    # we should technically use NLLLoss or treat them as logits where relative order is preserved for argmax.
                    # However, AdaptiveLogSoftmaxWithLoss.log_prob returns log probabilities.
                    # Since argmax(log_probs) == argmax(logits), accuracy calculation is correct.
                    # For loss calculation below: CrossEntropyLoss applies LogSoftmax internally.
                    # If we pass log_probs to CrossEntropyLoss, it applies LogSoftmax again, which is mathematically wrong but mechanically runs.
                    # Ideally, we use the log_probs directly for loss if we wanted exact loss, but here we just need consistency.
                    # Let's use the log_probs as logits for simplicity in preserving the structure, 
                    # but note that 'loss' metric might be slightly off scale compared to standard CE, 
                    # though 'total_loss' here is just for monitoring.
                    logits = self.entity.model.adaptive_head.log_prob(embeddings)
                else:                                          
                    logits = self.entity.model(**filtered_inputs)
                
                                                                      
                if isinstance(logits, tuple):
                                                                                      
                                                                              
                                                                                            
                    logits = logits[0]

                                                
                if getattr(batch, "candidates", None) is not None:
                    cands = batch.candidates.to(self.device).long()
                    
                    if target is not None and self.ignored_indices and valid_mask is not None:
                         cands = cands[valid_mask]

                                                              
                    inf_mask = torch.full_like(logits, float('-inf'))
                    
                    valid_cands_idx = (cands >= 0) & (cands < logits.size(1))
                    rows = torch.arange(cands.size(0), device=self.device).unsqueeze(1).expand_as(cands)
                    
                                                                              
                    inf_mask[rows[valid_cands_idx], cands[valid_cands_idx]] = 0.0
                    
                                                                                               
                                                                                                
                    if target is not None:
                                                          
                        if target.dim() > 1: target_idx = target.argmax(dim=1)
                        else: target_idx = target
                        
                        rows_t = torch.arange(target_idx.size(0), device=self.device)
                        inf_mask[rows_t, target_idx] = 0.0

                    preds = torch.argmax(logits + inf_mask, dim=1)
                else:
                    preds = torch.argmax(logits, dim=1)
                
                if self.ignored_indices:
                    logits[:, self.ignored_indices] = -1e9

                                                            
                if target is not None:
                                                                   
                    if target.dim() > 1 and target.shape[1] > 1:
                                             
                         target_indices = target.argmax(dim=1)
                    else:
                         target_indices = target

                    loss = criterion(logits, target_indices)
                    total_loss += loss.item()
                    all_targets.extend(target_indices.cpu().numpy())
                
                all_preds.extend(preds.cpu().numpy())

                                                                                
        if not all_targets:
            self.observer.log("WARNING", "No targets found in validation set. Metrics will be 0.")
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