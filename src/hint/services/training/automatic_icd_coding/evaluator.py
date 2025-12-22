import torch
import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ..common.base import BaseEvaluator
from ....domain.entities import ICDModelEntity
from ....domain.vo import ICDConfig

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

    def evaluate(self, loader) -> Dict[str, float]:
        self.observer.log("INFO", "ICDEvaluator: Starting validation pass.")
        self.entity.model.eval()
        
        val_logits_list = []
        val_labels_list = []
        
        with torch.no_grad():
            for batch in loader:
                ids = batch.input_ids.to(self.device)
                mask = batch.attention_mask.to(self.device)
                num = batch.x_num.to(self.device).float()
                y = batch.y.cpu().numpy()
                
                logits = self.entity.model(input_ids=ids, attention_mask=mask, x_num=num)
                
                if self.ignored_indices:
                    logits[:, self.ignored_indices] = -1e9
                
                val_logits_list.append(logits.cpu().numpy())
                val_labels_list.append(y)
        
        if not val_logits_list:
            self.observer.log("WARNING", "ICDEvaluator: Validation loader returned no batches.")
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_macro": 0.0}

        X_val = np.vstack(val_logits_list)
        y_val = np.concatenate(val_labels_list)
        
        preds = np.argmax(X_val, axis=1)
        
        # [Removed] Stacker fitting/prediction logic

        return {
            "accuracy": float(accuracy_score(y_val, preds)),
            "precision": float(precision_score(y_val, preds, average='macro', zero_division=0)),
            "recall": float(recall_score(y_val, preds, average='macro', zero_division=0)),
            "f1_macro": float(f1_score(y_val, preds, average='macro', zero_division=0))
        }