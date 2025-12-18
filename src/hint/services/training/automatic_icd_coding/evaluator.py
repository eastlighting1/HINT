import torch
import numpy as np
from typing import Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ..common.base import BaseEvaluator
from ....domain.entities import ICDModelEntity
from ....domain.vo import ICDConfig

class ICDEvaluator(BaseEvaluator):
    def __init__(self, config: ICDConfig, entity: ICDModelEntity, registry, observer, device):
        super().__init__(registry, observer, device)
        self.cfg = config
        self.entity = entity

    def evaluate(self, loader, fit_stacker: bool = False) -> Dict[str, float]:
        self.observer.log("INFO", "ICDEvaluator: Starting validation pass and stacker fit.")
        self.entity.head1.eval()
        self.entity.head2.eval()
        
        val_logits_list = []
        val_labels_list = []
        
        with torch.no_grad():
            for batch in loader:
                ids = batch['input_ids'].to(self.device)
                mask = batch['attention_mask'].to(self.device)
                num = batch['num'].to(self.device)
                y = batch['lab'].cpu().numpy()
                
                o1 = self.entity.head1(ids, mask, num)
                o2 = self.entity.head2(ids, mask, num)
                avg = (o1 + o2) / 2
                
                val_logits_list.append(avg.cpu().numpy())
                val_labels_list.append(y)
        
        if not val_logits_list:
            self.observer.log("WARNING", "ICDEvaluator: Validation loader returned no batches.")
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_macro": 0.0}

        X_val_ens = np.vstack(val_logits_list)
        y_val_ens = np.concatenate(val_labels_list)
        
        if fit_stacker:
            if self.entity.stacker.pca is None:
                X_pca = self.entity.stacker.fit_pca(X_val_ens, self.cfg.pca_components)
            else:
                X_pca = self.entity.stacker.transform_pca(X_val_ens)
            self.entity.stacker.fit(X_pca, y_val_ens)
            preds = self.entity.stacker.predict(X_pca)
        else:
            # Inference using fitted stacker or simple argmax fallback
            if self.entity.stacker.pca:
                X_pca = self.entity.stacker.transform_pca(X_val_ens)
                preds = self.entity.stacker.predict(X_pca)
            else:
                preds = np.argmax(X_val_ens, axis=1)
        
        return {
            "accuracy": float(accuracy_score(y_val_ens, preds)),
            "precision": float(precision_score(y_val_ens, preds, average='macro', zero_division=0)),
            "recall": float(recall_score(y_val_ens, preds, average='macro', zero_division=0)),
            "f1_macro": float(f1_score(y_val_ens, preds, average='macro', zero_division=0))
        }