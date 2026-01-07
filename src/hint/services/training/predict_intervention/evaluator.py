import torch
import numpy as np
from typing import Dict, List
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize

from ..common.base import BaseTrainer, BaseEvaluator
from ....domain.entities import InterventionModelEntity
from ....domain.vo import CNNConfig

class InterventionEvaluator(BaseEvaluator):
    """Evaluator for intervention prediction models.

    Attributes:
        cfg (CNNConfig): Evaluation configuration.
        entity (InterventionModelEntity): Model entity wrapper.
        class_names (List[str]): Ordered class labels.
    """
    def __init__(self, config: CNNConfig, entity: InterventionModelEntity, registry, observer, device):
        """Initialize the evaluator with model and configuration.

        Args:
            config (CNNConfig): Evaluation configuration.
            entity (InterventionModelEntity): Model entity wrapper.
            registry (Any): Artifact registry.
            observer (Any): Logging observer.
            device (Any): Target device.
        """
        super().__init__(registry, observer, device)
        self.cfg = config
        self.entity = entity
                                                    
        self.class_names = ["ONSET", "WEAN", "STAY ON", "STAY OFF"]

    def evaluate(self, loader, **kwargs) -> Dict[str, float]:
        """Evaluate the model on the provided data loader.

        Args:
            loader (Any): Evaluation data loader.
            **kwargs (Any): Additional options.

        Returns:
            Dict[str, float]: Aggregated evaluation metrics.
        """
        self.entity.network.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with self.entity.ema.average_parameters():
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(self.device)
                    logits = self.entity.network(batch.x_num, batch.x_cat, batch.x_icd)
                    probs = torch.softmax(logits, dim=1)
                    preds = logits.argmax(dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch.y.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
        
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_score = np.array(all_probs)
        
                     
        acc = float(accuracy_score(y_true, y_pred))
        
                             
        f1 = float(f1_score(y_true, y_pred, average='macro'))
        
                                              
                                                            
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
        
                                                                           
        try:
            aucs = roc_auc_score(y_true_bin, y_score, average=None, multi_class='ovr')
            macro_auc = float(np.mean(aucs))
            onset_auc = float(aucs[0])
            wean_auc = float(aucs[1])
            stay_on_auc = float(aucs[2])
            stay_off_auc = float(aucs[3])
        except ValueError:
                                                                     
            macro_auc = 0.0
            onset_auc = 0.0
            wean_auc = 0.0
            stay_on_auc = 0.0
            stay_off_auc = 0.0

                        
        try:
            macro_auprc = float(average_precision_score(y_true_bin, y_score, average='macro'))
        except ValueError:
            macro_auprc = 0.0

        metrics = {
            "accuracy": acc,
            "f1_score": f1,
            "macro_auc": macro_auc,
            "onset_auc": onset_auc,
            "wean_auc": wean_auc,
            "stay_on_auc": stay_on_auc,
            "stay_off_auc": stay_off_auc,
            "macro_auprc": macro_auprc
        }
        
        return metrics
