import torch
import torch.nn as nn
import numpy as np
from typing import Dict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
from ..common.base import BaseEvaluator
from ....domain.entities import InterventionModelEntity
from ....domain.vo import CNNConfig

class ModelWithTemperature(nn.Module):
    """Wrap a model with temperature scaling for calibration.

    Attributes:
        model (nn.Module): Base model to calibrate.
        temperature (nn.Parameter): Learnable temperature parameter.
    """
    def __init__(self, model):
        """Initialize the temperature wrapper.

        Args:
            model (nn.Module): Base model to calibrate.
        """
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        """Forward pass with temperature scaling.

        Args:
            input (Any): Input tensor(s) for the wrapped model.

        Returns:
            torch.Tensor: Scaled logits.
        """
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """Scale logits by the temperature.

        Args:
            logits (torch.Tensor): Raw logits.

        Returns:
            torch.Tensor: Scaled logits.
        """
        return logits / self.temperature

class InterventionEvaluator(BaseEvaluator):
    """Evaluator with temperature scaling and thresholded metrics.

    This evaluator computes accuracy, F1, AUROC, and AUPRC with
    optional temperature calibration.
    """
    
    def __init__(self, config: CNNConfig, entity: InterventionModelEntity, registry, observer, device):
        """Initialize the intervention evaluator.

        Args:
            config (CNNConfig): Evaluation configuration.
            entity (InterventionModelEntity): Wrapped model entity.
            registry (Any): Artifact registry instance.
            observer (Any): Telemetry observer instance.
            device (str): Execution device.
        """
        super().__init__(registry, observer, device)
        self.cfg = config
        self.entity = entity
        self.temperature = 1.0

    def calibrate(self, loader) -> None:
        """Learn the optimal temperature on the validation set.

        Args:
            loader (DataLoader): Validation data loader.
        """
        self.entity.network.eval()
        logits_list = []
        labels_list = []
        
        with torch.no_grad():
            for batch in loader:
                x_val = batch.x_val.to(self.device).float()
                x_msk = batch.x_msk.to(self.device).float()
                x_delta = batch.x_delta.to(self.device).float()
                x_num = torch.cat([x_val, x_msk, x_delta], dim=1)
                
                x_icd = batch.x_icd.to(self.device).float() if batch.x_icd is not None else None
                y = batch.y[:, -1].to(self.device)
                
                logits = self.entity.network(x_num, x_icd)
                logits_list.append(logits)
                labels_list.append(y)
        
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)
        
        self.observer.log("INFO", "InterventionEvaluator: Temperature calibration start.")
        temperature = nn.Parameter(torch.ones(1, device=self.device) * 1.5)
        optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
        
        def eval():
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(logits / temperature, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval)
        self.temperature = temperature.item()
        self.observer.log("INFO", f"Optimal Temperature found: {self.temperature:.4f}")

    def evaluate(self, loader, **kwargs) -> Dict[str, float]:
        """Evaluate the model on a data loader.

        Args:
            loader (DataLoader): Evaluation data loader.
            **kwargs (Any): Optional keyword arguments.

        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        self.entity.network.eval()
        
        all_probs = []
        all_labels = []
        total_loss = 0.0
        self.observer.log("INFO", "InterventionEvaluator: Evaluation loop start.")
        
        with torch.no_grad():
            for batch in loader:
                x_val = batch.x_val.to(self.device).float()
                x_msk = batch.x_msk.to(self.device).float()
                x_delta = batch.x_delta.to(self.device).float()
                x_num = torch.cat([x_val, x_msk, x_delta], dim=1)
                x_icd = batch.x_icd.to(self.device).float() if batch.x_icd is not None else None
                y = batch.y[:, -1].to(self.device)
                
                logits = self.entity.network(x_num, x_icd)
                
                scaled_logits = logits / self.temperature
                probs = torch.softmax(scaled_logits, dim=1)
                
                all_probs.append(probs.cpu().numpy())
                all_labels.append(y.cpu().numpy())
                
                loss = nn.CrossEntropyLoss()(scaled_logits, y)
                total_loss += loss.item()

        y_true = np.concatenate(all_labels)
        y_prob = np.concatenate(all_probs)
        y_pred = np.argmax(y_prob, axis=1)
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        
        try:
            y_true_bin = label_binarize(y_true, classes=[0,1,2,3])
            macro_auc = roc_auc_score(y_true_bin, y_prob, average='macro', multi_class='ovr')
            macro_auprc = average_precision_score(y_true_bin, y_prob, average='macro')
        except Exception:
            macro_auc = 0.0
            macro_auprc = 0.0
            
        metrics = {
            "loss": total_loss / len(loader),
            "accuracy": acc,
            "f1_score": f1,
            "macro_auc": macro_auc,
            "macro_auprc": macro_auprc
        }
        return metrics
