import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xgboost as xgb
from typing import List, Optional, Dict, Any
from sklearn.decomposition import PCA

class FocalLoss(nn.Module):
    """
    Multi-class focal loss with optional class weights and label smoothing.
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, label_smoothing: float = 0.0, num_classes: int = 4):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)
        targets_ohe = F.one_hot(targets, self.num_classes).float()

        if self.label_smoothing > 0:
            targets_ohe = targets_ohe * (1.0 - self.label_smoothing) + self.label_smoothing / self.num_classes

        pt = (probs * targets_ohe).sum(dim=1)
        focal_weight = (1.0 - pt).pow(self.gamma)

        if self.alpha is not None:
            alpha_tensor = self.alpha.to(targets_ohe.device)
            alpha_weight = (alpha_tensor * targets_ohe).sum(dim=1)
            focal_weight = focal_weight * alpha_weight

        ce_loss = -(targets_ohe * log_probs).sum(dim=1)
        loss = focal_weight * ce_loss
        return loss.mean()

class CBFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss.
    """
    def __init__(self, class_counts: np.ndarray, beta: float = 0.999, gamma: float = 1.5, device: str = 'cpu'):
        super().__init__()
        # Fix: Handle zero counts to avoid divide by zero
        class_counts = np.array(class_counts)
        class_counts[class_counts == 0] = 1 # Avoid power of 0 issues temporarily
        
        effective_num = 1.0 - np.power(beta, class_counts)
        # Fix: Add epsilon to avoid division by zero
        weights = (1.0 - beta) / (np.array(effective_num) + 1e-6)
        
        # Restore 0 weight for 0 count classes if necessary, or just normalize
        weights = weights / np.sum(weights) * len(class_counts)
        
        self.class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        self.gamma = gamma

    def forward(self, logits, targets):
        if self.class_weights.device != logits.device:
            self.class_weights = self.class_weights.to(logits.device)
            
        ce = F.cross_entropy(logits, targets, weight=self.class_weights, reduction="none")
        pt = torch.exp(-ce)
        focal_loss = ((1 - pt) ** self.gamma) * ce
        return focal_loss.mean()

class CLPLLoss(nn.Module):
    """
    Convex Loss for Partial Labels (CLPL).
    Implementation of Equation (2) from 'Learning from Partial Labels' (Cour et al., 2011).
    Maximizes the average score of candidate labels while minimizing scores of non-candidate labels.
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (Batch, Num_Classes)
            candidate_mask: (Batch, Num_Classes), 1.0 for candidates, 0.0 otherwise.
        """
        # 1. Candidate Set Processing
        # Calculate average logit for candidates: (1/|y|) * sum(g_a(x))
        cand_sum = (logits * candidate_mask).sum(dim=1, keepdim=True)
        cand_count = candidate_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        cand_avg = cand_sum / cand_count
        
        # 2. Non-Candidate Set Processing
        # Mask for labels NOT in the candidate set
        non_candidate_mask = 1.0 - candidate_mask
        
        # 3. Loss Calculation using Softplus (convex surrogate for 0/1 loss)
        # Term 1: Maximize average candidate score -> Minimize softplus(-avg)
        term1 = F.softplus(-cand_avg)
        
        # Term 2: Minimize non-candidate scores -> Minimize softplus(logit)
        # Sum over all non-candidate labels
        term2 = (F.softplus(logits) * non_candidate_mask).sum(dim=1, keepdim=True)
        
        loss = term1 + term2
        return loss.mean()

class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling for probability calibration.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, logits_list: List[torch.Tensor], labels_list: List[torch.Tensor], device: str) -> float:
        self.to(device)
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        criterion = nn.CrossEntropyLoss().to(device)
        
        logits_all = torch.cat(logits_list).detach()
        labels_all = torch.cat(labels_list).detach()

        def _closure():
            optimizer.zero_grad()
            loss = criterion(logits_all / self.temperature, labels_all)
            loss.backward()
            return loss

        optimizer.step(_closure)
        return float(self.temperature.item())

class XGBoostStacker:
    """
    Wrapper for XGBoost + PCA stacking logic.
    """
    def __init__(self, params: Dict):
        self.params = params
        self.params["n_jobs"] = -1
        self.model = xgb.XGBClassifier(**self.params)
        self.pca: Optional[PCA] = None

    def fit_pca(self, X, n_components=0.95):
        self.pca = PCA(n_components=n_components)
        return self.pca.fit_transform(X)
        
    def transform_pca(self, X):
        if self.pca is None: return X
        return self.pca.transform(X)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)