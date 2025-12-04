import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xgboost as xgb
from sklearn.decomposition import PCA
from typing import List, Dict, Any, Optional

class CBFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss.

    Args:
        class_counts: Array of sample counts per class.
        beta: Hyperparameter for class balancing.
        gamma: Hyperparameter for focal loss.
        device: Torch device.
    """
    def __init__(self, class_counts: np.ndarray, beta: float = 0.999, gamma: float = 1.5, device: str = "cpu"):
        super().__init__()
        self.gamma = gamma
        
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / (effective_num + 1e-9)
        weights = weights / np.mean(weights)
        
        self.register_buffer("class_weights", torch.tensor(weights, device=device, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate loss.

        Args:
            logits: Prediction logits.
            targets: Target labels.
        """
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class CLPLLoss(nn.Module):
    """
    Candidate Label Partial Label Loss.

    Args:
        device: Torch device.
    """
    def __init__(self, device: str):
        super().__init__()
        self.device = device

    def forward(self, logits: torch.Tensor, candidates: List[List[int]]) -> torch.Tensor:
        """
        Calculate partial label loss.

        Args:
            logits: Prediction logits.
            candidates: List of candidate label indices for each sample.
        """
        loss = 0.0
        batch_size = logits.size(0)
        num_classes = logits.size(1)
        
        for i, cand_indices in enumerate(candidates):
            if not cand_indices:
                cand_tensor = torch.arange(num_classes, device=self.device)
            else:
                cand_tensor = torch.tensor(cand_indices, device=self.device, dtype=torch.long)
            
            pos_score = logits[i, cand_tensor].mean()
            
            mask = torch.ones(num_classes, dtype=torch.bool, device=self.device)
            mask[cand_tensor] = False
            neg_scores = logits[i, mask]
            
            neg_loss = torch.tensor(0.0, device=self.device)
            if neg_scores.numel() > 0:
                neg_loss = F.softplus(neg_scores).mean()
                
            loss += F.softplus(-pos_score) + neg_loss
            
        return loss / batch_size

class XGBoostStacker:
    """
    Wrapper for XGBoost classifier used in ensemble stacking.
    
    Args:
        params: XGBoost parameters.
    """
    def __init__(self, params: Dict[str, Any]):
        self.model = xgb.XGBClassifier(**params)
        self.pca: Optional[PCA] = None

    def fit_pca(self, X: np.ndarray, n_components: float) -> np.ndarray:
        """
        Fit PCA and transform data.

        Args:
            X: Input data.
            n_components: Number of components or variance ratio.
        
        Returns:
            Transformed data.
        """
        self.pca = PCA(n_components=n_components)
        return self.pca.fit_transform(np.nan_to_num(X))

    def transform_pca(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted PCA.

        Args:
            X: Input data.
        
        Returns:
            Transformed data.
        """
        if self.pca is None:
            return np.nan_to_num(X)
        return self.pca.transform(np.nan_to_num(X))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train XGBoost model.
        """
        self.model.fit(X, y, eval_set=[(X, y)], verbose=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using XGBoost model.
        """
        return self.model.predict(X)