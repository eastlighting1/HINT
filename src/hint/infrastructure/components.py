"""Summary of the components module.

Longer description of the module purpose and usage.
"""

import torch

import torch.nn as nn

import torch.nn.functional as F

import numpy as np

import xgboost as xgb

from typing import List, Optional, Dict, Any

from sklearn.decomposition import PCA



class FocalLoss(nn.Module):

    """Summary of FocalLoss purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
        alpha (Any): Description of alpha.
        gamma (Any): Description of gamma.
        label_smoothing (Any): Description of label_smoothing.
        num_classes (Any): Description of num_classes.
    """

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, label_smoothing: float = 0.0, num_classes: int = 4):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
            alpha (Any): Description of alpha.
            gamma (Any): Description of gamma.
            label_smoothing (Any): Description of label_smoothing.
            num_classes (Any): Description of num_classes.
        
        Returns:
            None: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        super().__init__()

        self.alpha = alpha

        self.gamma = gamma

        self.label_smoothing = label_smoothing

        self.num_classes = num_classes



    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        """Summary of forward.
        
        Longer description of the forward behavior and usage.
        
        Args:
            logits (Any): Description of logits.
            targets (Any): Description of targets.
        
        Returns:
            torch.Tensor: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

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

    """Summary of CBFocalLoss purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
        class_weights (Any): Description of class_weights.
        gamma (Any): Description of gamma.
    """

    def __init__(self, class_counts: np.ndarray, beta: float = 0.999, gamma: float = 1.5, device: str = 'cpu'):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
            class_counts (Any): Description of class_counts.
            beta (Any): Description of beta.
            gamma (Any): Description of gamma.
            device (Any): Description of device.
        
        Returns:
            None: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        super().__init__()



        class_counts = np.array(class_counts)

        class_counts[class_counts == 0] = 1



        effective_num = 1.0 - np.power(beta, class_counts)



        weights = (1.0 - beta) / (np.array(effective_num) + 1e-6)





        weights = weights / np.sum(weights) * len(class_counts)



        self.class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

        self.gamma = gamma



    def forward(self, logits, targets):

        """Summary of forward.
        
        Longer description of the forward behavior and usage.
        
        Args:
            logits (Any): Description of logits.
            targets (Any): Description of targets.
        
        Returns:
            Any: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        if self.class_weights.device != logits.device:

            self.class_weights = self.class_weights.to(logits.device)



        ce = F.cross_entropy(logits, targets, weight=self.class_weights, reduction="none")

        pt = torch.exp(-ce)

        focal_loss = ((1 - pt) ** self.gamma) * ce

        return focal_loss.mean()



class CLPLLoss(nn.Module):

    """Summary of CLPLLoss purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
        None (None): No documented attributes.
    """

    def __init__(self):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
            None (None): This function does not accept arguments.
        
        Returns:
            None: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        super().__init__()



    def forward(self, logits: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:

        """Summary of forward.
        
        Longer description of the forward behavior and usage.
        
        Args:
            logits (Any): Description of logits.
            candidate_mask (Any): Description of candidate_mask.
        
        Returns:
            torch.Tensor: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """





        cand_sum = (logits * candidate_mask).sum(dim=1, keepdim=True)

        cand_count = candidate_mask.sum(dim=1, keepdim=True).clamp(min=1.0)

        cand_avg = cand_sum / cand_count







        non_candidate_mask = 1.0 - candidate_mask







        term1 = F.softplus(-cand_avg)







        term2 = (F.softplus(logits) * non_candidate_mask).sum(dim=1, keepdim=True)



        loss = term1 + term2

        return loss.mean()



class XGBoostStacker:

    """Summary of XGBoostStacker purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
        model (Any): Description of model.
        params (Any): Description of params.
        pca (Any): Description of pca.
    """

    def __init__(self, params: Dict):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
            params (Any): Description of params.
        
        Returns:
            None: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        self.params = params

        self.params["n_jobs"] = -1

        self.model = xgb.XGBClassifier(**self.params)

        self.pca: Optional[PCA] = None



    def fit_pca(self, X, n_components=0.95):

        """Summary of fit_pca.
        
        Longer description of the fit_pca behavior and usage.
        
        Args:
            X (Any): Description of X.
            n_components (Any): Description of n_components.
        
        Returns:
            Any: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        self.pca = PCA(n_components=n_components)

        return self.pca.fit_transform(X)



    def transform_pca(self, X):

        """Summary of transform_pca.
        
        Longer description of the transform_pca behavior and usage.
        
        Args:
            X (Any): Description of X.
        
        Returns:
            Any: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        if self.pca is None: return X

        return self.pca.transform(X)



    def fit(self, X, y):

        """Summary of fit.
        
        Longer description of the fit behavior and usage.
        
        Args:
            X (Any): Description of X.
            y (Any): Description of y.
        
        Returns:
            Any: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        self.model.fit(X, y)



    def predict(self, X):

        """Summary of predict.
        
        Longer description of the predict behavior and usage.
        
        Args:
            X (Any): Description of X.
        
        Returns:
            Any: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        return self.model.predict(X)
