"""Calibration infrastructure module.

This module defines the VectorScaler network and its corresponding Entity wrapper
for the post-hoc calibration process.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from ..domain.entities import TrainableEntity


class VectorScaler(nn.Module):
    """
    A simple affine transformation layer for vector scaling calibration.
    It learns a weight and bias for each class logit independently.
    
    logits' = logits * weight + bias
    """
    def __init__(self, num_classes: int):
        super().__init__()
        # Initialize weights to 1.0 and bias to 0.0 to start as Identity
        self.weight = nn.Parameter(torch.ones(num_classes))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # Broadcasting: (Batch, K) * (K,) + (K,)
        return logits * self.weight + self.bias


class CalibrationEntity(TrainableEntity):
    """
    Entity wrapper for the VectorScaler network.
    Manages state persistence and lifecycle for the calibration module.
    """
    def __init__(self, num_classes: int, config: Any):
        super().__init__("calibration_scaler")
        self.network = VectorScaler(num_classes)
        self.config = config
        self.optimizer: Optional[torch.optim.Optimizer] = None

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state snapshot of the calibration entity."""
        state = {
            "network": self.network.state_dict(),
            "best_metric": self.best_metric,
            "epoch": self.epoch
        }
        if self.optimizer:
            state["optimizer"] = self.optimizer.state_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restores the state of the calibration entity."""
        self.network.load_state_dict(state["network"])
        self.best_metric = state.get("best_metric", 0.0)
        self.epoch = state.get("epoch", 0)
        
        # Optimizer state is loaded externally if needed, 
        # or we can assume re-initialization for new calibration runs.

    def step_calibrate(self, logits: torch.Tensor, candidates: torch.Tensor, loss_fn: nn.Module) -> float:
        """
        Performs a single optimization step for calibration.
        
        Args:
            logits: Output from the frozen main model.
            candidates: Ground truth multi-hot labels.
            loss_fn: Calibration loss function.
            
        Returns:
            loss value (float)
        """
        if self.optimizer is None:
            raise RuntimeError("Optimizer has not been initialized for CalibrationEntity.")

        self.network.train()
        self.optimizer.zero_grad()
        
        scaled_logits = self.network(logits)
        loss = loss_fn(scaled_logits, candidates)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()