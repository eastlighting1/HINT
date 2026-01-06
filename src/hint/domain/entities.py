import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from torch_ema import ExponentialMovingAverage
from abc import ABC, abstractmethod

class TrainableEntity(ABC):
    """Abstract base class for trainable models with state.

    Tracks training progress metadata alongside model parameters.
    """
    def __init__(self, name: str):
        self.name = name
        self.epoch: int = 0
        self.global_step: int = 0
        self.best_metric: float = -float('inf')
        self.network: Optional[nn.Module] = None
    
    @abstractmethod
    def state_dict(self) -> Dict[str, Any]: ...
    
    @abstractmethod
    def load_state_dict(self, state: Dict[str, Any]) -> None: ...

class ICDModelEntity(TrainableEntity):
    """Entity wrapper for ICD coding models.

    Stores model state and training metadata for ICD classifiers.
    """
    def __init__(self, model: nn.Module):
        super().__init__("icd_entity")
        self.model = model
        self.network = model
        
    def state_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "best_metric": self.best_metric,
            "epoch": self.epoch
        }
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.model.load_state_dict(state["model"])
        self.best_metric = state.get("best_metric", 0.0)
        self.epoch = state.get("epoch", 0)

    def to(self, device: str) -> None:
        self.model.to(device)

    def forward(self, *args, **kwargs):
        """Delegate the forward call to the underlying model."""
        return self.model(*args, **kwargs)

class InterventionModelEntity(TrainableEntity):
    """Entity for intervention prediction models with EMA tracking."""
    def __init__(self, network: nn.Module, ema_decay: float = 0.999):
        super().__init__("intervention_cnn")
        self.network = network
        self.ema = ExponentialMovingAverage(network.parameters(), decay=ema_decay)
        self.temperature: float = 1.0
        self.thresholds: Optional[Any] = None

    def update_ema(self) -> None:
        self.ema.update(self.network.parameters())

    def state_dict(self) -> Dict[str, Any]:
        with self.ema.average_parameters():
            net_state = self.network.state_dict()
        return {
            "network": net_state,
            "temperature": self.temperature,
            "thresholds": self.thresholds,
            "best_metric": self.best_metric,
            "epoch": self.epoch
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.network.load_state_dict(state["network"])
        self.temperature = state.get("temperature", 1.0)
        self.thresholds = state.get("thresholds", None)
        self.best_metric = state.get("best_metric", 0.0)
        self.epoch = state.get("epoch", 0)
        self.ema = ExponentialMovingAverage(self.network.parameters(), decay=0.999) 

    def to(self, device: str) -> None:
        self.network.to(device)
        self.ema.to(device)
