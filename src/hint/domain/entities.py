import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from torch_ema import ExponentialMovingAverage
from abc import ABC, abstractmethod

class TrainableEntity(ABC):
    """Abstract base class for trainable models with state."""
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
    """Entity for ICD coding model (Ensemble of BERT heads)."""
    def __init__(self, head1: nn.Module, head2: nn.Module, stacker: Any):
        super().__init__("icd_ensemble")
        self.head1 = head1
        self.head2 = head2
        self.stacker = stacker
        
    def state_dict(self) -> Dict[str, Any]:
        return {
            "head1": self.head1.state_dict(),
            "head2": self.head2.state_dict(),
            "best_metric": self.best_metric,
            "epoch": self.epoch
        }
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.head1.load_state_dict(state["head1"])
        self.head2.load_state_dict(state["head2"])
        self.best_metric = state.get("best_metric", 0.0)
        self.epoch = state.get("epoch", 0)

    def to(self, device: str) -> None:
        self.head1.to(device)
        self.head2.to(device)

class InterventionModelEntity(TrainableEntity):
    """Entity for Intervention Prediction CNN with EMA and Calibration."""
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
