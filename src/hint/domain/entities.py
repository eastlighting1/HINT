import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from torch_ema import ExponentialMovingAverage
from abc import ABC, abstractmethod

class TrainableEntity(ABC):
    """Base class for trainable entities in the pipeline.

    This class tracks training metadata and exposes serialization hooks
    for checkpoints.

    Attributes:
        name (str): Entity identifier used in artifacts.
        epoch (int): Current training epoch.
        global_step (int): Global step counter.
        best_metric (float): Best metric observed so far.
        network (Optional[nn.Module]): Underlying model reference.
    """
    def __init__(self, name: str):
        """Initialize the entity with a name and training counters.

        Args:
            name (str): Human-readable entity name.
        """
        self.name = name
        self.epoch: int = 0
        self.global_step: int = 0
        self.best_metric: float = -float('inf')
        self.network: Optional[nn.Module] = None
    
    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Serialize the entity state for checkpointing.

        Returns:
            Dict[str, Any]: Serialized state.
        """
        raise NotImplementedError
    
    @abstractmethod
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore the entity state from a checkpoint.

        Args:
            state (Dict[str, Any]): Serialized state to load.
        """
        raise NotImplementedError

class ICDModelEntity(TrainableEntity):
    """Wrapper for ICD prediction models.

    This entity stores the model and exposes serialization helpers
    compatible with the training workflow.

    Attributes:
        model (nn.Module): ICD model implementation.
        network (nn.Module): Alias for the underlying model.
    """
    def __init__(self, model: nn.Module):
        """Create an ICD entity around a model instance.

        Args:
            model (nn.Module): ICD model to wrap.
        """
        super().__init__("icd_entity")
        self.model = model
        self.network = model
        
    def state_dict(self) -> Dict[str, Any]:
        """Serialize model parameters and training metadata.

        Returns:
            Dict[str, Any]: State dictionary including metrics.
        """
        return {
            "model": self.model.state_dict(),
            "best_metric": self.best_metric,
            "epoch": self.epoch
        }
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore model parameters and metadata.

        Args:
            state (Dict[str, Any]): Serialized checkpoint state.
        """
        self.model.load_state_dict(state["model"])
        self.best_metric = state.get("best_metric", 0.0)
        self.epoch = state.get("epoch", 0)

    def to(self, device: str) -> None:
        """Move the model to the specified device.

        Args:
            device (str): Target device identifier.
        """
        self.model.to(device)

    def forward(self, *args, **kwargs):
        """Delegate the forward pass to the wrapped model.

        Args:
            *args (Any): Positional arguments for the model.
            **kwargs (Any): Keyword arguments for the model.

        Returns:
            Any: Model outputs.
        """
        return self.model(*args, **kwargs)

class InterventionModelEntity(TrainableEntity):
    """Wrapper for intervention prediction models with EMA support.

    This entity maintains an exponential moving average of model weights
    and additional calibration metadata.

    Attributes:
        network (nn.Module): Intervention model implementation.
        ema (ExponentialMovingAverage): EMA helper for parameters.
        temperature (float): Calibration temperature value.
        thresholds (Optional[Any]): Optional decision thresholds.
    """
    def __init__(self, network: nn.Module, ema_decay: float = 0.999):
        """Create an intervention entity and EMA tracker.

        Args:
            network (nn.Module): Intervention model to wrap.
            ema_decay (float): EMA decay factor.
        """
        super().__init__("intervention_cnn")
        self.network = network
        self.ema = ExponentialMovingAverage(network.parameters(), decay=ema_decay)
        self.temperature: float = 1.0
        self.thresholds: Optional[Any] = None

    def update_ema(self) -> None:
        """Update the exponential moving average of parameters."""
        self.ema.update(self.network.parameters())

    def state_dict(self) -> Dict[str, Any]:
        """Serialize EMA parameters and calibration metadata.

        Returns:
            Dict[str, Any]: State dictionary for checkpointing.
        """
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
        """Restore the entity from a serialized checkpoint.

        Args:
            state (Dict[str, Any]): Serialized checkpoint state.
        """
        self.network.load_state_dict(state["network"])
        self.temperature = state.get("temperature", 1.0)
        self.thresholds = state.get("thresholds", None)
        self.best_metric = state.get("best_metric", 0.0)
        self.epoch = state.get("epoch", 0)
        self.ema = ExponentialMovingAverage(self.network.parameters(), decay=0.999) 

    def to(self, device: str) -> None:
        """Move the model and EMA buffers to the target device.

        Args:
            device (str): Target device identifier.
        """
        self.network.to(device)
        self.ema.to(device)
