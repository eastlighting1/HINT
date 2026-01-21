"""Summary of the entities module.

Longer description of the module purpose and usage.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch_ema import ExponentialMovingAverage



class TrainableEntity(ABC):

    """Summary of TrainableEntity purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
        best_metric (Any): Description of best_metric.
        epoch (Any): Description of epoch.
        global_step (Any): Description of global_step.
        name (Any): Description of name.
        network (Any): Description of network.
    """

    def __init__(self, name: str):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
            name (Any): Description of name.
        
        Returns:
            None: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        self.name = name

        self.epoch: int = 0

        self.global_step: int = 0

        self.best_metric: float = -float('inf')

        self.network: Optional[nn.Module] = None



    @abstractmethod

    def state_dict(self) -> Dict[str, Any]:

        """Summary of state_dict.
        
        Longer description of the state_dict behavior and usage.
        
        Args:
            None (None): This function does not accept arguments.
        
        Returns:
            Dict[str, Any]: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        raise NotImplementedError



    @abstractmethod

    def load_state_dict(self, state: Dict[str, Any]) -> None:

        """Summary of load_state_dict.
        
        Longer description of the load_state_dict behavior and usage.
        
        Args:
            state (Any): Description of state.
        
        Returns:
            None: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        raise NotImplementedError



class ICDModelEntity(TrainableEntity):

    """Summary of ICDModelEntity purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
        best_metric (Any): Description of best_metric.
        epoch (Any): Description of epoch.
        model (Any): Description of model.
        network (Any): Description of network.
    """

    def __init__(self, model: nn.Module, num_classes: Optional[int] = None):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
            model (Any): Description of model.
        
        Returns:
            None: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        super().__init__("icd_entity")

        self.model = model

        self.network = model

        self.num_classes = num_classes
        self.best_metric = float("inf")



    def state_dict(self) -> Dict[str, Any]:

        """Summary of state_dict.
        
        Longer description of the state_dict behavior and usage.
        
        Args:
            None (None): This function does not accept arguments.
        
        Returns:
            Dict[str, Any]: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        return {

            "model": self.model.state_dict(),

            "best_metric": self.best_metric,

            "epoch": self.epoch,

            "num_classes": self.num_classes,

        }



    def load_state_dict(self, state: Dict[str, Any]) -> None:

        """Summary of load_state_dict.
        
        Longer description of the load_state_dict behavior and usage.
        
        Args:
            state (Any): Description of state.
        
        Returns:
            None: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        self.model.load_state_dict(state["model"])

        self.best_metric = state.get("best_metric", float("inf"))

        self.epoch = state.get("epoch", 0)

        if "num_classes" in state:
            self.num_classes = state["num_classes"]



    def to(self, device: str) -> None:

        """Summary of to.
        
        Longer description of the to behavior and usage.
        
        Args:
            device (Any): Description of device.
        
        Returns:
            None: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        self.model.to(device)



    def forward(self, *args, **kwargs):

        """Summary of forward.
        
        Longer description of the forward behavior and usage.
        
        Args:
            args (Any): Description of args.
            kwargs (Any): Description of kwargs.
        
        Returns:
            Any: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        return self.model(*args, **kwargs)



class InterventionModelEntity(TrainableEntity):

    """Summary of InterventionModelEntity purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
        best_metric (Any): Description of best_metric.
        ema (Any): Description of ema.
        epoch (Any): Description of epoch.
        network (Any): Description of network.
        temperature (Any): Description of temperature.
        thresholds (Any): Description of thresholds.
    """

    def __init__(self, network: nn.Module, ema_decay: float = 0.999):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
            network (Any): Description of network.
            ema_decay (Any): Description of ema_decay.
        
        Returns:
            None: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        super().__init__("intervention_cnn")

        self.network = network

        self.ema = ExponentialMovingAverage(network.parameters(), decay=ema_decay)

        self.temperature: float = 1.0

        self.thresholds: Optional[Any] = None



    def update_ema(self) -> None:

        """Summary of update_ema.
        
        Longer description of the update_ema behavior and usage.
        
        Args:
            None (None): This function does not accept arguments.
        
        Returns:
            None: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        self.ema.update(self.network.parameters())



    def state_dict(self) -> Dict[str, Any]:

        """Summary of state_dict.
        
        Longer description of the state_dict behavior and usage.
        
        Args:
            None (None): This function does not accept arguments.
        
        Returns:
            Dict[str, Any]: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
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

        """Summary of load_state_dict.
        
        Longer description of the load_state_dict behavior and usage.
        
        Args:
            state (Any): Description of state.
        
        Returns:
            None: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        self.network.load_state_dict(state["network"])

        self.temperature = state.get("temperature", 1.0)

        self.thresholds = state.get("thresholds", None)

        self.best_metric = state.get("best_metric", 0.0)

        self.epoch = state.get("epoch", 0)

        self.ema = ExponentialMovingAverage(self.network.parameters(), decay=0.999)



    def to(self, device: str) -> None:

        """Summary of to.
        
        Longer description of the to behavior and usage.
        
        Args:
            device (Any): Description of device.
        
        Returns:
            None: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        self.network.to(device)

        self.ema.to(device)
