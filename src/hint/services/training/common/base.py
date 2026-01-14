"""Summary of the base module.

Longer description of the module purpose and usage.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from hint.foundation.interfaces import Registry, TelemetryObserver



class BaseComponent(ABC):

    """Summary of BaseComponent purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    device (Any): Description of device.
    observer (Any): Description of observer.
    registry (Any): Description of registry.
    """

    def __init__(self, registry: Registry, observer: TelemetryObserver, device: str):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
        registry (Any): Description of registry.
        observer (Any): Description of observer.
        device (Any): Description of device.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        self.registry = registry

        self.observer = observer

        self.device = device



class BaseTrainer(BaseComponent):

    """Summary of BaseTrainer purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    None (None): No documented attributes.
    """

    @abstractmethod

    def train(self, train_loader: Any, val_loader: Any, evaluator: 'BaseEvaluator', **kwargs) -> None:

        """Summary of train.
        
        Longer description of the train behavior and usage.
        
        Args:
        train_loader (Any): Description of train_loader.
        val_loader (Any): Description of val_loader.
        evaluator (Any): Description of evaluator.
        kwargs (Any): Description of kwargs.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        raise NotImplementedError



class BaseEvaluator(BaseComponent):

    """Summary of BaseEvaluator purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    None (None): No documented attributes.
    """

    @abstractmethod

    def evaluate(self, loader: Any, **kwargs) -> Dict[str, float]:

        """Summary of evaluate.
        
        Longer description of the evaluate behavior and usage.
        
        Args:
        loader (Any): Description of loader.
        kwargs (Any): Description of kwargs.
        
        Returns:
        Dict[str, float]: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        raise NotImplementedError



class BaseDomainService(ABC):

    """Summary of BaseDomainService purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    observer (Any): Description of observer.
    """

    def __init__(self, observer: TelemetryObserver):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
        observer (Any): Description of observer.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        self.observer = observer



    @abstractmethod

    def execute(self) -> None:

        """Summary of execute.
        
        Longer description of the execute behavior and usage.
        
        Args:
        None (None): This function does not accept arguments.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        raise NotImplementedError
