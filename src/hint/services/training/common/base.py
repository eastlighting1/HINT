from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from hint.foundation.interfaces import Registry, TelemetryObserver

class BaseComponent(ABC):
    """Base class for training components.

    Attributes:
        registry (Registry): Artifact registry.
        observer (TelemetryObserver): Logging observer.
        device (str): Target device identifier.
    """
    def __init__(self, registry: Registry, observer: TelemetryObserver, device: str):
        """Initialize the component dependencies.

        Args:
            registry (Registry): Artifact registry.
            observer (TelemetryObserver): Logging observer.
            device (str): Target device identifier.
        """
        self.registry = registry
        self.observer = observer
        self.device = device

class BaseTrainer(BaseComponent):
    """Abstract base class for model trainers."""
    @abstractmethod
    def train(self, train_loader: Any, val_loader: Any, evaluator: 'BaseEvaluator', **kwargs) -> None:
        """Train a model using the provided data loaders.

        Args:
            train_loader (Any): Training data loader.
            val_loader (Any): Validation data loader.
            evaluator (BaseEvaluator): Evaluation helper.
            **kwargs (Any): Additional training options.
        """
        raise NotImplementedError

class BaseEvaluator(BaseComponent):
    """Abstract base class for evaluators."""
    @abstractmethod
    def evaluate(self, loader: Any, **kwargs) -> Dict[str, float]:
        """Evaluate a model on the provided data loader.

        Args:
            loader (Any): Evaluation data loader.
            **kwargs (Any): Additional evaluation options.

        Returns:
            Dict[str, float]: Aggregated metrics.
        """
        raise NotImplementedError

class BaseDomainService(ABC):
    """Abstract base class for domain services.

    Attributes:
        observer (TelemetryObserver): Logging observer.
    """
    def __init__(self, observer: TelemetryObserver):
        """Initialize the service with a telemetry observer.

        Args:
            observer (TelemetryObserver): Logging observer.
        """
        self.observer = observer

    @abstractmethod
    def execute(self) -> None:
        """Execute the service workflow."""
        raise NotImplementedError
