from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from hint.foundation.interfaces import Registry, TelemetryObserver

class BaseComponent(ABC):
    """Shared base for training components.

    Manages common infrastructure such as the artifact registry, telemetry
    observer, and device configuration for trainers and evaluators.
    """
    def __init__(self, registry: Registry, observer: TelemetryObserver, device: str):
        self.registry = registry
        self.observer = observer
        self.device = device

class BaseTrainer(BaseComponent):
    """Abstract trainer that encapsulates the training loop contract.

    Concrete implementations should run optimization and validation cycles.
    """
    @abstractmethod
    def train(self, train_loader: Any, val_loader: Any, evaluator: 'BaseEvaluator', **kwargs) -> None:
        """Run the training loop over the provided loaders.

        Args:
            train_loader (Any): Training data loader.
            val_loader (Any): Validation data loader.
            evaluator (BaseEvaluator): Evaluator used during validation.
        """
        pass

class BaseEvaluator(BaseComponent):
    """Abstract evaluator responsible for metrics and validation logic."""
    @abstractmethod
    def evaluate(self, loader: Any, **kwargs) -> Dict[str, float]:
        """Run evaluation and return metrics.

        Args:
            loader (Any): Evaluation data loader.

        Returns:
            Dict[str, float]: Metric name to value mapping.
        """
        pass

class BaseDomainService(ABC):
    """Orchestrator that prepares data and runs domain workflows.

    Coordinates component assembly and execution order for a service.
    """
    def __init__(self, observer: TelemetryObserver):
        self.observer = observer

    @abstractmethod
    def execute(self) -> None:
        """Execute the service pipeline."""
        pass
