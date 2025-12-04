from abc import ABC, abstractmethod
from typing import Any, Iterator, ContextManager, Optional
from .dtos import TensorBatch

class TelemetryObserver(ABC):
    """
    Abstract interface for observability (logging, metrics, tracing).

    This interface decouples the business logic from specific logging libraries
    like Loguru or Rich.
    """
    @abstractmethod
    def log(self, level: str, message: str) -> None:
        """
        Log a message with a specific severity level.

        Args:
            level: Severity level (e.g., 'INFO', 'ERROR').
            message: The message to log.
        """
        pass

    @abstractmethod
    def track_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """
        Track a numerical metric.

        Args:
            name: Name of the metric (e.g., 'loss').
            value: Value of the metric.
            step: Current step or epoch number.
        """
        pass
    
    @abstractmethod
    def trace(self, span_name: str) -> ContextManager:
        """
        Create a context manager for tracing a block of code.

        Args:
            span_name: Name of the span/operation to trace.

        Returns:
            A context manager that measures execution time.
        """
        pass

class StreamingSource(ABC):
    """
    Abstract interface for streaming data batches.

    This interface hides the details of data loading (e.g., HDF5, CSV, SQL).
    """
    @abstractmethod
    def stream_batches(self) -> Iterator[TensorBatch]:
        """
        Yield batches of data.

        Returns:
            An iterator yielding TensorBatch objects.
        """
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """
        Return the total number of batches.

        Returns:
            Total number of batches available.
        """
        pass

class ModelRegistry(ABC):
    """
    Abstract interface for saving and loading model artifacts.
    """
    @abstractmethod
    def save(self, entity: Any, tag: str = "latest") -> None:
        """
        Save the model entity.

        Args:
            entity: The entity to save.
            tag: Version tag (e.g., 'best', 'latest').
        """
        pass

    @abstractmethod
    def load(self, entity_id: str, tag: str = "latest") -> Any:
        """
        Load the model entity.

        Args:
            entity_id: The unique identifier of the entity.
            tag: Version tag to load.

        Returns:
            The loaded artifact (usually a state dictionary).
        """
        pass