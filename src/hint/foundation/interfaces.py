from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Union
from pathlib import Path
import torch

class TelemetryObserver(Protocol):
    """Protocol for logging and progress reporting.

    Implementations should provide structured logging and lightweight
    metric tracking for pipeline stages.
    """
    def log(self, level: str, message: str) -> None:
        """Record a log message with the provided severity level.

        Args:
            level (str): Logging level name.
            message (str): Log message content.
        """
        ...

    def track_metric(self, name: str, value: float, step: int) -> None:
        """Record a numeric metric at a given step.

        Args:
            name (str): Metric name.
            value (float): Metric value.
            step (int): Step index associated with the metric.
        """
        ...

    def create_progress(self, desc: str, total: int) -> Any:
        """Create a progress tracker for long-running steps.

        Args:
            desc (str): Human-readable task description.
            total (int): Total units to track.

        Returns:
            Any: Progress tracker instance.
        """
        ...

class Registry(ABC):
    """Abstract interface for model and artifact storage."""
    @abstractmethod
    def save_model(self, state_dict: Dict[str, Any], name: str, tag: str) -> Path:
        """Persist a model state dictionary.

        Args:
            state_dict (Dict[str, Any]): Model state to save.
            name (str): Artifact base name.
            tag (str): Version tag or checkpoint label.

        Returns:
            Path: Location of the saved artifact.
        """
        raise NotImplementedError

    @abstractmethod
    def load_model(self, name: str, tag: str, device: str) -> Dict[str, Any]:
        """Load a model state dictionary from storage.

        Args:
            name (str): Artifact base name.
            tag (str): Version tag or checkpoint label.
            device (str): Target device for tensor mapping.

        Returns:
            Dict[str, Any]: Loaded model state.
        """
        raise NotImplementedError

    @abstractmethod
    def save_dataframe(self, df: Any, name: str) -> Path:
        """Persist a dataframe artifact.

        Args:
            df (Any): Dataframe to save.
            name (str): Artifact name or file path.

        Returns:
            Path: Location of the saved artifact.
        """
        raise NotImplementedError

    @abstractmethod
    def load_dataframe(self, name: str) -> Any:
        """Load a dataframe artifact.

        Args:
            name (str): Artifact name or file path.

        Returns:
            Any: Loaded dataframe.
        """
        raise NotImplementedError

    @abstractmethod
    def save_json(self, data: Dict[str, Any], name: str) -> Path:
        """Persist a JSON-serializable artifact.

        Args:
            data (Dict[str, Any]): JSON-serializable data.
            name (str): Artifact name or file path.

        Returns:
            Path: Location of the saved artifact.
        """
        raise NotImplementedError

    @abstractmethod
    def load_json(self, name: str) -> Dict[str, Any]:
        """Load a JSON artifact.

        Args:
            name (str): Artifact name or file path.

        Returns:
            Dict[str, Any]: Parsed JSON object.
        """
        raise NotImplementedError

    @abstractmethod
    def get_artifact_path(self, name: str) -> Path:
        """Resolve an artifact path without loading it.

        Args:
            name (str): Artifact name or file path.

        Returns:
            Path: Resolved artifact path.
        """
        raise NotImplementedError

class PipelineComponent(ABC):
    """Abstract unit of work within the ETL pipeline."""
    @abstractmethod
    def execute(self) -> None:
        """Execute the component's work in the pipeline."""
        raise NotImplementedError

class StreamingSource(ABC):
    """Abstract iterable data source."""
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of items available in the source."""
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        """Yield items from the source."""
        raise NotImplementedError
