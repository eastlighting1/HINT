from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Union
from pathlib import Path
import torch

class TelemetryObserver(Protocol):
    """Telemetry adapter for logs, metrics, and progress reporting.

    Defines the contract for emitting structured messages, tracking metrics,
    and creating progress displays within services and pipelines.
    """
    def log(self, level: str, message: str) -> None: ...
    def track_metric(self, name: str, value: float, step: int) -> None: ...
    def create_progress(self, desc: str, total: int) -> Any: ...

class Registry(ABC):
    """Artifact store interface for models, dataframes, and metadata.

    Provides a unified API for saving and loading training artifacts across
    the application lifecycle.
    """
    @abstractmethod
    def save_model(self, state_dict: Dict[str, Any], name: str, tag: str) -> Path: ...
    @abstractmethod
    def load_model(self, name: str, tag: str, device: str) -> Dict[str, Any]: ...
    @abstractmethod
    def save_dataframe(self, df: Any, name: str) -> Path: ...
    @abstractmethod
    def load_dataframe(self, name: str) -> Any: ...
    @abstractmethod
    def save_json(self, data: Dict[str, Any], name: str) -> Path: ...
    @abstractmethod
    def load_json(self, name: str) -> Dict[str, Any]: ...
    @abstractmethod
    def get_artifact_path(self, name: str) -> Path: ...

class PipelineComponent(ABC):
    """ETL pipeline step interface.

    Each component encapsulates one stage in the ETL flow and exposes a
    single execution entry point.
    """
    @abstractmethod
    def execute(self) -> None: ...

class StreamingSource(ABC):
    """Streaming data source interface.

    Supports iterable access to large datasets and exposes a length for
    progress and batching.
    """
    @abstractmethod
    def __len__(self) -> int: ...
    @abstractmethod
    def __iter__(self): ...
