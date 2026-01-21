"""Summary of the interfaces module.

Longer description of the module purpose and usage.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

import torch



class TelemetryObserver(Protocol):

    """Summary of TelemetryObserver purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
        None (None): No documented attributes.
    """

    def log(self, level: str, message: str) -> None:

        """Summary of log.
        
        Longer description of the log behavior and usage.
        
        Args:
            level (Any): Description of level.
            message (Any): Description of message.
        
        Returns:
            None: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        ...



    def track_metric(self, name: str, value: float, step: int) -> None:

        """Summary of track_metric.
        
        Longer description of the track_metric behavior and usage.
        
        Args:
            name (Any): Description of name.
            value (Any): Description of value.
            step (Any): Description of step.
        
        Returns:
            None: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        ...



    def create_progress(self, desc: str, total: int) -> Any:

        """Summary of create_progress.
        
        Longer description of the create_progress behavior and usage.
        
        Args:
            desc (Any): Description of desc.
            total (Any): Description of total.
        
        Returns:
            Any: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        ...



class Registry(ABC):

    """Summary of Registry purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
        None (None): No documented attributes.
    """

    @abstractmethod

    def save_model(self, state_dict: Dict[str, Any], name: str, tag: str) -> Path:

        """Summary of save_model.
        
        Longer description of the save_model behavior and usage.
        
        Args:
            state_dict (Any): Description of state_dict.
            name (Any): Description of name.
            tag (Any): Description of tag.
        
        Returns:
            Path: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        raise NotImplementedError



    @abstractmethod

    def load_model(self, name: str, tag: str, device: str) -> Dict[str, Any]:

        """Summary of load_model.
        
        Longer description of the load_model behavior and usage.
        
        Args:
            name (Any): Description of name.
            tag (Any): Description of tag.
            device (Any): Description of device.
        
        Returns:
            Dict[str, Any]: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        raise NotImplementedError



    @abstractmethod

    def save_dataframe(self, df: Any, name: str) -> Path:

        """Summary of save_dataframe.
        
        Longer description of the save_dataframe behavior and usage.
        
        Args:
            df (Any): Description of df.
            name (Any): Description of name.
        
        Returns:
            Path: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        raise NotImplementedError



    @abstractmethod

    def load_dataframe(self, name: str) -> Any:

        """Summary of load_dataframe.
        
        Longer description of the load_dataframe behavior and usage.
        
        Args:
            name (Any): Description of name.
        
        Returns:
            Any: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        raise NotImplementedError



    @abstractmethod

    def save_json(self, data: Dict[str, Any], name: str) -> Path:

        """Summary of save_json.
        
        Longer description of the save_json behavior and usage.
        
        Args:
            data (Any): Description of data.
            name (Any): Description of name.
        
        Returns:
            Path: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        raise NotImplementedError



    @abstractmethod

    def load_json(self, name: str) -> Dict[str, Any]:

        """Summary of load_json.
        
        Longer description of the load_json behavior and usage.
        
        Args:
            name (Any): Description of name.
        
        Returns:
            Dict[str, Any]: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        raise NotImplementedError



    @abstractmethod

    def get_artifact_path(self, name: str) -> Path:

        """Summary of get_artifact_path.
        
        Longer description of the get_artifact_path behavior and usage.
        
        Args:
            name (Any): Description of name.
        
        Returns:
            Path: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        raise NotImplementedError



class PipelineComponent(ABC):

    """Summary of PipelineComponent purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
        None (None): No documented attributes.
    """

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



class StreamingSource(ABC):

    """Summary of StreamingSource purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
        None (None): No documented attributes.
    """

    @abstractmethod

    def __len__(self) -> int:

        """Summary of __len__.
        
        Longer description of the __len__ behavior and usage.
        
        Args:
            None (None): This function does not accept arguments.
        
        Returns:
            int: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        raise NotImplementedError



    @abstractmethod

    def __iter__(self):

        """Summary of __iter__.
        
        Longer description of the __iter__ behavior and usage.
        
        Args:
            None (None): This function does not accept arguments.
        
        Returns:
            Any: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        raise NotImplementedError
