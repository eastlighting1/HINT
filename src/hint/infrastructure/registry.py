"""Summary of the registry module.

Longer description of the module purpose and usage.
"""

import json

import torch

import polars as pl

import joblib

from pathlib import Path

from typing import Any, Dict, Optional, Union



from ..foundation.interfaces import Registry



class FileSystemRegistry(Registry):

    """Summary of FileSystemRegistry purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    base_dir (Any): Description of base_dir.
    dirs (Any): Description of dirs.
    """

    def __init__(self, base_dir: Union[str, Path]):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
        base_dir (Any): Description of base_dir.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        self.base_dir = Path(base_dir)

        self.base_dir.mkdir(parents=True, exist_ok=True)



        self.dirs = {

            "checkpoints": self.base_dir / "checkpoints",

            "data": self.base_dir / "data",

            "metrics": self.base_dir / "metrics",

            "configs": self.base_dir / "configs"

        }

        for d in self.dirs.values():

            d.mkdir(parents=True, exist_ok=True)



    def _resolve_path(self, name: Union[str, Path], category: str) -> Path:

        """Summary of _resolve_path.
        
        Longer description of the _resolve_path behavior and usage.
        
        Args:
        name (Any): Description of name.
        category (Any): Description of category.
        
        Returns:
        Path: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        if isinstance(name, Path) or "/" in str(name) or "\\" in str(name):

            path = Path(name)

            if not path.parent.exists() and path.parent != Path('.'):

                try:

                    path.parent.mkdir(parents=True, exist_ok=True)

                except OSError:

                    pass

            return path

        return self.dirs[category] / name



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

        return self._resolve_path(name, "data")



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

        filename = f"{name}_{tag}.pt"

        path = self._resolve_path(filename, "checkpoints")

        torch.save(state_dict, path)

        return path



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

        filename = f"{name}_{tag}.pt"

        path = self._resolve_path(filename, "checkpoints")



        if not path.exists():

            fallback = self.dirs["checkpoints"] / filename

            if fallback.exists():

                path = fallback

            else:

                raise FileNotFoundError(f"Model artifact {filename} not found at {path}")



        return torch.load(path, map_location=device)



    def save_dataframe(self, df: Any, name: Union[str, Path]) -> Path:

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

        path = self._resolve_path(name, "data")

        if isinstance(df, pl.DataFrame):

            df.write_parquet(path)

        else:

            raise ValueError("Only Polars DataFrame supported.")

        return path



    def load_dataframe(self, name: Union[str, Path]) -> pl.DataFrame:

        """Summary of load_dataframe.
        
        Longer description of the load_dataframe behavior and usage.
        
        Args:
        name (Any): Description of name.
        
        Returns:
        pl.DataFrame: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        path = self._resolve_path(name, "data")

        if not path.exists():

            raise FileNotFoundError(f"Data artifact {name} not found at {path}")

        return pl.read_parquet(path)



    def save_labels(self, df: Any, name: Union[str, Path]) -> Path:

        """Summary of save_labels.
        
        Longer description of the save_labels behavior and usage.
        
        Args:
        df (Any): Description of df.
        name (Any): Description of name.
        
        Returns:
        Path: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        return self.save_dataframe(df, name)



    def load_labels(self, name: Union[str, Path]) -> pl.DataFrame:

        """Summary of load_labels.
        
        Longer description of the load_labels behavior and usage.
        
        Args:
        name (Any): Description of name.
        
        Returns:
        pl.DataFrame: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        return self.load_dataframe(name)



    def save_json(self, data: Dict[str, Any], name: Union[str, Path]) -> Path:

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

        path = self._resolve_path(name, "metrics")

        with open(path, "w", encoding="utf-8") as f:

            json.dump(data, f, indent=2, default=str)

        return path



    def load_json(self, name: Union[str, Path]) -> Dict[str, Any]:

        """Summary of load_json.
        
        Longer description of the load_json behavior and usage.
        
        Args:
        name (Any): Description of name.
        
        Returns:
        Dict[str, Any]: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        path = self._resolve_path(name, "metrics")

        if not path.exists():

            raise FileNotFoundError(f"JSON artifact {name} not found at {path}")

        with open(path, "r", encoding="utf-8") as f:

            return json.load(f)



    def save_sklearn(self, model: Any, name: str) -> Path:

        """Summary of save_sklearn.
        
        Longer description of the save_sklearn behavior and usage.
        
        Args:
        model (Any): Description of model.
        name (Any): Description of name.
        
        Returns:
        Path: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        path = self._resolve_path(f"{name}.joblib", "checkpoints")

        joblib.dump(model, path)

        return path



    def load_sklearn(self, name: str) -> Any:

        """Summary of load_sklearn.
        
        Longer description of the load_sklearn behavior and usage.
        
        Args:
        name (Any): Description of name.
        
        Returns:
        Any: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        path = self._resolve_path(f"{name}.joblib", "checkpoints")

        if not path.exists():

            raise FileNotFoundError(f"Sklearn artifact {name} not found at {path}")

        return joblib.load(path)
