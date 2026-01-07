import json
import torch
import polars as pl
import joblib
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..foundation.interfaces import Registry

class FileSystemRegistry(Registry):
    """Filesystem-backed registry for model and data artifacts.

    This registry stores checkpoints, datasets, metrics, and configs
    under a configurable base directory.

    Attributes:
        base_dir (Path): Root directory for artifacts.
        dirs (Dict[str, Path]): Mapped subdirectories by category.
    """
    def __init__(self, base_dir: Union[str, Path]):
        """Initialize the registry and create subdirectories.

        Args:
            base_dir (Union[str, Path]): Root directory for artifacts.
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
        """Resolve an artifact path, supporting absolute or relative inputs.

        Args:
            name (Union[str, Path]): Artifact name or explicit path.
            category (str): Registry subdirectory name.

        Returns:
            Path: Resolved artifact path.
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
        """Return the resolved path for a data artifact.

        Args:
            name (str): Artifact name.

        Returns:
            Path: Resolved artifact path.
        """
        return self._resolve_path(name, "data")

    def save_model(self, state_dict: Dict[str, Any], name: str, tag: str) -> Path:
        """Save a model checkpoint to disk.

        Args:
            state_dict (Dict[str, Any]): Model state dictionary.
            name (str): Base model name.
            tag (str): Checkpoint label.

        Returns:
            Path: Path to the saved checkpoint.
        """
        filename = f"{name}_{tag}.pt"
        path = self._resolve_path(filename, "checkpoints")
        torch.save(state_dict, path)
        return path

    def load_model(self, name: str, tag: str, device: str) -> Dict[str, Any]:
        """Load a model checkpoint from disk.

        Args:
            name (str): Base model name.
            tag (str): Checkpoint label.
            device (str): Target device for tensor mapping.

        Returns:
            Dict[str, Any]: Loaded model state dictionary.

        Raises:
            FileNotFoundError: If the checkpoint is missing.
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
        """Save a Polars dataframe as parquet.

        Args:
            df (Any): Polars dataframe to save.
            name (Union[str, Path]): Artifact name or path.

        Returns:
            Path: Path to the saved parquet file.

        Raises:
            ValueError: If the dataframe type is unsupported.
        """
        path = self._resolve_path(name, "data")
        if isinstance(df, pl.DataFrame):
            df.write_parquet(path)
        else:
            raise ValueError("Only Polars DataFrame supported.")
        return path

    def load_dataframe(self, name: Union[str, Path]) -> pl.DataFrame:
        """Load a Polars dataframe from parquet.

        Args:
            name (Union[str, Path]): Artifact name or path.

        Returns:
            pl.DataFrame: Loaded dataframe.

        Raises:
            FileNotFoundError: If the parquet file is missing.
        """
        path = self._resolve_path(name, "data")
        if not path.exists():
            raise FileNotFoundError(f"Data artifact {name} not found at {path}")
        return pl.read_parquet(path)

    def save_labels(self, df: Any, name: Union[str, Path]) -> Path:
        """Save labels as a parquet artifact.

        Args:
            df (Any): Labels dataframe.
            name (Union[str, Path]): Artifact name or path.

        Returns:
            Path: Path to the saved labels file.
        """
        return self.save_dataframe(df, name)

    def load_labels(self, name: Union[str, Path]) -> pl.DataFrame:
        """Load labels dataframe from storage.

        Args:
            name (Union[str, Path]): Artifact name or path.

        Returns:
            pl.DataFrame: Loaded labels dataframe.
        """
        return self.load_dataframe(name)

    def save_json(self, data: Dict[str, Any], name: Union[str, Path]) -> Path:
        """Save a JSON metrics or metadata artifact.

        Args:
            data (Dict[str, Any]): JSON-serializable data.
            name (Union[str, Path]): Artifact name or path.

        Returns:
            Path: Path to the saved JSON file.
        """
        path = self._resolve_path(name, "metrics")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    def load_json(self, name: Union[str, Path]) -> Dict[str, Any]:
        """Load a JSON artifact from storage.

        Args:
            name (Union[str, Path]): Artifact name or path.

        Returns:
            Dict[str, Any]: Parsed JSON content.

        Raises:
            FileNotFoundError: If the JSON file is missing.
        """
        path = self._resolve_path(name, "metrics")
        if not path.exists():
            raise FileNotFoundError(f"JSON artifact {name} not found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
            
    def save_sklearn(self, model: Any, name: str) -> Path:
        """Persist a scikit-learn model using joblib.

        Args:
            model (Any): Trained model instance.
            name (str): Artifact base name.

        Returns:
            Path: Path to the saved model file.
        """
        path = self._resolve_path(f"{name}.joblib", "checkpoints")
        joblib.dump(model, path)
        return path
        
    def load_sklearn(self, name: str) -> Any:
        """Load a scikit-learn model from joblib.

        Args:
            name (str): Artifact base name.

        Returns:
            Any: Loaded model instance.

        Raises:
            FileNotFoundError: If the joblib file is missing.
        """
        path = self._resolve_path(f"{name}.joblib", "checkpoints")
        if not path.exists():
            raise FileNotFoundError(f"Sklearn artifact {name} not found at {path}")
        return joblib.load(path)
