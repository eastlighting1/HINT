import json
import torch
import polars as pl
import joblib
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..foundation.interfaces import Registry

class FileSystemRegistry(Registry):
    """
    Implementation of Registry that stores artifacts on the local file system.
    """
    def __init__(self, base_dir: Union[str, Path]):
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
        """
        If name is a Path object or contains separator, use it directly (relative to CWD).
        Otherwise, prepend the default category directory.
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
        return self._resolve_path(name, "data")

    def save_model(self, state_dict: Dict[str, Any], name: str, tag: str) -> Path:
        filename = f"{name}_{tag}.pt"
        path = self._resolve_path(filename, "checkpoints")
        torch.save(state_dict, path)
        return path

    def load_model(self, name: str, tag: str, device: str) -> Dict[str, Any]:
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
        path = self._resolve_path(name, "data")
        if isinstance(df, pl.DataFrame):
            df.write_parquet(path)
        else:
            raise ValueError("Only Polars DataFrame supported.")
        return path

    def load_dataframe(self, name: Union[str, Path]) -> pl.DataFrame:
        path = self._resolve_path(name, "data")
        if not path.exists():
            raise FileNotFoundError(f"Data artifact {name} not found at {path}")
        return pl.read_parquet(path)

    def save_labels(self, df: Any, name: Union[str, Path]) -> Path:
        """Specifically saves label dataframes."""
        return self.save_dataframe(df, name)

    def load_labels(self, name: Union[str, Path]) -> pl.DataFrame:
        """Specifically loads label dataframes."""
        return self.load_dataframe(name)

    def save_json(self, data: Dict[str, Any], name: Union[str, Path]) -> Path:
        path = self._resolve_path(name, "metrics")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    def load_json(self, name: Union[str, Path]) -> Dict[str, Any]:
        path = self._resolve_path(name, "metrics")
        if not path.exists():
            raise FileNotFoundError(f"JSON artifact {name} not found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
            
    def save_sklearn(self, model: Any, name: str) -> Path:
        path = self._resolve_path(f"{name}.joblib", "checkpoints")
        joblib.dump(model, path)
        return path
        
    def load_sklearn(self, name: str) -> Any:
        path = self._resolve_path(f"{name}.joblib", "checkpoints")
        if not path.exists():
            raise FileNotFoundError(f"Sklearn artifact {name} not found at {path}")
        return joblib.load(path)