import json
import torch
import polars as pl
import joblib
from pathlib import Path
from typing import Any, Dict, Optional, Union

from hint.foundation.interfaces import Registry
from hint.foundation.exceptions import ConfigurationError

class FileSystemRegistry(Registry):
    """
    Implementation of Registry that stores artifacts on the local file system.
    """
    def __init__(self, base_dir: Union[str, Path]):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories structure
        self.dirs = {
            "checkpoints": self.base_dir / "checkpoints",
            "data": self.base_dir / "data",
            "metrics": self.base_dir / "metrics",
            "configs": self.base_dir / "configs"
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    def get_artifact_path(self, name: str) -> Path:
        # Simple heuristic to find file
        for d in self.dirs.values():
            if (d / name).exists():
                return d / name
        return self.base_dir / name

    def save_model(self, state_dict: Dict[str, Any], name: str, tag: str) -> Path:
        filename = f"{name}_{tag}.pt"
        path = self.dirs["checkpoints"] / filename
        torch.save(state_dict, path)
        return path

    def load_model(self, name: str, tag: str, device: str) -> Dict[str, Any]:
        filename = f"{name}_{tag}.pt"
        path = self.dirs["checkpoints"] / filename
        if not path.exists():
            # Fallback search
            path = self.get_artifact_path(filename)
            
        if not path.exists():
            raise FileNotFoundError(f"Model artifact {filename} not found.")
            
        return torch.load(path, map_location=device)

    def save_dataframe(self, df: Any, name: str) -> Path:
        path = self.dirs["data"] / name
        if isinstance(df, pl.DataFrame):
            df.write_parquet(path)
        else:
            # Assume pandas if not polars, or raise error
            raise ValueError("Only Polars DataFrame supported for now.")
        return path

    def load_dataframe(self, name: str) -> pl.DataFrame:
        path = self.dirs["data"] / name
        if not path.exists():
             path = self.get_artifact_path(name)
             
        if not path.exists():
            raise FileNotFoundError(f"Data artifact {name} not found.")
        return pl.read_parquet(path)

    def save_json(self, data: Dict[str, Any], name: str) -> Path:
        path = self.dirs["metrics"] / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    def load_json(self, name: str) -> Dict[str, Any]:
        path = self.dirs["metrics"] / name
        if not path.exists():
             path = self.get_artifact_path(name)
             
        if not path.exists():
            raise FileNotFoundError(f"JSON artifact {name} not found.")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
            
    def save_sklearn(self, model: Any, name: str) -> Path:
        """Helper for saving sklearn/xgboost models via joblib"""
        path = self.dirs["checkpoints"] / f"{name}.joblib"
        joblib.dump(model, path)
        return path
        
    def load_sklearn(self, name: str) -> Any:
        path = self.dirs["checkpoints"] / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Sklearn artifact {name} not found.")
        return joblib.load(path)
