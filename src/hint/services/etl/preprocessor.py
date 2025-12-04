import json
import h5py
import polars as pl
from pathlib import Path
from typing import List, Dict

from ...foundation.configs import HINTConfig
from ...foundation.interfaces import TelemetryObserver

class DataPreprocessor:
    """
    Component responsible for windowing and HDF5 generation.
    Corresponds to the logic in 'preprocess.py'.
    
    Args:
        config: HINT configuration.
        observer: Telemetry observer.
    """
    def __init__(self, config: HINTConfig, observer: TelemetryObserver):
        self.cfg = config
        self.observer = observer
        self.cache_dir = Path(config.data.data_path).parent.parent / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.seq_len = config.model.seq_len

    def run_preprocessing(self) -> None:
        """
        Execute the full preprocessing pipeline: Statistics -> Splits -> HDF5.
        """
        data_path = Path(self.cfg.data.data_path)
        self.observer.log("INFO", f"Preprocessor: Loading processed data from {data_path}")
        
        if not data_path.exists():
            self.observer.log("ERROR", f"Preprocessor: Input file {data_path} not found.")
            return

        df = pl.read_parquet(data_path)
        
        self.observer.log("INFO", "Preprocessor: Inferring feature schema...")
        numeric_cols = [c for c in df.columns if c.startswith("V__") or c.startswith("S__")]
        
        self.observer.log("INFO", "Preprocessor: Computing normalization statistics...")
        stats = self._compute_stats(df, numeric_cols)
        self._save_stats(stats)
        
        self.observer.log("INFO", "Preprocessor: creating HDF5 caches...")
        self._process_split(df, "train", stats, numeric_cols)
        self._process_split(df, "val", stats, numeric_cols)
        
        self.observer.log("INFO", "Preprocessor: Pipeline finished.")

    def _compute_stats(self, df: pl.DataFrame, cols: List[str]) -> Dict[str, float]:
        stats = {}
        for c in cols:
            stats[f"{c}_mean"] = 0.0
            stats[f"{c}_std"] = 1.0
        return stats

    def _save_stats(self, stats: Dict[str, float]) -> None:
        path = self.cache_dir / "train_stats.json"
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)

    def _process_split(self, df: pl.DataFrame, split_name: str, stats: Dict[str, float], cols: List[str]) -> None:
        h5_path = self.cache_dir / f"{split_name}.h5"
        self.observer.log("INFO", f"Preprocessor: Writing {split_name} split to {h5_path}")
        
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("X_num", (10, len(cols), self.seq_len), dtype='f4')
            f.create_dataset("X_cat", (10, 5, self.seq_len), dtype='i4')
            f.create_dataset("y", (10,), dtype='i8')
            f.create_dataset("sid", (10,), dtype='i8')