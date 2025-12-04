import h5py
import torch
import polars as pl
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Generator
from torch.utils.data import Dataset, DataLoader

from hint.foundation.interfaces import StreamingSource
from hint.foundation.dtos import TensorBatch
from hint.foundation.exceptions import DataError

class HDF5StreamingSource(StreamingSource, Dataset):
    """
    Streaming source for pre-windowed ICU time-series stored in HDF5 files.
    Ported from CNN.py's SeqDataset.
    """
    def __init__(self, h5_path: Path, seq_len: int):
        self.h5_path = h5_path
        self.seq_len = seq_len
        self.h5_file: Optional[h5py.File] = None
        self._len = 0
        
        if not self.h5_path.exists():
            raise DataError(f"HDF5 file not found at {self.h5_path}")

        try:
            # Open efficiently to check length without keeping open
            with h5py.File(self.h5_path, "r") as f:
                if "X_num" not in f or "X_cat" not in f:
                    raise DataError(f"HDF5 missing required datasets in {self.h5_path}")
                self._len = len(f["X_num"])
        except Exception as e:
            raise DataError(f"Failed to initialize HDF5 source: {e}")

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> TensorBatch:
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")

        # Lazy loading
        x_num = self.h5_file["X_num"][idx]
        x_cat = self.h5_file["X_cat"][idx]
        y = int(self.h5_file["y"][idx])
        sid = int(self.h5_file["sid"][idx])

        return TensorBatch(
            x_num=torch.from_numpy(x_num).float(),
            x_cat=torch.from_numpy(x_cat).long(),
            y=torch.tensor(y, dtype=torch.long),
            ids=torch.tensor(sid, dtype=torch.long)
        )

    def __iter__(self):
        # Allow iteration directly if needed, though DataLoader is preferred
        for i in range(len(self)):
            yield self[i]

    def close(self):
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

class ParquetSource(StreamingSource):
    """
    Source for loading ICD data from Parquet files.
    """
    def __init__(self, file_path: Path):
        self.file_path = file_path
        if not self.file_path.exists():
            raise DataError(f"Parquet file not found at {self.file_path}")
        
    def load(self) -> pl.DataFrame:
        try:
            return pl.read_parquet(self.file_path)
        except Exception as e:
            raise DataError(f"Failed to load parquet file: {e}")

    def __len__(self) -> int:
        # This might be expensive for large parquet files without scanning
        return pl.scan_parquet(self.file_path).select(pl.len()).collect().item()

    def __iter__(self):
        raise NotImplementedError("ParquetSource is designed for bulk load, not streaming iteration yet.")
