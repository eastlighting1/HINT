import h5py
import torch
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from pathlib import Path
from typing import Optional, List, Any, Union, Generator, Dict
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from ..foundation.interfaces import StreamingSource
from ..foundation.dtos import TensorBatch
from ..foundation.exceptions import DataError

def _to_python_list(val: Any) -> List[str]:
    """
    Helper to ensure value is a python list of strings.
    Handles numpy arrays, lists, and string representations.
    """
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            from ast import literal_eval
            parsed = literal_eval(val)
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        except:
            if val.startswith("[") and val.endswith("]"):
                return [t.strip(" '\"") for t in val.strip("[]").split(",") if t.strip()]
            return [val]
    return []

def custom_collate(batch, tokenizer, max_length):
    """
    Collate function for ICD data: batches numerical features and tokenizes text.
    """
    nums, labs, lists, cands = zip(*[(b["num"], b["lab"], b["lst"], b["cand"]) for b in batch])

    texts = []
    for lst in lists:
        if isinstance(lst, (list, tuple, np.ndarray)):
            valid_tokens = [str(t) for t in lst if t]
            texts.append(" ".join(valid_tokens))
        else:
            texts.append("")

    bt = default_collate([{"num": n, "lab": l} for n, l in zip(nums, labs)])
    bt["lst"] = lists
    bt["cand"] = cands

    if tokenizer is not None:
        tok = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        )
        bt["input_ids"] = tok["input_ids"]
        bt["attention_mask"] = tok["attention_mask"]
    
    return bt

def collate_tensor_batch(batch: List[TensorBatch]) -> TensorBatch:
    """
    Custom collate function to stack a list of TensorBatch objects into a single TensorBatch.
    """
    x_num = torch.stack([b.x_num for b in batch])
    y = torch.stack([b.y for b in batch])

    x_cat = None
    if batch[0].x_cat is not None:
        x_cat = torch.stack([b.x_cat for b in batch])
        
    ids = None
    if batch[0].ids is not None:
        ids = torch.stack([b.ids for b in batch])
        
    mask = None
    if batch[0].mask is not None:
        mask = torch.stack([b.mask for b in batch])

    return TensorBatch(
        x_num=x_num,
        x_cat=x_cat,
        y=y,
        ids=ids,
        mask=mask
    )

class ICDDataset(Dataset):
    """
    Dataset wrapper for combined numerical features and ICD code lists.
    Used by ICDService for training and evaluation.
    """
    def __init__(self, df, feats, label_col, list_col, cand_col="candidate_indices"):
        self.X = df[feats].to_numpy(dtype=np.float32, copy=True)
        self.y = df[label_col].astype(np.int64).to_numpy()
        self.lst = df[list_col].tolist()
        
        if cand_col in df.columns:
            self.cand = df[cand_col].tolist()
        else:
            self.cand = [[int(lbl)] for lbl in self.y]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        num = torch.tensor(self.X[i], dtype=torch.float32)
        num = torch.nan_to_num(num, nan=0.0, posinf=0.0, neginf=0.0)
        lab = torch.tensor(self.y[i], dtype=torch.long)
        
        return {
            "num": num,
            "lab": lab,
            "lst": self.lst[i],
            "cand": self.cand[i],
        }

class HDF5StreamingSource(StreamingSource, Dataset):
    """
    Streaming source for pre-windowed ICU time-series stored in HDF5 files.
    Used by CNN TrainingService.
    """
    def __init__(self, h5_path: Path, seq_len: int):
        self.h5_path = h5_path
        self.seq_len = seq_len
        self.h5_file: Optional[h5py.File] = None
        self._len = 0
        
        if not self.h5_path.exists():
            raise DataError(f"HDF5 file not found at {self.h5_path}")

        try:
            with h5py.File(self.h5_path, "r") as f:
                if "X_num" not in f or "X_cat" not in f:
                    raise DataError(f"HDF5 missing required datasets in {self.h5_path}")
                self._len = len(f["X_num"])
        except Exception as e:
            raise DataError(f"Failed to initialize HDF5 source: {e}")

    def get_real_vocab_sizes(self) -> List[int]:
        """
        Scan the categorical tensor to derive the maximum index per feature.

        Returns:
            Vocabulary sizes derived from the stored categorical tensor.
        """
        should_close = False
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")
            should_close = True
        
        try:
            data = self.h5_file["X_cat"][:] 
            max_indices = np.max(data, axis=(0, 2))
            
            real_vocab_sizes = (max_indices + 1).tolist()
            return [int(s) for s in real_vocab_sizes]
            
        except Exception as e:
            print(f"[Warning] Failed to calculate real vocab sizes: {e}")
            return []
        finally:
            if should_close:
                self.close()

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> TensorBatch:
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")

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
        for i in range(len(self)):
            yield self[i]

    def close(self):
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

class ParquetSource(StreamingSource):
    """
    Source for loading tabular data (default Parquet).
    Supports both bulk loading (load) and memory-efficient streaming iteration (__iter__).
    """
    def __init__(self, file_path: Path):
        self.file_path = file_path
        if not self.file_path.exists():
            if not file_path.is_absolute():
                self.file_path = Path.cwd() / file_path
                
            if not self.file_path.exists():
                raise DataError(f"Data file not found at {self.file_path}")
        
    def load(self) -> pl.DataFrame:
        try:
            return pl.read_parquet(self.file_path)
        except Exception as e:
            raise DataError(f"Failed to load parquet file: {e}")

    def __len__(self) -> int:
        try:
            return pl.scan_parquet(self.file_path).select(pl.len()).collect().item()
        except Exception:
            return 0

    def __iter__(self) -> Generator[Dict[str, Any], None, None]:
        try:
            parquet_file = pq.ParquetFile(self.file_path)
            
            for batch in parquet_file.iter_batches():
                batch_df = pl.from_arrow(batch)
                for row in batch_df.iter_rows(named=True):
                    yield row
                    
        except Exception as e:
            raise DataError(f"Failed to stream data from {self.file_path}: {e}")
