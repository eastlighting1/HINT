import h5py
import torch
import numpy as np
import polars as pl
from pathlib import Path
from typing import Optional, List, Any, Union, Dict, Generator
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from ..foundation.interfaces import StreamingSource
from ..foundation.dtos import TensorBatch
from ..foundation.exceptions import DataError

def _to_python_list(val: Any) -> List[str]:
    """Normalize a value to a list of strings.

    Handles numpy arrays, Python lists, and stringified list values.

    Args:
        val (Any): Input value to normalize.

    Returns:
        List[str]: List of string tokens.
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
    """Batch ICD samples and optionally tokenize text fields.

    Args:
        batch (Any): Batch of sample dictionaries.
        tokenizer (Any): Optional tokenizer instance.
        max_length (int): Maximum token length.

    Returns:
        Any: Collated batch with optional tokenized fields.
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

class ICDDataset(Dataset):
    """Dataset wrapper for numerical features and ICD code lists.

    Used by ICDService for legacy training and evaluation paths.
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

def collate_tensor_batch(batch: List[TensorBatch]) -> TensorBatch:
    """Stack TensorBatch objects into a single batch.

    Args:
        batch (List[TensorBatch]): List of TensorBatch objects.

    Returns:
        TensorBatch: Stacked batch with optional dynamic fields.
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

                                              
    delta = None
    if hasattr(batch[0], "delta") and batch[0].delta is not None:
        delta = torch.stack([b.delta for b in batch])
        
    x_icd = None
    if batch[0].x_icd is not None:
        x_icd = torch.stack([b.x_icd for b in batch])

                              
    input_ids = None
    if getattr(batch[0], "input_ids", None) is not None:
        input_ids = torch.stack([b.input_ids for b in batch])

    attention_mask = None
    if getattr(batch[0], "attention_mask", None) is not None:
        attention_mask = torch.stack([b.attention_mask for b in batch])
        
    candidates = None
    if getattr(batch[0], "candidates", None) is not None:
        candidates = torch.stack([b.candidates for b in batch])

    tb = TensorBatch(
        x_num=x_num,
        x_cat=x_cat,
        y=y,
        ids=ids,
        mask=mask,
        x_icd=x_icd,
        input_ids=input_ids,
        attention_mask=attention_mask,
        candidates=candidates
    )
    
                               
    if delta is not None:
        setattr(tb, "delta", delta)
        
    return tb

class ParquetSource(StreamingSource):
    """Streaming source for parquet files.

    Used in legacy ICD paths and factory loading.
    """
    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            if not self.file_path.is_absolute():
                self.file_path = Path.cwd() / self.file_path
                
            if not self.file_path.exists():
                raise DataError(f"Parquet file not found at {self.file_path}")
        
        self._df: Optional[pl.DataFrame] = None
        
    def load(self) -> pl.DataFrame:
        try:
            if self._df is None:
                self._df = pl.read_parquet(self.file_path)
            return self._df
        except Exception as e:
            raise DataError(f"Failed to load parquet file: {e}")

    def __iter__(self) -> Generator[Dict[str, Any], None, None]:
        try:
            if self._df is None:
                self._df = pl.read_parquet(self.file_path)
            for row in self._df.iter_rows(named=True):
                yield row
        except Exception as e:
            raise DataError(f"Failed to stream data: {e}")

    def __len__(self) -> int:
        """Return the number of rows in the parquet file.

        Returns:
            int: Row count in the parquet file.
        """
        try:
            return pl.scan_parquet(self.file_path).select(pl.len()).collect().item()
        except Exception as e:
            raise DataError(f"Failed to get length of parquet file: {e}")


class HDF5StreamingSource(Dataset):
    """Optimized HDF5 source for ICD coding tasks.

    Reads flat-schema HDF5 data, converts float16 to float32, computes
    masks and deltas for GRU-D, and exposes PLL fields when available.
    """
    def __init__(self, h5_path: Path, seq_len: int = 0, label_key: str = "y"):
        self.h5_path = Path(h5_path)
        self.h5_file: Optional[h5py.File] = None
        self._len = 0
        self.label_key = label_key
        
        if not self.h5_path.exists():
            raise DataError(f"HDF5 file not found at {self.h5_path}")

        try:
            with h5py.File(self.h5_path, "r") as f:
                if "X_num" not in f:
                    raise DataError(f"HDF5 missing required dataset X_num in {self.h5_path}")
                self._len = len(f["X_num"])
        except Exception as e:
            raise DataError(f"Failed to initialize HDF5 source: {e}")

    def __len__(self) -> int:
        return self._len

    def _compute_delta(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute time intervals between observations.

        Args:
            mask (torch.Tensor): (Channels, Time) tensor with 1=observed, 0=missing.

        Returns:
            torch.Tensor: (Channels, Time) delta tensor.
        """
        channels, seq_len = mask.shape
                                                
        m_np = mask.numpy()
        d_np = np.zeros_like(m_np, dtype=np.float32)
        
                                                       
                                                                       
        for c in range(channels):
            last_valid = 0.0
            for t in range(seq_len):
                if m_np[c, t] == 1:
                    last_valid = 0.0
                else:
                    last_valid += 1.0
                d_np[c, t] = last_valid + 1.0                      
                
        return torch.from_numpy(d_np)

    def __getitem__(self, idx: int) -> TensorBatch:
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")

                                  
                                                                     
        x_num_np = self.h5_file["X_num"][idx]
        x_num = torch.from_numpy(x_num_np).float()
        
                                       
                                                 
        is_nan = torch.isnan(x_num)
        mask = (~is_nan).float()
        
                        
                                                                      
        x_num = torch.nan_to_num(x_num, nan=0.0)
        
                                                     
        delta = self._compute_delta(mask)

                         
        x_cat = None
        if "X_cat" in self.h5_file:
            x_cat = torch.from_numpy(self.h5_file["X_cat"][idx]).long()
        
        y_val = self.h5_file[self.label_key][idx]
        
        # [FIX] Check dimensions to handle both scalar labels (ICD) and time-series/vector labels (Ventilation)
        if y_val.ndim == 0:
            # Scalar label for classification (e.g. ICD code index)
            y = torch.tensor(int(y_val), dtype=torch.long)
        else:
            # Array label for regression or dense prediction (e.g. Ventilation mask)
            y = torch.from_numpy(y_val).float()
        
        sid = -1
        if "sid" in self.h5_file:
            sid = int(self.h5_file["sid"][idx])
        sid_tensor = torch.tensor(sid, dtype=torch.long)
        
        x_icd = None
        if "X_icd" in self.h5_file:
            x_icd = torch.from_numpy(self.h5_file["X_icd"][idx]).float()

                                       
        input_ids = None
        if "input_ids" in self.h5_file:
             input_ids = torch.from_numpy(self.h5_file["input_ids"][idx]).long()
        
        attention_mask = None
        if "attention_mask" in self.h5_file:
             attention_mask = torch.from_numpy(self.h5_file["attention_mask"][idx]).long()
             
        candidates = None
        if "candidates" in self.h5_file:
             candidates = torch.from_numpy(self.h5_file["candidates"][idx]).long()

                                  
        tb = TensorBatch(
            x_num=x_num,
            x_cat=x_cat,
            y=y,
            ids=sid_tensor,
            mask=mask,
            x_icd=x_icd,
            input_ids=input_ids,
            attention_mask=attention_mask,
            candidates=candidates
        )
        
                                  
        setattr(tb, "delta", delta)
        
        return tb
    
    def get_real_vocab_sizes(self) -> List[int]:
        should_close = False
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")
            should_close = True
        try:
            if "X_cat" not in self.h5_file:
                return []
                
            data = self.h5_file["X_cat"][:]
            if data.size == 0: return []
            
            # [FIX] Dynamically handle 2D (Static) or 3D (Temporal) categorical features
            if data.ndim == 2:
                # Shape: (Samples, Features) -> Max along axis 0
                max_indices = np.max(data, axis=0)
            else:
                # Shape: (Samples, Features, Time) -> Max along axis 0 and 2
                max_indices = np.max(data, axis=(0, 2))
                
            return [int(s) + 1 for s in max_indices]
        finally:
            if should_close: self.close()

    def close(self):
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None