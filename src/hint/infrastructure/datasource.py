"""Summary of the datasource module.

Longer description of the module purpose and usage.
"""

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

    """Summary of _to_python_list.
    
    Longer description of the _to_python_list behavior and usage.
    
    Args:
    val (Any): Description of val.
    
    Returns:
    List[str]: Description of the return value.
    
    Raises:
    Exception: Description of why this exception might be raised.
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

        except Exception:

            if val.startswith("[") and val.endswith("]"):

                return [t.strip(" '\"") for t in val.strip("[]").split(",") if t.strip()]

            return [val]

    return []



def custom_collate(batch, tokenizer, max_length):

    """Summary of custom_collate.
    
    Longer description of the custom_collate behavior and usage.
    
    Args:
    batch (Any): Description of batch.
    tokenizer (Any): Description of tokenizer.
    max_length (Any): Description of max_length.
    
    Returns:
    Any: Description of the return value.
    
    Raises:
    Exception: Description of why this exception might be raised.
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

        tok = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=max_length)

        bt["input_ids"] = tok["input_ids"]

        bt["attention_mask"] = tok["attention_mask"]

    return bt



class ICDDataset(Dataset):

    """Summary of ICDDataset purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    X (Any): Description of X.
    cand (Any): Description of cand.
    lst (Any): Description of lst.
    y (Any): Description of y.
    """

    def __init__(self, df, feats, label_col, list_col, cand_col="candidate_indices"):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
        df (Any): Description of df.
        feats (Any): Description of feats.
        label_col (Any): Description of label_col.
        list_col (Any): Description of list_col.
        cand_col (Any): Description of cand_col.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        self.X = df[feats].to_numpy(dtype=np.float32, copy=True)

        self.y = df[label_col].astype(np.int64).to_numpy()

        self.lst = df[list_col].tolist()

        if cand_col in df.columns:

            self.cand = df[cand_col].tolist()

        else:

            self.cand = [[int(lbl)] for lbl in self.y]



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

        return len(self.y)



    def __getitem__(self, i: int) -> Dict[str, Any]:

        """Summary of __getitem__.
        
        Longer description of the __getitem__ behavior and usage.
        
        Args:
        i (Any): Description of i.
        
        Returns:
        Dict[str, Any]: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        num = torch.tensor(self.X[i], dtype=torch.float32)

        num = torch.nan_to_num(num, nan=0.0, posinf=0.0, neginf=0.0)

        lab = torch.tensor(self.y[i], dtype=torch.long)

        return {"num": num, "lab": lab, "lst": self.lst[i], "cand": self.cand[i]}



def collate_tensor_batch(batch: List[TensorBatch]) -> TensorBatch:

    """Summary of collate_tensor_batch.
    
    Longer description of the collate_tensor_batch behavior and usage.
    
    Args:
    batch (Any): Description of batch.
    
    Returns:
    TensorBatch: Description of the return value.
    
    Raises:
    Exception: Description of why this exception might be raised.
    """

    x_num = torch.stack([b.x_num for b in batch])

    y = torch.stack([b.y for b in batch])



    x_cat = torch.stack([b.x_cat for b in batch]) if batch[0].x_cat is not None else None

    ids = torch.stack([b.ids for b in batch]) if batch[0].ids is not None else None

    mask = torch.stack([b.mask for b in batch]) if batch[0].mask is not None else None

    delta = torch.stack([b.delta for b in batch]) if hasattr(batch[0], "delta") and batch[0].delta is not None else None

    x_icd = torch.stack([b.x_icd for b in batch]) if batch[0].x_icd is not None else None

    input_ids = torch.stack([b.input_ids for b in batch]) if getattr(batch[0], "input_ids", None) is not None else None

    attention_mask = torch.stack([b.attention_mask for b in batch]) if getattr(batch[0], "attention_mask", None) is not None else None

    candidates = torch.stack([b.candidates for b in batch]) if getattr(batch[0], "candidates", None) is not None else None



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

    """Summary of ParquetSource purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    _df (Any): Description of _df.
    file_path (Any): Description of file_path.
    """

    def __init__(self, file_path: Union[str, Path]):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
        file_path (Any): Description of file_path.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        self.file_path = Path(file_path)

        if not self.file_path.exists():

            if not self.file_path.is_absolute():

                self.file_path = Path.cwd() / self.file_path

            if not self.file_path.exists():

                raise DataError(f"Parquet file not found at {self.file_path}")

        self._df: Optional[pl.DataFrame] = None



    def load(self) -> pl.DataFrame:

        """Summary of load.
        
        Longer description of the load behavior and usage.
        
        Args:
        None (None): This function does not accept arguments.
        
        Returns:
        pl.DataFrame: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        try:

            if self._df is None:

                self._df = pl.read_parquet(self.file_path)

            return self._df

        except Exception as e:

            raise DataError(f"Failed to load parquet file: {e}")



    def __iter__(self) -> Generator[Dict[str, Any], None, None]:

        """Summary of __iter__.
        
        Longer description of the __iter__ behavior and usage.
        
        Args:
        None (None): This function does not accept arguments.
        
        Returns:
        Generator[Dict[str, Any], None, None]: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        try:

            if self._df is None:

                self._df = pl.read_parquet(self.file_path)

            for row in self._df.iter_rows(named=True):

                yield row

        except Exception as e:

            raise DataError(f"Failed to stream data: {e}")



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

        try:

            return pl.scan_parquet(self.file_path).select(pl.len()).collect().item()

        except Exception as e:

            raise DataError(f"Failed to get length of parquet file: {e}")



class HDF5StreamingSource(Dataset):

    """Summary of HDF5StreamingSource purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    _len (Any): Description of _len.
    h5_file (Any): Description of h5_file.
    h5_path (Any): Description of h5_path.
    label_key (Any): Description of label_key.
    """

    def __init__(self, h5_path: Path, seq_len: int = 0, label_key: str = "y"):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
        h5_path (Any): Description of h5_path.
        seq_len (Any): Description of seq_len.
        label_key (Any): Description of label_key.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

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

        """Summary of __len__.
        
        Longer description of the __len__ behavior and usage.
        
        Args:
        None (None): This function does not accept arguments.
        
        Returns:
        int: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        return self._len



    def __getitem__(self, idx: int) -> TensorBatch:

        """Summary of __getitem__.
        
        Longer description of the __getitem__ behavior and usage.
        
        Args:
        idx (Any): Description of idx.
        
        Returns:
        TensorBatch: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        if self.h5_file is None:

            self.h5_file = h5py.File(self.h5_path, "r")



        x_num_np = self.h5_file["X_num"][idx]

        x_num = torch.from_numpy(x_num_np).float()



        x_cat = None

        if "X_cat" in self.h5_file:

            x_cat = torch.from_numpy(self.h5_file["X_cat"][idx]).long()



        y_val = self.h5_file[self.label_key][idx]









        if y_val.ndim == 0:

            y = torch.tensor(y_val.item(), dtype=torch.long)

        else:

            y = torch.from_numpy(y_val).long()



        sid = -1

        if "sid" in self.h5_file:

            sid = int(self.h5_file["sid"][idx])

        sid_tensor = torch.tensor(sid, dtype=torch.long)



        x_icd = None

        if "X_icd" in self.h5_file:

            x_icd = torch.from_numpy(self.h5_file["X_icd"][idx]).float()



        input_ids = torch.from_numpy(self.h5_file["input_ids"][idx]).long() if "input_ids" in self.h5_file else None

        attention_mask = torch.from_numpy(self.h5_file["attention_mask"][idx]).long() if "attention_mask" in self.h5_file else None

        candidates = None

        if self.label_key == "y":

            candidates = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.long)



        tb = TensorBatch(

            x_num=x_num,

            x_cat=x_cat,

            y=y,

            ids=sid_tensor,

            mask=None,

            x_icd=x_icd,

            input_ids=input_ids,

            attention_mask=attention_mask,

            candidates=candidates

        )

        return tb



    def get_real_vocab_sizes(self) -> List[int]:

        """Summary of get_real_vocab_sizes.
        
        Longer description of the get_real_vocab_sizes behavior and usage.
        
        Args:
        None (None): This function does not accept arguments.
        
        Returns:
        List[int]: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        should_close = False

        if self.h5_file is None:

            self.h5_file = h5py.File(self.h5_path, "r")

            should_close = True

        try:

            if "X_cat" not in self.h5_file:

                return []

            data = self.h5_file["X_cat"][:]

            if data.size == 0:

                return []

            if data.ndim == 2:

                max_indices = np.max(data, axis=0)

            else:

                max_indices = np.max(data, axis=(0, 2))

            return [int(s) + 1 for s in max_indices]

        finally:

            if should_close:

                self.close()



    def close(self):

        """Summary of close.
        
        Longer description of the close behavior and usage.
        
        Args:
        None (None): This function does not accept arguments.
        
        Returns:
        Any: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        if self.h5_file is not None:

            self.h5_file.close()

            self.h5_file = None
