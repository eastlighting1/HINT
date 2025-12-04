import h5py
import numpy as np
import polars as pl
from pathlib import Path

def create_dummy_hdf5(path: Path, num_samples: int = 20, seq_len: int = 10, n_feats: int = 6):
    """
    Create a dummy HDF5 file for testing streaming sources.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("X_num", (num_samples, n_feats, seq_len), dtype='f4', data=np.random.randn(num_samples, n_feats, seq_len))
        f.create_dataset("X_cat", (num_samples, 2, seq_len), dtype='i4', data=np.random.randint(0, 5, (num_samples, 2, seq_len)))
        f.create_dataset("y", (num_samples,), dtype='i8', data=np.random.randint(0, 4, (num_samples,)))
        f.create_dataset("sid", (num_samples,), dtype='i8', data=np.arange(num_samples))

def create_dummy_parquet(path: Path, num_rows: int = 50):
    """
    Create a dummy Parquet file for testing ETL/ICD pipelines.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "SUBJECT_ID": np.arange(num_rows),
        "HADM_ID": np.arange(num_rows) + 1000,
        "ICD9_CODES": [["401.9", "250.00"] for _ in range(num_rows)],
        "target_label": np.random.randint(0, 2, num_rows),
    }
    # Add numeric features
    for i in range(5):
        data[f"V__feat_{i}"] = np.random.randn(num_rows)
    
    df = pl.DataFrame(data)
    df.write_parquet(path)