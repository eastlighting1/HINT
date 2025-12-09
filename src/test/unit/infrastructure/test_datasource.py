import torch
import tempfile
import h5py
import numpy as np
from pathlib import Path
from loguru import logger
from src.hint.infrastructure.datasource import HDF5StreamingSource, ParquetSource

def test_hdf5_streaming_source_getitem() -> None:
    """
    Validates data retrieval from HDF5StreamingSource.
    Test Case ID: INF-DS-01
    """
    logger.info("Starting test: test_hdf5_streaming_source_getitem")

    with tempfile.TemporaryDirectory() as tmp_dir:
        h5_path = Path(tmp_dir) / "test.h5"
        
        # [Fix] Use correct key names: 'X_num' (Capital), 'sid' (not ids)
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("X_num", data=np.random.randn(5, 10, 4))
            f.create_dataset("X_cat", data=np.random.randint(0, 5, (5, 10, 2)))
            f.create_dataset("y", data=np.array([0, 1, 0, 1, 0]))
            f.create_dataset("sid", data=np.arange(5))
            f.create_dataset("mask", data=np.ones((5, 10))) 

        # [Fix] Added seq_len argument
        source = HDF5StreamingSource(h5_path, seq_len=10)
        
        sample = source[0]
        assert isinstance(sample.x_num, torch.Tensor)
        assert sample.x_num.shape == (4, 10) 
        assert sample.y.item() == 0

    logger.info("HDF5 streaming source retrieval verified.")

def test_parquet_source_loading() -> None:
    """
    Validates initialization and length checking of ParquetSource.
    Test Case ID: INF-DS-02
    """
    logger.info("Starting test: test_parquet_source_loading")
    import polars as pl

    with tempfile.TemporaryDirectory() as tmp_dir:
        pq_path = Path(tmp_dir) / "data.parquet"
        
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df.write_parquet(pq_path)

        source = ParquetSource(pq_path)
        assert len(source) == 3

    logger.info("Parquet source loading verified.")