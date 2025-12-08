import torch
import tempfile
import h5py
import numpy as np
from pathlib import Path
from loguru import logger
from hint.infrastructure.datasource import HDF5StreamingSource, ParquetSource

def test_hdf5_streaming_source_getitem() -> None:
    """
    Verify HDF5StreamingSource returns tensors with expected shapes.

    This test validates that an HDF5 file containing numeric, categorical, labels, and IDs can be read through `HDF5StreamingSource`, returning tensors with the correct layout.
    - Test Case ID: INF-DS-01
    - Scenario: Read a single sample from a temporary HDF5 dataset.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_hdf5_streaming_source_getitem")

    with tempfile.TemporaryDirectory() as tmp_dir:
        h5_path = Path(tmp_dir) / "test.h5"
        
        logger.debug("Creating dummy HDF5 file")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("x_num", data=np.random.randn(5, 10, 4))
            f.create_dataset("x_cat", data=np.random.randint(0, 5, (5, 10, 2)))
            f.create_dataset("y", data=np.array([0, 1, 0, 1, 0]))
            f.create_dataset("ids", data=np.arange(5))

        source = HDF5StreamingSource(h5_path, max_seq_len=10)
        
        logger.debug("Retrieving sample at index 0")
        sample = source[0]

        assert isinstance(sample.x_num, torch.Tensor)
        assert sample.x_num.shape == (4, 10) 
        assert sample.y.item() == 0

    logger.info("HDF5 streaming source retrieval verified.")

def test_parquet_source_loading() -> None:
    """
    Confirm ParquetSource reports the correct dataset length.

    This test validates that `ParquetSource` can read a simple Parquet file and exposes the expected length value to consumers.
    - Test Case ID: INF-DS-02
    - Scenario: Load a synthetic Parquet file and compare reported rows.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_parquet_source_loading")
    import polars as pl

    with tempfile.TemporaryDirectory() as tmp_dir:
        pq_path = Path(tmp_dir) / "data.parquet"
        
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df.write_parquet(pq_path)

        source = ParquetSource(pq_path)
        
        logger.debug(f"Verifying source length: {len(source)}")
        assert len(source) == 3

    logger.info("Parquet source loading verified.")
