import torch
import tempfile
import h5py
import numpy as np
from pathlib import Path
from loguru import logger
from hint.infrastructure.datasource import HDF5StreamingSource, ParquetSource

def test_hdf5_streaming_source_getitem() -> None:
    """
    Validates data retrieval from HDF5StreamingSource.
    
    Test Case ID: INF-DS-01
    Description:
        Creates a temporary HDF5 file.
        Reads a sample using the source class.
        Verifies the shape and type of the returned TensorBatch components.
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
    Validates initialization and length checking of ParquetSource.
    
    Test Case ID: INF-DS-02
    Description:
        Creates a temporary Parquet file using Polars.
        Initializes ParquetSource.
        Verifies the reported length matches the row count.
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