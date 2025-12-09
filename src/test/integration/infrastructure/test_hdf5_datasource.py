import torch
import tempfile
import h5py
import numpy as np
from pathlib import Path
from loguru import logger
from src.hint.infrastructure.datasource import HDF5StreamingSource

def test_hdf5_datasource_read_integrity() -> None:
    """
    [One-line Summary] Validate HDF5StreamingSource reads datasets with expected keys.

    [Description]
    Persist numeric, categorical, label, and mask datasets into an on-disk HDF5 file and
    ensure HDF5StreamingSource returns tensor samples with the expected dimensions.

    Test Case ID: TS-08
    Scenario: Load a sample from a temporary HDF5 file and confirm tensor integrity.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_hdf5_datasource_read_integrity")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        h5_path = tmp_path / "test_data.h5"
        
        seq_len = 15
        n_feat = 8
        num_samples = 5
        
        logger.info("Writing HDF5 datasets with expected numeric, categorical, and mask keys.")
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset("X_num", data=np.random.randn(num_samples, seq_len, n_feat).astype(np.float32))
            f.create_dataset("X_cat", data=np.random.randint(0, 5, (num_samples, seq_len, 2)).astype(np.int64))
            f.create_dataset("y", data=np.random.randint(0, 4, (num_samples,)).astype(np.int64))
            f.create_dataset("sid", data=np.arange(num_samples).astype(np.int64))
            f.create_dataset("mask", data=np.ones((num_samples, seq_len)).astype(np.float32))
        
        logger.info("Instantiating HDF5StreamingSource with explicit seq_len.")
        source = HDF5StreamingSource(h5_path, seq_len)
        
        sample = source[0]
        
        assert sample.x_num.dim() == 2
        assert isinstance(sample.x_num, torch.Tensor)
        
        logger.info("HDF5 DataSource integrity verified.")
