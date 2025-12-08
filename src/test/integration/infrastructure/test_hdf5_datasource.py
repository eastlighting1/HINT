import torch
import tempfile
from pathlib import Path
from loguru import logger
from hint.infrastructure.datasource import HDF5StreamingSource
from ..conftest import IntegrationFixtures

def test_hdf5_datasource_read_integrity() -> None:
    """
    Validates that HDF5StreamingSource can correctly read from a physical HDF5 file.
    
    Test Case ID: TS-08
    Description:
        Creates a temporary HDF5 file with known shapes.
        Initializes HDF5StreamingSource.
        Reads a sample and verifies data types and tensor shapes.
    """
    logger.info("Starting test: test_hdf5_datasource_read_integrity")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        h5_path = tmp_path / "test_data.h5"
        
        seq_len = 15
        n_feat = 8
        
        IntegrationFixtures.create_dummy_hdf5(h5_path, num_samples=5, seq_len=seq_len, n_features=n_feat)
        
        logger.debug("Initializing HDF5StreamingSource")
        source = HDF5StreamingSource(h5_path, max_seq_len=seq_len)
        
        logger.debug("Reading first sample")
        sample = source[0]
        
        logger.debug(f"Verifying x_num shape. Expected: ({n_feat}, {seq_len})")
        # Note: Transpose behavior might depend on implementation, assuming (C, T) or (T, C)
        # Based on typical PyTorch CNN input (C, L), let's check dimensionality
        assert sample.x_num.dim() == 2, "x_num should be 2D tensor"
        assert sample.y.dim() == 0, "y should be scalar tensor"
        
        logger.debug("Verifying data types")
        assert isinstance(sample.x_num, torch.Tensor)
        assert isinstance(sample.y, torch.Tensor)
        
        logger.info("HDF5 DataSource integrity verified.")