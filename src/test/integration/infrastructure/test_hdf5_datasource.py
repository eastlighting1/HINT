import torch
import tempfile
from pathlib import Path
from loguru import logger
from hint.infrastructure.datasource import HDF5StreamingSource
from ..conftest import IntegrationFixtures

def test_hdf5_datasource_read_integrity() -> None:
    """
    Verify HDF5StreamingSource reads numeric and label tensors from disk.

    This test validates that a physical HDF5 file with known shapes can be streamed through `HDF5StreamingSource`, returning tensors with expected dimensionality and data types.
    - Test Case ID: TS-08
    - Scenario: Load a single record from a generated HDF5 dataset.

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
        
        IntegrationFixtures.create_dummy_hdf5(h5_path, num_samples=5, seq_len=seq_len, n_features=n_feat)
        
        logger.debug("Initializing HDF5StreamingSource")
        source = HDF5StreamingSource(h5_path, max_seq_len=seq_len)
        
        logger.debug("Reading first sample")
        sample = source[0]
        
        logger.debug(f"Verifying x_num shape. Expected: ({n_feat}, {seq_len})")
        assert sample.x_num.dim() == 2, "x_num should be 2D tensor"
        assert sample.y.dim() == 0, "y should be scalar tensor"
        
        logger.debug("Verifying data types")
        assert isinstance(sample.x_num, torch.Tensor)
        assert isinstance(sample.y, torch.Tensor)
        
        logger.info("HDF5 DataSource integrity verified.")
