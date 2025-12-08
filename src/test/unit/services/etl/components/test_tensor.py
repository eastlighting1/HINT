import polars as pl
import tempfile
import h5py
from pathlib import Path
from unittest.mock import MagicMock
from loguru import logger
from hint.services.etl.components.tensor import TensorConverter
from ...conftest import UnitFixtures

def test_tensor_converter_hdf5_generation() -> None:
    """
    Validates that TensorConverter converts parquet datasets into HDF5 tensors.
    
    Test Case ID: ETL-TNS-01
    Description:
        Creates labeled dataset parquet.
        Executes TensorConverter.
        Verifies HDF5 file structure (x_num, y, etc.).
    """
    logger.info("Starting test: test_tensor_converter_hdf5_generation")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        (tmp_path / "processed").mkdir()
        (tmp_path / "cache").mkdir()

        # Configs
        etl_config = UnitFixtures.get_minimal_etl_config().model_copy(update={"proc_dir": str(tmp_path / "processed")})
        cnn_config = UnitFixtures.get_minimal_cnn_config().model_copy(update={"data_cache_dir": str(tmp_path / "cache")})

        # Mock Labeled Data
        # Needs to have columns expected by converter
        pl.DataFrame({
            "SUBJECT_ID": [1, 1], "HADM_ID": [10, 10], "ICUSTAY_ID": [100, 100],
            "START_HOUR": [0, 6], "END_HOUR": [6, 12],
            "LABEL": [0, 1],
            "V__hr": [80.0, 85.0], "S__AGE__AGE_15_39": [1, 1]
        }).write_parquet(tmp_path / "processed" / "labeled_dataset.parquet")

        converter = TensorConverter(cnn_config, MagicMock(), MagicMock())
        # Inject ETL config path dependency if needed, usually passed in init or read from context
        # Assuming converter reads from cnn_config.data_path or fixed path
        # Here we mock the input path resolution in the component or ensure it looks in proc_dir
        # Since TensorConverter signature in snippet matches Component(config, registry, observer)
        # We need to ensure it knows where to look. Assuming it uses cnn_config.data_cache_dir or similar.
        
        # Override input path logic for test if hardcoded in component
        converter.input_path = tmp_path / "processed" / "labeled_dataset.parquet"
        
        converter.execute()

        h5_file = tmp_path / "cache" / "train.h5" # Assuming default split includes train
        # Check if any h5 created
        h5_files = list((tmp_path / "cache").glob("*.h5"))
        assert len(h5_files) > 0
        
        with h5py.File(h5_files[0], "r") as f:
            assert "x_num" in f
            assert "y" in f

    logger.info("TensorConverter HDF5 generation verified.")