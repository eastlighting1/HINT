import polars as pl
import tempfile
import h5py
from pathlib import Path
from unittest.mock import MagicMock
from loguru import logger
from src.hint.services.etl.components.tensor import TensorConverter
from src.test.unit.conftest import UnitFixtures

def test_tensor_converter_hdf5_generation() -> None:
    logger.info("Starting test: test_tensor_converter_hdf5_generation")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        (tmp_path / "processed").mkdir()
        (tmp_path / "cache").mkdir()
        
        etl_config = UnitFixtures.get_minimal_etl_config().model_copy(update={"proc_dir": str(tmp_path / "processed")})
        cnn_config = UnitFixtures.get_minimal_cnn_config().model_copy(update={"data_cache_dir": str(tmp_path / "cache")})

        pl.DataFrame({
            "SUBJECT_ID": [1, 1], "HADM_ID": [10, 10], "ICUSTAY_ID": [100, 100],
            "START_HOUR": [0, 6], "END_HOUR": [6, 12], "LABEL": [0, 1],
            "V__hr": [80.0, 85.0], "S__AGE__AGE_15_39": [1, 1]
        }).write_parquet(tmp_path / "processed" / "labeled_dataset.parquet")

        converter = TensorConverter(cnn_config, MagicMock(), MagicMock())
        converter.input_path = tmp_path / "processed" / "labeled_dataset.parquet"
        converter.execute()

        assert list((tmp_path / "cache").glob("*.h5"))