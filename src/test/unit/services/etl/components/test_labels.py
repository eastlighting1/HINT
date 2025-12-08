import polars as pl
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from loguru import logger
from hint.services.etl.components.labels import LabelGenerator
from ...conftest import UnitFixtures

def test_label_generator_execution() -> None:
    """
    Validates that LabelGenerator creates window-based labels.
    
    Test Case ID: ETL-LBL-01
    Description:
        Creates a dummy dataset with hourly VENT flags.
        Executes LabelGenerator.
        Verifies that windowed labels (ONSET, etc.) are generated.
    """
    logger.info("Starting test: test_label_generator_execution")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        (tmp_path / "processed").mkdir()
        
        config = UnitFixtures.get_minimal_etl_config().model_copy(update={
            "proc_dir": str(tmp_path / "processed")
        })

        # Mock Dataset
        pl.DataFrame({
            "SUBJECT_ID": [1]*10, "ICUSTAY_ID": [100]*10,
            "HOUR_IN": list(range(10)),
            "VENT": [0, 0, 0, 1, 1, 1, 0, 0, 0, 0], # Onset at 3
            "V__hr": [80]*10
        }).write_parquet(tmp_path / "processed" / "dataset_123.parquet")

        gen = LabelGenerator(config, MagicMock(), MagicMock())
        gen.execute()

        out_file = tmp_path / "processed" / "labeled_dataset.parquet"
        assert out_file.exists()
        
        df = pl.read_parquet(out_file)
        assert "LABEL_ONSET" in df.columns
        
    logger.info("LabelGenerator execution verified.")