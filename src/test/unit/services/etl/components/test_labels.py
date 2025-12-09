import polars as pl
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from loguru import logger
from src.hint.services.etl.components.labels import LabelGenerator
from src.test.unit.conftest import UnitFixtures

def test_label_generator_execution() -> None:
    """
    [One-line Summary] Verify LabelGenerator produces answer parquet with ventilation labels.

    [Description]
    Provide a processed dataset containing required identifiers and ventilation indicators,
    run the LabelGenerator, and ensure the derived answer parquet file is created.

    Test Case ID: ETL-LABEL-01
    Scenario: Execute label generation on a processed dataset with ventilation flags present.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_label_generator_execution")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        (tmp_path / "processed").mkdir()
        config = UnitFixtures.get_minimal_etl_config().model_copy(update={"proc_dir": str(tmp_path / "processed")})
        
        logger.info("Writing processed dataset with identifiers and ventilation markers.")
        pl.DataFrame({
            "SUBJECT_ID": [1]*10, "ICUSTAY_ID": [100]*10, "HADM_ID": [10]*10,
            "HOUR_IN": list(range(10)),
            "VENT": [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            "V__hr": [80]*10
        }).write_parquet(tmp_path / "processed" / "dataset_123.parquet")

        gen = LabelGenerator(config, MagicMock(), MagicMock())
        gen.execute()
        assert (tmp_path / "processed" / "dataset_123_answer.parquet").exists()
