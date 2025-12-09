import polars as pl
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from loguru import logger
from src.hint.services.etl.components.static import StaticExtractor
from src.test.unit.conftest import UnitFixtures

def test_static_extractor_filtering() -> None:
    """
    [One-line Summary] Validate StaticExtractor filters and writes patient demographics.

    [Description]
    Prepare minimal raw patient, admissions, and ICU stay data with required columns, run the
    StaticExtractor, and verify the processed patients parquet file is produced.

    Test Case ID: ETL-STATIC-01
    Scenario: Execute static extraction with complete demographic inputs and confirm output artifact.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_static_extractor_filtering")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        (tmp_path / "raw").mkdir()
        (tmp_path / "processed").mkdir()
        config = UnitFixtures.get_minimal_etl_config().model_copy(update={
            "raw_dir": str(tmp_path / "raw"), "proc_dir": str(tmp_path / "processed"), "min_age": 18
        })
        
        logger.info("Writing PATIENTS dataset with DOB and DOD columns.")
        pl.DataFrame({
            "SUBJECT_ID": [1], 
            "DOB": ["2050-01-01 00:00:00"], 
            "DOD": [None],
            "GENDER": ["M"]
        }).write_csv(tmp_path / "raw" / "PATIENTS.csv")

        logger.info("Writing ADMISSIONS dataset including DEATHTIME metadata.")
        pl.DataFrame({
            "SUBJECT_ID": [1], "HADM_ID": [10], 
            "ADMITTIME": ["2100-01-01 00:00:00"], 
            "DISCHTIME": ["2100-01-05 00:00:00"], 
            "DEATHTIME": [None],
            "ETHNICITY": ["W"], "ADMISSION_TYPE": ["E"], "INSURANCE": ["P"]
        }).write_csv(tmp_path / "raw" / "ADMISSIONS.csv")
        
        logger.info("Writing ICUSTAYS dataset with LOS and FIRST_CAREUNIT fields.")
        pl.DataFrame({
            "SUBJECT_ID": [1], "HADM_ID": [10], "ICUSTAY_ID": [100], 
            "INTIME": ["2100-01-01 10:00:00"], "OUTTIME": ["2100-01-03 10:00:00"],
            "LOS": [2.0], "FIRST_CAREUNIT": ["MICU"]
        }).write_csv(tmp_path / "raw" / "ICUSTAYS.csv")

        extractor = StaticExtractor(config, MagicMock(), MagicMock())
        extractor.execute()
        assert (tmp_path / "processed" / "patients.parquet").exists()
