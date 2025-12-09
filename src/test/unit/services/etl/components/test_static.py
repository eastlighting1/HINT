import polars as pl
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from loguru import logger
from src.hint.services.etl.components.static import StaticExtractor
from src.test.unit.conftest import UnitFixtures

def test_static_extractor_filtering() -> None:
    logger.info("Starting test: test_static_extractor_filtering")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        (tmp_path / "raw").mkdir()
        (tmp_path / "processed").mkdir()
        config = UnitFixtures.get_minimal_etl_config().model_copy(update={
            "raw_dir": str(tmp_path / "raw"), "proc_dir": str(tmp_path / "processed"), "min_age": 18
        })
        
        # [Fix] Added DOD column to PATIENTS
        pl.DataFrame({
            "SUBJECT_ID": [1], 
            "DOB": ["2050-01-01 00:00:00"], 
            "DOD": [None],
            "GENDER": ["M"]
        }).write_csv(tmp_path / "raw" / "PATIENTS.csv")

        # [Fix] Added DEATHTIME column to ADMISSIONS
        pl.DataFrame({
            "SUBJECT_ID": [1], "HADM_ID": [10], 
            "ADMITTIME": ["2100-01-01 00:00:00"], 
            "DISCHTIME": ["2100-01-05 00:00:00"], 
            "DEATHTIME": [None],
            "ETHNICITY": ["W"], "ADMISSION_TYPE": ["E"], "INSURANCE": ["P"]
        }).write_csv(tmp_path / "raw" / "ADMISSIONS.csv")
        
        # [Fix] Added FIRST_CAREUNIT and LOS columns required by extractor
        pl.DataFrame({
            "SUBJECT_ID": [1], "HADM_ID": [10], "ICUSTAY_ID": [100], 
            "INTIME": ["2100-01-01 10:00:00"], "OUTTIME": ["2100-01-03 10:00:00"],
            "LOS": [2.0], "FIRST_CAREUNIT": ["MICU"]
        }).write_csv(tmp_path / "raw" / "ICUSTAYS.csv")

        extractor = StaticExtractor(config, MagicMock(), MagicMock())
        extractor.execute()
        assert (tmp_path / "processed" / "patients.parquet").exists()