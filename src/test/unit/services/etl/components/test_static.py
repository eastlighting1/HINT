import polars as pl
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from loguru import logger
from hint.services.etl.components.static import StaticExtractor
from ...conftest import UnitFixtures

def test_static_extractor_filtering() -> None:
    """
    Validates that StaticExtractor filters patients based on age and LOS.
    
    Test Case ID: ETL-STA-01
    Description:
        Creates ADMISSIONS and PATIENTS and ICUSTAYS raw files.
        Executes StaticExtractor.
        Verifies that only valid patients are saved to patients.parquet.
    """
    logger.info("Starting test: test_static_extractor_filtering")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        (tmp_path / "raw").mkdir()
        (tmp_path / "processed").mkdir()

        config = UnitFixtures.get_minimal_etl_config().model_copy(update={
            "raw_dir": str(tmp_path / "raw"),
            "proc_dir": str(tmp_path / "processed"),
            "min_age": 18
        })

        # Mock Raw Files
        pl.DataFrame({"SUBJECT_ID": [1], "DOB": ["2050-01-01"], "GENDER": ["M"]}).write_csv(tmp_path / "raw" / "PATIENTS.csv")
        pl.DataFrame({"SUBJECT_ID": [1], "HADM_ID": [10], "ADMITTIME": ["2100-01-01"], "DISCHTIME": ["2100-01-05"], "ETHNICITY": ["W"], "ADMISSION_TYPE": ["E"], "INSURANCE": ["P"]}).write_csv(tmp_path / "raw" / "ADMISSIONS.csv")
        pl.DataFrame({"SUBJECT_ID": [1], "HADM_ID": [10], "ICUSTAY_ID": [100], "INTIME": ["2100-01-01 10:00:00"], "OUTTIME": ["2100-01-03 10:00:00"]}).write_csv(tmp_path / "raw" / "ICUSTAYS.csv")

        extractor = StaticExtractor(config, MagicMock(), MagicMock())
        extractor.execute()

        out_file = tmp_path / "processed" / "patients.parquet"
        assert out_file.exists()
        
        df = pl.read_parquet(out_file)
        assert df.height == 1
        assert df["AGE"][0] >= 18

    logger.info("StaticExtractor filtering verified.")