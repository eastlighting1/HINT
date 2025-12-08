import polars as pl
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from loguru import logger
from hint.services.etl.components.assembler import FeatureAssembler
from ...conftest import UnitFixtures

def test_feature_assembler_execution() -> None:
    """
    Validates the execution logic of FeatureAssembler.
    
    Test Case ID: ETL-ASM-01
    Description:
        Sets up required input files (patients, vitals, interventions).
        Executes the assembler.
        Verifies that the output dataset parquet file is created and has expected columns.
    """
    logger.info("Starting test: test_feature_assembler_execution")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        (tmp_path / "raw").mkdir()
        (tmp_path / "processed").mkdir()
        (tmp_path / "resources").mkdir()

        config = UnitFixtures.get_minimal_etl_config().model_copy(update={
            "raw_dir": str(tmp_path / "raw"),
            "proc_dir": str(tmp_path / "processed"),
            "resources_dir": str(tmp_path / "resources")
        })

        # Mock Inputs
        pl.DataFrame({
            "SUBJECT_ID": [1], "HADM_ID": [10], "ICUSTAY_ID": [100], 
            "INTIME": ["2100-01-01"], "AGE": [30], "STAY_HOURS": [48],
            "ETHNICITY": ["WHITE"], "ADMISSION_TYPE": ["EMERGENCY"], "INSURANCE": ["Public"]
        }).write_parquet(tmp_path / "processed" / "patients.parquet")

        pl.DataFrame({
            "SUBJECT_ID": [1], "HADM_ID": [10], "ICUSTAY_ID": [100], "HOUR_IN": [0],
            "LABEL": ["HR"], "MEAN": [80.0]
        }).write_parquet(tmp_path / "processed" / "vitals_labs_mean.parquet")

        pl.DataFrame({
            "SUBJECT_ID": [1], "HADM_ID": [10], "ICUSTAY_ID": [100], "HOUR_IN": [0],
            "VENT": [0], "OUTCOME_FLAG": [0]
        }).write_parquet(tmp_path / "processed" / "interventions.parquet")

        pl.DataFrame({
            "MIMIC LABEL": ["HR"], "LEVEL2": ["heart rate"]
        }).write_csv(tmp_path / "resources" / "itemid_to_variable_map.csv")
        
        # Write dummy raw ICD file
        pl.DataFrame({"SUBJECT_ID": [1], "HADM_ID": [10], "ICD9_CODE": ["428"]}).write_csv(tmp_path / "raw" / "DIAGNOSES_ICD.csv")

        assembler = FeatureAssembler(config, MagicMock(), MagicMock())
        assembler.execute()

        out_file = tmp_path / "processed" / "dataset_123.parquet"
        assert out_file.exists()
        
        df = pl.read_parquet(out_file)
        assert "V__heart_rate" in df.columns
        assert "S__AGE__AGE_15_39" in df.columns

    logger.info("FeatureAssembler execution verified.")