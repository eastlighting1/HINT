import polars as pl
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from loguru import logger
from src.hint.services.etl.components.assembler import FeatureAssembler
from src.test.unit.conftest import UnitFixtures


def test_etl_pipeline_execution() -> None:
    """
    [One-line Summary] Validate ETL pipeline creates a processed dataset from minimal inputs.

    [Description]
    Build a temporary ETL workspace with patient, vitals, intervention, and resource files,
    execute the FeatureAssembler, and confirm the processed dataset parquet is written.

    Test Case ID: ETL-PIPE-01
    Scenario: Execute end-to-end ETL assembly with single-row inputs and verify output materialization.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_etl_pipeline_execution")

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

        logger.info("Preparing patient demographics for ETL execution.")
        pl.DataFrame({
            "SUBJECT_ID": [1], "HADM_ID": [10], "ICUSTAY_ID": [100], 
            "INTIME": ["2100-01-01 00:00:00"], 
            "AGE": [30], "STAY_HOURS": [48],
            "ETHNICITY": ["WHITE"], "ADMISSION_TYPE": ["EMERGENCY"], "INSURANCE": ["Public"]
        }).with_columns(
            pl.col("INTIME").str.to_datetime(strict=False)
        ).write_parquet(tmp_path / "processed" / "patients.parquet")

        logger.info("Writing vitals_labs_mean parquet with HOURS_IN column for assembler.")
        pl.DataFrame({
            "SUBJECT_ID": [1], "HADM_ID": [10], "ICUSTAY_ID": [100], 
            "HOURS_IN": [0],
            "LABEL": ["HR"], "MEAN": [80.0]
        }).write_parquet(tmp_path / "processed" / "vitals_labs_mean.parquet")

        logger.info("Writing interventions parquet with HOUR_IN column for ventilation join.")
        pl.DataFrame({
            "SUBJECT_ID": [1], "HADM_ID": [10], "ICUSTAY_ID": [100], 
            "HOUR_IN": [0], 
            "VENT": [0], "OUTCOME_FLAG": [0]
        }).write_parquet(tmp_path / "processed" / "interventions.parquet")

        logger.info("Adding resources and raw diagnosis artifacts required by assembler.")
        pl.DataFrame({
            "MIMIC LABEL": ["HR"], "LEVEL2": ["heart rate"]
        }).write_csv(tmp_path / "resources" / "itemid_to_variable_map.csv")
        
        pl.DataFrame({
            "SUBJECT_ID": [1], "HADM_ID": [10], "ICD9_CODE": ["428"]
        }).write_csv(tmp_path / "raw" / "DIAGNOSES_ICD.csv")

        logger.info("Executing feature assembler.")
        assembler = FeatureAssembler(config, MagicMock(), MagicMock())
        assembler.execute()

        out_file = tmp_path / "processed" / "dataset_123.parquet"
        assert out_file.exists()
        
    logger.info("ETL pipeline execution verified.")
