import tempfile
import polars as pl
from pathlib import Path
from unittest.mock import MagicMock
from loguru import logger
from src.hint.services.etl.service import ETLService
from src.hint.services.etl.components.assembler import FeatureAssembler
from src.test.integration.conftest import IntegrationFixtures
from src.test.unit.conftest import UnitFixtures

def test_etl_pipeline_execution() -> None:
    """
    Validates the execution of a minimal ETL pipeline workflow.
    
    Test Case ID: TS-10
    """
    logger.info("Starting test: test_etl_pipeline_execution")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        (tmp_path / "raw").mkdir()
        (tmp_path / "processed").mkdir()
        (tmp_path / "resources").mkdir()
        
        IntegrationFixtures.setup_etl_raw_files(tmp_path / "raw")
        
        # [Fix] Use strict=False to handle potential format mismatches safely
        pl.DataFrame({
            "SUBJECT_ID": [101, 102], "HADM_ID": [1001, 1002], 
            "ICUSTAY_ID": [5001, 5002], "INTIME": ["2150-01-01 00:00:00", "2150-01-01 00:00:00"],
            "AGE": [60, 70], "STAY_HOURS": [50, 50],
            "ETHNICITY": ["WHITE", "WHITE"], "ADMISSION_TYPE": ["EMERGENCY", "EMERGENCY"],
            "INSURANCE": ["Medicare", "Medicare"]
        }).with_columns(
            pl.col("INTIME").str.to_datetime(strict=False)
        ).write_parquet(tmp_path / "processed" / "patients.parquet")
        
        # [Fix] Use HOUR_IN (singular) to match component logic
        pl.DataFrame({
            "SUBJECT_ID": [101], "HADM_ID": [1001], "ICUSTAY_ID": [5001],
            "HOUR_IN": [1], "LABEL": ["Heart Rate"], "MEAN": [80.0]
        }).write_parquet(tmp_path / "processed" / "vitals_labs_mean.parquet")
        
        pl.DataFrame({
            "SUBJECT_ID": [101], "HADM_ID": [1001], "ICUSTAY_ID": [5001],
            "HOUR_IN": [1], "VENT": [0], "OUTCOME_FLAG": [0]
        }).write_parquet(tmp_path / "processed" / "interventions.parquet")
        
        pl.DataFrame({
            "MIMIC LABEL": ["Heart Rate"], "LEVEL2": ["heart rate"]
        }).write_csv(tmp_path / "resources" / "itemid_to_variable_map.csv")
        
        config = IntegrationFixtures.get_integrated_etl_config(tmp_path)
        cnn_config = UnitFixtures.get_minimal_cnn_config()
        
        mock_registry = MagicMock()
        mock_observer = MagicMock()
        mock_observer.create_progress.return_value.__enter__.return_value = MagicMock()

        components = [
            FeatureAssembler(config, mock_registry, mock_observer)
        ]
        
        service = ETLService(config, cnn_config, components, mock_observer)
        
        service.run_pipeline()
        
        expected_output = tmp_path / "processed" / "dataset_123.parquet"
        assert expected_output.exists()
        
        logger.info("ETL pipeline execution verified.")