import tempfile
import polars as pl
from pathlib import Path
from unittest.mock import MagicMock
from loguru import logger
from hint.services.etl.service import ETLService
from hint.services.etl.components.static import StaticExtractor
from hint.services.etl.components.assembler import FeatureAssembler
from ..conftest import IntegrationFixtures
from ...unit.conftest import UnitFixtures

def test_etl_pipeline_execution() -> None:
    """
    Verify execution of a minimal ETL workflow using static inputs.

    This test validates that `ETLService` configured with synthetic raw data and a feature assembler writes the expected processed Parquet output into a temporary directory.
    - Test Case ID: TS-10
    - Scenario: Run ETL with static extractor and assembler on deterministic inputs.

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
        
        logger.debug("Setting up raw data")
        IntegrationFixtures.setup_etl_raw_files(tmp_path / "raw")
        
        pl.DataFrame({
            "SUBJECT_ID": [101, 102], "HADM_ID": [1001, 1002], 
            "ICUSTAY_ID": [5001, 5002], "INTIME": ["2150-01-01", "2150-01-01"],
            "AGE": [60, 70], "STAY_HOURS": [50, 50],
            "ETHNICITY": ["WHITE", "WHITE"], "ADMISSION_TYPE": ["EMERGENCY", "EMERGENCY"],
            "INSURANCE": ["Medicare", "Medicare"]
        }).write_parquet(tmp_path / "processed" / "patients.parquet")
        
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
        
        logger.info("Running ETL pipeline")
        service.run_pipeline()
        
        logger.debug("Verifying output artifacts")
        expected_output = tmp_path / "processed" / "dataset_123.parquet"
        assert expected_output.exists(), f"Output file {expected_output} was not created"
        
        logger.info("ETL pipeline execution verified.")
