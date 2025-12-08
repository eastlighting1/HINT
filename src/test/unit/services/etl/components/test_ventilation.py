import polars as pl
import tempfile
from pathlib import Path
from loguru import logger
from hint.services.etl.components.ventilation import VentilationTagger
from ...conftest import UnitFixtures
from ....conftest import TestFixtures

def test_ventilation_tagger_marks_vent_correctly() -> None:
    """
    Verify VentilationTagger derives VENT flags from item IDs.

    This test validates that VentilationTagger reads mapping and vitals files to label intervention rows with the correct ventilation indicator for matching timestamps.
    - Test Case ID: ETL-01
    - Scenario: Generate ventilation labels for synthetic intervention records using mocked resources.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_ventilation_tagger_marks_vent_correctly")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        res_dir = tmp_path / "resources"
        proc_dir = tmp_path / "processed"
        res_dir.mkdir()
        proc_dir.mkdir()
        
        logger.debug(f"Created temporary directories at {tmp_path}")

        base_config = UnitFixtures.get_minimal_etl_config()
        config = base_config.model_copy(update={
            "resources_dir": str(res_dir), 
            "proc_dir": str(proc_dir)
        })

        logger.debug("Creating dummy resource files")
        pl.DataFrame({
            "ITEMID": [445, 999],
            "MIMIC LABEL": ["Ventilator", "Other"],
            "LINKSTO": ["chartevents", "chartevents"]
        }).write_csv(res_dir / "itemid_to_variable_map.csv")

        logger.debug("Creating dummy vitals_labs.parquet")
        pl.DataFrame({
            "SUBJECT_ID": [101], "HADM_ID": [1001], "ICUSTAY_ID": [5001],
            "HOURS_IN": [10],
            "LABEL": ["Ventilator"],
            "VALUENUM": [1]
        }).write_parquet(proc_dir / "vitals_labs.parquet")

        logger.debug("Creating dummy interventions.parquet (Target)")
        pl.DataFrame({
            "SUBJECT_ID": [101, 101], 
            "HADM_ID": [1001, 1001], 
            "ICUSTAY_ID": [5001, 5001],
            "HOUR_IN": [10, 11],
            "OUTCOME_FLAG": [0, 0]
        }).write_parquet(proc_dir / "interventions.parquet")

        mock_registry = TestFixtures.get_mock_registry()
        mock_observer = TestFixtures.get_mock_observer()
        
        tagger = VentilationTagger(config, mock_registry, mock_observer)
        
        logger.info("Executing VentilationTagger component")
        tagger.execute()

        logger.debug("Verifying results in interventions.parquet")
        result = pl.read_parquet(proc_dir / "interventions.parquet")
        
        row_10 = result.filter(pl.col("HOUR_IN") == 10)
        vent_10 = row_10["VENT"][0]
        assert vent_10 == 1, f"Expected VENT=1 at hour 10, got {vent_10}"
        
        row_11 = result.filter(pl.col("HOUR_IN") == 11)
        vent_11 = row_11["VENT"][0]
        assert vent_11 == 0, f"Expected VENT=0 at hour 11, got {vent_11}"
        
        logger.info("Ventilation tagging logic verified successfully.")
