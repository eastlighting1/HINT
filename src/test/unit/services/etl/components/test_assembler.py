import polars as pl
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from loguru import logger
from src.hint.services.etl.components.assembler import FeatureAssembler
from src.test.unit.conftest import UnitFixtures

def test_feature_assembler_execution() -> None:
    """
    Validates the execution logic of FeatureAssembler.
    Test Case ID: ETL-ASM-01
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

        pl.DataFrame({
            "SUBJECT_ID": [1], "HADM_ID": [10], "ICUSTAY_ID": [100], 
            "INTIME": ["2100-01-01 00:00:00"], 
            "AGE": [30], "STAY_HOURS": [48],
            "ETHNICITY": ["WHITE"], "ADMISSION_TYPE": ["EMERGENCY"], "INSURANCE": ["Public"]
        }).with_columns(
            pl.col("INTIME").str.to_datetime(strict=False)
        ).write_parquet(tmp_path / "processed" / "patients.parquet")

        # [Fix] Renamed HOUR_IN -> HOURS_IN. 
        # The component code expects "HOURS_IN" in the input file and renames it to "HOUR_IN".
        pl.DataFrame({
            "SUBJECT_ID": [1], "HADM_ID": [10], "ICUSTAY_ID": [100], 
            "HOURS_IN": [0], 
            "LABEL": ["HR"], "MEAN": [80.0]
        }).write_parquet(tmp_path / "processed" / "vitals_labs_mean.parquet")

        # [Fix] Consistent naming (though intervention might use HOUR_IN, keeping consistent is safer)
        # Based on assemblers code: vitals uses HOURS_IN, intervs uses HOUR_IN in provided code.
        # But let's check assembler.py: `vent_df = intervs.select(["...HOUR_IN..."])`.
        # So interventions.parquet needs "HOUR_IN".
        pl.DataFrame({
            "SUBJECT_ID": [1], "HADM_ID": [10], "ICUSTAY_ID": [100], 
            "HOUR_IN": [0], 
            "VENT": [0], "OUTCOME_FLAG": [0]
        }).write_parquet(tmp_path / "processed" / "interventions.parquet")

        pl.DataFrame({
            "MIMIC LABEL": ["HR"], "LEVEL2": ["heart rate"]
        }).write_csv(tmp_path / "resources" / "itemid_to_variable_map.csv")
        
        pl.DataFrame({"SUBJECT_ID": [1], "HADM_ID": [10], "ICD9_CODE": ["428"]}).write_csv(tmp_path / "raw" / "DIAGNOSES_ICD.csv")

        assembler = FeatureAssembler(config, MagicMock(), MagicMock())
        assembler.execute()

        out_file = tmp_path / "processed" / "dataset_123.parquet"
        assert out_file.exists()
        
    logger.info("FeatureAssembler execution verified.")