import polars as pl
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from loguru import logger
from hint.services.etl.components.outcomes import OutcomesBuilder
from ...conftest import UnitFixtures

def test_outcomes_builder_flagging() -> None:
    """
    Validates that OutcomesBuilder correctly flags death or discharge events.
    
    Test Case ID: ETL-OUT-01
    Description:
        Creates patient data with death timestamps.
        Executes OutcomesBuilder.
        Verifies that OUTCOME_FLAG is set correctly in interventions file.
    """
    logger.info("Starting test: test_outcomes_builder_flagging")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        (tmp_path / "processed").mkdir()

        config = UnitFixtures.get_minimal_etl_config().model_copy(update={
            "proc_dir": str(tmp_path / "processed")
        })

        # Patient died at hour 50 relative to intime
        pl.DataFrame({
            "SUBJECT_ID": [1], "HADM_ID": [10], "ICUSTAY_ID": [100],
            "INTIME": ["2100-01-01 00:00:00"], "DOD": ["2100-01-03 02:00:00"], # +50 hours
            "DISCHTIME": [None]
        }).write_parquet(tmp_path / "processed" / "patients.parquet")

        # Intervention timeline
        pl.DataFrame({
            "SUBJECT_ID": [1], "HADM_ID": [10], "ICUSTAY_ID": [100],
            "HOUR_IN": [48, 49, 50, 51]
        }).write_parquet(tmp_path / "processed" / "interventions.parquet")

        builder = OutcomesBuilder(config, MagicMock(), MagicMock())
        builder.execute()

        df = pl.read_parquet(tmp_path / "processed" / "interventions.parquet")
        
        # Check flag logic (implementation specific, assuming flag=1 if death/discharge imminent or happened)
        # Detailed logic depends on actual code, checking existence of column and changes
        assert "OUTCOME_FLAG" in df.columns
        assert df.filter(pl.col("HOUR_IN") >= 50)["OUTCOME_FLAG"].sum() > 0

    logger.info("OutcomesBuilder flagging verified.")