import polars as pl
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from loguru import logger
from hint.services.etl.components.outcomes import OutcomesBuilder
from ...conftest import UnitFixtures

def test_outcomes_builder_flagging() -> None:
    """
    Verify OutcomesBuilder flags records near death or discharge events.

    This test validates that executing `OutcomesBuilder` on patient timelines containing a death timestamp enriches the interventions file with `OUTCOME_FLAG` indicators.
    - Test Case ID: ETL-OUT-01
    - Scenario: Process intervention rows around a recorded death to confirm flag propagation.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_outcomes_builder_flagging")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        (tmp_path / "processed").mkdir()

        config = UnitFixtures.get_minimal_etl_config().model_copy(update={
            "proc_dir": str(tmp_path / "processed")
        })

        pl.DataFrame({
            "SUBJECT_ID": [1], "HADM_ID": [10], "ICUSTAY_ID": [100],
            "INTIME": ["2100-01-01 00:00:00"], "DOD": ["2100-01-03 02:00:00"],
            "DISCHTIME": [None]
        }).write_parquet(tmp_path / "processed" / "patients.parquet")

        pl.DataFrame({
            "SUBJECT_ID": [1], "HADM_ID": [10], "ICUSTAY_ID": [100],
            "HOUR_IN": [48, 49, 50, 51]
        }).write_parquet(tmp_path / "processed" / "interventions.parquet")

        builder = OutcomesBuilder(config, MagicMock(), MagicMock())
        builder.execute()

        df = pl.read_parquet(tmp_path / "processed" / "interventions.parquet")
        
        assert "OUTCOME_FLAG" in df.columns
        assert df.filter(pl.col("HOUR_IN") >= 50)["OUTCOME_FLAG"].sum() > 0

    logger.info("OutcomesBuilder flagging verified.")
