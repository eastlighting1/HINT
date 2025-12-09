import polars as pl
import tempfile
import gzip
from pathlib import Path
from unittest.mock import MagicMock
from loguru import logger
from src.hint.services.etl.components.outcomes import OutcomesBuilder
from src.test.unit.conftest import UnitFixtures

def test_outcomes_builder_flagging() -> None:
    """
    [One-line Summary] Verify OutcomesBuilder flags interventions based on output events.

    [Description]
    Create ICU stay metadata and output events within the stay window, execute the outcomes
    builder, and assert the resulting interventions parquet includes outcome flags.

    Test Case ID: ETL-OUT-01
    Scenario: Execute outcomes building with matching ICU stay and output event timestamps.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_outcomes_builder_flagging")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        (tmp_path / "processed").mkdir()
        (tmp_path / "raw").mkdir()
        
        config = UnitFixtures.get_minimal_etl_config().model_copy(update={
            "proc_dir": str(tmp_path / "processed"),
            "raw_dir": str(tmp_path / "raw")
        })
        
        logger.info("Creating ICU stay metadata for outcomes builder.")
        icu_df = pl.DataFrame({
            "SUBJECT_ID": [1], "HADM_ID": [10], "ICUSTAY_ID": [100],
            "INTIME": ["2100-01-01 00:00:00"], "OUTTIME": ["2100-01-03 00:00:00"]
        })
        with gzip.open(tmp_path / "raw" / "ICUSTAYS.csv.gz", "wb") as f:
            icu_df.write_csv(f)
            
        logger.info("Creating output events within ICU stay window.")
        out_df = pl.DataFrame({
            "SUBJECT_ID": [1], "HADM_ID": [10], "ICUSTAY_ID": [100],
            "CHARTTIME": ["2100-01-01 12:00:00"],
            "ITEMID": [999]
        })
        with gzip.open(tmp_path / "raw" / "OUTPUTEVENTS.csv.gz", "wb") as f:
            out_df.write_csv(f)

        builder = OutcomesBuilder(config, MagicMock(), MagicMock())
        builder.execute()
        
        df = pl.read_parquet(tmp_path / "processed" / "interventions.parquet")
        assert "OUTCOME_FLAG" in df.columns
        assert df.height > 0
