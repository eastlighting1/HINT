import polars as pl
import tempfile
import gzip
from pathlib import Path
from unittest.mock import MagicMock
from loguru import logger
from src.hint.services.etl.components.outcomes import OutcomesBuilder
from src.test.unit.conftest import UnitFixtures

def test_outcomes_builder_flagging() -> None:
    logger.info("Starting test: test_outcomes_builder_flagging")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        (tmp_path / "processed").mkdir()
        (tmp_path / "raw").mkdir() # Need raw dir for ICUSTAYS
        
        config = UnitFixtures.get_minimal_etl_config().model_copy(update={
            "proc_dir": str(tmp_path / "processed"),
            "raw_dir": str(tmp_path / "raw")
        })
        
        # [Fix] Create dummy ICUSTAYS.csv.gz
        icu_df = pl.DataFrame({
            "SUBJECT_ID": [1], "HADM_ID": [10], "ICUSTAY_ID": [100],
            "INTIME": ["2100-01-01 00:00:00"], "OUTTIME": ["2100-01-03 00:00:00"]
        })
        with gzip.open(tmp_path / "raw" / "ICUSTAYS.csv.gz", "wb") as f:
            icu_df.write_csv(f)

        pl.DataFrame({
            "SUBJECT_ID": [1], "HADM_ID": [10], "ICUSTAY_ID": [100],
            "INTIME": ["2100-01-01 00:00:00"], "DOD": ["2100-01-03 02:00:00"], "DISCHTIME": [None]
        }).write_parquet(tmp_path / "processed" / "patients.parquet")
        
        # Match lengths for valid DataFrame creation
        pl.DataFrame({
            "SUBJECT_ID": [1, 1, 1, 1], "HADM_ID": [10, 10, 10, 10], "ICUSTAY_ID": [100, 100, 100, 100],
            "HOUR_IN": [48, 49, 50, 51]
        }).write_parquet(tmp_path / "processed" / "interventions.parquet")

        builder = OutcomesBuilder(config, MagicMock(), MagicMock())
        builder.execute()
        
        df = pl.read_parquet(tmp_path / "processed" / "interventions.parquet")
        assert "OUTCOME_FLAG" in df.columns