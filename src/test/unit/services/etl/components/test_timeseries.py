import polars as pl
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from loguru import logger
from src.hint.services.etl.components.timeseries import TimeSeriesAggregator
from src.test.unit.conftest import UnitFixtures

def test_timeseries_aggregation_hourly() -> None:
    logger.info("Starting test: test_timeseries_aggregation_hourly")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        (tmp_path / "raw").mkdir()
        (tmp_path / "processed").mkdir()
        config = UnitFixtures.get_minimal_etl_config().model_copy(update={
            "raw_dir": str(tmp_path / "raw"), "proc_dir": str(tmp_path / "processed")
        })
        pl.DataFrame({
            "SUBJECT_ID": [1, 1], "HADM_ID": [10, 10], "ICUSTAY_ID": [100, 100],
            "CHARTTIME": ["2100-01-01 10:15:00", "2100-01-01 10:45:00"],
            "ITEMID": [220045, 220045], "VALUENUM": [80, 100]
        }).write_csv(tmp_path / "raw" / "CHARTEVENTS.csv")
        
        pl.DataFrame({
            "SUBJECT_ID": [1], "HADM_ID": [10], "ICUSTAY_ID": [100], "INTIME": ["2100-01-01 10:00:00"]
        }).write_parquet(tmp_path / "processed" / "patients.parquet")

        aggregator = TimeSeriesAggregator(config, MagicMock(), MagicMock())
        aggregator.execute()
        assert (tmp_path / "processed" / "vitals_labs_mean.parquet").exists()