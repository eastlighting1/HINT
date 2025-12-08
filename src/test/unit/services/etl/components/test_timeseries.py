import polars as pl
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from loguru import logger
from hint.services.etl.components.timeseries import TimeSeriesAggregator
from ...conftest import UnitFixtures

def test_timeseries_aggregation_hourly() -> None:
    """
    Validates that TimeSeriesAggregator aggregates raw events into hourly mean values.
    
    Test Case ID: ETL-TS-01
    Description:
        Creates raw CHARTEVENTS.
        Executes aggregator.
        Verifies result is grouped by hour and averaged.
    """
    logger.info("Starting test: test_timeseries_aggregation_hourly")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        (tmp_path / "raw").mkdir()
        (tmp_path / "processed").mkdir()

        config = UnitFixtures.get_minimal_etl_config().model_copy(update={
            "raw_dir": str(tmp_path / "raw"),
            "proc_dir": str(tmp_path / "processed")
        })

        # Mock CHARTEVENTS
        pl.DataFrame({
            "SUBJECT_ID": [1, 1], "HADM_ID": [10, 10], "ICUSTAY_ID": [100, 100],
            "CHARTTIME": ["2100-01-01 10:15:00", "2100-01-01 10:45:00"],
            "ITEMID": [220045, 220045], "VALUENUM": [80, 100]
        }).write_csv(tmp_path / "raw" / "CHARTEVENTS.csv")

        # Mock Patients for Intime
        pl.DataFrame({
            "SUBJECT_ID": [1], "HADM_ID": [10], "ICUSTAY_ID": [100], "INTIME": ["2100-01-01 10:00:00"]
        }).write_parquet(tmp_path / "processed" / "patients.parquet")

        aggregator = TimeSeriesAggregator(config, MagicMock(), MagicMock())
        aggregator.execute()

        out_file = tmp_path / "processed" / "vitals_labs_mean.parquet"
        assert out_file.exists()
        
        df = pl.read_parquet(out_file)
        # Should be 1 row for hour 0 with mean 90
        assert df.height == 1
        assert df["MEAN"][0] == 90.0

    logger.info("TimeSeriesAggregator hourly aggregation verified.")