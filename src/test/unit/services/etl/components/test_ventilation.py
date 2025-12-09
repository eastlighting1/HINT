import polars as pl
import tempfile
from pathlib import Path
from loguru import logger
from src.hint.services.etl.components.ventilation import VentilationTagger
from src.test.unit.conftest import UnitFixtures
from src.test.conftest import TestFixtures

def test_ventilation_tagger_marks_vent_correctly() -> None:
    logger.info("Starting test: test_ventilation_tagger_marks_vent_correctly")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        res_dir = tmp_path / "resources"
        proc_dir = tmp_path / "processed"
        res_dir.mkdir()
        proc_dir.mkdir()
        
        config = UnitFixtures.get_minimal_etl_config().model_copy(update={
            "resources_dir": str(res_dir), "proc_dir": str(proc_dir)
        })

        pl.DataFrame({
            "ITEMID": [445, 999], "MIMIC LABEL": ["Ventilator", "Other"], "LINKSTO": ["chartevents", "chartevents"]
        }).write_csv(res_dir / "itemid_to_variable_map.csv")

        pl.DataFrame({
            "SUBJECT_ID": [101], "HADM_ID": [1001], "ICUSTAY_ID": [5001],
            "HOURS_IN": [10], "LABEL": ["Ventilator"], "VALUENUM": [1]
        }).write_parquet(proc_dir / "vitals_labs.parquet")

        pl.DataFrame({
            "SUBJECT_ID": [101, 101], "HADM_ID": [1001, 1001], "ICUSTAY_ID": [5001, 5001],
            "HOURS_IN": [10, 11], "OUTCOME_FLAG": [0, 0]
        }).write_parquet(proc_dir / "interventions.parquet")

        tagger = VentilationTagger(config, TestFixtures.get_mock_registry(), TestFixtures.get_mock_observer())
        tagger.execute()
        
        result = pl.read_parquet(proc_dir / "interventions.parquet")
        assert result.filter(pl.col("HOUR_IN") == 10)["VENT"][0] == 1
        assert result.filter(pl.col("HOUR_IN") == 11)["VENT"][0] == 0