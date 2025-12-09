import polars as pl
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from loguru import logger
from src.hint.services.etl.components.notes import NoteTokenizer
from src.test.unit.conftest import UnitFixtures

def test_note_tokenizer_execution() -> None:
    logger.info("Starting test: test_note_tokenizer_execution")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        (tmp_path / "raw").mkdir()
        (tmp_path / "processed").mkdir()
        config = UnitFixtures.get_minimal_etl_config().model_copy(update={
            "raw_dir": str(tmp_path / "raw"), "proc_dir": str(tmp_path / "processed")
        })
        pl.DataFrame({
            "SUBJECT_ID": [1], "HADM_ID": [10], "CHARTDATE": ["2100-01-01"],
            "TEXT": ["Patient is stable."]
        }).write_csv(tmp_path / "raw" / "NOTEEVENTS.csv")

        tokenizer = NoteTokenizer(config, MagicMock(), MagicMock())
        tokenizer.execute()
        assert (tmp_path / "processed" / "notes_tokenized.parquet").exists()