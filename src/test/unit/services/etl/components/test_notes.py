import polars as pl
import tempfile
import gzip
import shutil
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
        
        csv_path = tmp_path / "raw" / "NOTEEVENTS.csv"
        # [Fix] Added ISERROR column (required for filtering) and CHARTTIME
        pl.DataFrame({
            "SUBJECT_ID": [1], "HADM_ID": [10], 
            "CHARTDATE": ["2100-01-01"], "CHARTTIME": ["2100-01-01 12:00:00"],
            "TEXT": ["Patient is stable."],
            "ISERROR": [None] 
        }).write_csv(csv_path)
        
        with open(csv_path, 'rb') as f_in:
            with gzip.open(str(csv_path) + ".gz", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Create ICUSTAYS for the join filter
        pl.DataFrame({
            "HADM_ID": [10], "ICUSTAY_ID": [100],
            "OUTTIME": ["2100-01-02 12:00:00"]
        }).write_csv(tmp_path / "raw" / "ICUSTAYS.csv.gz")

        tokenizer = NoteTokenizer(config, MagicMock(), MagicMock())
        tokenizer.execute()
        assert (tmp_path / "processed" / "notes_sentences.csv").exists()