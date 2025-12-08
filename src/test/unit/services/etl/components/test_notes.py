import polars as pl
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from loguru import logger
from hint.services.etl.components.notes import NoteTokenizer
from ...conftest import UnitFixtures

def test_note_tokenizer_execution() -> None:
    """
    Verify NoteTokenizer converts raw notes into tokenized outputs.

    This test validates that `NoteTokenizer` reads raw note text, tokenizes the content, and writes processed Parquet output containing token ID lists.
    - Test Case ID: ETL-NOT-01
    - Scenario: Tokenize a single synthetic NOTEEVENTS record.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_note_tokenizer_execution")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        (tmp_path / "raw").mkdir()
        (tmp_path / "processed").mkdir()

        config = UnitFixtures.get_minimal_etl_config().model_copy(update={
            "raw_dir": str(tmp_path / "raw"),
            "proc_dir": str(tmp_path / "processed")
        })

        pl.DataFrame({
            "SUBJECT_ID": [1], "HADM_ID": [10], "CHARTDATE": ["2100-01-01"],
            "TEXT": ["Patient is stable. No complaints."]
        }).write_csv(tmp_path / "raw" / "NOTEEVENTS.csv")

        tokenizer = NoteTokenizer(config, MagicMock(), MagicMock())
        tokenizer.execute()

        out_file = tmp_path / "processed" / "notes_tokenized.parquet"
        assert out_file.exists()
        
        df = pl.read_parquet(out_file)
        assert "INPUT_IDS" in df.columns
        assert isinstance(df["INPUT_IDS"][0], list)

    logger.info("NoteTokenizer execution verified.")
