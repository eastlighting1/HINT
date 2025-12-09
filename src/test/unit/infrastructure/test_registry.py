import torch
import tempfile
from pathlib import Path
from loguru import logger
from src.hint.infrastructure.registry import FileSystemRegistry

def test_filesystem_registry_save_load() -> None:
    """
    Validates saving and loading artifacts using FileSystemRegistry.
    Test Case ID: INF-REG-01
    """
    logger.info("Starting test: test_filesystem_registry_save_load")

    with tempfile.TemporaryDirectory() as tmp_dir:
        registry = FileSystemRegistry(tmp_dir)
        
        state = {"weights": torch.tensor([1.0, 2.0]), "epoch": 10}
        
        registry.save_model(state, "my_model", "best")
        
        # Check if any file was created starting with my_model
        found_files = list(Path(tmp_dir).rglob("my_model*"))
        assert len(found_files) > 0, f"No model file found in {tmp_dir}"
        
        saved_path = found_files[0]
        logger.debug(f"Found saved artifact at {saved_path}")
        
        loaded = torch.load(saved_path)
        assert loaded["epoch"] == 10
        assert torch.equal(loaded["weights"], state["weights"])

    logger.info("FileSystemRegistry save/load verified.")