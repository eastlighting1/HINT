import torch
import tempfile
from pathlib import Path
from loguru import logger
from src.hint.infrastructure.registry import FileSystemRegistry

def test_filesystem_registry_save_load() -> None:
    """
    [One-line Summary] Validate FileSystemRegistry persists and restores model artifacts.

    [Description]
    Save a model state dict to a temporary directory, confirm the artifact is written, and
    reload it to verify epoch and weight tensors match the original values.

    Test Case ID: INF-REG-01
    Scenario: Persist and reload model weights and metadata using FileSystemRegistry.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_filesystem_registry_save_load")

    with tempfile.TemporaryDirectory() as tmp_dir:
        registry = FileSystemRegistry(tmp_dir)
        
        state = {"weights": torch.tensor([1.0, 2.0]), "epoch": 10}
        
        registry.save_model(state, "my_model", "best")
        
        logger.info("Searching for persisted model artifact on disk.")
        found_files = list(Path(tmp_dir).rglob("my_model*"))
        assert len(found_files) > 0, f"No model file found in {tmp_dir}"
        
        saved_path = found_files[0]
        logger.debug(f"Found saved artifact at {saved_path}")
        
        loaded = torch.load(saved_path)
        assert loaded["epoch"] == 10
        assert torch.equal(loaded["weights"], state["weights"])

    logger.info("FileSystemRegistry save/load verified.")
