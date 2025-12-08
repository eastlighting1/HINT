import torch
import tempfile
from pathlib import Path
from loguru import logger
from hint.infrastructure.registry import FileSystemRegistry

def test_filesystem_registry_save_load() -> None:
    """
    Verify FileSystemRegistry saves and loads artifacts correctly.

    This test validates that `FileSystemRegistry` persists a model state to disk and the saved payload can be reloaded with matching content.
    - Test Case ID: INF-REG-01
    - Scenario: Persist and reload a mock model state within a temporary directory.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_filesystem_registry_save_load")

    with tempfile.TemporaryDirectory() as tmp_dir:
        registry = FileSystemRegistry(artifacts_dir=tmp_dir)
        
        state = {"weights": torch.tensor([1.0, 2.0]), "epoch": 10}
        
        logger.debug("Saving model artifact")
        registry.save_model(state, "my_model", "best")
        
        expected_path = Path(tmp_dir) / "my_model_best.pt"
        assert expected_path.exists()
        
        logger.debug("Loading model artifact manually to verify content")
        loaded = torch.load(expected_path)
        assert loaded["epoch"] == 10
        assert torch.equal(loaded["weights"], state["weights"])

    logger.info("FileSystemRegistry save/load verified.")
