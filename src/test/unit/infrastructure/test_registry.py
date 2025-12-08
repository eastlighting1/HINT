import torch
import tempfile
from pathlib import Path
from loguru import logger
from hint.infrastructure.registry import FileSystemRegistry

def test_filesystem_registry_save_load() -> None:
    """
    Validates saving and loading artifacts using FileSystemRegistry.
    
    Test Case ID: INF-REG-01
    Description:
        Initializes the registry in a temp directory.
        Saves a dictionary artifact (simulating a state_dict).
        Verifies the file exists and can be loaded back.
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