import torch
import tempfile
from loguru import logger
from hint.domain.entities import InterventionModelEntity
from hint.infrastructure.registry import FileSystemRegistry

def test_model_persistence_cycle() -> None:
    """
    [One-line Summary] Validate registry save/load preserves model weights and metadata.

    [Description]
    Save an InterventionModelEntity via FileSystemRegistry, reload it into a fresh entity, and confirm
    predictions and epoch metadata match to prove the persistence cycle is lossless.

    Test Case ID: TS-09
    Scenario: Persist and restore a model state within a temporary artifacts directory.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_model_persistence_cycle")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        registry = FileSystemRegistry(base_dir=tmp_dir)
        
        logger.debug("Initializing original entity")
        net = torch.nn.Linear(10, 2)
        entity = InterventionModelEntity(net)
        entity.epoch = 3
        
        x = torch.randn(1, 10)
        y_orig = net(x)
        
        logger.debug("Saving model to registry")
        registry.save_model(entity.state_dict(), "test_model", "v1")
        
        logger.debug("Initializing new entity for loading")
        new_net = torch.nn.Linear(10, 2)
        new_entity = InterventionModelEntity(new_net)
        
        logger.debug("Loading model from registry")
        state = registry.load_model("test_model", "v1", device="cpu")
        new_entity.load_state_dict(state)

        y_loaded = new_net(x)
        
        logger.debug("Comparing predictions")
        assert torch.allclose(y_orig, y_loaded, atol=1e-6), "Predictions mismatch after loading"
        assert new_entity.epoch == 3, "Epoch metadata mismatch"
        
        logger.info("Model persistence cycle verified.")
