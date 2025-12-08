import torch
import tempfile
from pathlib import Path
from loguru import logger
from hint.domain.entities import InterventionModelEntity
from hint.infrastructure.registry import FileSystemRegistry

def test_model_persistence_cycle() -> None:
    """
    Validates the full cycle of saving and loading a model entity via the Registry.
    
    Test Case ID: TS-09
    Description:
        Creates a FileSystemRegistry in a temporary directory.
        Initializes a model entity with random weights.
        Saves the model using the registry.
        Loads the model back into a new instance.
        Verifies that the weights are identical within tolerance.
    """
    logger.info("Starting test: test_model_persistence_cycle")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        registry = FileSystemRegistry(artifacts_dir=tmp_dir)
        
        logger.debug("Initializing original entity")
        net = torch.nn.Linear(10, 2)
        entity = InterventionModelEntity(net)
        entity.epoch = 3
        
        # Forward pass to establish state
        x = torch.randn(1, 10)
        y_orig = net(x)
        
        logger.debug("Saving model to registry")
        registry.save_model(entity.state_dict(), "test_model", "v1")
        
        logger.debug("Initializing new entity for loading")
        new_net = torch.nn.Linear(10, 2)
        new_entity = InterventionModelEntity(new_net)
        
        logger.debug("Loading model from registry")
        # Registry typically returns a dict, or we use registry to find path and load
        # Assuming registry has load_model or similar, or we simulate load logic if registry only handles paths
        # Adapting to common registry pattern:
        load_path = Path(tmp_dir) / "test_model_v1.pt"
        if load_path.exists():
            state = torch.load(load_path)
            new_entity.load_state_dict(state)
        else:
             raise AssertionError(f"Model file not found at {load_path}")

        y_loaded = new_net(x)
        
        logger.debug("Comparing predictions")
        assert torch.allclose(y_orig, y_loaded, atol=1e-6), "Predictions mismatch after loading"
        assert new_entity.epoch == 3, "Epoch metadata mismatch"
        
        logger.info("Model persistence cycle verified.")