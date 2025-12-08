import torch
import torch.nn as nn
from loguru import logger
from hint.domain.entities import InterventionModelEntity
from ...utils.custom_assertions import assert_raises

class MockCNN(nn.Module):
    """
    A simple mock CNN for testing Entity wrapping logic.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

def test_intervention_entity_initialization() -> None:
    """
    Verify InterventionModelEntity initializes with expected defaults.

    This test validates that constructing `InterventionModelEntity` with a mock CNN sets the name, best metric, EMA tracker, and epoch counter to their expected starting values.
    - Test Case ID: ENT-01
    - Scenario: Instantiate InterventionModelEntity using a mock CNN dependency.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_intervention_entity_initialization")
    
    net = MockCNN()
    entity = InterventionModelEntity(net, ema_decay=0.9)
    
    logger.debug("Verifying entity properties")
    assert entity.name == "intervention_cnn", f"Expected name 'intervention_cnn', got {entity.name}"
    assert entity.best_metric == -float('inf'), "Initial best_metric should be -inf"
    assert entity.ema is not None, "EMA object should be initialized"
    assert entity.epoch == 0, "Initial epoch should be 0"
    
    logger.info("InterventionModelEntity initialization verified.")

def test_intervention_entity_state_dict_cycle() -> None:
    """
    Validate state_dict save and load cycle for InterventionModelEntity.

    This test modifies entity attributes, exports `state_dict`, loads it into a new instance, and confirms the restored fields match to guarantee persistence correctness.
    - Test Case ID: ENT-02
    - Scenario: Persist and restore an entity after mutating training state.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_intervention_entity_state_dict_cycle")
    
    net = MockCNN()
    entity = InterventionModelEntity(net)
    entity.epoch = 5
    entity.best_metric = 0.8
    entity.temperature = 1.5
    
    logger.debug("Exporting state_dict from source entity")
    state = entity.state_dict()
    
    new_entity = InterventionModelEntity(MockCNN())
    
    logger.debug("Loading state_dict into new entity")
    new_entity.load_state_dict(state)
    
    logger.debug("Verifying restored state values")
    assert new_entity.epoch == 5, f"Expected epoch 5, got {new_entity.epoch}"
    assert new_entity.best_metric == 0.8, f"Expected best_metric 0.8, got {new_entity.best_metric}"
    assert new_entity.temperature == 1.5, f"Expected temperature 1.5, got {new_entity.temperature}"
    
    logger.info("Entity state restore cycle verified.")

def test_entity_device_move() -> None:
    """
    Confirm InterventionModelEntity moves its network to the requested device.

    This test ensures calling `to` relocates model parameters to the target device string so downstream training uses the correct hardware placement.
    - Test Case ID: ENT-03
    - Scenario: Move an entity backed by a mock CNN onto the CPU device.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_entity_device_move")
    
    net = MockCNN()
    entity = InterventionModelEntity(net)
    
    target_device = "cpu"
    logger.debug(f"Moving entity to {target_device}")
    entity.to(target_device)
    
    param = next(entity.network.parameters())
    assert str(param.device).startswith(target_device), f"Parameter device mismatch: {param.device}"
    
    logger.info("Entity device move verified.")
