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
    [One-line Summary] Verify InterventionModelEntity initializes with expected defaults.

    [Description]
    Construct InterventionModelEntity with a mock CNN and confirm the name, best metric,
    EMA tracker, and epoch counter are set to the baseline values expected before training.

    Test Case ID: ENT-01
    Scenario: Instantiate InterventionModelEntity using a mock CNN dependency.

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
    [One-line Summary] Validate state_dict save/load cycle for InterventionModelEntity.

    [Description]
    Mutate entity training state, export the state_dict, load it into a new instance, and
    verify epoch, metric, and temperature fields are restored to prove persistence integrity.

    Test Case ID: ENT-02
    Scenario: Persist and restore an entity after mutating training state.

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
    [One-line Summary] Confirm InterventionModelEntity moves its network to the requested device.

    [Description]
    Invoke `to` on an entity backed by a mock CNN and assert that model parameters reside on
    the requested device string so subsequent training occurs on the intended hardware.

    Test Case ID: ENT-03
    Scenario: Move an entity backed by a mock CNN onto the CPU device.

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
