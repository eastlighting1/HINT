import torch
from unittest.mock import MagicMock
from loguru import logger
from hint.services.training.evaluator import EvaluationService
from hint.domain.entities import InterventionModelEntity

def test_evaluator_metrics_calculation() -> None:
    """
    Verify EvaluationService initializes with injected dependencies.

    This test validates that EvaluationService accepts mocked registry, observer, and entity instances while preserving the configured device for later evaluation work.
    - Test Case ID: TRN-EVAL-01
    - Scenario: Construct evaluation service with minimal mocks and inspect attributes.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_evaluator_metrics_calculation")

    mock_registry = MagicMock()
    mock_observer = MagicMock()
    mock_config = MagicMock()
    
    net = torch.nn.Linear(10, 2)
    entity = InterventionModelEntity(net)
    
    service = EvaluationService(mock_config, mock_registry, mock_observer, entity, "cpu")
    
    assert service.entity == entity
    assert service.device == "cpu"
    
    logger.info("EvaluationService initialized successfully.")
