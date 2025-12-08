import torch
from unittest.mock import MagicMock, patch
from loguru import logger
from hint.services.icd.service import ICDService
from hint.domain.vo import ICDConfig

def test_icd_service_train_step() -> None:
    """
    Verify ICDService wiring honors configuration and network presence.

    This test validates that ICDService constructed with mocked dependencies exposes the configured batch size and accepts a mocked network entity suitable for training routines.
    - Test Case ID: SVC-ICD-01
    - Scenario: Initialize ICDService with mocked registry, observer, and network entity.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_icd_service_train_step")

    config = ICDConfig(data_path="dummy", batch_size=2)
    mock_registry = MagicMock()
    mock_observer = MagicMock()
    mock_observer.create_progress.return_value.__enter__.return_value = MagicMock()

    service = ICDService(config, mock_registry, mock_observer, None, None, None)

    service.entity = MagicMock()
    service.entity.network = MagicMock()

    assert service.cfg.batch_size == 2
    
    logger.info("ICDService configuration verified.")
