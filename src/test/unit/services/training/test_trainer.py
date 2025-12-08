import torch
from unittest.mock import MagicMock, patch
from loguru import logger
from hint.services.training.trainer import TrainingService
from hint.domain.entities import InterventionModelEntity
from hint.domain.vo import CNNConfig
from hint.foundation.dtos import TensorBatch
from ....conftest import TestFixtures

def test_train_model_runs_one_epoch() -> None:
    """
    Validates that TrainingService runs a full training epoch and saves the model.
    
    Test Case ID: TRN-01
    Description:
        Mocks the DataLoader to return a single dummy batch.
        Runs 'train_model' for 1 epoch.
        Verifies that the entity's epoch counter increases and the save_model method is called.
        Checks if metrics are tracked via the observer.
    """
    logger.info("Starting test: test_train_model_runs_one_epoch")

    mock_registry = TestFixtures.get_mock_registry()
    mock_observer = TestFixtures.get_mock_observer()
    
    config = CNNConfig(
        data_path="dummy", 
        data_cache_dir="dummy",
        batch_size=2, 
        epochs=1, 
        patience=1
    )
    
    logger.debug("Initializing dummy network and entity")
    net = torch.nn.Linear(10, 4)
    entity = InterventionModelEntity(net)
    
    service = TrainingService(
        config, mock_registry, mock_observer, entity, device="cpu"
    )

    logger.debug("Mocking DataLoader and Batch")
    batch = TensorBatch(
        x_num=torch.randn(2, 10),
        x_cat=None,
        y=torch.tensor([0, 1]),
        ids=None
    )

    mock_loader = MagicMock()
    mock_loader.__len__.return_value = 1
    mock_loader.__iter__.return_value = iter([batch])

    with patch("hint.services.training.trainer.DataLoader", return_value=mock_loader) as MockDL:
        logger.info("Calling train_model")
        service.train_model(MagicMock(), MagicMock())
        
        logger.debug("Verifying entity state after training")
        assert service.entity.epoch == 1, f"Expected entity epoch 1, got {service.entity.epoch}"
        
        logger.debug("Verifying model checkpoint saving")
        mock_registry.save_model.assert_called()
        
        logger.debug("Verifying metric tracking")
        mock_observer.track_metric.assert_any_call("cnn_train_loss", torch.tensor(0.0).float(), step=1) # approx check skipped for simplicity in mock
        
        logger.info("Training service workflow verified successfully.")