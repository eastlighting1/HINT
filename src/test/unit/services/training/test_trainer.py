import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from loguru import logger
from src.hint.services.training.trainer import TrainingService
from src.hint.domain.entities import InterventionModelEntity
from src.hint.domain.vo import CNNConfig
from src.hint.foundation.dtos import TensorBatch
from src.test.conftest import TestFixtures  # Fixed Import

class MockNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 4)
    def forward(self, x_num, x_cat):
        if x_num.dim() == 3: x_pooled = x_num.mean(dim=2)
        else: x_pooled = x_num
        return self.fc(x_pooled)

def test_train_model_runs_one_epoch() -> None:
    """
    Validates that TrainingService runs a full training epoch.
    Test Case ID: TRN-01
    """
    logger.info("Starting test: test_train_model_runs_one_epoch")

    mock_registry = TestFixtures.get_mock_registry()
    mock_observer = TestFixtures.get_mock_observer()
    
    config = CNNConfig(data_path="dummy", data_cache_dir="dummy", batch_size=2, epochs=1, patience=1)
    
    net = MockNet()
    entity = InterventionModelEntity(net)
    
    service = TrainingService(config, mock_registry, mock_observer, entity, device="cpu")

    batch = TensorBatch(
        x_num=torch.randn(2, 10),
        x_cat=torch.zeros(2, 2).long(),
        y=torch.tensor([0, 1]),
        ids=None
    )

    mock_loader = MagicMock()
    mock_loader.__len__.return_value = 1
    mock_loader.__iter__.return_value = iter([batch])

    with patch("src.hint.services.training.trainer.DataLoader", return_value=mock_loader):
        logger.info("Calling train_model")
        service.train_model(MagicMock(), MagicMock())
        
        assert service.entity.epoch == 1
        mock_registry.save_model.assert_called()