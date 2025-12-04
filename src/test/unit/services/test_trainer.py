import pytest
from unittest.mock import MagicMock, patch
from hint.services.trainer import TrainingService
from hint.foundation.dtos import TensorBatch
import torch

@pytest.fixture
def mock_components():
    registry = MagicMock()
    observer = MagicMock()
    entity = MagicMock()
    entity.best_metric = 0.0
    entity.predict.return_value = torch.tensor([[0.1, 0.9], [0.8, 0.2]]) # 2 samples, 2 classes
    
    # Mock Stream
    batch = MagicMock(spec=TensorBatch)
    batch.targets = torch.tensor([1, 0])
    batch.to.return_value = batch
    
    source = MagicMock()
    source.stream_batches.return_value = iter([batch])
    source.__len__.return_value = 1
    
    return registry, observer, entity, source

def test_training_service_workflow(mock_components):
    registry, observer, entity, source = mock_components
    
    service = TrainingService(registry, observer, device="cpu")
    
    # Run 1 epoch
    service.train_model(entity, source, source, epochs=1)
    
    # Verification
    # 1. Training step called?
    entity.step_train.assert_called()
    
    # 2. Validation called?
    entity.predict.assert_called()
    
    # 3. Metric logged?
    observer.track_metric.assert_any_call("train_loss", pytest.any(float), step=1)
    observer.track_metric.assert_any_call("val_acc", pytest.any(float), step=1)
    
    # 4. Checkpoint saved? (Acc 100% > 0.0)
    registry.save.assert_called_with(entity, tag="best")