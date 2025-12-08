import torch
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from loguru import logger
from hint.services.training.trainer import TrainingService
from hint.domain.entities import InterventionModelEntity
from hint.infrastructure.datasource import HDF5StreamingSource
from ..conftest import IntegrationFixtures
from ...unit.conftest import UnitFixtures

def test_full_training_loop_integration() -> None:
    """
    Verify a minimal training loop executes end-to-end with HDF5 sources.

    This test validates that `TrainingService` can consume synthetic HDF5 datasets, run a single epoch, and trigger checkpoint persistence while advancing the entity epoch counter.
    - Test Case ID: TS-11
    - Scenario: Train for one epoch on synthetic data using a compatible mock network.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_full_training_loop_integration")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        train_path = tmp_path / "train.h5"
        val_path = tmp_path / "val.h5"
        
        IntegrationFixtures.create_dummy_hdf5(train_path, num_samples=20, seq_len=10)
        IntegrationFixtures.create_dummy_hdf5(val_path, num_samples=10, seq_len=10)
        
        cnn_config = UnitFixtures.get_minimal_cnn_config()
        cnn_config = cnn_config.model_copy(update={"batch_size": 4, "epochs": 1})
        
        train_src = HDF5StreamingSource(train_path, max_seq_len=10)
        val_src = HDF5StreamingSource(val_path, max_seq_len=10)
        
        class CompatMockNet(torch.nn.Module):
            """
            Minimal network compatible with TrainingService expectations.
            """
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(5, 4)

            def forward(self, x_num, x_cat):
                x_pooled = x_num.mean(dim=2)
                return self.fc(x_pooled)

        entity = InterventionModelEntity(CompatMockNet())
        
        mock_registry = MagicMock()
        mock_observer = MagicMock()
        mock_observer.create_progress.return_value.__enter__.return_value = MagicMock()
        
        service = TrainingService(
            cnn_config, mock_registry, mock_observer, entity, device="cpu"
        )
        
        logger.info("Executing training loop")
        service.train_model(train_src, val_src)
        
        logger.debug("Verifying execution results")
        assert entity.epoch == 1, "Epoch count did not increment"
        mock_registry.save_model.assert_called(), "Model checkpoint was not saved"
        
        logger.info("Full training loop integration verified.")
