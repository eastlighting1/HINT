import torch
import tempfile
import h5py
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock
from loguru import logger
from src.hint.services.training.trainer import TrainingService
from src.hint.domain.entities import InterventionModelEntity
from src.hint.infrastructure.datasource import HDF5StreamingSource
from src.test.unit.conftest import UnitFixtures

def test_full_training_loop_integration() -> None:
    """
    Validates the full training loop with real data sources and entity updates.
    
    Test Case ID: TS-11
    """
    logger.info("Starting test: test_full_training_loop_integration")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        train_path = tmp_path / "train.h5"
        val_path = tmp_path / "val.h5"
        
        seq_len = 10
        for path in [train_path, val_path]:
            with h5py.File(path, 'w') as f:
                f.create_dataset("x_num", data=np.random.randn(10, seq_len, 5).astype(np.float32))
                f.create_dataset("x_cat", data=np.random.randint(0, 5, (10, seq_len, 2)).astype(np.int64))
                f.create_dataset("y", data=np.random.randint(0, 4, (10,)).astype(np.int64))
                f.create_dataset("ids", data=np.arange(10).astype(np.int64))
                f.create_dataset("mask", data=np.ones((10, seq_len)).astype(np.float32))
        
        cnn_config = UnitFixtures.get_minimal_cnn_config()
        cnn_config = cnn_config.model_copy(update={"batch_size": 4, "epochs": 1})
        
        # [Fix] Pass seq_len
        train_src = HDF5StreamingSource(train_path, seq_len)
        val_src = HDF5StreamingSource(val_path, seq_len)
        
        class CompatMockNet(torch.nn.Module):
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
        
        service.train_model(train_src, val_src)
        
        assert entity.epoch == 1
        mock_registry.save_model.assert_called()
        
        logger.info("Full training loop integration verified.")