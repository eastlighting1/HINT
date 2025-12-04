import pytest
import torch
from pathlib import Path
from omegaconf import OmegaConf
from hint.foundation.configs import HINTConfig, DataConfig, ModelConfig, TrainingConfig, ICDConfig

@pytest.fixture(scope="session")
def test_config():
    """
    Global test configuration fixture.
    """
    return HINTConfig(
        data=DataConfig(
            data_path="tests/resources/data/tiny.h5",
            seq_len=10,
            batch_size=4,
            num_workers=0
        ),
        model=ModelConfig(
            embed_dim=16,
            dropout=0.1,
            tcn_kernel_size=3,
            tcn_layers=2,
            n_classes=4,
            vocab_sizes={"cat1": 10, "cat2": 5},
            g1_indices=[0, 1],
            g2_indices=[2, 3],
            rest_indices=[4, 5]
        ),
        train=TrainingConfig(
            epochs=2,
            lr=0.01,
            device="cpu"
        ),
        icd=ICDConfig(
            model_name="prajjwal1/bert-tiny", # Small model for fast testing
            batch_size=4,
            epochs=1
        ),
        artifact_dir="tests/artifacts"
    )

@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"