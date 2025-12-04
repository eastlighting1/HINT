from dataclasses import dataclass, field
from typing import List

@dataclass(frozen=True)
class DataConfig:
    """
    Configuration for data loading and processing.

    Args:
        data_path: Path to the input data file.
        seq_len: Sequence length for the time-series data.
        batch_size: Number of samples per batch.
        num_workers: Number of subprocesses for data loading.
    """
    data_path: str
    seq_len: int = 120
    batch_size: int = 512
    num_workers: int = 4

@dataclass(frozen=True)
class ModelConfig:
    """
    Configuration for the neural network model structure.

    Args:
        embed_dim: Dimension of the embedding layers.
        dropout: Dropout rate for the classifier head.
        tcn_kernel_size: Kernel size for the TCN layers.
        tcn_layers: Number of TCN layers.
        tcn_dropout: Dropout rate for the TCN layers.
        n_classes: Number of target classes.
        g1_indices: Indices for the first group of features.
        g2_indices: Indices for the second group of features.
        rest_indices: Indices for the remaining features.
        vocab_sizes: Dictionary mapping categorical feature names to vocabulary sizes.
    """
    embed_dim: int = 128
    dropout: float = 0.5
    tcn_kernel_size: int = 5
    tcn_layers: int = 5
    tcn_dropout: float = 0.4
    n_classes: int = 4
    g1_indices: List[int] = field(default_factory=list)
    g2_indices: List[int] = field(default_factory=list)
    rest_indices: List[int] = field(default_factory=list)
    vocab_sizes: dict = field(default_factory=dict)

@dataclass(frozen=True)
class TrainingConfig:
    """
    Configuration for the training process.

    Args:
        epochs: Number of total epochs to run.
        lr: Learning rate for the optimizer.
        patience: Patience for early stopping.
        focal_gamma: Gamma parameter for Focal Loss.
        ema_decay: Decay rate for Exponential Moving Average.
        device: Computation device (e.g., 'cuda', 'cpu').
    """
    epochs: int = 100
    lr: float = 0.001
    patience: int = 10
    focal_gamma: float = 2.0
    ema_decay: float = 0.999
    device: str = "cuda"

@dataclass(frozen=True)
class HINTConfig:
    """
    Root configuration aggregating all sub-configurations.

    Args:
        data: Data configuration.
        model: Model configuration.
        train: Training configuration.
        project_name: Name of the project.
        artifact_dir: Directory to store artifacts.
    """
    data: DataConfig
    model: ModelConfig
    train: TrainingConfig
    project_name: str = "HINT"
    artifact_dir: str = "artifacts"