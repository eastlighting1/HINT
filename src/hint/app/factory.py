import json
from pathlib import Path
from omegaconf import DictConfig

from ..foundation.configs import HINTConfig, DataConfig, ModelConfig, TrainingConfig
from ..domain.entities import TrainableEntity
from ..infrastructure.networks import HINT_CNN
from ..infrastructure.datasource import HDF5StreamingSource
from ..infrastructure.registry import FileSystemRegistry
from ..infrastructure.telemetry import RichTelemetryObserver
from ..services.trainer import TrainingService

class AppFactory:
    """
    Factory class responsible for object creation and dependency injection.

    Converts raw Hydra DictConfig into domain-specific Value Objects and
    assembles the application components.

    Args:
        cfg: Raw Hydra configuration.
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._load_feature_info()

    def _load_feature_info(self) -> None:
        """
        Load feature metadata from feature_info.json in the cache directory.
        """
        cache_dir = Path(self.cfg.cnn.data.data_cache_dir)
        info_path = cache_dir / "feature_info.json"
        
        if info_path.exists():
            with open(info_path, "r") as f:
                self.feat_info = json.load(f)
        else:
            self.feat_info = {}

    def create_configs(self) -> HINTConfig:
        """
        Create domain configuration objects from Hydra config.

        Returns:
            Fully populated HINTConfig object.
        """
        d_cfg = DataConfig(
            data_path=self.cfg.cnn.data.path,
            batch_size=self.cfg.cnn.model.batch_size,
            seq_len=self.cfg.cnn.model.seq_len
        )
        
        nbf = self.feat_info.get("n_base_feats_numeric", 0)
        
        g1 = list(range(nbf)) 
        g2 = list(range(nbf, 2 * nbf))
        rest = list(range(2 * nbf, 3 * nbf))
        
        m_cfg = ModelConfig(
            embed_dim=self.cfg.cnn.model.embed_dim,
            dropout=self.cfg.cnn.model.dropout,
            g1_indices=g1,
            g2_indices=g2,
            rest_indices=rest,
            vocab_sizes=self.feat_info.get("vocab_info", {})
        )
        
        t_cfg = TrainingConfig(
            epochs=self.cfg.cnn.model.epochs,
            lr=self.cfg.cnn.model.lr,
            device="cuda"
        )
        
        return HINTConfig(
            data=d_cfg, 
            model=m_cfg, 
            train=t_cfg, 
            artifact_dir=self.cfg.cnn.artifact_dir
        )

    def create_entity(self, hint_cfg: HINTConfig) -> TrainableEntity:
        """
        Create the TrainableEntity with the neural network.

        Args:
            hint_cfg: HINT configuration.

        Returns:
            Initialized TrainableEntity.
        """
        in_chs = [
            len(hint_cfg.model.g1_indices), 
            len(hint_cfg.model.g2_indices), 
            len(hint_cfg.model.rest_indices)
        ]
        
        network = HINT_CNN(
            in_chs=in_chs,
            n_cls=hint_cfg.model.n_classes,
            g1=hint_cfg.model.g1_indices,
            g2=hint_cfg.model.g2_indices,
            rest=hint_cfg.model.rest_indices,
            cat_vocab_sizes=hint_cfg.model.vocab_sizes,
            embed_dim=hint_cfg.model.embed_dim
        )
        return TrainableEntity(network=network, config=hint_cfg.train)

    def create_service(self, hint_cfg: HINTConfig) -> TrainingService:
        """
        Create the TrainingService with necessary infrastructure.

        Args:
            hint_cfg: HINT configuration.

        Returns:
            Initialized TrainingService.
        """
        registry = FileSystemRegistry(root_dir=f"{hint_cfg.artifact_dir}/checkpoints")
        observer = RichTelemetryObserver(log_filename="train.log")
        return TrainingService(
            registry=registry, 
            observer=observer, 
            device=hint_cfg.train.device
        )

    def create_sources(self, hint_cfg: HINTConfig):
        """
        Create streaming sources for training and validation.

        Args:
            hint_cfg: HINT configuration.

        Returns:
            Tuple of (train_source, val_source).
        """
        cache_dir = self.cfg.cnn.data.data_cache_dir
        train_path = f"{cache_dir}/train.h5"
        val_path = f"{cache_dir}/val.h5"
        
        train_d_cfg = DataConfig(data_path=train_path, batch_size=hint_cfg.data.batch_size)
        val_d_cfg = DataConfig(data_path=val_path, batch_size=hint_cfg.data.batch_size)
        
        return HDF5StreamingSource(train_d_cfg), HDF5StreamingSource(val_d_cfg)