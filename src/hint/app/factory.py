from pathlib import Path
import torch
from typing import List

from ..foundation.configs import HydraConfigLoader, load_app_context
from ..foundation.interfaces import Registry, TelemetryObserver, PipelineComponent
from ..domain.vo import ETLConfig, ICDConfig, CNNConfig
from ..domain.entities import InterventionModelEntity, ICDModelEntity
from ..infrastructure.registry import FileSystemRegistry
from ..infrastructure.telemetry import RichTelemetryObserver
from ..infrastructure.datasource import HDF5StreamingSource, ParquetSource
from ..infrastructure.networks import GFINet_CNN
from ..services.etl.service import ETLService
from ..services.etl.components.assembler import FeatureAssembler
from ..services.etl.components.tensor import TensorConverter
from ..services.etl.components.labels import LabelGenerator
from ..services.icd.service import ICDService
from ..services.training.trainer import TrainingService

class AppFactory:
    """
    Factory class to assemble components and services (Dependency Injection).
    """
    def __init__(self, config_name: str = "config", config_path: str = "configs"):
        self.loader = HydraConfigLoader(config_name=config_name, config_path=config_path)
        self.raw_cfg = self.loader.load()
        self.ctx = load_app_context(self.raw_cfg)

    def create_registry(self) -> Registry:
        path = self.raw_cfg.get("logging", {}).get("artifacts_dir", "artifacts")
        return FileSystemRegistry(base_dir=path)

    def create_telemetry(self) -> TelemetryObserver:
        return RichTelemetryObserver()

    def create_etl_service(self) -> ETLService:
        etl_cfg = self.ctx.etl
        cnn_cfg = self.ctx.cnn
        
        registry = self.create_registry()
        observer = self.create_telemetry()
        
        assembler = FeatureAssembler(etl_cfg, registry, observer)
        label_gen = LabelGenerator(etl_cfg, registry, observer)
        
        tensor_converter = TensorConverter(
            etl_config=etl_cfg,
            cnn_config=cnn_cfg,
            registry=registry,
            observer=observer
        )
        
        components: List[PipelineComponent] = [
            assembler,
            label_gen,
            tensor_converter
        ]
        
        return ETLService(registry, observer, components)

    def create_icd_service(self) -> ICDService:
        cfg = self.ctx.icd
        registry = self.create_registry()
        observer = self.create_telemetry()
        
        train_path = Path(cfg.data_path)
        source = ParquetSource(train_path)
        
        return ICDService(
            config=cfg,
            registry=registry,
            observer=observer,
            train_source=source,
            val_source=source,
            test_source=source
        )

    def create_cnn_service(self) -> TrainingService:
        cfg = self.ctx.cnn
        registry = self.create_registry()
        observer = self.create_telemetry()
        
        cache_dir = Path(cfg.data_cache_dir)
        train_path = cache_dir / "train.h5"
        val_path = cache_dir / "val.h5"
        
        try:
            train_source = HDF5StreamingSource(train_path, seq_len=cfg.seq_len)
            val_source = HDF5StreamingSource(val_path, seq_len=cfg.seq_len)
            vocab_sizes = train_source.get_real_vocab_sizes()
            
            if len(train_source) > 0:
                dummy = train_source[0]
                num_channels = dummy.x_num.shape[0]
                icd_dim = dummy.x_icd.shape[0] if dummy.x_icd is not None else 0
            else:
                num_channels = 0
                icd_dim = 0
        except Exception as e:
            observer.log("WARNING", f"CNN Factory: Could not initialize data sources ({e}). Using dummy dims.")
            train_source = None
            val_source = None
            vocab_sizes = []
            num_channels = 1
            icd_dim = 0

        network = GFINet_CNN(
            in_chs=[num_channels],
            n_cls=4,
            vocab_sizes=vocab_sizes,
            icd_dim=icd_dim,
            embed_dim=cfg.embed_dim,
            cat_embed_dim=cfg.cat_embed_dim,
            head_drop=cfg.dropout,
            tcn_drop=cfg.tcn_dropout,
            kernel=cfg.tcn_kernel_size,
            layers=cfg.tcn_layers
        )
        
        entity = InterventionModelEntity(network)
        
        return TrainingService(
            config=cfg,
            registry=registry,
            observer=observer,
            entity=entity,
            device="cuda" if torch.cuda.is_available() else "cpu",
            train_dataset=train_source,
            val_dataset=val_source
        )
