from pathlib import Path
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
from ..services.etl.components.static import StaticExtractor
from ..services.etl.components.timeseries import TimeSeriesAggregator
from ..services.etl.components.outcomes import OutcomesBuilder
from ..services.etl.components.ventilation import VentilationTagger
from ..services.etl.components.assembler import FeatureAssembler
from ..services.etl.components.tensor import TensorConverter
from ..services.etl.components.labels import LabelGenerator
from ..services.training.automatic_icd_coding.service import ICDService
from ..services.training.predict_intervention.service import InterventionService

class AppFactory:
    """Factory to assemble application services and components.

    Initializes configuration, telemetry, registry, and downstream services
    using dependency injection.
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
        icd_cfg = self.ctx.icd
        registry = self.create_registry()
        observer = self.create_telemetry()

        observer.log("INFO", "AppFactory: Building ETL pipeline components")
        static_extractor = StaticExtractor(etl_cfg, registry, observer)
        ts_aggregator = TimeSeriesAggregator(etl_cfg, registry, observer)
        outcomes_builder = OutcomesBuilder(etl_cfg, registry, observer)
        vent_tagger = VentilationTagger(etl_cfg, registry, observer)

        assembler = FeatureAssembler(etl_cfg, registry, observer)
        label_gen = LabelGenerator(
            etl_config=etl_cfg,
            icd_config=icd_cfg,
            cnn_config=cnn_cfg,
            registry=registry,
            observer=observer
        )
        
        tensor_converter = TensorConverter(
            etl_config=etl_cfg,
            cnn_config=cnn_cfg,
            icd_config=icd_cfg,
            registry=registry,
            observer=observer
        )

        observer.log("INFO", "AppFactory: Ordering ETL pipeline stages")
        components: List[PipelineComponent] = [
            static_extractor,
            ts_aggregator,
            outcomes_builder,
            vent_tagger,
            assembler,
            label_gen,
            tensor_converter
        ]

        observer.log("INFO", "AppFactory: ETL service ready")
        return ETLService(registry, observer, components)

    def create_icd_service(self) -> ICDService:
        cfg = self.ctx.icd
        registry = self.create_registry()
        observer = self.create_telemetry()

        observer.log("INFO", "AppFactory: Preparing ICD training data sources")
        cache_dir = Path(cfg.data.data_cache_dir)
        prefix = cfg.data.input_h5_prefix
        
        train_path = cache_dir / f"{prefix}_train.h5"
        val_path = cache_dir / f"{prefix}_val.h5"
        test_path = cache_dir / f"{prefix}_test.h5"
        
        try:
            train_source = HDF5StreamingSource(train_path, label_key="y")
            val_source = HDF5StreamingSource(val_path, label_key="y")
            test_source = HDF5StreamingSource(test_path, label_key="y") if test_path.exists() else None
        except Exception as e:
            observer.log("WARNING", f"Factory: Could not initialize ICD H5 sources ({e}). Ensure ETL has run.")
            train_source = None
            val_source = None
            test_source = None

        observer.log("INFO", "AppFactory: ICD service ready")
        return ICDService(
            config=cfg,
            registry=registry,
            observer=observer,
            train_source=train_source,
            val_source=val_source,
            test_source=test_source
        )

    def create_intervention_service(self) -> InterventionService:
        cfg = self.ctx.cnn
        registry = self.create_registry()
        observer = self.create_telemetry()

        observer.log("INFO", "AppFactory: Preparing intervention training data sources")
        cache_dir = Path(cfg.data.data_cache_dir)
        prefix = cfg.data.input_h5_prefix 
        train_path = cache_dir / f"{prefix}_train.h5"
        val_path = cache_dir / f"{prefix}_val.h5"
        
        # [FIX] Explicitly use 'y_vent' (or configured key) as label_key
        # Defaulting to 'y' causes it to read ICD targets which leads to CUDA errors
        target_key = cfg.keys.TARGET_VENT_STATE if hasattr(cfg.keys, "TARGET_VENT_STATE") else "y_vent"

        try:
            train_source = HDF5StreamingSource(train_path, seq_len=cfg.seq_len, label_key=target_key)
            val_source = HDF5StreamingSource(val_path, seq_len=cfg.seq_len, label_key=target_key)
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

        observer.log("INFO", "AppFactory: Building intervention model network")
        
        # Note: n_cls=4 is hardcoded here. If binary task (0/1), consider changing to 2.
        # But keeping as 4 to match FocalLoss default and potentially multi-state logic.
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

        observer.log("INFO", "AppFactory: Intervention service ready")
        return InterventionService(
            config=cfg,
            registry=registry,
            observer=observer,
            entity=entity,
            train_dataset=train_source,
            val_dataset=val_source
        )