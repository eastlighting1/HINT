"""Summary of the factory module.

Longer description of the module purpose and usage.
"""

from pathlib import Path

from typing import List



from ..foundation.configs import HydraConfigLoader, load_app_context

from ..foundation.interfaces import Registry, TelemetryObserver, PipelineComponent

from ..domain.vo import ETLConfig, ICDConfig, CNNConfig

from ..domain.entities import InterventionModelEntity, ICDModelEntity

from ..infrastructure.registry import FileSystemRegistry

from ..infrastructure.telemetry import RichTelemetryObserver

from ..infrastructure.datasource import HDF5StreamingSource, ParquetSource

from ..infrastructure.networks import TCNClassifier















from ..services.etl.components.static import StaticExtractor

from ..services.etl.components.timeseries import TimeSeriesAggregator

from ..services.etl.components.outcomes import OutcomesBuilder

from ..services.etl.components.ventilation import VentilationTagger

from ..services.etl.components.assembler import FeatureAssembler

from ..services.etl.components.tensor import TensorConverter

from ..services.etl.components.labels import LabelGenerator



class AppFactory:

    """Summary of AppFactory purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    ctx (Any): Description of ctx.
    loader (Any): Description of loader.
    raw_cfg (Any): Description of raw_cfg.
    """

    def __init__(self, config_name: str = "config", config_path: str = "configs"):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
        config_name (Any): Description of config_name.
        config_path (Any): Description of config_path.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        self.loader = HydraConfigLoader(config_name=config_name, config_path=config_path)

        self.raw_cfg = self.loader.load()

        self.ctx = load_app_context(self.raw_cfg)



    def create_registry(self) -> Registry:

        """Summary of create_registry.
        
        Longer description of the create_registry behavior and usage.
        
        Args:
        None (None): This function does not accept arguments.
        
        Returns:
        Registry: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        path = self.raw_cfg.get("logging", {}).get("artifacts_dir", "artifacts")

        return FileSystemRegistry(base_dir=path)



    def create_telemetry(self) -> TelemetryObserver:

        """Summary of create_telemetry.
        
        Longer description of the create_telemetry behavior and usage.
        
        Args:
        None (None): This function does not accept arguments.
        
        Returns:
        TelemetryObserver: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        return RichTelemetryObserver()



    def create_etl_service(self):

        """Summary of create_etl_service.
        
        Longer description of the create_etl_service behavior and usage.
        
        Args:
        None (None): This function does not accept arguments.
        
        Returns:
        Any: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """



        from ..services.etl.service import ETLService



        etl_cfg = self.ctx.etl

        cnn_cfg = self.ctx.cnn

        icd_cfg = self.ctx.icd

        registry = self.create_registry()

        observer = self.create_telemetry()



        observer.log("INFO", "AppFactory: ETL stage 1/4 starting component initialization.")
        observer.log("INFO", f"AppFactory: ETL config loaded dataset_root={etl_cfg.raw_dir}.")

        static_extractor = StaticExtractor(etl_cfg, registry, observer)

        ts_aggregator = TimeSeriesAggregator(etl_cfg, registry, observer)

        outcomes_builder = OutcomesBuilder(etl_cfg, cnn_cfg, registry, observer)

        vent_tagger = VentilationTagger(etl_cfg, registry, observer)



        assembler = FeatureAssembler(etl_cfg, cnn_cfg, registry, observer)

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



        observer.log("INFO", "AppFactory: ETL stage 2/4 assembling pipeline order.")

        components: List[PipelineComponent] = [

            static_extractor,

            ts_aggregator,

            outcomes_builder,

            vent_tagger,

            assembler,

            label_gen,

            tensor_converter

        ]



        observer.log("INFO", f"AppFactory: ETL stage 3/4 pipeline components count={len(components)}.")
        observer.log("INFO", "AppFactory: ETL stage 4/4 service construction complete.")

        return ETLService(registry, observer, components)



    def create_icd_service(self):

        """Summary of create_icd_service.
        
        Longer description of the create_icd_service behavior and usage.
        
        Args:
        None (None): This function does not accept arguments.
        
        Returns:
        Any: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """



        from ..services.training.automatic_icd_coding.service import ICDService



        cfg = self.ctx.icd

        registry = self.create_registry()

        observer = self.create_telemetry()



        cache_dir = Path(cfg.data.data_cache_dir)
        prefix = cfg.data.input_h5_prefix

        observer.log("INFO", "AppFactory: ICD stage 1/3 resolving training data sources.")
        observer.log("INFO", f"AppFactory: ICD cache_dir={cache_dir} prefix={prefix}.")



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



        observer.log("INFO", "AppFactory: ICD stage 2/3 service dependencies ready.")
        observer.log("INFO", "AppFactory: ICD stage 3/3 service construction complete.")

        return ICDService(

            config=cfg,

            registry=registry,

            observer=observer,

            train_source=train_source,

            val_source=val_source,

            test_source=test_source

        )



    def create_intervention_service(self):

        """Summary of create_intervention_service.
        
        Longer description of the create_intervention_service behavior and usage.
        
        Args:
        None (None): This function does not accept arguments.
        
        Returns:
        Any: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """



        from ..services.training.predict_intervention.service import InterventionService



        cfg = self.ctx.cnn

        registry = self.create_registry()

        observer = self.create_telemetry()



        cache_dir = Path(cfg.data.data_cache_dir)
        prefix = cfg.data.input_h5_prefix

        observer.log("INFO", "AppFactory: Intervention stage 1/4 resolving training data sources.")
        observer.log("INFO", f"AppFactory: Intervention cache_dir={cache_dir} prefix={prefix}.")

        train_path = cache_dir / f"{prefix}_train.h5"

        val_path = cache_dir / f"{prefix}_val.h5"

        test_path = cache_dir / f"{prefix}_test.h5"



        target_key = cfg.keys.TARGET_VENT_STATE if hasattr(cfg.keys, "TARGET_VENT_STATE") else "y_vent"



        try:

            train_source = HDF5StreamingSource(train_path, seq_len=cfg.seq_len, label_key=target_key)

            val_source = HDF5StreamingSource(val_path, seq_len=cfg.seq_len, label_key=target_key)

            test_source = HDF5StreamingSource(test_path, seq_len=cfg.seq_len, label_key=target_key) if test_path.exists() else None



            vocab_sizes = train_source.get_real_vocab_sizes()



            if len(train_source) > 0:

                dummy = train_source[0]



                if hasattr(dummy, 'x_num') and dummy.x_num is not None:

                    num_channels = dummy.x_num.shape[0]

                elif hasattr(dummy, 'x_val'):

                    num_channels = dummy.x_val.shape[0] * 3

                else:

                    num_channels = 0



                icd_dim = dummy.x_icd.shape[0] if dummy.x_icd is not None else 0

            else:

                num_channels = 0

                icd_dim = 0

        except Exception as e:

            observer.log("WARNING", f"CNN Factory: Could not initialize data sources ({e}). Using dummy dims.")

            train_source = None

            val_source = None

            test_source = None

            vocab_sizes = []

            num_channels = 1

            icd_dim = 0



        observer.log("INFO", "AppFactory: Intervention stage 2/4 building model network.")



        network = TCNClassifier(

            in_chs=num_channels,

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



        observer.log("INFO", f"AppFactory: Intervention stage 3/4 model in_chs={num_channels} icd_dim={icd_dim}.")
        observer.log("INFO", "AppFactory: Intervention stage 4/4 service construction complete.")

        return InterventionService(

            config=cfg,

            registry=registry,

            observer=observer,

            entity=entity,

            train_dataset=train_source,

            val_dataset=val_source,

            test_dataset=test_source

        )
