"""Summary of the factory module.

Longer description of the module purpose and usage.
"""

from pathlib import Path
from datetime import datetime
import json
import platform
import sys

from typing import List



from ..foundation.configs import HydraConfigLoader, load_app_context
from omegaconf import OmegaConf

from ..foundation.interfaces import Registry, TelemetryObserver, PipelineComponent

from ..domain.vo import ETLConfig, ICDConfig, CNNConfig

from ..domain.entities import InterventionModelEntity, ICDModelEntity

from ..infrastructure.registry import FileSystemRegistry

from ..infrastructure.telemetry import RichTelemetryObserver

from ..infrastructure.datasource import HDF5StreamingSource, ParquetSource

from ..infrastructure.networks import TCNClassifier
import inspect















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

        self.run_dir = self._init_run_dir()

        self._write_initial_artifacts()


    def _init_run_dir(self) -> Path:

        """Summary of _init_run_dir.
        
        Longer description of the _init_run_dir behavior and usage.
        
        Args:
        None (None): This function does not accept arguments.
        
        Returns:
        Path: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        logging_cfg = self.raw_cfg.get("logging", {}) or {}

        output_root = Path(logging_cfg.get("output_dir", "outputs"))

        run_dir_cfg = logging_cfg.get("run_dir")
        use_hydra = bool(logging_cfg.get("use_hydra_run_dir", False))

        if use_hydra:
            try:
                from hydra.core.hydra_config import HydraConfig

                hydra_cfg = HydraConfig.get()
                run_dir = Path(hydra_cfg.runtime.output_dir)
            except Exception:
                run_dir = None
        else:
            run_dir = None

        if run_dir is None and run_dir_cfg:

            run_dir = Path(run_dir_cfg)

            if not run_dir.is_absolute():

                run_dir = output_root / run_dir

        elif run_dir is None:

            timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")

            run_dir = output_root / timestamp

        for subdir in ["logs", "metrics", "traces", "artifacts"]:

            (run_dir / subdir).mkdir(parents=True, exist_ok=True)

        return run_dir

    def _write_initial_artifacts(self) -> None:

        """Write config and system metadata into the run artifacts directory."""

        artifacts_dir = self.run_dir / "artifacts"

        artifacts_dir.mkdir(parents=True, exist_ok=True)

        config_path = artifacts_dir / "config_snapshot.yaml"

        config_path.write_text(OmegaConf.to_yaml(self.raw_cfg), encoding="utf-8")

        system_spec = {
            "python_version": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "processor": platform.processor(),
        }

        try:
            import torch

            system_spec["torch_version"] = torch.__version__
            system_spec["cuda_available"] = torch.cuda.is_available()
        except Exception:
            system_spec["torch_version"] = None
            system_spec["cuda_available"] = None

        try:
            import numpy as np

            system_spec["numpy_version"] = np.__version__
        except Exception:
            system_spec["numpy_version"] = None

        system_path = artifacts_dir / "system_spec.json"

        system_path.write_text(json.dumps(system_spec, indent=2), encoding="utf-8")



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

        return FileSystemRegistry(base_dir=self.run_dir / path)



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

        return RichTelemetryObserver(run_dir=self.run_dir)



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



        observer.log("INFO", "[1.0] ETL component initialization started.")
        observer.log("INFO", f"[1.0] ETL config loaded. dataset_root={etl_cfg.raw_dir}")

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



        observer.log("INFO", "[1.0] ETL pipeline order assembled.")

        components: List[PipelineComponent] = [

            static_extractor,

            ts_aggregator,

            outcomes_builder,

            vent_tagger,

            assembler,

            label_gen,

            tensor_converter

        ]



        observer.log("INFO", f"[1.0] ETL components count={len(components)}.")
        observer.log("INFO", "[1.0] ETL service construction complete.")

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

        observer.log("INFO", "[2.0] ICD training data resolution started.")
        observer.log("INFO", f"[2.0] ICD cache_dir={cache_dir} prefix={prefix}")



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



        observer.log("INFO", "[2.0] ICD service dependencies ready.")
        observer.log("INFO", "[2.0] ICD service construction complete.")

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

        observer.log("INFO", "[3.0] Intervention data resolution started.")
        observer.log("INFO", f"[3.0] Intervention cache_dir={cache_dir} prefix={prefix}")

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



        observer.log("INFO", "[3.0] Intervention model construction started.")



        net_kwargs = dict(
            in_chs=num_channels,
            n_cls=4,
            vocab_sizes=vocab_sizes,
            icd_dim=icd_dim,
            embed_dim=cfg.embed_dim,
            cat_embed_dim=cfg.cat_embed_dim,
            head_drop=cfg.dropout,
            tcn_drop=cfg.tcn_dropout,
            kernel=cfg.tcn_kernel_size,
            layers=cfg.tcn_layers,
        )
        if "use_icd_gating" in inspect.signature(TCNClassifier.__init__).parameters:
            net_kwargs["use_icd_gating"] = cfg.use_icd_gating

        network = TCNClassifier(**net_kwargs)



        entity = InterventionModelEntity(network)



        observer.log("INFO", f"[3.0] Intervention model dims in_chs={num_channels} icd_dim={icd_dim}.")
        observer.log("INFO", "[3.0] Intervention service construction complete.")

        return InterventionService(

            config=cfg,

            registry=registry,

            observer=observer,

            entity=entity,

            train_dataset=train_source,

            val_dataset=val_source,

            test_dataset=test_source

        )
