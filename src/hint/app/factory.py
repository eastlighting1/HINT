from omegaconf import DictConfig
from pathlib import Path
import torch
import json

from ..foundation.configs import load_app_context
from ..foundation.interfaces import Registry, TelemetryObserver
from ..infrastructure.registry import FileSystemRegistry
from ..infrastructure.telemetry import RichTelemetryObserver
from ..infrastructure.datasource import HDF5StreamingSource, ParquetSource
from ..infrastructure.networks import GFINet_CNN, MedBERTClassifier
from ..infrastructure.components import XGBoostStacker
from ..domain.entities import ICDModelEntity, InterventionModelEntity
from ..services.etl.service import ETLService
from ..services.icd.service import ICDService
from ..services.training.trainer import TrainingService
from ..services.training.evaluator import EvaluationService

# ETL Components
from ..services.etl.components.static import StaticExtractor
from ..services.etl.components.timeseries import TimeSeriesAggregator
from ..services.etl.components.outcomes import OutcomesBuilder
from ..services.etl.components.ventilation import VentilationTagger
from ..services.etl.components.notes import NoteTokenizer
from ..services.etl.components.assembler import FeatureAssembler
from ..services.etl.components.labels import LabelGenerator
from ..services.etl.components.tensor import TensorConverter

class AppFactory:
    """
    Dependency Injection Factory for the HINT application.
    Constructs services, entities, and infrastructure based on configuration.
    """
    def __init__(self, hydra_cfg: DictConfig):
        self.ctx = load_app_context(hydra_cfg)
        self.registry = FileSystemRegistry(hydra_cfg.get("logging", {}).get("artifacts_dir", "artifacts"))
        self.observer = RichTelemetryObserver(Path(hydra_cfg.get("logging", {}).get("core_logs_dir", "logs")))

    def create_etl_service(self) -> ETLService:
        components = [
            StaticExtractor(self.ctx.etl, self.registry, self.observer),
            TimeSeriesAggregator(self.ctx.etl, self.registry, self.observer),
            OutcomesBuilder(self.ctx.etl, self.registry, self.observer),
            VentilationTagger(self.ctx.etl, self.registry, self.observer),
            NoteTokenizer(self.ctx.etl, self.registry, self.observer),
            FeatureAssembler(self.ctx.etl, self.registry, self.observer),
            LabelGenerator(self.ctx.etl, self.registry, self.observer),
            TensorConverter(self.ctx.cnn, self.registry, self.observer)
        ]
        return ETLService(self.ctx.etl, self.ctx.cnn, components, self.observer)

    def create_icd_service(self) -> ICDService:
        # Load data source
        # Use configured absolute/relative path instead of registry default
        # Use configured absolute/relative path instead of registry default
        train_path = Path(self.ctx.icd.data_path)
        source = ParquetSource(train_path)
        
        # [FIXED] Removed 'entity' argument completely.
        # The service will initialize the entity internally after inspecting data dimensions.
        # [FIX] Do NOT create entity here. Pass None.
        # The service will create it after inspecting data dimensions.
        entity = None 
        
        return ICDService(
            self.ctx.icd, self.registry, self.observer, 
            train_source=source, val_source=source, test_source=source
        )

    def create_cnn_services(self) -> tuple[TrainingService, EvaluationService]:
        # Sources
        # Use configured cache dir (data/cache)
        # Use configured cache dir (data/cache)
        data_dir = Path(self.ctx.cnn.data_cache_dir)
        
        tr_src = HDF5StreamingSource(data_dir / "train.h5", self.ctx.cnn.seq_len)
        val_src = HDF5StreamingSource(data_dir / "val.h5", self.ctx.cnn.seq_len)
        te_src = HDF5StreamingSource(data_dir / "test.h5", self.ctx.cnn.seq_len)
        
        # Feature info for model build
        # We need to know input dimensions to initialize GFINet correctly
        feature_info_path = data_dir / "feature_info.json"
        if not feature_info_path.exists():
             # Fallback if not running full pipeline or file missing
             vocab = {}
             n_num = 123 * 3 # Estimate
        else:
            with open(feature_info_path, 'r') as f:
                feat_info = json.load(f)
            vocab = feat_info.get("vocab_info", {})
            n_num = feat_info.get("n_feats_numeric", 0)
        
        # Entity
        # Dynamically set channel sizes to avoid shape mismatch errors.
        # Dynamically set channel sizes to avoid shape mismatch errors.
        # Assign all numeric features to the first group (g1) for robust initialization
        net = GFINet_CNN(
            in_chs=[n_num, 0, 0], 
            n_cls=4, 
            g1=list(range(n_num)), 
            g2=[], 
            rest=[], 
            cat_vocab_sizes=vocab, 
            embed_dim=self.ctx.cnn.embed_dim
        )
        entity = InterventionModelEntity(net, self.ctx.cnn.ema_decay)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        trainer = TrainingService(self.ctx.cnn, self.registry, self.observer, entity, device)
        evaluator = EvaluationService(self.ctx.cnn, self.registry, self.observer, entity, device)
        
        # Patch sources into services
        trainer.tr_src = tr_src
        trainer.val_src = val_src
        evaluator.te_src = te_src
        evaluator.val_src = val_src
        
        return trainer, evaluator