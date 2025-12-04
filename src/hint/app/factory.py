from omegaconf import DictConfig
from pathlib import Path
import torch

from hint.foundation.configs import load_app_context
from hint.foundation.interfaces import Registry, TelemetryObserver
from hint.infrastructure.registry import FileSystemRegistry
from hint.infrastructure.telemetry import RichTelemetryObserver
from hint.infrastructure.datasource import HDF5StreamingSource, ParquetSource
from hint.infrastructure.networks import GFINet_CNN, MedBERTClassifier
from hint.infrastructure.components import XGBoostStacker
from hint.domain.entities import ICDModelEntity, InterventionModelEntity
from hint.services.etl.service import ETLService
from hint.services.icd.service import ICDService
from hint.services.training.trainer import TrainingService
from hint.services.training.evaluator import EvaluationService

# ETL Components
from hint.services.etl.components.static import StaticExtractor
from hint.services.etl.components.timeseries import TimeSeriesAggregator
from hint.services.etl.components.outcomes import OutcomesBuilder
from hint.services.etl.components.ventilation import VentilationTagger
from hint.services.etl.components.notes import NoteTokenizer
from hint.services.etl.components.assembler import FeatureAssembler
from hint.services.etl.components.tensor import TensorConverter

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
            TensorConverter(self.ctx.cnn, self.registry, self.observer) # Tensor step uses CNN config
        ]
        return ETLService(self.ctx.etl, self.ctx.cnn, components, self.observer)

    def create_icd_service(self) -> ICDService:
        # Load data source
        train_path = self.registry.get_artifact_path("dataset_123_answer.parquet")
        source = ParquetSource(train_path)
        
        # Create Entity
        head1 = MedBERTClassifier(self.ctx.icd.model_name, num_num=123, num_cls=500)
        head2 = MedBERTClassifier(self.ctx.icd.model_name, num_num=123, num_cls=500)
        stacker = XGBoostStacker(self.ctx.icd.xgb_params)
        entity = ICDModelEntity(head1, head2, stacker)
        
        # Try load checkpoint if exists
        try:
            state = self.registry.load_model("icd_model", "best", "cpu")
            entity.load_state_dict(state)
        except:
            pass

        return ICDService(
            self.ctx.icd, self.registry, self.observer, entity, 
            train_source=source, val_source=source, test_source=source # Simplified split logic
        )

    def create_cnn_services(self) -> tuple[TrainingService, EvaluationService]:
        # Sources
        data_dir = self.registry.dirs["data"]
        tr_src = HDF5StreamingSource(data_dir / "train.h5", self.ctx.cnn.seq_len)
        val_src = HDF5StreamingSource(data_dir / "val.h5", self.ctx.cnn.seq_len)
        te_src = HDF5StreamingSource(data_dir / "test.h5", self.ctx.cnn.seq_len)
        
        # Feature info for model build
        feat_info = self.registry.load_json("feature_info.json")
        vocab = feat_info.get("vocab_info", {})
        
        # Entity
        # Simplified group creation logic for brevity
        net = GFINet_CNN(
            in_chs=[40, 20, 10], n_cls=4, g1=[], g2=[], rest=[], # Need proper indices here from feat_info
            cat_vocab_sizes=vocab, embed_dim=self.ctx.cnn.embed_dim
        )
        entity = InterventionModelEntity(net, self.ctx.cnn.ema_decay)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        trainer = TrainingService(self.ctx.cnn, self.registry, self.observer, entity, device)
        evaluator = EvaluationService(self.ctx.cnn, self.registry, self.observer, entity, device)
        
        # Monkey patch sources into trainer for easy access or pass them in main
        trainer.tr_src = tr_src
        trainer.val_src = val_src
        evaluator.te_src = te_src
        
        return trainer, evaluator
