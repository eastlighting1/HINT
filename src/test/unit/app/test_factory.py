from omegaconf import OmegaConf
from unittest.mock import MagicMock, patch
from loguru import logger
from hint.app.factory import AppFactory
from hint.services.etl.service import ETLService
from hint.services.icd.service import ICDService
from hint.services.training.trainer import TrainingService
from hint.services.training.evaluator import EvaluationService

def test_factory_initialization() -> None:
    """
    Verify AppFactory initializes with a valid Hydra configuration.

    This test validates that `AppFactory` builds its context, registry, and telemetry observer when provided with minimal Hydra-style settings for data, ICD, and CNN components.
    - Test Case ID: APP-01
    - Scenario: Construct AppFactory from a synthetic configuration.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_factory_initialization")

    cfg = OmegaConf.create({
        "data": {"raw_dir": "raw", "proc_dir": "proc"},
        "icd": {"data_path": "icd"},
        "cnn": {"data": {"path": "cnn", "data_cache_dir": "cache"}, "model": {}},
        "logging": {"artifacts_dir": "artifacts"}
    })

    factory = AppFactory(cfg)

    logger.debug("Verifying factory attributes")
    assert factory.ctx is not None
    assert factory.registry is not None
    assert factory.observer is not None

    logger.info("Factory initialization verified.")

def test_create_icd_service() -> None:
    """
    Confirm AppFactory produces an ICDService instance.

    This test validates that the factory wires dependencies and returns `ICDService` when backing sources are patched, ensuring correct service construction without file I/O.
    - Test Case ID: APP-02
    - Scenario: Build an ICD service using mocked ParquetSource dependencies.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_create_icd_service")

    cfg = OmegaConf.create({
        "data": {"raw_dir": "raw"},
        "icd": {"data_path": "dummy_path.parquet"},
        "cnn": {"data": {"path": "cnn"}, "model": {}},
        "logging": {"artifacts_dir": "artifacts"}
    })

    factory = AppFactory(cfg)

    with patch("hint.app.factory.ParquetSource") as MockSource:
        service = factory.create_icd_service()
        
        logger.debug("Verifying ICDService creation")
        assert isinstance(service, ICDService)
        assert service.registry == factory.registry
        
    logger.info("ICDService creation verified.")

def test_create_cnn_services() -> None:
    """
    Verify AppFactory constructs paired Training and Evaluation services.

    This test validates that patched data sources allow `create_cnn_services` to return both trainer and evaluator bound to the same entity, confirming wiring without real file access.
    - Test Case ID: APP-03
    - Scenario: Instantiate CNN training and evaluation services using mocked storage and metadata.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_create_cnn_services")

    cfg = OmegaConf.create({
        "data": {"raw_dir": "raw"},
        "icd": {"data_path": "icd"},
        "cnn": {
            "data": {"path": "cnn", "data_cache_dir": "cache"},
            "model": {"embed_dim": 128}
        },
        "logging": {"artifacts_dir": "artifacts"}
    })

    factory = AppFactory(cfg)

    with patch("hint.app.factory.HDF5StreamingSource") as MockH5, \
         patch("builtins.open", new_callable=MagicMock), \
         patch("json.load", return_value={"n_feats_numeric": 10, "vocab_info": {}}):
        
        trainer, evaluator = factory.create_cnn_services()
        
        logger.debug("Verifying CNN services creation")
        assert isinstance(trainer, TrainingService)
        assert isinstance(evaluator, EvaluationService)
        assert trainer.entity is evaluator.entity
        
    logger.info("CNN services creation verified.")
