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
    Validates that AppFactory initializes correctly with a valid Hydra configuration.
    
    Test Case ID: APP-01
    Description:
        Passes a dummy OmegaConf configuration.
        Verifies that the context, registry, and observer are instantiated.
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
    Validates the creation of the ICDService via the factory.
    
    Test Case ID: APP-02
    Description:
        Mocks the ParquetSource to avoid file I/O.
        Calls create_icd_service and checks the returned instance type.
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
    Validates the creation of TrainingService and EvaluationService.
    
    Test Case ID: APP-03
    Description:
        Mocks HDF5StreamingSource and file operations.
        Verifies that both trainer and evaluator are returned and linked correctly.
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