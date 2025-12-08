from omegaconf import OmegaConf
from loguru import logger
from hint.foundation.configs import load_app_context
from hint.foundation.dtos import AppContext

def test_load_app_context_valid_yaml() -> None:
    """
    Verify AppContext creation from a populated Hydra configuration.

    This test validates `load_app_context` when it receives a complete OmegaConf mapping for ETL, ICD, and CNN settings, ensuring the resulting `AppContext` carries the expected values.
    - Test Case ID: FND-01
    - Scenario: Build AppContext from complete Hydra configuration.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_load_app_context_valid_yaml")
    
    cfg = OmegaConf.create({
        "data": {
            "raw_dir": "./raw",
            "proc_dir": "./proc"
        },
        "icd": {
            "data_path": "./icd_data"
        },
        "cnn": {
            "data": {"path": "./cnn_data"},
            "model": {"epochs": 50}
        },
        "mode": "train",
        "seed": 1234
    })
    
    logger.debug("Loading AppContext from OmegaConf")
    ctx = load_app_context(cfg)
    
    logger.debug("Verifying context fields")
    assert isinstance(ctx, AppContext), "Result should be an instance of AppContext"
    assert ctx.etl.raw_dir == "./raw", f"Expected raw_dir './raw', got {ctx.etl.raw_dir}"
    assert ctx.icd.data_path == "./icd_data", f"Expected icd.data_path './icd_data', got {ctx.icd.data_path}"
    assert ctx.cnn.epochs == 50, f"Expected cnn.epochs 50, got {ctx.cnn.epochs}"
    assert ctx.seed == 1234, f"Expected seed 1234, got {ctx.seed}"
    
    logger.info("AppContext loading verified.")
