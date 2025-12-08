from pydantic import ValidationError
from loguru import logger
from hint.domain.vo import ETLConfig, CNNConfig
from ...utils.custom_assertions import assert_raises
from ..conftest import UnitFixtures

def test_etl_config_immutability() -> None:
    """
    Validates that ETLConfig enforces immutability.
    
    Test Case ID: VO-01
    Description: Attempts to modify a frozen field in ETLConfig and verifies that a ValidationError is raised.
    """
    logger.info("Starting test: test_etl_config_immutability")
    
    cfg = UnitFixtures.get_minimal_etl_config()
    
    logger.debug("Attempting to modify frozen field 'min_age'")
    with assert_raises(ValidationError):
        cfg.min_age = 100
        
    logger.info("Successfully verified that ETLConfig is immutable.")

def test_cnn_config_defaults() -> None:
    """
    Validates that CNNConfig initializes with correct default values.
    
    Test Case ID: VO-02
    Description: Checks default values for batch_size, learning rate, and excluded columns.
    """
    logger.info("Starting test: test_cnn_config_defaults")
    
    cfg = CNNConfig(data_path="dummy", data_cache_dir="dummy")
    
    logger.debug(f"Checking batch_size: {cfg.batch_size} (Expected: 512)")
    assert cfg.batch_size == 512, f"Expected batch_size 512, got {cfg.batch_size}"
    
    logger.debug(f"Checking learning rate: {cfg.lr} (Expected: 0.001)")
    assert cfg.lr == 0.001, f"Expected lr 0.001, got {cfg.lr}"
    
    logger.debug(f"Checking excluded columns: {cfg.exclude_cols}")
    assert cfg.exclude_cols == ["ICD9_CODES"], f"Expected ['ICD9_CODES'], got {cfg.exclude_cols}"
    
    logger.info("CNNConfig defaults verified successfully.")

def test_etl_config_exact_level2_list() -> None:
    """
    Validates that ETLConfig populates the exact_level2_104 list by default.
    
    Test Case ID: VO-03
    Description: Ensures that the list of vital signs is not empty and contains expected values.
    """
    logger.info("Starting test: test_etl_config_exact_level2_list")
    
    cfg = ETLConfig(raw_dir=".", proc_dir=".", resources_dir=".")
    
    logger.debug(f"Checking exact_level2_104 list length: {len(cfg.exact_level2_104)}")
    assert len(cfg.exact_level2_104) > 10, "exact_level2_104 list is too short."
    
    logger.debug("Verifying presence of 'heart rate' in the list")
    assert "heart rate" in cfg.exact_level2_104, "'heart rate' missing from exact_level2_104"
    
    logger.info("ETLConfig exact_level2_104 list verified successfully.")