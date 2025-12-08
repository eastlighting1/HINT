from pydantic import ValidationError
from loguru import logger
from hint.domain.vo import ETLConfig, CNNConfig
from ...utils.custom_assertions import assert_raises
from ..conftest import UnitFixtures

def test_etl_config_immutability() -> None:
    """
    Verify ETLConfig rejects mutation of frozen fields.

    This test validates that attempting to reassign immutable attributes on `ETLConfig` raises `ValidationError`, preserving configuration integrity.
    - Test Case ID: VO-01
    - Scenario: Mutate a frozen ETL configuration attribute after creation.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_etl_config_immutability")
    
    cfg = UnitFixtures.get_minimal_etl_config()
    
    logger.debug("Attempting to modify frozen field 'min_age'")
    with assert_raises(ValidationError):
        cfg.min_age = 100
        
    logger.info("Successfully verified that ETLConfig is immutable.")

def test_cnn_config_defaults() -> None:
    """
    Confirm CNNConfig initializes with expected default hyperparameters.

    This test validates default values such as batch size, learning rate, and excluded columns to guarantee predictable initialization.
    - Test Case ID: VO-02
    - Scenario: Instantiate CNNConfig with minimal paths and inspect defaults.

    Args:
        None

    Returns:
        None
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
    Ensure ETLConfig populates the default exact_level2_104 list.

    This test validates that `ETLConfig` seeds the vital sign list with expected entries such as `heart rate`, indicating required schema coverage.
    - Test Case ID: VO-03
    - Scenario: Construct ETLConfig with minimal paths and inspect default vital sign list.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_etl_config_exact_level2_list")
    
    cfg = ETLConfig(raw_dir=".", proc_dir=".", resources_dir=".")
    
    logger.debug(f"Checking exact_level2_104 list length: {len(cfg.exact_level2_104)}")
    assert len(cfg.exact_level2_104) > 10, "exact_level2_104 list is too short."
    
    logger.debug("Verifying presence of 'heart rate' in the list")
    assert "heart rate" in cfg.exact_level2_104, "'heart rate' missing from exact_level2_104"
    
    logger.info("ETLConfig exact_level2_104 list verified successfully.")
