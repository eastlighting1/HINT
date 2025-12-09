from loguru import logger
from src.hint.foundation.exceptions import (
    ConfigurationError, 
    DataValidationError, 
    ModelError
)
from src.test.utils.custom_assertions import assert_raises

def test_custom_exceptions_inheritance() -> None:
    """
    Validates that custom exceptions inherit from the base Exception class.
    
    Test Case ID: FND-EXC-01
    """
    logger.info("Starting test: test_custom_exceptions_inheritance")

    assert issubclass(ConfigurationError, Exception)
    assert issubclass(DataValidationError, Exception)
    assert issubclass(ModelError, Exception)

    logger.info("Exception inheritance verified.")

def test_exception_raising() -> None:
    """
    Validates that custom exceptions can be raised and caught correctly.
    
    Test Case ID: FND-EXC-02
    """
    logger.info("Starting test: test_exception_raising")

    def raiser():
        raise DataValidationError("Invalid data")

    with assert_raises(DataValidationError):
        raiser()

    logger.info("Exception raising verified.")