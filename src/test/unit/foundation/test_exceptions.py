from loguru import logger
from hint.foundation.exceptions import (
    HintException, 
    ConfigurationError, 
    DataValidationError, 
    ModelError
)
from ...utils.custom_assertions import assert_raises

def test_custom_exceptions_inheritance() -> None:
    """
    Validates that custom exceptions inherit from the base HintException.
    
    Test Case ID: FND-EXC-01
    Description:
        Checks inheritance relationships for ConfigurationError, DataValidationError, and ModelError.
    """
    logger.info("Starting test: test_custom_exceptions_inheritance")

    assert issubclass(ConfigurationError, HintException)
    assert issubclass(DataValidationError, HintException)
    assert issubclass(ModelError, HintException)

    logger.info("Exception inheritance verified.")

def test_exception_raising() -> None:
    """
    Validates that custom exceptions can be raised and caught correctly.
    
    Test Case ID: FND-EXC-02
    Description:
        Raises a DataValidationError and verifies it is caught as a HintException.
    """
    logger.info("Starting test: test_exception_raising")

    def raiser():
        raise DataValidationError("Invalid data")

    with assert_raises(HintException):
        raiser()

    logger.info("Exception raising verified.")