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
    Verify custom exceptions inherit from the base HintException.

    This test validates the inheritance chain for `ConfigurationError`, `DataValidationError`, and `ModelError`, ensuring each derives from `HintException` to support unified error handling.
    - Test Case ID: FND-EXC-01
    - Scenario: Inspect inheritance relationships for custom exceptions.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_custom_exceptions_inheritance")

    assert issubclass(ConfigurationError, HintException)
    assert issubclass(DataValidationError, HintException)
    assert issubclass(ModelError, HintException)

    logger.info("Exception inheritance verified.")

def test_exception_raising() -> None:
    """
    Validate raising and catching custom exceptions.

    This test ensures that `DataValidationError` can be raised and is caught as a `HintException`, confirming compatibility with shared exception handling paths.
    - Test Case ID: FND-EXC-02
    - Scenario: Raise and capture a data validation error via the HintException hierarchy.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_exception_raising")

    def raiser():
        raise DataValidationError("Invalid data")

    with assert_raises(HintException):
        raiser()

    logger.info("Exception raising verified.")
