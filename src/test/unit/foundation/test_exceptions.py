from loguru import logger
from src.hint.foundation.exceptions import (
    ConfigurationError, 
    DataError, 
    ModelError
)
from src.test.utils.custom_assertions import assert_raises

def test_custom_exceptions_inheritance() -> None:
    """
    [One-line Summary] Validate that custom exceptions inherit from the base Exception class.

    [Description]
    Ensure ConfigurationError, DataError, and ModelError are subclasses of Exception so they
    integrate with standard error handling semantics.

    Test Case ID: FND-EXC-01
    Scenario: Inspect custom exception inheritance relationships.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_custom_exceptions_inheritance")

    assert issubclass(ConfigurationError, Exception)
    assert issubclass(DataError, Exception)
    assert issubclass(ModelError, Exception)

    logger.info("Exception inheritance verified.")

def test_exception_raising() -> None:
    """
    [One-line Summary] Validate that custom exceptions can be raised and caught correctly.

    [Description]
    Raise a DataError via a helper function and assert it is caught by the assertion helper,
    proving custom exceptions follow expected control flow.

    Test Case ID: FND-EXC-02
    Scenario: Trigger a DataError and assert it is captured by the testing helper.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_exception_raising")

    def raiser():
        raise DataError("Invalid data")

    with assert_raises(DataError):
        raiser()

    logger.info("Exception raising verified.")
