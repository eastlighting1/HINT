class HINTError(Exception):
    """Base exception for the HINT application.

    Use this type as the root for domain-specific errors across the
    pipeline to enable consistent exception handling.
    """
    pass

class ConfigurationError(HINTError):
    """Raised when configuration loading or validation fails.

    This error signals missing files, invalid settings, or unsupported
    configuration values.
    """
    pass

class PipelineError(HINTError):
    """Raised when a pipeline stage fails to execute.

    This error captures failures during ETL or training workflows.
    """
    pass

class DataError(HINTError):
    """Raised when data loading or processing fails.

    This error is used for missing data, schema mismatches, and invalid
    input values.
    """
    pass

class ModelError(HINTError):
    """Raised when model initialization or inference fails.

    This error signals invalid model configuration or runtime issues
    during forward passes.
    """
    pass
