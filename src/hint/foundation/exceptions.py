class HINTError(Exception):
    """Base exception for the HINT application.

    Serves as the common root for domain-specific errors raised across
    ETL, training, and inference workflows.
    """
    pass

class ConfigurationError(HINTError):
    """Raised when configuration is invalid or missing.

    Indicates that required settings are absent or malformed.
    """
    pass

class PipelineError(HINTError):
    """Raised when a pipeline component fails.

    Wraps failures in ETL or training stages to simplify error handling.
    """
    pass

class DataError(HINTError):
    """Raised when data validation fails or files are missing.

    Signals missing inputs or inconsistent data shape/format issues.
    """
    pass

class ModelError(HINTError):
    """Raised during model training or inference failures.

    Covers runtime errors encountered during forward/backward passes.
    """
    pass
