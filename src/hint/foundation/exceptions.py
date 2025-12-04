class HINTError(Exception):
    """Base exception for the HINT application."""
    pass

class ConfigurationError(HINTError):
    """Raised when configuration is invalid or missing."""
    pass

class PipelineError(HINTError):
    """Raised when a pipeline component fails."""
    pass

class DataError(HINTError):
    """Raised when data validation fails or files are missing."""
    pass

class ModelError(HINTError):
    """Raised during model training or inference failures."""
    pass