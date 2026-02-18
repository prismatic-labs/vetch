"""Vetch exception hierarchy.

Vetch-specific exceptions allow:
1. Distinguishing Vetch errors from application errors
2. Avoiding accidental swallowing of KeyboardInterrupt/MemoryError
3. Providing actionable error context

All Vetch exceptions inherit from VetchError (which inherits from ValueError).
"""

from __future__ import annotations


class VetchError(ValueError):
    """Base exception for all Vetch errors.

    Inherits from ValueError for backwards compatibility.
    All Vetch-specific exceptions should inherit from this class.
    """

    pass


class RegistryError(VetchError):
    """Raised when registry lookup fails.

    Examples:
        - Model not found in registry
        - Corrupted registry file
        - Missing required registry fields
    """

    def __init__(self, message: str, model: str | None = None) -> None:
        super().__init__(message)
        self.model = model


class ProviderError(VetchError):
    """Raised when provider SDK interaction fails.

    Examples:
        - Failed to patch SDK
        - SDK version incompatible
        - Provider response parsing failed
    """

    def __init__(self, message: str, provider: str | None = None) -> None:
        super().__init__(message)
        self.provider = provider


class ConfigurationError(VetchError):
    """Raised when Vetch configuration is invalid.

    Examples:
        - Invalid energy_override values
        - Missing required configuration
        - Invalid environment variable format
    """

    def __init__(self, message: str, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


class CalibrationError(VetchError):
    """Raised when GPU calibration fails.

    Examples:
        - GPU not available
        - pynvml initialization failed
        - Invalid workload function
    """

    def __init__(self, message: str, gpu_error: str | None = None) -> None:
        super().__init__(message)
        self.gpu_error = gpu_error


class StorageError(VetchError):
    """Raised when storage operations fail.

    Examples:
        - Database connection failed
        - Query execution failed
        - Schema migration failed
    """

    def __init__(self, message: str, db_path: str | None = None) -> None:
        super().__init__(message)
        self.db_path = db_path
