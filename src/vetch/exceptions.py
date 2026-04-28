"""Vetch exception hierarchy.

Two parallel families:

1. ``VetchError`` (ValueError) — validation / configuration / data errors.
   Things the user did wrong, or that Vetch detected as invalid.

2. ``VetchInterrupt`` (RuntimeError) — system-level interventions where Vetch
   is taking control to prevent waste, runaway costs, or budget overruns.
   These are NOT validation errors and must propagate past
   ``except ValueError:`` handlers.

Vetch-specific exceptions allow:
1. Distinguishing Vetch errors from application errors
2. Avoiding accidental swallowing of KeyboardInterrupt/MemoryError
3. Providing actionable error context
"""

from __future__ import annotations


class VetchError(ValueError):
    """Base exception for Vetch validation / configuration / data errors.

    Inherits from ValueError for backwards compatibility.
    Use this for errors caused by invalid user input, missing config,
    bad data, etc. NOT for system-level interventions — see VetchInterrupt.
    """

    pass


class VetchInterrupt(RuntimeError):
    """Base for Vetch system-level interventions (circuit breaker, budget kill, etc.).

    Distinct from ``VetchError`` — these are not validation errors. They
    signal that Vetch is taking control to prevent waste or runaway cost.
    Inherits from ``RuntimeError`` (not ``ValueError``) so a generic
    ``except ValueError:`` handler in user code will not swallow them.

    Catch ``VetchInterrupt`` to handle any Vetch intervention generically,
    or catch the specific subclass (``StallDetected``) for fine-grained control.
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


class StallDetected(VetchInterrupt):
    """Raised when STALL-001 fires and ``stall_action="kill"`` is configured.

    Indicates an agentic loop is wasting calls (low-output + high input
    similarity) and Vetch is stopping the loop to prevent further spend.

    Attributes:
        wasted_cost_usd: Estimated cost of the stalled calls so far.
        request_count: Number of calls in the stalled window.
        fallback_model: Suggested fallback model from configuration, or None.
    """

    def __init__(
        self,
        message: str,
        wasted_cost_usd: float = 0.0,
        request_count: int = 0,
        fallback_model: str | None = None,
    ) -> None:
        super().__init__(message)
        self.wasted_cost_usd = wasted_cost_usd
        self.request_count = request_count
        self.fallback_model = fallback_model
