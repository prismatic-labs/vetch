"""SDK compatibility and version detection.

This module handles:
- Detection of installed SDK versions
- Compatibility checks against tested versions
- Python version validation
"""

import sys
from typing import NamedTuple, Union

# Minimum Python version
MIN_PYTHON_VERSION = (3, 9)

# Tested SDK versions (we warn if not in this range)
TESTED_OPENAI_VERSIONS = ("1.0.0", "3.0.0")  # [min, max)
TESTED_VERTEXAI_VERSIONS = ("1.0.0", "3.0.0")  # [min, max)


class VersionInfo(NamedTuple):
    """SDK version information."""

    name: str
    version: Union[str, None]
    installed: bool
    tested: bool


def check_python_version() -> bool:
    """Check if Python version meets minimum requirement.

    Returns:
        True if Python >= 3.9, False otherwise.
    """
    return sys.version_info >= MIN_PYTHON_VERSION


def get_python_version() -> str:
    """Get current Python version string."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def parse_version(version_str: str) -> tuple[int, ...]:
    """Parse version string into comparable tuple.

    Handles versions like "1.0.0", "1.2.3b1", "2.0.0rc1".
    Returns at least a 3-tuple (major, minor, patch).
    """
    # Strip any pre-release suffix
    clean = version_str.split("a")[0].split("b")[0].split("rc")[0].split("-")[0]
    parts = clean.split(".")
    result: list[int] = []
    for part in parts:
        try:
            result.append(int(part))
        except ValueError:
            break

    # Pad to at least 3 parts
    while len(result) < 3:
        result.append(0)

    return tuple(result)


def version_in_range(version: str, min_version: str, max_version: str) -> bool:
    """Check if version is in [min_version, max_version) range."""
    v = parse_version(version)
    min_v = parse_version(min_version)
    max_v = parse_version(max_version)
    return min_v <= v < max_v


def get_openai_version() -> VersionInfo:
    """Detect OpenAI SDK version.

    Returns:
        VersionInfo with version details and tested status.
    """
    try:
        import openai  # type: ignore[import-not-found]

        version = getattr(openai, "__version__", None)
        if version is None:
            return VersionInfo("openai", None, True, False)

        tested = version_in_range(version, *TESTED_OPENAI_VERSIONS)
        return VersionInfo("openai", version, True, tested)
    except ImportError:
        return VersionInfo("openai", None, False, False)


def get_vertexai_version() -> VersionInfo:
    """Detect Vertex AI SDK version.

    Returns:
        VersionInfo with version details and tested status.
    """
    try:
        import google.cloud.aiplatform as aiplatform

        version = getattr(aiplatform, "__version__", None)
        if version is None:
            return VersionInfo("vertexai", None, True, False)

        tested = version_in_range(version, *TESTED_VERTEXAI_VERSIONS)
        return VersionInfo("vertexai", version, True, tested)
    except ImportError:
        return VersionInfo("vertexai", None, False, False)


def get_all_sdk_versions() -> dict[str, VersionInfo]:
    """Get version info for all supported SDKs.

    Returns:
        Dict mapping SDK name to VersionInfo.
    """
    return {
        "openai": get_openai_version(),
        "vertexai": get_vertexai_version(),
    }


def is_datadog_patched(func: object) -> bool:
    """Check if function is patched by Datadog.

    Datadog's ddtrace marks patched functions with _datadog_patch.
    Note: __wrapped__ is too generic (used by functools.wraps).
    """
    return hasattr(func, "_datadog_patch")


def is_opentelemetry_patched(func: object) -> bool:
    """Check if function is patched by OpenTelemetry.

    OpenTelemetry instrumentations typically wrap functions.
    """
    return hasattr(func, "_otel_patched") or hasattr(func, "__otel_original__")


def is_sentry_patched(func: object) -> bool:
    """Check if function is patched by Sentry.

    Sentry SDK wraps functions for tracing.
    """
    return hasattr(func, "_sentry_wrapped")


def detect_existing_patches(func: object) -> list[str]:
    """Detect any existing observability patches on a function.

    Returns:
        List of detected patch sources (e.g., ["datadog", "opentelemetry"]).
    """
    patches: list[str] = []

    if is_datadog_patched(func):
        patches.append("datadog")
    if is_opentelemetry_patched(func):
        patches.append("opentelemetry")
    if is_sentry_patched(func):
        patches.append("sentry")

    # Generic __wrapped__ detection
    if hasattr(func, "__wrapped__") and not patches:
        patches.append("unknown")

    return patches
