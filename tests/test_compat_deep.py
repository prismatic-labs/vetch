"""Deep tests for compatibility logic.

Verifies version parsing, dependency detection, and patch detection.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from vetch.compat import (
    check_python_version,
    detect_existing_patches,
    get_python_version,
    parse_version,
    version_in_range,
)


def test_parse_version():
    """Verify semantic version parsing."""
    assert parse_version("1.2.3") == (1, 2, 3)
    assert parse_version("2.0.0-alpha") == (2, 0, 0)
    assert parse_version("invalid") == (0, 0, 0)

def test_version_in_range():
    """Verify version range checks."""
    assert version_in_range("1.5.0", "1.0.0", "2.0.0") is True
    assert version_in_range("2.0.0", "1.0.0", "2.0.0") is False
    assert version_in_range("0.9.0", "1.0.0", "2.0.0") is False

def test_python_version():
    """Verify Python version detection."""
    assert check_python_version() is True
    assert isinstance(get_python_version(), str)

def test_detect_existing_patches_none():
    """Verify detection returns empty list when no patches exist."""
    def clean_func(): pass
    assert detect_existing_patches(clean_func) == []

def test_detect_existing_patches_multiple():
    """Verify detection of multiple observability patches."""
    def patched_func(): pass
    patched_func._datadog_patch = True
    patched_func._otel_patched = True

    patches = detect_existing_patches(patched_func)
    assert "datadog" in patches
    assert "opentelemetry" in patches

def test_get_sdk_versions_not_installed():
    """Verify version detection when SDKs are not installed."""
    import sys

    from vetch.compat import get_all_sdk_versions, get_openai_version, get_vertexai_version

    with patch.dict(sys.modules, {"openai": None, "google.cloud.aiplatform": None}):
        info = get_openai_version()
        assert info.installed is False

        info = get_vertexai_version()
        assert info.installed is False

        all_info = get_all_sdk_versions()
        assert all_info["openai"].installed is False

def test_get_sdk_versions_installed_no_version():
    """Verify version detection when SDK installed but __version__ missing."""
    import sys
    mock_openai = MagicMock()
    del mock_openai.__version__

    with patch.dict(sys.modules, {"openai": mock_openai}):
        from vetch.compat import get_openai_version
        info = get_openai_version()
        assert info.installed is True
        assert info.version is None

