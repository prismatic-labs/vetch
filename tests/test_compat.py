"""Tests for compat module."""



from vetch.compat import (
    MIN_PYTHON_VERSION,
    check_python_version,
    detect_existing_patches,
    get_python_version,
    is_datadog_patched,
    is_opentelemetry_patched,
    is_sentry_patched,
    parse_version,
    version_in_range,
)


class TestPythonVersion:
    """Tests for Python version checking."""

    def test_min_python_version(self) -> None:
        """Minimum Python version should be 3.9."""
        assert MIN_PYTHON_VERSION == (3, 9)

    def test_check_python_version_passes(self) -> None:
        """Current Python should pass version check."""
        # We're running tests, so Python must be >= 3.9
        assert check_python_version() is True

    def test_get_python_version_format(self) -> None:
        """Python version string should be X.Y.Z format."""
        version = get_python_version()
        parts = version.split(".")
        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)


class TestParseVersion:
    """Tests for version string parsing."""

    def test_simple_version(self) -> None:
        """Parse simple X.Y.Z version."""
        assert parse_version("1.2.3") == (1, 2, 3)

    def test_two_part_version(self) -> None:
        """Parse X.Y version."""
        assert parse_version("1.2") == (1, 2, 0)

    def test_single_part_version(self) -> None:
        """Parse X version."""
        assert parse_version("1") == (1, 0, 0)

    def test_beta_version(self) -> None:
        """Parse version with beta suffix."""
        assert parse_version("1.2.3b1") == (1, 2, 3)

    def test_alpha_version(self) -> None:
        """Parse version with alpha suffix."""
        assert parse_version("2.0.0a1") == (2, 0, 0)

    def test_rc_version(self) -> None:
        """Parse version with rc suffix."""
        assert parse_version("1.0.0rc1") == (1, 0, 0)

    def test_empty_string(self) -> None:
        """Empty string returns (0, 0, 0)."""
        assert parse_version("") == (0, 0, 0)


class TestVersionInRange:
    """Tests for version range checking."""

    def test_in_range(self) -> None:
        """Version in range returns True."""
        assert version_in_range("1.5.0", "1.0.0", "2.0.0") is True

    def test_at_min_boundary(self) -> None:
        """Version at min boundary is included."""
        assert version_in_range("1.0.0", "1.0.0", "2.0.0") is True

    def test_at_max_boundary(self) -> None:
        """Version at max boundary is excluded."""
        assert version_in_range("2.0.0", "1.0.0", "2.0.0") is False

    def test_below_range(self) -> None:
        """Version below range returns False."""
        assert version_in_range("0.9.0", "1.0.0", "2.0.0") is False

    def test_above_range(self) -> None:
        """Version above range returns False."""
        assert version_in_range("2.1.0", "1.0.0", "2.0.0") is False


class TestPatchDetection:
    """Tests for observability patch detection."""

    def test_unpatched_function(self) -> None:
        """Unpatched function has no patches detected."""

        def sample_func() -> None:
            pass

        assert is_datadog_patched(sample_func) is False
        assert is_opentelemetry_patched(sample_func) is False
        assert is_sentry_patched(sample_func) is False
        assert detect_existing_patches(sample_func) == []

    def test_datadog_patched_detection(self) -> None:
        """Detect Datadog patch marker."""

        def sample_func() -> None:
            pass

        sample_func._datadog_patch = True  # type: ignore[attr-defined]
        assert is_datadog_patched(sample_func) is True
        assert "datadog" in detect_existing_patches(sample_func)

    def test_opentelemetry_patched_detection(self) -> None:
        """Detect OpenTelemetry patch marker."""

        def sample_func() -> None:
            pass

        sample_func._otel_patched = True  # type: ignore[attr-defined]
        assert is_opentelemetry_patched(sample_func) is True
        assert "opentelemetry" in detect_existing_patches(sample_func)

    def test_sentry_patched_detection(self) -> None:
        """Detect Sentry patch marker."""

        def sample_func() -> None:
            pass

        sample_func._sentry_wrapped = True  # type: ignore[attr-defined]
        assert is_sentry_patched(sample_func) is True
        assert "sentry" in detect_existing_patches(sample_func)

    def test_generic_wrapped_detection(self) -> None:
        """Detect generic __wrapped__ attribute."""

        def sample_func() -> None:
            pass

        def original() -> None:
            pass

        sample_func.__wrapped__ = original  # type: ignore[attr-defined]
        patches = detect_existing_patches(sample_func)
        assert "unknown" in patches

    def test_multiple_patches(self) -> None:
        """Detect multiple patches on same function."""

        def sample_func() -> None:
            pass

        sample_func._datadog_patch = True  # type: ignore[attr-defined]
        sample_func._sentry_wrapped = True  # type: ignore[attr-defined]
        patches = detect_existing_patches(sample_func)
        assert "datadog" in patches
        assert "sentry" in patches
        assert len(patches) == 2
