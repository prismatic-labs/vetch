"""Tests for Vetch exception hierarchy.

These tests verify:
- Exception inheritance chain
- Exception context fields
- Proper error messages
"""

from __future__ import annotations

import pytest

from vetch.exceptions import (
    VetchError,
    RegistryError,
    ProviderError,
    ConfigurationError,
    CalibrationError,
    StorageError,
)


class TestVetchError:
    """Tests for base VetchError."""

    def test_inherits_from_value_error(self) -> None:
        """VetchError should inherit from ValueError."""
        assert issubclass(VetchError, ValueError)

    def test_can_be_raised(self) -> None:
        """VetchError can be raised and caught."""
        with pytest.raises(VetchError, match="test error"):
            raise VetchError("test error")

    def test_can_catch_as_value_error(self) -> None:
        """VetchError can be caught as ValueError."""
        with pytest.raises(ValueError):
            raise VetchError("caught as ValueError")


class TestRegistryError:
    """Tests for RegistryError."""

    def test_inherits_from_vetch_error(self) -> None:
        """RegistryError should inherit from VetchError."""
        assert issubclass(RegistryError, VetchError)

    def test_stores_model_context(self) -> None:
        """RegistryError stores model name for debugging."""
        error = RegistryError("Model not found", model="gpt-5-turbo")
        assert error.model == "gpt-5-turbo"
        assert "Model not found" in str(error)

    def test_model_optional(self) -> None:
        """Model context is optional."""
        error = RegistryError("Generic registry error")
        assert error.model is None


class TestProviderError:
    """Tests for ProviderError."""

    def test_inherits_from_vetch_error(self) -> None:
        """ProviderError should inherit from VetchError."""
        assert issubclass(ProviderError, VetchError)

    def test_stores_provider_context(self) -> None:
        """ProviderError stores provider name."""
        error = ProviderError("SDK patch failed", provider="openai")
        assert error.provider == "openai"
        assert "SDK patch failed" in str(error)

    def test_provider_optional(self) -> None:
        """Provider context is optional."""
        error = ProviderError("Generic provider error")
        assert error.provider is None


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_inherits_from_vetch_error(self) -> None:
        """ConfigurationError should inherit from VetchError."""
        assert issubclass(ConfigurationError, VetchError)

    def test_stores_field_context(self) -> None:
        """ConfigurationError stores field name."""
        error = ConfigurationError("Invalid value", field="wh_per_1k_input")
        assert error.field == "wh_per_1k_input"

    def test_field_optional(self) -> None:
        """Field context is optional."""
        error = ConfigurationError("Generic config error")
        assert error.field is None


class TestCalibrationError:
    """Tests for CalibrationError."""

    def test_inherits_from_vetch_error(self) -> None:
        """CalibrationError should inherit from VetchError."""
        assert issubclass(CalibrationError, VetchError)

    def test_stores_gpu_error_context(self) -> None:
        """CalibrationError stores GPU error details."""
        error = CalibrationError("GPU init failed", gpu_error="pynvml not found")
        assert error.gpu_error == "pynvml not found"

    def test_gpu_error_optional(self) -> None:
        """GPU error context is optional."""
        error = CalibrationError("Generic calibration error")
        assert error.gpu_error is None


class TestStorageError:
    """Tests for StorageError."""

    def test_inherits_from_vetch_error(self) -> None:
        """StorageError should inherit from VetchError."""
        assert issubclass(StorageError, VetchError)

    def test_stores_db_path_context(self) -> None:
        """StorageError stores database path."""
        error = StorageError("Connection failed", db_path="/home/user/.vetch/usage.db")
        assert error.db_path == "/home/user/.vetch/usage.db"

    def test_db_path_optional(self) -> None:
        """Database path context is optional."""
        error = StorageError("Generic storage error")
        assert error.db_path is None


class TestExceptionChain:
    """Tests for exception hierarchy chain."""

    def test_all_exceptions_derive_from_vetch_error(self) -> None:
        """All custom exceptions inherit from VetchError."""
        exceptions = [
            RegistryError,
            ProviderError,
            ConfigurationError,
            CalibrationError,
            StorageError,
        ]
        for exc in exceptions:
            assert issubclass(exc, VetchError)

    def test_can_catch_all_with_vetch_error(self) -> None:
        """Single except VetchError catches all custom exceptions."""
        errors_to_test = [
            RegistryError("test"),
            ProviderError("test"),
            ConfigurationError("test"),
            CalibrationError("test"),
            StorageError("test"),
        ]

        for error in errors_to_test:
            try:
                raise error
            except VetchError as e:
                assert True  # Successfully caught
            except Exception:
                pytest.fail(f"{type(error).__name__} not caught by VetchError")
