"""Tests for vetch.instrument() global auto-patching."""

from __future__ import annotations

import logging
import sys
from unittest.mock import patch

import pytest


class TestInstrument:
    """Tests for the instrument() function."""

    def test_instrument_returns_false_when_disabled(self) -> None:
        """instrument() returns False when VETCH_DISABLED=true."""
        import vetch

        # Temporarily set _DISABLED
        original = vetch._DISABLED
        vetch._DISABLED = True
        vetch._instrumented = False

        try:
            result = vetch.instrument()
            assert result is False
        finally:
            vetch._DISABLED = original

    def test_instrument_returns_true_when_already_instrumented(self) -> None:
        """instrument() returns True immediately if already instrumented."""
        import vetch

        vetch._instrumented = True
        try:
            result = vetch.instrument()
            assert result is True
        finally:
            vetch._instrumented = False

    def test_instrument_sets_region_env_var(self) -> None:
        """instrument() sets VETCH_REGION if provided."""
        import os

        import vetch

        vetch._instrumented = False
        original_region = os.environ.get("VETCH_REGION")

        try:
            # Clear region first
            os.environ.pop("VETCH_REGION", None)
            vetch._default_region = None

            with patch.object(vetch, "add_global_tags"):
                vetch.instrument(region="test-region")

            # Check that region is stored in module state, not os.environ
            assert vetch._default_region == "test-region"
            assert vetch.get_default_region() == "test-region"
            # Environment should NOT be mutated
            assert os.environ.get("VETCH_REGION") is None
        finally:
            vetch._instrumented = False
            vetch._default_region = None
            if original_region:
                os.environ["VETCH_REGION"] = original_region
            else:
                os.environ.pop("VETCH_REGION", None)

    def test_instrument_adds_global_tags(self) -> None:
        """instrument() calls add_global_tags if tags provided."""
        import vetch

        vetch._instrumented = False

        with patch.object(vetch, "add_global_tags") as mock_tags:
            vetch.instrument(tags={"service": "test"})
            mock_tags.assert_called_once_with({"service": "test"})

        vetch._instrumented = False

    def test_instrument_handles_missing_sdks(self) -> None:
        """instrument() handles missing SDKs gracefully."""
        import vetch

        vetch._instrumented = False

        # Should not raise even if no SDKs are installed
        result = vetch.instrument()

        # Result depends on whether SDKs are in sys.modules
        assert isinstance(result, bool)
        vetch._instrumented = False

    def test_instrument_logs_debug_on_unexpected_error(self) -> None:
        """instrument() logs debug message when provider instrumentation fails."""
        import vetch

        vetch._instrumented = False

        # Mock the OpenAI instrumentation to raise an unexpected error
        with patch(
            "vetch.providers.openai.instrument_openai_module",
            side_effect=RuntimeError("Unexpected error"),
        ):
            # Should not raise
            result = vetch.instrument()

        assert isinstance(result, bool)
        vetch._instrumented = False

    def test_instrument_handles_anthropic_error(self) -> None:
        """instrument() handles errors from Anthropic instrumentation."""
        import vetch

        vetch._instrumented = False

        with patch(
            "vetch.providers.anthropic.instrument_anthropic_module",
            side_effect=RuntimeError("Anthropic error"),
        ):
            result = vetch.instrument()

        assert isinstance(result, bool)
        vetch._instrumented = False

    def test_instrument_handles_vertexai_error(self) -> None:
        """instrument() handles errors from Vertex AI instrumentation."""
        import vetch

        vetch._instrumented = False

        with patch(
            "vetch.providers.vertexai.instrument_vertexai_module",
            side_effect=RuntimeError("VertexAI error"),
        ):
            result = vetch.instrument()

        assert isinstance(result, bool)
        vetch._instrumented = False

    def test_instrument_handles_azure_openai_error(self) -> None:
        """instrument() handles errors from Azure OpenAI instrumentation."""
        import vetch

        vetch._instrumented = False

        with patch(
            "vetch.providers.azure_openai.instrument_azure_openai_module",
            side_effect=RuntimeError("Azure OpenAI error"),
        ):
            result = vetch.instrument()

        assert isinstance(result, bool)
        vetch._instrumented = False

    def test_instrument_handles_genai_error(self) -> None:
        """instrument() handles errors from Google GenAI instrumentation."""
        import vetch

        vetch._instrumented = False

        with patch(
            "vetch.providers.genai.instrument_genai_module",
            side_effect=RuntimeError("GenAI error"),
        ):
            result = vetch.instrument()

        assert isinstance(result, bool)
        vetch._instrumented = False

    def test_instrument_logs_exceptions(self, caplog: pytest.LogCaptureFixture) -> None:
        """instrument() logs debug messages when provider instrumentation raises exceptions."""
        import vetch

        vetch._instrumented = False

        with caplog.at_level(logging.DEBUG, logger="vetch"):
            with patch(
                "vetch.providers.openai.instrument_openai_module",
                side_effect=ValueError("Test OpenAI error"),
            ):
                with patch(
                    "vetch.providers.anthropic.instrument_anthropic_module",
                    side_effect=TypeError("Test Anthropic error"),
                ):
                    vetch.instrument()

        # Verify debug messages were logged
        assert "Failed to instrument OpenAI" in caplog.text
        assert "Failed to instrument Anthropic" in caplog.text
        vetch._instrumented = False


class TestInstrumentOpenAIModule:
    """Tests for instrument_openai_module()."""

    def test_returns_false_when_openai_not_imported(self) -> None:
        """Returns False if openai not in sys.modules."""
        from vetch.providers.openai import instrument_openai_module

        # Temporarily remove openai from sys.modules if present
        openai_module = sys.modules.pop("openai", None)

        try:
            # Reset instrumented state
            import vetch.providers.openai as openai_provider

            openai_provider._module_instrumented = False

            result = instrument_openai_module()
            assert result is False
        finally:
            if openai_module:
                sys.modules["openai"] = openai_module

    def test_returns_true_when_already_instrumented(self) -> None:
        """Returns True immediately if already instrumented."""
        import vetch.providers.openai as openai_provider

        openai_provider._module_instrumented = True
        try:
            result = openai_provider.instrument_openai_module()
            assert result is True
        finally:
            openai_provider._module_instrumented = False


class TestInstrumentAnthropicModule:
    """Tests for instrument_anthropic_module()."""

    def test_returns_false_when_anthropic_not_imported(self) -> None:
        """Returns False if anthropic not in sys.modules."""
        from vetch.providers.anthropic import instrument_anthropic_module

        # Temporarily remove anthropic from sys.modules if present
        anthropic_module = sys.modules.pop("anthropic", None)

        try:
            import vetch.providers.anthropic as anthropic_provider

            anthropic_provider._module_instrumented = False

            result = instrument_anthropic_module()
            assert result is False
        finally:
            if anthropic_module:
                sys.modules["anthropic"] = anthropic_module

    def test_returns_true_when_already_instrumented(self) -> None:
        """Returns True immediately if already instrumented."""
        import vetch.providers.anthropic as anthropic_provider

        anthropic_provider._module_instrumented = True
        try:
            result = anthropic_provider.instrument_anthropic_module()
            assert result is True
        finally:
            anthropic_provider._module_instrumented = False


class TestInstrumentVertexAIModule:
    """Tests for instrument_vertexai_module()."""

    def test_returns_false_when_vertexai_not_imported(self) -> None:
        """Returns False if vertexai not in sys.modules."""
        from vetch.providers.vertexai import instrument_vertexai_module

        # Temporarily remove vertexai modules
        removed = {}
        for mod in ["google.cloud.aiplatform", "vertexai"]:
            if mod in sys.modules:
                removed[mod] = sys.modules.pop(mod)

        try:
            import vetch.providers.vertexai as vertexai_provider

            vertexai_provider._module_instrumented = False

            result = instrument_vertexai_module()
            assert result is False
        finally:
            sys.modules.update(removed)

    def test_returns_true_when_already_instrumented(self) -> None:
        """Returns True immediately if already instrumented."""
        import vetch.providers.vertexai as vertexai_provider

        vertexai_provider._module_instrumented = True
        try:
            result = vertexai_provider.instrument_vertexai_module()
            assert result is True
        finally:
            vertexai_provider._module_instrumented = False


class TestWrapDisabled:
    """Tests for wrap() when vetch is disabled."""

    def test_wrap_returns_noop_context_when_disabled(self) -> None:
        """wrap() returns no-op context when VETCH_DISABLED=true."""
        import vetch

        original = vetch._DISABLED
        try:
            vetch._DISABLED = True
            ctx = vetch.wrap(region="us-east-1", tags={"test": "value"})
            # Context should be created even when disabled (no-op mode)
            assert ctx is not None
        finally:
            vetch._DISABLED = original


class TestLazyImports:
    """Tests for lazy import __getattr__ paths."""

    def test_session_lazy_import(self) -> None:
        """Session class can be imported via __getattr__."""
        import vetch

        # Access Session via lazy import
        session_class = vetch.Session
        assert session_class is not None
        assert session_class.__name__ == "Session"

    def test_get_session_stats_lazy_import(self) -> None:
        """get_session_stats can be imported via __getattr__."""
        import vetch

        func = vetch.get_session_stats
        assert func is not None
        assert callable(func)

    def test_generate_advisories_lazy_import(self) -> None:
        """generate_advisories can be imported via __getattr__."""
        import vetch

        func = vetch.generate_advisories
        assert func is not None
        assert callable(func)


class TestConfigFunctions:
    """Tests for configuration functions."""

    def test_set_tag_cardinality_limit(self) -> None:
        """set_tag_cardinality_limit sets the cardinality limit."""
        import vetch

        # Should not raise
        vetch.set_tag_cardinality_limit(500)
        vetch.set_tag_cardinality_limit(1000)

    def test_set_tag_allowlist(self) -> None:
        """set_tag_allowlist sets allowed tag keys."""
        import vetch

        # Should not raise
        vetch.set_tag_allowlist(["team", "env", "service"])
        vetch.set_tag_allowlist([])

    def test_set_redacted_tags(self) -> None:
        """set_redacted_tags sets tags to be hashed."""
        import vetch

        # Should not raise
        vetch.set_redacted_tags(["user_email", "customer_id"])
        vetch.set_redacted_tags([])

    def test_get_default_region_from_env(self) -> None:
        """get_default_region falls back to VETCH_REGION env var."""
        import os

        import vetch

        # Reset module state
        vetch._default_region = None

        original = os.environ.get("VETCH_REGION")
        try:
            os.environ["VETCH_REGION"] = "test-region-123"
            result = vetch.get_default_region()
            assert result == "test-region-123"
        finally:
            vetch._default_region = None
            if original:
                os.environ["VETCH_REGION"] = original
            else:
                os.environ.pop("VETCH_REGION", None)

    def test_get_default_region_module_takes_precedence(self) -> None:
        """get_default_region returns module-level region over env var."""
        import os

        import vetch

        original = os.environ.get("VETCH_REGION")
        try:
            os.environ["VETCH_REGION"] = "env-region"
            vetch._default_region = "module-region"
            result = vetch.get_default_region()
            assert result == "module-region"
        finally:
            vetch._default_region = None
            if original:
                os.environ["VETCH_REGION"] = original
            else:
                os.environ.pop("VETCH_REGION", None)


class TestSetLogLevel:
    """Tests for set_log_level() function."""

    def test_set_log_level_with_string(self) -> None:
        """set_log_level accepts string log levels."""
        import vetch

        # Store original level
        logger = logging.getLogger("vetch")
        original_level = logger.level

        try:
            vetch.set_log_level("ERROR")
            assert logger.level == logging.ERROR

            vetch.set_log_level("DEBUG")
            assert logger.level == logging.DEBUG

            vetch.set_log_level("warning")  # lowercase
            assert logger.level == logging.WARNING
        finally:
            logger.setLevel(original_level)

    def test_set_log_level_with_int(self) -> None:
        """set_log_level accepts integer log levels."""
        import vetch

        logger = logging.getLogger("vetch")
        original_level = logger.level

        try:
            vetch.set_log_level(logging.CRITICAL)
            assert logger.level == logging.CRITICAL

            vetch.set_log_level(logging.INFO)
            assert logger.level == logging.INFO
        finally:
            logger.setLevel(original_level)

    def test_set_log_level_invalid_string_defaults_to_warning(self) -> None:
        """Invalid string level defaults to WARNING."""
        import vetch

        logger = logging.getLogger("vetch")
        original_level = logger.level

        try:
            vetch.set_log_level("INVALID_LEVEL")
            assert logger.level == logging.WARNING
        finally:
            logger.setLevel(original_level)

    def test_set_log_level_exported(self) -> None:
        """set_log_level is in __all__ exports."""
        import vetch

        assert "set_log_level" in vetch.__all__


class TestLazyImportsAdvanced:
    """Tests for lazy __getattr__ imports in vetch.__init__."""

    def test_vetch_context_lazy_import(self) -> None:
        """VetchContext is accessible via lazy import."""
        import vetch

        # Access triggers lazy import
        ctx_class = vetch.VetchContext
        assert ctx_class is not None
        assert ctx_class.__name__ == "VetchContext"

    def test_get_session_stats_lazy_import(self) -> None:
        """get_session_stats is accessible via lazy import."""
        import vetch

        func = vetch.get_session_stats
        assert callable(func)
        assert func.__name__ == "get_session_stats"

    def test_generate_advisories_lazy_import(self) -> None:
        """generate_advisories is accessible via lazy import."""
        import vetch

        func = vetch.generate_advisories
        assert callable(func)
        assert func.__name__ == "generate_advisories"

    def test_configure_storage_lazy_import(self) -> None:
        """configure_storage is accessible via lazy import."""
        import vetch

        func = vetch.configure_storage
        assert callable(func)
        assert func.__name__ == "configure_storage"

    def test_query_usage_lazy_import(self) -> None:
        """query_usage is accessible via lazy import."""
        import vetch

        func = vetch.query_usage
        assert callable(func)
        assert func.__name__ == "query_usage"

    def test_budget_apis_lazy_import(self) -> None:
        """Budget APIs are accessible via lazy import."""
        import vetch

        assert callable(vetch.set_budget)
        assert callable(vetch.on_budget_alert)
        assert callable(vetch.get_budget_status)

    def test_configure_otlp_export_lazy_import(self) -> None:
        """configure_otlp_export is accessible via lazy import."""
        import vetch

        func = vetch.configure_otlp_export
        assert callable(func)
        assert func.__name__ == "configure_otlp_export"

    def test_get_cleanest_region_lazy_import(self) -> None:
        """get_cleanest_region is accessible via lazy import."""
        import vetch

        func = vetch.get_cleanest_region
        assert callable(func)
        assert func.__name__ == "get_cleanest_region"

    def test_invalid_attribute_raises_attribute_error(self) -> None:
        """Invalid attribute raises AttributeError."""
        import vetch

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = vetch.nonexistent_attribute


class TestWrapDisabledAdvanced:
    """Tests for wrap() when VETCH_DISABLED is True."""

    def test_wrap_returns_disabled_context_when_disabled(self) -> None:
        """wrap() returns disabled context when VETCH_DISABLED=true."""
        import vetch

        original = vetch._DISABLED
        vetch._DISABLED = True

        try:
            ctx = vetch.wrap(region="us-east-1")
            assert ctx._globally_disabled is True
        finally:
            vetch._DISABLED = original

    def test_wrap_disabled_context_is_noop(self) -> None:
        """Disabled context manager is a no-op."""
        import vetch

        original = vetch._DISABLED
        vetch._DISABLED = True

        try:
            with vetch.wrap(region="us-east-1") as ctx:
                # Should not track anything
                pass
            # Context should exist but be globally disabled
            assert ctx._globally_disabled is True
            # tracking_disabled should be set after entering context
            assert ctx.tracking_disabled is True
        finally:
            vetch._DISABLED = original


class TestAddGlobalTags:
    """Tests for add_global_tags function."""

    def test_add_global_tags_sets_tags(self) -> None:
        """add_global_tags stores tags in config."""
        import vetch
        from vetch.config import _reset_config, get_global_tags

        try:
            vetch.add_global_tags({"env": "test", "service": "test-svc"})
            tags = get_global_tags()
            assert tags["env"] == "test"
            assert tags["service"] == "test-svc"
        finally:
            _reset_config()


class TestRequireTags:
    """Tests for require_tags function."""

    def test_require_tags_sets_required(self) -> None:
        """require_tags stores required tag names."""
        import vetch
        from vetch.config import _reset_config, get_required_tags

        try:
            vetch.require_tags(["cost_center", "feature_id"])
            required = get_required_tags()
            assert "cost_center" in required
            assert "feature_id" in required
        finally:
            _reset_config()


class TestUninstrument:
    """Tests for uninstrument() function."""

    def test_uninstrument_returns_true_when_not_instrumented(self) -> None:
        """uninstrument() returns True when nothing was instrumented."""
        import vetch

        vetch._instrumented = False
        result = vetch.uninstrument()
        assert result is True

    def test_uninstrument_exported(self) -> None:
        """uninstrument is in __all__ exports."""
        import vetch

        assert "uninstrument" in vetch.__all__

    def test_uninstrument_resets_instrumented_flag(self) -> None:
        """uninstrument() resets _instrumented flag."""
        import vetch

        vetch._instrumented = True
        vetch.uninstrument()
        assert vetch._instrumented is False

    def test_uninstrument_handles_openai_error(self) -> None:
        """uninstrument() handles errors from OpenAI uninstrumentation."""
        import vetch

        vetch._instrumented = True

        with patch(
            "vetch.providers.openai.uninstrument_openai_module",
            side_effect=RuntimeError("OpenAI uninstrument error"),
        ):
            # Should not raise and should continue uninstrumenting other providers
            result = vetch.uninstrument()

        # Returns False because one provider failed, but doesn't crash
        assert vetch._instrumented is False

    def test_uninstrument_handles_anthropic_error(self) -> None:
        """uninstrument() handles errors from Anthropic uninstrumentation."""
        import vetch

        vetch._instrumented = True

        with patch(
            "vetch.providers.anthropic.uninstrument_anthropic_module",
            side_effect=RuntimeError("Anthropic uninstrument error"),
        ):
            result = vetch.uninstrument()

        assert vetch._instrumented is False

    def test_uninstrument_logs_exceptions(self, caplog: pytest.LogCaptureFixture) -> None:
        """uninstrument() logs debug messages when provider uninstrumentation raises exceptions."""
        import vetch

        vetch._instrumented = True

        with caplog.at_level(logging.DEBUG, logger="vetch"):
            with patch(
                "vetch.providers.openai.uninstrument_openai_module",
                side_effect=ValueError("Test OpenAI uninstrument error"),
            ):
                vetch.uninstrument()

        # Verify debug message was logged
        assert "Failed to uninstrument OpenAI" in caplog.text
        assert vetch._instrumented is False


class TestUninstrumentOpenAIModule:
    """Tests for uninstrument_openai_module()."""

    def test_returns_true_when_not_instrumented(self) -> None:
        """Returns True if module wasn't instrumented."""
        import vetch.providers.openai as openai_provider

        openai_provider._module_instrumented = False
        result = openai_provider.uninstrument_openai_module()
        assert result is True

    def test_returns_true_when_openai_not_in_modules(self) -> None:
        """Returns True if openai not in sys.modules."""
        import vetch.providers.openai as openai_provider

        # Temporarily remove openai from sys.modules if present
        openai_module = sys.modules.pop("openai", None)
        openai_provider._module_instrumented = True

        try:
            result = openai_provider.uninstrument_openai_module()
            assert result is True
            assert openai_provider._module_instrumented is False
        finally:
            if openai_module:
                sys.modules["openai"] = openai_module


class TestUninstrumentAnthropicModule:
    """Tests for uninstrument_anthropic_module()."""

    def test_returns_true_when_not_instrumented(self) -> None:
        """Returns True if module wasn't instrumented."""
        import vetch.providers.anthropic as anthropic_provider

        anthropic_provider._module_instrumented = False
        result = anthropic_provider.uninstrument_anthropic_module()
        assert result is True


class TestUninstrumentVertexAIModule:
    """Tests for uninstrument_vertexai_module()."""

    def test_returns_true_when_not_instrumented(self) -> None:
        """Returns True if module wasn't instrumented."""
        import vetch.providers.vertexai as vertexai_provider

        vertexai_provider._module_instrumented = False
        result = vertexai_provider.uninstrument_vertexai_module()
        assert result is True


class TestAutomaticContextCreation:
    """Tests for automatic context creation when using instrument()."""

    def test_instrument_stores_default_tags(self) -> None:
        """instrument() stores default tags for auto-context creation."""
        import vetch

        vetch._instrumented = False
        vetch._default_tags = None

        try:
            with patch.object(vetch, "add_global_tags"):
                vetch.instrument(tags={"service": "test", "env": "dev"})

            assert vetch._default_tags == {"service": "test", "env": "dev"}
            assert vetch.get_default_tags() == {"service": "test", "env": "dev"}
        finally:
            vetch._instrumented = False
            vetch._default_tags = None

    def test_get_default_tags_returns_none_when_not_set(self) -> None:
        """get_default_tags() returns None when no tags were set."""
        import vetch

        vetch._default_tags = None
        result = vetch.get_default_tags()
        assert result is None


class TestAwrap:
    """Tests for awrap() async context manager."""

    def test_awrap_exported(self) -> None:
        """awrap is in __all() exports."""
        import vetch

        assert "awrap" in vetch.__all__

    @pytest.mark.asyncio
    async def test_awrap_basic(self) -> None:
        """awrap() returns an async context manager."""
        import vetch

        async with vetch.awrap(region="us-east-1") as ctx:
            # Context should be created
            assert ctx is not None

    @pytest.mark.asyncio
    async def test_awrap_disabled(self) -> None:
        """awrap() returns disabled context when VETCH_DISABLED=true."""
        import vetch

        original = vetch._DISABLED
        vetch._DISABLED = True

        try:
            async with vetch.awrap(region="us-east-1") as ctx:
                assert ctx._globally_disabled is True
        finally:
            vetch._DISABLED = original
