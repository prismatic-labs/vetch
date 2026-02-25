"""Tests for Azure OpenAI provider wrapper."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch


class TestAzureRegionInference:
    """Tests for Azure region inference from URLs."""

    def test_cognitive_services_url(self) -> None:
        """Infer region from cognitive services URL."""
        from vetch.providers.azure_openai import infer_region_from_azure_url

        result = infer_region_from_azure_url(
            "https://eastus.api.cognitive.microsoft.com/openai/deployments/gpt-4"
        )
        assert result == "US-EAST"

    def test_westeurope_cognitive_services(self) -> None:
        """Infer region from West Europe cognitive services URL."""
        from vetch.providers.azure_openai import infer_region_from_azure_url

        result = infer_region_from_azure_url(
            "https://westeurope.api.cognitive.microsoft.com/openai"
        )
        assert result == "NL"

    def test_openai_azure_com_with_region_in_name(self) -> None:
        """Infer region from resource name containing region hint."""
        from vetch.providers.azure_openai import infer_region_from_azure_url

        result = infer_region_from_azure_url(
            "https://my-eastus-resource.openai.azure.com/"
        )
        assert result == "US-EAST"

    def test_openai_azure_com_no_region(self) -> None:
        """Return None when resource name doesn't contain region."""
        from vetch.providers.azure_openai import infer_region_from_azure_url

        result = infer_region_from_azure_url(
            "https://my-custom-resource.openai.azure.com/"
        )
        assert result is None

    def test_none_url(self) -> None:
        """Return None for None URL."""
        from vetch.providers.azure_openai import infer_region_from_azure_url

        assert infer_region_from_azure_url(None) is None

    def test_non_azure_url(self) -> None:
        """Return None for non-Azure URLs."""
        from vetch.providers.azure_openai import infer_region_from_azure_url

        assert infer_region_from_azure_url("https://api.openai.com/v1") is None


class TestAzureRegionMap:
    """Tests for Azure region mapping coverage."""

    def test_all_major_regions_mapped(self) -> None:
        """Verify key Azure regions are in the map."""
        from vetch.providers.azure_openai import AZURE_REGION_MAP

        # Verify major regions exist
        assert "eastus" in AZURE_REGION_MAP
        assert "westeurope" in AZURE_REGION_MAP
        assert "japaneast" in AZURE_REGION_MAP
        assert "australiaeast" in AZURE_REGION_MAP
        assert "canadacentral" in AZURE_REGION_MAP

    def test_region_map_values_are_strings(self) -> None:
        """All region map values should be strings."""
        from vetch.providers.azure_openai import AZURE_REGION_MAP

        for key, value in AZURE_REGION_MAP.items():
            assert isinstance(key, str), f"Key {key} is not a string"
            assert isinstance(value, str), f"Value for {key} is not a string"


class TestInstrumentAzureOpenAI:
    """Tests for Azure OpenAI module instrumentation."""

    def test_returns_false_when_openai_not_imported(self) -> None:
        """Returns False if openai not in sys.modules."""
        import vetch.providers.azure_openai as azure_provider
        from vetch.providers.azure_openai import instrument_azure_openai_module

        # Temporarily remove openai from sys.modules
        openai_module = sys.modules.pop("openai", None)
        azure_provider._module_instrumented = False

        try:
            result = instrument_azure_openai_module()
            assert result is False
        finally:
            if openai_module:
                sys.modules["openai"] = openai_module

    def test_returns_true_when_already_instrumented(self) -> None:
        """Returns True if already instrumented."""
        import vetch.providers.azure_openai as azure_provider

        azure_provider._module_instrumented = True
        try:
            result = azure_provider.instrument_azure_openai_module()
            assert result is True
        finally:
            azure_provider._module_instrumented = False

    def test_instrument_patches_azure_openai_init(self) -> None:
        """Instrumentation patches AzureOpenAI.__init__."""
        import vetch.providers.azure_openai as azure_provider

        azure_provider._module_instrumented = False

        # Create a mock openai module with AzureOpenAI
        mock_openai = MagicMock()
        mock_openai.AzureOpenAI = type("AzureOpenAI", (), {"__init__": lambda s: None})
        mock_openai.AsyncAzureOpenAI = type(
            "AsyncAzureOpenAI", (), {"__init__": lambda s: None}
        )

        with patch.dict(sys.modules, {"openai": mock_openai}):
            try:
                result = azure_provider.instrument_azure_openai_module()
                assert result is True
                assert azure_provider._module_instrumented is True
            finally:
                azure_provider.uninstrument_azure_openai_module()


class TestUninstrumentAzureOpenAI:
    """Tests for Azure OpenAI module uninstrumentation."""

    def test_returns_true_when_not_instrumented(self) -> None:
        """Returns True if not instrumented."""
        import vetch.providers.azure_openai as azure_provider

        azure_provider._module_instrumented = False
        result = azure_provider.uninstrument_azure_openai_module()
        assert result is True

    def test_returns_true_when_openai_not_in_modules(self) -> None:
        """Returns True when openai not in sys.modules."""
        import vetch.providers.azure_openai as azure_provider

        openai_module = sys.modules.pop("openai", None)
        azure_provider._module_instrumented = True

        try:
            result = azure_provider.uninstrument_azure_openai_module()
            assert result is True
            assert azure_provider._module_instrumented is False
        finally:
            if openai_module:
                sys.modules["openai"] = openai_module


class TestAzureOpenAIIntegration:
    """Integration tests for Azure OpenAI in instrument/uninstrument."""

    def test_instrument_includes_azure(self) -> None:
        """vetch.instrument() tries to instrument Azure OpenAI."""
        import vetch

        vetch._instrumented = False

        # instrument() should handle Azure OpenAI gracefully
        result = vetch.instrument()
        assert isinstance(result, bool)
        vetch._instrumented = False

    def test_uninstrument_includes_azure(self) -> None:
        """vetch.uninstrument() tries to uninstrument Azure OpenAI."""
        import vetch

        vetch._instrumented = True
        result = vetch.uninstrument()
        assert result is True
        assert vetch._instrumented is False
