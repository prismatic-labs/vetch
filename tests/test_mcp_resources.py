"""Tests for MCP resource handlers.

These tests exercise the resource functions directly (no MCP transport).
"""

from __future__ import annotations

from vetch.mcp.resources import (
    get_config,
    get_energy_data,
    get_pricing_data,
    get_version,
    list_models,
)


class TestListModels:
    """Tests for list_models resource."""

    def test_returns_sorted_list(self) -> None:
        """Model list is sorted alphabetically."""
        models = list_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert models == sorted(models)

    def test_contains_known_models(self) -> None:
        """Model list includes well-known models."""
        models = list_models()
        # At minimum, gpt-4o should be in the registry
        assert "gpt-4o" in models


class TestGetEnergyData:
    """Tests for energy data lookup."""

    def test_known_model(self) -> None:
        """Returns energy data for a known model."""
        data = get_energy_data("gpt-4o")
        assert data["model"] == "gpt-4o"
        assert "error" not in data

    def test_unknown_model(self) -> None:
        """Returns error for unknown model."""
        data = get_energy_data("nonexistent-model-xyz")
        assert "error" in data


class TestGetPricingData:
    """Tests for pricing data lookup."""

    def test_known_model(self) -> None:
        """Returns pricing data for a known model."""
        data = get_pricing_data("gpt-4o")
        assert data["model"] == "gpt-4o"
        assert "error" not in data

    def test_unknown_model(self) -> None:
        """Returns error for unknown model."""
        data = get_pricing_data("nonexistent-model-xyz")
        assert "error" in data


class TestGetConfig:
    """Tests for config resource."""

    def test_returns_config_dict(self) -> None:
        """Config returns expected keys."""
        config = get_config()
        assert "region" in config
        assert "output" in config
        assert "default_pue" in config
        assert "cache_mode" in config


class TestGetVersion:
    """Tests for version resource."""

    def test_returns_version_string(self) -> None:
        """Version is a non-empty string."""
        version = get_version()
        assert isinstance(version, str)
        assert len(version) > 0
