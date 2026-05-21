"""Tests for MCP tool handlers.

These tests exercise the tool functions directly (no MCP transport),
verifying that each handler returns correct data and handles errors gracefully.
"""

from __future__ import annotations

import pytest

from vetch.budget import clear_budgets, set_budget
from vetch.calculation import calculate_water
from vetch.mcp.tools import (
    vetch_check_budget,
    vetch_cleanest_region,
    vetch_compare,
    vetch_estimate,
    vetch_grid_intensity,
    vetch_registry_lookup,
    vetch_session_stats,
    vetch_status,
)
from vetch.stats import _reset_session_stats


class TestVetchEstimate:
    """Tests for vetch_estimate tool."""

    def test_returns_all_fields(self) -> None:
        """Estimate returns energy, carbon, water, cost, confidence, training context."""
        result = vetch_estimate(
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            region="us-east-1",
        )

        assert "error" not in result
        assert result["model"] == "gpt-4o"
        assert result["input_tokens"] == 1000
        assert result["output_tokens"] == 500
        assert isinstance(result["energy_wh"], float)
        assert isinstance(result["carbon_g"], float)
        assert isinstance(result["water_l"], float)
        assert isinstance(result["water_ml"], float)
        assert isinstance(result["cost_usd"], float)
        assert result["confidence"] in ("high", "medium", "low")
        assert isinstance(result["training_context"], list)

    def test_water_units_are_reported_correctly(self) -> None:
        """Estimate reports water in liters and milliliters without unit drift."""
        result = vetch_estimate(
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
        )

        assert "error" not in result
        assert result["water_l"] > 0
        assert result["water_ml"] == pytest.approx(result["water_l"] * 1000, rel=1e-4)

        expected_l = calculate_water(
            energy_wh=result["energy_wh"],
            model="gpt-4o",
            region="us-east-1",
        )
        assert result["water_l"] == pytest.approx(expected_l, abs=1e-8)

    def test_cost_breakdown(self) -> None:
        """Estimate includes cost breakdown."""
        result = vetch_estimate(model="gpt-4o", input_tokens=1000, output_tokens=500)

        assert "cost_breakdown" in result
        assert "input_usd" in result["cost_breakdown"]
        assert "output_usd" in result["cost_breakdown"]
        assert "billing_tier" in result["cost_breakdown"]

    def test_unknown_model_does_not_crash(self) -> None:
        """Unknown model returns a result (may use heuristic), never crashes."""
        result = vetch_estimate(
            model="totally-fake-model-xyz",
            input_tokens=100,
            output_tokens=50,
        )
        # Should either succeed with heuristic or return error dict
        assert isinstance(result, dict)

    def test_with_region(self) -> None:
        """Estimate with explicit region."""
        result = vetch_estimate(
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            region="US-CAL-CISO",
        )
        assert "error" not in result
        assert result["region"] == "US-CAL-CISO"


class TestVetchCompare:
    """Tests for vetch_compare tool."""

    def test_compare_multiple_models(self) -> None:
        """Compare returns sorted results with cheapest/greenest flags."""
        result = vetch_compare(
            models=["gpt-4o", "gpt-4o-mini"],
            input_tokens=1000,
            output_tokens=500,
        )

        assert "error" not in result
        assert isinstance(result["comparisons"], list)
        assert len(result["comparisons"]) >= 1

    def test_marks_cheapest_and_greenest(self) -> None:
        """Compare flags the cheapest and greenest models."""
        result = vetch_compare(
            models=["gpt-4o", "gpt-4o-mini"],
            input_tokens=1000,
            output_tokens=500,
        )

        valid = result["comparisons"]
        if len(valid) >= 2:
            cheapest_count = sum(1 for r in valid if r.get("is_cheapest"))
            greenest_count = sum(1 for r in valid if r.get("is_greenest"))
            assert cheapest_count == 1
            assert greenest_count == 1

    def test_sort_by_carbon(self) -> None:
        """Compare can sort by carbon_g."""
        result = vetch_compare(
            models=["gpt-4o", "gpt-4o-mini"],
            input_tokens=1000,
            output_tokens=500,
            sort_by="carbon_g",
        )
        assert result["sort_by"] == "carbon_g"


class TestVetchSessionStats:
    """Tests for vetch_session_stats tool."""

    def setup_method(self) -> None:
        _reset_session_stats()

    def teardown_method(self) -> None:
        _reset_session_stats()

    def test_returns_session_data(self) -> None:
        """Session stats returns totals, advisories, and training context."""
        result = vetch_session_stats()

        assert "error" not in result
        assert "total_requests" in result
        assert "total_energy_wh" in result
        assert "total_carbon_g" in result
        assert "total_cost_usd" in result
        assert isinstance(result["advisories"], list)
        assert isinstance(result["training_context"], list)
        assert isinstance(result["models_used"], list)


class TestVetchStatus:
    """Tests for vetch_status tool."""

    def test_returns_health_and_version(self) -> None:
        """Status returns version, health, and budget."""
        result = vetch_status()

        assert "error" not in result
        assert "version" in result
        assert "health" in result
        assert "budget" in result


class TestVetchGridIntensity:
    """Tests for vetch_grid_intensity tool."""

    def test_returns_intensity(self) -> None:
        """Grid intensity returns region and signal quality."""
        result = vetch_grid_intensity(region="US-CAL-CISO")

        assert "error" not in result
        assert result["region"] == "US-CAL-CISO"
        assert "intensity_gco2e_kwh" in result
        assert "signal_quality" in result


class TestVetchCleanestRegion:
    """Tests for vetch_cleanest_region tool."""

    def test_returns_cleanest(self) -> None:
        """Cleanest region returns one of the candidates."""
        result = vetch_cleanest_region(regions=["US-CAL-CISO", "DE", "FR"])

        assert "error" not in result
        assert result["cleanest_region"] in ["US-CAL-CISO", "DE", "FR"]
        assert isinstance(result["intensity_gco2e_kwh"], float)


class TestVetchRegistryLookup:
    """Tests for vetch_registry_lookup tool."""

    def test_known_model(self) -> None:
        """Lookup returns energy and/or pricing data for a known model."""
        result = vetch_registry_lookup(model="gpt-4o")

        assert result["model"] == "gpt-4o"
        # Should have at least energy or pricing
        assert "energy" in result or "pricing" in result

    def test_unknown_model_returns_error(self) -> None:
        """Lookup returns error dict for unknown model."""
        result = vetch_registry_lookup(model="nonexistent-model-xyz")

        assert "error" in result
        assert "nonexistent-model-xyz" in result["error"]


class TestVetchCheckBudget:
    """Tests for vetch_check_budget tool."""

    def setup_method(self) -> None:
        clear_budgets()

    def teardown_method(self) -> None:
        clear_budgets()

    def test_no_budgets_returns_empty_with_message(self) -> None:
        """No budgets configured returns helpful message."""
        result = vetch_check_budget()

        assert "error" not in result
        assert result["budgets"] == {}
        assert "message" in result

    def test_returns_budget_detail(self) -> None:
        """Configured budget returns threshold and remaining."""
        set_budget("session", cost_usd=10.0, window="session")

        result = vetch_check_budget()

        assert "error" not in result
        assert "session" in result["budgets"]
        cost = result["budgets"]["session"]["cost_usd"]
        assert cost["threshold"] == 10.0
        assert cost["remaining"] == 10.0
        assert cost["percentage_used"] == 0.0
