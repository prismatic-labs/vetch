"""Tests for energy, carbon, and cost calculation.

These tests verify:
- Model alias resolution
- Energy calculation (Wh)
- Carbon calculation (gCO2e)
- Cost calculation (USD)
- Conservative fallbacks for unknown models
- Energy overrides
- Token estimation (with and without tiktoken)
"""

from __future__ import annotations

import pytest

from vetch.calculation import (
    calculate_carbon,
    calculate_cost,
    calculate_energy,
    estimate_tokens,
    resolve_model,
)


class TestModelResolution:
    """Tests for model name resolution and aliases."""

    def test_resolve_exact_match(self) -> None:
        """Resolve model when it exists exactly in registry."""
        resolved, known = resolve_model("gpt-4o")
        assert resolved == "gpt-4o"
        assert known is True

    def test_resolve_alias(self) -> None:
        """Resolve model via explicit alias."""
        resolved, known = resolve_model("gpt-4-0613")
        assert resolved == "gpt-4"
        assert known is True

    def test_resolve_prefix(self) -> None:
        """Resolve model via prefix matching."""
        resolved, known = resolve_model("gpt-4o-2024-05-13")
        assert resolved == "gpt-4o"
        assert known is True

    def test_resolve_unknown(self) -> None:
        """Handle completely unknown model."""
        resolved, known = resolve_model("mysterious-llm-v1")
        assert resolved == "mysterious-llm-v1"
        assert known is False


class TestEnergyCalculation:
    """Tests for energy consumption calculation."""

    def test_calculate_energy_known_model(self) -> None:
        """Calculate energy for a known model (gpt-4o with prompt-length-aware coefficients)."""
        # gpt-4o now uses Tier 1 data from Jegham et al. (2025)
        # With 1500 total tokens (1000 in + 500 out), uses "medium" category
        # Medium: 0.304 Wh/1k input, 0.911 Wh/1k output
        # Energy = (1000 * 0.304 + 500 * 0.911) / 1000 = (304 + 455.5) / 1000 = 0.7595 Wh
        energy, tier, uncertainty_pct, source, basis, known = calculate_energy(
            1000, 500, "gpt-4o"
        )

        assert energy == pytest.approx(0.7595)
        assert tier == 1
        assert uncertainty_pct == 50  # Tier 1 = ±50%
        assert source == "registry"
        assert known is True
        assert "Jegham" in basis

    def test_calculate_energy_unknown_model(self) -> None:
        """Calculate energy for unknown model using conservative fallback."""
        energy, tier, uncertainty_pct, source, basis, known = calculate_energy(
            1000, 500, "unknown-model"
        )

        # Fallback uses 1.4 in, 4.2 out
        # (1000 * 1.4 + 500 * 4.2) / 1000 = (1400 + 2100) / 1000 = 3.5 Wh
        assert energy == 3.5
        assert tier == 3
        assert uncertainty_pct == 1000
        assert source == "fallback"
        assert known is False

    def test_calculate_energy_override(self) -> None:
        """Use user-provided energy values."""
        override = {
            "wh_per_1k_input": 2.0,
            "wh_per_1k_output": 6.0,
            "tier": 2,
            "source": "benchmarks",
            "basis": "Custom measurement",
        }
        # (1000 * 2.0 + 500 * 6.0) / 1000 = (2000 + 3000) / 1000 = 5.0 Wh
        energy, tier, uncertainty_pct, source, basis, known = calculate_energy(
            1000, 500, "gpt-4o", override
        )

        assert energy == 5.0
        assert tier == 2
        assert uncertainty_pct == 100  # Tier 2 = ±100%
        assert source == "override"
        assert basis == "Custom measurement"


class TestCarbonCalculation:
    """Tests for carbon emissions calculation."""

    def test_calculate_carbon(self) -> None:
        """Calculate carbon from energy and grid intensity."""
        # energy_wh * pue * grid_intensity / 1000
        # 2.0 Wh * 1.1 PUE * 400 gCO2e/kWh / 1000 = 0.88g
        carbon, pue, pue_tier, pue_source = calculate_carbon(2.0, 400.0, pue=1.1)
        assert carbon == pytest.approx(0.88)
        assert pue == 1.1
        assert pue_tier == 1
        assert pue_source == "explicit override"


class TestCostCalculation:
    """Tests for cost estimation."""

    def test_calculate_cost_known_model(self) -> None:
        """Calculate cost for a known model (gpt-4o: $0.005 in, $0.015 out)."""
        # (1000 * 0.005 + 500 * 0.015) / 1000 = (5.0 + 7.5) / 1000 = $0.0125
        total, cost_in, cost_out, tier = calculate_cost(1000, 500, "gpt-4o")

        assert total == pytest.approx(0.0125)
        assert cost_in == pytest.approx(0.005)
        assert cost_out == pytest.approx(0.0075)
        assert tier == "list"

    def test_calculate_cost_unknown_model(self) -> None:
        """Return zero cost for unknown models."""
        total, cost_in, cost_out, tier = calculate_cost(1000, 500, "unknown-model")

        assert total == 0.0
        assert tier == "none"


class TestTokenEstimation:
    """Tests for heuristic token estimation."""

    def test_estimate_tokens_english(self) -> None:
        """English prose uses ~4 chars/token heuristic."""
        text = "Hello world!"  # 12 chars
        # 12 // 4 = 3 tokens
        assert estimate_tokens(text) == 3

    def test_estimate_tokens_cjk(self) -> None:
        """CJK text uses ~1.5 chars/token ratio or tiktoken."""
        text = "你好世界，很高兴见到你。"  # 12 chars (including punctuation)

        # Check if tiktoken is available
        try:
            import tiktoken  # noqa: F401

            # tiktoken returns 15 tokens for this specific string
            expected = 15
        except ImportError:
            # Heuristic: 12 / 1.5 = 8 tokens
            expected = 8

        assert estimate_tokens(text) == expected

    def test_estimate_tokens_code(self) -> None:
        """Code text uses ~3 chars/token ratio or tiktoken."""
        text = "def hello():\n    print('world')"  # High punctuation

        try:
            import tiktoken  # noqa: F401

            # tiktoken returns 8 tokens for this specific string
            expected = 8
        except ImportError:
            # Heuristic: 32 chars // 3 = 10 tokens
            expected = 10

        assert estimate_tokens(text) == expected

    def test_estimate_tokens_empty(self) -> None:
        """Empty or None text returns 0."""
        assert estimate_tokens("") == 0
        assert estimate_tokens(None) == 0


class TestTokenEstimationWithTiktoken:
    """Tests specifically for tiktoken integration."""

    def test_tiktoken_available(self) -> None:
        """Check if tiktoken is available (informational)."""
        try:
            import tiktoken  # noqa: F401

            available = True
        except ImportError:
            available = False
        assert isinstance(available, bool)

    def test_tiktoken_usage_if_installed(self) -> None:
        """If tiktoken is available, estimate_tokens should use it."""
        try:
            import tiktoken  # noqa: F401

            text = "Hello world"
            # tiktoken for "Hello world" gives exactly 2 tokens
            assert estimate_tokens(text) == 2
        except ImportError:
            # Skip if not installed in this environment
            pytest.skip("tiktoken not installed")
