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
        total, cost_in, cost_out, cache_write, cache_read, tier = calculate_cost(
            1000, 500, "gpt-4o"
        )

        assert total == pytest.approx(0.0125)
        assert cost_in == pytest.approx(0.005)
        assert cost_out == pytest.approx(0.0075)
        assert cache_write == 0.0
        assert cache_read == 0.0
        assert tier == "list"

    def test_calculate_cost_unknown_model(self) -> None:
        """Return zero cost for unknown models."""
        total, cost_in, cost_out, cache_write, cache_read, tier = calculate_cost(
            1000, 500, "unknown-model"
        )

        assert total == 0.0
        assert cache_write == 0.0
        assert cache_read == 0.0
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


class TestWaterCalculation:
    """Test water consumption calculations for scientific completeness."""

    def test_calculate_water_for_standard_inference(self) -> None:
        """Water calculation returns reasonable values for standard inference."""
        from vetch.calculation import calculate_water

        # Standard 1Wh inference should use ~1.8L water (data center cooling)
        water_liters = calculate_water(
            energy_wh=1.0,
            model="gpt-4o",
            provider_hint="openai",
            region="us-east-1",
        )

        # Water usage should be positive and in reasonable range (0.5-5 L/Wh)
        assert water_liters > 0.0
        assert water_liters < 10.0  # Sanity check: shouldn't be > 10L per Wh

    def test_water_scales_with_energy(self) -> None:
        """Water consumption scales linearly with energy."""
        from vetch.calculation import calculate_water

        water_1wh = calculate_water(
            energy_wh=1.0, model="gpt-4", provider_hint="openai", region="us-west-2"
        )

        water_10wh = calculate_water(
            energy_wh=10.0, model="gpt-4", provider_hint="openai", region="us-west-2"
        )

        # Should scale approximately linearly
        ratio = water_10wh / water_1wh
        assert 8.0 < ratio < 12.0  # Allow some variance for region differences


class TestEmbodiedCarbonCalculation:
    """Test embodied carbon calculations for lifecycle assessment."""

    def test_calculate_embodied_carbon_for_inference(self) -> None:
        """Embodied carbon calculation accounts for hardware manufacturing."""
        from vetch.calculation import calculate_embodied_carbon

        # Calculate embodied carbon for standard inference (1K input, 100 output tokens)
        embodied_gco2e = calculate_embodied_carbon(
            input_tokens=1000, output_tokens=100, model="gpt-4o"
        )

        # Embodied carbon should be non-zero and reasonable
        # Typical values: 0.01-1.0 gCO2e per 1K tokens
        assert embodied_gco2e > 0.0
        assert embodied_gco2e < 10.0  # Sanity check

    def test_embodied_carbon_scales_with_tokens(self) -> None:
        """Embodied carbon scales with token count."""
        from vetch.calculation import calculate_embodied_carbon

        embodied_small = calculate_embodied_carbon(
            input_tokens=100, output_tokens=10, model="gpt-3.5-turbo"
        )

        embodied_large = calculate_embodied_carbon(
            input_tokens=10000, output_tokens=1000, model="gpt-4"
        )

        # More tokens should have higher embodied carbon
        assert embodied_large > embodied_small

    def test_full_lifecycle_assessment(self) -> None:
        """Complete lifecycle: operational carbon + embodied carbon + water."""
        from vetch.calculation import calculate_carbon, calculate_embodied_carbon, calculate_water

        # Standard inference scenario
        energy_wh = 1.0
        grid_intensity = 400.0  # gCO2e/kWh (typical US average)
        model = "gpt-4o"
        provider = "openai"
        input_tokens = 1000
        output_tokens = 500

        # Operational carbon (grid electricity)
        operational_gco2e, _, _, _ = calculate_carbon(
            energy_wh=energy_wh,
            grid_intensity_gco2e_kwh=grid_intensity,
            model=model,
            provider_hint=provider,
            pue=1.2,
        )

        # Embodied carbon (hardware manufacturing)
        embodied_gco2e = calculate_embodied_carbon(
            input_tokens=input_tokens, output_tokens=output_tokens, model=model
        )

        # Water usage (cooling)
        water_liters = calculate_water(
            energy_wh=energy_wh, model=model, provider_hint=provider, region="us-east-1"
        )

        # All components should contribute to full assessment
        assert operational_gco2e > 0
        assert embodied_gco2e > 0
        assert water_liters > 0

        # Total carbon = operational + embodied
        total_gco2e = operational_gco2e + embodied_gco2e
        # Total should be dominated by operational (typically 80-95%)
        assert operational_gco2e > embodied_gco2e


class TestTieredPricing:
    """Test tiered pricing functionality."""

    def test_calculate_tiered_cost_no_tiers(self) -> None:
        """Test tiered cost calculation without tiers (standard pricing)."""
        from vetch.calculation import _calculate_tiered_cost

        # 100k tokens @ $1.25/M = $0.125
        cost = _calculate_tiered_cost(
            tokens=100000, base_rate_per_1k=0.00125, tier_threshold=None, tier_multiplier=None
        )
        assert abs(cost - 0.125) < 0.0001

    def test_calculate_tiered_cost_under_threshold(self) -> None:
        """Test tiered cost when under threshold (no tier applies)."""
        from vetch.calculation import _calculate_tiered_cost

        # 100k tokens @ $1.25/M, threshold 128k, multiplier 2x
        # Should use base rate only
        cost = _calculate_tiered_cost(
            tokens=100000, base_rate_per_1k=0.00125, tier_threshold=128000, tier_multiplier=2.0
        )
        assert abs(cost - 0.125) < 0.0001

    def test_calculate_tiered_cost_at_threshold(self) -> None:
        """Test tiered cost exactly at threshold."""
        from vetch.calculation import _calculate_tiered_cost

        # 128k tokens @ $1.25/M = $0.16
        cost = _calculate_tiered_cost(
            tokens=128000, base_rate_per_1k=0.00125, tier_threshold=128000, tier_multiplier=2.0
        )
        assert abs(cost - 0.16) < 0.0001

    def test_calculate_tiered_cost_over_threshold(self) -> None:
        """Test tiered cost when over threshold (split calculation)."""
        from vetch.calculation import _calculate_tiered_cost

        # 200k tokens @ $1.25/M base, 2x over 128k
        # Base tier: 128k @ $1.25/M = $0.16
        # Over tier: 72k @ $2.50/M = $0.18
        # Total: $0.34
        cost = _calculate_tiered_cost(
            tokens=200000, base_rate_per_1k=0.00125, tier_threshold=128000, tier_multiplier=2.0
        )
        assert abs(cost - 0.34) < 0.0001

    def test_tiered_pricing_with_output_tokens(self) -> None:
        """Test tiered pricing applies to both input and output tokens."""
        from vetch.calculation import _calculate_tiered_cost

        # Output tokens should also use tiered pricing
        # 200k output @ $10/M base, 2x over 128k
        # Base: 128k @ $10/M = $1.28
        # Over: 72k @ $20/M = $1.44
        # Total: $2.72
        cost = _calculate_tiered_cost(
            tokens=200000, base_rate_per_1k=0.010, tier_threshold=128000, tier_multiplier=2.0
        )
        assert abs(cost - 2.72) < 0.01

    def test_calculate_tiered_cost_zero_tokens(self) -> None:
        """Test tiered cost with zero tokens."""
        from vetch.calculation import _calculate_tiered_cost

        # Zero tokens should cost $0
        cost = _calculate_tiered_cost(
            tokens=0, base_rate_per_1k=0.00125, tier_threshold=128000, tier_multiplier=2.0
        )
        assert cost == 0.0

    def test_calculate_tiered_cost_with_none_threshold_only(self) -> None:
        """Test tiered cost when only threshold is None."""
        from vetch.calculation import _calculate_tiered_cost

        # None threshold but non-None multiplier should use flat rate
        cost = _calculate_tiered_cost(
            tokens=100000, base_rate_per_1k=0.00125, tier_threshold=None, tier_multiplier=2.0
        )
        assert abs(cost - 0.125) < 0.0001
