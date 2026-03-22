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

        # 200k tokens @ $1.25/M base, 2x over 128k (THRESHOLD pricing)
        # Since 200k > 128k threshold: ALL 200k @ $2.50/M (base * multiplier)
        # Total: 200k * $0.0025/1k = $0.50
        cost = _calculate_tiered_cost(
            tokens=200000, base_rate_per_1k=0.00125, tier_threshold=128000, tier_multiplier=2.0
        )
        assert abs(cost - 0.50) < 0.0001

    def test_tiered_pricing_with_output_tokens(self) -> None:
        """Test tiered pricing applies to both input and output tokens."""
        from vetch.calculation import _calculate_tiered_cost

        # Output tokens should also use threshold-based tiered pricing
        # 200k output @ $10/M base, 2x over 128k (THRESHOLD pricing)
        # Since 200k > 128k threshold: ALL 200k @ $20/M (base * multiplier)
        # Total: 200k * $0.020/1k = $4.00
        cost = _calculate_tiered_cost(
            tokens=200000, base_rate_per_1k=0.010, tier_threshold=128000, tier_multiplier=2.0
        )
        assert abs(cost - 4.00) < 0.01

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


class TestV024RegistryUpdates:
    """Tests for v0.2.4 energy registry upgrades and new entries."""

    def test_claude_37_sonnet_prompt_length_aware(self) -> None:
        """claude-3.7-sonnet should have different energy per token at short vs long prompts."""
        short_e, _, _, _, _, _ = calculate_energy(100, 300, "claude-3.7-sonnet")
        long_e, _, _, _, _, _ = calculate_energy(10000, 1500, "claude-3.7-sonnet")
        assert short_e is not None and long_e is not None
        # short per-token cost should be higher than long per-token cost
        short_per_token = short_e / (100 + 300)
        long_per_token = long_e / (10000 + 1500)
        assert short_per_token > long_per_token

    def test_claude_37_sonnet_thinking_higher_energy(self) -> None:
        """Extended Thinking variant should consume more energy than standard."""
        standard_e, _, _, _, _, known_s = calculate_energy(1000, 1000, "claude-3.7-sonnet")
        thinking_e, _, _, _, _, known_t = calculate_energy(
            1000, 1000, "claude-3.7-sonnet-thinking"
        )
        assert known_s is True
        assert known_t is True
        assert standard_e is not None and thinking_e is not None
        assert thinking_e > standard_e

    def test_o3_mini_tier1(self) -> None:
        """o3-mini should be a Tier 1 entry with reasonable energy."""
        energy, tier, _, _, _, known = calculate_energy(1000, 1000, "o3-mini")
        assert known is True
        assert tier == 1
        assert energy is not None and energy > 0

    def test_gpt41_prompt_length_aware(self) -> None:
        """gpt-4.1 should be prompt-length-aware (short != long)."""
        short_e, _, _, _, _, _ = calculate_energy(100, 300, "gpt-4.1")
        long_e, _, _, _, _, _ = calculate_energy(10000, 1500, "gpt-4.1")
        assert short_e is not None and long_e is not None
        assert short_e != long_e

    def test_deepseek_v3_tier1(self) -> None:
        """deepseek-v3 should be a Tier 1 entry."""
        energy, tier, _, _, _, known = calculate_energy(1000, 1000, "deepseek-v3")
        assert known is True
        assert tier == 1
        assert energy is not None and energy > 0

    def test_llama_33_more_efficient_than_31(self) -> None:
        """llama-3.3-70b should be significantly more efficient than llama-3.1-70b."""
        e31, _, _, _, _, _ = calculate_energy(1000, 1000, "llama-3.1-70b")
        e33, _, _, _, _, _ = calculate_energy(1000, 1000, "llama-3.3-70b")
        assert e31 is not None and e33 is not None
        # Jegham: 3.3 is ~4x more efficient
        assert e33 < e31 * 0.5

    def test_claude_35_haiku_separate_from_claude_3_haiku(self) -> None:
        """claude-3-5-haiku alias should resolve to claude-3.5-haiku, not claude-3-haiku."""
        model_35, _ = resolve_model("claude-3-5-haiku")
        model_3, _ = resolve_model("claude-3-haiku")
        assert model_35 != model_3
        assert model_35 == "claude-3.5-haiku"

    def test_gpt4_turbo_upgraded_to_tier1(self) -> None:
        """gpt-4-turbo should now be Tier 1 (upgraded from Tier 3)."""
        _, tier, _, _, _, known = calculate_energy(1000, 1000, "gpt-4-turbo")
        assert known is True
        assert tier == 1

    def test_gpt4o_mini_alias_corrected(self) -> None:
        """gpt-4o-mini energy should be much lower than gpt-4o (alias fix)."""
        mini_e, _, _, _, _, _ = calculate_energy(1000, 1000, "gpt-4o-mini")
        full_e, _, _, _, _, _ = calculate_energy(1000, 1000, "gpt-4o")
        assert mini_e is not None and full_e is not None
        # gpt-4o-mini is Tier 3 (~0.10+0.30 per-1k), gpt-4o is Tier 1 (~0.304+0.911 per-1k)
        assert mini_e < full_e

    def test_pricing_backfill_o1(self) -> None:
        """o1 should now have non-null cost estimate."""
        cost, *_ = calculate_cost(1000, 1000, "o1")
        assert cost is not None and cost > 0

    def test_pricing_backfill_o3(self) -> None:
        """o3 should now have non-null cost estimate."""
        cost, *_ = calculate_cost(1000, 1000, "o3")
        assert cost is not None and cost > 0

    def test_pricing_backfill_claude_37_sonnet(self) -> None:
        """claude-3.7-sonnet should now have non-null cost estimate."""
        cost, *_ = calculate_cost(1000, 1000, "claude-3.7-sonnet")
        assert cost is not None and cost > 0

    def test_pricing_backfill_gpt41(self) -> None:
        """gpt-4.1 should have non-null cost estimate."""
        cost, *_ = calculate_cost(1000, 1000, "gpt-4.1")
        assert cost is not None and cost > 0


class TestV024TokenizationFallback:
    """Tests for v0.2.4 script-aware token estimation improvements."""

    def test_prepare_metrics_tiktoken_fallback(self) -> None:
        """When accumulated_tik_tokens > 0 and no usage, use tiktoken count."""
        from vetch.calculation import prepare_inference_metrics

        metrics = prepare_inference_metrics(
            model="gpt-4o",
            provider="openai",
            usage=None,
            accumulated_chars=400,
            region=None,
            price_multiplier=1.0,
            energy_override=None,
            cache_read_tokens=None,
            cache_creation_tokens=None,
            existing_warnings=[],
            accumulated_tik_tokens=100,
            content_type_hint="en",
        )
        assert metrics.usage_estimated is True
        assert metrics.usage_estimation_method == "tiktoken"
        assert metrics.usage is not None
        assert metrics.usage["text"]["output_tokens"] == 100  # type: ignore[index]

    def test_prepare_metrics_cjk_ratio(self) -> None:
        """CJK content_type_hint should use 1.5 chars/token ratio."""
        from vetch.calculation import prepare_inference_metrics

        metrics = prepare_inference_metrics(
            model="gpt-4o",
            provider="openai",
            usage=None,
            accumulated_chars=300,
            region=None,
            price_multiplier=1.0,
            energy_override=None,
            cache_read_tokens=None,
            cache_creation_tokens=None,
            existing_warnings=[],
            accumulated_tik_tokens=0,
            content_type_hint="cjk",
        )
        assert metrics.usage_estimated is True
        assert metrics.usage is not None
        # 300 chars / 1.5 = 200 output tokens
        assert metrics.usage["text"]["output_tokens"] == 200  # type: ignore[index]

    def test_prepare_metrics_japanese_ratio(self) -> None:
        """Japanese content_type_hint should use 1.7 chars/token ratio."""
        from vetch.calculation import prepare_inference_metrics

        metrics = prepare_inference_metrics(
            model="gpt-4o",
            provider="openai",
            usage=None,
            accumulated_chars=170,
            region=None,
            price_multiplier=1.0,
            energy_override=None,
            cache_read_tokens=None,
            cache_creation_tokens=None,
            existing_warnings=[],
            accumulated_tik_tokens=0,
            content_type_hint="ja",
        )
        assert metrics.usage_estimated is True
        assert metrics.usage is not None
        # 170 chars / 1.7 = 100 output tokens
        assert metrics.usage["text"]["output_tokens"] == 100  # type: ignore[index]

    def test_uncertainty_floor_when_usage_estimated(self) -> None:
        """When token counts are estimated, uncertainty should be floored at 50%."""
        from vetch.calculation import prepare_inference_metrics

        metrics = prepare_inference_metrics(
            model="gpt-4o",  # Tier 1 = 50% normally
            provider="openai",
            usage=None,
            accumulated_chars=400,
            region=None,
            price_multiplier=1.0,
            energy_override=None,
            cache_read_tokens=None,
            cache_creation_tokens=None,
            existing_warnings=[],
            accumulated_tik_tokens=0,
            content_type_hint="en",
        )
        assert metrics.usage_estimated is True
        assert metrics.energy_uncertainty_pct is not None
        assert metrics.energy_uncertainty_pct >= 50


class TestV024CacheEnergyDiscount:
    """Tests for v0.2.4 cache-hit energy discounting."""

    def test_cache_read_tokens_reduce_energy(self) -> None:
        """cache_read_tokens should reduce energy vs. uncached baseline."""
        full_energy, *_ = calculate_energy(1000, 500, "gpt-4o")
        cached_energy, *_ = calculate_energy(1000, 500, "gpt-4o", cache_read_tokens=800)
        assert full_energy is not None
        assert cached_energy is not None
        assert cached_energy < full_energy

    def test_cache_discount_factor_approx_15pct(self) -> None:
        """Cache reads should cost ~15% of normal prefill energy."""
        # 1000 input tokens, all from cache, no output
        full_energy, *_ = calculate_energy(1000, 0, "gpt-4o")
        cached_energy, *_ = calculate_energy(1000, 0, "gpt-4o", cache_read_tokens=1000)
        assert full_energy is not None and full_energy > 0
        assert cached_energy is not None and cached_energy > 0
        # Cached should be ~15% of full prefill
        ratio = cached_energy / full_energy
        assert 0.10 <= ratio <= 0.25, f"Expected ~0.15, got {ratio:.3f}"

    def test_zero_cache_tokens_no_discount(self) -> None:
        """cache_read_tokens=0 should give same result as no cache arg."""
        energy_no_arg, *_ = calculate_energy(1000, 500, "gpt-4o")
        energy_zero, *_ = calculate_energy(1000, 500, "gpt-4o", cache_read_tokens=0)
        assert energy_no_arg == energy_zero

    def test_cache_tokens_capped_at_input_tokens(self) -> None:
        """cache_read_tokens exceeding in_tokens should be clamped."""
        normal, *_ = calculate_energy(500, 200, "gpt-4o", cache_read_tokens=500)
        excess, *_ = calculate_energy(500, 200, "gpt-4o", cache_read_tokens=9999)
        assert normal == excess

    def test_prepare_metrics_cache_saving_populated(self) -> None:
        """prepare_inference_metrics should populate cache_energy_saving_wh."""
        from vetch.calculation import prepare_inference_metrics

        usage = {"text": {"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500}}
        metrics = prepare_inference_metrics(
            model="gpt-4o",
            provider="openai",
            usage=usage,
            accumulated_chars=0,
            region=None,
            price_multiplier=1.0,
            energy_override=None,
            cache_read_tokens=800,
            cache_creation_tokens=None,
            existing_warnings=[],
        )
        assert metrics.cache_energy_saving_wh is not None
        assert metrics.cache_energy_saving_wh > 0

    def test_prepare_metrics_no_cache_saving_when_no_cache(self) -> None:
        """With no cache_read_tokens, cache_energy_saving_wh should be None."""
        from vetch.calculation import prepare_inference_metrics

        usage = {"text": {"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500}}
        metrics = prepare_inference_metrics(
            model="gpt-4o",
            provider="openai",
            usage=usage,
            accumulated_chars=0,
            region=None,
            price_multiplier=1.0,
            energy_override=None,
            cache_read_tokens=None,
            cache_creation_tokens=None,
            existing_warnings=[],
        )
        assert metrics.cache_energy_saving_wh is None
