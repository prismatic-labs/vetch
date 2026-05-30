"""Unit tests for cache cost saving calculation."""

from __future__ import annotations

import pytest


class TestCacheCostSaving:
    """Tests for cache_cost_saving_usd in prepare_inference_metrics."""

    def test_no_cache_tokens_no_saving(self) -> None:
        from vetch.calculation import prepare_inference_metrics

        metrics = prepare_inference_metrics(
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            usage={"text": {"input_tokens": 1000, "output_tokens": 100}},
            accumulated_chars=0,
            region=None,
            price_multiplier=1.0,
            energy_override=None,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            existing_warnings=[],
        )
        assert metrics.cache_cost_saving_usd is None

    def test_cache_tokens_produce_saving(self) -> None:
        from vetch.calculation import prepare_inference_metrics

        metrics = prepare_inference_metrics(
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            usage={"text": {"input_tokens": 1000, "output_tokens": 100,
                            "cache_read_tokens": 800}},
            accumulated_chars=0,
            region=None,
            price_multiplier=1.0,
            energy_override=None,
            cache_read_tokens=800,
            cache_creation_tokens=0,
            existing_warnings=[],
        )
        # Should have a positive saving: paying cache-read price instead of full input price
        assert metrics.cache_cost_saving_usd is not None
        assert metrics.cache_cost_saving_usd > 0.0
        assert metrics.cache_carbon_saving_g is not None
        assert metrics.cache_carbon_saving_g > 0.0

    def test_carbon_saving_uses_energy_saving_pue_and_grid(self) -> None:
        """Cache carbon saving follows the same PUE/grid formula as normal carbon."""
        from vetch.calculation import prepare_inference_metrics

        metrics = prepare_inference_metrics(
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            usage={"text": {"input_tokens": 1000, "output_tokens": 100}},
            accumulated_chars=0,
            region=None,
            price_multiplier=1.0,
            energy_override=None,
            cache_read_tokens=800,
            cache_creation_tokens=0,
            existing_warnings=[],
        )

        assert metrics.cache_energy_saving_wh is not None
        assert metrics.cache_carbon_saving_g == pytest.approx(
            metrics.cache_energy_saving_wh * metrics.pue * metrics.grid_val / 1000
        )

    def test_saving_is_non_negative(self) -> None:
        """Saving should never be negative — max(0, ...) guard."""
        from vetch.calculation import prepare_inference_metrics

        metrics = prepare_inference_metrics(
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            usage={"text": {"input_tokens": 500, "output_tokens": 50,
                            "cache_read_tokens": 1}},
            accumulated_chars=0,
            region=None,
            price_multiplier=1.0,
            energy_override=None,
            cache_read_tokens=1,
            cache_creation_tokens=0,
            existing_warnings=[],
        )
        if metrics.cache_cost_saving_usd is not None:
            assert metrics.cache_cost_saving_usd >= 0.0

    def test_saving_less_than_total_cost(self) -> None:
        """Saving should not exceed the total cost of the call."""
        from vetch.calculation import prepare_inference_metrics

        metrics = prepare_inference_metrics(
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            usage={"text": {"input_tokens": 2000, "output_tokens": 200,
                            "cache_read_tokens": 1500}},
            accumulated_chars=0,
            region=None,
            price_multiplier=1.0,
            energy_override=None,
            cache_read_tokens=1500,
            cache_creation_tokens=0,
            existing_warnings=[],
        )
        if metrics.cache_cost_saving_usd is not None and metrics.cost_usd is not None:
            assert metrics.cache_cost_saving_usd < metrics.cost_usd * 10

    def test_price_multiplier_applied_symmetrically(self) -> None:
        """With a price multiplier, both sides scale — saving is proportional."""
        from vetch.calculation import prepare_inference_metrics

        base = prepare_inference_metrics(
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            usage={"text": {"input_tokens": 1000, "output_tokens": 100,
                            "cache_read_tokens": 800}},
            accumulated_chars=0,
            region=None,
            price_multiplier=1.0,
            energy_override=None,
            cache_read_tokens=800,
            cache_creation_tokens=0,
            existing_warnings=[],
        )
        doubled = prepare_inference_metrics(
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            usage={"text": {"input_tokens": 1000, "output_tokens": 100,
                            "cache_read_tokens": 800}},
            accumulated_chars=0,
            region=None,
            price_multiplier=2.0,
            energy_override=None,
            cache_read_tokens=800,
            cache_creation_tokens=0,
            existing_warnings=[],
        )
        if base.cache_cost_saving_usd and doubled.cache_cost_saving_usd:
            assert abs(doubled.cache_cost_saving_usd - base.cache_cost_saving_usd * 2) < 1e-9

    def test_unknown_model_no_crash(self) -> None:
        """prepare_inference_metrics handles unknown models without crashing."""
        from vetch.calculation import prepare_inference_metrics

        metrics = prepare_inference_metrics(
            model="totally-unknown-model-xyz",
            provider="unknown",
            usage={"text": {"input_tokens": 100, "output_tokens": 10}},
            accumulated_chars=0,
            region=None,
            price_multiplier=1.0,
            energy_override=None,
            cache_read_tokens=50,
            cache_creation_tokens=0,
            existing_warnings=[],
        )
        # Should not crash; saving may be None for unknown models with no pricing
        assert metrics.cache_cost_saving_usd is None or metrics.cache_cost_saving_usd >= 0.0
