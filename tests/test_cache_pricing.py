"""Tests for cache-aware cost calculation."""

from __future__ import annotations

import pytest

from vetch.calculation import _reset_registries, calculate_cost

# Index of cache_write_cost in the calculate_cost return tuple:
# (total, input_cost, output_cost, cache_write_cost, cache_read_cost, tier)
_WRITE = 3


class TestCacheAwarePricing:
    """Tests for cache-aware cost calculation."""

    def setup_method(self) -> None:
        """Reset registries before each test."""
        _reset_registries()

    def teardown_method(self) -> None:
        """Reset registries after each test."""
        _reset_registries()

    def test_basic_cost_unchanged_without_cache(self) -> None:
        """Cost calculation without cache tokens is unchanged."""
        total, input_cost, output_cost, cache_write, cache_read, tier = calculate_cost(
            1000, 500, "claude-3.5-sonnet"
        )
        assert total > 0
        assert tier == "list"
        assert cache_write == 0.0
        assert cache_read == 0.0

    def test_cache_read_reduces_cost(self) -> None:
        """Cache read tokens reduce input cost."""
        # Without cache
        total_no_cache, _, _, _, _, _ = calculate_cost(
            1000, 500, "claude-3.5-sonnet"
        )

        # With cache (500 of 1000 input tokens from cache)
        total_with_cache, _, _, _, _, _ = calculate_cost(
            1000, 500, "claude-3.5-sonnet", cache_read_tokens=500
        )

        # Cache should reduce cost (Anthropic cache read is 10% of input price)
        assert total_with_cache < total_no_cache

    def test_cache_read_all_tokens(self) -> None:
        """All input tokens from cache gives maximum discount."""
        total_no_cache, _, _, _, _, _ = calculate_cost(
            1000, 500, "claude-3.5-sonnet"
        )

        total_all_cache, _, _, _, _, _ = calculate_cost(
            1000, 500, "claude-3.5-sonnet", cache_read_tokens=1000
        )

        # With all tokens cached at 10%, input cost should be ~10% of original
        assert total_all_cache < total_no_cache

    def test_cache_creation_adds_cost(self) -> None:
        """Cache creation tokens add premium to cost."""
        total_no_cache, _, _, _, _, _ = calculate_cost(
            1000, 500, "claude-3.5-sonnet"
        )

        total_with_creation, _, _, _, _, _ = calculate_cost(
            1000, 500, "claude-3.5-sonnet", cache_creation_tokens=500
        )

        # Cache creation adds 25% premium on those tokens
        assert total_with_creation > total_no_cache

    def test_openai_cache_discount_different(self) -> None:
        """OpenAI has different cache discount (50%)."""
        total_no_cache, _, _, _, _, _ = calculate_cost(
            1000, 500, "gpt-4o"
        )

        total_with_cache, _, _, _, _, _ = calculate_cost(
            1000, 500, "gpt-4o", cache_read_tokens=500
        )

        # GPT-4o cache read is 50% discount (not 90%)
        assert total_with_cache < total_no_cache

    def test_unknown_model_returns_zero(self) -> None:
        """Unknown model returns zero cost even with cache tokens."""
        total, _, _, _, _, tier = calculate_cost(
            1000, 500, "unknown-model",
            cache_read_tokens=100, cache_creation_tokens=100
        )
        assert total == 0.0
        assert tier == "none"

    def test_zero_cache_tokens_same_as_none(self) -> None:
        """Zero cache tokens produces same result as None."""
        total_none, _, _, _, _, _ = calculate_cost(1000, 500, "claude-3.5-sonnet")
        total_zero, _, _, _, _, _ = calculate_cost(
            1000, 500, "claude-3.5-sonnet",
            cache_read_tokens=0, cache_creation_tokens=0
        )
        assert total_none == total_zero

    def test_negative_cache_tokens_ignored(self) -> None:
        """Negative cache tokens are treated as if not provided."""
        total_base, _, _, _, _, _ = calculate_cost(1000, 500, "claude-3.5-sonnet")
        total_neg, _, _, _, _, _ = calculate_cost(
            1000, 500, "claude-3.5-sonnet",
            cache_read_tokens=-100
        )
        # Negative cache_read_tokens should not trigger discount
        # (the conditional checks for > 0)
        assert total_neg == total_base


class TestCacheWriteTTLTiering:
    """Cache writes must be priced by TTL: Anthropic 5-min = 1.25x, 1-hour = 2.0x."""

    RATE_IN = 0.003  # claude-3.5-sonnet usd_per_1k_input

    def setup_method(self) -> None:
        _reset_registries()

    def teardown_method(self) -> None:
        _reset_registries()

    def test_1h_write_costs_more_than_5m(self) -> None:
        """A 1-hour write is priced strictly above the same 5-minute write."""
        w5m = calculate_cost(
            1000, 100, "claude-3.5-sonnet", cache_creation_tokens=1000
        )[_WRITE]
        w1h = calculate_cost(
            1000, 100, "claude-3.5-sonnet",
            cache_creation_tokens=1000, cache_creation_1h_tokens=1000,
        )[_WRITE]
        assert w1h > w5m
        # 2.0x vs 1.25x premium → exact 1.6x ratio
        assert w1h == pytest.approx(w5m * (2.0 / 1.25))

    def test_unspecified_1h_defaults_to_5m_pricing(self) -> None:
        """Legacy behavior: no 1h count prices every write at the 5-minute premium."""
        w_default = calculate_cost(
            1000, 100, "claude-3.5-sonnet", cache_creation_tokens=1000
        )[_WRITE]
        w_zero_1h = calculate_cost(
            1000, 100, "claude-3.5-sonnet",
            cache_creation_tokens=1000, cache_creation_1h_tokens=0,
        )[_WRITE]
        assert w_default == pytest.approx(w_zero_1h)
        assert w_default == pytest.approx(1000 * self.RATE_IN * 1.25 / 1000)

    def test_mixed_ttl_split_pricing(self) -> None:
        """A mixed write bills the 1h subset at 2.0x and the remainder at 1.25x."""
        write = calculate_cost(
            1000, 100, "claude-3.5-sonnet",
            cache_creation_tokens=1000, cache_creation_1h_tokens=400,
        )[_WRITE]
        expected = (600 * self.RATE_IN * 1.25 + 400 * self.RATE_IN * 2.0) / 1000
        assert write == pytest.approx(expected)

    def test_1h_count_clamped_to_total(self) -> None:
        """A 1h count exceeding the total write is clamped, not double-counted."""
        clamped = calculate_cost(
            1000, 100, "claude-3.5-sonnet",
            cache_creation_tokens=500, cache_creation_1h_tokens=9999,
        )[_WRITE]
        all_1h = calculate_cost(
            1000, 100, "claude-3.5-sonnet",
            cache_creation_tokens=500, cache_creation_1h_tokens=500,
        )[_WRITE]
        assert clamped == pytest.approx(all_1h)
