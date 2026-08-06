"""Regression tests for provider cache-token conventions in cost accounting.

The cost/energy math assumes the OpenAI convention where ``input_tokens`` already
includes cache-read tokens (cached is a subset of the prompt). Anthropic reports
them disjoint: ``usage.input_tokens`` is the fresh, uncached count and
``cache_read_input_tokens`` is separate. Before the normalization fix, feeding
Anthropic's disjoint counts into the subtraction zeroed out the fresh-input cost
whenever ``cache_read >= input_tokens`` — the norm for agentic / Claude Code
traffic. These tests pin the corrected behavior.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from vetch.calculation import _reset_registries, prepare_inference_metrics

# claude-3.5-sonnet list price (usd per 1k input tokens) and cache read discount.
RATE_IN = 0.003
READ_DISCOUNT = 0.1
MODEL = "claude-3-5-sonnet-20241022"


def _anthropic_metrics(fresh: int, cache_read: int, cache_creation: int = 0):
    return prepare_inference_metrics(
        model=MODEL,
        provider="anthropic",
        usage={"text": {"input_tokens": fresh, "output_tokens": 100}},
        accumulated_chars=0,
        region=None,
        price_multiplier=1.0,
        energy_override=None,
        cache_read_tokens=cache_read,
        cache_creation_tokens=cache_creation,
        existing_warnings=[],
    )


class TestAnthropicDisjointConvention:
    def setup_method(self) -> None:
        _reset_registries()

    def teardown_method(self) -> None:
        _reset_registries()

    def test_fresh_input_not_zeroed_by_larger_cache_read(self) -> None:
        """Fresh input is billed at full rate even when cache_read >> input_tokens.

        This is the exact regression: 1000 fresh tokens with 4000 cache reads.
        Under the old subtraction, effective_input clamped to 0 and the fresh
        1000 tokens were billed at $0.
        """
        m = _anthropic_metrics(fresh=1000, cache_read=4000)
        assert m.cost_in_usd is not None
        expected_fresh = 1000 * RATE_IN / 1000  # 0.003
        expected_read = 4000 * RATE_IN * READ_DISCOUNT / 1000  # 0.0012
        assert m.cost_in_usd == pytest.approx(expected_fresh + expected_read)
        # Fresh cost must be present, not swallowed by the cache-read discount.
        assert m.cost_in_usd > expected_read

    def test_all_input_fresh_no_cache_matches_plain_pricing(self) -> None:
        """With no cache reads, cost is just fresh input at full rate."""
        m = _anthropic_metrics(fresh=1000, cache_read=0)
        assert m.cost_in_usd is not None
        assert m.cost_in_usd == pytest.approx(1000 * RATE_IN / 1000)

    def test_cache_read_still_discounts_relative_to_uncached(self) -> None:
        """A cached call is cheaper than the same tokens billed entirely fresh."""
        cached = _anthropic_metrics(fresh=1000, cache_read=4000)
        assert cached.cost_in_usd is not None
        # Baseline: all 5000 tokens fresh (no discount).
        baseline_in = 5000 * RATE_IN / 1000
        assert cached.cost_in_usd < baseline_in

    @settings(max_examples=50, deadline=None)
    @given(
        fresh=st.integers(min_value=1, max_value=100_000),
        cache_read=st.integers(min_value=0, max_value=500_000),
    )
    def test_fresh_input_never_free_invariant(self, fresh: int, cache_read: int) -> None:
        """Invariant: fresh input tokens are always billed at least at full rate.

        This single property would have caught the original bug for any token split.
        """
        _reset_registries()
        m = _anthropic_metrics(fresh=fresh, cache_read=cache_read)
        assert m.cost_in_usd is not None
        floor = fresh * RATE_IN / 1000
        assert m.cost_in_usd >= floor - 1e-9


class TestOpenAIInclusiveConventionUnchanged:
    """The fix must not alter inclusive-convention providers (OpenAI)."""

    def setup_method(self) -> None:
        _reset_registries()

    def teardown_method(self) -> None:
        _reset_registries()

    def test_openai_input_includes_cache_read_no_double_add(self) -> None:
        """For OpenAI, input_tokens already includes cache reads; no re-adding.

        1000 prompt tokens fully cached must cost less than 1000 fresh tokens,
        proving cache_read is still subtracted (not added back) for OpenAI.
        """
        cached = prepare_inference_metrics(
            model="gpt-4o",
            provider="openai",
            usage={"text": {"input_tokens": 1000, "output_tokens": 100}},
            accumulated_chars=0,
            region=None,
            price_multiplier=1.0,
            energy_override=None,
            cache_read_tokens=1000,
            cache_creation_tokens=0,
            existing_warnings=[],
        )
        uncached = prepare_inference_metrics(
            model="gpt-4o",
            provider="openai",
            usage={"text": {"input_tokens": 1000, "output_tokens": 100}},
            accumulated_chars=0,
            region=None,
            price_multiplier=1.0,
            energy_override=None,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            existing_warnings=[],
        )
        assert cached.cost_in_usd is not None and uncached.cost_in_usd is not None
        assert cached.cost_in_usd < uncached.cost_in_usd
