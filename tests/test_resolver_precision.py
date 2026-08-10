"""Tests for resolve_model_match precision tiers and the uncertainty floor (B2).

Covers the golden precision table, case-insensitivity, the conservative family
fallback, the prefix/family Tier-3 downgrade, and property-based invariants that
protect numerical credibility (energy monotonicity, uncertainty ordering). The
"never OpenAI-priced for a non-OpenAI provider" invariant lives in
test_self_hosted_routing.py, where the cost path is exercised.
"""

from __future__ import annotations

import pytest

from vetch.calculation import (
    calculate_energy,
    get_uncertainty_pct,
    resolve_model,
    resolve_model_match,
)

# Skip the whole module (rather than erroring at collection) when the optional
# hypothesis dependency is absent.
pytest.importorskip("hypothesis")

from hypothesis import given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

# model id -> expected precision rung
GOLDEN_PRECISION = {
    "gpt-4o": "exact",
    "gemini-3.1-pro": "exact",
    "claude-sonnet-4-6": "exact",
    "gpt-4o-2024-05-13": "alias",
    "gemini-3-flash-preview": "alias",
    "claude-sonnet-4-6-experimental": "prefix",  # not a curated alias -> algorithmic shorten
    "gemini-9-ultra": "family",  # unknown google -> family proxy
    "gemma-4-31b-it": "exact",  # first-class self-hosted registry row
    "gemma-9-31b-it": "family",  # unknown google open model -> family proxy (31b -> large)
    "totally-unknown-model-xyz": "fallback",
}


class TestPrecisionGoldenTable:
    @pytest.mark.parametrize("model,expected", GOLDEN_PRECISION.items())
    def test_precision(self, model: str, expected: str) -> None:
        assert resolve_model_match(model).precision == expected

    def test_known_flag_only_false_for_fallback(self) -> None:
        for model, expected in GOLDEN_PRECISION.items():
            match = resolve_model_match(model)
            assert match.known == (expected != "fallback")


class TestCaseInsensitivity:
    @pytest.mark.parametrize("model", ["GPT-4O", "Gpt-4O", "gPt-4o"])
    def test_uppercase_resolves_like_lowercase(self, model: str) -> None:
        assert resolve_model_match(model).name == "gpt-4o"
        assert resolve_model_match(model).precision == "exact"

    def test_mixed_case_family(self) -> None:
        # Regression: "31b" must not match the "1b" small hint; mid-size biases large.
        # Uses an UNKNOWN gemma (gemma-4-31b-it is now a first-class row) so the
        # family-proxy heuristic is still exercised.
        match = resolve_model_match("Gemma-9-31b-it")
        assert match.precision == "family"
        assert match.name == "gemini-3.1-pro"  # conservative (larger) representative

    def test_back_compat_wrapper(self) -> None:
        resolved, known = resolve_model("GPT-4O")
        assert (resolved, known) == ("gpt-4o", True)


class TestUncertaintyFloor:
    def test_exact_keeps_measured_tier(self) -> None:
        _, tier, unc, *_ = calculate_energy(1000, 500, "gpt-4o")
        assert tier == 1  # gpt-4o is Tier 1 (inferred)
        assert unc == get_uncertainty_pct(1)

    def test_prefix_proxy_into_tier1_is_floored_to_tier3(self) -> None:
        # prefix-shortens to gpt-4o (Tier 1) but must not inherit that confidence
        _, tier, unc, _, basis, known = calculate_energy(1000, 500, "gpt-4o-frontier-2099")
        assert known is True
        assert tier == 3
        assert unc == get_uncertainty_pct(3)
        assert "Proxy match" in basis

    def test_family_proxy_is_tier3(self) -> None:
        _, tier, _, _, _, known = calculate_energy(1000, 500, "gemini-9-ultra")
        assert known is True
        assert tier == 3


class TestConservativeFamilyFallback:
    def test_large_hint_picks_frontier(self) -> None:
        assert resolve_model_match("gemini-9-pro").name == "gemini-3.1-pro"

    def test_small_hint_picks_small(self) -> None:
        assert resolve_model_match("gemini-9-flash-lite").name == "gemini-3-flash"

    def test_ambiguous_biases_large(self) -> None:
        # No tier keyword, no size -> default == frontier (never undercount)
        assert resolve_model_match("claude-9").name == "claude-sonnet-4-6"

    def test_unknown_family_is_fallback(self) -> None:
        # qwen has no provider family mapping
        assert resolve_model_match("qwen-72b").precision == "fallback"


class TestDottedVersionDegrade:
    """A dotted minor-version bump with no exact/alias entry degrades to its
    major-version sibling when that sibling exists, rather than jumping to a
    generic family default. Covers multiple vendors and identifier shapes so
    this remains a resolver invariant, not a model-specific accommodation.
    """

    @pytest.mark.parametrize(
        "model,expected",
        [
            ("gpt-5.9-experimental", "gpt-5"),
            ("gpt-5.8", "gpt-5"),
            ("claude-sonnet-4.7-experimental", "claude-sonnet-4"),
            ("deepseek-r1.5", "deepseek-r1"),
        ],
    )
    def test_dotted_bump_degrades_to_existing_major(
        self, model: str, expected: str
    ) -> None:
        match = resolve_model_match(model)
        assert (match.name, match.precision) == (expected, "prefix")

    def test_does_not_fall_through_to_family_default(self) -> None:
        assert resolve_model_match("gpt-5.9-experimental").name != "gpt-4o"


class TestPropertyInvariants:
    @given(
        in_tok=st.integers(min_value=0, max_value=200_000),
        extra_out=st.integers(min_value=1, max_value=50_000),
    )
    def test_energy_monotonic_in_output(self, in_tok: int, extra_out: int) -> None:
        e_low, *_ = calculate_energy(in_tok, 100, "gpt-4o")
        e_high, *_ = calculate_energy(in_tok, 100 + extra_out, "gpt-4o")
        assert e_high >= e_low

    @given(extra_in=st.integers(min_value=1, max_value=50_000))
    def test_energy_monotonic_in_input(self, extra_in: int) -> None:
        e_low, *_ = calculate_energy(1000, 500, "claude-sonnet-4-6")
        e_high, *_ = calculate_energy(1000 + extra_in, 500, "claude-sonnet-4-6")
        assert e_high >= e_low

    @given(suffix=st.text(alphabet="abcdefghijklmnop", min_size=1, max_size=8))
    def test_prefix_uncertainty_never_below_exact(self, suffix: str) -> None:
        # A prefix proxy of gpt-4o must never report narrower uncertainty than exact.
        _, _, exact_unc, *_ = calculate_energy(1000, 500, "gpt-4o")
        _, _, proxy_unc, *_ = calculate_energy(1000, 500, f"gpt-4o-{suffix}-99")
        assert proxy_unc >= exact_unc


class TestHFOrgPrefix:
    """HF/hub ids carry an org prefix that bare registry keys don't; the resolver
    must strip it (and casefold) so a self-hosted model tagged with its repo id
    resolves exactly instead of a wrong-family proxy."""

    def test_org_prefixed_resolves_exact(self):
        m = resolve_model_match("google/gemma-4-31B-it")
        assert m.precision == "exact" and m.name == "gemma-4-31b-it"

    def test_org_prefix_various(self):
        assert resolve_model_match("openai/gpt-4o").precision == "exact"
        assert resolve_model_match("google/gemini-3.1-pro").name == "gemini-3.1-pro"

    def test_bare_key_still_exact(self):
        assert resolve_model_match("gemma-4-31b-it").precision == "exact"

    def test_unknown_org_prefixed_does_not_false_match(self):
        # A genuinely-unknown org/model must not be forced to exact.
        assert resolve_model_match("acme/not-a-real-model-xyz").precision in ("family", "fallback")
