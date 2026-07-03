"""Property-based invariants for capability observability (Hypothesis).

These assert the invariants that must hold for *all* inputs, complementing the
example-based tests in test_capabilities.py.
"""

from __future__ import annotations

import string

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from vetch.capabilities import (
    extract_openai_tools_invoked,
    extract_openai_tools_offered,
    normalize_function_tools,
    reset_capability_state,
)
from vetch.stats import SessionStats

_junk = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text(max_size=10),
    lambda children: (
        st.lists(children, max_size=5) | st.dictionaries(st.text(max_size=5), children, max_size=5)
    ),
    max_leaves=15,
)


@pytest.fixture(autouse=True)
def _clean_capability_state():
    reset_capability_state()
    yield
    reset_capability_state()


# --- normalizer fail-open + shape invariants -------------------------------

_names = st.lists(st.text(max_size=40), max_size=30)


@given(names=_names)
@settings(max_examples=200)
def test_normalize_never_raises_and_is_well_formed(names):
    refs = normalize_function_tools(names)
    assert isinstance(refs, list)
    seen = set()
    for ref in refs:
        assert set(ref.keys()) == {"name", "kind"}
        assert ref["kind"] == "function"
        assert isinstance(ref["name"], str)
        assert ref["name"]  # no empty names
        assert ref["name"] not in seen  # de-duped
        seen.add(ref["name"])


@given(names=_names)
@settings(max_examples=200)
def test_normalize_is_sorted_and_subset_of_input(names):
    refs = normalize_function_tools(names)
    out_names = [r["name"] for r in refs]
    assert out_names == sorted(out_names)  # stable, deterministic order
    # every emitted name derives from a non-empty input name (no redaction here)
    non_empty_inputs = {n for n in names if n}
    assert set(out_names) <= non_empty_inputs


@given(names=_names)
@settings(max_examples=100)
def test_normalize_is_idempotent_on_names(names):
    once = normalize_function_tools(names)
    twice = normalize_function_tools([r["name"] for r in once])
    assert once == twice


# --- rollup helpers --------------------------------------------------------


def _event(
    *,
    offered=None,
    invoked=None,
    schema_tokens=None,
    cost_in=0.0,
    cost_cache_read=0.0,
    in_tok=0,
    cache_read_tokens=0,
    estimated_cost_usd=0.0,
    caps_invoked=None,
):
    ev = {
        "usage": {"text": {"input_tokens": in_tok, "output_tokens": 0}},
        "estimated_cost_usd": estimated_cost_usd,
        "estimated_cost_input_usd": cost_in,
        "estimated_cost_cache_read_usd": cost_cache_read,
        "cache_read_tokens": cache_read_tokens,
        "model": "gpt-4o",
    }
    if offered is not None:
        ev["tools_offered"] = [{"name": n, "kind": "function"} for n in offered]
    if invoked is not None:
        ev["tools_invoked"] = [{"name": n, "kind": "function"} for n in invoked]
    if schema_tokens is not None:
        ev["tool_schema_tokens"] = schema_tokens
    if caps_invoked is not None:
        ev["capabilities_invoked"] = caps_invoked
    return ev


_tool_names = st.lists(
    st.text(alphabet="abcdefghijklmnop_", min_size=1, max_size=8),
    min_size=0,
    max_size=8,
)


@given(offered=_tool_names, invoked=_tool_names)
@settings(max_examples=200)
def test_never_called_is_offered_minus_invoked(offered, invoked):
    stats = SessionStats()
    stats.update(
        _event(
            offered=offered,
            invoked=invoked,
            schema_tokens={n: 10 for n in offered},
            cost_in=0.01,
            in_tok=100,
        )
    )
    s = stats.summary()
    never = set(s["function_tools_never_called"])
    offered_set, invoked_set = set(offered), set(invoked)
    # core invariants
    assert never <= offered_set
    assert never.isdisjoint(invoked_set)
    assert never == offered_set - invoked_set


@given(offered=_tool_names, invoked=_tool_names)
@settings(max_examples=150)
def test_wasted_cost_is_nonnegative(offered, invoked):
    stats = SessionStats()
    stats.update(
        _event(
            offered=offered,
            invoked=invoked,
            schema_tokens={n: 25 for n in offered},
            cost_in=0.05,
            in_tok=500,
        )
    )
    s = stats.summary()
    assert s["wasted_tool_schema_tokens"] >= 0
    assert s["wasted_tool_schema_cost_usd"] >= 0.0


@given(offered=st.lists(st.sampled_from(["a", "b", "c", "d"]), min_size=1, max_size=4, unique=True))
@settings(max_examples=100)
def test_invoking_a_tool_is_monotonic(offered):
    """Invoking a previously-uncalled tool can only shrink never_called."""
    base = SessionStats()
    base.update(
        _event(
            offered=offered,
            invoked=[],
            schema_tokens={n: 10 for n in offered},
            cost_in=0.01,
            in_tok=100,
        )
    )
    before = set(base.summary()["function_tools_never_called"])

    after_stats = SessionStats()
    after_stats.update(
        _event(
            offered=offered,
            invoked=[offered[0]],
            schema_tokens={n: 10 for n in offered},
            cost_in=0.01,
            in_tok=100,
        )
    )
    after = set(after_stats.summary()["function_tools_never_called"])
    assert after <= before
    assert offered[0] not in after


# --- cache-aware cost invariants -------------------------------------------


def test_fully_cached_session_reports_zero_cost():
    """billable_input_tokens == 0 with dead tools -> $0 + note, never div/0."""
    stats = SessionStats()
    # all input tokens are cache reads -> billable == 0
    stats.update(
        _event(
            offered=["dead"],
            invoked=[],
            schema_tokens={"dead": 500},
            cost_in=0.10,
            cost_cache_read=0.10,
            in_tok=1000,
            cache_read_tokens=1000,
        )
    )
    s = stats.summary()
    assert s["wasted_tool_schema_tokens_per_request"] == 500
    assert s["wasted_tool_schema_tokens"] == 500
    assert s["wasted_tool_schema_cost_usd"] == 0.0
    assert "wasted_tool_schema_cost_note" in s


def test_rate_ignores_output_cost():
    """Guardrail #1: wasted cost uses the input rate, never blended total cost."""
    small_output = SessionStats()
    small_output.update(
        _event(
            offered=["dead"],
            invoked=[],
            schema_tokens={"dead": 100},
            cost_in=0.01,
            in_tok=1000,
            estimated_cost_usd=0.02,
        )
    )  # small total
    huge_output = SessionStats()
    huge_output.update(
        _event(
            offered=["dead"],
            invoked=[],
            schema_tokens={"dead": 100},
            cost_in=0.01,
            in_tok=1000,
            estimated_cost_usd=99.0,
        )
    )  # huge output cost
    # Same input economics -> identical wasted cost despite 5000x total cost gap.
    assert (
        small_output.summary()["wasted_tool_schema_cost_usd"]
        == huge_output.summary()["wasted_tool_schema_cost_usd"]
    )


def test_cost_matches_manual_input_rate():
    stats = SessionStats()
    stats.update(
        _event(offered=["dead"], invoked=[], schema_tokens={"dead": 200}, cost_in=0.05, in_tok=1000)
    )
    # rate = 0.05 / 1000 = 5e-5 ; wasted = 200 * 5e-5 = 0.01
    assert stats.summary()["wasted_tool_schema_cost_usd"] == round(200 * (0.05 / 1000), 6)


# --- extractor fail-open over arbitrary objects (#2) ------------------------


@given(obj=_junk)
@settings(max_examples=200)
def test_invoked_extractor_never_raises_on_junk(obj):
    out = extract_openai_tools_invoked(obj)
    assert isinstance(out, tuple) and len(out) == 2


@given(tools=_junk)
@settings(max_examples=200)
def test_offered_extractor_never_raises_on_junk(tools):
    out = extract_openai_tools_offered({"tools": tools})
    assert isinstance(out, tuple) and len(out) == 2


# --- redaction privacy invariant (#7) --------------------------------------


@given(
    names=st.lists(
        st.text(alphabet=string.ascii_letters, min_size=1, max_size=12), min_size=1, max_size=8
    )
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_redaction_leaves_no_plaintext_name(names, monkeypatch):
    monkeypatch.setenv("VETCH_REDACTION_KEY", "prop-test-key")
    refs = normalize_function_tools(names)
    original = set(names)
    out_names = [r["name"] for r in refs]
    # every emitted name is an opaque hash, and no plaintext input survives
    assert all(n.startswith("redacted-") for n in out_names)
    assert original.isdisjoint(out_names)
    assert out_names == sorted(out_names)  # still deterministic under redaction
