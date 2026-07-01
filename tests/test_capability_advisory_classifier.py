"""Treat the advisories as classifiers + metamorphic estimator checks.

TOOL-DEAD-001 and CAP-001 are detectors, so characterize their error rates over
a labeled corpus rather than a single example. Plus metamorphic relations for the
cost estimate where ground truth is unavailable.
"""

from __future__ import annotations

import pytest

import vetch.capabilities as cap
from vetch.advisory import generate_advisories
from vetch.audit_report import _build_cap_findings
from vetch.capabilities import reset_capability_state, set_expected_capabilities
from vetch.stats import SessionStats


@pytest.fixture(autouse=True)
def _clean():
    reset_capability_state()
    yield
    reset_capability_state()


def _session(*, requests, offered, invoked, cached=False, schema_tokens=50):
    stats = SessionStats()
    for _ in range(requests):
        ev = {
            "usage": {"text": {"input_tokens": 1000, "output_tokens": 100}},
            "estimated_cost_usd": 0.05,
            "estimated_cost_input_usd": 0.02,
            "estimated_cost_cache_read_usd": 0.02 if cached else 0.0,
            "cache_read_tokens": 1000 if cached else 0,
            "model": "gpt-4o",
            "tools_offered": [{"name": n, "kind": "function"} for n in offered],
            "tools_invoked": [{"name": n, "kind": "function"} for n in invoked],
            "tool_call_count": len(invoked),
            "tool_schema_tokens": {n: schema_tokens for n in offered},
        }
        stats.update(ev)
    return stats


def _fires_tool_dead(stats):
    return any(a.code == "TOOL-DEAD-001" for a in generate_advisories(stats))


def test_tool_dead_classifier_precision_and_recall():
    # Labeled corpus: (stats, should_fire)
    corpus = [
        # positives: enough requests, dead tools, real billable cost
        (_session(requests=20, offered=["a", "b", "c"], invoked=["a"]), True),
        (_session(requests=15, offered=["x", "y"], invoked=[]), True),
        (_session(requests=10, offered=["p", "q"], invoked=["p"]), True),
        # negatives:
        (_session(requests=20, offered=["a", "b"], invoked=["a", "b"]), False),  # all used
        (_session(requests=5, offered=["a", "b"], invoked=["a"]), False),  # too few requests
        (_session(requests=20, offered=[], invoked=[]), False),  # no tools
        # fully cached -> wasted cost is $0 -> must not fire
        (_session(requests=20, offered=["a", "b"], invoked=["a"], cached=True), False),
    ]

    tp = fp = fn = tn = 0
    for stats, should in corpus:
        fired = _fires_tool_dead(stats)
        if should and fired:
            tp += 1
        elif should and not fired:
            fn += 1
        elif not should and fired:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    assert precision == 1.0, f"false positives: fp={fp}"
    assert recall == 1.0, f"false negatives: fn={fn}"


def test_cap001_fires_only_when_declared_route_silent():
    set_expected_capabilities(["model:image", "model:embedding"])
    # embedding fires, image never does
    events = [{"capabilities_invoked": [{"name": "embedding", "kind": "model"}]} for _ in range(5)]
    findings = _build_cap_findings(events, window_days=1.0)
    assert len(findings) == 1
    f = findings[0]
    assert f.code == "CAP-001"
    assert f.evidence["declared_capabilities_silent"] == ["model:image"]


def test_cap001_silent_when_all_declared_routes_fire():
    set_expected_capabilities(["model:image"])
    events = [{"capabilities_invoked": [{"name": "image", "kind": "model"}]}]
    assert _build_cap_findings(events, window_days=1.0) == []


def test_cap001_noop_without_manifest():
    set_expected_capabilities([])
    events = [{"capabilities_invoked": []}]
    assert _build_cap_findings(events, window_days=1.0) == []


def test_cap001_no_false_positive_when_embedding_declared_and_fires():
    # Explicit FP guard: declared route fires -> no finding.
    set_expected_capabilities(["model:embedding"])
    events = [{"capabilities_invoked": [{"name": "embedding", "kind": "model"}]}]
    assert _build_cap_findings(events, window_days=1.0) == []


def test_cap001_partial_manifest_only_flags_silent_ones():
    set_expected_capabilities(["model:image", "model:embedding", "model:audio"])
    events = [
        {
            "capabilities_invoked": [
                {"name": "embedding", "kind": "model"},
                {"name": "audio", "kind": "model"},
            ]
        }
    ]
    findings = _build_cap_findings(events, window_days=1.0)
    assert len(findings) == 1
    assert findings[0].evidence["declared_capabilities_silent"] == ["model:image"]


# --- metamorphic estimator relations ---------------------------------------


def test_estimator_is_monotonic_in_schema_size():
    small = {"type": "function", "function": {"name": "t", "description": "x"}}
    large = {
        "type": "function",
        "function": {
            "name": "t",
            "description": "x" * 5000,
            "parameters": {"a": 1, "b": 2, "c": 3},
        },
    }
    assert cap._estimate_tool_json_tokens(large) > cap._estimate_tool_json_tokens(small)


def test_estimator_positive_and_deterministic():
    obj = {"type": "function", "function": {"name": "t"}}
    v1 = cap._estimate_tool_json_tokens(obj)
    v2 = cap._estimate_tool_json_tokens(obj)
    assert v1 == v2 and v1 >= 1


def test_wasted_cost_invariant_to_offered_order():
    def waste(order):
        stats = SessionStats()
        stats.update(
            {
                "usage": {"text": {"input_tokens": 1000, "output_tokens": 0}},
                "estimated_cost_input_usd": 0.02,
                "estimated_cost_cache_read_usd": 0.0,
                "cache_read_tokens": 0,
                "model": "gpt-4o",
                "tools_offered": [{"name": n, "kind": "function"} for n in order],
                "tools_invoked": [{"name": "a", "kind": "function"}],
                "tool_schema_tokens": {"a": 10, "b": 20, "c": 30},
            }
        )
        return stats.summary()["wasted_tool_schema_cost_usd"]

    assert waste(["a", "b", "c"]) == waste(["c", "b", "a"]) == waste(["b", "a", "c"])
