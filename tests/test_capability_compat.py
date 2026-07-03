"""Backward-compatibility and golden-snapshot tests.

The audit/rollup path reads historical events that predate the v0.10.0 fields.
Those must roll up None-safe. The summary numbers are the product surface, so a
snapshot over a deterministic scenario locks them into diffs.
"""

from __future__ import annotations

import pytest

from vetch.audit_report import _build_cap_findings, _summarize_events
from vetch.capabilities import reset_capability_state, set_expected_capabilities
from vetch.stats import SessionStats


@pytest.fixture(autouse=True)
def _clean():
    reset_capability_state()
    yield
    reset_capability_state()


def _legacy_event():
    """A pre-0.10.0 event: no tools_* / capabilities_invoked / cache fields."""
    return {
        "usage": {"text": {"input_tokens": 100, "output_tokens": 50}},
        "estimated_cost_usd": 0.02,
        "model": "gpt-4o",
    }


def test_legacy_events_roll_up_without_new_fields():
    stats = SessionStats()
    for _ in range(5):
        stats.update(_legacy_event())
    s = stats.summary()
    # capability keys present and empty, never KeyError/None crash
    assert s["function_tools_never_called"] == []
    assert s["wasted_tool_schema_tokens"] == 0
    assert s["wasted_tool_schema_cost_usd"] == 0.0
    assert s["declared_capabilities_silent"] == []
    assert s["capability_invocation_counts"] == {}


def test_missing_cost_fields_do_not_divide_by_zero():
    stats = SessionStats()
    stats.update(
        {
            "usage": {"text": {"input_tokens": 0, "output_tokens": 0}},
            "model": "gpt-4o",
            "tools_offered": [{"name": "x", "kind": "function"}],
            "tools_invoked": [],
            "tool_schema_tokens": {"x": 10},
        }
    )
    s = stats.summary()
    assert s["function_tools_never_called"] == ["x"]
    assert s["wasted_tool_schema_cost_usd"] == 0.0  # billable == 0 guard


def test_summary_snapshot_is_stable():
    """Deterministic scenario -> locked numbers. Update intentionally if design changes."""
    set_expected_capabilities(["model:image", "model:embedding"])
    stats = SessionStats()

    # 10 calls: offer {search, refund, lookup}, only ever call search.
    for _ in range(10):
        stats.update(
            {
                "usage": {"text": {"input_tokens": 1000, "output_tokens": 100}},
                "estimated_cost_usd": 0.05,
                "estimated_cost_input_usd": 0.01,
                "estimated_cost_cache_read_usd": 0.0,
                "cache_read_tokens": 0,
                "model": "gpt-4o",
                "tools_offered": [
                    {"name": "search", "kind": "function"},
                    {"name": "refund", "kind": "function"},
                    {"name": "lookup", "kind": "function"},
                ],
                "tools_invoked": [{"name": "search", "kind": "function"}],
                "tool_call_count": 1,
                "tool_schema_tokens": {"search": 40, "refund": 60, "lookup": 50},
                "capabilities_invoked": [{"name": "embedding", "kind": "model"}],
            }
        )

    s = stats.summary()
    assert s["function_tools_never_called"] == ["lookup", "refund"]
    assert s["wasted_tool_schema_tokens_per_request"] == 110  # 60 + 50
    assert s["wasted_tool_schema_tokens"] == 110 * 10
    assert s["wasted_tool_schema_session_tokens"] == 110 * 10
    # rate = total_effective_input (10 * 0.01) / total_billable (10 * 1000) = 1e-6/tok
    # per-request = 110 * 1e-6 = 0.00011; session = 0.00011 * 10 requests
    assert s["wasted_tool_schema_cost_per_request_usd"] == round(110 * (0.10 / 10000), 6)
    assert s["wasted_tool_schema_cost_usd"] == round(10 * 110 * (0.10 / 10000), 6)
    assert s["dead_tool_offer_request_count"] == 10
    assert s["declared_capabilities_silent"] == ["model:image"]  # embedding fired
    assert s["capability_invocation_counts"] == {"model:embedding": 10}
    assert s["tool_call_event_rate"] == 1.0


# --- audit-path backward compatibility (#8) --------------------------------


def _legacy_stored_events():
    """Stored JSON that predates v0.10.0 capability fields."""
    return [
        {
            "model": "gpt-4o",
            "provider": "openai",
            "estimated_cost_usd": 0.02,
            "usage": {"text": {"input_tokens": 100, "output_tokens": 50}},
        }
        for _ in range(3)
    ]


def test_audit_summarize_handles_legacy_events():
    # audit_report rollup over pre-0.10.0 stored events must not crash.
    out = _summarize_events(_legacy_stored_events())
    assert isinstance(out, dict)


def test_audit_cap_findings_none_safe_on_legacy_events():
    set_expected_capabilities(["model:image"])
    # legacy events have no capabilities_invoked -> everything reads as silent,
    # but must not raise on the missing field.
    findings = _build_cap_findings(_legacy_stored_events(), window_days=1.0)
    assert len(findings) == 1
    assert findings[0].evidence["declared_capabilities_silent"] == ["model:image"]


def test_audit_cap_no_finding_when_declared_route_fires():
    set_expected_capabilities(["model:image"])
    events = [
        {
            "capabilities_invoked": [{"kind": "model", "name": "image"}],
        }
    ]
    findings = _build_cap_findings(events, window_days=1.0)
    assert findings == []


def test_rollup_capability_summary_from_stored_events():
    from vetch.capabilities import rollup_capability_summary_from_events

    set_expected_capabilities(["model:image"])
    events = [
        {
            "usage": {"text": {"input_tokens": 500, "output_tokens": 10}},
            "estimated_cost_input_usd": 0.01,
            "cache_read_tokens": 0,
            "tools_offered": [
                {"name": "a", "kind": "function"},
                {"name": "b", "kind": "function"},
            ],
            "tools_invoked": [{"name": "a", "kind": "function"}],
            "tool_call_count": 1,
            "tool_schema_tokens": {"a": 20, "b": 30},
        }
        for _ in range(4)
    ]
    rollup = rollup_capability_summary_from_events(events)
    assert rollup["function_tools_never_called"] == ["b"]
    assert rollup["dead_tool_offer_request_count"] == 4
    assert rollup["wasted_tool_schema_tokens"] == 30 * 4
    assert rollup["wasted_tool_schema_cost_usd"] == rollup["wasted_tool_schema_session_cost_usd"]


def test_dead_tool_retransmit_skipped_when_invoked_unknown():
    """tools_offered with tools_invoked=None (unknown) must not attribute retransmit cost."""
    stats = SessionStats()
    for _ in range(5):
        stats.update(
            {
                "usage": {"text": {"input_tokens": 1000, "output_tokens": 0}},
                "estimated_cost_input_usd": 0.02,
                "cache_read_tokens": 0,
                "model": "gpt-4o",
                "tools_offered": [
                    {"name": "a", "kind": "function"},
                    {"name": "b", "kind": "function"},
                ],
                # tools_invoked intentionally omitted -> unknown
                "tool_schema_tokens": {"a": 20, "b": 30},
            }
        )
    s = stats.summary()
    assert s["dead_tool_offer_request_count"] == 0
    assert s["wasted_tool_schema_cost_usd"] == 0.0  # no cost attributed when invoked unknown
    assert s["function_tools_never_called"] == []

    # Regression: a known-empty tools_invoked is a genuine dead offer and still counts.
    stats2 = SessionStats()
    stats2.update(
        {
            "usage": {"text": {"input_tokens": 1000, "output_tokens": 0}},
            "estimated_cost_input_usd": 0.02,
            "cache_read_tokens": 0,
            "model": "gpt-4o",
            "tools_offered": [{"name": "a", "kind": "function"}],
            "tools_invoked": [],
            "tool_schema_tokens": {"a": 20},
        }
    )
    assert stats2.summary()["dead_tool_offer_request_count"] == 1
