"""Regression tests for v0.10.0 capability review fixes."""

from __future__ import annotations

import pytest

from vetch.capabilities import (
    reset_capability_state,
    resolve_model_capability,
    rollup_capability_summary_from_events,
    stage_request_tools,
    truncate_capability_lists_for_transport,
)
from vetch.context import TrackingContext
from vetch.stats import SessionStats


@pytest.fixture(autouse=True)
def _clean():
    reset_capability_state()
    yield
    reset_capability_state()


def test_resolve_model_capability_uses_longest_prefix():
    assert resolve_model_capability("gpt-4o-2024-11-20") == "chat"
    assert resolve_model_capability("text-embedding-3-small") == "embedding"
    assert resolve_model_capability("claude-sonnet-4-6") == "chat"


def test_stage_request_tools_before_reroute_model_mutation():
    """Offered tools are staged before kwargs['model'] is substituted."""
    kwargs = {
        "model": "gpt-4o",
        "tools": [{"type": "function", "function": {"name": "search", "parameters": {}}}],
    }
    with TrackingContext() as ctx:
        stage_request_tools("openai", kwargs)
        kwargs["model"] = "gpt-4o-mini"
        assert ctx.pending_tools_offered is not None
        assert [r["name"] for r in ctx.pending_tools_offered] == ["search"]


def test_audit_rollup_matches_runtime_for_truncated_transport_lists():
    offered = [{"name": f"t{i:03d}", "kind": "function"} for i in range(80)]
    invoked = [{"name": "t000", "kind": "function"}]
    events = [
        {
            "usage": {"text": {"input_tokens": 500, "output_tokens": 10}},
            "estimated_cost_input_usd": 0.01,
            "cache_read_tokens": 0,
            "tools_offered": offered,
            "tools_invoked": invoked,
            "tool_call_count": 1,
            "tool_schema_tokens": {f"t{i:03d}": 10 for i in range(80)},
        }
    ]
    runtime = SessionStats()
    runtime.update(events[0])
    stored = truncate_capability_lists_for_transport(dict(events[0]), [])
    assert len(stored["tools_offered"]) == 64

    runtime_summary = runtime.summary()
    audit_summary = rollup_capability_summary_from_events(events)
    runtime_never = runtime_summary["function_tools_never_called"]
    audit_never = audit_summary["function_tools_never_called"]
    assert runtime_never == audit_never
    assert len(runtime_never) == 79


def test_session_scoped_expected_capabilities():
    import vetch
    from vetch.capabilities import get_expected_capabilities, set_expected_capabilities

    set_expected_capabilities(["model:global"])
    with vetch.Session(expected_capabilities=["model:image"]) as _session:
        assert get_expected_capabilities() == ["model:image"]
    assert get_expected_capabilities() == ["model:global"]


def test_manual_capture_redacts_tool_names(monkeypatch):
    from vetch.context import TrackingContext

    monkeypatch.setenv("VETCH_REDACTION_KEY", "unit-test-key")
    ctx = TrackingContext()
    ctx.capture(
        model="gpt-4o",
        provider="openai",
        usage={"text": {"input_tokens": 1, "output_tokens": 1}},
        tools_offered=[{"name": "send_email_to_alice", "kind": "function"}],
        tools_invoked=[],
        tool_schema_tokens={"send_email_to_alice": 42},
    )
    assert ctx.captured_call is not None
    name = ctx.captured_call.tools_offered[0]["name"]
    assert name.startswith("redacted-")
    assert "alice" not in name
    assert all(k.startswith("redacted-") for k in (ctx.captured_call.tool_schema_tokens or {}))


def test_build_cap_findings_uses_explicit_manifest_without_global_mutation():
    from vetch.audit_report import _build_cap_findings
    from vetch.capabilities import get_expected_capabilities, set_expected_capabilities

    set_expected_capabilities([])
    events = [{"capabilities_invoked": []}]
    findings = _build_cap_findings(
        events,
        window_days=1.0,
        expected_capabilities=["model:image"],
    )
    assert get_expected_capabilities() == []
    assert len(findings) == 1
    assert findings[0].code == "CAP-001"


def test_offered_memo_is_bounded():
    import vetch.capabilities as cap

    cap.reset_capability_state()

    def tools(i: int):
        return [{"type": "function", "function": {"name": f"t{i}", "parameters": {}}}]

    for i in range(300):
        extract = cap.extract_openai_tools_offered({"tools": tools(i)})
        assert extract[0] is not None
    assert len(cap._offered_memo) <= cap._OFFERED_MEMO_MAX
