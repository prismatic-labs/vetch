"""Tests for capability observability (v0.10.0)."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from vetch.capabilities import (
    configure_capabilities,
    derive_capabilities_invoked,
    extract_anthropic_tools_invoked,
    extract_anthropic_tools_offered,
    extract_openai_tools_invoked,
    extract_openai_tools_offered,
    finalize_openai_stream_tools,
    load_model_capability_map,
    normalize_function_tools,
    redact_capability_name,
    reset_capability_state,
    set_redacted_capability_names,
    truncate_capability_lists_for_transport,
)
from vetch.schema import CapabilityRef
from vetch.stats import SessionStats


@pytest.fixture(autouse=True)
def _reset_caps() -> None:
    reset_capability_state()
    yield
    reset_capability_state()


def test_normalize_function_tools_dedupes_and_sorts() -> None:
    refs = normalize_function_tools(["b", "a", "a"])
    assert refs == [
        {"name": "a", "kind": "function"},
        {"name": "b", "kind": "function"},
    ]


def test_extract_openai_tools_offered_and_invoked() -> None:
    kwargs = {
        "tools": [
            {"type": "function", "function": {"name": "get_weather"}},
            {"type": "function", "function": {"name": "refund_order"}},
        ]
    }
    offered, sizes = extract_openai_tools_offered(kwargs)
    assert offered is not None
    assert {r["name"] for r in offered} == {"get_weather", "refund_order"}
    assert sizes is not None
    assert "get_weather" in sizes

    result = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(name="get_weather"),
                        )
                    ]
                )
            )
        ]
    )
    invoked, count = extract_openai_tools_invoked(result)
    assert count == 1
    assert invoked is not None
    assert invoked[0]["name"] == "get_weather"


def test_extract_anthropic_tools() -> None:
    offered, _ = extract_anthropic_tools_offered({"tools": [{"name": "search"}]})
    assert offered is not None
    assert offered[0]["name"] == "search"

    result = SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", name="search")]
    )
    invoked, count = extract_anthropic_tools_invoked(result)
    assert count == 1
    assert invoked is not None


def test_stream_finalize_none_on_error() -> None:
    invoked, count = finalize_openai_stream_tools({0: {"name": "x"}}, complete=False, error=True)
    assert invoked is None
    assert count is None


def test_cache_aware_wasted_cost_fully_cached() -> None:
    stats = SessionStats()
    event: dict[str, Any] = {
        "usage": {"text": {"input_tokens": 1000, "output_tokens": 10, "total_tokens": 1010}},
        "estimated_cost_usd": 0.01,
        "estimated_cost_input_usd": 0.005,
        "estimated_cost_cache_read_usd": 0.005,
        "cache_read_tokens": 1000,
        "tools_offered": [{"name": "dead_tool", "kind": "function"}],
        "tools_invoked": [],
        "tool_call_count": 0,
        "tool_schema_tokens": {"dead_tool": 500},
    }
    stats.update(event)
    summary = stats.summary()
    assert summary["function_tools_never_called"] == ["dead_tool"]
    assert summary["wasted_tool_schema_tokens_per_request"] == 500
    assert summary["wasted_tool_schema_tokens"] == 500
    assert summary["wasted_tool_schema_cost_usd"] == 0.0


def test_cache_aware_wasted_cost_mixed() -> None:
    stats = SessionStats()
    for _ in range(3):
        stats.update(
            {
                "usage": {"text": {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}},
                "estimated_cost_input_usd": 0.01,
                "estimated_cost_cache_read_usd": 0.002,
                "cache_read_tokens": 20,
                "tools_offered": [{"name": "unused", "kind": "function"}],
                "tools_invoked": [],
                "tool_schema_tokens": {"unused": 100},
            }
        )
    summary = stats.summary()
    assert summary["wasted_tool_schema_tokens_per_request"] == 100
    assert summary["wasted_tool_schema_tokens"] == 300
    assert summary["wasted_tool_schema_cost_usd"] > 0


def test_declared_capabilities_silent() -> None:
    configure_capabilities(expected=["model:image"])
    stats = SessionStats()
    stats.update(
        {
            "usage": {"text": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
            "capabilities_invoked": [{"name": "embedding", "kind": "model"}],
        }
    )
    summary = stats.summary()
    assert "model:image" in summary["declared_capabilities_silent"]


def test_derive_capabilities_invoked_embedding_and_registry() -> None:
    refs = derive_capabilities_invoked(
        is_embedding=True,
        usage={"text": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}},
        model="text-embedding-3-small",
    )
    assert refs is not None
    names = {r["name"] for r in refs}
    assert "embedding" in names


def test_registry_model_capabilities_loads() -> None:
    mapping = load_model_capability_map()
    assert mapping.get("gpt-image-1") == "image"


def test_truncate_transport_only() -> None:
    offered = [{"name": f"t{i}", "kind": "function"} for i in range(70)]
    event = {"tools_offered": offered}
    out = truncate_capability_lists_for_transport(event, [])
    assert len(out["tools_offered"]) == 64


def test_redact_capability_name_with_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VETCH_REDACTION_KEY", "test-secret-key")
    set_redacted_capability_names(["secret_tool"])
    assert redact_capability_name("secret_tool").startswith("redacted-")


def test_redaction_key_does_not_hash_tool_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VETCH_REDACTION_KEY", "k")
    assert redact_capability_name("search_index") == "search_index"


def test_redact_capability_name_opted_in_hashes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VETCH_REDACTION_KEY", "k")
    set_redacted_capability_names(["internal_tool"])
    assert redact_capability_name("internal_tool").startswith("redacted-")
    assert redact_capability_name("public_tool") == "public_tool"


def test_concurrent_stats_update() -> None:
    stats = SessionStats()

    def worker(i: int) -> None:
        stats.update(
            {
                "usage": {"text": {"input_tokens": 10, "output_tokens": 1, "total_tokens": 11}},
                "tools_offered": [{"name": f"tool_{i}", "kind": "function"}],
            }
        )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert stats.total_requests == 20


def test_event_json_round_trip() -> None:
    """New nullable capability fields survive JSON serialization."""
    refs: list[CapabilityRef] = [{"name": "a", "kind": "function"}]
    payload = {
        "tools_offered": refs,
        "tools_invoked": refs,
        "tool_call_count": 1,
        "capabilities_invoked": [{"name": "image", "kind": "model"}],
        "tool_schema_tokens": {"a": 42},
    }
    restored = json.loads(json.dumps(payload))
    assert restored["tool_call_count"] == 1
    assert restored["tool_schema_tokens"]["a"] == 42
