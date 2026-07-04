"""Adversarial privacy + fault injection.

Privacy: tool arguments/schemas must never survive extraction; redaction must
leave no plaintext. Fault injection: extraction and derivation must never break
the caller's inference, whatever the tokenizer/registry does.
"""

from __future__ import annotations

import pytest

import vetch.capabilities as cap
from vetch.capabilities import (
    derive_capabilities_invoked,
    extract_openai_tools_offered,
    reset_capability_state,
    set_otel_capability_attributes,
    set_redacted_capability_names,
    truncate_capability_lists_for_transport,
)


@pytest.fixture(autouse=True)
def _clean():
    reset_capability_state()
    yield
    reset_capability_state()


# --- privacy ---------------------------------------------------------------


def test_no_tool_arguments_or_schema_survive_extraction():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "charge_card",
                "description": "SECRET internal description",
                "parameters": {"properties": {"pan": {"type": "string"}}},
            },
        }
    ]
    refs, sizes = extract_openai_tools_offered({"tools": tools})
    # only name + kind survive; no description/parameters/pan anywhere
    assert refs == [{"name": "charge_card", "kind": "function"}]
    blob = repr(refs) + repr(sizes)
    for secret in ("SECRET", "description", "parameters", "pan"):
        assert secret not in blob
    # schema size is an int count, not the payload
    assert isinstance(sizes["charge_card"], int)


def test_redaction_leaves_no_plaintext(monkeypatch):
    monkeypatch.setenv("VETCH_REDACTION_KEY", "unit-test-key")
    set_redacted_capability_names(["notify_user_acme"])
    tools = [{"type": "function", "function": {"name": "notify_user_acme"}}]
    refs, _ = extract_openai_tools_offered({"tools": tools})
    name = refs[0]["name"]
    assert name.startswith("redacted-")
    assert "acme" not in name and "notify_user" not in name


def test_redaction_is_stable_within_process(monkeypatch):
    monkeypatch.setenv("VETCH_REDACTION_KEY", "unit-test-key")
    set_redacted_capability_names(["tool_x"])
    a = cap.redact_capability_name("tool_x")
    b = cap.redact_capability_name("tool_x")
    assert a == b and a != "tool_x"


# --- fault injection (hot path must never raise) ---------------------------


def test_tokenizer_failure_is_fail_open(monkeypatch):
    def boom(_obj):
        raise RuntimeError("tokenizer exploded")

    monkeypatch.setattr("vetch.capabilities._estimate_tool_json_tokens", boom)
    tools = [{"type": "function", "function": {"name": "t1"}}]
    # must not raise; fail-open returns a tuple (may be (None, None))
    refs, sizes = extract_openai_tools_offered({"tools": tools})
    assert isinstance((refs, sizes), tuple)


def test_registry_load_failure_is_fail_open(monkeypatch, tmp_path):
    monkeypatch.setattr("vetch.capabilities.REGISTRY_DIR", tmp_path / "does-not-exist")
    # map load fails -> {}, but derivation still works off usage flags
    assert cap.load_model_capability_map() == {}
    refs = derive_capabilities_invoked(is_embedding=True, usage=None, model="whatever")
    assert refs == [{"name": "embedding", "kind": "model"}]


def test_derive_never_raises_on_garbage_usage():
    for usage in (None, {}, {"image": "not-a-dict"}, {"audio": {"input_tokens": None}}):
        out = derive_capabilities_invoked(is_embedding=False, usage=usage, model="m")
        assert out is None or isinstance(out, list)


# --- OTel semconv + transport cap ------------------------------------------


class _FakeSpan:
    def __init__(self):
        self.attrs = {}

    def set_attribute(self, k, v):
        self.attrs[k] = v


def test_otel_attributes_are_semconv_arrays():
    span = _FakeSpan()
    event = {
        "tools_offered": [{"name": "a", "kind": "function"}, {"name": "b", "kind": "function"}],
        "tools_invoked": [{"name": "a", "kind": "function"}],
        "tool_call_count": 1,
        "capabilities_invoked": [{"name": "image", "kind": "model"}],
        "tool_schema_tokens": {"a": 10, "b": 20},
    }
    set_otel_capability_attributes(span, event)
    assert span.attrs["gen_ai.tool.definitions"] == ["a", "b"]
    assert span.attrs["gen_ai.tool.calls"] == ["a"]
    assert span.attrs["gen_ai.tool.call.count"] == 1
    assert span.attrs["vetch.capabilities_invoked"] == ["model:image"]
    assert span.attrs["vetch.tools_never_called"] == ["b"]
    assert span.attrs["vetch.wasted_tool_schema_tokens"] == 20
    # arrays, not CSV strings
    assert isinstance(span.attrs["gen_ai.tool.definitions"], list)


def test_otel_emits_wasted_tool_schema_tokens():
    span = _FakeSpan()
    event = {
        "tools_offered": [{"name": "a", "kind": "function"}, {"name": "b", "kind": "function"}],
        "tools_invoked": [{"name": "a", "kind": "function"}],
        "tool_schema_tokens": {"a": 10, "b": 30},
    }
    set_otel_capability_attributes(span, event)
    # only the never-called tool (b) counts toward wasted schema tokens
    assert span.attrs["vetch.wasted_tool_schema_tokens"] == 30


def test_transport_cap_truncates_and_warns_without_touching_source():
    offered = [{"name": f"t{i:03d}", "kind": "function"} for i in range(100)]
    event = {"tools_offered": offered}
    warnings: list[str] = []
    out = truncate_capability_lists_for_transport(event, warnings)
    assert len(out["tools_offered"]) == 64
    assert any("tools_offered_truncated" in w for w in out["vetch_warnings"])
    # source event object is untouched (rollup sees full list)
    assert len(event["tools_offered"]) == 100
