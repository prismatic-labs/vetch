"""End-to-end privacy regression: planted sensitive data must never leak.

Vetch is metadata-only — it captures token counts, model, timing, finish reason,
and tool *names*, but never prompt/completion text or tool-call arguments. This
suite is the belt-and-suspenders proof of that guarantee across sensitive-data
categories (PHI, financial, PII) and across every channel Vetch observes:
prompt, completion, tool-call arguments, and tags.

All tokens below are obviously synthetic (``FAKE`` markers, ``example.invalid``)
so they cannot be mistaken for real data and do not match secret-scanner patterns.
"""

from __future__ import annotations

import pytest

from vetch.emitter import BufferedEmitter, serialize_event, set_test_emitter

# Unique, searchable, clearly-synthetic tokens per category.
SENSITIVE: dict[str, list[str]] = {
    "phi": ["mrn-FAKE-000111", "dx-code-FAKE-z00", "patient-ref-FAKE-42"],
    "financial": ["cardnum-FAKE-411111", "acct-FAKE-99887766", "routing-FAKE-021000"],
    "pii": ["ssn-FAKE-000000000", "nobody@example.invalid", "person-FAKE-xyz"],
}
_CATEGORIES = sorted(SENSITIVE)


@pytest.fixture
def buf():
    emitter = BufferedEmitter()
    set_test_emitter(emitter)
    yield emitter
    set_test_emitter(None)


@pytest.mark.parametrize("category", _CATEGORIES)
def test_langchain_event_leaks_no_content_or_tool_args(category, buf):
    """Plant category secrets in the prompt, the completion, and tool-call args.
    The emitted event must contain none of them. The (non-sensitive) tool *name*
    is captured, proving we keep names but never content/arguments."""
    pytest.importorskip("langchain_core")
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    from vetch.integrations.langchain import VetchCallbackHandler

    secrets = SENSITIVE[category]
    ai = AIMessage(
        content=f"answer includes {secrets[1]} and {secrets[2]}",
        tool_calls=[
            {
                "name": "run_query",
                "args": {"value": secrets[0], "note": secrets[1]},
                "id": "1",
            }
        ],
        usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
    )
    model = GenericFakeChatModel(messages=iter([ai]))
    model.invoke(f"process {secrets[0]} now", config={"callbacks": [VetchCallbackHandler()]})

    assert buf.events, "no event emitted"
    blob = serialize_event(buf.events[-1])
    for secret in secrets:
        assert secret not in blob, f"{category} leaked into event: {secret!r}"
    # tool name (non-sensitive, developer-defined) IS captured
    assert "run_query" in blob


@pytest.mark.parametrize("category", _CATEGORIES)
def test_wrap_event_leaks_no_filtered_tag_key_or_value(category, buf):
    """A sensitive tag whose key is filtered by the allowlist must not leak its
    key or value into the emitted event (or its warnings)."""
    import vetch
    import vetch.config as config

    secret = SENSITIVE[category][0]
    prev_allowlist = config._tag_allowlist
    try:
        vetch.set_tag_allowlist(["feature"])
        with vetch.wrap(tags={"feature": "ok", secret: "1", "detail": secret}, emit=True):
            pass
        assert buf.events, "no event emitted"
        blob = serialize_event(buf.events[-1])
        assert secret not in blob, f"{category} tag leaked into event: {secret!r}"
    finally:
        config._tag_allowlist = prev_allowlist


@pytest.mark.parametrize("category", _CATEGORIES)
def test_wrap_event_redacts_sensitive_tag_value(category, buf):
    """A sensitive value on an allowed tag key must be hashed, not emitted in the
    clear, when the key is registered via set_redacted_tags."""
    import vetch
    import vetch.config as config

    secret = SENSITIVE[category][0]
    prev_redacted = config._redacted_tags
    prev_allowlist = config._tag_allowlist
    try:
        config._tag_allowlist = None  # no allowlist: keep the tag but redact its value
        vetch.set_redacted_tags(["detail"])
        with vetch.wrap(tags={"detail": secret}, emit=True):
            pass
        assert buf.events, "no event emitted"
        blob = serialize_event(buf.events[-1])
        assert secret not in blob, f"{category} tag value leaked (not redacted): {secret!r}"
    finally:
        config._redacted_tags = prev_redacted
        config._tag_allowlist = prev_allowlist
