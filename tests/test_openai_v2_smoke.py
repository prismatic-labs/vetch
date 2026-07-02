"""Task 3: Vetch instruments the OpenAI 2.x client (mock-based, no network).

Covers instance-level patching (sync + async) and a regression test for the
module double-instrumentation RecursionError that previously appeared when a
real client was constructed after instrument()/uninstrument() cycles.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("openai")

import openai  # noqa: E402

import vetch  # noqa: E402
from vetch.emitter import BufferedEmitter, set_test_emitter  # noqa: E402
from vetch.providers.openai import (  # noqa: E402
    instrument_openai_module,
    patch_openai_client,
    uninstrument_openai_module,
)


@pytest.fixture(autouse=True)
def _emitter():
    emitter = BufferedEmitter()
    set_test_emitter(emitter)
    yield emitter
    set_test_emitter(None)
    # Safety net: never leak module instrumentation into other tests.
    try:
        uninstrument_openai_module()
    except Exception:
        pass


def _response(model="gpt-4o", prompt=50, completion=100, total=150):
    usage = NS(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)
    message = NS(content="hello", tool_calls=None)
    choice = NS(message=message, finish_reason="stop", delta=NS(content=None))
    return NS(model=model, usage=usage, choices=[choice])


def test_sync_client_patch_emits_event(_emitter):
    client = openai.OpenAI(api_key="test-key")
    # Replace the network call BEFORE patching so the wrapper's original is a mock.
    client.chat.completions.create = MagicMock(return_value=_response())

    assert patch_openai_client(client) is True  # openai 2.x now tested (gate bumped)

    with vetch.wrap(region="us-east-1"):
        client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}]
        )

    assert len(_emitter.events) == 1
    text = _emitter.events[0]["usage"]["text"]
    assert (text["input_tokens"], text["output_tokens"], text["total_tokens"]) == (50, 100, 150)


def test_async_client_patch_emits_event(_emitter):
    client = openai.AsyncOpenAI(api_key="test-key")
    client.chat.completions.create = AsyncMock(
        return_value=_response(prompt=11, completion=22, total=33)
    )

    assert patch_openai_client(client) is True

    async def _run():
        with vetch.wrap(region="us-east-1"):
            await client.chat.completions.create(
                model="gpt-4o", messages=[{"role": "user", "content": "hi"}]
            )

    asyncio.run(_run())

    assert len(_emitter.events) == 1
    text = _emitter.events[0]["usage"]["text"]
    assert (text["input_tokens"], text["output_tokens"], text["total_tokens"]) == (11, 22, 33)


def test_reinstrument_cycle_does_not_recurse(_emitter):
    """Regression: instrument -> uninstrument -> instrument, then build a real
    client. Previously raised RecursionError in patched_init."""
    try:
        assert instrument_openai_module() is True
        assert uninstrument_openai_module() is True
        assert instrument_openai_module() is True
        # Must not recurse; module auto-patches the new instance.
        client = openai.OpenAI(api_key="test-key")
        create = client.chat.completions.create
        assert getattr(create, "vetch_patched", False) is True
    finally:
        uninstrument_openai_module()


def test_double_instrument_is_idempotent(_emitter):
    """Calling instrument twice must not re-wrap __init__ into self-recursion."""
    try:
        assert instrument_openai_module() is True
        first_init = openai.OpenAI.__init__
        assert instrument_openai_module() is True
        # Idempotent: __init__ unchanged, and constructing a client is safe.
        assert openai.OpenAI.__init__ is first_init
        openai.OpenAI(api_key="test-key")
    finally:
        uninstrument_openai_module()
