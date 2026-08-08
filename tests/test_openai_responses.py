"""Tests for OpenAI Responses API instrumentation.

Covers:
- Usage extraction from the Responses shape (input/output/reasoning/cache tokens)
- Sync + async patching of responses.create AND responses.parse
- Non-zero usage assertion on a realistic parse fixture (the customer's path)
- Streaming aggregation into a single event
- is_client_instrumented coverage of responses
- Unpatch restoring a working (bound) client — fail-open regression
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
    AsyncResponsesStreamWrapper,
    ResponsesStreamWrapper,
    extract_responses_diagnostics,
    extract_responses_usage,
    patch_openai_client,
    uninstrument_openai_module,
    unpatch_openai_client,
)


@pytest.fixture(autouse=True)
def _emitter():
    emitter = BufferedEmitter()
    set_test_emitter(emitter)
    yield emitter
    set_test_emitter(None)
    try:
        uninstrument_openai_module()
    except Exception:
        pass


def _response(model="gpt-5.4", inp=120, out=80, reasoning=30, cached=0, status="completed"):
    """Realistic non-streaming Responses object."""
    usage = NS(
        input_tokens=inp,
        output_tokens=out,
        total_tokens=inp + out,
        input_tokens_details=NS(cached_tokens=cached),
        output_tokens_details=NS(reasoning_tokens=reasoning),
    )
    return NS(model=model, usage=usage, status=status, output_text="hello world")


class TestExtractResponsesUsage:
    def test_reasoning_subtracted_from_visible_output(self):
        usage, cache_read, cache_create = extract_responses_usage(_response())
        assert usage is not None
        assert usage["text"] == {"input_tokens": 120, "output_tokens": 50, "total_tokens": 200}
        assert usage["reasoning"]["output_tokens"] == 30
        assert cache_read == 0
        assert cache_create is None

    def test_cache_read_tokens_extracted(self):
        usage, cache_read, _ = extract_responses_usage(_response(cached=64))
        assert cache_read == 64

    def test_no_reasoning_block_when_zero(self):
        usage, _, _ = extract_responses_usage(_response(reasoning=0))
        assert "reasoning" not in usage
        assert usage["text"]["output_tokens"] == 80

    def test_none_usage_returns_none(self):
        usage, cache_read, cache_create = extract_responses_usage(NS(usage=None))
        assert usage is None and cache_read is None and cache_create is None

    def test_diagnostics_status_and_chars(self):
        chars, finish = extract_responses_diagnostics(_response(status="incomplete"))
        assert chars == len("hello world")
        assert finish == "incomplete"


class TestResponsesPatching:
    def test_sync_parse_emits_event_with_nonzero_usage(self, _emitter):
        client = openai.OpenAI(api_key="test-key")
        client.responses.parse = MagicMock(return_value=_response())
        assert patch_openai_client(client) is True

        with vetch.wrap(region="us-east-1"):
            client.responses.parse(model="gpt-5.4", input="hi")

        assert len(_emitter.events) == 1
        ev = _emitter.events[0]
        text = ev["usage"]["text"]
        assert (text["input_tokens"], text["output_tokens"]) == (120, 50)
        assert ev["usage"]["reasoning"]["output_tokens"] == 30
        # AC: usage is non-zero, not just "an event emitted".
        assert text["input_tokens"] > 0 and text["output_tokens"] > 0

    def test_sync_create_emits_event(self, _emitter):
        client = openai.OpenAI(api_key="test-key")
        client.responses.create = MagicMock(return_value=_response(inp=10, out=20, reasoning=0))
        assert patch_openai_client(client) is True

        with vetch.wrap(region="us-east-1"):
            client.responses.create(model="gpt-5.4", input="hi")

        assert len(_emitter.events) == 1
        assert _emitter.events[0]["usage"]["text"] == {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
        }

    def test_async_create_emits_event(self, _emitter):
        client = openai.AsyncOpenAI(api_key="test-key")
        client.responses.create = AsyncMock(
            return_value=_response(inp=11, out=22, reasoning=0)
        )
        assert patch_openai_client(client) is True

        async def _run():
            with vetch.wrap(region="us-east-1"):
                await client.responses.create(model="gpt-5.4", input="hi")

        asyncio.run(_run())
        assert len(_emitter.events) == 1
        assert _emitter.events[0]["usage"]["text"]["total_tokens"] == 33

    def test_async_parse_emits_event(self, _emitter):
        client = openai.AsyncOpenAI(api_key="test-key")
        client.responses.parse = AsyncMock(return_value=_response(inp=7, out=13, reasoning=3))
        assert patch_openai_client(client) is True

        async def _run():
            with vetch.wrap(region="us-east-1"):
                await client.responses.parse(model="gpt-5.4", input="hi")

        asyncio.run(_run())
        assert len(_emitter.events) == 1
        assert _emitter.events[0]["usage"]["text"]["output_tokens"] == 10  # 13 - 3

    def test_error_path_captures_and_reraises(self, _emitter):
        client = openai.OpenAI(api_key="test-key")
        client.responses.create = MagicMock(side_effect=RuntimeError("boom"))
        assert patch_openai_client(client) is True

        with vetch.wrap(region="us-east-1"):
            with pytest.raises(RuntimeError, match="boom"):
                client.responses.create(model="gpt-5.4", input="hi")

        assert len(_emitter.events) == 1
        assert _emitter.events[0].get("error") is True
        assert _emitter.events[0].get("error_type") == "RuntimeError"

    def test_is_client_instrumented_sees_responses(self):
        client = openai.OpenAI(api_key="test-key")
        assert vetch.is_client_instrumented(client) is False
        patch_openai_client(client)
        assert vetch.is_client_instrumented(client) is True

    def test_unpatch_restores_working_bound_client(self):
        """Fail-open: after unpatch a real client must still dispatch (self bound)."""
        client = openai.OpenAI(api_key="test-key")
        responses = client.responses
        patch_openai_client(client)
        assert getattr(responses.create, "vetch_patched", False) is True
        assert getattr(responses.parse, "vetch_patched", False) is True

        unpatch_openai_client(client)
        # Class-method dispatch restored: bound methods, no instance shadow with
        # an unbound function that would map the first kwarg onto `self`.
        assert hasattr(responses.create, "__self__")
        assert hasattr(responses.parse, "__self__")
        assert "create" not in responses.__dict__
        assert "parse" not in responses.__dict__


class TestMeteringFailOpen:
    """Metering must never break the host call (CLAUDE.md #1 non-negotiable).
    response.output_text is a computed SDK property that can raise on
    tool-call-only / refusal responses; if it does, the caller must still get
    its result.
    """

    def test_responses_metering_exception_does_not_crash_host(self, _emitter):
        class BadResponse:
            model = "gpt-5.4"
            status = "completed"
            usage = NS(
                input_tokens=10,
                output_tokens=20,
                total_tokens=30,
                input_tokens_details=NS(cached_tokens=0),
                output_tokens_details=NS(reasoning_tokens=0),
            )

            @property
            def output_text(self):  # realistic: property raises
                raise RuntimeError("output_text boom")

        client = openai.OpenAI(api_key="test-key")
        sentinel = BadResponse()
        client.responses.parse = MagicMock(return_value=sentinel)
        patch_openai_client(client)

        with vetch.wrap(region="us-east-1"):
            result = client.responses.parse(model="gpt-5.4", input="hi")

        assert result is sentinel  # host got its result; no exception propagated

    def test_chat_metering_exception_does_not_crash_host(self, _emitter):
        """Same guarantee for the shared chat hook (project-wide hardening)."""

        class BadChat:
            model = "gpt-4o"
            usage = NS(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
                prompt_tokens_details=None,
                completion_tokens_details=None,
            )

            @property
            def choices(self):
                raise RuntimeError("choices boom")

        client = openai.OpenAI(api_key="test-key")
        sentinel = BadChat()
        client.chat.completions.create = MagicMock(return_value=sentinel)
        patch_openai_client(client)

        with vetch.wrap(region="us-east-1"):
            result = client.chat.completions.create(model="gpt-4o", messages=[])

        assert result is sentinel

    def test_negative_visible_output_clamped_to_zero(self):
        usage, _, _ = extract_responses_usage(
            NS(
                usage=NS(
                    input_tokens=10,
                    output_tokens=5,
                    total_tokens=15,
                    input_tokens_details=NS(cached_tokens=0),
                    output_tokens_details=NS(reasoning_tokens=20),  # > output
                )
            )
        )
        assert usage["text"]["output_tokens"] == 0
        assert usage["reasoning"]["output_tokens"] == 20


class TestUnpatchFailOpen:
    """Shared restore helper must leave a working (bound) client after unpatch,
    for chat completions and embeddings as well as responses. Regression for the
    latent bug where the stored unbound function was restored as an instance
    attribute, mapping the next call's first keyword onto ``self``.
    """

    def test_chat_completions_unpatch_restores_bound(self):
        client = openai.OpenAI(api_key="test-key")
        completions = client.chat.completions
        patch_openai_client(client)
        assert getattr(completions.create, "vetch_patched", False) is True

        unpatch_openai_client(client)
        assert hasattr(completions.create, "__self__")  # bound, not raw function
        assert "create" not in completions.__dict__  # no unbound shadow left

    def test_embeddings_unpatch_restores_bound(self):
        client = openai.OpenAI(api_key="test-key")
        embeddings = client.embeddings
        patch_openai_client(client)
        assert getattr(embeddings.create, "vetch_patched", False) is True

        unpatch_openai_client(client)
        assert hasattr(embeddings.create, "__self__")
        assert "create" not in embeddings.__dict__

    def test_module_uninstrument_restores_bound_chat(self):
        from vetch.providers.openai import instrument_openai_module

        assert instrument_openai_module() is True
        client = openai.OpenAI(api_key="test-key")  # auto-patched via __init__
        assert getattr(client.chat.completions.create, "vetch_patched", False) is True
        uninstrument_openai_module()
        assert hasattr(client.chat.completions.create, "__self__")


class TestResponsesStreaming:
    def test_stream_aggregates_single_event(self, _emitter):
        # Text delta events, then a terminal completed event carrying usage.
        events = [
            NS(delta="Hel"),
            NS(delta="lo!"),
            NS(response=_response(inp=40, out=15, reasoning=5)),
        ]
        wrapper = ResponsesStreamWrapper(iter(events), model_hint="gpt-5.4")
        with vetch.wrap(region="us-east-1"):
            collected = list(wrapper)

        assert len(collected) == 3
        assert len(_emitter.events) == 1
        text = _emitter.events[0]["usage"]["text"]
        assert text["input_tokens"] == 40
        assert text["output_tokens"] == 10  # 15 output - 5 reasoning

    def test_stream_without_usage_falls_back_to_chars(self, _emitter):
        events = [NS(delta="abcd"), NS(delta="efgh")]
        wrapper = ResponsesStreamWrapper(iter(events), model_hint="gpt-5.4")
        with vetch.wrap(region="us-east-1"):
            list(wrapper)
        # One event still emits; char count is available for estimation.
        assert len(_emitter.events) == 1

    def test_wrapper_forwards_response_attr(self, _emitter):
        """The SDK ResponseStream reads raw_stream.response; must be forwarded."""
        underlying = NS(response=NS(closed=False), close=lambda: None)
        wrapper = ResponsesStreamWrapper(underlying, model_hint="gpt-5.4")
        assert wrapper.response is underlying.response
        # close() is forwarded too (used by the streaming manager teardown).
        assert callable(wrapper.close)

    def test_backs_real_responsestream_manager(self, _emitter):
        """Regression for client.responses.stream(): the SDK wraps the object
        returned by create(stream=True) in a ResponseStream whose __init__ reads
        raw_stream.response. Before attribute forwarding this raised
        AttributeError. Constructing it here reproduces that path."""
        rs_mod = pytest.importorskip("openai.lib.streaming.responses._responses")
        try:
            from openai import omit  # openai >= 2.x sentinel
        except ImportError:  # pragma: no cover - older SDKs
            from openai import NOT_GIVEN as omit  # type: ignore[no-redef]

        httpx_response = NS(close=lambda: None)
        underlying = NS(response=httpx_response)
        wrapper = ResponsesStreamWrapper(iter([]), model_hint="gpt-5.4")
        wrapper._stream = underlying

        stream = rs_mod.ResponseStream(
            raw_stream=wrapper, text_format=omit, input_tools=omit, starting_after=None
        )
        assert stream._response is httpx_response

    def test_async_stream_aggregates_single_event(self, _emitter):
        events = [
            NS(delta="Hi "),
            NS(delta="there"),
            NS(response=_response(inp=25, out=12, reasoning=2)),
        ]

        class _AsyncIter:
            def __init__(self, items):
                self._it = iter(items)

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return next(self._it)
                except StopIteration:
                    raise StopAsyncIteration from None

        wrapper = AsyncResponsesStreamWrapper(_AsyncIter(events), model_hint="gpt-5.4")

        async def _run():
            collected = []
            with vetch.wrap(region="us-east-1"):
                async for ev in wrapper:
                    collected.append(ev)
            return collected

        collected = asyncio.run(_run())
        assert len(collected) == 3
        assert len(_emitter.events) == 1
        assert _emitter.events[0]["usage"]["text"]["output_tokens"] == 10  # 12 - 2
