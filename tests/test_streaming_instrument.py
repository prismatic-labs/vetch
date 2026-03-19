"""Tests for streaming auto-instrumentation (instrument() without wrap()).

Verifies that streaming calls emit events when using instrument() alone,
and that manual wrap() still works correctly with no double-emission.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

import vetch
from vetch.emitter import BufferedEmitter, set_test_emitter
from vetch.providers.anthropic import (
    AsyncStreamWrapper as AnthropicAsyncStreamWrapper,
)
from vetch.providers.anthropic import (
    StreamWrapper as AnthropicStreamWrapper,
)
from vetch.providers.openai import AsyncStreamWrapper
from vetch.providers.openai import StreamWrapper as OpenAIStreamWrapper
from vetch.providers.vertexai import (
    StreamWrapper as VertexStreamWrapper,
)


def _make_openai_chunk(model: str = "gpt-4o", content: str = "hi", usage: bool = False) -> Mock:
    chunk = Mock()
    chunk.model = model
    chunk.choices = [Mock()]
    chunk.choices[0].delta = Mock()
    chunk.choices[0].delta.content = content
    if usage:
        chunk.usage = Mock()
        chunk.usage.prompt_tokens = 10
        chunk.usage.completion_tokens = 5
        chunk.usage.total_tokens = 15
        chunk.usage.prompt_tokens_details = None
    else:
        chunk.usage = None
    return chunk


def _make_anthropic_chunk(event_type: str = "content_block_delta", text: str = "hi") -> Mock:
    chunk = Mock()
    chunk.type = event_type
    if event_type == "message_start":
        chunk.message = Mock()
        chunk.message.model = "claude-3-5-sonnet-20241022"
        chunk.message.usage = Mock()
        chunk.message.usage.input_tokens = 10
        chunk.message.usage.cache_read_input_tokens = None
        chunk.message.usage.cache_creation_input_tokens = None
    elif event_type == "content_block_delta":
        chunk.delta = Mock()
        chunk.delta.text = text
    elif event_type == "message_delta":
        chunk.usage = Mock()
        chunk.usage.output_tokens = 5
    return chunk


def _make_vertex_chunk(text: str = "hi", with_usage: bool = False) -> Mock:
    chunk = Mock()
    chunk.text = text
    if with_usage:
        chunk.usage_metadata = Mock()
        chunk.usage_metadata.prompt_token_count = 10
        chunk.usage_metadata.candidates_token_count = 5
        chunk.usage_metadata.total_token_count = 15
    else:
        chunk.usage_metadata = None
    return chunk


@pytest.fixture(autouse=True)
def reset_vetch_state():
    """Reset vetch instrumentation state between tests."""
    emitter = BufferedEmitter()
    set_test_emitter(emitter)

    original_instrumented = vetch._instrumented
    original_region = vetch._default_region
    original_tags = vetch._default_tags

    yield emitter

    set_test_emitter(None)
    vetch._instrumented = original_instrumented
    vetch._default_region = original_region
    vetch._default_tags = original_tags

    try:
        vetch.uninstrument()
    except Exception:
        pass


class TestOpenAIStreamingInstrumented:
    """OpenAI streaming with instrument() only (no wrap())."""

    def test_stream_emits_event_when_instrumented(self, reset_vetch_state: BufferedEmitter) -> None:
        """Stream completion emits event when instrument() is used without wrap()."""
        emitter = reset_vetch_state
        vetch._instrumented = True
        vetch._default_region = "us-east-1"
        vetch._default_tags = {"env": "test"}

        chunks = [
            _make_openai_chunk(content="Hello"),
            _make_openai_chunk(content=" world", usage=True),
        ]
        mock_stream = iter(chunks)

        wrapper = OpenAIStreamWrapper(mock_stream)
        for _ in wrapper:
            pass

        assert len(emitter.events) == 1
        event = emitter.events[0]
        assert event["provider"] == "openai"
        assert event["is_stream"] is True
        assert event["complete"] is True
        assert event["region"] == "us-east-1"

    def test_stream_with_manual_wrap_emits_once(self, reset_vetch_state: BufferedEmitter) -> None:
        """Stream inside manual wrap() emits exactly one event (no double-emit).

        The StreamWrapper captures to the active wrap() context on StopIteration.
        wrap() then emits a single event on __exit__. auto_context_for_instrumented_call
        is a no-op when a manual context is already active.
        """
        emitter = reset_vetch_state
        vetch._instrumented = True
        vetch._default_region = "us-east-1"
        vetch._default_tags = {}

        chunks = [_make_openai_chunk(content="hi")]
        mock_stream = iter(chunks)

        with vetch.wrap(region="us-east-1"):
            wrapper = OpenAIStreamWrapper(mock_stream)
            for _ in wrapper:
                pass
            # Stream wrapper calls _capture_to_context() → finds active wrap() context
            # and captures to it. wrap().__exit__ will emit the single event.

        assert len(emitter.events) == 1

    def test_stream_error_mid_iteration_emits_event(
        self, reset_vetch_state: BufferedEmitter
    ) -> None:
        """Stream that raises mid-iteration still emits an event with error=True."""
        emitter = reset_vetch_state
        vetch._instrumented = True
        vetch._default_region = "us-east-1"
        vetch._default_tags = {}

        def error_stream():
            yield _make_openai_chunk(content="partial")
            raise RuntimeError("connection lost")

        wrapper = OpenAIStreamWrapper(error_stream())
        with pytest.raises(RuntimeError):
            for _ in wrapper:
                pass

        assert len(emitter.events) == 1
        event = emitter.events[0]
        assert event["error"] is True
        assert event["error_type"] == "RuntimeError"

    def test_stream_no_double_emit_instrumented_plus_wrap(
        self, reset_vetch_state: BufferedEmitter
    ) -> None:
        """instrument() + wrap() together → exactly one event, not two."""
        emitter = reset_vetch_state
        vetch._instrumented = True
        vetch._default_region = "us-east-1"
        vetch._default_tags = {}

        # Simulate what happens when StreamWrapper._capture_to_context runs
        # while a manual wrap() context is active: it should use the existing
        # context, not create a new one.
        with vetch.wrap(region="us-east-1"):
            chunks = [_make_openai_chunk(content="hi")]
            mock_stream = iter(chunks)
            wrapper = OpenAIStreamWrapper(mock_stream)
            # Inside wrap(), _capture_to_context should find the active context
            # and NOT create an auto-context (so no double emit)
            for _ in wrapper:
                pass
            # The stream wrapper captures to the manual wrap() context
            # which emits on __exit__

        # The stream wrapper captures to wrap() context → 1 event from wrap().__exit__
        # No extra event from auto_context_for_instrumented_call
        assert len(emitter.events) == 1


class TestAnthropicStreamingInstrumented:
    """Anthropic streaming with instrument() only (no wrap())."""

    def test_stream_emits_event_when_instrumented(self, reset_vetch_state: BufferedEmitter) -> None:
        """Anthropic stream emits event when instrument() is used without wrap()."""
        emitter = reset_vetch_state
        vetch._instrumented = True
        vetch._default_region = "us-east-1"
        vetch._default_tags = {}

        chunks = [
            _make_anthropic_chunk("message_start"),
            _make_anthropic_chunk("content_block_delta", text="Hello world"),
            _make_anthropic_chunk("message_delta"),
        ]
        mock_stream = iter(chunks)

        wrapper = AnthropicStreamWrapper(mock_stream)
        for _ in wrapper:
            pass

        assert len(emitter.events) == 1
        event = emitter.events[0]
        assert event["provider"] == "anthropic"
        assert event["is_stream"] is True
        assert event["complete"] is True

    def test_stream_error_emits_event(self, reset_vetch_state: BufferedEmitter) -> None:
        """Anthropic stream error still emits an event."""
        emitter = reset_vetch_state
        vetch._instrumented = True
        vetch._default_region = "us-east-1"
        vetch._default_tags = {}

        def error_stream():
            yield _make_anthropic_chunk("message_start")
            raise ConnectionError("API error")

        wrapper = AnthropicStreamWrapper(error_stream())
        with pytest.raises(ConnectionError):
            for _ in wrapper:
                pass

        assert len(emitter.events) == 1
        assert emitter.events[0]["error"] is True


class TestVertexStreamingInstrumented:
    """Vertex AI streaming with instrument() only (no wrap())."""

    def test_stream_emits_event_when_instrumented(self, reset_vetch_state: BufferedEmitter) -> None:
        """Vertex AI stream emits event when instrument() is used without wrap()."""
        emitter = reset_vetch_state
        vetch._instrumented = True
        vetch._default_region = "us-central1"
        vetch._default_tags = {}

        chunks = [
            _make_vertex_chunk("Hello "),
            _make_vertex_chunk("world", with_usage=True),
        ]
        mock_stream = iter(chunks)

        wrapper = VertexStreamWrapper(mock_stream, model_name="gemini-2.0-flash")
        for _ in wrapper:
            pass

        assert len(emitter.events) == 1
        event = emitter.events[0]
        assert event["provider"] == "vertexai"
        assert event["is_stream"] is True
        assert event["model"] == "gemini-2.0-flash"

    def test_stream_error_emits_event(self, reset_vetch_state: BufferedEmitter) -> None:
        """Vertex AI stream error still emits an event."""
        emitter = reset_vetch_state
        vetch._instrumented = True
        vetch._default_region = "us-central1"
        vetch._default_tags = {}

        def error_stream():
            yield _make_vertex_chunk("partial")
            raise ValueError("model error")

        wrapper = VertexStreamWrapper(error_stream(), model_name="gemini-2.0-flash")
        with pytest.raises(ValueError):
            for _ in wrapper:
                pass

        assert len(emitter.events) == 1
        assert emitter.events[0]["error"] is True


class TestAsyncStreamingInstrumented:
    """Async streaming with instrument() only (no wrap())."""

    @pytest.mark.asyncio
    async def test_openai_async_stream_emits_event(
        self, reset_vetch_state: BufferedEmitter
    ) -> None:
        """OpenAI async stream emits event when instrument() is used."""
        emitter = reset_vetch_state
        vetch._instrumented = True
        vetch._default_region = "us-east-1"
        vetch._default_tags = {}

        async def async_chunks():
            for chunk in [
                _make_openai_chunk(content="Hello"),
                _make_openai_chunk(content=" world"),
            ]:
                yield chunk

        wrapper = AsyncStreamWrapper(async_chunks())
        async for _ in wrapper:
            pass

        assert len(emitter.events) == 1
        event = emitter.events[0]
        assert event["provider"] == "openai"
        assert event["is_stream"] is True

    @pytest.mark.asyncio
    async def test_anthropic_async_stream_emits_event(
        self, reset_vetch_state: BufferedEmitter
    ) -> None:
        """Anthropic async stream emits event when instrument() is used."""
        emitter = reset_vetch_state
        vetch._instrumented = True
        vetch._default_region = "us-east-1"
        vetch._default_tags = {}

        async def async_chunks():
            for chunk in [
                _make_anthropic_chunk("message_start"),
                _make_anthropic_chunk("content_block_delta", text="hi"),
                _make_anthropic_chunk("message_delta"),
            ]:
                yield chunk

        wrapper = AnthropicAsyncStreamWrapper(async_chunks())
        async for _ in wrapper:
            pass

        assert len(emitter.events) == 1
        assert emitter.events[0]["provider"] == "anthropic"
