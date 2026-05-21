"""Tests for OpenAI provider wrapper.

These tests verify:
- Usage extraction from OpenAI responses
- Model name extraction
- Streaming with memory safety
- Patching and unpatching behavior
"""

from __future__ import annotations

from typing import Any

from vetch.context import TrackingContext
from vetch.providers.openai import (
    StreamWrapper,
    extract_response_diagnostics,
    extract_usage,
)


class MockUsage:
    """Mock OpenAI usage object."""

    def __init__(self, prompt: int, completion: int, total: int) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = total


class MockResponse:
    """Mock OpenAI response."""

    def __init__(
        self,
        model: str = "gpt-4",
        usage: MockUsage | None = None,
        content: str | None = None,
        finish_reason: str | None = None,
    ) -> None:
        self.model = model
        self.usage = usage
        self.choices = [MockChoice(content, finish_reason=finish_reason)]


class MockChunk:
    """Mock OpenAI streaming chunk."""

    def __init__(
        self,
        content: str | None = None,
        model: str | None = None,
        usage: MockUsage | None = None,
    ) -> None:
        self.model = model
        self.usage = usage
        if content is not None:
            self.choices = [MockChoice(content)]
        else:
            self.choices = []


class MockChoice:
    """Mock OpenAI choice."""

    def __init__(self, content: str | None, finish_reason: str | None = None) -> None:
        self.delta = MockDelta(content)
        self.message = MockMessage(content)
        self.finish_reason = finish_reason


class MockDelta:
    """Mock OpenAI delta."""

    def __init__(self, content: str | None) -> None:
        self.content = content


class MockMessage:
    """Mock OpenAI message."""

    def __init__(self, content: str | None) -> None:
        self.content = content


class TestExtractUsage:
    """Tests for usage extraction from OpenAI responses."""

    def test_extract_usage_with_usage(self) -> None:
        """Extract usage when present."""
        usage = MockUsage(prompt=100, completion=50, total=150)
        response = MockResponse(usage=usage)

        result, cache_read, cache_create = extract_usage(response)

        assert result is not None
        assert result["text"]["input_tokens"] == 100
        assert result["text"]["output_tokens"] == 50
        assert result["text"]["total_tokens"] == 150
        assert cache_read is None  # No cache tokens in mock
        assert cache_create is None

    def test_extract_usage_no_usage(self) -> None:
        """Return None when no usage."""
        response = MockResponse(usage=None)

        result, cache_read, cache_create = extract_usage(response)

        assert result is None
        assert cache_read is None
        assert cache_create is None

    def test_extract_usage_missing_attribute(self) -> None:
        """Handle missing usage attribute."""

        class NoUsage:
            model = "gpt-4"

        result, cache_read, cache_create = extract_usage(NoUsage())

        assert result is None
        assert cache_read is None
        assert cache_create is None


class TestExtractResponseDiagnostics:
    """Tests for privacy-safe response diagnostics."""

    def test_extract_visible_chars_and_finish_reason(self) -> None:
        response = MockResponse(content="Hello world", finish_reason="stop")

        visible_chars, finish_reason = extract_response_diagnostics(response)

        assert visible_chars == 11
        assert finish_reason == "stop"

    def test_extract_zero_visible_chars(self) -> None:
        response = MockResponse(content="", finish_reason="length")

        visible_chars, finish_reason = extract_response_diagnostics(response)

        assert visible_chars == 0
        assert finish_reason == "length"


class TestStreamWrapper:
    """Tests for OpenAI stream wrapper."""

    def test_stream_yields_all_chunks(self) -> None:
        """Stream yields all original chunks."""
        chunks = [
            MockChunk(content="Hello", model="gpt-4"),
            MockChunk(content=" World"),
            MockChunk(content="!"),
        ]
        wrapper = StreamWrapper(iter(chunks))

        result = list(wrapper)

        assert len(result) == 3
        assert result[0].choices[0].delta.content == "Hello"

    def test_stream_counts_chars(self) -> None:
        """Stream counts accumulated characters."""
        chunks = [
            MockChunk(content="Hello"),
            MockChunk(content=" World"),
            MockChunk(content="!"),
        ]
        wrapper = StreamWrapper(iter(chunks))

        list(wrapper)

        assert wrapper._accumulated_chars == 12

    def test_stream_captures_model(self) -> None:
        """Stream captures model from first chunk."""
        chunks = [
            MockChunk(content="Hello", model="gpt-4o"),
            MockChunk(content="!"),
        ]
        wrapper = StreamWrapper(iter(chunks))

        list(wrapper)

        assert wrapper._model == "gpt-4o"

    def test_stream_captures_final_usage(self) -> None:
        """Stream captures usage from final chunk."""
        usage = MockUsage(prompt=100, completion=50, total=150)
        chunks = [
            MockChunk(content="Hello"),
            MockChunk(content="!", usage=usage),
        ]
        wrapper = StreamWrapper(iter(chunks))

        list(wrapper)

        assert wrapper._final_usage is not None
        assert wrapper._final_usage["text"]["input_tokens"] == 100

    def test_stream_handles_empty_choices(self) -> None:
        """Stream handles chunks with no choices."""
        chunks = [
            MockChunk(content="Hello"),
            MockChunk(),  # No choices
            MockChunk(content="World"),
        ]
        wrapper = StreamWrapper(iter(chunks))

        list(wrapper)

        assert wrapper._accumulated_chars == 10

    def test_stream_handles_none_content(self) -> None:
        """Stream handles None content."""
        chunks = [
            MockChunk(content=None),
            MockChunk(content="Hello"),
            MockChunk(content=None),
        ]
        wrapper = StreamWrapper(iter(chunks))

        list(wrapper)

        assert wrapper._accumulated_chars == 5

    def test_stream_complete_flag(self) -> None:
        """Stream sets complete flag on natural finish."""
        chunks = [MockChunk(content="Done")]
        wrapper = StreamWrapper(iter(chunks))

        assert wrapper._complete is False

        list(wrapper)

        assert wrapper._complete is True
        assert wrapper._error is False

    def test_stream_error_flag(self) -> None:
        """Stream sets error flag on exception."""

        def failing_stream() -> Any:
            yield MockChunk(content="Start")
            raise ValueError("Stream error")

        wrapper = StreamWrapper(failing_stream())

        try:
            list(wrapper)
        except ValueError:
            pass

        assert wrapper._error is True
        assert wrapper._error_type == "ValueError"
        assert wrapper._complete is False

    def test_stream_captures_to_context(self) -> None:
        """Stream captures to active context on completion."""
        with TrackingContext() as ctx:
            chunks = [MockChunk(content="Hello", model="gpt-4")]
            wrapper = StreamWrapper(iter(chunks))

            list(wrapper)

            assert ctx.captured_call is not None
            assert ctx.captured_call.model == "gpt-4"
            assert ctx.captured_call.provider == "openai"
            assert ctx.captured_call.is_stream is True
            assert ctx.captured_call.accumulated_chars == 5

    def test_stream_captures_error_to_context(self) -> None:
        """Stream captures error to context on exception."""

        def failing_stream() -> Any:
            yield MockChunk(content="Start", model="gpt-4")
            raise RuntimeError("Network error")

        with TrackingContext() as ctx:
            wrapper = StreamWrapper(failing_stream())

            try:
                list(wrapper)
            except RuntimeError:
                pass

            assert ctx.captured_call is not None
            assert ctx.captured_call.error is True
            assert ctx.captured_call.error_type == "RuntimeError"

    def test_stream_context_manager_protocol(self) -> None:
        """Stream supports context manager protocol."""
        chunks = [MockChunk(content="Hello")]
        wrapper = StreamWrapper(iter(chunks))

        with wrapper as w:
            result = list(w)

        assert len(result) == 1
        assert wrapper._complete is True

    def test_stream_context_manager_on_error(self) -> None:
        """Context manager captures error on exit."""
        with TrackingContext() as ctx:
            chunks = [MockChunk(content="Hi", model="gpt-4")]
            wrapper = StreamWrapper(iter(chunks))

            try:
                with wrapper:
                    next(wrapper)
                    raise ValueError("User error")
            except ValueError:
                pass

            assert ctx.captured_call is not None
            assert ctx.captured_call.error is True
            assert ctx.captured_call.error_type == "ValueError"

    def test_stream_iterator_protocol(self) -> None:
        """Stream implements iterator protocol correctly."""
        chunks = [MockChunk(content="A"), MockChunk(content="B")]
        wrapper = StreamWrapper(iter(chunks))

        # __iter__ returns self
        assert iter(wrapper) is wrapper

        # __next__ works
        first = next(wrapper)
        assert first.choices[0].delta.content == "A"


class TestStreamWrapperNoContext:
    """Tests for stream behavior without active context."""

    def test_stream_works_without_context(self) -> None:
        """Stream functions normally without active context."""
        chunks = [MockChunk(content="Hello")]
        wrapper = StreamWrapper(iter(chunks))

        result = list(wrapper)

        assert len(result) == 1
        assert wrapper._accumulated_chars == 5
        assert wrapper._complete is True

    def test_stream_error_without_context(self) -> None:
        """Stream error handling works without context."""

        def failing_stream() -> Any:
            yield MockChunk(content="Start")
            raise ValueError("Error")

        wrapper = StreamWrapper(failing_stream())

        try:
            list(wrapper)
        except ValueError:
            pass

        assert wrapper._error is True
        assert wrapper._error_type == "ValueError"


class TestEmbeddingsExtraction:
    """Tests for embeddings usage extraction."""

    def test_extract_embeddings_usage(self) -> None:
        """Extract usage from embeddings response."""
        from vetch.providers.openai import extract_embeddings_usage

        usage = MockUsage(prompt=50, completion=0, total=50)
        response = MockResponse(model="text-embedding-3-small", usage=usage)

        result = extract_embeddings_usage(response)

        assert result is not None
        assert result["text"]["input_tokens"] == 50
        assert result["text"]["output_tokens"] == 0  # Embeddings don't generate output
        assert result["text"]["total_tokens"] == 50

    def test_extract_embeddings_usage_no_usage(self) -> None:
        """Return None when usage is missing."""
        from vetch.providers.openai import extract_embeddings_usage

        response = MockResponse(model="text-embedding-3-small", usage=None)

        result = extract_embeddings_usage(response)

        assert result is None

    def test_embeddings_capture_sets_is_embedding_flag(self) -> None:
        """Embeddings capture should set is_embedding=True."""
        from vetch.providers.openai import _after_embeddings_create

        with TrackingContext(region="us-east-1") as ctx:
            usage = MockUsage(prompt=100, completion=0, total=100)
            response = MockResponse(model="text-embedding-3-small", usage=usage)

            _after_embeddings_create(response)

            assert ctx.captured_call is not None
            assert ctx.captured_call.model == "text-embedding-3-small"
            assert ctx.captured_call.provider == "openai"
            assert ctx.captured_call.is_embedding is True
            assert ctx.captured_call.complete is True

    def test_embeddings_error_sets_flag(self) -> None:
        """Embeddings error capture should set is_embedding=True."""
        from vetch.providers.openai import _on_embeddings_error

        with TrackingContext(region="us-east-1") as ctx:
            error = ValueError("API error")
            _on_embeddings_error(error)

            assert ctx.captured_call is not None
            assert ctx.captured_call.is_embedding is True
            assert ctx.captured_call.error is True
            assert ctx.captured_call.error_type == "ValueError"
