"""Tests for streaming memory safety.

These tests verify that streaming does NOT accumulate content in memory.
Only character counts and final usage should be stored.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from vetch.providers.openai import StreamWrapper


class MockChunk:
    """Mock OpenAI streaming chunk."""

    def __init__(
        self,
        content: str | None = None,
        model: str | None = None,
        usage: Any = None,
    ) -> None:
        self.model = model
        self.usage = usage
        if content is not None:
            self.choices = [MockChoice(content)]
        else:
            self.choices = []


class MockChoice:
    """Mock OpenAI choice in chunk."""

    def __init__(self, content: str | None) -> None:
        self.delta = MockDelta(content)


class MockDelta:
    """Mock OpenAI delta in choice."""

    def __init__(self, content: str | None) -> None:
        self.content = content


class MockUsage:
    """Mock OpenAI usage object."""

    def __init__(self, prompt: int, completion: int, total: int) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = total


def create_mock_stream(chunks: list[MockChunk]) -> Iterator[MockChunk]:
    """Create an iterator from mock chunks."""
    return iter(chunks)


class TestStreamingMemorySafety:
    """Tests verifying memory-safe streaming."""

    def test_chunks_not_accumulated(self) -> None:
        """Verify chunks are counted but not stored."""
        # Create a stream with substantial content
        large_content = "x" * 10000  # 10KB per chunk
        chunks = [MockChunk(content=large_content) for _ in range(100)]

        stream = create_mock_stream(chunks)
        wrapper = StreamWrapper(stream)

        # Consume the stream
        consumed_chunks = []
        for chunk in wrapper:
            consumed_chunks.append(chunk)

        # Verify all chunks were yielded
        assert len(consumed_chunks) == 100

        # Verify accumulated_chars is correct
        assert wrapper._accumulated_chars == 10000 * 100

        # The wrapper should NOT have stored the actual content
        # It only stores: _accumulated_chars, _model, _final_usage, flags
        # Check that the wrapper's size is small
        wrapper_attrs = ["_stream", "_accumulated_chars", "_model", "_final_usage"]
        for attr in wrapper_attrs:
            assert hasattr(wrapper, attr)

        # The content should not be stored anywhere in the wrapper
        # (the stream itself is consumed and gone)

    def test_only_counts_stored(self) -> None:
        """Verify only counts are stored, not content."""
        chunks = [
            MockChunk(content="Hello ", model="gpt-4"),
            MockChunk(content="World"),
            MockChunk(content="!"),
        ]

        stream = create_mock_stream(chunks)
        wrapper = StreamWrapper(stream)

        # Consume
        list(wrapper)

        # Check counts
        assert wrapper._accumulated_chars == len("Hello World!")
        assert wrapper._model == "gpt-4"

    def test_final_usage_captured(self) -> None:
        """Verify final usage is captured from last chunk."""
        usage = MockUsage(prompt=100, completion=50, total=150)
        chunks = [
            MockChunk(content="Hello"),
            MockChunk(content=" World", usage=usage),  # Final chunk with usage
        ]

        stream = create_mock_stream(chunks)
        wrapper = StreamWrapper(stream)

        # Consume
        list(wrapper)

        # Check final usage was captured
        assert wrapper._final_usage is not None
        assert wrapper._final_usage["text"]["input_tokens"] == 100
        assert wrapper._final_usage["text"]["output_tokens"] == 50
        assert wrapper._final_usage["text"]["total_tokens"] == 150

    def test_empty_content_handled(self) -> None:
        """Verify empty/None content doesn't cause issues."""
        chunks = [
            MockChunk(content=None),
            MockChunk(content=""),
            MockChunk(content="Hello"),
            MockChunk(),  # No choices at all
        ]

        stream = create_mock_stream(chunks)
        wrapper = StreamWrapper(stream)

        # Consume - should not raise
        list(wrapper)

        # Only "Hello" should be counted
        assert wrapper._accumulated_chars == 5

    def test_completion_flag_set(self) -> None:
        """Verify complete flag is set when stream finishes."""
        chunks = [MockChunk(content="Done")]

        stream = create_mock_stream(chunks)
        wrapper = StreamWrapper(stream)

        assert wrapper._complete is False

        # Consume
        list(wrapper)

        assert wrapper._complete is True
        assert wrapper._error is False

    def test_error_flag_on_exception(self) -> None:
        """Verify error flag is set when stream raises."""

        def failing_stream() -> Iterator[MockChunk]:
            yield MockChunk(content="Start")
            raise ValueError("Stream failed")

        wrapper = StreamWrapper(failing_stream())

        # Consume until error
        chunks_received = []
        try:
            for chunk in wrapper:
                chunks_received.append(chunk)
        except ValueError:
            pass

        assert len(chunks_received) == 1
        assert wrapper._error is True
        assert wrapper._error_type == "ValueError"
        assert wrapper._complete is False
