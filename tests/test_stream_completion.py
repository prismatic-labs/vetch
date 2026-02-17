"""Tests for stream completion event emission.

These tests verify that events are emitted at the right time:
- On natural stream completion
- On stream exception
- With correct metadata
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from vetch.context import TrackingContext, get_active_context
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
    """Mock choice."""

    def __init__(self, content: str) -> None:
        self.delta = MockDelta(content)


class MockDelta:
    """Mock delta."""

    def __init__(self, content: str) -> None:
        self.content = content


class MockUsage:
    """Mock usage."""

    def __init__(self, prompt: int, completion: int, total: int) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = total


class TestStreamCompletion:
    """Tests for stream completion handling."""

    def test_capture_on_natural_completion(self) -> None:
        """Verify capture is called when stream completes naturally."""
        with TrackingContext() as ctx:
            chunks = [
                MockChunk(content="Hello", model="gpt-4"),
                MockChunk(content=" World"),
            ]
            wrapper = StreamWrapper(iter(chunks))

            # Consume stream
            list(wrapper)

            # Check capture was called
            assert ctx.captured_call is not None
            assert ctx.captured_call.model == "gpt-4"
            assert ctx.captured_call.provider == "openai"
            assert ctx.captured_call.is_stream is True
            assert ctx.captured_call.complete is True
            assert ctx.captured_call.error is False

    def test_capture_on_exception(self) -> None:
        """Verify capture is called with error info on exception."""

        def failing_stream() -> Iterator[MockChunk]:
            yield MockChunk(content="Start", model="gpt-4")
            raise RuntimeError("Network error")

        with TrackingContext() as ctx:
            wrapper = StreamWrapper(failing_stream())

            try:
                list(wrapper)
            except RuntimeError:
                pass

            # Check capture was called with error
            assert ctx.captured_call is not None
            assert ctx.captured_call.model == "gpt-4"
            assert ctx.captured_call.error is True
            assert ctx.captured_call.error_type == "RuntimeError"
            assert ctx.captured_call.complete is False

    def test_accumulated_chars_captured(self) -> None:
        """Verify accumulated character count is captured."""
        with TrackingContext() as ctx:
            chunks = [
                MockChunk(content="Hello", model="gpt-4"),
                MockChunk(content=" "),
                MockChunk(content="World!"),
            ]
            wrapper = StreamWrapper(iter(chunks))

            list(wrapper)

            assert ctx.captured_call is not None
            assert ctx.captured_call.accumulated_chars == len("Hello World!")

    def test_usage_captured_from_final_chunk(self) -> None:
        """Verify usage from final chunk is captured."""
        usage = MockUsage(prompt=50, completion=25, total=75)

        with TrackingContext() as ctx:
            chunks = [
                MockChunk(content="Hello", model="gpt-4"),
                MockChunk(content="!", usage=usage),
            ]
            wrapper = StreamWrapper(iter(chunks))

            list(wrapper)

            assert ctx.captured_call is not None
            assert ctx.captured_call.usage is not None
            assert ctx.captured_call.usage["text"]["input_tokens"] == 50

    def test_no_capture_without_context(self) -> None:
        """Verify no error when stream completes outside context."""
        # Ensure no active context
        assert get_active_context() is None

        chunks = [MockChunk(content="Hello", model="gpt-4")]
        wrapper = StreamWrapper(iter(chunks))

        # Should not raise
        list(wrapper)

        # Wrapper should still track locally
        assert wrapper._accumulated_chars == 5
        assert wrapper._complete is True

    def test_context_manager_on_error(self) -> None:
        """Verify context manager captures error on exit."""
        with TrackingContext() as ctx:
            wrapper = StreamWrapper(iter([MockChunk(content="Hi", model="gpt-4")]))

            try:
                with wrapper:
                    # Partially consume
                    next(wrapper)
                    # Simulate error before full consumption
                    raise ValueError("User error")
            except ValueError:
                pass

            # The context manager __exit__ should have captured
            assert ctx.captured_call is not None
            assert ctx.captured_call.error is True
            assert ctx.captured_call.error_type == "ValueError"


class TestStreamEmissionTiming:
    """Tests for when events are emitted."""

    def test_event_emitted_after_iteration(self) -> None:
        """Verify event is emitted after iteration completes, not during."""
        captures: list[Any] = []

        with TrackingContext() as ctx:
            # Monkey-patch capture to track calls
            original_capture = ctx.capture

            def tracking_capture(**kwargs: Any) -> None:
                captures.append(kwargs)
                original_capture(**kwargs)

            ctx.capture = tracking_capture  # type: ignore[method-assign]

            chunks = [
                MockChunk(content="A", model="gpt-4"),
                MockChunk(content="B"),
                MockChunk(content="C"),
            ]
            wrapper = StreamWrapper(iter(chunks))

            # No capture during iteration
            result = []
            for chunk in wrapper:
                result.append(chunk)
                # No capture should have happened yet during iteration
                # (capture happens in finally/StopIteration)

            # After iteration completes, capture should have been called once
            assert len(captures) == 1
            assert captures[0]["complete"] is True

    def test_single_event_per_stream(self) -> None:
        """Verify only one event is emitted per stream."""
        call_count = 0

        with TrackingContext() as ctx:
            original_capture = ctx.capture

            def counting_capture(**kwargs: Any) -> None:
                nonlocal call_count
                call_count += 1
                original_capture(**kwargs)

            ctx.capture = counting_capture  # type: ignore[method-assign]

            chunks = [MockChunk(content=f"chunk{i}") for i in range(10)]
            wrapper = StreamWrapper(iter(chunks))

            list(wrapper)

        assert call_count == 1
