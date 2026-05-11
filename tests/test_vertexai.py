"""Tests for Vertex AI provider wrapper.

These tests verify:
- Usage extraction from Vertex AI responses
- Model name extraction and cleanup
- Region inference from endpoints
- Streaming with memory safety
"""

from __future__ import annotations

from typing import Any

from vetch.context import TrackingContext
from vetch.providers.vertexai import (
    StreamWrapper,
    extract_model,
    extract_usage,
    infer_region_from_endpoint,
)


class MockUsageMetadata:
    """Mock Vertex AI usage metadata."""

    def __init__(
        self,
        prompt: int,
        candidates: int,
        total: int,
        thoughts: int = 0,
    ) -> None:
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates
        self.total_token_count = total
        self.thoughts_token_count = thoughts


class MockResponse:
    """Mock Vertex AI response."""

    def __init__(self, usage: MockUsageMetadata | None = None) -> None:
        self.usage_metadata = usage


class MockModel:
    """Mock Vertex AI GenerativeModel."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name


class MockChunk:
    """Mock Vertex AI streaming chunk."""

    def __init__(
        self,
        text: str | None = None,
        usage_metadata: MockUsageMetadata | None = None,
    ) -> None:
        self.text = text
        self.usage_metadata = usage_metadata


class TestExtractUsage:
    """Tests for usage extraction."""

    def test_extract_usage_with_metadata(self) -> None:
        """Extract usage when metadata is present."""
        usage_metadata = MockUsageMetadata(prompt=100, candidates=50, total=150)
        response = MockResponse(usage=usage_metadata)

        usage = extract_usage(response)

        assert usage is not None
        assert usage["text"]["input_tokens"] == 100
        assert usage["text"]["output_tokens"] == 50
        assert usage["text"]["total_tokens"] == 150

    def test_extract_usage_no_metadata(self) -> None:
        """Return None when no usage metadata."""
        response = MockResponse(usage=None)

        usage = extract_usage(response)

        assert usage is None

    def test_extract_usage_missing_fields(self) -> None:
        """Handle missing fields gracefully."""

        class PartialUsage:
            prompt_token_count = 100
            # No candidates_token_count or total_token_count

        response = MockResponse()
        response.usage_metadata = PartialUsage()  # type: ignore[assignment]

        usage = extract_usage(response)

        assert usage is not None
        assert usage["text"]["input_tokens"] == 100
        assert usage["text"]["output_tokens"] == 0
        assert usage["text"]["total_tokens"] == 0

    def test_extract_usage_with_thinking_and_modalities(self) -> None:
        """Extract thinking tokens and modality breakdowns from Vertex metadata."""

        class Detail:
            def __init__(self, modality: str, token_count: int) -> None:
                self.modality = modality
                self.token_count = token_count

        usage_metadata = MockUsageMetadata(
            prompt=800,
            candidates=100,
            total=1200,
            thoughts=300,
        )
        usage_metadata.prompt_tokens_details = [
            Detail("TEXT", 100),
            Detail("IMAGE", 450),
            Detail("AUDIO", 250),
        ]
        usage_metadata.candidates_tokens_details = [Detail("VIDEO", 40)]
        response = MockResponse(usage=usage_metadata)

        usage = extract_usage(response)

        assert usage is not None
        assert usage["reasoning"]["output_tokens"] == 300
        assert usage["image"]["input_tokens"] == 450
        assert usage["audio"]["input_tokens"] == 250
        assert usage["video"]["output_tokens"] == 40


class TestExtractModel:
    """Tests for model name extraction."""

    def test_extract_model_full_path(self) -> None:
        """Extract model name from full path."""
        model = MockModel(model_name="models/gemini-1.5-pro")

        name = extract_model(model)

        assert name == "gemini-1.5-pro"

    def test_extract_model_simple_name(self) -> None:
        """Extract simple model name."""
        model = MockModel(model_name="gemini-1.5-flash")

        name = extract_model(model)

        assert name == "gemini-1.5-flash"

    def test_extract_model_none(self) -> None:
        """Return unknown when no model name."""
        model = MockModel(model_name=None)

        name = extract_model(model)

        assert name == "unknown"

    def test_extract_model_no_attribute(self) -> None:
        """Return unknown when attribute missing."""

        class NoModelName:
            pass

        name = extract_model(NoModelName())

        assert name == "unknown"


class TestInferRegion:
    """Tests for region inference from endpoint."""

    def test_infer_region_from_endpoint(self) -> None:
        """Infer region from standard endpoint."""
        region = infer_region_from_endpoint("us-central1-aiplatform.googleapis.com")

        assert region == "us-central1"

    def test_infer_region_europe(self) -> None:
        """Infer European region."""
        region = infer_region_from_endpoint("europe-west1-aiplatform.googleapis.com")

        assert region == "europe-west1"

    def test_infer_region_asia(self) -> None:
        """Infer Asian region."""
        region = infer_region_from_endpoint("asia-southeast1-aiplatform.googleapis.com")

        assert region == "asia-southeast1"

    def test_infer_region_none_endpoint(self) -> None:
        """Return None for None endpoint."""
        region = infer_region_from_endpoint(None)

        assert region is None

    def test_infer_region_unknown_format(self) -> None:
        """Return None for unknown endpoint format."""
        region = infer_region_from_endpoint("custom.example.com")

        assert region is None


class TestStreamWrapper:
    """Tests for Vertex AI stream wrapper."""

    def test_stream_counts_chars(self) -> None:
        """Stream wrapper counts characters."""
        chunks = [
            MockChunk(text="Hello"),
            MockChunk(text=" World"),
            MockChunk(text="!"),
        ]
        wrapper = StreamWrapper(iter(chunks), "gemini-1.5-pro")

        # Consume stream
        list(wrapper)

        assert wrapper._accumulated_chars == 12
        assert wrapper._complete is True

    def test_stream_captures_final_usage(self) -> None:
        """Stream wrapper captures final usage from last chunk."""
        usage = MockUsageMetadata(prompt=100, candidates=50, total=150)
        chunks = [
            MockChunk(text="Hello"),
            MockChunk(text="!", usage_metadata=usage),
        ]
        wrapper = StreamWrapper(iter(chunks), "gemini-1.5-pro")

        list(wrapper)

        assert wrapper._final_usage is not None
        assert wrapper._final_usage["text"]["input_tokens"] == 100

    def test_stream_handles_none_text(self) -> None:
        """Stream wrapper handles None text gracefully."""
        chunks = [
            MockChunk(text=None),
            MockChunk(text="Hello"),
            MockChunk(text=None),
        ]
        wrapper = StreamWrapper(iter(chunks), "gemini-1.5-pro")

        list(wrapper)

        assert wrapper._accumulated_chars == 5

    def test_stream_tracks_error(self) -> None:
        """Stream wrapper tracks errors."""

        def failing_stream() -> Any:
            yield MockChunk(text="Start")
            raise ValueError("Stream error")

        wrapper = StreamWrapper(failing_stream(), "gemini-1.5-pro")

        try:
            list(wrapper)
        except ValueError:
            pass

        assert wrapper._error is True
        assert wrapper._error_type == "ValueError"
        assert wrapper._complete is False

    def test_stream_captures_to_context(self) -> None:
        """Stream wrapper captures to active context."""
        with TrackingContext() as ctx:
            chunks = [MockChunk(text="Hello", usage_metadata=None)]
            wrapper = StreamWrapper(iter(chunks), "gemini-1.5-pro")

            list(wrapper)

            assert ctx.captured_call is not None
            assert ctx.captured_call.model == "gemini-1.5-pro"
            assert ctx.captured_call.provider == "vertexai"
            assert ctx.captured_call.is_stream is True

    def test_stream_context_manager_protocol(self) -> None:
        """Stream wrapper supports context manager."""
        chunks = [MockChunk(text="Hello")]
        wrapper = StreamWrapper(iter(chunks), "gemini-1.5-pro")

        with wrapper as w:
            result = list(w)

        assert len(result) == 1
        assert wrapper._complete is True

    def test_stream_context_manager_on_error(self) -> None:
        """Stream wrapper captures error when used as context manager."""
        with TrackingContext() as ctx:
            chunks = [MockChunk(text="Start")]
            wrapper = StreamWrapper(iter(chunks), "gemini-1.5-pro")

            try:
                with wrapper:
                    next(wrapper)
                    raise RuntimeError("User error")
            except RuntimeError:
                pass

            assert ctx.captured_call is not None
            assert ctx.captured_call.error is True
            assert ctx.captured_call.error_type == "RuntimeError"
