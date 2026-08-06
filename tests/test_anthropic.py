"""Tests for Anthropic provider wrapper.

These tests verify:
- Usage extraction from responses
- Model extraction from responses
- Stream wrapper functionality
- Client patching
"""

from __future__ import annotations

from unittest.mock import MagicMock

from vetch.providers.anthropic import (
    StreamWrapper,
    _extract_cache_creation_1h,
    _extract_stop_reason,
    _extract_visible_chars,
    extract_model,
    extract_usage,
    patch_anthropic_client,
    unpatch_anthropic_client,
)


class TestExtractCacheCreation1h:
    """Tests for _extract_cache_creation_1h (1-hour-TTL cache write breakdown)."""

    def test_extracts_1h_tokens_when_present(self) -> None:
        usage = MagicMock()
        usage.cache_creation = MagicMock()
        usage.cache_creation.ephemeral_1h_input_tokens = 512
        assert _extract_cache_creation_1h(usage) == 512

    def test_none_when_usage_missing(self) -> None:
        assert _extract_cache_creation_1h(None) is None

    def test_none_when_no_breakdown(self) -> None:
        usage = MagicMock(spec=["cache_creation_input_tokens"])
        assert _extract_cache_creation_1h(usage) is None

    def test_none_when_breakdown_lacks_1h_field(self) -> None:
        usage = MagicMock()
        usage.cache_creation = MagicMock(spec=["ephemeral_5m_input_tokens"])
        assert _extract_cache_creation_1h(usage) is None


class TestExtractUsage:
    """Tests for extract_usage function."""

    def test_extracts_usage_from_response(self) -> None:
        """Extract usage dict from Anthropic response."""
        response = MagicMock()
        response.usage = MagicMock()
        response.usage.input_tokens = 100
        response.usage.output_tokens = 50
        response.usage.cache_read_input_tokens = None
        response.usage.cache_creation_input_tokens = None

        result, cache_read, cache_create = extract_usage(response)

        assert result is not None
        assert result["text"]["input_tokens"] == 100
        assert result["text"]["output_tokens"] == 50
        assert result["text"]["total_tokens"] == 150
        assert cache_read is None
        assert cache_create is None

    def test_returns_none_when_no_usage(self) -> None:
        """Return None when response has no usage."""
        response = MagicMock()
        response.usage = None

        result, cache_read, cache_create = extract_usage(response)

        assert result is None
        assert cache_read is None
        assert cache_create is None

    def test_handles_missing_tokens(self) -> None:
        """Handle missing token counts gracefully."""
        response = MagicMock()
        response.usage = MagicMock(spec=[])  # No attributes

        result, cache_read, cache_create = extract_usage(response)

        assert result is not None
        assert result["text"]["input_tokens"] == 0
        assert result["text"]["output_tokens"] == 0
        assert cache_read is None
        assert cache_create is None


class TestExtractModel:
    """Tests for extract_model function."""

    def test_extracts_model_name(self) -> None:
        """Extract model name from response."""
        response = MagicMock()
        response.model = "claude-3-opus-20240229"

        result = extract_model(response)

        assert result == "claude-3-opus-20240229"

    def test_returns_unknown_when_missing(self) -> None:
        """Return 'unknown' when model not present."""
        response = MagicMock(spec=[])  # No model attribute

        result = extract_model(response)

        assert result == "unknown"


class TestStreamWrapper:
    """Tests for StreamWrapper class."""

    def test_counts_characters_from_content_blocks(self) -> None:
        """Count characters from content_block_delta events."""
        # Create mock chunks
        chunk1 = MagicMock()
        chunk1.type = "content_block_delta"
        chunk1.delta = MagicMock()
        chunk1.delta.text = "Hello"

        chunk2 = MagicMock()
        chunk2.type = "content_block_delta"
        chunk2.delta = MagicMock()
        chunk2.delta.text = " world"

        mock_stream = iter([chunk1, chunk2])
        wrapper = StreamWrapper(mock_stream)

        # Consume the stream
        chunks = list(wrapper)

        assert len(chunks) == 2
        assert wrapper._accumulated_chars == 11  # "Hello" + " world"

    def test_captures_model_from_message_start(self) -> None:
        """Capture model name from message_start event."""
        chunk = MagicMock()
        chunk.type = "message_start"
        chunk.message = MagicMock()
        chunk.message.model = "claude-3-sonnet-20240229"
        chunk.message.usage = MagicMock()
        chunk.message.usage.input_tokens = 100

        mock_stream = iter([chunk])
        wrapper = StreamWrapper(mock_stream)

        list(wrapper)

        assert wrapper._model == "claude-3-sonnet-20240229"
        assert wrapper._input_tokens == 100

    def test_captures_output_tokens_from_message_delta(self) -> None:
        """Capture output tokens from message_delta event."""
        chunk = MagicMock()
        chunk.type = "message_delta"
        chunk.usage = MagicMock()
        chunk.usage.output_tokens = 75

        mock_stream = iter([chunk])
        wrapper = StreamWrapper(mock_stream)

        list(wrapper)

        assert wrapper._output_tokens == 75

    def test_marks_complete_on_stream_end(self) -> None:
        """Mark complete when stream ends normally."""
        mock_stream = iter([])
        wrapper = StreamWrapper(mock_stream)

        list(wrapper)

        assert wrapper._complete is True
        assert wrapper._error is False

    def test_marks_error_on_exception(self) -> None:
        """Mark error when stream raises exception."""

        def failing_generator():
            raise RuntimeError("Stream failed")
            yield  # Make it a generator

        wrapper = StreamWrapper(failing_generator())

        try:
            list(wrapper)
        except RuntimeError:
            pass

        assert wrapper._error is True
        assert wrapper._error_type == "RuntimeError"

    def test_context_manager_protocol(self) -> None:
        """StreamWrapper supports context manager protocol."""
        mock_stream = MagicMock()
        wrapper = StreamWrapper(mock_stream)

        with wrapper as w:
            assert w is wrapper

    def test_context_manager_closes_stream(self) -> None:
        """Context manager calls close on stream."""
        mock_stream = MagicMock()
        wrapper = StreamWrapper(mock_stream)

        with wrapper:
            pass

        mock_stream.close.assert_called_once()

    def test_handles_empty_delta_text(self) -> None:
        """Handle empty text in delta gracefully."""
        chunk = MagicMock()
        chunk.type = "content_block_delta"
        chunk.delta = MagicMock()
        chunk.delta.text = ""

        mock_stream = iter([chunk])
        wrapper = StreamWrapper(mock_stream)

        chunks = list(wrapper)

        assert len(chunks) == 1
        assert wrapper._accumulated_chars == 0

    def test_handles_missing_delta(self) -> None:
        """Handle missing delta attribute gracefully."""
        chunk = MagicMock()
        chunk.type = "content_block_delta"
        chunk.delta = None

        mock_stream = iter([chunk])
        wrapper = StreamWrapper(mock_stream)

        # Should not raise
        list(wrapper)

    def test_captures_stop_reason_from_message_delta(self) -> None:
        """stop_reason from message_delta.delta must be stored as _stop_reason."""
        chunk = MagicMock()
        chunk.type = "message_delta"
        chunk.delta = MagicMock()
        chunk.delta.stop_reason = "max_tokens"
        chunk.usage = MagicMock()
        chunk.usage.output_tokens = 10

        wrapper = StreamWrapper(iter([chunk]))
        list(wrapper)

        assert wrapper._stop_reason == "max_tokens"

    def test_stop_reason_none_when_end_turn(self) -> None:
        """end_turn stop_reason is captured correctly."""
        chunk = MagicMock()
        chunk.type = "message_delta"
        chunk.delta = MagicMock()
        chunk.delta.stop_reason = "end_turn"
        chunk.usage = MagicMock()
        chunk.usage.output_tokens = 5

        wrapper = StreamWrapper(iter([chunk]))
        list(wrapper)

        assert wrapper._stop_reason == "end_turn"

    def test_stop_reason_unset_when_not_in_delta(self) -> None:
        """No message_delta means _stop_reason stays None."""
        chunk = MagicMock()
        chunk.type = "content_block_delta"
        chunk.delta = MagicMock()
        chunk.delta.text = "hello"

        wrapper = StreamWrapper(iter([chunk]))
        list(wrapper)

        assert wrapper._stop_reason is None


class TestPatchClient:
    """Tests for client patching."""

    def test_patches_client_messages_create(self) -> None:
        """Patch client.messages.create method."""
        client = MagicMock()
        client.messages = MagicMock()
        client.messages.create = MagicMock()

        result = patch_anthropic_client(client)

        assert result is True
        assert hasattr(client.messages.create, "vetch_patched")

    def test_returns_false_when_no_messages(self) -> None:
        """Return False when client has no messages attribute."""
        client = MagicMock()
        client.messages = None

        result = patch_anthropic_client(client)

        assert result is False

    def test_returns_false_when_no_create(self) -> None:
        """Return False when messages has no create method."""
        client = MagicMock()
        client.messages = MagicMock()
        client.messages.create = None

        result = patch_anthropic_client(client)

        assert result is False

    def test_skips_already_patched(self) -> None:
        """Skip patching if already patched."""
        client = MagicMock()
        client.messages = MagicMock()
        mock_create = MagicMock()
        mock_create.vetch_patched = True
        client.messages.create = mock_create

        result = patch_anthropic_client(client)

        # Should succeed without re-patching
        assert result is True

    def test_unpatch_returns_true_after_patch(self) -> None:
        """Unpatch returns True after successful patch."""
        client = MagicMock()
        client.messages = MagicMock()
        original_create = MagicMock()
        client.messages.create = original_create

        patch_anthropic_client(client)

        # After patching, should have vetch marker
        assert hasattr(client.messages.create, "vetch_patched")
        assert client.messages.create.vetch_patched is True

        # Unpatch should return True
        result = unpatch_anthropic_client(client)
        assert result is True

    def test_unpatch_returns_true_when_not_patched(self) -> None:
        """Unpatch returns True when never patched (no-op)."""
        # Reset module state
        import vetch.providers.anthropic as anthropic_mod
        anthropic_mod._original_create = None

        client = MagicMock()
        client.messages = MagicMock()

        result = unpatch_anthropic_client(client)
        assert result is True


class TestExtractVisibleChars:
    """Tests for _extract_visible_chars — visible text extraction for EMPTY-001."""

    def _make_block(self, block_type: str, text: str = "") -> MagicMock:
        block = MagicMock()
        block.type = block_type
        block.text = text
        return block

    def test_counts_text_block(self) -> None:
        response = MagicMock()
        response.content = [self._make_block("text", "hello world")]
        assert _extract_visible_chars(response) == 10

    def test_skips_thinking_block(self) -> None:
        response = MagicMock()
        response.content = [
            self._make_block("thinking", "long internal reasoning " * 50),
            self._make_block("text", "ok"),
        ]
        assert _extract_visible_chars(response) == 2

    def test_empty_text_block(self) -> None:
        response = MagicMock()
        response.content = [
            self._make_block("thinking", "lots of thinking"),
            self._make_block("text", ""),
        ]
        assert _extract_visible_chars(response) == 0

    def test_no_content_attribute(self) -> None:
        response = MagicMock(spec=[])
        assert _extract_visible_chars(response) is None

    def test_content_not_list(self) -> None:
        response = MagicMock()
        response.content = "not a list"
        assert _extract_visible_chars(response) is None

    def test_multiple_text_blocks(self) -> None:
        response = MagicMock()
        response.content = [
            self._make_block("text", "abc"),
            self._make_block("text", "de"),
        ]
        assert _extract_visible_chars(response) == 5

    def test_whitespace_only_not_counted(self) -> None:
        """Whitespace-only output must not count as visible chars (EMPTY-001 bypass)."""
        response = MagicMock()
        response.content = [self._make_block("text", "\n\n  \n  ")]
        assert _extract_visible_chars(response) == 0

    def test_counts_non_whitespace_visible_chars(self) -> None:
        """Visible char counting is consistent with the streaming path."""
        response = MagicMock()
        response.content = [self._make_block("text", "  hello world  ")]
        assert _extract_visible_chars(response) == 10


class TestExtractStopReason:
    """Tests for _extract_stop_reason — Anthropic stop_reason → finish_reason."""

    def _make_response(self, stop_reason: object) -> MagicMock:
        r = MagicMock()
        r.stop_reason = stop_reason
        return r

    def test_end_turn(self) -> None:
        assert _extract_stop_reason(self._make_response("end_turn")) == "end_turn"

    def test_max_tokens(self) -> None:
        assert _extract_stop_reason(self._make_response("max_tokens")) == "max_tokens"

    def test_stop_sequence(self) -> None:
        assert _extract_stop_reason(self._make_response("stop_sequence")) == "stop_sequence"

    def test_tool_use(self) -> None:
        assert _extract_stop_reason(self._make_response("tool_use")) == "tool_use"

    def test_none_stop_reason(self) -> None:
        assert _extract_stop_reason(self._make_response(None)) is None

    def test_missing_attribute(self) -> None:
        r = MagicMock(spec=[])
        assert _extract_stop_reason(r) is None

    def test_streaming_visible_chars_excludes_whitespace(self) -> None:
        """Streaming path must count non-whitespace chars, matching non-streaming behaviour."""
        chunk = MagicMock()
        chunk.type = "content_block_delta"
        chunk.delta = MagicMock()
        chunk.delta.text = "\n\n   \n"

        wrapper = StreamWrapper(iter([chunk]))
        list(wrapper)

        assert wrapper._visible_chars == 0
        assert wrapper._accumulated_chars == len("\n\n   \n")

    def test_streaming_visible_chars_counts_content(self) -> None:
        """Non-whitespace content increments _visible_chars correctly."""
        chunk = MagicMock()
        chunk.type = "content_block_delta"
        chunk.delta = MagicMock()
        chunk.delta.text = "  hello world  "

        wrapper = StreamWrapper(iter([chunk]))
        list(wrapper)

        assert wrapper._visible_chars == len("helloworld")  # 10 non-ws chars
