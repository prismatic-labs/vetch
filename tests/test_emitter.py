"""Tests for emitter module."""

import os
import tempfile

from vetch.emitter import (
    BufferedEmitter,
    _configure_logging,
    emit,
    emit_event,
    get_test_emitter,
    serialize_event,
    set_test_emitter,
)
from vetch.schema import InferenceEvent


class TestSerializeEvent:
    """Tests for serialize_event function."""

    def test_serializes_to_json(self) -> None:
        """Event is serialized to valid JSON."""
        event: InferenceEvent = {
            "schema_version": "1",
            "vetch_version": "0.1.0",
            "event_id": "test-123",
            "timestamp": "2026-02-12T00:00:00Z",
            "signal_quality": "live",
        }
        json_str = serialize_event(event)
        assert '"schema_version":"1"' in json_str
        assert '"event_id":"test-123"' in json_str

    def test_filters_none_values(self) -> None:
        """None values are filtered from output."""
        event: InferenceEvent = {
            "schema_version": "1",
            "model": None,  # type: ignore[typeddict-item]
            "signal_quality": "live",
        }
        json_str = serialize_event(event)
        assert "model" not in json_str

    def test_compact_format(self) -> None:
        """Output uses compact JSON format."""
        event: InferenceEvent = {
            "schema_version": "1",
            "signal_quality": "live",
        }
        json_str = serialize_event(event)
        # No spaces after colons or commas
        assert ": " not in json_str
        assert ", " not in json_str


class TestEmit:
    """Tests for emit function."""

    def test_emit_to_none(self) -> None:
        """Emit with target 'none' does nothing."""
        original = os.environ.get("VETCH_OUTPUT")
        try:
            os.environ["VETCH_OUTPUT"] = "none"
            _configure_logging()
            event: InferenceEvent = {
                "schema_version": "1",
                "signal_quality": "live",
            }
            # Should not raise
            emit(event)
        finally:
            if original:
                os.environ["VETCH_OUTPUT"] = original
            else:
                os.environ.pop("VETCH_OUTPUT", None)
            _configure_logging()

    def test_emit_to_file(self) -> None:
        """Emit writes to file when configured."""
        import logging

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            temp_path = f.name

        original = os.environ.get("VETCH_OUTPUT")
        try:
            os.environ["VETCH_OUTPUT"] = temp_path
            # Reconfigure logging to pick up new env var
            _configure_logging()

            event: InferenceEvent = {
                "schema_version": "1",
                "event_id": "test-file",
                "signal_quality": "live",
            }
            emit(event)

            # Flush handlers to ensure write completes
            logger = logging.getLogger("vetch.emitter")
            for handler in logger.handlers:
                handler.flush()

            with open(temp_path) as f:
                content = f.read()
            assert "test-file" in content
        finally:
            if original:
                os.environ["VETCH_OUTPUT"] = original
            else:
                os.environ.pop("VETCH_OUTPUT", None)
            # Reconfigure back to default
            _configure_logging()
            os.unlink(temp_path)


class TestBufferedEmitter:
    """Tests for BufferedEmitter class."""

    def test_stores_events(self) -> None:
        """BufferedEmitter stores emitted events."""
        emitter = BufferedEmitter()
        event: InferenceEvent = {
            "schema_version": "1",
            "signal_quality": "live",
        }
        emitter.emit(event)
        assert len(emitter) == 1
        assert emitter.events[0] == event

    def test_clear(self) -> None:
        """BufferedEmitter can be cleared."""
        emitter = BufferedEmitter()
        event: InferenceEvent = {"schema_version": "1", "signal_quality": "live"}
        emitter.emit(event)
        emitter.clear()
        assert len(emitter) == 0


class TestTestEmitter:
    """Tests for test emitter functions."""

    def test_set_and_get(self) -> None:
        """Can set and get test emitter."""
        original = get_test_emitter()
        try:
            emitter = BufferedEmitter()
            set_test_emitter(emitter)
            assert get_test_emitter() is emitter
        finally:
            set_test_emitter(original)

    def test_emit_event_uses_test_emitter(self) -> None:
        """emit_event uses test emitter when set."""
        original = get_test_emitter()
        try:
            emitter = BufferedEmitter()
            set_test_emitter(emitter)

            event: InferenceEvent = {
                "schema_version": "1",
                "event_id": "test-event",
                "signal_quality": "live",
            }
            emit_event(event)

            assert len(emitter) == 1
            assert emitter.events[0]["event_id"] == "test-event"
        finally:
            set_test_emitter(original)
