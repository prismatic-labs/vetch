"""Deep tests for emitter logic.

Verifies file output and error handling in event emission.
"""

from __future__ import annotations

import logging
import os
import tempfile

from vetch.emitter import _configure_logging, emit, serialize_event
from vetch.schema import SCHEMA_VERSION


def test_serialize_event_filters_none() -> None:
    """Verify None values are filtered from JSON."""
    event = {
        "schema_version": SCHEMA_VERSION,
        "model": "gpt-4o",
        "usage": None,
        "error": False,
    }
    json_str = serialize_event(event)
    assert "usage" not in json_str
    assert "model" in json_str


def test_emit_to_file() -> None:
    """Verify emission to a file."""
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as tmp:
        path = tmp.name

    original = os.environ.get("VETCH_OUTPUT")
    try:
        os.environ["VETCH_OUTPUT"] = path
        _configure_logging()

        event = {"schema_version": SCHEMA_VERSION, "event_id": "test"}
        emit(event)

        # Flush handlers
        logger = logging.getLogger("vetch.emitter")
        for handler in logger.handlers:
            handler.flush()

        with open(path) as f:
            content = f.read()
            assert "test" in content
    finally:
        if original:
            os.environ["VETCH_OUTPUT"] = original
        else:
            os.environ.pop("VETCH_OUTPUT", None)
        _configure_logging()
        if os.path.exists(path):
            os.unlink(path)


def test_emit_file_error_silent() -> None:
    """Verify file write errors are handled silently via fallback."""
    event = {"schema_version": SCHEMA_VERSION}
    # Path to a directory instead of a file should trigger fallback to stderr
    with tempfile.TemporaryDirectory() as tmpdir:
        original = os.environ.get("VETCH_OUTPUT")
        try:
            os.environ["VETCH_OUTPUT"] = tmpdir
            _configure_logging()  # Should fallback to stderr
            # Should not raise
            emit(event)
        finally:
            if original:
                os.environ["VETCH_OUTPUT"] = original
            else:
                os.environ.pop("VETCH_OUTPUT", None)
            _configure_logging()


def test_emit_stderr_error_silent() -> None:
    """Verify stderr write errors are handled by logging module."""
    # The logging module has its own error handling - we just verify
    # emit doesn't crash
    event = {"schema_version": SCHEMA_VERSION}
    original = os.environ.get("VETCH_OUTPUT")
    try:
        os.environ["VETCH_OUTPUT"] = "stderr"
        _configure_logging()
        # Should not raise even if there's a logging issue
        emit(event)
    finally:
        if original:
            os.environ["VETCH_OUTPUT"] = original
        else:
            os.environ.pop("VETCH_OUTPUT", None)
        _configure_logging()
