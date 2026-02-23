"""Event emission to configured outputs.

This module handles:
- JSON serialization of InferenceEvents
- Output via standard Python logging (vetch.emitter)
- Environment variable configuration (VETCH_OUTPUT)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vetch.schema import InferenceEvent

# Create logger
logger = logging.getLogger("vetch.emitter")
logger.setLevel(logging.INFO)
# Default to NullHandler so we don't spam unless configured
logger.addHandler(logging.NullHandler())


def _configure_logging() -> None:
    """Configure logging based on VETCH_OUTPUT env var.

    Defaults to 'none' (quiet mode). Set VETCH_OUTPUT=stderr to see JSON output.
    This is a convenience helper. Advanced users should configure
    the 'vetch.emitter' logger directly.
    """
    target = os.environ.get("VETCH_OUTPUT", "none")

    if target == "none":
        return

    # Remove default handlers to avoid duplication if re-configured
    for h in logger.handlers[:]:
        if isinstance(h, (logging.StreamHandler, logging.FileHandler)):
            logger.removeHandler(h)

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            return record.getMessage()

    if target == "stderr":
        handler: logging.Handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    elif target.startswith(("http://", "https://")):
        # HTTP Emitter is "Dark Launched" for Alpha.
        # Enabling it now requires an explicit opt-in flag to avoid privacy concerns.
        if os.environ.get("VETCH_ENABLE_REMOTE") == "true":
            handler = HttpHandler(target)
            handler.setFormatter(JsonFormatter())
            logger.addHandler(handler)
        else:
            # Log a warning to stderr instead of sending data
            sys.stderr.write(
                "Vetch INFO: Remote emission (HTTP) is disabled in this version. "
                "Use VETCH_ENABLE_REMOTE=true to enable beta testing.\n"
            )
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(JsonFormatter())
            logger.addHandler(handler)
    else:
        # Assume file path
        try:
            handler = logging.FileHandler(target, encoding="utf-8")
            handler.setFormatter(JsonFormatter())
            logger.addHandler(handler)
        except Exception:
            # Fallback to stderr if file fails
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(JsonFormatter())
            logger.addHandler(handler)


class HttpHandler(logging.Handler):
    """Logging handler that POSTs JSON to an HTTP endpoint."""

    def __init__(self, url: str) -> None:
        super().__init__()
        self.url = url
        self._timeout = 0.5  # Tight timeout to avoid blocking

    def emit(self, record: logging.LogRecord) -> None:
        import urllib.error
        import urllib.request

        try:
            payload = self.format(record).encode("utf-8")
            req = urllib.request.Request(
                self.url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            # Synchronous but with a very short timeout
            with urllib.request.urlopen(req, timeout=self._timeout):
                pass
        except Exception:
            # Fail silently - never block or crash the host app
            pass


# Auto-configure on import if env var is set (best effort)
_configure_logging()


def serialize_event(event: InferenceEvent) -> str:
    """Serialize an InferenceEvent to JSON.

    Args:
        event: The event to serialize.

    Returns:
        Compact JSON string (single line).
    """
    # Filter out None values for cleaner output
    filtered = {k: v for k, v in event.items() if v is not None}
    return json.dumps(filtered, separators=(",", ":"))


def emit(event: InferenceEvent) -> None:
    """Emit an inference event via logger.

    Args:
        event: The event to emit.
    """
    json_line = serialize_event(event)
    logger.info(json_line)


class BufferedEmitter:
    """Emitter that buffers events for testing.

    Useful for capturing events in tests without stderr pollution.
    """

    def __init__(self) -> None:
        """Initialize empty buffer."""
        self.events: list[InferenceEvent] = []

    def emit(self, event: InferenceEvent) -> None:
        """Store event in buffer.

        Args:
            event: Event to store.
        """
        self.events.append(event)

    def clear(self) -> None:
        """Clear buffered events."""
        self.events.clear()

    def __len__(self) -> int:
        """Get number of buffered events."""
        return len(self.events)


# Global buffered emitter for testing
_test_emitter: BufferedEmitter | None = None


def set_test_emitter(emitter: BufferedEmitter | None) -> None:
    """Set a test emitter to capture events.

    Args:
        emitter: BufferedEmitter instance, or None to disable.
    """
    global _test_emitter
    _test_emitter = emitter


def get_test_emitter() -> BufferedEmitter | None:
    """Get the current test emitter.

    Returns:
        BufferedEmitter if set, None otherwise.
    """
    return _test_emitter


def emit_event(event: InferenceEvent) -> None:
    """Emit an event, using test emitter if set.

    This is the main entry point for event emission.

    Args:
        event: The event to emit.
    """
    if _test_emitter is not None:
        _test_emitter.emit(event)
    else:
        emit(event)
