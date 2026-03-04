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
import queue
import sys
import threading
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
        # Assume file path - validate against path traversal (defense in depth)
        from pathlib import Path
        from tempfile import gettempdir

        from vetch._security import is_safe_output_path

        try:
            target_path = Path(target).resolve()
        except (OSError, RuntimeError):
            sys.stderr.write(
                f"Vetch WARNING: VETCH_OUTPUT path '{target}' is invalid. "
                "Falling back to stderr.\n"
            )
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(JsonFormatter())
            logger.addHandler(handler)
            return

        # Security: Only allow writing to current directory, subdirectories, or temp
        cwd = Path.cwd().resolve()
        tmp = Path(gettempdir()).resolve()
        home_vetch = (Path.home() / ".vetch").resolve()

        if not is_safe_output_path(target_path, [cwd, tmp, home_vetch]):
            sys.stderr.write(
                f"Vetch WARNING: VETCH_OUTPUT path '{target}' is outside allowed directories. "
                "Allowed: current directory, temp, or ~/.vetch. Falling back to stderr.\n"
            )
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(JsonFormatter())
            logger.addHandler(handler)
            return

        try:
            # Ensure parent directory exists
            target_path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(str(target_path), encoding="utf-8")
            handler.setFormatter(JsonFormatter())
            logger.addHandler(handler)
        except Exception as e:
            # Fallback to stderr if file fails
            sys.stderr.write(f"Vetch WARNING: Failed to open {target_path}: {e}. Using stderr.\n")
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(JsonFormatter())
            logger.addHandler(handler)


class HttpHandler(logging.Handler):
    """Logging handler that POSTs JSON to an HTTP endpoint asynchronously.

    Uses a background thread and queue to avoid blocking the main thread
    or async event loops (FastAPI/Tornado).
    """

    def __init__(self, url: str) -> None:
        super().__init__()
        self.url = url
        self._timeout = 0.5  # Tight timeout to avoid blocking
        self._queue: queue.Queue[tuple[str, bytes] | None] = queue.Queue(maxsize=1000)
        self._worker_thread: threading.Thread | None = None
        self._shutdown = False
        self._start_worker()

    def _start_worker(self) -> None:
        """Start background worker thread for async HTTP requests."""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._shutdown = False
            self._worker_thread = threading.Thread(
                target=self._worker,
                daemon=True,
                name="vetch-http-worker",
            )
            self._worker_thread.start()

    def _worker(self) -> None:
        """Background worker that sends HTTP requests."""
        import urllib.error
        import urllib.request

        while not self._shutdown:
            try:
                item = self._queue.get(timeout=0.5)
                if item is None:  # Sentinel for shutdown
                    break

                url, payload = item
                try:
                    req = urllib.request.Request(
                        url,
                        data=payload,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=self._timeout):
                        pass
                except Exception:
                    # Fail silently - never block or crash the host app
                    pass
                finally:
                    self._queue.task_done()
            except queue.Empty:
                continue

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record asynchronously (non-blocking)."""
        try:
            payload = self.format(record).encode("utf-8")
            # Non-blocking: drop if queue is full (fail-open)
            self._queue.put_nowait((self.url, payload))
        except queue.Full:
            # Queue full - drop message (fail-open behavior)
            pass
        except Exception:
            # Fail silently - never block or crash the host app
            pass

    def close(self) -> None:
        """Shutdown background worker gracefully."""
        self._shutdown = True
        try:
            self._queue.put_nowait(None)  # Sentinel
        except queue.Full:
            pass
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
        super().close()


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
