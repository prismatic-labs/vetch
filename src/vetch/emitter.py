"""Event emission to configured outputs.

This module handles:
- JSON serialization of InferenceEvents
- Output via standard Python logging (vetch.emitter)
- Environment variable configuration (VETCH_OUTPUT, VETCH_ENDPOINT)
"""

from __future__ import annotations

import json
import logging
import os
import queue
import sys
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vetch.schema import InferenceEvent

# Create logger
logger = logging.getLogger("vetch.emitter")
logger.setLevel(logging.INFO)
# Default to NullHandler so we don't spam unless configured
logger.addHandler(logging.NullHandler())


def _configure_logging() -> None:
    """Configure logging based on VETCH_OUTPUT / VETCH_ENDPOINT env vars.

    Defaults to 'none' (quiet mode). Set VETCH_OUTPUT=stderr to see JSON output.

    Environment variables:
        VETCH_OUTPUT: 'stderr', 'none', or a file path (default: 'none')
        VETCH_ENDPOINT: HTTP/HTTPS URL to POST events to (takes precedence over
            VETCH_OUTPUT=https://... and does not require VETCH_ENABLE_REMOTE)
        VETCH_API_KEY: Optional Bearer token for VETCH_ENDPOINT authentication
    """
    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            return record.getMessage()

    # VETCH_ENDPOINT is first-class: wire it up unconditionally if set
    endpoint = os.environ.get("VETCH_ENDPOINT", "").strip()
    if endpoint:
        api_key = os.environ.get("VETCH_API_KEY") or None
        # Remove any previously added HttpHandler for this endpoint to avoid duplication
        for h in logger.handlers[:]:
            if isinstance(h, HttpHandler) and h.url == endpoint:
                logger.removeHandler(h)
        handler: logging.Handler = HttpHandler(endpoint, api_key=api_key)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)

    # VETCH_OUTPUT handles local / legacy targets
    target = os.environ.get("VETCH_OUTPUT", "none")

    if target == "none":
        return

    # Remove duplicate stream/file handlers to avoid double-writing on re-configure
    for h in logger.handlers[:]:
        if isinstance(h, (logging.StreamHandler, logging.FileHandler)) and not isinstance(
            h, HttpHandler
        ):
            logger.removeHandler(h)

    if target == "stderr":
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    elif target.startswith(("http://", "https://")):
        # Legacy: VETCH_OUTPUT=https://... still works but requires VETCH_ENABLE_REMOTE
        # Prefer VETCH_ENDPOINT for new setups (no flag required)
        if os.environ.get("VETCH_ENABLE_REMOTE") == "true":
            handler = HttpHandler(target)
            handler.setFormatter(JsonFormatter())
            logger.addHandler(handler)
        else:
            sys.stderr.write(
                "Vetch INFO: Use VETCH_ENDPOINT instead of VETCH_OUTPUT for HTTP emission "
                "(no opt-in flag required). Or set VETCH_ENABLE_REMOTE=true for legacy mode.\n"
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

    Configure via environment variables:
        VETCH_ENDPOINT=https://your-endpoint.example.com/ingest
        VETCH_API_KEY=your-api-key  (optional, sent as Bearer token)

    Or programmatically via vetch.configure_http_endpoint(url, api_key).
    """

    def __init__(self, url: str, api_key: str | None = None) -> None:
        super().__init__()
        self.url = url
        self._api_key = api_key
        self._timeout = 0.5  # Tight timeout to avoid blocking
        self._queue: queue.Queue[tuple[str, bytes] | None] = queue.Queue(maxsize=1000)
        self._worker_thread: threading.Thread | None = None
        self._shutdown = False
        self._last_error_log: float = 0.0  # For rate-limited error logging
        self._error_log_interval = 60.0  # Log at most once per minute
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
                    headers: dict[str, str] = {"Content-Type": "application/json"}
                    if self._api_key:
                        headers["Authorization"] = f"Bearer {self._api_key}"

                    req = urllib.request.Request(
                        url,
                        data=payload,
                        headers=headers,
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=self._timeout):
                        pass
                except Exception as e:
                    # Rate-limited warning: log at most once per minute
                    now = time.monotonic()
                    if now - self._last_error_log >= self._error_log_interval:
                        self._last_error_log = now
                        sys.stderr.write(
                            f"Vetch WARNING: Failed to POST event to {url}: {e}. "
                            "Events may be dropped. Check VETCH_ENDPOINT and network access.\n"
                        )
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


def configure_http_endpoint(url: str, api_key: str | None = None) -> None:
    """Configure HTTP endpoint for event emission.

    Events are POSTed as JSON to the specified URL asynchronously
    (non-blocking, fire-and-forget with a background thread).

    This is the programmatic equivalent of setting VETCH_ENDPOINT and
    VETCH_API_KEY environment variables.

    Args:
        url: HTTP or HTTPS endpoint URL to POST events to.
             Example: "https://analytics.internal.corp/vetch/ingest"
        api_key: Optional API key sent as ``Authorization: Bearer {key}``.
                 Leave None for endpoints that don't require authentication
                 (e.g., internal services behind a firewall).

    Example::

        import vetch

        vetch.configure_http_endpoint(
            "https://analytics.internal.corp/ingest",
            api_key="my-secret-key",  # omit if behind firewall
        )
        vetch.instrument()

        # All subsequent LLM calls will POST events to the endpoint.
    """

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            return record.getMessage()

    # Remove any existing HttpHandler for this URL to avoid duplication
    for h in logger.handlers[:]:
        if isinstance(h, HttpHandler) and h.url == url:
            logger.removeHandler(h)

    handler: logging.Handler = HttpHandler(url, api_key=api_key)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)


# Auto-configure on import if env vars are set (best effort)
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
