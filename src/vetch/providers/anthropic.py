"""Anthropic SDK provider wrapper.

This module handles patching the Anthropic Python SDK to capture
inference metadata without reading prompt/completion content.

Supports:
- Sync messages (client.messages.create)
- Async messages (await client.messages.create)
- Streaming messages (stream=True)
- Async streaming messages (stream=True)
"""

from __future__ import annotations

import contextlib
import logging
import threading
import weakref
from typing import TYPE_CHECKING, Any, cast
from weakref import WeakKeyDictionary

from vetch.context import get_active_context
from vetch.proxy import is_vetch_patched

if TYPE_CHECKING:
    from vetch.schema import Usage

logger = logging.getLogger(__name__)

# Thread-safe per-client storage for original methods
_client_originals: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()
_client_lock = threading.Lock()


class _WeakMessagesWrapper:
    """Wrapper for sync messages.create with weak reference.

    Problem: Closures that capture `original` (bound method) create reference cycles:
      messages -> create (wrapper) -> closure -> original (bound) -> messages

    Solution: Use weak reference to messages object and retrieve original from dict.
    """

    __slots__ = ("_messages_ref", "_originals_dict")

    def __init__(self, messages: Any, originals_dict: WeakKeyDictionary) -> None:
        self._messages_ref = weakref.ref(messages)
        self._originals_dict = originals_dict

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        messages = self._messages_ref()
        if messages is None:
            raise RuntimeError("Messages object was garbage collected")

        original = self._originals_dict[messages]
        is_stream = kwargs.get("stream", False)

        try:
            result = original(*args, **kwargs)

            if is_stream:
                return StreamWrapper(result)

            _after_create(result, *args, **kwargs)
            return result

        except Exception as e:
            _on_create_error(e)
            raise


class _WeakAsyncMessagesWrapper:
    """Async wrapper for messages.create with weak reference."""

    __slots__ = ("_messages_ref", "_originals_dict")

    def __init__(self, messages: Any, originals_dict: WeakKeyDictionary) -> None:
        self._messages_ref = weakref.ref(messages)
        self._originals_dict = originals_dict

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        messages = self._messages_ref()
        if messages is None:
            raise RuntimeError("Messages object was garbage collected")

        original = self._originals_dict[messages]
        is_stream = kwargs.get("stream", False)

        try:
            result = await original(*args, **kwargs)

            if is_stream:
                return AsyncStreamWrapper(result)

            _after_create(result, *args, **kwargs)
            return result

        except Exception as e:
            _on_create_error(e)
            raise


def extract_usage(response: Any) -> tuple[Usage | None, int | None, int | None]:
    """Extract usage metadata from Anthropic response.

    Args:
        response: Anthropic Message object.

    Returns:
        Tuple of (Usage dict, cache_read_tokens, cache_creation_tokens).
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None, None

    # Anthropic prompt caching tokens
    cache_read_tokens = getattr(usage, "cache_read_input_tokens", None)
    cache_creation_tokens = getattr(usage, "cache_creation_input_tokens", None)

    return (
        cast(
            "Usage",
            {
                "text": {
                    "input_tokens": getattr(usage, "input_tokens", 0),
                    "output_tokens": getattr(usage, "output_tokens", 0),
                    "total_tokens": getattr(usage, "input_tokens", 0)
                    + getattr(usage, "output_tokens", 0),
                }
            },
        ),
        cache_read_tokens,
        cache_creation_tokens,
    )


def extract_model(response: Any) -> str:
    """Extract model name from Anthropic response.

    Args:
        response: Anthropic Message object.

    Returns:
        Model identifier string.
    """
    return getattr(response, "model", "unknown")


def _after_create(result: Any, *args: Any, **kwargs: Any) -> None:
    """Hook called after messages.create.

    Captures metadata from the response into the active context.
    """
    ctx = get_active_context()
    if ctx is None:
        return

    # Check if this is a streaming response
    is_stream = kwargs.get("stream", False)

    if is_stream:
        return

    # Non-streaming: capture immediately
    usage, cache_read, cache_create = extract_usage(result)
    model = extract_model(result)

    ctx.capture(
        model=model,
        provider="anthropic",
        usage=usage,
        is_stream=False,
        complete=True,
        cache_read_tokens=cache_read,
        cache_creation_tokens=cache_create,
    )


def _on_create_error(error: BaseException) -> None:
    """Hook called when messages.create fails."""
    ctx = get_active_context()
    if ctx is None:
        return

    ctx.capture(
        model="unknown",
        provider="anthropic",
        error=True,
        error_type=type(error).__name__,
        complete=False,
    )


class StreamWrapper:
    """Wrapper for Anthropic streaming responses.

    Counts characters without accumulating content.
    Captures usage from message_start and message_delta events.
    """

    def __init__(self, stream: Any) -> None:
        self._stream = stream
        self._accumulated_chars = 0
        self._model = "unknown"
        self._input_tokens = 0
        self._output_tokens = 0
        self._cache_read_tokens: int | None = None
        self._cache_creation_tokens: int | None = None
        self._complete = False
        self._error = False
        self._error_type: str | None = None

    def __iter__(self) -> StreamWrapper:
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._stream)
            self._process_chunk(chunk)
            return chunk
        except StopIteration:
            self._complete = True
            self._capture_to_context()
            raise
        except Exception as e:
            self._error = True
            self._error_type = type(e).__name__
            self._capture_to_context()
            raise

    def _process_chunk(self, chunk: Any) -> None:
        # Anthropic chunks are events: message_start, content_block_delta, message_delta, etc.

        event_type = getattr(chunk, "type", "")

        if event_type == "message_start":
            msg = getattr(chunk, "message", None)
            if msg:
                self._model = getattr(msg, "model", "unknown")
                usage = getattr(msg, "usage", None)
                if usage:
                    self._input_tokens += getattr(usage, "input_tokens", 0)
                    # Extract cache tokens if present
                    cache_read = getattr(usage, "cache_read_input_tokens", None)
                    cache_create = getattr(usage, "cache_creation_input_tokens", None)
                    if cache_read is not None:
                        self._cache_read_tokens = cache_read
                    if cache_create is not None:
                        self._cache_creation_tokens = cache_create

        elif event_type == "content_block_delta":
            delta = getattr(chunk, "delta", None)
            if delta:
                text = getattr(delta, "text", "")
                self._accumulated_chars += len(text)

        elif event_type == "message_delta":
            usage = getattr(chunk, "usage", None)
            if usage:
                self._output_tokens += getattr(usage, "output_tokens", 0)

    def _capture_to_context(self) -> None:
        ctx = get_active_context()
        if ctx is None:
            return

        final_usage = {
            "text": {
                "input_tokens": self._input_tokens,
                "output_tokens": self._output_tokens,
                "total_tokens": self._input_tokens + self._output_tokens,
            }
        }

        ctx.capture(
            model=self._model,
            provider="anthropic",
            usage=final_usage,  # type: ignore[arg-type]
            is_stream=True,
            accumulated_chars=self._accumulated_chars,
            complete=self._complete,
            error=self._error,
            error_type=self._error_type,
            cache_read_tokens=self._cache_read_tokens,
            cache_creation_tokens=self._cache_creation_tokens,
        )

    def __enter__(self) -> StreamWrapper:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        if exc_type is not None:
            self._error = True
            self._error_type = exc_type.__name__
            self._capture_to_context()
        close = getattr(self._stream, "close", None)
        if close:
            close()


class AsyncStreamWrapper(StreamWrapper):
    """Async wrapper for Anthropic streaming responses."""

    def __aiter__(self) -> AsyncStreamWrapper:
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._stream.__anext__()
            self._process_chunk(chunk)
            return chunk
        except StopAsyncIteration:
            self._complete = True
            self._capture_to_context()
            raise
        except Exception as e:
            self._error = True
            self._error_type = type(e).__name__
            self._capture_to_context()
            raise

    async def __aenter__(self) -> AsyncStreamWrapper:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        if exc_type is not None:
            self._error = True
            self._error_type = exc_type.__name__
            self._capture_to_context()

        close = getattr(self._stream, "close", None)
        if close:
            try:
                if hasattr(close, "__await__"):
                    await close()
                else:
                    close()
            except Exception:
                pass


def _wrapped_create(original: Any) -> Any:
    """Create wrapped version of messages.create."""
    import inspect

    if inspect.iscoroutinefunction(original):

        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            is_stream = kwargs.get("stream", False)
            try:
                result = await original(*args, **kwargs)
                if is_stream:
                    return AsyncStreamWrapper(result)
                _after_create(result, *args, **kwargs)
                return result
            except Exception as e:
                _on_create_error(e)
                raise

        async_wrapper.vetch_patched = True  # type: ignore[attr-defined]
        async_wrapper._vetch_original = original  # type: ignore[attr-defined]
        return async_wrapper

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        is_stream = kwargs.get("stream", False)
        try:
            result = original(*args, **kwargs)
            if is_stream:
                return StreamWrapper(result)
            _after_create(result, *args, **kwargs)
            return result
        except Exception as e:
            _on_create_error(e)
            raise

    wrapper.vetch_patched = True  # type: ignore[attr-defined]
    wrapper._vetch_original = original  # type: ignore[attr-defined]
    return wrapper


def patch_anthropic_client(client: Any) -> bool:
    """Patch an Anthropic client instance.

    Thread-safe. Each client's original method is stored separately.

    Args:
        client: Anthropic client instance.

    Returns:
        True if patching succeeded, False otherwise.
    """
    try:
        messages = getattr(client, "messages", None)
        if messages is None:
            return False

        create = getattr(messages, "create", None)
        if create is None:
            return False

        if is_vetch_patched(create):
            return True

        # Thread-safe: store original per-client before patching
        with _client_lock:
            # Double-check inside lock
            if is_vetch_patched(getattr(messages, "create", None)):
                return True

            # Store unbound function to avoid circular reference
            # Bound methods hold a reference to the object, preventing garbage collection
            _client_originals[messages] = (
                create.__func__ if hasattr(create, "__func__") else create
            )

            # Apply patch using weak reference wrapper to avoid GC cycles
            import inspect
            if inspect.iscoroutinefunction(create):
                wrapper = _WeakAsyncMessagesWrapper(messages, _client_originals)
            else:
                wrapper = _WeakMessagesWrapper(messages, _client_originals)

            wrapper.vetch_patched = True  # type: ignore[attr-defined]
            wrapper._vetch_original = _client_originals[messages]  # type: ignore[attr-defined]
            messages.create = wrapper

        logger.debug("Anthropic client patched successfully")
        return True

    except Exception as e:
        logger.warning(f"Failed to patch Anthropic client: {e}")
        return False


def unpatch_anthropic_client(client: Any) -> bool:
    """Remove patch from Anthropic client.

    Thread-safe. Restores the original method for this specific client.

    Args:
        client: Anthropic client instance.

    Returns:
        True if unpatching succeeded, False otherwise.
    """
    try:
        messages = getattr(client, "messages", None)
        if messages is None:
            return False

        with _client_lock:
            original = _client_originals.pop(messages, None)
            if original is None:
                return True

            # Restore original (may be unbound function or direct method)
            if hasattr(original, "__get__"):
                # Unbound function - bind it back to the object
                messages.create = original.__get__(messages, type(messages))
            else:
                messages.create = original

        logger.debug("Anthropic client unpatched successfully")
        return True

    except Exception as e:
        logger.warning(f"Failed to unpatch Anthropic client: {e}")
        return False


def detect_anthropic_client() -> Any | None:
    """Detect default Anthropic client."""
    import sys

    if "anthropic" not in sys.modules:
        return None

    # Anthropic doesn't have a global default client easily accessible
    return None


# Track if module is instrumented
_module_instrumented = False

# Store original __init__ methods for uninstrumentation
_original_anthropic_init: Any | None = None
_original_async_anthropic_init: Any | None = None


def instrument_anthropic_module() -> bool:
    """Instrument the Anthropic module to auto-track all client instances.

    Patches the Anthropic class __init__ to automatically call patch_anthropic_client
    on every new client instance.

    Returns:
        True if instrumentation succeeded, False otherwise.
    """
    global _module_instrumented, _original_anthropic_init, _original_async_anthropic_init
    import sys

    if _module_instrumented:
        return True

    if "anthropic" not in sys.modules:
        return False

    try:
        import anthropic  # type: ignore[import-not-found]

        # Store original __init__ for later restoration
        _original_anthropic_init = anthropic.Anthropic.__init__

        def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
            _original_anthropic_init(self, *args, **kwargs)
            # Auto-patch this client instance
            with contextlib.suppress(Exception):
                patch_anthropic_client(self)

        anthropic.Anthropic.__init__ = patched_init

        # Also patch AsyncAnthropic if available
        if hasattr(anthropic, "AsyncAnthropic"):
            _original_async_anthropic_init = anthropic.AsyncAnthropic.__init__

            def patched_async_init(self: Any, *args: Any, **kwargs: Any) -> None:
                _original_async_anthropic_init(self, *args, **kwargs)
                with contextlib.suppress(Exception):
                    patch_anthropic_client(self)

            anthropic.AsyncAnthropic.__init__ = patched_async_init

        _module_instrumented = True
        logger.debug("Anthropic module instrumented")
        return True

    except Exception as e:
        logger.debug(f"Failed to instrument Anthropic module: {e}")
        return False


def uninstrument_anthropic_module() -> bool:
    """Remove Vetch instrumentation from Anthropic module.

    Restores the original __init__ methods and clears tracking state.

    Returns:
        True if uninstrumentation succeeded, False otherwise.
    """
    global _module_instrumented, _original_anthropic_init, _original_async_anthropic_init
    import sys

    if not _module_instrumented:
        return True

    if "anthropic" not in sys.modules:
        _module_instrumented = False
        return True

    try:
        import anthropic

        # Atomic: restore per-client methods first, then __init__
        with _client_lock:
            for messages, original_create in list(_client_originals.items()):
                with contextlib.suppress(Exception):
                    messages.create = original_create
            _client_originals.clear()

        if _original_anthropic_init is not None:
            anthropic.Anthropic.__init__ = _original_anthropic_init

        if _original_async_anthropic_init is not None and hasattr(anthropic, "AsyncAnthropic"):
            anthropic.AsyncAnthropic.__init__ = _original_async_anthropic_init

        _module_instrumented = False
        _original_anthropic_init = None
        _original_async_anthropic_init = None
        logger.debug("Anthropic module uninstrumented")
        return True

    except Exception as e:
        logger.debug(f"Failed to uninstrument Anthropic module: {e}")
        return False
