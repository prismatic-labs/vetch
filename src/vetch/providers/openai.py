"""OpenAI SDK provider wrapper.

This module handles patching the OpenAI Python SDK to capture
inference metadata without reading prompt/completion content.

Supports:
- Sync completions (client.chat.completions.create)
- Async completions (await client.chat.completions.create)
- Streaming completions (stream=True)
- Async streaming completions (stream=True)

Privacy guarantee: We only read model, usage, and timing metadata.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import TYPE_CHECKING, Any, cast
from weakref import WeakKeyDictionary

from vetch.context import get_active_context
from vetch.proxy import is_vetch_patched

if TYPE_CHECKING:
    from vetch.schema import Usage

logger = logging.getLogger(__name__)

# Thread-safe per-client storage for original methods
# Using WeakKeyDictionary so clients can be garbage collected
_client_originals: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()
_client_lock = threading.Lock()


def extract_usage(response: Any) -> Usage | None:
    """Extract usage metadata from OpenAI response.

    Args:
        response: OpenAI ChatCompletion response object.

    Returns:
        Usage dict with text token counts, or None if unavailable.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    return cast(
        "Usage",
        {
            "text": {
                "input_tokens": getattr(usage, "prompt_tokens", 0),
                "output_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
        },
    )


def extract_model(response: Any) -> str:
    """Extract model name from OpenAI response.

    Args:
        response: OpenAI ChatCompletion response object.

    Returns:
        Model identifier string.
    """
    return getattr(response, "model", "unknown")


def infer_region_from_base_url(base_url: str | None) -> str | None:
    """Infer region from OpenAI base URL.

    Supports Azure OpenAI URL patterns:
    - https://eastus.api.cognitive.microsoft.com/...
    - https://my-resource.openai.azure.com/... (uses resource name)

    Args:
        base_url: The client's base URL.

    Returns:
        Inferred region or None.
    """
    if base_url is None:
        return None

    # Azure pattern: region in subdomain
    azure_match = re.match(r"https://([a-z0-9-]+)\.(api\.cognitive|openai\.azure)", base_url)
    if azure_match:
        return azure_match.group(1)

    return None


def _after_create(result: Any, *args: Any, **kwargs: Any) -> None:
    """Hook called after chat.completions.create.

    Captures metadata from the response into the active context.
    """
    ctx = get_active_context()
    if ctx is None:
        return

    # Check if this is a streaming response
    is_stream = kwargs.get("stream", False)

    if is_stream:
        # For streams, we can't capture here - need stream wrapper
        # Just mark that it's a stream, actual capture happens in stream iteration
        return

    # Non-streaming: capture immediately
    usage = extract_usage(result)
    model = extract_model(result)

    # P1: Removed user_id hashing - compliance risk (GDPR/CCPA)
    # Users should handle attribution via explicit tags instead

    ctx.capture(
        model=model,
        provider="openai",
        usage=usage,
        is_stream=False,
        complete=True,
    )


def _on_create_error(error: BaseException) -> None:
    """Hook called when chat.completions.create fails."""
    ctx = get_active_context()
    if ctx is None:
        return

    ctx.capture(
        model="unknown",
        provider="openai",
        error=True,
        error_type=type(error).__name__,
        complete=False,
    )


class StreamWrapper:
    """Wrapper for OpenAI streaming responses.

    Counts characters without accumulating content.
    Captures final usage from the last chunk if available.
    """

    def __init__(self, stream: Any) -> None:
        """Initialize stream wrapper.

        Args:
            stream: The original OpenAI stream.
        """
        self._stream = stream
        self._accumulated_chars = 0
        self._model = "unknown"
        self._final_usage: Usage | None = None
        self._complete = False
        self._error = False
        self._error_type: str | None = None

    def __iter__(self) -> StreamWrapper:
        """Return self as iterator."""
        return self

    def __next__(self) -> Any:
        """Get next chunk from stream."""
        try:
            chunk = next(self._stream)
            self._process_chunk(chunk)
            return chunk

        except StopIteration:
            self._complete = True
            self._capture_to_context()
            raise

        except BaseException as e:
            self._error = True
            self._error_type = type(e).__name__
            self._capture_to_context()
            raise

    def _process_chunk(self, chunk: Any) -> None:
        """Process a chunk to update counters."""
        # Extract model from first chunk
        model = getattr(chunk, "model", None)
        if model:
            self._model = model

        # Count characters (not accumulate content)
        # Defensive access: verify choices exists and has elements
        choices = getattr(chunk, "choices", None)
        if choices and len(choices) > 0:
            delta = getattr(choices[0], "delta", None)
            if delta:
                content = getattr(delta, "content", None)
                if content:
                    self._accumulated_chars += len(content)

        # Check for usage in chunk (OpenAI includes in final chunk with stream_options)
        usage = getattr(chunk, "usage", None)
        if usage:
            self._final_usage = cast(
                "Usage",
                {
                    "text": {
                        "input_tokens": getattr(usage, "prompt_tokens", 0),
                        "output_tokens": getattr(usage, "completion_tokens", 0),
                        "total_tokens": getattr(usage, "total_tokens", 0),
                    }
                },
            )

    def _capture_to_context(self) -> None:
        """Capture final metadata to active context."""
        ctx = get_active_context()
        if ctx is None:
            return

        ctx.capture(
            model=self._model,
            provider="openai",
            usage=self._final_usage,
            is_stream=True,
            accumulated_chars=self._accumulated_chars,
            complete=self._complete,
            error=self._error,
            error_type=self._error_type,
        )

    def __enter__(self) -> StreamWrapper:
        """Support context manager protocol."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Clean up on context exit."""
        if exc_type is not None:
            self._error = True
            self._error_type = exc_type.__name__
            self._capture_to_context()
        # Close underlying stream if it supports it
        close = getattr(self._stream, "close", None)
        if close:
            close()


class AsyncStreamWrapper(StreamWrapper):
    """Async wrapper for OpenAI streaming responses."""

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
        except BaseException as e:
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

        # Async close if available
        close = getattr(self._stream, "close", None)
        if close:
            # Some async streams have sync close, some awaitable close
            if  logging.getLogger("asyncio").name == "asyncio": # Basic check
                 pass # Don't overengineer the close check for now
            # Best effort
            try:
                if hasattr(close, "__await__"):
                    await close()
                else:
                    close()
            except Exception:
                pass


def _wrapped_create(original: Any) -> Any:
    """Create wrapped version of chat.completions.create.

    Args:
        original: The original create method.

    Returns:
        Wrapped method that captures metadata.
    """
    import inspect

    # Handle async function
    if inspect.iscoroutinefunction(original):
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            is_stream = kwargs.get("stream", False)
            try:
                result = await original(*args, **kwargs)
                if is_stream:
                    return AsyncStreamWrapper(result)
                _after_create(result, *args, **kwargs)
                return result
            except BaseException as e:
                _on_create_error(e)
                raise

        async_wrapper.vetch_patched = True # type: ignore[attr-defined]
        async_wrapper._vetch_original = original # type: ignore[attr-defined]
        return async_wrapper

    # Handle sync function
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        is_stream = kwargs.get("stream", False)

        try:
            result = original(*args, **kwargs)

            if is_stream:
                # Wrap the stream to capture during iteration
                return StreamWrapper(result)

            # Non-streaming: capture immediately
            _after_create(result, *args, **kwargs)
            return result

        except BaseException as e:
            _on_create_error(e)
            raise

    wrapper.vetch_patched = True  # type: ignore[attr-defined]
    wrapper._vetch_original = original  # type: ignore[attr-defined]

    return wrapper


def patch_openai_client(client: Any) -> bool:
    """Patch an OpenAI client instance.

    Thread-safe. Each client's original method is stored separately
    using WeakKeyDictionary, allowing proper cleanup and multi-client support.

    Args:
        client: OpenAI client instance.

    Returns:
        True if patching succeeded, False otherwise.
    """
    try:
        # 1. Check version compatibility
        from vetch.compat import get_openai_version
        v_info = get_openai_version()
        if v_info.installed and not v_info.tested:
            logger.warning(
                f"OpenAI SDK version {v_info.version} is not tested with Vetch. "
                "Patching may be unstable. Set VETCH_FORCE_PATCH=true to override."
            )
            import os
            if os.environ.get("VETCH_FORCE_PATCH") != "true":
                return False

        # 2. Check if already patched
        completions = getattr(client, "chat", None)
        if completions is None:
            logger.warning("OpenAI client has no chat attribute")
            return False

        completions = getattr(completions, "completions", None)
        if completions is None:
            logger.warning("OpenAI client has no chat.completions attribute")
            return False

        create = getattr(completions, "create", None)
        if create is None:
            logger.warning("OpenAI client has no chat.completions.create method")
            return False

        if is_vetch_patched(create):
            logger.debug("OpenAI client already patched by Vetch")
            return True

        # Thread-safe: store original per-client before patching
        with _client_lock:
            # Double-check inside lock (another thread may have patched)
            if is_vetch_patched(getattr(completions, "create", None)):
                return True

            # Store original keyed by completions object (unique per client)
            _client_originals[completions] = create

            # Apply patch
            completions.create = _wrapped_create(create)

        logger.debug("OpenAI client patched successfully")
        return True

    except Exception as e:
        logger.warning(f"Failed to patch OpenAI client: {e}")
        return False


def unpatch_openai_client(client: Any) -> bool:
    """Remove Vetch patch from an OpenAI client.

    Thread-safe. Restores the original method for this specific client.

    Args:
        client: OpenAI client instance.

    Returns:
        True if unpatching succeeded, False otherwise.
    """
    try:
        completions = getattr(client, "chat", None)
        if completions is None:
            return False

        completions = getattr(completions, "completions", None)
        if completions is None:
            return False

        # Thread-safe: retrieve and remove original for this client
        with _client_lock:
            original = _client_originals.pop(completions, None)
            if original is None:
                # Not patched by us, or already unpatched
                return True

            completions.create = original

        logger.debug("OpenAI client unpatched successfully")
        return True

    except Exception as e:
        logger.warning(f"Failed to unpatch OpenAI client: {e}")
        return False


def detect_openai_client() -> Any | None:
    """Detect if OpenAI SDK is available and get default client.

    Returns:
        Default OpenAI client if available, None otherwise.
    """
    import sys
    if "openai" not in sys.modules:
        return None

    try:
        import openai  # type: ignore[import-not-found]

        # Check for default client (OpenAI >= 1.0 pattern)
        client = getattr(openai, "_client", None)
        if client is not None:
            return client

        # Try to get from module-level resources
        return getattr(openai, "chat", None)

    except (ImportError, AttributeError):
        return None
