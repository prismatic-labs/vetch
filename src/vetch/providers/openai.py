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

import contextlib
import logging
import re
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
# Using WeakKeyDictionary so clients can be garbage collected
_client_originals: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()
_client_lock = threading.Lock()


class _WeakChatWrapper:
    """Wrapper for sync chat.completions.create with weak reference.

    Problem: Closures that capture `original` (bound method) create reference cycles:
      completions -> create (wrapper) -> closure -> original (bound) -> completions

    Solution: Use weak reference to completions object and retrieve original from dict.
    """

    __slots__ = ("_completions_ref", "_originals_dict", "vetch_patched", "_vetch_original")

    def __init__(self, completions: Any, originals_dict: WeakKeyDictionary[Any, Any]) -> None:
        self._completions_ref = weakref.ref(completions)
        self._originals_dict = originals_dict

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        completions = self._completions_ref()
        if completions is None:
            raise RuntimeError("Completions object was garbage collected")

        original = self._originals_dict[completions]
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


class _WeakAsyncChatWrapper:
    """Async wrapper for chat.completions.create with weak reference."""

    __slots__ = ("_completions_ref", "_originals_dict", "vetch_patched", "_vetch_original")

    def __init__(self, completions: Any, originals_dict: WeakKeyDictionary[Any, Any]) -> None:
        self._completions_ref = weakref.ref(completions)
        self._originals_dict = originals_dict

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        completions = self._completions_ref()
        if completions is None:
            raise RuntimeError("Completions object was garbage collected")

        original = self._originals_dict[completions]
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


class _WeakEmbeddingsWrapper:
    """Wrapper for sync embeddings.create with weak reference."""

    __slots__ = ("_embeddings_ref", "_originals_dict", "vetch_patched", "_vetch_original")

    def __init__(self, embeddings: Any, originals_dict: WeakKeyDictionary[Any, Any]) -> None:
        self._embeddings_ref = weakref.ref(embeddings)
        self._originals_dict = originals_dict

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        embeddings = self._embeddings_ref()
        if embeddings is None:
            raise RuntimeError("Embeddings object was garbage collected")

        original = self._originals_dict[embeddings]

        try:
            result = original(*args, **kwargs)
            _after_embeddings_create(result, *args, **kwargs)
            return result

        except Exception as e:
            _on_embeddings_error(e)
            raise


class _WeakAsyncEmbeddingsWrapper:
    """Async wrapper for embeddings.create with weak reference."""

    __slots__ = ("_embeddings_ref", "_originals_dict", "vetch_patched", "_vetch_original")

    def __init__(self, embeddings: Any, originals_dict: WeakKeyDictionary[Any, Any]) -> None:
        self._embeddings_ref = weakref.ref(embeddings)
        self._originals_dict = originals_dict

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        embeddings = self._embeddings_ref()
        if embeddings is None:
            raise RuntimeError("Embeddings object was garbage collected")

        original = self._originals_dict[embeddings]

        try:
            result = await original(*args, **kwargs)
            _after_embeddings_create(result, *args, **kwargs)
            return result

        except Exception as e:
            _on_embeddings_error(e)
            raise


def extract_usage(response: Any) -> tuple[Usage | None, int | None, int | None]:
    """Extract usage metadata from OpenAI response including image tokens.

    Args:
        response: OpenAI ChatCompletion response object.

    Returns:
        Tuple of (Usage dict, cache_read_tokens, cache_creation_tokens).
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None, None

    # Extract cache tokens if available (OpenAI prompt caching)
    # OpenAI includes these in prompt_tokens_details
    cache_read_tokens = None
    cache_creation_tokens = None

    prompt_details = getattr(usage, "prompt_tokens_details", None)
    if prompt_details:
        cache_read_tokens = getattr(prompt_details, "cached_tokens", None)

    # Extract image tokens if available (GPT-4 Vision, GPT-4o)
    # OpenAI API includes image tokens in prompt_tokens_details
    image_input_tokens = 0
    if prompt_details:
        # GPT-4 Vision includes image tokens separately
        image_input_tokens = getattr(prompt_details, "image_tokens", 0) or getattr(
            prompt_details, "cached_image_tokens", 0
        )

    # Extract reasoning tokens if available (o1, o3, o1-mini models)
    # OpenAI includes these in completion_tokens_details
    reasoning_tokens = 0
    completion_details = getattr(usage, "completion_tokens_details", None)
    if completion_details:
        reasoning_tokens = getattr(completion_details, "reasoning_tokens", 0) or 0

    # Build usage dict with text and optional image/reasoning
    usage_dict: Usage = {
        "text": {
            "input_tokens": getattr(usage, "prompt_tokens", 0),
            "output_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }
    }

    # Add image usage if present
    if isinstance(image_input_tokens, int) and image_input_tokens > 0:
        usage_dict["image"] = {
            "input_tokens": image_input_tokens,
            "output_tokens": 0,
            "total_tokens": image_input_tokens,
            "image_count": 0,  # Not provided by OpenAI API
            "total_pixels": 0,  # Not provided by OpenAI API
        }

    # Add reasoning usage if present (o1/o3 thinking models)
    if isinstance(reasoning_tokens, int) and reasoning_tokens > 0:
        usage_dict["reasoning"] = {
            "input_tokens": reasoning_tokens,
            "output_tokens": 0,  # Reasoning tokens are input-side (thinking)
            "total_tokens": reasoning_tokens,
        }

    return usage_dict, cache_read_tokens, cache_creation_tokens


def extract_model(response: Any) -> str:
    """Extract model name from OpenAI response.

    Args:
        response: OpenAI ChatCompletion response object.

    Returns:
        Model identifier string.
    """
    return getattr(response, "model", "unknown")


def extract_embeddings_usage(response: Any) -> Usage | None:
    """Extract usage metadata from OpenAI embeddings response.

    Args:
        response: OpenAI CreateEmbeddingResponse object.

    Returns:
        Usage dict with input tokens only (embeddings don't generate output).
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    # Embeddings only consume input tokens, no output generation
    usage_dict: Usage = {
        "text": {
            "input_tokens": getattr(usage, "prompt_tokens", 0),
            "output_tokens": 0,  # Embeddings don't generate tokens
            "total_tokens": getattr(usage, "total_tokens", 0),
        }
    }

    return usage_dict


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
    from vetch.wrapper import auto_context_for_instrumented_call

    # Check if this is a streaming response
    is_stream = kwargs.get("stream", False)

    if is_stream:
        # For streams, we can't auto-wrap here
        # Stream wrapper handles context creation
        return

    # Auto-create context if needed, or use existing manual wrap() context
    with auto_context_for_instrumented_call("openai"):
        # Non-streaming: capture immediately
        usage, cache_read, cache_create = extract_usage(result)
        model = extract_model(result)

        # Get active context (either auto-created or manual wrap())
        ctx = get_active_context()
        if ctx is not None:
            ctx.capture(
                model=model,
                provider="openai",
                usage=usage,
                is_stream=False,
                complete=True,
                cache_read_tokens=cache_read,
                cache_creation_tokens=cache_create,
            )


def _on_create_error(error: BaseException) -> None:
    """Hook called when chat.completions.create fails."""
    from vetch.wrapper import auto_context_for_instrumented_call

    # Auto-create context if needed, or use existing manual wrap() context
    with auto_context_for_instrumented_call("openai"):
        ctx = get_active_context()
        if ctx is not None:
            ctx.capture(
                model="unknown",
                provider="openai",
                error=True,
                error_type=type(error).__name__,
                complete=False,
            )


def _after_embeddings_create(result: Any, *args: Any, **kwargs: Any) -> None:
    """Hook called after embeddings.create.

    Captures metadata from the embeddings response into the active context.
    """
    from vetch.wrapper import auto_context_for_instrumented_call

    # Auto-create context if needed, or use existing manual wrap() context
    with auto_context_for_instrumented_call("openai"):
        usage = extract_embeddings_usage(result)
        model = extract_model(result)

        ctx = get_active_context()
        if ctx is not None:
            ctx.capture(
                model=model,
                provider="openai",
                usage=usage,
                is_stream=False,
                is_embedding=True,  # Mark as embedding request
                complete=True,
            )


def _on_embeddings_error(error: BaseException) -> None:
    """Hook called when embeddings.create fails."""
    from vetch.wrapper import auto_context_for_instrumented_call

    # Auto-create context if needed, or use existing manual wrap() context
    with auto_context_for_instrumented_call("openai"):
        ctx = get_active_context()
        if ctx is not None:
            ctx.capture(
                model="unknown",
                provider="openai",
                error=True,
                error_type=type(error).__name__,
                is_embedding=True,
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
        self._cache_read_tokens: int | None = None
        self._cache_creation_tokens: int | None = None
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

        except Exception as e:
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
            # Extract cache tokens if available
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            if prompt_details:
                self._cache_read_tokens = getattr(prompt_details, "cached_tokens", None)

    def _capture_to_context(self) -> None:
        """Capture final metadata to active context (or create auto-context)."""
        from vetch.wrapper import auto_context_for_instrumented_call

        ctx = get_active_context()

        if ctx is not None:
            # Manual wrap() is active — capture to it; it emits on exit
            ctx.capture(
                model=self._model,
                provider="openai",
                usage=self._final_usage,
                is_stream=True,
                accumulated_chars=self._accumulated_chars,
                complete=self._complete,
                error=self._error,
                error_type=self._error_type,
                cache_read_tokens=self._cache_read_tokens,
                cache_creation_tokens=self._cache_creation_tokens,
            )
            return

        # Instrumented mode (no manual wrap()) — create auto-context at stream completion
        with auto_context_for_instrumented_call("openai"):
            ctx = get_active_context()
            if ctx is not None:
                ctx.capture(
                    model=self._model,
                    provider="openai",
                    usage=self._final_usage,
                    is_stream=True,
                    accumulated_chars=self._accumulated_chars,
                    complete=self._complete,
                    error=self._error,
                    error_type=self._error_type,
                    cache_read_tokens=self._cache_read_tokens,
                    cache_creation_tokens=self._cache_creation_tokens,
                )
        # auto-context exits here → event emitted

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
            except Exception as e:
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

        except Exception as e:
            _on_create_error(e)
            raise

    wrapper.vetch_patched = True  # type: ignore[attr-defined]
    wrapper._vetch_original = original  # type: ignore[attr-defined]

    return wrapper


def _wrapped_embeddings_create(original: Any) -> Any:
    """Create wrapped version of embeddings.create.

    Args:
        original: The original embeddings create method.

    Returns:
        Wrapped method that captures metadata.
    """
    import inspect

    # Handle async function
    if inspect.iscoroutinefunction(original):
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                result = await original(*args, **kwargs)
                _after_embeddings_create(result, *args, **kwargs)
                return result
            except Exception as e:
                _on_embeddings_error(e)
                raise

        async_wrapper.vetch_patched = True  # type: ignore[attr-defined]
        async_wrapper._vetch_original = original  # type: ignore[attr-defined]
        return async_wrapper

    # Handle sync function
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            result = original(*args, **kwargs)
            _after_embeddings_create(result, *args, **kwargs)
            return result
        except Exception as e:
            _on_embeddings_error(e)
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

            # Store original function (not bound method) to avoid circular reference
            # Bound methods hold a reference to the object, preventing garbage collection
            _client_originals[completions] = (
                create.__func__ if hasattr(create, "__func__") else create
            )

            # Apply patch using weak reference wrapper to avoid GC cycles
            import inspect
            wrapper: Any
            if inspect.iscoroutinefunction(create):
                wrapper = _WeakAsyncChatWrapper(completions, _client_originals)
            else:
                wrapper = _WeakChatWrapper(completions, _client_originals)

            wrapper.vetch_patched = True
            wrapper._vetch_original = _client_originals[completions]
            completions.create = wrapper

        # 3. Patch embeddings.create if available
        embeddings = getattr(client, "embeddings", None)
        if embeddings:
            embeddings_create = getattr(embeddings, "create", None)
            if embeddings_create and not is_vetch_patched(embeddings_create):
                with _client_lock:
                    # Double-check inside lock
                    if not is_vetch_patched(getattr(embeddings, "create", None)):
                        # Store original function (not bound method) to avoid circular reference
                        _client_originals[embeddings] = (
                            embeddings_create.__func__
                            if hasattr(embeddings_create, "__func__")
                            else embeddings_create
                        )

                        # Apply patch using weak reference wrapper to avoid GC cycles
                        import inspect
                        emb_wrapper: Any
                        if inspect.iscoroutinefunction(embeddings_create):
                            emb_wrapper = _WeakAsyncEmbeddingsWrapper(embeddings, _client_originals)
                        else:
                            emb_wrapper = _WeakEmbeddingsWrapper(embeddings, _client_originals)

                        emb_wrapper.vetch_patched = True
                        emb_wrapper._vetch_original = _client_originals[embeddings]
                        embeddings.create = emb_wrapper
                        logger.debug("OpenAI embeddings endpoint patched successfully")

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

        # Thread-safe: retrieve and remove original for chat completions
        with _client_lock:
            original = _client_originals.pop(completions, None)
            if original is not None:
                completions.create = original

        # Unpatch embeddings if it was patched
        embeddings = getattr(client, "embeddings", None)
        if embeddings:
            with _client_lock:
                embeddings_original = _client_originals.pop(embeddings, None)
                if embeddings_original is not None:
                    embeddings.create = embeddings_original

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


# Track if module is instrumented
_module_instrumented = False
_module_instrumentation_lock = threading.Lock()

# Store original __init__ methods for uninstrumentation
_original_openai_init: Any | None = None
_original_async_openai_init: Any | None = None


def instrument_openai_module() -> bool:
    """Instrument the OpenAI module to auto-track all client instances.

    Patches the OpenAI class __init__ to automatically call patch_openai_client
    on every new client instance.

    Thread-safe: uses lock to prevent race conditions during instrumentation.

    Returns:
        True if instrumentation succeeded, False otherwise.
    """
    global _module_instrumented, _original_openai_init, _original_async_openai_init
    import sys

    # Fast path without lock
    if _module_instrumented:
        return True

    if "openai" not in sys.modules:
        return False

    # Acquire lock for instrumentation
    with _module_instrumentation_lock:
        # Double-check after acquiring lock (another thread may have instrumented)
        if _module_instrumented:
            return True

        try:
            import openai

            # Store original __init__ for later restoration
            _original_openai_init = openai.OpenAI.__init__

            def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
                _original_openai_init(self, *args, **kwargs)
                # Auto-patch this client instance
                with contextlib.suppress(Exception):
                    patch_openai_client(self)

            openai.OpenAI.__init__ = patched_init

            # Also patch AsyncOpenAI if available
            if hasattr(openai, "AsyncOpenAI"):
                _original_async_openai_init = openai.AsyncOpenAI.__init__

                def patched_async_init(self: Any, *args: Any, **kwargs: Any) -> None:
                    _original_async_openai_init(self, *args, **kwargs)
                    with contextlib.suppress(Exception):
                        patch_openai_client(self)

                openai.AsyncOpenAI.__init__ = patched_async_init

            _module_instrumented = True
            logger.debug("OpenAI module instrumented")
            return True

        except Exception as e:
            logger.debug(f"Failed to instrument OpenAI module: {e}")
            return False


def uninstrument_openai_module() -> bool:
    """Remove Vetch instrumentation from OpenAI module.

    Restores the original __init__ methods and clears tracking state.

    Returns:
        True if uninstrumentation succeeded, False otherwise.
    """
    global _module_instrumented, _original_openai_init, _original_async_openai_init
    import sys

    if not _module_instrumented:
        return True

    if "openai" not in sys.modules:
        _module_instrumented = False
        return True

    try:
        import openai

        # Atomic: restore all methods under lock, then clear state.
        # Order: restore per-client methods first (so in-flight calls
        # finish against originals), then restore __init__ (so new
        # clients stop getting patched), then clear registry.
        with _client_lock:
            for completions, original_create in list(_client_originals.items()):
                with contextlib.suppress(Exception):
                    completions.create = original_create
            _client_originals.clear()

        # Restore original __init__ (after per-client cleanup)
        if _original_openai_init is not None:
            openai.OpenAI.__init__ = _original_openai_init

        # Restore AsyncOpenAI if we patched it
        if _original_async_openai_init is not None and hasattr(openai, "AsyncOpenAI"):
            openai.AsyncOpenAI.__init__ = _original_async_openai_init

        _module_instrumented = False
        _original_openai_init = None
        _original_async_openai_init = None
        logger.debug("OpenAI module uninstrumented")
        return True

    except Exception as e:
        logger.debug(f"Failed to uninstrument OpenAI module: {e}")
        return False
