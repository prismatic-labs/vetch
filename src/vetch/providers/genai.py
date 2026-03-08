"""Google GenAI SDK provider wrapper.

This module handles patching the Google GenAI Python SDK to capture
inference metadata without reading prompt/completion content.

Supports:
- Sync content generation (client.models.generate_content)
- Async content generation (client.aio.models.generate_content)
- Embeddings (client.models.embed_content)
- Streaming responses (stream=True)

Privacy guarantee: We only read model, usage, and timing metadata.
"""

from __future__ import annotations

import contextlib
import logging
import re
import threading
import types
import weakref
from collections.abc import AsyncGenerator, Generator
from typing import TYPE_CHECKING, Any, Callable
from weakref import WeakKeyDictionary

from vetch.context import get_active_context
from vetch.proxy import is_vetch_patched

if TYPE_CHECKING:
    from vetch.schema import Usage

logger = logging.getLogger(__name__)

# Pre-compiled regex for model name normalization
_VERSION_SUFFIX_PATTERN = re.compile(r"-\d{3,4}$")

# Thread-safe per-client storage for original methods
_client_originals: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()
_client_lock = threading.Lock()

# Module-level storage for original Client.__init__ (strong reference)
_module_original_init: Any = None


class _WeakMethodWrapper:
    """Wrapper that holds weak reference to client to avoid GC cycles.

    Problem: Closures that capture `client` create reference cycles:
      client -> method -> closure -> client

    Solution: Use weak reference to client in wrapper class.
    """

    __slots__ = ("_client_ref", "_method_name", "_originals_dict")

    def __init__(self, client: Any, method_name: str, originals_dict: WeakKeyDictionary) -> None:
        self._client_ref = weakref.ref(client)
        self._method_name = method_name
        self._originals_dict = originals_dict

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        client = self._client_ref()
        if client is None:
            raise RuntimeError("Client was garbage collected")

        ctx = get_active_context()
        original = self._originals_dict[client][self._method_name]

        if isinstance(original, tuple):
            orig_func, orig_self = original
            if ctx is None:
                return orig_func(orig_self, *args, **kwargs)
            response = orig_func(orig_self, *args, **kwargs)
        else:
            if ctx is None:
                return original(*args, **kwargs)
            response = original(*args, **kwargs)

        # Extract metadata
        usage, cache_read, cache_create = extract_usage(response)
        model = extract_model(response)

        # Capture metadata
        ctx.capture(
            model=model,
            provider="google_genai",
            usage=usage,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_create,
        )

        return response


class _WeakAsyncMethodWrapper:
    """Async version of _WeakMethodWrapper."""

    __slots__ = ("_client_ref", "_method_name", "_originals_dict")

    def __init__(self, client: Any, method_name: str, originals_dict: WeakKeyDictionary) -> None:
        self._client_ref = weakref.ref(client)
        self._method_name = method_name
        self._originals_dict = originals_dict

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        client = self._client_ref()
        if client is None:
            raise RuntimeError("Client was garbage collected")

        ctx = get_active_context()
        original = self._originals_dict[client][self._method_name]

        if isinstance(original, tuple):
            orig_func, orig_self = original
            if ctx is None:
                return await orig_func(orig_self, *args, **kwargs)
            response = await orig_func(orig_self, *args, **kwargs)
        else:
            if ctx is None:
                return await original(*args, **kwargs)
            response = await original(*args, **kwargs)

        # Extract metadata
        usage, cache_read, cache_create = extract_usage(response)
        model = extract_model(response)

        # Capture metadata
        ctx.capture(
            model=model,
            provider="google_genai",
            usage=usage,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_create,
        )

        return response


class _WeakEmbedWrapper:
    """Wrapper for embed_content with weak reference."""

    __slots__ = ("_client_ref", "_method_name", "_originals_dict")

    def __init__(self, client: Any, method_name: str, originals_dict: WeakKeyDictionary) -> None:
        self._client_ref = weakref.ref(client)
        self._method_name = method_name
        self._originals_dict = originals_dict

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        client = self._client_ref()
        if client is None:
            raise RuntimeError("Client was garbage collected")

        ctx = get_active_context()
        original = self._originals_dict[client][self._method_name]

        if isinstance(original, tuple):
            orig_func, orig_self = original
            if ctx is None:
                return orig_func(orig_self, *args, **kwargs)
            response = orig_func(orig_self, *args, **kwargs)
        else:
            if ctx is None:
                return original(*args, **kwargs)
            response = original(*args, **kwargs)

        # Extract metadata (embeddings use prompt_token_count)
        usage_metadata = getattr(response, "usage_metadata", None)
        if usage_metadata:
            input_tokens = getattr(usage_metadata, "prompt_token_count", 0)

            # Get model from kwargs or args
            model = kwargs.get("model", "text-embedding-004")
            if isinstance(model, str):
                model = _normalize_model_name(model)

            usage: Usage = {
                "text": {
                    "input_tokens": input_tokens,
                    "output_tokens": 0,  # Embeddings don't generate tokens
                    "total_tokens": input_tokens,
                }
            }

            ctx.capture(
                model=model,
                provider="google_genai",
                usage=usage,
                is_embedding=True,
            )

        return response


def extract_usage(response: Any) -> tuple[Usage | None, int | None, int | None]:
    """Extract usage metadata from Google GenAI response.

    Args:
        response: Google GenAI GenerateContentResponse object.

    Returns:
        Tuple of (Usage dict, cache_read_tokens, cache_creation_tokens).
    """
    usage_metadata = getattr(response, "usage_metadata", None)
    if usage_metadata is None:
        return None, None, None

    # Extract token counts
    input_tokens = getattr(usage_metadata, "prompt_token_count", 0)
    output_tokens = getattr(usage_metadata, "candidates_token_count", 0)
    total_tokens = getattr(usage_metadata, "total_token_count", 0)

    # Build usage dict
    usage_dict: Usage = {
        "text": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
    }

    # Extract reasoning tokens for extended thinking models (e.g., Gemini 2.0 Flash Thinking)
    # These tokens represent internal chain-of-thought reasoning and can be 10x+ the visible output
    reasoning_tokens = getattr(usage_metadata, "thought_token_count", 0)
    if reasoning_tokens > 0:
        usage_dict["reasoning"] = {
            "input_tokens": reasoning_tokens,
            "output_tokens": 0,  # Reasoning tokens are input-side (thinking)
            "total_tokens": reasoning_tokens,
        }

    # Google GenAI doesn't currently expose cache tokens in the same way
    # as Vertex AI, so we return None for now
    cache_read_tokens = None
    cache_creation_tokens = None

    return usage_dict, cache_read_tokens, cache_creation_tokens


def _normalize_model_name(model: str) -> str:
    """Normalize Google GenAI model names.

    Strips "models/" prefix and version suffixes like "-001".

    Args:
        model: Raw model name from Google GenAI SDK.

    Returns:
        Normalized model name (e.g., "gemini-1.5-pro").
    """
    # Strip "models/" prefix
    if model.startswith("models/"):
        model = model[7:]

    # Strip version suffixes like "-001", "-002" (pattern: -NNN or -NNNN at end)
    model = _VERSION_SUFFIX_PATTERN.sub("", model)

    return model


def extract_model(response: Any) -> str:
    """Extract and normalize model name from Google GenAI response.

    Args:
        response: Google GenAI response object.

    Returns:
        Normalized model identifier string.
    """
    # Google GenAI returns models in format "models/gemini-1.5-pro-001"
    # We normalize to just "gemini-1.5-pro"
    model = getattr(response, "model_name", None) or "unknown"

    if model == "unknown":
        return model

    return _normalize_model_name(model)


def patch_client(client: Any) -> None:
    """Patch a Google GenAI client to track inference.

    Args:
        client: google.genai.Client instance to patch.
    """
    if is_vetch_patched(client):
        return

    with _client_lock:
        if is_vetch_patched(client):
            return

        # Store originals
        _client_originals[client] = {
            "generate_content": None,
            "embed_content": None,
        }

        # Patch models.generate_content (sync)
        if hasattr(client, "models") and hasattr(client.models, "generate_content"):
            # Store as unbound method to avoid circular reference
            # Try to extract __func__ for real methods, fall back to storing directly for Mocks
            method = client.models.generate_content
            if hasattr(method, "__func__"):
                original_generate = method.__func__
                original_self = client.models
                _client_originals[client]["generate_content"] = (original_generate, original_self)
            else:
                # Mock object or already unbound - store directly
                _client_originals[client]["generate_content"] = method

            # Use wrapper class with weak reference to avoid GC cycle
            client.models.generate_content = _WeakMethodWrapper(
                client, "generate_content", _client_originals
            )

        # Patch aio.models.generate_content (async)
        if (
            hasattr(client, "aio")
            and hasattr(client.aio, "models")
            and hasattr(client.aio.models, "generate_content")
        ):
            # Store as unbound method to avoid circular reference
            method = client.aio.models.generate_content
            if hasattr(method, "__func__"):
                original_aio_generate = method.__func__
                original_aio_self = client.aio.models
                _client_originals[client]["aio_generate_content"] = (
                    original_aio_generate,
                    original_aio_self,
                )
            else:
                _client_originals[client]["aio_generate_content"] = method

            # Use async wrapper class with weak reference
            client.aio.models.generate_content = _WeakAsyncMethodWrapper(
                client, "aio_generate_content", _client_originals
            )

        # Patch models.embed_content (embeddings)
        if hasattr(client, "models") and hasattr(client.models, "embed_content"):
            # Store as unbound method to avoid circular reference
            method = client.models.embed_content
            if hasattr(method, "__func__"):
                original_embed = method.__func__
                original_embed_self = client.models
                _client_originals[client]["embed_content"] = (original_embed, original_embed_self)
            else:
                _client_originals[client]["embed_content"] = method

            # Use embed wrapper class with weak reference
            client.models.embed_content = _WeakEmbedWrapper(
                client, "embed_content", _client_originals
            )

        # Mark as patched (use vetch_patched not __vetch_patched__)
        client.vetch_patched = True

        logger.debug(f"Patched Google GenAI client: {client}")


def unpatch_client(client: Any) -> None:
    """Remove vetch patches from a Google GenAI client.

    Args:
        client: google.genai.Client instance to unpatch.
    """
    if not is_vetch_patched(client):
        return

    with _client_lock:
        if client not in _client_originals:
            return

        originals = _client_originals[client]

        # Restore originals (stored as either (unbound_func, self) tuples or direct Mock)
        if originals.get("generate_content"):
            original = originals["generate_content"]
            if isinstance(original, tuple):
                func, self_obj = original
                client.models.generate_content = types.MethodType(func, self_obj)
            else:
                client.models.generate_content = original

        if originals.get("aio_generate_content"):
            original = originals["aio_generate_content"]
            if isinstance(original, tuple):
                func, self_obj = original
                client.aio.models.generate_content = types.MethodType(func, self_obj)
            else:
                client.aio.models.generate_content = original

        if originals.get("embed_content"):
            original = originals["embed_content"]
            if isinstance(original, tuple):
                func, self_obj = original
                client.models.embed_content = types.MethodType(func, self_obj)
            else:
                client.models.embed_content = original

        # Clean up
        del _client_originals[client]
        delattr(client, "vetch_patched")

        logger.debug(f"Unpatched Google GenAI client: {client}")


@contextlib.contextmanager
def track_genai(client: Any) -> Generator[Any, None, None]:
    """Context manager to temporarily track a Google GenAI client (sync).

    Example:
        >>> import google.genai as genai
        >>> from vetch.providers.genai import track_genai
        >>>
        >>> client = genai.Client(api_key="...")
        >>> with track_genai(client):
        ...     response = client.models.generate_content(
        ...         model="gemini-1.5-pro",
        ...         contents="Hello!",
        ...     )

    Args:
        client: google.genai.Client instance to track.

    Yields:
        The patched client.
    """
    patch_client(client)
    try:
        yield client
    finally:
        unpatch_client(client)


@contextlib.asynccontextmanager
async def atrack_genai(client: Any) -> AsyncGenerator[Any, None]:
    """Async context manager to temporarily track a Google GenAI client.

    Example:
        >>> import google.genai as genai
        >>> from vetch.providers.genai import atrack_genai
        >>>
        >>> client = genai.Client(api_key="...")
        >>> async with atrack_genai(client):
        ...     response = await client.aio.models.generate_content(
        ...         model="gemini-2.0-flash",
        ...         contents="Hello!",
        ...     )

    Args:
        client: google.genai.Client instance to track.

    Yields:
        The patched client.
    """
    patch_client(client)
    try:
        yield client
    finally:
        unpatch_client(client)


# Module-level instrumentation (for vetch.instrument())
_module_instrumented = False


def instrument_genai_module() -> bool:
    """Instrument the google.genai module for automatic tracking.

    This patches the default Client constructor to automatically track
    all GenAI API calls without needing explicit context managers.

    Returns:
        True if instrumentation succeeded, False if module not available.

    Example:
        >>> import vetch
        >>> vetch.instrument()  # Automatically calls this function
        >>> import google.genai as genai
        >>> client = genai.Client(api_key="...")
        >>> # All calls are now automatically tracked!
    """
    global _module_instrumented, _module_original_init

    if _module_instrumented:
        return True

    try:
        import google.genai as genai  # type: ignore[import-not-found, import-untyped]
    except ImportError:
        return False

    # Store original Client.__init__ at MODULE level (strong reference)
    _module_original_init = genai.Client.__init__

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        """Patched Client.__init__ that auto-instruments."""
        # Call original init
        _module_original_init(self, *args, **kwargs)
        # Auto-patch this client
        patch_client(self)

    # Patch Client.__init__
    genai.Client.__init__ = patched_init

    _module_instrumented = True
    logger.info("Google GenAI module instrumented for automatic tracking")
    return True


def uninstrument_genai_module() -> bool:
    """Remove instrumentation from the google.genai module.

    Returns:
        True if uninstrumentation succeeded, False otherwise.
    """
    global _module_instrumented, _module_original_init

    if not _module_instrumented:
        return True

    # Restore original Client.__init__ if we have it
    if _module_original_init is not None:
        try:
            import google.genai as genai

            genai.Client.__init__ = _module_original_init
            _module_original_init = None
        except ImportError:
            pass

    _module_instrumented = False
    logger.info("Google GenAI module uninstrumented")
    return True
