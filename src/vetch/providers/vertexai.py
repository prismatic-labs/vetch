"""Vertex AI SDK provider wrapper.

This module handles patching the Google Vertex AI Python SDK to capture
inference metadata without reading prompt/completion content.

Supports:
- Sync completions (model.generate_content)
- Async completions (model.generate_content_async)
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
from typing import TYPE_CHECKING, Any, NamedTuple, cast
from weakref import WeakKeyDictionary

from vetch.context import get_active_context
from vetch.proxy import is_vetch_patched

if TYPE_CHECKING:
    from vetch.schema import Usage

logger = logging.getLogger(__name__)


class _ModelOriginals(NamedTuple):
    """Stores original methods for a single model instance."""

    generate: Any
    generate_async: Any


# Thread-safe per-model storage for original methods
_model_originals: WeakKeyDictionary[Any, _ModelOriginals] = WeakKeyDictionary()
_model_lock = threading.Lock()


class _WeakGenerateWrapper:
    """Wrapper for sync generate_content with weak reference.

    Problem: Closures that capture `original` and `model_obj` create reference cycles:
      model -> generate_content (wrapper) -> closure -> original/model_obj -> model

    Solution: Use weak reference to model object and retrieve original from dict.
    """

    __slots__ = ("_model_ref", "_originals_dict", "vetch_patched", "_vetch_original")

    def __init__(self, model: Any, originals_dict: WeakKeyDictionary[Any, _ModelOriginals]) -> None:
        self._model_ref = weakref.ref(model)
        self._originals_dict = originals_dict

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        model = self._model_ref()
        if model is None:
            raise RuntimeError("Model object was garbage collected")

        originals = self._originals_dict[model]
        original = originals.generate
        model_name = extract_model(model)
        is_stream = kwargs.get("stream", False)

        try:
            result = original(*args, **kwargs)

            if is_stream:
                return StreamWrapper(result, model_name)

            _after_generate(result, model, *args, **kwargs)
            return result

        except Exception as e:
            _on_generate_error(e)
            raise


class _WeakGenerateAsyncWrapper:
    """Async wrapper for generate_content_async with weak reference."""

    __slots__ = ("_model_ref", "_originals_dict", "vetch_patched", "_vetch_original")

    def __init__(self, model: Any, originals_dict: WeakKeyDictionary[Any, _ModelOriginals]) -> None:
        self._model_ref = weakref.ref(model)
        self._originals_dict = originals_dict

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        model = self._model_ref()
        if model is None:
            raise RuntimeError("Model object was garbage collected")

        originals = self._originals_dict[model]
        original = originals.generate_async
        model_name = extract_model(model)
        is_stream = kwargs.get("stream", False)

        try:
            result = await original(*args, **kwargs)

            if is_stream:
                return AsyncStreamWrapper(result, model_name)

            _after_generate(result, model, *args, **kwargs)
            return result

        except Exception as e:
            _on_generate_error(e)
            raise


def extract_usage(response: Any) -> Usage | None:
    """Extract usage metadata from Vertex AI response.

    Args:
        response: Vertex AI GenerateContentResponse object.

    Returns:
        Usage dict with text token counts, or None if unavailable.
    """
    # Vertex AI uses usage_metadata attribute
    usage_metadata = getattr(response, "usage_metadata", None)
    if usage_metadata is None:
        return None

    return cast(
        "Usage",
        {
            "text": {
                "input_tokens": getattr(usage_metadata, "prompt_token_count", 0),
                "output_tokens": getattr(usage_metadata, "candidates_token_count", 0),
                "total_tokens": getattr(usage_metadata, "total_token_count", 0),
            }
        },
    )


def extract_model(model_obj: Any) -> str:
    """Extract model name from Vertex AI model object.

    Args:
        model_obj: Vertex AI GenerativeModel instance.

    Returns:
        Model identifier string.
    """
    # Try to get model name from the model object
    model_name = getattr(model_obj, "_model_name", None)
    if model_name:
        # Clean up the model name (e.g., "models/gemini-1.5-pro" -> "gemini-1.5-pro")
        if "/" in model_name:
            return str(model_name.split("/")[-1])
        return str(model_name)
    return "unknown"


def infer_region_from_endpoint(endpoint: str | None) -> str | None:
    """Infer region from Vertex AI endpoint URL.

    Supports patterns like:
    - us-central1-aiplatform.googleapis.com

    Args:
        endpoint: The API endpoint URL.

    Returns:
        Inferred region or None.
    """
    if endpoint is None:
        return None

    # Pattern: region-aiplatform.googleapis.com
    match = re.match(r"([a-z]+-[a-z]+\d+)-aiplatform\.googleapis\.com", endpoint)
    if match:
        return match.group(1)

    return None


def _after_generate(result: Any, model_obj: Any, *args: Any, **kwargs: Any) -> None:
    """Hook called after model.generate_content.

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
    with auto_context_for_instrumented_call("vertexai"):
        # Non-streaming: capture immediately
        usage = extract_usage(result)
        model = extract_model(model_obj)

        ctx = get_active_context()
        if ctx is not None:
            ctx.capture(
                model=model,
                provider="vertexai",
                usage=usage,
                is_stream=False,
                complete=True,
            )


def _on_generate_error(error: BaseException) -> None:
    """Hook called when model.generate_content fails."""
    from vetch.wrapper import auto_context_for_instrumented_call

    # Auto-create context if needed, or use existing manual wrap() context
    with auto_context_for_instrumented_call("vertexai"):
        ctx = get_active_context()
        if ctx is not None:
            ctx.capture(
                model="unknown",
                provider="vertexai",
                error=True,
                error_type=type(error).__name__,
                complete=False,
            )


class StreamWrapper:
    """Wrapper for Vertex AI streaming responses.

    Counts characters without accumulating content.
    Captures final usage from the last chunk if available.
    """

    def __init__(self, stream: Any, model_name: str) -> None:
        """Initialize stream wrapper.

        Args:
            stream: The original Vertex AI stream.
            model_name: Model identifier for capture.
        """
        self._stream = stream
        self._model = model_name
        self._accumulated_chars = 0
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

        except Exception as e:
            self._error = True
            self._error_type = type(e).__name__
            self._capture_to_context()
            raise

    def _process_chunk(self, chunk: Any) -> None:
        """Process a single chunk to update counters."""
        # Count characters from text content (not accumulate)
        text = getattr(chunk, "text", None)
        if text:
            self._accumulated_chars += len(text)

        # Check for usage in chunk
        usage_metadata = getattr(chunk, "usage_metadata", None)
        if usage_metadata:
            self._final_usage = cast(
                "Usage",
                {
                    "text": {
                        "input_tokens": getattr(usage_metadata, "prompt_token_count", 0),
                        "output_tokens": getattr(usage_metadata, "candidates_token_count", 0),
                        "total_tokens": getattr(usage_metadata, "total_token_count", 0),
                    }
                },
            )

    def _capture_to_context(self) -> None:
        """Capture final metadata to active context (or create auto-context)."""
        from vetch.wrapper import auto_context_for_instrumented_call

        ctx = get_active_context()

        if ctx is not None:
            # Manual wrap() is active — capture to it; it emits on exit
            ctx.capture(
                model=self._model,
                provider="vertexai",
                usage=self._final_usage,
                is_stream=True,
                accumulated_chars=self._accumulated_chars,
                complete=self._complete,
                error=self._error,
                error_type=self._error_type,
            )
            return

        # Instrumented mode (no manual wrap()) — create auto-context at stream completion
        with auto_context_for_instrumented_call("vertexai"):
            ctx = get_active_context()
            if ctx is not None:
                ctx.capture(
                    model=self._model,
                    provider="vertexai",
                    usage=self._final_usage,
                    is_stream=True,
                    accumulated_chars=self._accumulated_chars,
                    complete=self._complete,
                    error=self._error,
                    error_type=self._error_type,
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


class AsyncStreamWrapper(StreamWrapper):
    """Async wrapper for Vertex AI streaming responses."""

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


def _wrapped_generate(original: Any, model_obj: Any) -> Any:
    """Create wrapped version of model.generate_content.

    Args:
        original: The original generate_content method.
        model_obj: The GenerativeModel instance.

    Returns:
        Wrapped method that captures metadata.
    """
    model_name = extract_model(model_obj)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        is_stream = kwargs.get("stream", False)

        try:
            result = original(*args, **kwargs)

            if is_stream:
                # Wrap the stream to capture during iteration
                return StreamWrapper(result, model_name)

            # Non-streaming: capture immediately
            _after_generate(result, model_obj, *args, **kwargs)
            return result

        except Exception as e:
            _on_generate_error(e)
            raise

    wrapper.vetch_patched = True  # type: ignore[attr-defined]
    wrapper._vetch_original = original  # type: ignore[attr-defined]

    return wrapper


def _wrapped_generate_async(original: Any, model_obj: Any) -> Any:
    """Create wrapped version of model.generate_content_async.

    Args:
        original: The original generate_content_async method.
        model_obj: The GenerativeModel instance.

    Returns:
        Wrapped method that captures metadata.
    """
    model_name = extract_model(model_obj)

    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        is_stream = kwargs.get("stream", False)

        try:
            result = await original(*args, **kwargs)

            if is_stream:
                # Wrap the stream to capture during iteration
                return AsyncStreamWrapper(result, model_name)

            # Non-streaming: capture immediately
            _after_generate(result, model_obj, *args, **kwargs)
            return result

        except Exception as e:
            _on_generate_error(e)
            raise

    wrapper.vetch_patched = True  # type: ignore[attr-defined]
    wrapper._vetch_original = original  # type: ignore[attr-defined]

    return wrapper


def patch_vertexai_model(model: Any) -> bool:
    """Patch a Vertex AI GenerativeModel instance.

    Thread-safe. Each model's original methods are stored separately.

    Args:
        model: GenerativeModel instance.

    Returns:
        True if patching succeeded, False otherwise.
    """
    try:
        # 1. Check version compatibility
        from vetch.compat import get_vertexai_version
        v_info = get_vertexai_version()
        if v_info.installed and not v_info.tested:
            logger.warning(
                f"Vertex AI SDK version {v_info.version} is not tested with Vetch. "
                "Patching may be unstable. Set VETCH_FORCE_PATCH=true to override."
            )
            import os
            if os.environ.get("VETCH_FORCE_PATCH") != "true":
                return False

        # Thread-safe: check and patch atomically
        with _model_lock:
            # Check if already patched
            generate = getattr(model, "generate_content", None)
            generate_async = getattr(model, "generate_content_async", None)

            if generate and is_vetch_patched(generate):
                return True  # Already patched

            # Store unbound functions to avoid circular reference
            # Bound methods hold a reference to the object, preventing garbage collection
            generate_unbound = None
            if generate:
                generate_unbound = (
                    generate.__func__ if hasattr(generate, "__func__") else generate
                )

            generate_async_unbound = None
            if generate_async:
                generate_async_unbound = (
                    generate_async.__func__
                    if hasattr(generate_async, "__func__")
                    else generate_async
                )

            _model_originals[model] = _ModelOriginals(
                generate=generate_unbound,
                generate_async=generate_async_unbound,
            )

            # Patch sync method using weak reference wrapper to avoid GC cycles
            if generate:
                sync_wrapper = _WeakGenerateWrapper(model, _model_originals)
                sync_wrapper.vetch_patched = True  # type: ignore[attr-defined]
                sync_wrapper._vetch_original = generate_unbound  # type: ignore[attr-defined]
                model.generate_content = sync_wrapper

            # Patch async method if it exists using weak reference wrapper to avoid GC cycles
            if generate_async:
                async_wrapper = _WeakGenerateAsyncWrapper(model, _model_originals)
                async_wrapper.vetch_patched = True  # type: ignore[attr-defined]
                async_wrapper._vetch_original = generate_async_unbound  # type: ignore[attr-defined]
                model.generate_content_async = async_wrapper

        logger.debug("Vertex AI model patched successfully")
        return True

    except Exception as e:
        logger.warning(f"Failed to patch Vertex AI model: {e}")
        return False


def unpatch_vertexai_model(model: Any) -> bool:
    """Remove Vetch patch from a Vertex AI model.

    Thread-safe. Restores original methods for this specific model.

    Args:
        model: GenerativeModel instance.

    Returns:
        True if unpatching succeeded, False otherwise.
    """
    try:
        with _model_lock:
            originals = _model_originals.pop(model, None)
            if originals is None:
                return True  # Not patched by us

            # Restore originals (may be unbound functions or direct methods)
            if originals.generate:
                if hasattr(originals.generate, "__get__"):
                    # Unbound function - bind it back to the model
                    model.generate_content = originals.generate.__get__(model, type(model))
                else:
                    model.generate_content = originals.generate

            if originals.generate_async:
                if hasattr(originals.generate_async, "__get__"):
                    # Unbound function - bind it back to the model
                    model.generate_content_async = originals.generate_async.__get__(
                        model, type(model)
                    )
                else:
                    model.generate_content_async = originals.generate_async

        logger.debug("Vertex AI model unpatched successfully")
        return True

    except Exception as e:
        logger.warning(f"Failed to unpatch Vertex AI model: {e}")
        return False


def detect_vertexai_model() -> Any | None:
    """Detect if Vertex AI SDK is available.

    Note: Unlike OpenAI, Vertex AI doesn't have a global client.
    Users create model instances explicitly. This function just
    checks if the SDK is importable.

    Returns:
        None (Vertex AI requires explicit model creation).
    """
    import sys
    # Check for common Vertex AI entry points in sys.modules
    if not any(m in sys.modules for m in ["google.cloud.aiplatform", "vertexai"]):
        return None

    try:
        import google.generativeai  # noqa: F401

        return None  # SDK available but no default model
    except ImportError:
        return None


# Track if module is instrumented
_module_instrumented = False

# Store original __init__ for uninstrumentation
_original_generative_model_init: Any | None = None


def instrument_vertexai_module() -> bool:
    """Instrument the Vertex AI module to auto-track all model instances.

    Patches GenerativeModel.__init__ to automatically call patch_vertexai_client
    on every new model instance.

    Returns:
        True if instrumentation succeeded, False otherwise.
    """
    global _module_instrumented, _original_generative_model_init
    import sys

    if _module_instrumented:
        return True

    # Check for Vertex AI modules
    if not any(m in sys.modules for m in ["google.cloud.aiplatform", "vertexai"]):
        return False

    try:
        from vertexai.generative_models import GenerativeModel  # type: ignore[import-not-found]

        # Store original __init__ for later restoration
        _original_generative_model_init = GenerativeModel.__init__

        def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
            _original_generative_model_init(self, *args, **kwargs)
            # Auto-patch this model instance
            with contextlib.suppress(Exception):
                patch_vertexai_model(self)

        GenerativeModel.__init__ = patched_init

        _module_instrumented = True
        logger.debug("Vertex AI module instrumented")
        return True

    except Exception as e:
        logger.debug(f"Failed to instrument Vertex AI module: {e}")
        return False


def uninstrument_vertexai_module() -> bool:
    """Remove Vetch instrumentation from Vertex AI module.

    Restores the original __init__ method and clears tracking state.

    Returns:
        True if uninstrumentation succeeded, False otherwise.
    """
    global _module_instrumented, _original_generative_model_init
    import sys

    if not _module_instrumented:
        return True

    # Check for Vertex AI modules
    if not any(m in sys.modules for m in ["google.cloud.aiplatform", "vertexai"]):
        _module_instrumented = False
        return True

    try:
        from vertexai.generative_models import GenerativeModel

        # Atomic: restore per-model methods first, then __init__
        with _model_lock:
            for model, originals in list(_model_originals.items()):
                try:
                    if originals.generate:
                        model.generate_content = originals.generate
                    if originals.generate_async:
                        model.generate_content_async = originals.generate_async
                except Exception:
                    pass  # Model may have been garbage collected
            _model_originals.clear()

        if _original_generative_model_init is not None:
            GenerativeModel.__init__ = _original_generative_model_init

        _module_instrumented = False
        _original_generative_model_init = None
        logger.debug("Vertex AI module uninstrumented")
        return True

    except Exception as e:
        logger.debug(f"Failed to uninstrument Vertex AI module: {e}")
        return False
