"""Transparent proxy wrapper for SDK methods.

This module provides a proxy pattern that:
- Forwards all attribute access to the original
- Preserves __name__, __doc__, __module__
- Marks wrapped functions with vetch_patched = True
- Detects and respects existing patches (Datadog, OpenTelemetry, etc.)
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from vetch.compat import detect_existing_patches

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class TransparentProxy:
    """Proxy that forwards all attribute access to the wrapped object.

    This ensures observability tools (Datadog, Sentry, etc.) continue
    to function after Vetch patching.
    """

    def __init__(self, wrapped: object) -> None:
        """Initialize proxy.

        Args:
            wrapped: The original object to proxy.
        """
        object.__setattr__(self, "_wrapped", wrapped)

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to wrapped object."""
        return getattr(object.__getattribute__(self, "_wrapped"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Forward attribute setting to wrapped object."""
        if name == "_wrapped":
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_wrapped"), name, value)

    def __delattr__(self, name: str) -> None:
        """Forward attribute deletion to wrapped object."""
        delattr(object.__getattribute__(self, "_wrapped"), name)

    def __repr__(self) -> str:
        """Transparent repr."""
        return repr(object.__getattribute__(self, "_wrapped"))

    def __str__(self) -> str:
        """Transparent str."""
        return str(object.__getattribute__(self, "_wrapped"))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Forward calls to wrapped object."""
        wrapped = object.__getattribute__(self, "_wrapped")
        return wrapped(*args, **kwargs)


def create_wrapper(
    original: F,
    before_call: Callable[..., dict[str, Any] | None] | None = None,
    after_call: Callable[..., None] | None = None,
    on_error: Callable[[BaseException], None] | None = None,
) -> F:
    """Create a transparent wrapper around a function.

    The wrapper:
    - Calls before_call before the original (can modify kwargs)
    - Calls the original function
    - Calls after_call with the result
    - Calls on_error if an exception occurs
    - Always re-raises exceptions (fail-open)

    Args:
        original: The function to wrap.
        before_call: Called before original. Returns extra kwargs or None.
        after_call: Called after original with (result, *args, **kwargs).
        on_error: Called on exception with the exception.

    Returns:
        Wrapped function with same signature.
    """
    # Check for existing patches
    existing_patches = detect_existing_patches(original)
    if existing_patches:
        logger.debug(f"Wrapping already-patched function: {existing_patches}")

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Call before hook
        if before_call is not None:
            try:
                extra_kwargs = before_call(*args, **kwargs)
                if extra_kwargs:
                    kwargs.update(extra_kwargs)
            except Exception as e:
                logger.warning(f"Vetch before_call failed: {e}")

        try:
            result = original(*args, **kwargs)

            # Call after hook
            if after_call is not None:
                try:
                    after_call(result, *args, **kwargs)
                except Exception as e:
                    logger.warning(f"Vetch after_call failed: {e}")

            return result
        except Exception as e:
            # Call error hook
            if on_error is not None:
                try:
                    on_error(e)
                except Exception as hook_error:
                    logger.warning(f"Vetch on_error failed: {hook_error}")
            raise

    # Mark as patched by Vetch
    wrapper.vetch_patched = True  # type: ignore[attr-defined]

    # Preserve original for introspection
    wrapper._vetch_original = original  # type: ignore[attr-defined]

    return wrapper  # type: ignore[return-value]


def is_vetch_patched(func: object) -> bool:
    """Check if a function is already patched by Vetch.

    Args:
        func: Function to check.

    Returns:
        True if already patched by Vetch.
    """
    return getattr(func, "vetch_patched", False) is True


def get_original(func: object) -> object:
    """Get the original function from a Vetch-wrapped function.

    Args:
        func: Potentially wrapped function.

    Returns:
        The original unwrapped function, or func if not wrapped.
    """
    return getattr(func, "_vetch_original", func)


def create_stream_wrapper(
    original: F,
    on_chunk: Callable[[Any], None] | None = None,
    on_complete: Callable[[Any], None] | None = None,
    on_error: Callable[[BaseException], None] | None = None,
) -> F:
    """Create a wrapper for streaming responses.

    The wrapper yields chunks from the original iterator while:
    - Calling on_chunk for each chunk (for counting)
    - Calling on_complete when iteration finishes
    - Calling on_error on exceptions
    - Never accumulating chunks in memory

    Args:
        original: The streaming function to wrap.
        on_chunk: Called for each chunk (for counting, not storing).
        on_complete: Called when stream completes.
        on_error: Called on exception.

    Returns:
        Wrapped streaming function.
    """

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Iterable[Any]:
        stream = original(*args, **kwargs)
        final_chunk = None

        try:
            for chunk in stream:
                final_chunk = chunk
                if on_chunk is not None:
                    try:
                        on_chunk(chunk)
                    except Exception as e:
                        logger.warning(f"Vetch on_chunk failed: {e}")
                yield chunk

            # Stream completed successfully
            if on_complete is not None:
                try:
                    on_complete(final_chunk)
                except Exception as e:
                    logger.warning(f"Vetch on_complete failed: {e}")

        except Exception as e:
            if on_error is not None:
                try:
                    on_error(e)
                except Exception as hook_error:
                    logger.warning(f"Vetch on_error failed: {hook_error}")
            raise

    wrapper.vetch_patched = True  # type: ignore[attr-defined]
    wrapper._vetch_original = original  # type: ignore[attr-defined]

    return wrapper  # type: ignore[return-value]
