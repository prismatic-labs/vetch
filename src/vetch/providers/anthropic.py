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

from vetch._stall import apply_stall_action, looks_like_param_mismatch
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

    __slots__ = ("_messages_ref", "_originals_dict", "vetch_patched", "_vetch_original")

    def __init__(self, messages: Any, originals_dict: WeakKeyDictionary[Any, Any]) -> None:
        self._messages_ref = weakref.ref(messages)
        self._originals_dict = originals_dict

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        messages = self._messages_ref()
        if messages is None:
            raise RuntimeError("Messages object was garbage collected")

        original = self._originals_dict[messages]

        # original is stored as create.__func__ (unbound) to avoid GC cycles.
        # Pass the messages instance explicitly when original has no __self__.
        bound_args = args if hasattr(original, "__self__") else (messages, *args)

        from vetch.capabilities import stage_request_tools

        stage_request_tools("anthropic", kwargs)
        ctx = get_active_context()

        # v0.4.0: Stall circuit breaker.
        rerouted, original_model = apply_stall_action(kwargs, ctx)
        if rerouted and original_model and ctx is not None:
            ctx.attribution_model = original_model

        is_stream = kwargs.get("stream", False)
        thinking_param = kwargs.get("thinking", {})
        is_thinking = (
            isinstance(thinking_param, dict) and thinking_param.get("type") == "enabled"
        )

        try:
            result = original(*bound_args, **kwargs)

            if is_stream:
                model_hint = kwargs.get("model", "unknown")
                return StreamWrapper(result, model_hint=model_hint, is_thinking=is_thinking)

            _after_create(result, *bound_args, **kwargs)
            return result

        except Exception as e:
            # Fail-open reroute: retry with original model on param mismatch.
            if rerouted and original_model and looks_like_param_mismatch(e):
                ctx = get_active_context()
                if ctx is not None:
                    ctx.warnings.append(
                        f"STALL-001 reroute failed ({type(e).__name__}); "
                        f"falling back to original model {original_model}"
                    )
                kwargs["model"] = original_model
                try:
                    result = original(*bound_args, **kwargs)
                    if is_stream:
                        model_hint = kwargs.get("model", "unknown")
                        return StreamWrapper(
                            result, model_hint=model_hint, is_thinking=is_thinking
                        )
                    _after_create(result, *bound_args, **kwargs)
                    return result
                except Exception as fallback_err:
                    _on_create_error(fallback_err)
                    raise
            _on_create_error(e)
            raise


class _WeakAsyncMessagesWrapper:
    """Async wrapper for messages.create with weak reference."""

    __slots__ = ("_messages_ref", "_originals_dict", "vetch_patched", "_vetch_original")

    def __init__(self, messages: Any, originals_dict: WeakKeyDictionary[Any, Any]) -> None:
        self._messages_ref = weakref.ref(messages)
        self._originals_dict = originals_dict

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        messages = self._messages_ref()
        if messages is None:
            raise RuntimeError("Messages object was garbage collected")

        original = self._originals_dict[messages]
        bound_args = args if hasattr(original, "__self__") else (messages, *args)

        from vetch.capabilities import stage_request_tools

        stage_request_tools("anthropic", kwargs)
        ctx = get_active_context()

        # v0.4.0: Stall circuit breaker.
        rerouted, original_model = apply_stall_action(kwargs, ctx)
        if rerouted and original_model and ctx is not None:
            ctx.attribution_model = original_model

        is_stream = kwargs.get("stream", False)
        thinking_param = kwargs.get("thinking", {})
        is_thinking = (
            isinstance(thinking_param, dict) and thinking_param.get("type") == "enabled"
        )

        try:
            result = await original(*bound_args, **kwargs)

            if is_stream:
                model_hint = kwargs.get("model", "unknown")
                return AsyncStreamWrapper(result, model_hint=model_hint, is_thinking=is_thinking)

            _after_create(result, *args, **kwargs)
            return result

        except Exception as e:
            # Fail-open reroute: retry with original model on param mismatch.
            if rerouted and original_model and looks_like_param_mismatch(e):
                ctx = get_active_context()
                if ctx is not None:
                    ctx.warnings.append(
                        f"STALL-001 reroute failed ({type(e).__name__}); "
                        f"falling back to original model {original_model}"
                    )
                kwargs["model"] = original_model
                try:
                    result = await original(*bound_args, **kwargs)
                    if is_stream:
                        model_hint = kwargs.get("model", "unknown")
                        return AsyncStreamWrapper(
                            result, model_hint=model_hint, is_thinking=is_thinking
                        )
                    _after_create(result, *bound_args, **kwargs)
                    return result
                except Exception as fallback_err:
                    _on_create_error(fallback_err)
                    raise
            _on_create_error(e)
            raise


def _extract_visible_chars(response: Any) -> int | None:
    """Count visible text chars from an Anthropic response, skipping thinking blocks.

    Counts non-whitespace characters so whitespace-only responses are not
    mistaken for meaningful output (EMPTY-001 bypass via e.g. bare newlines).
    """
    content = getattr(response, "content", None)
    if not isinstance(content, list):
        return None
    total = 0
    for block in content:
        if getattr(block, "type", "") == "text":
            text = getattr(block, "text", "") or ""
            total += sum(1 for char in text if not char.isspace())
    return total


def _extract_stop_reason(response: Any) -> str | None:
    """Extract stop_reason from an Anthropic response and return as finish_reason."""
    value = getattr(response, "stop_reason", None)
    return value if isinstance(value, str) else None


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


def _extract_cache_creation_1h(usage: Any) -> int | None:
    """Extract 1-hour-TTL cache-write tokens from an Anthropic usage object.

    Anthropic reports the per-TTL breakdown under ``usage.cache_creation`` as
    ``ephemeral_5m_input_tokens`` / ``ephemeral_1h_input_tokens`` (extended cache
    TTL). The top-level ``cache_creation_input_tokens`` is their sum. We only need
    the 1-hour portion here; the remainder is priced at the 5-minute premium.

    Returns None when the breakdown is absent (older SDKs / 5-minute-only usage),
    which keeps cost pricing on the legacy 5-minute assumption.
    """
    if usage is None:
        return None
    breakdown = getattr(usage, "cache_creation", None)
    if breakdown is None:
        return None
    value = getattr(breakdown, "ephemeral_1h_input_tokens", None)
    return value if isinstance(value, int) else None


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
    from vetch.wrapper import auto_context_for_instrumented_call

    # Check if this is a streaming response
    is_stream = kwargs.get("stream", False)

    if is_stream:
        # For streams, we can't auto-wrap here
        # Stream wrapper handles context creation
        return

    # Auto-create context if needed, or use existing manual wrap() context
    with auto_context_for_instrumented_call("anthropic"):
        # Non-streaming: capture immediately
        usage, cache_read, cache_create = extract_usage(result)
        cache_create_1h = _extract_cache_creation_1h(getattr(result, "usage", None))
        model = extract_model(result)

        # Extended Thinking auto-detection for non-streaming
        thinking_param = kwargs.get("thinking", {})
        if isinstance(thinking_param, dict) and thinking_param.get("type") == "enabled":
            model = model + "-thinking"

        # Extract visible chars (text blocks only; thinking blocks excluded).
        # Required for EMPTY-001 to fire on thinking-mode calls where output_tokens
        # includes hidden reasoning tokens but visible text may be near zero.
        visible_chars = _extract_visible_chars(result)
        finish_reason = _extract_stop_reason(result)

        ctx = get_active_context()
        if ctx is not None:
            if ctx.attribution_model:
                model = ctx.attribution_model

            from vetch.capabilities import (
                extract_anthropic_tools_invoked,
                merge_capability_capture,
            )

            invoked, tool_count = extract_anthropic_tools_invoked(result)
            cap_kwargs = merge_capability_capture(
                tools_invoked=invoked,
                tool_call_count=tool_count,
            )
            ctx.capture(
                model=model,
                provider="anthropic",
                usage=usage,
                is_stream=False,
                complete=True,
                cache_read_tokens=cache_read,
                cache_creation_tokens=cache_create,
                cache_creation_1h_tokens=cache_create_1h,
                visible_output_chars=visible_chars,
                finish_reason=finish_reason,
                **cap_kwargs,
            )


def _on_create_error(error: BaseException) -> None:
    """Hook called when messages.create fails."""
    from vetch.wrapper import auto_context_for_instrumented_call

    # Auto-create context if needed, or use existing manual wrap() context
    with auto_context_for_instrumented_call("anthropic"):
        ctx = get_active_context()
        if ctx is not None:
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

    def __init__(
        self, stream: Any, model_hint: str = "unknown", is_thinking: bool = False
    ) -> None:
        self._stream = stream
        self._accumulated_chars = 0
        # Non-whitespace count. This matches _extract_visible_chars without
        # storing streamed output content.
        self._visible_chars = 0
        self._model = "unknown"
        self._model_hint = model_hint
        self._is_thinking = is_thinking
        self._input_tokens = 0
        self._output_tokens = 0
        self._cache_read_tokens: int | None = None
        self._cache_creation_tokens: int | None = None
        self._cache_creation_1h_tokens: int | None = None
        self._complete = False
        self._error = False
        self._error_type: str | None = None
        self._stop_reason: str | None = None
        self._captured = False
        self._stream_tool_names: list[str] = []

        # Tier 1: tiktoken buffered counting (~100-char buffer reduces encode() call frequency)
        from vetch.calculation import _get_tiktoken_encoding

        self._tiktoken_enc = _get_tiktoken_encoding(model_hint)
        self._tik_token_count = 0
        self._tik_buffer = ""  # flushed every ~100 chars

        # Tier 2: script-aware char counting (only first _SCRIPT_SAMPLE_LIMIT chars sampled)
        self._hiragana_katakana_chars = 0  # \u3040-\u30ff
        self._cjk_ideograph_chars = 0  # \u4e00-\u9fff
        self._hangul_chars = 0  # \uac00-\ud7a3
        self._script_sample_chars = 0  # chars seen during sampling window

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

        from vetch.capabilities import accumulate_anthropic_stream_tool_use

        accumulate_anthropic_stream_tool_use(self._stream_tool_names, chunk)

        if event_type == "message_start":
            msg = getattr(chunk, "message", None)
            if msg:
                self._model = getattr(msg, "model", "unknown")
                if self._is_thinking:
                    self._model = self._model + "-thinking"
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
                    cache_create_1h = _extract_cache_creation_1h(usage)
                    if cache_create_1h is not None:
                        self._cache_creation_1h_tokens = cache_create_1h

        elif event_type == "content_block_delta":
            delta = getattr(chunk, "delta", None)
            if delta:
                text = getattr(delta, "text", "")
                if text:
                    self._accumulated_chars += len(text)
                    self._visible_chars += sum(1 for c in text if not c.isspace())
                    if self._tiktoken_enc is not None:
                        # Tier 1: buffer chunks; encode when buffer reaches ~100 chars
                        self._tik_buffer += text
                        if len(self._tik_buffer) >= 100:
                            self._tik_token_count += len(
                                self._tiktoken_enc.encode(self._tik_buffer)
                            )
                            self._tik_buffer = ""
                    else:
                        # Tier 2: sample only the first _SCRIPT_SAMPLE_LIMIT chars
                        from vetch.calculation import _SCRIPT_SAMPLE_LIMIT

                        if self._script_sample_chars < _SCRIPT_SAMPLE_LIMIT:
                            for ch in text:
                                cp = ord(ch)
                                if 0x3040 <= cp <= 0x30FF:
                                    self._hiragana_katakana_chars += 1
                                elif 0x4E00 <= cp <= 0x9FFF:
                                    self._cjk_ideograph_chars += 1
                                elif 0xAC00 <= cp <= 0xD7A3:
                                    self._hangul_chars += 1
                            self._script_sample_chars += len(text)

        elif event_type == "message_delta":
            delta = getattr(chunk, "delta", None)
            if delta:
                stop_reason = getattr(delta, "stop_reason", None)
                if isinstance(stop_reason, str):
                    self._stop_reason = stop_reason
            usage = getattr(chunk, "usage", None)
            if usage:
                self._output_tokens += getattr(usage, "output_tokens", 0)

    def _capture_to_context(self) -> None:
        if self._captured:
            return
        self._captured = True

        from vetch.calculation import _detect_content_type_hint
        from vetch.wrapper import auto_context_for_instrumented_call

        # Flush any remaining tiktoken buffer
        if self._tiktoken_enc is not None and self._tik_buffer:
            self._tik_token_count += len(self._tiktoken_enc.encode(self._tik_buffer))
            self._tik_buffer = ""

        final_usage = {
            "text": {
                "input_tokens": self._input_tokens,
                "output_tokens": self._output_tokens,
                "total_tokens": self._input_tokens + self._output_tokens,
            }
        }

        content_type_hint = (
            "en"
            if self._tiktoken_enc is not None
            else _detect_content_type_hint(
                self._hiragana_katakana_chars,
                self._cjk_ideograph_chars,
                self._hangul_chars,
                self._script_sample_chars,
            )
        )

        ctx = get_active_context()

        from vetch.capabilities import finalize_anthropic_stream_tools, merge_capability_capture

        invoked, tool_count = finalize_anthropic_stream_tools(
            self._stream_tool_names,
            complete=self._complete,
            error=self._error,
        )
        cap_kwargs = merge_capability_capture(
            tools_invoked=invoked,
            tool_call_count=tool_count,
        )
        model = self._model
        if ctx is not None and ctx.attribution_model:
            model = ctx.attribution_model

        if ctx is not None:
            # Manual wrap() is active — capture to it; it emits on exit
            ctx.capture(
                model=model,
                provider="anthropic",
                usage=final_usage,  # type: ignore[arg-type]
                is_stream=True,
                accumulated_chars=self._accumulated_chars,
                visible_output_chars=self._visible_chars,
                complete=self._complete,
                error=self._error,
                error_type=self._error_type,
                cache_read_tokens=self._cache_read_tokens,
                cache_creation_tokens=self._cache_creation_tokens,
                cache_creation_1h_tokens=self._cache_creation_1h_tokens,
                accumulated_tik_tokens=self._tik_token_count,
                content_type_hint=content_type_hint,
                finish_reason=self._stop_reason,
                **cap_kwargs,
            )
            return

        # Instrumented mode (no manual wrap()) — create auto-context at stream completion
        with auto_context_for_instrumented_call("anthropic"):
            ctx = get_active_context()
            if ctx is not None:
                model = self._model
                if ctx.attribution_model:
                    model = ctx.attribution_model
                ctx.capture(
                    model=model,
                    provider="anthropic",
                    usage=final_usage,  # type: ignore[arg-type]
                    is_stream=True,
                    accumulated_chars=self._accumulated_chars,
                    visible_output_chars=self._visible_chars,
                    complete=self._complete,
                    error=self._error,
                    error_type=self._error_type,
                    cache_read_tokens=self._cache_read_tokens,
                    cache_creation_tokens=self._cache_creation_tokens,
                    cache_creation_1h_tokens=self._cache_creation_1h_tokens,
                    accumulated_tik_tokens=self._tik_token_count,
                    content_type_hint=content_type_hint,
                    finish_reason=self._stop_reason,
                    **cap_kwargs,
                )
        # auto-context exits here → event emitted

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
            thinking_param = kwargs.get("thinking", {})
            is_thinking = (
                isinstance(thinking_param, dict) and thinking_param.get("type") == "enabled"
            )
            try:
                result = await original(*args, **kwargs)
                if is_stream:
                    model_hint = kwargs.get("model", "unknown")
                    return AsyncStreamWrapper(
                        result, model_hint=model_hint, is_thinking=is_thinking
                    )
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
        thinking_param = kwargs.get("thinking", {})
        is_thinking = (
            isinstance(thinking_param, dict) and thinking_param.get("type") == "enabled"
        )
        try:
            result = original(*args, **kwargs)
            if is_stream:
                model_hint = kwargs.get("model", "unknown")
                return StreamWrapper(result, model_hint=model_hint, is_thinking=is_thinking)
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
            wrapper: Any
            if inspect.iscoroutinefunction(create):
                wrapper = _WeakAsyncMessagesWrapper(messages, _client_originals)
            else:
                wrapper = _WeakMessagesWrapper(messages, _client_originals)

            wrapper.vetch_patched = True
            wrapper._vetch_original = _client_originals[messages]
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

        anthropic.Anthropic.__init__ = patched_init  # type: ignore[method-assign]

        # Also patch AsyncAnthropic if available
        if hasattr(anthropic, "AsyncAnthropic"):
            _original_async_anthropic_init = anthropic.AsyncAnthropic.__init__

            def patched_async_init(self: Any, *args: Any, **kwargs: Any) -> None:
                _original_async_anthropic_init(self, *args, **kwargs)
                with contextlib.suppress(Exception):
                    patch_anthropic_client(self)

            anthropic.AsyncAnthropic.__init__ = patched_async_init  # type: ignore[method-assign]

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
            anthropic.Anthropic.__init__ = _original_anthropic_init  # type: ignore[method-assign]

        if _original_async_anthropic_init is not None and hasattr(anthropic, "AsyncAnthropic"):
            anthropic.AsyncAnthropic.__init__ = _original_async_anthropic_init  # type: ignore[method-assign]

        _module_instrumented = False
        _original_anthropic_init = None
        _original_async_anthropic_init = None
        logger.debug("Anthropic module uninstrumented")
        return True

    except Exception as e:
        logger.debug(f"Failed to uninstrument Anthropic module: {e}")
        return False
