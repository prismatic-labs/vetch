"""Ollama Python SDK instrumentation for Vetch.

Captures model, token usage, and image counts without reading prompt content.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from vetch.context import get_active_context
from vetch.schema import ImageUsage, TextUsage, Usage
from vetch.wrapper import auto_context_for_instrumented_call

logger = logging.getLogger(__name__)

_module_instrumented = False
_module_lock = threading.Lock()
_original_generate: Any | None = None
_original_chat: Any | None = None


def _count_images(kwargs: dict[str, Any]) -> int:
    images = kwargs.get("images")
    if images is None:
        image = kwargs.get("image")
        if image is not None:
            return 1
        return 0
    if isinstance(images, list):
        return len(images)
    return 1


def _usage_from_response(response: Any, n_images: int) -> Usage:
    prompt_eval = int(getattr(response, "prompt_eval_count", 0) or 0)
    eval_count = int(getattr(response, "eval_count", 0) or 0)
    text: TextUsage = {
        "input_tokens": prompt_eval,
        "output_tokens": eval_count,
        "total_tokens": prompt_eval + eval_count,
    }
    image: ImageUsage | None = None
    if n_images > 0:
        image = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "image_count": n_images,
        }
    result: Usage = {"text": text, "image": image, "audio": None, "video": None, "reasoning": None}
    return result


def _capture_after(
    model: str,
    response: Any,
    kwargs: dict[str, Any],
    *,
    error: bool = False,
) -> None:
    ctx = get_active_context()
    if ctx is None:
        return
    n_images = _count_images(kwargs)
    usage = None if error else _usage_from_response(response, n_images)
    visible_chars = 0
    if not error:
        text = getattr(response, "response", None) or getattr(response, "message", None)
        if isinstance(text, str):
            visible_chars = len(text)
        elif text is not None and hasattr(text, "content"):
            visible_chars = len(str(getattr(text, "content", "")))
    ctx.capture(
        model=model,
        provider="ollama",
        usage=usage,
        accumulated_chars=visible_chars,
        complete=not error,
        error=error,
    )


def _wrap_call(original: Any, method_name: str) -> Any:
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        model = str(kwargs.get("model") or (args[0] if args else "unknown"))
        with auto_context_for_instrumented_call("ollama", model=model):
            start = time.monotonic()
            try:
                response = original(self, *args, **kwargs)
            except Exception as exc:
                active = get_active_context()
                if active is not None:
                    active.capture(
                        model=model,
                        provider="ollama",
                        error=True,
                        error_type=type(exc).__name__,
                        complete=False,
                    )
                raise
            _capture_after(model, response, kwargs)
            _ = start
            return response

    wrapper.__name__ = getattr(original, "__name__", method_name)
    return wrapper


def instrument_ollama_module() -> bool:
    """Patch ollama.Client generate/chat when the package is already imported."""
    global _module_instrumented, _original_generate, _original_chat
    if _module_instrumented:
        return True

    with _module_lock:
        if _module_instrumented:
            return True
        try:
            import ollama
        except ImportError:
            return False

        client_cls = getattr(ollama, "Client", None)
        if client_cls is None:
            return False

        if hasattr(client_cls, "generate") and _original_generate is None:
            _original_generate = client_cls.generate
            client_cls.generate = _wrap_call(_original_generate, "generate")  # type: ignore[method-assign]

        if hasattr(client_cls, "chat") and _original_chat is None:
            _original_chat = client_cls.chat
            client_cls.chat = _wrap_call(_original_chat, "chat")  # type: ignore[method-assign]

        _module_instrumented = True
        logger.debug("Instrumented ollama.Client")
        return True


def uninstrument_ollama_module() -> None:
    """Restore ollama.Client methods after uninstrument()."""
    global _module_instrumented, _original_generate, _original_chat
    with _module_lock:
        if not _module_instrumented:
            return
        try:
            import ollama
        except ImportError:
            return
        client_cls = getattr(ollama, "Client", None)
        if client_cls is None:
            return
        if _original_generate is not None:
            client_cls.generate = _original_generate  # type: ignore[method-assign]
            _original_generate = None
        if _original_chat is not None:
            client_cls.chat = _original_chat  # type: ignore[method-assign]
            _original_chat = None
        _module_instrumented = False
