"""Resolve LangChain's callback handler base without a module-level optional import.

CI type-checks ``src/`` without LangChain installed. Importing
``langchain_core`` at module scope makes ``BaseCallbackHandler`` resolve to
``Any`` under ``ignore_missing_imports``, which strict mypy rejects as a
superclass. This module resolves the real base at runtime via ``importlib``
(so mypy never loads LangChain) and exposes a concrete fallback type for
static analysis.
"""

from __future__ import annotations

import importlib
from typing import cast


class BaseCallbackHandlerFallback:
    """Minimal callback-handler surface when LangChain is not installed."""

    raise_error: bool = False


def resolve_callback_handler_base() -> type[BaseCallbackHandlerFallback]:
    """Return LangChain's ``BaseCallbackHandler``, or a compatible fallback."""
    for module_path, attr in (
        ("langchain_core.callbacks.base", "BaseCallbackHandler"),
        ("langchain.callbacks.base", "BaseCallbackHandler"),
    ):
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            continue
        handler_base = getattr(module, attr)
        return cast(type[BaseCallbackHandlerFallback], handler_base)
    return BaseCallbackHandlerFallback
