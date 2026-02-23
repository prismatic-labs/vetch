"""Vetch: Planet-aware observability for LLM inference.

NOTE: We use `from __future__ import annotations` to enable Python 3.10+
style type hints (str | None) on Python 3.9. This defers evaluation.

Vetch wraps LLM API calls to log energy consumption, cost, and carbon
per inference using live grid data. It never reads prompt or completion
content - only usage metadata from the response.

Basic usage::

    from vetch import wrap

    with wrap(region="us-east-1") as ctx:
        response = client.chat.completions.create(...)

    print(f"Energy: {ctx.event['estimated_energy_wh']} Wh")

Emergency kill switch::

    VETCH_DISABLED=true python my_app.py

"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

# P0: Emergency kill switch - check immediately on import
_DISABLED = os.environ.get("VETCH_DISABLED", "").lower() in ("true", "1", "yes")
if _DISABLED:
    print("vetch: disabled via VETCH_DISABLED=true", file=sys.stderr)

if TYPE_CHECKING:
    from vetch.wrapper import VetchContext

__version__ = "0.1.5"
__all__ = [
    "wrap",
    "instrument",
    "require_tags",
    "add_global_tags",
    "configure_storage",
    "query_usage",
    "__version__",
    "get_session_stats",
    "generate_advisories",
    # v0.1.5: Budget alerts
    "set_budget",
    "on_budget_alert",
    "get_budget_status",
    # v0.1.5: OTLP export
    "configure_otlp_export",
    # v0.1.5: Green signal API
    "get_cleanest_region",
    # v0.1.5: Logging control
    "set_log_level",
]

# Track instrumented state
_instrumented = False


def add_global_tags(tags: dict[str, str]) -> None:
    """Set tags that will be automatically added to every inference event.

    Example::

        vetch.add_global_tags({"env": "production", "service": "chat-api"})
    """
    from vetch.config import add_global_tags as _add_global_tags

    _add_global_tags(tags)


def require_tags(tags: list[str]) -> None:
    """Set global mandatory tags for compliance.

    Example::

        vetch.require_tags(["feature_id", "cost_center"])
    """
    from vetch.config import require_tags as _require_tags

    _require_tags(tags)


def instrument(
    region: str | None = None,
    tags: dict[str, str] | None = None,
) -> bool:
    """Auto-instrument all detected LLM SDK clients.

    Call once at application startup to automatically track all LLM calls
    without needing to use the `wrap()` context manager.

    This is ideal for frameworks like LangChain, LlamaIndex, or any code
    where you don't control the client instantiation.

    Args:
        region: Default grid region for carbon calculation.
        tags: Default tags to add to all events.

    Returns:
        True if any clients were instrumented, False otherwise.

    Example::

        import vetch
        import openai

        # Call once at startup
        vetch.instrument(region="us-east-1", tags={"service": "chat-api"})

        # All OpenAI calls are now automatically tracked
        client = openai.OpenAI()
        response = client.chat.completions.create(...)
        # Events emitted automatically!

    Note:
        - Works with OpenAI, Anthropic, and Vertex AI SDKs
        - Safe to call multiple times (idempotent)
        - Set VETCH_DISABLED=true to disable all instrumentation
    """
    global _instrumented

    if _DISABLED:
        return False

    if _instrumented:
        return True

    # Store default config for auto-instrumentation
    if region:
        os.environ.setdefault("VETCH_REGION", region)
    if tags:
        add_global_tags(tags)

    instrumented_any = False

    # Try to instrument OpenAI
    try:
        from vetch.providers.openai import instrument_openai_module

        if instrument_openai_module():
            instrumented_any = True
    except (ImportError, ModuleNotFoundError):
        pass  # SDK not installed
    except Exception as e:
        import logging

        logging.getLogger("vetch").debug(f"Failed to instrument OpenAI: {e}")

    # Try to instrument Anthropic
    try:
        from vetch.providers.anthropic import instrument_anthropic_module

        if instrument_anthropic_module():
            instrumented_any = True
    except (ImportError, ModuleNotFoundError):
        pass  # SDK not installed
    except Exception as e:
        import logging

        logging.getLogger("vetch").debug(f"Failed to instrument Anthropic: {e}")

    # Try to instrument Vertex AI
    try:
        from vetch.providers.vertexai import instrument_vertexai_module

        if instrument_vertexai_module():
            instrumented_any = True
    except (ImportError, ModuleNotFoundError):
        pass  # SDK not installed
    except Exception as e:
        import logging

        logging.getLogger("vetch").debug(f"Failed to instrument Vertex AI: {e}")

    _instrumented = instrumented_any
    return instrumented_any


def set_log_level(level: str | int) -> None:
    """Set Vetch's internal logging verbosity.

    Args:
        level: Logging level (e.g., "DEBUG", "INFO", "WARNING", "ERROR", or int).

    Example::

        import vetch
        vetch.set_log_level("ERROR")  # Silence info/warning messages
    """
    import logging

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.WARNING)
    logging.getLogger("vetch").setLevel(level)


def wrap(
    region: str | None = None,
    tags: dict[str, str] | None = None,
    energy_override: dict[str, object] | None = None,
) -> VetchContext:
    """Context manager for tracking LLM inference energy and carbon.

    Args:
        region: Grid region for carbon calculation. Falls back to VETCH_REGION
            env var, base_url parsing, or 'unknown'.
        tags: Key-value pairs for cost attribution. Avoid high-cardinality
            values (user IDs, request IDs).
        energy_override: User-provided energy values with keys:
            - wh_per_1k_input (float, required)
            - wh_per_1k_output (float, required)
            - tier (int 1-3, optional)
            - source (str, optional)

    Returns:
        VetchContext: Context manager that logs inference events.

    Example::

        with wrap(region="eu-west-1", tags={"team": "ml"}) as ctx:
            response = client.chat.completions.create(...)

        # Access the logged event
        print(ctx.event["estimated_energy_wh"])

    Note:
        Set VETCH_DISABLED=true to completely disable Vetch (emergency kill switch).

    """
    from vetch.wrapper import VetchContext

    # P0: Kill switch - return no-op context when disabled
    if _DISABLED:
        return VetchContext(
            region=region, tags=tags, energy_override=energy_override, _disabled=True
        )

    return VetchContext(region=region, tags=tags, energy_override=energy_override)


# Lazy imports to avoid circular dependencies
def __getattr__(name: str) -> object:
    if name == "VetchContext":
        from vetch.wrapper import VetchContext

        return VetchContext
    if name == "get_session_stats":
        from vetch.stats import get_session_stats

        return get_session_stats
    if name == "generate_advisories":
        from vetch.advisory import generate_advisories

        return generate_advisories
    if name == "configure_storage":
        from vetch.storage import configure_storage

        return configure_storage
    if name == "query_usage":
        from vetch.storage import query_usage

        return query_usage
    # v0.1.5: Budget alerts
    if name == "set_budget":
        from vetch.budget import set_budget

        return set_budget
    if name == "on_budget_alert":
        from vetch.budget import on_budget_alert

        return on_budget_alert
    if name == "get_budget_status":
        from vetch.budget import get_budget_status

        return get_budget_status
    # v0.1.5: OTLP export
    if name == "configure_otlp_export":
        from vetch.otel import configure_otlp_export

        return configure_otlp_export
    # v0.1.5: Green signal API
    if name == "get_cleanest_region":
        from vetch.sensing.grid import get_cleanest_region

        return get_cleanest_region
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
