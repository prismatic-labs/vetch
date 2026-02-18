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

__version__ = "0.1.1"
__all__ = [
    "wrap",
    "require_tags",
    "add_global_tags",
    "configure_storage",
    "query_usage",
    "__version__",
    "get_session_stats",
    "generate_advisories",
]


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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
