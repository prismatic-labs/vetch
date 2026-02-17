"""Global configuration for Vetch.

This module handles:
- Required tags for FinOps compliance
- Global tracking overrides
- Default coefficients
"""

from __future__ import annotations

from collections.abc import Iterable

# Global state
_required_tags: set[str] = set()
_global_tags: dict[str, str] = {}


def add_global_tags(tags: dict[str, str]) -> None:
    """Set tags that will be automatically added to every inference event.
    
    Args:
        tags: Key-value pairs (e.g. {'env': 'prod', 'version': '1.2.0'})
    """
    global _global_tags
    _global_tags.update(tags)


def get_global_tags() -> dict[str, str]:
    """Get the currently configured global tags."""
    return _global_tags


def require_tags(tags: Iterable[str]) -> None:
    """Specify tags that MUST be present in every wrap() call.

    If required tags are missing, the inference event will be
    marked with an error and 'tracking_disabled=True'.

    Args:
        tags: List of tag keys (e.g. ['feature_id', 'cost_center'])
    """
    global _required_tags
    _required_tags = set(tags)


def get_required_tags() -> set[str]:
    """Get the set of currently required tags."""
    return _required_tags


def validate_tags(tags: dict[str, str] | None) -> list[str]:
    """Validate that all required tags are present.

    Returns:
        List of missing tag keys.
    """
    if not _required_tags:
        return []

    current_keys = set(tags.keys()) if tags else set()
    missing = _required_tags - current_keys
    return sorted(list(missing))
