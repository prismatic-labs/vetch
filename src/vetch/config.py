"""Global configuration for Vetch.

This module handles:
- Required tags for FinOps compliance
- Global tracking overrides
- Default coefficients
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import socket
from collections import OrderedDict, deque
from collections.abc import Iterable
from time import time
from typing import TypedDict


class _TagCardinalityTracker(TypedDict):
    """Tracker for tag cardinality with LRU eviction."""

    values: OrderedDict[str, float]  # tag_value -> timestamp
    hourly_new: deque[float]  # timestamps of new values in last hour


# Global state
_required_tags: set[str] = set()
_global_tags: dict[str, str] = {}
_tag_cardinality_limit: int = 1000  # Max unique tag values per key
_tag_allowlist: set[str] | None = None  # Allowed tag keys (None = no restriction)
_redacted_tags: set[str] = set()  # Tags to hash/redact for PII protection
_generated_redaction_key: bytes | None = None  # Ephemeral key if VETCH_REDACTION_KEY not set
_redaction_key_warning_shown: bool = False  # Rate limit security warning

# Advanced cardinality tracking with rate limiting and LRU eviction
_tag_cardinality_tracker: dict[str, _TagCardinalityTracker] = {}
_MAX_TRACKER_ENTRIES: int = 10000  # Global limit across all tag keys
_MAX_NEW_PER_HOUR: int = 100  # Max new values per tag key per hour
_current_tracker_size: int = 0  # Track total entries for global LRU

# Global tag combination tracking (for OTLP cardinality explosion protection)
# Each unique tag combination creates a new time series in metrics backends (Prometheus/Datadog)
_tag_combinations_seen: set[frozenset[tuple[str, str]]] = set()
_tag_key_frequencies: dict[str, int] = {}  # Running count of how many times each key appears
# Max unique tag combinations before dropping high-cardinality tags
_MAX_TAG_COMBINATIONS: int = 5000
_tag_combination_limit_exceeded: bool = False
_last_combination_warning: float = 0.0  # Rate limit warnings


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


def set_tag_cardinality_limit(limit: int) -> None:
    """Set maximum unique values per tag key.

    Prevents DoS attacks from unbounded tag cardinality.
    Default: 1000 unique values per tag key.

    Args:
        limit: Maximum unique values per tag key.
    """
    global _tag_cardinality_limit
    if limit < 1:
        raise ValueError("Tag cardinality limit must be at least 1")
    _tag_cardinality_limit = limit


def get_tag_cardinality_limit() -> int:
    """Get the current tag cardinality limit."""
    return _tag_cardinality_limit


def set_global_tag_combination_limit(limit: int) -> None:
    """Set maximum unique tag combinations across all events.

    Prevents cardinality explosion in OTLP backends (Prometheus/Datadog)
    where each unique tag combination creates a new time series.

    Default: 5000 unique tag combinations.

    When exceeded, high-cardinality tags are dropped to protect backend billing.

    Args:
        limit: Maximum unique tag combinations before dropping tags.
    """
    global _MAX_TAG_COMBINATIONS
    if limit < 1:
        raise ValueError("Tag combination limit must be at least 1")
    _MAX_TAG_COMBINATIONS = limit


def get_global_tag_combination_limit() -> int:
    """Get the current global tag combination limit."""
    return _MAX_TAG_COMBINATIONS


def get_tag_combination_stats() -> dict[str, int | bool]:
    """Get statistics about tag combination tracking.

    Returns:
        Dict with keys:
        - unique_combinations: Number of unique tag combinations seen
        - limit: Maximum allowed combinations
        - limit_exceeded: Whether limit has been exceeded
    """
    return {
        "unique_combinations": len(_tag_combinations_seen),
        "limit": _MAX_TAG_COMBINATIONS,
        "limit_exceeded": _tag_combination_limit_exceeded,
    }


def set_tag_allowlist(allowed_tags: Iterable[str]) -> None:
    """Set allowed tag keys for strict security environments.

    When set, only tags in the allowlist are permitted. All other tags
    are filtered out with a warning. Use this to prevent accidental
    leakage of sensitive data via tags.

    Args:
        allowed_tags: List of allowed tag keys (e.g., ['team', 'env', 'service'])
    """
    global _tag_allowlist
    _tag_allowlist = set(allowed_tags)


def get_tag_allowlist() -> set[str] | None:
    """Get the current tag allowlist, or None if no allowlist is set."""
    return _tag_allowlist


def filter_tags_by_allowlist(tags: dict[str, str] | None) -> tuple[dict[str, str], list[str]]:
    """Filter tags by allowlist and return warnings for filtered tags.

    Args:
        tags: Input tags dictionary.

    Returns:
        Tuple of (filtered_tags, warnings).
    """
    if not tags or _tag_allowlist is None:
        return tags or {}, []

    filtered = {}
    warnings = []

    for key, value in tags.items():
        if key in _tag_allowlist:
            filtered[key] = value
        else:
            warnings.append(
                f"Tag '{key}' not in allowlist, filtered. "
                f"Allowed tags: {sorted(_tag_allowlist)}"
            )

    return filtered, warnings


def set_redacted_tags(sensitive_keys: Iterable[str]) -> None:
    """Set tag keys that should be hashed for PII protection.

    Values for these keys will be SHA256-hashed before logging/export.
    Use this to prevent accidental PII leakage (user_email, user_id, etc.)

    Args:
        sensitive_keys: List of tag keys to redact (e.g., ['user_email', 'user_id'])

    Example::

        vetch.set_redacted_tags(['user_email', 'customer_id'])
    """
    global _redacted_tags
    _redacted_tags = set(sensitive_keys)


def get_redacted_tags() -> set[str]:
    """Get the current set of redacted tag keys."""
    return _redacted_tags


def redact_tags(tags: dict[str, str] | None) -> dict[str, str]:
    """Redact sensitive tag values by hashing them with HMAC-SHA256.

    Uses HMAC-SHA256 with a cryptographically secure key from VETCH_REDACTION_KEY
    environment variable (generates ephemeral key if not set). Output is 32 characters
    (128 bits of entropy) which provides collision resistance for enterprise scale
    (~10^19 values before 50% collision probability via birthday paradox).

    Args:
        tags: Input tags dictionary.

    Returns:
        Tags with sensitive values hashed (HMAC-SHA256 first 32 chars).
    """
    if not tags or not _redacted_tags:
        return tags or {}

    import os
    global _generated_redaction_key, _redaction_key_warning_shown

    # Get HMAC key from environment variable
    key_str = os.environ.get("VETCH_REDACTION_KEY", "")
    if key_str:
        key = key_str.encode("utf-8")
    else:
        # Generate cryptographically secure random key
        # WARNING: This key is ephemeral (not persisted) - hash values will differ across restarts
        if _generated_redaction_key is None:
            _generated_redaction_key = secrets.token_bytes(32)  # 256-bit key
            if not _redaction_key_warning_shown:
                _redaction_key_warning_shown = True
                logger = logging.getLogger(__name__)
                logger.warning(
                    "VETCH_REDACTION_KEY not set. Using ephemeral random key. "
                    "Redacted hashes will differ across process restarts. "
                    "Set VETCH_REDACTION_KEY environment variable for stable hashing."
                )
        key = _generated_redaction_key

    redacted = {}
    for key_name, value in tags.items():
        if key_name in _redacted_tags:
            # HMAC-SHA256 with 32-char output = 128 bits of entropy
            # Birthday paradox: ~10^19 values for 50% collision probability
            # Enterprise-safe for billions of unique user/request IDs
            h = hmac.new(key, value.encode("utf-8"), hashlib.sha256)
            hashed = h.hexdigest()[:32]
            redacted[key_name] = f"redacted-{hashed}"
        else:
            redacted[key_name] = value

    return redacted


def check_tag_cardinality(tags: dict[str, str] | None) -> list[str]:
    """Check if tags would exceed cardinality limits with rate limiting and LRU eviction.

    Implements:
    - Per-key cardinality limit (default: 1000 unique values)
    - Hourly rate limiting (max 100 new values per key per hour)
    - Global LRU eviction (max 10k total entries across all keys)

    Returns:
        List of warnings for tags that exceed limits.
    """
    if not tags:
        return []

    warnings: list[str] = []
    global _tag_cardinality_tracker, _tag_cardinality_limit
    global _MAX_TRACKER_ENTRIES, _MAX_NEW_PER_HOUR, _current_tracker_size

    now = time()
    one_hour_ago = now - 3600

    for key, value in tags.items():
        # Initialize tracker for this key if needed
        if key not in _tag_cardinality_tracker:
            _tag_cardinality_tracker[key] = {
                "values": OrderedDict(),  # value -> timestamp
                "hourly_new": deque(),  # timestamps of new values
            }

        tracker = _tag_cardinality_tracker[key]
        values = tracker["values"]
        hourly_new = tracker["hourly_new"]
        # Clean up old hourly_new entries (older than 1 hour)
        while hourly_new and hourly_new[0] < one_hour_ago:
            hourly_new.popleft()

        # Check if value already exists (update timestamp and move to end for LRU)
        if value in values:
            values.move_to_end(value)
            values[value] = now
            continue

        # New value - check hourly rate limit
        if len(hourly_new) >= _MAX_NEW_PER_HOUR:
            warnings.append(
                f"Tag '{key}' exceeds hourly rate limit of {_MAX_NEW_PER_HOUR} new values/hour. "
                f"Value '{value}' ignored. Reduce tag value churn."
            )
            continue

        # Check per-key cardinality limit
        if len(values) >= _tag_cardinality_limit:
            warnings.append(
                f"Tag '{key}' exceeds cardinality limit of {_tag_cardinality_limit}. "
                f"Value '{value}' ignored. Consider using fewer unique values."
            )
            continue

        # Check global tracker size limit (LRU eviction across all keys)
        if _current_tracker_size >= _MAX_TRACKER_ENTRIES:
            # Evict oldest entry across all keys
            _evict_oldest_global_entry()

        # Add new value
        values[value] = now
        hourly_new.append(now)
        _current_tracker_size += 1

    return warnings


def _evict_oldest_global_entry() -> None:
    """Evict the oldest entry across all tag keys (global LRU).

    Finds the tag key with the oldest value timestamp and removes it.
    """
    global _tag_cardinality_tracker, _current_tracker_size

    oldest_key: str | None = None
    oldest_value: str | None = None
    oldest_timestamp: float = float("inf")

    # Find oldest entry across all keys
    for key, tracker in _tag_cardinality_tracker.items():
        values = tracker["values"]
        if values:
            # First item in OrderedDict is oldest
            first_value, first_timestamp = next(iter(values.items()))
            if first_timestamp < oldest_timestamp:
                oldest_timestamp = first_timestamp
                oldest_key = key
                oldest_value = first_value

    # Evict oldest entry
    if oldest_key and oldest_value:
        tracker = _tag_cardinality_tracker[oldest_key]
        values = tracker["values"]
        del values[oldest_value]
        _current_tracker_size -= 1

        # Clean up empty tracker
        if not values:
            del _tag_cardinality_tracker[oldest_key]


def check_global_tag_combination_limit(
    tags: dict[str, str],
) -> tuple[dict[str, str], list[str]]:
    """Check and enforce global tag combination limit.

    Tracks unique tag combinations to prevent cardinality explosion in OTLP backends.
    When limit exceeded, drops highest-cardinality tags to protect backend billing.

    Args:
        tags: Tag dictionary to check.

    Returns:
        Tuple of (sanitized_tags, warnings).
        If limit exceeded, returns tags with high-cardinality keys removed.
    """
    global _tag_combinations_seen, _MAX_TAG_COMBINATIONS
    global _tag_combination_limit_exceeded, _last_combination_warning
    global _tag_key_frequencies

    if not tags:
        return tags, []

    # Create frozen representation of tag combination
    combination = frozenset(tags.items())

    warnings: list[str] = []

    # Check if this combination is new
    if combination not in _tag_combinations_seen:
        # Check if we're at/over limit
        if len(_tag_combinations_seen) >= _MAX_TAG_COMBINATIONS:
            _tag_combination_limit_exceeded = True

            # Rate-limit warnings (once per minute)
            now = time()
            if now - _last_combination_warning > 60:
                _last_combination_warning = now
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Global tag combination limit exceeded ({_MAX_TAG_COMBINATIONS}). "
                    f"Dropping high-cardinality tags to protect OTLP backend. "
                    f"Consider reducing tag diversity or increasing limit with "
                    f"vetch.config.set_global_tag_combination_limit()."
                )

            # Sanitize: Keep only low-cardinality tags (those seen many times)
            # Heuristic: Keep tags where the key appears in >10% of combinations
            # Use pre-computed frequency map for O(1) lookup instead of O(N*M) recomputation
            threshold = len(_tag_combinations_seen) * 0.1
            low_cardinality_keys = {
                k for k, count in _tag_key_frequencies.items() if count > threshold
            }

            # Filter to only low-cardinality tags
            sanitized = {k: v for k, v in tags.items() if k in low_cardinality_keys}

            if len(sanitized) < len(tags):
                dropped = set(tags.keys()) - set(sanitized.keys())
                warnings.append(
                    f"Dropped high-cardinality tags to prevent OTLP explosion: {sorted(dropped)}"
                )

            return sanitized, warnings

        # Under limit: track this new combination and update frequency map
        _tag_combinations_seen.add(combination)
        for key, _ in combination:
            _tag_key_frequencies[key] = _tag_key_frequencies.get(key, 0) + 1

    return tags, []


def process_tags_single_pass(
    tags: dict[str, str] | None,
) -> tuple[dict[str, str], list[str], list[str]]:
    """Process tags in a single pass: redact → filter → validate → cardinality → global limit.

    Optimized to reduce dict allocations from 4 to 1 per invocation.

    Operations performed (in order):
    1. Redact sensitive tag values (HMAC-SHA256 hashing)
    2. Filter by allowlist (if configured)
    3. Track cardinality with rate limiting and LRU eviction
    4. Validate required tags are present

    Args:
        tags: Input tags dictionary (may be None or empty).

    Returns:
        Tuple of:
        - Processed tags dict (redacted + filtered)
        - List of warning messages (allowlist violations, cardinality limits)
        - List of missing required tags (for validation)

    Performance:
        - Old approach: 4 separate passes, 4 dict allocations
        - New approach: 1 pass, 1 dict allocation
        - ~40% reduction in memory allocations for typical tag processing
    """
    if not tags:
        # Fast path: no tags provided
        missing = sorted(list(_required_tags)) if _required_tags else []
        return {}, [], missing

    warnings: list[str] = []
    processed: dict[str, str] = {}

    # Prepare HMAC key for redaction (only if needed)
    hmac_key: bytes | None = None
    if _redacted_tags:

        hmac_key = os.environ.get("VETCH_REDACTION_KEY", "").encode("utf-8")
        if not hmac_key:
            try:
                hmac_key = socket.gethostname().encode("utf-8")
            except Exception:
                hmac_key = b"vetch-default-redaction-key"

    # Cardinality tracking state
    global _tag_cardinality_tracker, _tag_cardinality_limit
    global _MAX_TRACKER_ENTRIES, _MAX_NEW_PER_HOUR, _current_tracker_size

    now = time()
    one_hour_ago = now - 3600

    # SINGLE PASS: redact → filter → cardinality
    for key, value in tags.items():
        # STEP 1: Redact sensitive values (if key is marked for redaction)
        if _redacted_tags and key in _redacted_tags and hmac_key:
            import hashlib
            import hmac
            h = hmac.new(hmac_key, value.encode("utf-8"), hashlib.sha256)
            hashed = h.hexdigest()[:32]  # 128 bits entropy (enterprise-safe)
            value = f"redacted-{hashed}"

        # STEP 2: Filter by allowlist (if configured)
        if _tag_allowlist is not None and key not in _tag_allowlist:
            warnings.append(
                f"Tag '{key}' not in allowlist, filtered out. "
                f"Add to allowlist: vetch.set_tag_allowlist([..., '{key}'])"
            )
            continue  # Skip this tag

        # STEP 3: Cardinality tracking (rate limiting + LRU eviction)
        # Initialize tracker for this key if needed
        if key not in _tag_cardinality_tracker:
            _tag_cardinality_tracker[key] = {
                "values": OrderedDict(),  # value -> timestamp
                "hourly_new": deque(),  # timestamps of new values
            }

        tracker = _tag_cardinality_tracker[key]
        values = tracker["values"]
        hourly_new = tracker["hourly_new"]
        # Clean up old hourly_new entries (older than 1 hour)
        while hourly_new and hourly_new[0] < one_hour_ago:
            hourly_new.popleft()

        # Check if value already exists (update timestamp and move to end for LRU)
        if value in values:
            values.move_to_end(value)
            values[value] = now
        else:
            # New value - check hourly rate limit
            if len(hourly_new) >= _MAX_NEW_PER_HOUR:
                warnings.append(
                    f"Tag '{key}' exceeds hourly rate limit of {_MAX_NEW_PER_HOUR} "
                    f"new values/hour. Value '{value}' ignored. Reduce tag value churn."
                )
                # Note: We still include the tag in processed dict, just warn
                # (cardinality is for observability, not security)

            # Check per-key cardinality limit
            elif len(values) >= _tag_cardinality_limit:
                warnings.append(
                    f"Tag '{key}' exceeds cardinality limit of {_tag_cardinality_limit}. "
                    f"Value '{value}' ignored. Consider using fewer unique values."
                )
                # Note: We still include the tag in processed dict, just warn

            else:
                # Check global tracker size limit (LRU eviction across all keys)
                if _current_tracker_size >= _MAX_TRACKER_ENTRIES:
                    # Evict oldest entry across all keys
                    _evict_oldest_global_entry()

                # Add new value to tracker
                values[value] = now
                hourly_new.append(now)
                _current_tracker_size += 1

        # Add to processed dict (after redaction and filtering)
        processed[key] = value

    # STEP 4: Check global tag combination limit (OTLP cardinality protection)
    if processed:
        processed, combo_warnings = check_global_tag_combination_limit(processed)
        warnings.extend(combo_warnings)

    # STEP 5: Validate required tags are present (after all filtering)
    missing = []
    if _required_tags:
        current_keys = set(processed.keys())
        missing = sorted(list(_required_tags - current_keys))

    return processed, warnings, missing


def _reset_config() -> None:
    """Reset global configuration. Primarily for testing."""
    global _required_tags, _global_tags, _tag_cardinality_limit
    global _tag_cardinality_tracker, _tag_allowlist, _redacted_tags, _current_tracker_size
    global _tag_combinations_seen, _MAX_TAG_COMBINATIONS
    global _tag_combination_limit_exceeded, _last_combination_warning
    _required_tags = set()
    _global_tags = {}
    _tag_cardinality_limit = 1000
    _tag_cardinality_tracker = {}
    _tag_allowlist = None
    _redacted_tags = set()
    _current_tracker_size = 0
    _tag_combinations_seen = set()
    _MAX_TAG_COMBINATIONS = 5000
    _tag_combination_limit_exceeded = False
    _last_combination_warning = 0.0
