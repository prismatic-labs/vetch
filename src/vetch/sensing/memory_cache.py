"""In-memory cache for grid intensity data.

Provides fast, per-process caching of carbon intensity values
with configurable TTL. This is the first tier in the multi-tier
caching strategy.

Default TTL: 5 minutes (configurable via VETCH_MEMORY_CACHE_TTL)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")

# Default TTL in seconds (5 minutes)
DEFAULT_TTL_SECONDS = 300


@dataclass
class CacheEntry(Generic[T]):
    """A cached value with timestamp."""

    value: T
    timestamp: float


class MemoryCache(Generic[T]):
    """Thread-safe in-memory cache with TTL.

    Values are cached per-key with a configurable time-to-live.
    After TTL expires, the value is considered stale and will
    be refreshed on next access.

    This cache is designed for grid intensity data where:
    - Data changes slowly (5-15 minute updates)
    - Stale data is acceptable (fail-open)
    - Per-process isolation is fine
    """

    def __init__(self, ttl_seconds: int | None = None) -> None:
        """Initialize cache with TTL.

        Args:
            ttl_seconds: Time-to-live in seconds. If None, uses
                VETCH_MEMORY_CACHE_TTL env var or default (300s).
        """
        if ttl_seconds is not None:
            self._ttl = ttl_seconds
        else:
            env_ttl = os.environ.get("VETCH_MEMORY_CACHE_TTL")
            if env_ttl:
                try:
                    self._ttl = int(env_ttl)
                except ValueError:
                    self._ttl = DEFAULT_TTL_SECONDS
            else:
                self._ttl = DEFAULT_TTL_SECONDS

        self._cache: dict[str, CacheEntry[T]] = {}

    @property
    def ttl(self) -> int:
        """Get the TTL in seconds."""
        return self._ttl

    def get(self, key: str) -> tuple[T | None, bool]:
        """Get a cached value.

        Args:
            key: The cache key (typically a region identifier).

        Returns:
            Tuple of (value or None, is_fresh). If value is None,
            there's no cached entry. is_fresh indicates whether
            the value is within TTL.
        """
        entry = self._cache.get(key)
        if entry is None:
            return None, False

        age = time.time() - entry.timestamp
        is_fresh = age < self._ttl

        return entry.value, is_fresh

    def set(self, key: str, value: T) -> None:
        """Set a cached value.

        Args:
            key: The cache key.
            value: The value to cache.
        """
        self._cache[key] = CacheEntry(value=value, timestamp=time.time())

    def get_age(self, key: str) -> float | None:
        """Get the age of a cached entry in seconds.

        Args:
            key: The cache key.

        Returns:
            Age in seconds, or None if not cached.
        """
        entry = self._cache.get(key)
        if entry is None:
            return None

        return time.time() - entry.timestamp

    def invalidate(self, key: str) -> bool:
        """Remove a cached entry.

        Args:
            key: The cache key to remove.

        Returns:
            True if entry was removed, False if it didn't exist.
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    def size(self) -> int:
        """Get the number of cached entries."""
        return len(self._cache)

    def keys(self) -> list[str]:
        """Get all cached keys."""
        return list(self._cache.keys())


# Global instance for grid intensity caching
_grid_intensity_cache: MemoryCache[float] | None = None


def get_grid_cache() -> MemoryCache[float]:
    """Get the global grid intensity cache.

    Creates the cache on first access (lazy initialization).

    Returns:
        The shared MemoryCache instance for grid data.
    """
    global _grid_intensity_cache
    if _grid_intensity_cache is None:
        _grid_intensity_cache = MemoryCache[float]()
    return _grid_intensity_cache


def reset_grid_cache() -> None:
    """Reset the global grid intensity cache.

    Primarily for testing. Clears and reinitializes the cache.
    """
    global _grid_intensity_cache
    _grid_intensity_cache = None
