"""Tests for in-memory cache.

These tests verify:
- Basic get/set operations
- TTL expiration
- Cache invalidation
- Global cache singleton
"""

from __future__ import annotations

import os
import time

from vetch.sensing.memory_cache import (
    DEFAULT_TTL_SECONDS,
    MemoryCache,
    get_grid_cache,
    reset_grid_cache,
)


class TestMemoryCacheBasic:
    """Tests for basic cache operations."""

    def test_set_and_get(self) -> None:
        """Set and get a value."""
        cache: MemoryCache[float] = MemoryCache()

        cache.set("region-a", 450.0)
        value, is_fresh = cache.get("region-a")

        assert value == 450.0
        assert is_fresh is True

    def test_get_missing_key(self) -> None:
        """Get returns None for missing key."""
        cache: MemoryCache[float] = MemoryCache()

        value, is_fresh = cache.get("nonexistent")

        assert value is None
        assert is_fresh is False

    def test_overwrite_value(self) -> None:
        """Overwriting updates the value."""
        cache: MemoryCache[float] = MemoryCache()

        cache.set("region", 100.0)
        cache.set("region", 200.0)
        value, _ = cache.get("region")

        assert value == 200.0

    def test_multiple_keys(self) -> None:
        """Cache supports multiple independent keys."""
        cache: MemoryCache[float] = MemoryCache()

        cache.set("a", 1.0)
        cache.set("b", 2.0)
        cache.set("c", 3.0)

        assert cache.get("a")[0] == 1.0
        assert cache.get("b")[0] == 2.0
        assert cache.get("c")[0] == 3.0


class TestMemoryCacheTTL:
    """Tests for TTL behavior."""

    def test_fresh_within_ttl(self) -> None:
        """Value is fresh when within TTL."""
        cache: MemoryCache[float] = MemoryCache(ttl_seconds=60)

        cache.set("region", 450.0)
        _, is_fresh = cache.get("region")

        assert is_fresh is True

    def test_stale_after_ttl(self) -> None:
        """Value is stale after TTL expires."""
        cache: MemoryCache[float] = MemoryCache(ttl_seconds=1)

        cache.set("region", 450.0)
        time.sleep(1.1)  # Wait for TTL to expire
        value, is_fresh = cache.get("region")

        assert value == 450.0  # Value still accessible
        assert is_fresh is False  # But marked as stale

    def test_default_ttl(self) -> None:
        """Default TTL is 300 seconds."""
        cache: MemoryCache[float] = MemoryCache()

        assert cache.ttl == DEFAULT_TTL_SECONDS

    def test_custom_ttl(self) -> None:
        """Custom TTL is respected."""
        cache: MemoryCache[float] = MemoryCache(ttl_seconds=120)

        assert cache.ttl == 120

    def test_env_var_ttl(self) -> None:
        """TTL can be set via environment variable."""
        original = os.environ.get("VETCH_MEMORY_CACHE_TTL")
        try:
            os.environ["VETCH_MEMORY_CACHE_TTL"] = "600"
            cache: MemoryCache[float] = MemoryCache()

            assert cache.ttl == 600
        finally:
            if original is not None:
                os.environ["VETCH_MEMORY_CACHE_TTL"] = original
            else:
                os.environ.pop("VETCH_MEMORY_CACHE_TTL", None)

    def test_invalid_env_var_ttl(self) -> None:
        """Invalid env var falls back to default."""
        original = os.environ.get("VETCH_MEMORY_CACHE_TTL")
        try:
            os.environ["VETCH_MEMORY_CACHE_TTL"] = "not-a-number"
            cache: MemoryCache[float] = MemoryCache()

            assert cache.ttl == DEFAULT_TTL_SECONDS
        finally:
            if original is not None:
                os.environ["VETCH_MEMORY_CACHE_TTL"] = original
            else:
                os.environ.pop("VETCH_MEMORY_CACHE_TTL", None)


class TestMemoryCacheAge:
    """Tests for cache age tracking."""

    def test_get_age_exists(self) -> None:
        """Get age of cached entry."""
        cache: MemoryCache[float] = MemoryCache()

        cache.set("region", 450.0)
        time.sleep(0.05)  # Small delay
        age = cache.get_age("region")

        assert age is not None
        assert age >= 0.05

    def test_get_age_missing(self) -> None:
        """Get age returns None for missing key."""
        cache: MemoryCache[float] = MemoryCache()

        age = cache.get_age("nonexistent")

        assert age is None


class TestMemoryCacheInvalidation:
    """Tests for cache invalidation."""

    def test_invalidate_existing(self) -> None:
        """Invalidate removes existing entry."""
        cache: MemoryCache[float] = MemoryCache()

        cache.set("region", 450.0)
        result = cache.invalidate("region")

        assert result is True
        assert cache.get("region")[0] is None

    def test_invalidate_missing(self) -> None:
        """Invalidate returns False for missing key."""
        cache: MemoryCache[float] = MemoryCache()

        result = cache.invalidate("nonexistent")

        assert result is False

    def test_clear(self) -> None:
        """Clear removes all entries."""
        cache: MemoryCache[float] = MemoryCache()

        cache.set("a", 1.0)
        cache.set("b", 2.0)
        cache.clear()

        assert cache.size() == 0
        assert cache.get("a")[0] is None
        assert cache.get("b")[0] is None


class TestMemoryCacheMetadata:
    """Tests for cache metadata operations."""

    def test_size(self) -> None:
        """Size returns number of entries."""
        cache: MemoryCache[float] = MemoryCache()

        assert cache.size() == 0

        cache.set("a", 1.0)
        assert cache.size() == 1

        cache.set("b", 2.0)
        assert cache.size() == 2

    def test_keys(self) -> None:
        """Keys returns all cached keys."""
        cache: MemoryCache[float] = MemoryCache()

        cache.set("us-east-1", 380.0)
        cache.set("eu-west-1", 340.0)

        keys = cache.keys()

        assert set(keys) == {"us-east-1", "eu-west-1"}


class TestGlobalGridCache:
    """Tests for global grid cache singleton."""

    def test_get_grid_cache_singleton(self) -> None:
        """get_grid_cache returns the same instance."""
        reset_grid_cache()

        cache1 = get_grid_cache()
        cache2 = get_grid_cache()

        assert cache1 is cache2

    def test_reset_grid_cache(self) -> None:
        """reset_grid_cache creates a new instance."""
        reset_grid_cache()
        cache1 = get_grid_cache()

        reset_grid_cache()
        cache2 = get_grid_cache()

        assert cache1 is not cache2

    def test_grid_cache_is_float_typed(self) -> None:
        """Grid cache stores float values."""
        reset_grid_cache()
        cache = get_grid_cache()

        cache.set("us-east-1", 380.5)
        value, _ = cache.get("us-east-1")

        assert value == 380.5
