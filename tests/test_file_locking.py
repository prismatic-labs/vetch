"""Tests for file-based cache with locking.

These tests verify:
- Basic get/set operations
- Atomic writes
- Lock acquisition and timeout
- Error handling
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

from vetch.sensing.cache import (
    CachedIntensity,
    FileCache,
    FileLock,
    get_file_cache,
    reset_file_cache,
)


class TestFileLock:
    """Tests for file locking."""

    def test_lock_acquire_release(self) -> None:
        """Lock can be acquired and released."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = Path(tmpdir) / "test.lock"

            with FileLock(lock_path):
                assert lock_path.exists()

    def test_lock_creates_directory(self) -> None:
        """Lock creates parent directory if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = Path(tmpdir) / "nested" / "dir" / "test.lock"

            with FileLock(lock_path):
                assert lock_path.parent.exists()

    def test_lock_context_manager_cleanup(self) -> None:
        """Lock releases on exception."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = Path(tmpdir) / "test.lock"

            try:
                with FileLock(lock_path):
                    raise ValueError("Test error")
            except ValueError:
                pass

            # Should be able to acquire lock again
            with FileLock(lock_path):
                pass


class TestFileCacheBasic:
    """Tests for basic file cache operations."""

    def test_set_and_get(self) -> None:
        """Set and get a cached entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FileCache(Path(tmpdir) / "cache.json")

            entry = CachedIntensity(
                intensity_gco2e_kwh=380.0,
                timestamp=time.time(),
                signal_quality="live",
            )
            cache.set("us-east-1", entry)

            result = cache.get("us-east-1")

            assert result is not None
            assert result.intensity_gco2e_kwh == 380.0
            assert result.signal_quality == "live"

    def test_get_missing_region(self) -> None:
        """Get returns None for missing region."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FileCache(Path(tmpdir) / "cache.json")

            result = cache.get("nonexistent")

            assert result is None

    def test_get_missing_file(self) -> None:
        """Get returns None when cache file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FileCache(Path(tmpdir) / "nonexistent.json")

            result = cache.get("us-east-1")

            assert result is None

    def test_multiple_regions(self) -> None:
        """Cache supports multiple regions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FileCache(Path(tmpdir) / "cache.json")

            cache.set(
                "us-east-1",
                CachedIntensity(380.0, time.time(), "live"),
            )
            cache.set(
                "eu-west-1",
                CachedIntensity(340.0, time.time(), "delayed"),
            )

            assert cache.get("us-east-1") is not None
            assert cache.get("us-east-1").intensity_gco2e_kwh == 380.0  # type: ignore
            assert cache.get("eu-west-1") is not None
            assert cache.get("eu-west-1").intensity_gco2e_kwh == 340.0  # type: ignore

    def test_overwrite_region(self) -> None:
        """Overwriting a region updates the value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FileCache(Path(tmpdir) / "cache.json")

            cache.set(
                "us-east-1",
                CachedIntensity(380.0, time.time(), "live"),
            )
            cache.set(
                "us-east-1",
                CachedIntensity(400.0, time.time(), "delayed"),
            )

            result = cache.get("us-east-1")
            assert result is not None
            assert result.intensity_gco2e_kwh == 400.0


class TestFileCacheAtomicWrite:
    """Tests for atomic write behavior."""

    def test_atomic_write_creates_file(self) -> None:
        """Atomic write creates the cache file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.json"
            cache = FileCache(cache_path)

            cache.set(
                "us-east-1",
                CachedIntensity(380.0, time.time(), "live"),
            )

            assert cache_path.exists()
            data = json.loads(cache_path.read_text())
            assert "us-east-1" in data

    def test_atomic_write_creates_directory(self) -> None:
        """Atomic write creates parent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "nested" / "cache.json"
            cache = FileCache(cache_path)

            cache.set(
                "us-east-1",
                CachedIntensity(380.0, time.time(), "live"),
            )

            assert cache_path.exists()


class TestFileCacheAge:
    """Tests for cache age tracking."""

    def test_get_age(self) -> None:
        """Get age of cached entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FileCache(Path(tmpdir) / "cache.json")

            timestamp = time.time() - 60  # 60 seconds ago
            cache.set(
                "us-east-1",
                CachedIntensity(380.0, timestamp, "delayed"),
            )

            age = cache.get_age("us-east-1")

            assert age is not None
            assert age >= 60

    def test_get_age_missing(self) -> None:
        """Get age returns None for missing region."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FileCache(Path(tmpdir) / "cache.json")

            age = cache.get_age("nonexistent")

            assert age is None


class TestFileCacheOperations:
    """Tests for cache operations."""

    def test_clear(self) -> None:
        """Clear removes cache file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.json"
            cache = FileCache(cache_path)

            cache.set(
                "us-east-1",
                CachedIntensity(380.0, time.time(), "live"),
            )
            cache.clear()

            assert not cache_path.exists()

    def test_clear_missing_file(self) -> None:
        """Clear succeeds when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FileCache(Path(tmpdir) / "nonexistent.json")

            result = cache.clear()

            assert result is True

    def test_list_regions(self) -> None:
        """List regions returns all cached regions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FileCache(Path(tmpdir) / "cache.json")

            cache.set(
                "us-east-1",
                CachedIntensity(380.0, time.time(), "live"),
            )
            cache.set(
                "eu-west-1",
                CachedIntensity(340.0, time.time(), "live"),
            )

            regions = cache.list_regions()

            assert set(regions) == {"us-east-1", "eu-west-1"}

    def test_list_regions_empty(self) -> None:
        """List regions returns empty list for new cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FileCache(Path(tmpdir) / "cache.json")

            regions = cache.list_regions()

            assert regions == []


class TestFileCacheEnvVar:
    """Tests for environment variable configuration."""

    def test_cache_path_from_env(self) -> None:
        """Cache path can be set via environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / "env_cache.json"
            original = os.environ.get("VETCH_CACHE_PATH")

            try:
                os.environ["VETCH_CACHE_PATH"] = str(env_path)
                reset_file_cache()
                cache = get_file_cache()

                assert cache.path == env_path
            finally:
                if original is not None:
                    os.environ["VETCH_CACHE_PATH"] = original
                else:
                    os.environ.pop("VETCH_CACHE_PATH", None)
                reset_file_cache()


class TestFileCacheErrorHandling:
    """Tests for error handling."""

    def test_get_corrupted_json(self) -> None:
        """Get returns None for corrupted JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.json"
            cache_path.write_text("not valid json {{{")

            cache = FileCache(cache_path)
            result = cache.get("us-east-1")

            assert result is None

    def test_set_recovers_from_corrupted_json(self) -> None:
        """Set overwrites corrupted JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.json"
            cache_path.write_text("not valid json {{{")

            cache = FileCache(cache_path)
            cache.set(
                "us-east-1",
                CachedIntensity(380.0, time.time(), "live"),
            )

            result = cache.get("us-east-1")
            assert result is not None
            assert result.intensity_gco2e_kwh == 380.0


class TestGlobalFileCache:
    """Tests for global file cache singleton."""

    def test_get_file_cache_singleton(self) -> None:
        """get_file_cache returns the same instance."""
        reset_file_cache()

        cache1 = get_file_cache()
        cache2 = get_file_cache()

        assert cache1 is cache2

    def test_reset_file_cache(self) -> None:
        """reset_file_cache creates a new instance."""
        reset_file_cache()
        cache1 = get_file_cache()

        reset_file_cache()
        cache2 = get_file_cache()

        assert cache1 is not cache2
