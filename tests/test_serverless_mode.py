"""Tests for serverless mode.

In serverless mode (VETCH_CACHE_MODE=memory-only), Vetch should:
- Skip file cache entirely
- Only use in-memory cache
- Work correctly even if temp directory is read-only
"""

from __future__ import annotations

import os
from unittest.mock import patch

from vetch.sensing.cache import get_file_cache


class TestServerlessMode:
    """Tests for memory-only cache behavior."""

    def test_memory_only_mode_skips_file(self) -> None:
        """Verify that memory-only mode doesn't write to disk."""
        from vetch.sensing.cache import CachedIntensity, NoOpCache, reset_file_cache

        reset_file_cache()
        try:
            with patch.dict(os.environ, {"VETCH_CACHE_MODE": "memory-only"}):
                cache = get_file_cache()
                assert isinstance(cache, NoOpCache)

                # Should not raise even if it does nothing
                cache.set("us-east-1", CachedIntensity(100, 1234, "live"))
                assert cache.get("us-east-1") is None
        finally:
            reset_file_cache()

    def test_get_file_cache_respects_env(self) -> None:
        """Verify get_file_cache returns appropriate instance."""
        from vetch.sensing.cache import FileCache, NoOpCache, reset_file_cache

        reset_file_cache()
        try:
            with patch.dict(os.environ, {"VETCH_CACHE_MODE": "memory-only"}):
                cache = get_file_cache()
                assert isinstance(cache, NoOpCache)

            reset_file_cache()
            with patch.dict(os.environ, {"VETCH_CACHE_MODE": "file"}):
                cache = get_file_cache()
                assert isinstance(cache, FileCache)
        finally:
            reset_file_cache()
