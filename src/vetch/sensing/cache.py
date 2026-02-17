"""File-based cache for grid intensity data.

Provides persistent, cross-process caching with file locking.
This is the second tier in the multi-tier caching strategy.

Supports:
- Cross-platform file locking (Unix: fcntl, Windows: msvcrt)
- Atomic writes via temp file + rename
- Lock timeout to prevent deadlocks
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default cache path
DEFAULT_CACHE_DIR = Path(tempfile.gettempdir()) / "vetch"
DEFAULT_CACHE_FILE = "grid_intensity_cache.json"

# Lock timeout in seconds
LOCK_TIMEOUT_SECONDS = 0.1  # 100ms


@dataclass
class CachedIntensity:
    """Cached grid intensity data."""

    intensity_gco2e_kwh: float
    timestamp: float
    signal_quality: str  # "live", "delayed", "blind"


class FileLock:
    """Cross-platform file lock context manager.

    Uses fcntl on Unix and msvcrt on Windows.
    Supports timeout to prevent deadlocks.
    """

    def __init__(self, path: Path, timeout: float = LOCK_TIMEOUT_SECONDS) -> None:
        """Initialize file lock.

        Args:
            path: Path to the lock file.
            timeout: Maximum time to wait for lock in seconds.
        """
        self._path = path
        self._timeout = timeout
        self._fd: int | None = None

    def __enter__(self) -> FileLock:
        """Acquire the lock."""
        # Ensure directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Open file for writing (creates if doesn't exist)
        self._fd = os.open(
            str(self._path),
            os.O_RDWR | os.O_CREAT,
            0o644,
        )

        start_time = time.time()
        while True:
            try:
                self._lock()
                return self
            except BlockingIOError:
                if time.time() - start_time > self._timeout:
                    # Close fd on timeout
                    if self._fd is not None:
                        os.close(self._fd)
                        self._fd = None
                    raise TimeoutError(
                        f"Could not acquire file lock within {self._timeout}s"
                    ) from None
                time.sleep(0.01)  # 10ms retry interval

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Release the lock."""
        if self._fd is not None:
            try:
                self._unlock()
            finally:
                os.close(self._fd)
                self._fd = None

    def _lock(self) -> None:
        """Acquire platform-specific lock."""
        if self._fd is None:
            raise RuntimeError("File not opened")

        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(self._fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(self) -> None:
        """Release platform-specific lock."""
        if self._fd is None:
            return

        if sys.platform == "win32":
            import msvcrt

            with contextlib.suppress(OSError):
                msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self._fd, fcntl.LOCK_UN)


class FileCache:
    """Persistent file-based cache for grid intensity data.

    Features:
    - Atomic writes using temp file + rename
    - Cross-platform file locking
    - JSON serialization
    - Graceful degradation on errors
    """

    def __init__(self, cache_path: Path | None = None) -> None:
        """Initialize file cache.

        Args:
            cache_path: Path to cache file. If None, uses
                VETCH_CACHE_PATH env var or default location.
        """
        if cache_path is not None:
            self._path = cache_path
        else:
            env_path = os.environ.get("VETCH_CACHE_PATH")
            if env_path:
                self._path = Path(env_path)
            else:
                self._path = DEFAULT_CACHE_DIR / DEFAULT_CACHE_FILE

        self._lock_path = self._path.with_suffix(".lock")

    @property
    def path(self) -> Path:
        """Get the cache file path."""
        return self._path

    def get(self, region: str) -> CachedIntensity | None:
        """Get cached intensity for a region.

        Args:
            region: Grid region identifier.

        Returns:
            CachedIntensity if found, None otherwise.
        """
        try:
            with FileLock(self._lock_path):
                if not self._path.exists():
                    return None

                data = json.loads(self._path.read_text())
                entry = data.get(region)
                if entry is None:
                    return None

                return CachedIntensity(
                    intensity_gco2e_kwh=entry["intensity_gco2e_kwh"],
                    timestamp=entry["timestamp"],
                    signal_quality=entry["signal_quality"],
                )
        except TimeoutError:
            logger.debug(f"Cache lock timeout reading {region}")
            return None
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.debug(f"Cache read error for {region}: {e}")
            return None

    def set(self, region: str, intensity: CachedIntensity) -> bool:
        """Set cached intensity for a region.

        Uses atomic write (temp file + rename) for safety.

        Args:
            region: Grid region identifier.
            intensity: The intensity data to cache.

        Returns:
            True if successful, False otherwise.
        """
        try:
            with FileLock(self._lock_path):
                # Read existing data
                data: dict[str, Any] = {}
                if self._path.exists():
                    try:
                        data = json.loads(self._path.read_text())
                    except json.JSONDecodeError:
                        data = {}

                # Update with new entry
                data[region] = {
                    "intensity_gco2e_kwh": intensity.intensity_gco2e_kwh,
                    "timestamp": intensity.timestamp,
                    "signal_quality": intensity.signal_quality,
                }

                # Atomic write
                self._path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = self._path.with_suffix(".tmp")
                temp_path.write_text(json.dumps(data, indent=2))
                os.replace(temp_path, self._path)

                return True

        except TimeoutError:
            logger.debug(f"Cache lock timeout writing {region}")
            return False
        except OSError as e:
            logger.debug(f"Cache write error for {region}: {e}")
            return False

    def get_age(self, region: str) -> float | None:
        """Get the age of a cached entry in seconds.

        Args:
            region: Grid region identifier.

        Returns:
            Age in seconds, or None if not cached.
        """
        entry = self.get(region)
        if entry is None:
            return None
        return time.time() - entry.timestamp

    def clear(self) -> bool:
        """Clear all cached data.

        Returns:
            True if successful, False otherwise.
        """
        try:
            with FileLock(self._lock_path):
                if self._path.exists():
                    self._path.unlink()
                return True
        except (TimeoutError, OSError) as e:
            logger.debug(f"Cache clear error: {e}")
            return False

    def list_regions(self) -> list[str]:
        """List all cached regions.

        Returns:
            List of region identifiers in cache.
        """
        try:
            with FileLock(self._lock_path):
                if not self._path.exists():
                    return []
                data = json.loads(self._path.read_text())
                return list(data.keys())
        except (TimeoutError, json.JSONDecodeError, OSError):
            return []


class NoOpCache:
    """Non-operational cache for serverless mode."""

    def __init__(self, cache_path: Path | None = None) -> None:
        """Initialize NoOp cache.

        Args:
            cache_path: Ignored.
        """
        self._path = Path("/dev/null")

    @property
    def path(self) -> Path:
        """Get the cache file path."""
        return self._path

    def get(self, region: str) -> CachedIntensity | None:
        """Always return None."""
        return None

    def set(self, region: str, intensity: CachedIntensity) -> bool:
        """Always return True (pretend to succeed)."""
        return True

    def get_age(self, region: str) -> float | None:
        """Always return None."""
        return None

    def clear(self) -> bool:
        """Always return True."""
        return True

    def list_regions(self) -> list[str]:
        """Always return empty list."""
        return []


# Global instance
_file_cache: Any = None


def get_file_cache() -> Any:
    """Get the global file cache.

    Respects VETCH_CACHE_MODE env var:
    - 'memory-only': returns NoOpCache
    - anything else (default): returns FileCache

    Returns:
        The shared cache instance.
    """
    global _file_cache
    if _file_cache is None:
        mode = os.environ.get("VETCH_CACHE_MODE")
        if mode == "memory-only":
            _file_cache = NoOpCache()
        else:
            _file_cache = FileCache()
    return _file_cache


def reset_file_cache() -> None:
    """Reset the global file cache.

    Primarily for testing.
    """
    global _file_cache
    _file_cache = None


# Extraordinary Move: Zombie Lock Cleaner
def _cleanup_stale_locks() -> None:
    """Check for and remove stale lock files."""
    try:
        # Only clean the default cache dir
        if not DEFAULT_CACHE_DIR.exists():
            return

        for lock_file in DEFAULT_CACHE_DIR.glob("*.lock"):
            try:
                # If lock file is more than 1 hour old, it's likely a zombie
                if time.time() - lock_file.stat().st_mtime > 3600:
                    lock_file.unlink(missing_ok=True)
            except Exception:
                continue
    except Exception:
        pass


atexit.register(_cleanup_stale_locks)
