"""Remote registry fetcher for dynamic model data updates.

This module provides background fetching of energy and pricing data
from a remote endpoint (CDN/GitHub), merging with bundled defaults.

Key design decisions:
- Fail-silent: network errors log debug, use cached/bundled fallback
- Memory-only cache by default (no file writes for privacy)
- ETag caching to reduce bandwidth
- Decorrelated jitter to avoid thundering herd
- Opt-out via VETCH_REGISTRY_REMOTE=false
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import random
import threading
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# Default remote registry URL (GitHub raw)
DEFAULT_REGISTRY_URL = (
    "https://raw.githubusercontent.com/prismatic-labs/vetch/main/src/vetch/registry"
)

# Default refresh interval
DEFAULT_REFRESH_HOURS = 24

# VETCH_HOME default
DEFAULT_VETCH_HOME = Path.home() / ".vetch"


def get_vetch_home() -> Path:
    """Get the Vetch home directory.

    Respects VETCH_HOME environment variable, defaults to ~/.vetch/.

    Returns:
        Path to Vetch home directory.
    """
    env_home = os.environ.get("VETCH_HOME")
    if env_home:
        return Path(env_home)
    return DEFAULT_VETCH_HOME


class RemoteRegistryFetcher:
    """Fetches and caches remote registry data.

    Thread-safe. Uses background timer for periodic refresh.
    ETag caching minimizes bandwidth usage.
    """

    def __init__(
        self,
        base_url: str | None = None,
        refresh_hours: float | None = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        """Initialize the remote registry fetcher.

        Args:
            base_url: Base URL for registry files. Defaults to GitHub raw URL.
            refresh_hours: Hours between refresh attempts.
            timeout_seconds: HTTP request timeout in seconds.
        """
        self._base_url = base_url or os.environ.get(
            "VETCH_REGISTRY_URL", DEFAULT_REGISTRY_URL
        )
        self._refresh_hours = refresh_hours or float(
            os.environ.get("VETCH_REGISTRY_REFRESH_HOURS", str(DEFAULT_REFRESH_HOURS))
        )
        self._timeout = timeout_seconds
        self._lock = threading.Lock()

        # In-memory cache
        self._energy_cache: dict[str, Any] | None = None
        self._pricing_cache: dict[str, Any] | None = None

        # ETag tracking per file
        self._etags: dict[str, str] = {}

        # Last fetch timestamps
        self._last_fetch: float = 0.0

        # Background timer
        self._timer: threading.Timer | None = None
        self._stopped = False

    def _is_enabled(self) -> bool:
        """Check if remote registry is enabled."""
        disabled = os.environ.get("VETCH_REGISTRY_REMOTE", "").lower()
        return disabled not in ("false", "0", "no")

    def _fetch_json(self, filename: str) -> dict[str, Any] | None:
        """Fetch a JSON file from remote, using ETag caching.

        Args:
            filename: Name of the file (e.g., "energy.json").

        Returns:
            Parsed JSON dict, or None if fetch failed or not modified.
        """
        url = f"{self._base_url.rstrip('/')}/{filename}"

        try:
            req = Request(url)
            req.add_header("User-Agent", "vetch-sdk")

            # Add ETag for conditional request
            etag = self._etags.get(filename)
            if etag:
                req.add_header("If-None-Match", etag)

            response = urlopen(req, timeout=self._timeout)  # noqa: S310

            # Store new ETag
            new_etag = response.headers.get("ETag")
            if new_etag:
                self._etags[filename] = new_etag

            data = json.loads(response.read().decode("utf-8"))
            return data  # type: ignore[no-any-return]

        except URLError as e:
            # Check for 304 Not Modified
            if hasattr(e, "code") and getattr(e, "code", None) == 304:
                logger.debug(f"Registry {filename} not modified (304)")
                return None
            logger.debug(f"Failed to fetch remote registry {filename}: {e}")
            return None
        except Exception as e:
            logger.debug(f"Failed to fetch remote registry {filename}: {e}")
            return None

    def _merge_registry(
        self, bundled: dict[str, Any], remote: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge remote registry data with bundled defaults.

        Remote values override bundled if the remote entry has a newer
        or equal version field. Entries only in bundled are preserved.

        Args:
            bundled: The bundled (local) registry data.
            remote: The remote registry data.

        Returns:
            Merged registry dict.
        """
        merged = dict(bundled)

        for key, remote_entry in remote.items():
            if key.startswith("_"):
                # Skip metadata keys
                continue

            if key not in merged:
                # New model from remote
                merged[key] = remote_entry
                continue

            # Compare versions if both have them
            bundled_version = merged[key].get("version", "0") if isinstance(
                merged[key], dict
            ) else "0"
            remote_version = remote_entry.get("version", "0") if isinstance(
                remote_entry, dict
            ) else "0"

            if remote_version >= bundled_version:
                merged[key] = remote_entry

        return merged

    def fetch_and_cache(self) -> bool:
        """Fetch remote registries and update cache.

        Thread-safe. Called by background timer or manually.

        Returns:
            True if any data was updated, False otherwise.
        """
        if not self._is_enabled():
            return False

        updated = False

        with self._lock:
            # Fetch energy registry
            energy_data = self._fetch_json("energy.json")
            if energy_data is not None:
                self._energy_cache = energy_data
                updated = True

            # Fetch pricing registry
            pricing_data = self._fetch_json("pricing.json")
            if pricing_data is not None:
                self._pricing_cache = pricing_data
                updated = True

            self._last_fetch = time.monotonic()

        return updated

    def get_energy(self, bundled: dict[str, Any]) -> dict[str, Any]:
        """Get energy registry, merging remote cache with bundled.

        Args:
            bundled: The bundled energy registry.

        Returns:
            Merged energy registry.
        """
        with self._lock:
            if self._energy_cache is not None:
                return self._merge_registry(bundled, self._energy_cache)
        return bundled

    def get_pricing(self, bundled: dict[str, Any]) -> dict[str, Any]:
        """Get pricing registry, merging remote cache with bundled.

        Args:
            bundled: The bundled pricing registry.

        Returns:
            Merged pricing registry.
        """
        with self._lock:
            if self._pricing_cache is not None:
                return self._merge_registry(bundled, self._pricing_cache)
        return bundled

    @property
    def last_fetch_time(self) -> float:
        """Monotonic timestamp of last successful fetch."""
        return self._last_fetch

    @property
    def has_remote_data(self) -> bool:
        """Whether remote data has been fetched."""
        with self._lock:
            return self._energy_cache is not None or self._pricing_cache is not None

    def start_background_refresh(self) -> None:
        """Start background refresh timer with decorrelated jitter."""
        if self._stopped or not self._is_enabled():
            return

        # Decorrelated jitter: next interval = random(base, last * 3), capped
        base_seconds = self._refresh_hours * 3600
        jitter = random.uniform(0.5 * base_seconds, 1.5 * base_seconds)

        self._timer = threading.Timer(jitter, self._background_refresh)
        self._timer.daemon = True
        self._timer.name = "vetch-registry-refresh"
        self._timer.start()

    def _background_refresh(self) -> None:
        """Background refresh callback."""
        if self._stopped:
            return

        try:
            self.fetch_and_cache()
        except Exception as e:
            logger.debug(f"Background registry refresh failed: {e}")

        # Schedule next refresh
        self.start_background_refresh()

    def stop(self) -> None:
        """Stop background refresh."""
        self._stopped = True
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None


def load_offline_registry(
    registry_path: str | Path, filename: str
) -> dict[str, Any] | None:
    """Load registry from offline file path.

    For air-gapped environments using VETCH_REGISTRY_PATH.

    Args:
        registry_path: Path to directory containing registry files.
        filename: Name of the file (e.g., "energy.json").

    Returns:
        Parsed JSON dict, or None if load failed.
    """
    path = Path(registry_path) / filename
    try:
        if path.exists():
            return json.loads(path.read_text())  # type: ignore[no-any-return]
    except Exception as e:
        logger.warning(f"Failed to load offline registry {path}: {e}")
    return None


def freeze_registry(output_path: str | Path, base_url: str | None = None) -> bool:
    """Download current remote registry to a local file.

    For CI/CD and serverless environments where cold-start latency matters.

    Usage:
        vetch registry freeze --output ./vetch_registry.json

    Args:
        output_path: Path to write the frozen registry.
        base_url: Override registry URL.

    Returns:
        True if freeze succeeded, False otherwise.
    """
    fetcher = RemoteRegistryFetcher(base_url=base_url)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    frozen: dict[str, Any] = {}

    energy = fetcher._fetch_json("energy.json")
    if energy is not None:
        frozen["energy"] = energy

    pricing = fetcher._fetch_json("pricing.json")
    if pricing is not None:
        frozen["pricing"] = pricing

    aliases = fetcher._fetch_json("aliases.json")
    if aliases is not None:
        frozen["aliases"] = aliases

    if not frozen:
        logger.error("Failed to fetch any registry data for freeze")
        return False

    frozen["_frozen_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    frozen["_source"] = fetcher._base_url

    output.write_text(json.dumps(frozen, indent=2))
    logger.info(f"Registry frozen to {output}")
    return True


# Global fetcher instance (lazy-initialized)
_fetcher: RemoteRegistryFetcher | None = None
_fetcher_lock = threading.Lock()


def get_remote_fetcher() -> RemoteRegistryFetcher | None:
    """Get or create the global remote registry fetcher.

    Returns None if remote registry is disabled.

    Returns:
        RemoteRegistryFetcher instance or None.
    """
    global _fetcher

    disabled = os.environ.get("VETCH_REGISTRY_REMOTE", "").lower()
    if disabled in ("false", "0", "no"):
        return None

    if _fetcher is None:
        with _fetcher_lock:
            if _fetcher is None:
                _fetcher = RemoteRegistryFetcher()
                # Do initial fetch (non-blocking, best-effort)
                with contextlib.suppress(Exception):
                    _fetcher.fetch_and_cache()
                # Start background refresh
                _fetcher.start_background_refresh()

    return _fetcher


def reset_remote_fetcher() -> None:
    """Reset the global fetcher. Primarily for testing."""
    global _fetcher
    with _fetcher_lock:
        if _fetcher is not None:
            _fetcher.stop()
            _fetcher = None
