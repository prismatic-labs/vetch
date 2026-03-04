"""Remote registry fetcher for dynamic model data updates.

This module provides background fetching of energy and pricing data
from a remote endpoint (CDN/GitHub), merging with bundled defaults.

Key design decisions:
- Fail-silent: network errors log debug, use cached/bundled fallback
- Memory-only cache by default (no file writes for privacy)
- ETag caching to reduce bandwidth
- Decorrelated jitter to avoid thundering herd
- Opt-in via VETCH_REGISTRY_REMOTE=true (disabled by default in 0.1.7)
"""

from __future__ import annotations

import contextlib
import hashlib
import ipaddress
import json
import logging
import os
import random
import socket
import threading
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
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


def _is_private_ip(ip: str) -> bool:
    """Check if IP address is private/internal.

    Args:
        ip: IP address string (IPv4 or IPv6).

    Returns:
        True if IP is private, loopback, link-local, or reserved.
    """
    try:
        addr = ipaddress.ip_address(ip)
        return addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved
    except ValueError:
        return False


def validate_registry_url(url: str) -> tuple[bool, str | None]:
    """Validate registry URL to prevent SSRF attacks.

    Blocks:
    - Private/internal IP addresses (10.x, 192.168.x, 127.x, etc.)
    - Non-HTTP(S) schemes (file://, ftp://, etc.)
    - Invalid URLs

    Args:
        url: URL to validate.

    Returns:
        Tuple of (is_valid, error_message).
    """
    try:
        parsed = urlparse(url)

        # Only allow HTTP/HTTPS
        if parsed.scheme not in ("http", "https"):
            return False, f"Invalid URL scheme '{parsed.scheme}', only http/https allowed"

        # Ensure hostname is present
        if not parsed.hostname:
            return False, "URL must have a hostname"

        # Resolve hostname to IP and check if private
        try:
            ip = socket.gethostbyname(parsed.hostname)
            if _is_private_ip(ip):
                return False, f"Registry URL resolves to private IP {ip}, SSRF blocked"
        except socket.gaierror:
            # Hostname doesn't resolve - let urlopen handle the error
            pass

        return True, None

    except Exception as e:
        return False, f"Invalid URL: {e}"


class RemoteRegistryFetcher:
    """Fetches and caches remote registry data.

    Thread-safe. Uses background timer for periodic refresh.
    ETag caching minimizes bandwidth usage.
    Circuit breaker prevents hammering remote on repeated failures.
    """

    def __init__(
        self,
        base_url: str | None = None,
        refresh_hours: float | None = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        """Initialize the remote registry fetcher with SSRF protection.

        Args:
            base_url: Base URL for registry files. Defaults to GitHub raw URL.
            refresh_hours: Hours between refresh attempts.
            timeout_seconds: HTTP request timeout in seconds.
        """
        self._base_url = base_url or os.environ.get(
            "VETCH_REGISTRY_URL", DEFAULT_REGISTRY_URL
        )

        # Validate base URL for SSRF
        valid, error = validate_registry_url(self._base_url)
        if not valid:
            logger.error(f"Invalid registry URL: {error}. Remote registry disabled.")
            self._base_url = ""  # Disable remote fetching

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

        # Merged registry caching (optimization to avoid re-merging when ETag unchanged)
        # Only re-merge if remote data's ETag changes
        self._merged_energy_cache: dict[str, Any] | None = None
        self._merged_pricing_cache: dict[str, Any] | None = None
        self._last_energy_etag: str | None = None
        self._last_pricing_etag: str | None = None

        # Last fetch timestamps
        self._last_fetch: float = 0.0

        # Background timer
        self._timer: threading.Timer | None = None
        self._stopped = False

        # Circuit breaker state
        self._failure_count: int = 0
        self._circuit_open_until: float = 0.0  # Monotonic timestamp
        self._max_failures: int = 3  # Open circuit after 3 consecutive failures
        self._circuit_timeout_seconds: float = 300.0  # 5 minutes

        # Signature verification (opt-in for high-security environments)
        self._verify_signatures: bool = os.environ.get(
            "VETCH_REGISTRY_VERIFY_SIGNATURES", ""
        ).lower() in ("true", "1", "yes")
        self._expected_checksums: dict[str, str] = {}  # filename -> sha256 hex

        # Load persisted circuit breaker state from disk
        self._load_circuit_state()

    def _is_enabled(self) -> bool:
        """Check if remote registry is enabled.

        Default: DISABLED (opt-in) to prevent silent accuracy regressions
        when bundled registry is newer than remote (e.g., during releases).
        """
        enabled = os.environ.get("VETCH_REGISTRY_REMOTE", "").lower()
        return enabled in ("true", "1", "yes")

    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open (blocking requests).

        Returns:
            True if circuit is open and should block requests.
        """
        now = time.monotonic()
        if now < self._circuit_open_until:
            return True
        # Circuit timeout expired, close the circuit and reset failure count
        if self._circuit_open_until > 0:
            logger.info("Remote registry circuit breaker closed, resuming requests")
            self._circuit_open_until = 0.0
            self._failure_count = 0
        return False

    def _record_failure(self) -> None:
        """Record a fetch failure and potentially open circuit breaker."""
        self._failure_count += 1
        if self._failure_count >= self._max_failures:
            self._circuit_open_until = time.monotonic() + self._circuit_timeout_seconds
            logger.warning(
                f"Remote registry circuit breaker opened after {self._failure_count} "
                f"consecutive failures. Blocking requests for {self._circuit_timeout_seconds}s"
            )
        self._save_circuit_state()

    def _record_success(self) -> None:
        """Record a successful fetch and reset circuit breaker."""
        if self._failure_count > 0:
            logger.info(f"Remote registry fetch succeeded after {self._failure_count} failures")
        self._failure_count = 0
        self._circuit_open_until = 0.0
        self._save_circuit_state()

    def _load_circuit_state(self) -> None:
        """Load circuit breaker state from persisted file.

        Loads state from ~/.vetch/circuit_state.json with 24-hour TTL.
        This allows circuit breaker to survive process restarts and prevents
        hammering remote registry immediately after restart.
        """
        try:
            vetch_home = get_vetch_home()
            state_path = vetch_home / "circuit_state.json"

            if not state_path.exists():
                return

            with open(state_path) as f:
                state = json.load(f)

            # Check TTL (24 hours = 86400 seconds)
            timestamp = state.get("timestamp", 0)
            if time.time() - timestamp > 86400:
                # State expired, clean up
                with contextlib.suppress(Exception):
                    state_path.unlink()
                return

            # Load state
            self._failure_count = state.get("failure_count", 0)
            self._circuit_open_until = state.get("open_until", 0.0)

            if self._is_circuit_open():
                logger.info(
                    f"Circuit breaker loaded from disk: open until "
                    f"{self._circuit_open_until - time.monotonic():.0f}s from now"
                )

        except Exception as e:
            logger.debug(f"Failed to load circuit breaker state: {e}")

    def _save_circuit_state(self) -> None:
        """Persist circuit breaker state to disk.

        Saves state to ~/.vetch/circuit_state.json with timestamp.
        """
        try:
            vetch_home = get_vetch_home()
            vetch_home.mkdir(parents=True, exist_ok=True)
            state_path = vetch_home / "circuit_state.json"

            state = {
                "failure_count": self._failure_count,
                "open_until": self._circuit_open_until,
                "timestamp": time.time(),
            }

            with open(state_path, "w") as f:
                json.dump(state, f)

        except Exception as e:
            logger.debug(f"Failed to save circuit breaker state: {e}")

    def _verify_checksum(self, filename: str, data: bytes) -> bool:
        """Verify SHA256 checksum of fetched data.

        Checks against expected checksums loaded from checksums.json.
        If verification is not enabled, always returns True.

        Args:
            filename: Name of the file being verified.
            data: Raw bytes of the file content.

        Returns:
            True if checksum matches or verification is disabled.
        """
        if not self._verify_signatures:
            return True

        if filename not in self._expected_checksums:
            logger.warning(
                f"No checksum found for {filename}, skipping verification. "
                f"Set VETCH_REGISTRY_VERIFY_SIGNATURES=false to suppress this warning."
            )
            return True

        computed = hashlib.sha256(data).hexdigest()
        expected = self._expected_checksums[filename]

        if computed != expected:
            logger.error(
                f"Checksum verification failed for {filename}. "
                f"Expected: {expected}, Got: {computed}. "
                f"Possible supply chain attack or corrupted download."
            )
            return False

        logger.debug(f"Checksum verified for {filename}")
        return True

    def _load_checksums(self) -> None:
        """Load expected checksums from remote checksums.json file.

        This file should contain a JSON object mapping filenames to SHA256 hashes.
        """
        if not self._verify_signatures:
            return

        try:
            url = f"{self._base_url.rstrip('/')}/checksums.json"
            req = Request(url)
            req.add_header("User-Agent", "vetch-sdk")
            response = urlopen(req, timeout=self._timeout)  # noqa: S310
            checksums = json.loads(response.read().decode("utf-8"))

            if isinstance(checksums, dict):
                self._expected_checksums = checksums
                logger.info(f"Loaded {len(checksums)} registry checksums for verification")
            else:
                logger.warning("checksums.json is not a valid dict, skipping verification")
        except Exception as e:
            logger.warning(f"Failed to load checksums.json: {e}. Signature verification disabled.")

    def _fetch_json(self, filename: str) -> dict[str, Any] | None:
        """Fetch a JSON file from remote with ETag caching, circuit breaker, and verification.

        Args:
            filename: Name of the file (e.g., "energy.json").

        Returns:
            Parsed JSON dict, or None if fetch failed or not modified.
        """
        # Check circuit breaker
        if self._is_circuit_open():
            logger.debug(f"Circuit breaker open, skipping fetch of {filename}")
            return None

        url = f"{self._base_url.rstrip('/')}/{filename}"

        # Re-validate URL immediately before fetch to prevent TOCTOU
        # DNS could have changed since initialization
        is_valid, err = validate_registry_url(url)
        if not is_valid:
            logger.warning(f"Registry URL failed re-validation: {err}. Skipping fetch of {filename}")
            self._record_failure()
            return None

        try:
            req = Request(url)
            req.add_header("User-Agent", "vetch-sdk")

            # Add ETag for conditional request
            etag = self._etags.get(filename)
            if etag:
                req.add_header("If-None-Match", etag)

            response = urlopen(req, timeout=self._timeout)  # noqa: S310

            # Read response data
            raw_data = response.read()

            # Verify checksum before parsing
            if not self._verify_checksum(filename, raw_data):
                logger.error(f"Checksum verification failed for {filename}, rejecting update")
                self._record_failure()
                return None

            # Store new ETag
            new_etag = response.headers.get("ETag")
            if new_etag:
                self._etags[filename] = new_etag

            data = json.loads(raw_data.decode("utf-8"))
            self._record_success()
            return data  # type: ignore[no-any-return]

        except URLError as e:
            # Check for 304 Not Modified (not a failure)
            if hasattr(e, "code") and getattr(e, "code", None) == 304:
                logger.debug(f"Registry {filename} not modified (304)")
                self._record_success()  # 304 is a successful response
                return None
            logger.debug(f"Failed to fetch remote registry {filename}: {e}")
            self._record_failure()
            return None
        except Exception as e:
            logger.debug(f"Failed to fetch remote registry {filename}: {e}")
            self._record_failure()
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
        """Fetch remote registries and update cache with circuit breaker protection.

        Thread-safe. Called by background timer or manually.

        Returns:
            True if any data was updated, False otherwise.
        """
        if not self._is_enabled():
            return False

        # Check circuit breaker before attempting fetch
        if self._is_circuit_open():
            logger.debug("Circuit breaker open, skipping registry fetch")
            return False

        updated = False

        with self._lock:
            # Load checksums for verification (if enabled)
            if self._verify_signatures and not self._expected_checksums:
                self._load_checksums()

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

        Uses cached merge result if remote ETag unchanged (performance optimization).

        Args:
            bundled: The bundled energy registry.

        Returns:
            Merged energy registry.
        """
        with self._lock:
            if self._energy_cache is not None:
                # Get current ETag for energy.json
                current_etag = self._etags.get("energy.json")

                # Return cached merge if ETag unchanged
                if (
                    self._merged_energy_cache is not None
                    and current_etag == self._last_energy_etag
                ):
                    return self._merged_energy_cache

                # ETag changed or no cache - perform merge
                merged = self._merge_registry(bundled, self._energy_cache)

                # Cache the merge result with its ETag
                self._merged_energy_cache = merged
                self._last_energy_etag = current_etag

                return merged

        return bundled

    def get_pricing(self, bundled: dict[str, Any]) -> dict[str, Any]:
        """Get pricing registry, merging remote cache with bundled.

        Uses cached merge result if remote ETag unchanged (performance optimization).

        Args:
            bundled: The bundled pricing registry.

        Returns:
            Merged pricing registry.
        """
        with self._lock:
            if self._pricing_cache is not None:
                # Get current ETag for pricing.json
                current_etag = self._etags.get("pricing.json")

                # Return cached merge if ETag unchanged
                if (
                    self._merged_pricing_cache is not None
                    and current_etag == self._last_pricing_etag
                ):
                    return self._merged_pricing_cache

                # ETag changed or no cache - perform merge
                merged = self._merge_registry(bundled, self._pricing_cache)

                # Cache the merge result with its ETag
                self._merged_pricing_cache = merged
                self._last_pricing_etag = current_etag

                return merged

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

    @property
    def circuit_breaker_open(self) -> bool:
        """Whether circuit breaker is currently open."""
        return self._is_circuit_open()

    @property
    def failure_count(self) -> int:
        """Number of consecutive failures."""
        return self._failure_count

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

    enabled = os.environ.get("VETCH_REGISTRY_REMOTE", "").lower()
    if enabled not in ("true", "1", "yes"):
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
