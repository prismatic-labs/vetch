"""Electricity Maps API client for carbon intensity data.

Fetches real-time grid carbon intensity with:
- Multi-tier caching (memory + file)
- Graceful fallback to regional/global averages
- Jittered retry on transient errors
- Signal quality tracking
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from vetch.sensing.cache import CachedIntensity, FileCache, get_file_cache
from vetch.sensing.memory_cache import MemoryCache, get_grid_cache

logger = logging.getLogger(__name__)

# API configuration
ELECTRICITY_MAPS_API_URL = "https://api.electricitymaps.com/v3/carbon-intensity/latest"
CONNECT_TIMEOUT_SECONDS = 0.5
READ_TIMEOUT_SECONDS = 1.0
MAX_RETRIES = 2
RETRY_DELAY_BASE_SECONDS = 0.1

# Signal quality thresholds (in seconds)
LIVE_THRESHOLD_SECONDS = 300  # 5 minutes
DELAYED_THRESHOLD_SECONDS = 1800  # 30 minutes

# Load global averages
_AVERAGES_PATH = Path(__file__).parent / "global_averages.json"
_AVERAGES: dict[str, object] | None = None


def _load_averages() -> dict[str, object]:
    """Load global averages from JSON file."""
    global _AVERAGES
    if _AVERAGES is None:
        _AVERAGES = json.loads(_AVERAGES_PATH.read_text())
    return _AVERAGES


SignalQuality = Literal["live", "delayed", "blind", "unknown"]


@dataclass
class GridIntensity:
    """Grid carbon intensity result."""

    intensity_gco2e_kwh: float
    signal_quality: SignalQuality
    timestamp: float | None = None


def _get_signal_quality(age_seconds: float | None) -> SignalQuality:
    """Determine signal quality based on data age.

    Args:
        age_seconds: Age of data in seconds, or None if unknown.

    Returns:
        Signal quality: "live", "delayed", "blind", or "unknown".
    """
    if age_seconds is None:
        return "unknown"

    if age_seconds < LIVE_THRESHOLD_SECONDS:
        return "live"
    elif age_seconds < DELAYED_THRESHOLD_SECONDS:
        return "delayed"
    else:
        return "blind"


def _get_fallback_intensity(region: str | None) -> float:
    """Get fallback intensity from regional or global averages.

    Args:
        region: Region identifier, or None for global average.

    Returns:
        Carbon intensity in gCO2e/kWh.
    """
    averages = _load_averages()

    if region is not None:
        # Try region-specific average
        regions = averages.get("regions", {})
        if isinstance(regions, dict) and region in regions:
            value = regions[region]
            if isinstance(value, (int, float)):
                return float(value)

        # Try country code extraction (e.g., "us-east-1" -> "US")
        country_code = _extract_country_code(region)
        if country_code:
            countries = averages.get("country_defaults", {})
            if isinstance(countries, dict) and country_code in countries:
                value = countries[country_code]
                if isinstance(value, (int, float)):
                    return float(value)

    # Fall back to global average
    global_avg = averages.get("global", 436)
    if isinstance(global_avg, (int, float)):
        return float(global_avg)
    return 436.0


def _extract_country_code(region: str) -> str | None:
    """Extract country code from region identifier.

    Handles patterns like:
    - us-east-1 -> US
    - eu-west-1 -> (region-specific, no country)
    - europe-west1 -> (region-specific, no country)
    - ap-northeast-1 -> (region-specific, no country)

    Args:
        region: Region identifier.

    Returns:
        Two-letter country code or None.
    """
    region_lower = region.lower()

    # Direct country mappings
    country_prefixes = {
        "us-": "US",
        "ca-": "CA",
        "sa-": "BR",  # South America defaults to Brazil
    }

    for prefix, code in country_prefixes.items():
        if region_lower.startswith(prefix):
            return code

    # Australia
    if "australia" in region_lower:
        return "AU"

    # Japan
    if "northeast" in region_lower and "asia" in region_lower:
        return "JP"

    return None


def _fetch_from_api(
    region: str,
    api_key: str,
) -> GridIntensity | None:
    """Fetch carbon intensity from Electricity Maps API.

    Args:
        region: Grid region identifier.
        api_key: Electricity Maps API key.

    Returns:
        GridIntensity if successful, None on failure.
    """
    # Build request
    url = f"{ELECTRICITY_MAPS_API_URL}?zone={region}"
    headers = {
        "auth-token": api_key,
        "Accept": "application/json",
    }

    request = urllib.request.Request(url, headers=headers)

    for attempt in range(MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(
                request,
                timeout=CONNECT_TIMEOUT_SECONDS + READ_TIMEOUT_SECONDS,
            ) as response:
                data = json.loads(response.read().decode("utf-8"))

                # Extract carbon intensity
                intensity = data.get("carbonIntensity")
                if intensity is None:
                    logger.warning(f"No carbonIntensity in API response for {region}")
                    return None

                return GridIntensity(
                    intensity_gco2e_kwh=float(intensity),
                    signal_quality="live",
                    timestamp=time.time(),
                )

        except urllib.error.HTTPError as e:
            if e.code == 429:
                # Rate limited - retry with jitter
                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAY_BASE_SECONDS * (2**attempt) * (0.5 + random.random())
                    logger.debug(f"Rate limited, retrying in {delay:.2f}s")
                    time.sleep(delay)
                    continue
            elif e.code >= 500:
                # Server error - retry with jitter
                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAY_BASE_SECONDS * (2**attempt) * (0.5 + random.random())
                    logger.debug(f"Server error {e.code}, retrying in {delay:.2f}s")
                    time.sleep(delay)
                    continue

            logger.warning(f"API error for {region}: {e.code}")
            return None

        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
            if attempt < MAX_RETRIES:
                delay = RETRY_DELAY_BASE_SECONDS * (2**attempt) * (0.5 + random.random())
                logger.debug(f"Request failed ({e}), retrying in {delay:.2f}s")
                time.sleep(delay)
                continue

            logger.warning(f"Failed to fetch intensity for {region}: {e}")
            return None

    return None


def get_carbon_intensity(
    region: str | None,
    memory_cache: MemoryCache[float] | None = None,
    file_cache: FileCache | None = None,
    api_key: str | None = None,
    force_refresh: bool = False,
) -> GridIntensity:
    """Get carbon intensity for a region.

    Uses multi-tier caching strategy:
    1. In-memory cache (fastest, per-process)
    2. File cache (shared across processes)
    3. Electricity Maps API (live data)
    4. Regional/global averages (fallback)

    Args:
        region: Grid region identifier, or None for global average.
        memory_cache: Optional memory cache (defaults to global).
        file_cache: Optional file cache (defaults to global).
        api_key: Electricity Maps API key. If None, uses
            ELECTRICITY_MAPS_API_KEY env var.
        force_refresh: Skip caches and fetch fresh data.

    Returns:
        GridIntensity with value and signal quality.
    """
    # Default to global caches
    if memory_cache is None:
        memory_cache = get_grid_cache()
    if file_cache is None:
        file_cache = get_file_cache()

    # If no region, return global average
    if region is None:
        return GridIntensity(
            intensity_gco2e_kwh=_get_fallback_intensity(None),
            signal_quality="unknown",
        )

    # Tier 1: Memory cache
    if not force_refresh:
        value, is_fresh = memory_cache.get(region)
        if value is not None:
            age = memory_cache.get_age(region)
            return GridIntensity(
                intensity_gco2e_kwh=value,
                signal_quality=_get_signal_quality(age),
                timestamp=time.time() - (age or 0),
            )

    # Tier 2: File cache
    if not force_refresh:
        cached = file_cache.get(region)
        if cached is not None:
            # Promote to memory cache
            memory_cache.set(region, cached.intensity_gco2e_kwh)

            age = time.time() - cached.timestamp
            return GridIntensity(
                intensity_gco2e_kwh=cached.intensity_gco2e_kwh,
                signal_quality=_get_signal_quality(age),
                timestamp=cached.timestamp,
            )

    # Tier 3: API fetch
    if api_key is None:
        api_key = os.environ.get("ELECTRICITY_MAPS_API_KEY")

    if api_key:
        result = _fetch_from_api(region, api_key)
        if result is not None:
            # Update caches
            memory_cache.set(region, result.intensity_gco2e_kwh)
            file_cache.set(
                region,
                CachedIntensity(
                    intensity_gco2e_kwh=result.intensity_gco2e_kwh,
                    timestamp=result.timestamp or time.time(),
                    signal_quality="live",
                ),
            )
            return result

    # Tier 4: Fallback to averages
    return GridIntensity(
        intensity_gco2e_kwh=_get_fallback_intensity(region),
        signal_quality="blind",
    )
