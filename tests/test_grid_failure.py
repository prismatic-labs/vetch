"""Tests for grid sensing failure scenarios.

These tests verify fail-open behavior:
- Fallback to regional averages when API fails
- Fallback to global average when region unknown
- Signal quality tracking
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from vetch.sensing.cache import CachedIntensity, FileCache
from vetch.sensing.grid import (
    GridIntensity,
    _extract_country_code,
    _get_fallback_intensity,
    _get_signal_quality,
    get_carbon_intensity,
)
from vetch.sensing.memory_cache import MemoryCache


class TestSignalQuality:
    """Tests for signal quality determination."""

    def test_live_within_5_minutes(self) -> None:
        """Data under 5 minutes is live."""
        quality = _get_signal_quality(60)  # 1 minute
        assert quality == "live"

    def test_live_at_threshold(self) -> None:
        """Data at exactly 5 minutes is live."""
        quality = _get_signal_quality(299)
        assert quality == "live"

    def test_delayed_after_5_minutes(self) -> None:
        """Data over 5 minutes is delayed."""
        quality = _get_signal_quality(301)
        assert quality == "delayed"

    def test_delayed_within_30_minutes(self) -> None:
        """Data under 30 minutes is delayed."""
        quality = _get_signal_quality(1500)  # 25 minutes
        assert quality == "delayed"

    def test_blind_after_30_minutes(self) -> None:
        """Data over 30 minutes is blind."""
        quality = _get_signal_quality(1801)
        assert quality == "blind"

    def test_unknown_when_none(self) -> None:
        """None age results in unknown."""
        quality = _get_signal_quality(None)
        assert quality == "unknown"


class TestCountryCodeExtraction:
    """Tests for country code extraction from region."""

    def test_us_regions(self) -> None:
        """US regions extract US code."""
        assert _extract_country_code("us-east-1") == "US"
        assert _extract_country_code("us-west-2") == "US"
        assert _extract_country_code("us-central1") == "US"

    def test_canada_regions(self) -> None:
        """Canada regions extract CA code."""
        assert _extract_country_code("ca-central-1") == "CA"

    def test_south_america_regions(self) -> None:
        """South America regions extract BR code."""
        assert _extract_country_code("sa-east-1") == "BR"

    def test_australia_regions(self) -> None:
        """Australia regions extract AU code."""
        assert _extract_country_code("australia-southeast1") == "AU"

    def test_asia_northeast_regions(self) -> None:
        """Asia northeast regions extract JP code."""
        assert _extract_country_code("asia-northeast1") == "JP"

    def test_unknown_regions(self) -> None:
        """Unknown regions return None."""
        assert _extract_country_code("eu-west-1") is None
        assert _extract_country_code("europe-west1") is None


class TestFallbackIntensity:
    """Tests for fallback intensity values."""

    def test_known_region_fallback(self) -> None:
        """Known region returns specific intensity."""
        intensity = _get_fallback_intensity("us-east-1")
        assert intensity == 380.0

    def test_known_country_fallback(self) -> None:
        """Region with known country code uses country default."""
        # us-west-1 is not in regions, but US is in country_defaults
        intensity = _get_fallback_intensity("us-west-1")
        assert intensity == 230.0  # us-west-1 is in regions

    def test_global_fallback(self) -> None:
        """Unknown region returns global average."""
        intensity = _get_fallback_intensity("unknown-region-xyz")
        assert intensity == 436.0  # Global average

    def test_none_region_fallback(self) -> None:
        """None region returns global average."""
        intensity = _get_fallback_intensity(None)
        assert intensity == 436.0


class TestGetCarbonIntensityWithCache:
    """Tests for carbon intensity retrieval with caching."""

    def test_memory_cache_hit(self) -> None:
        """Return cached value from memory cache."""
        memory_cache: MemoryCache[float] = MemoryCache()
        memory_cache.set("us-east-1", 350.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_cache = FileCache(Path(tmpdir) / "cache.json")

            result = get_carbon_intensity(
                "us-east-1",
                memory_cache=memory_cache,
                file_cache=file_cache,
            )

            assert result.intensity_gco2e_kwh == 350.0
            assert result.signal_quality == "live"

    def test_file_cache_hit(self) -> None:
        """Return cached value from file cache and promote to memory."""
        memory_cache: MemoryCache[float] = MemoryCache()

        with tempfile.TemporaryDirectory() as tmpdir:
            file_cache = FileCache(Path(tmpdir) / "cache.json")
            file_cache.set(
                "us-east-1",
                CachedIntensity(
                    intensity_gco2e_kwh=360.0,
                    timestamp=time.time(),
                    signal_quality="live",
                ),
            )

            result = get_carbon_intensity(
                "us-east-1",
                memory_cache=memory_cache,
                file_cache=file_cache,
            )

            assert result.intensity_gco2e_kwh == 360.0
            # Should be promoted to memory cache
            assert memory_cache.get("us-east-1")[0] == 360.0

    def test_fallback_on_cache_miss(self) -> None:
        """Return fallback when no cache hit and no API key."""
        memory_cache: MemoryCache[float] = MemoryCache()

        with tempfile.TemporaryDirectory() as tmpdir:
            file_cache = FileCache(Path(tmpdir) / "cache.json")

            result = get_carbon_intensity(
                "us-east-1",
                memory_cache=memory_cache,
                file_cache=file_cache,
                api_key=None,
            )

            assert result.intensity_gco2e_kwh == 380.0  # Regional average
            assert result.signal_quality == "blind"

    def test_none_region_returns_global(self) -> None:
        """None region returns global average."""
        memory_cache: MemoryCache[float] = MemoryCache()

        with tempfile.TemporaryDirectory() as tmpdir:
            file_cache = FileCache(Path(tmpdir) / "cache.json")

            result = get_carbon_intensity(
                None,
                memory_cache=memory_cache,
                file_cache=file_cache,
            )

            assert result.intensity_gco2e_kwh == 436.0
            assert result.signal_quality == "unknown"

    def test_force_refresh_skips_cache(self) -> None:
        """force_refresh bypasses caches."""
        memory_cache: MemoryCache[float] = MemoryCache()
        memory_cache.set("us-east-1", 999.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_cache = FileCache(Path(tmpdir) / "cache.json")

            result = get_carbon_intensity(
                "us-east-1",
                memory_cache=memory_cache,
                file_cache=file_cache,
                force_refresh=True,
            )

            # Should get fallback, not cached value
            assert result.intensity_gco2e_kwh == 380.0
            assert result.signal_quality == "blind"


class TestGetCarbonIntensityAPIFailure:
    """Tests for API failure scenarios."""

    def test_api_failure_returns_fallback(self) -> None:
        """API failure returns regional fallback."""
        memory_cache: MemoryCache[float] = MemoryCache()

        with tempfile.TemporaryDirectory() as tmpdir:
            file_cache = FileCache(Path(tmpdir) / "cache.json")

            # Mock API to fail
            with patch(
                "vetch.sensing.grid._fetch_from_api",
                return_value=None,
            ):
                result = get_carbon_intensity(
                    "us-east-1",
                    memory_cache=memory_cache,
                    file_cache=file_cache,
                    api_key="test-key",
                )

                assert result.intensity_gco2e_kwh == 380.0
                assert result.signal_quality == "blind"

    def test_api_success_updates_caches(self) -> None:
        """Successful API call updates both caches."""
        memory_cache: MemoryCache[float] = MemoryCache()

        with tempfile.TemporaryDirectory() as tmpdir:
            file_cache = FileCache(Path(tmpdir) / "cache.json")

            mock_result = GridIntensity(
                intensity_gco2e_kwh=325.0,
                signal_quality="live",
                timestamp=time.time(),
            )

            with patch(
                "vetch.sensing.grid._fetch_from_api",
                return_value=mock_result,
            ):
                result = get_carbon_intensity(
                    "us-east-1",
                    memory_cache=memory_cache,
                    file_cache=file_cache,
                    api_key="test-key",
                )

                assert result.intensity_gco2e_kwh == 325.0
                assert result.signal_quality == "live"

                # Check caches were updated
                assert memory_cache.get("us-east-1")[0] == 325.0
                assert file_cache.get("us-east-1") is not None


class TestSignalQualityFromCache:
    """Tests for signal quality based on cached data age."""

    def test_fresh_cache_is_live(self) -> None:
        """Recently cached data is live."""
        memory_cache: MemoryCache[float] = MemoryCache()
        memory_cache.set("us-east-1", 350.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_cache = FileCache(Path(tmpdir) / "cache.json")

            result = get_carbon_intensity(
                "us-east-1",
                memory_cache=memory_cache,
                file_cache=file_cache,
            )

            assert result.signal_quality == "live"

    def test_old_file_cache_is_delayed(self) -> None:
        """File cache with old timestamp is delayed."""
        memory_cache: MemoryCache[float] = MemoryCache()

        with tempfile.TemporaryDirectory() as tmpdir:
            file_cache = FileCache(Path(tmpdir) / "cache.json")

            # Set cache with old timestamp (10 minutes ago)
            file_cache.set(
                "us-east-1",
                CachedIntensity(
                    intensity_gco2e_kwh=350.0,
                    timestamp=time.time() - 600,
                    signal_quality="live",
                ),
            )

            result = get_carbon_intensity(
                "us-east-1",
                memory_cache=memory_cache,
                file_cache=file_cache,
            )

            assert result.signal_quality == "delayed"
