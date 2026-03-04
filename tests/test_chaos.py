"""Chaos engineering tests for Vetch.

Verifies fail-open behavior under various failure modes:
1. Grid API timeout/error
2. Corrupt cache files (invalid JSON)
3. Read-only filesystem (PermissionError)
4. Corrupt registry files
5. Patch failures
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from vetch.emitter import BufferedEmitter, set_test_emitter
from vetch.sensing.cache import FileCache
from vetch.sensing.grid import get_carbon_intensity
from vetch.wrapper import VetchContext


class TestChaosGridAPI:
    """Chaos tests for Grid API failures."""

    def test_grid_api_timeout_fail_open(self) -> None:
        """Verify fallback to blind signal when API times out."""
        with patch("urllib.request.urlopen", side_effect=TimeoutError("API Timeout")):
            # Should not raise, should return blind/fallback
            intensity = get_carbon_intensity("us-east-1", api_key="fake")
            assert intensity.signal_quality == "blind"
            assert intensity.intensity_gco2e_kwh > 0

    def test_grid_api_500_fail_open(self) -> None:
        """Verify fallback when API returns 500."""
        from io import BytesIO
        from urllib.error import HTTPError

        mock_error = HTTPError("url", 500, "Internal Server Error", {}, BytesIO(b""))
        with patch("urllib.request.urlopen", side_effect=mock_error):
            intensity = get_carbon_intensity("us-east-1", api_key="fake")
            assert intensity.signal_quality == "blind"


class TestChaosCache:
    """Chaos tests for cache failures."""

    def test_corrupt_cache_file(self) -> None:
        """Verify recovery from corrupt (invalid JSON) cache file."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp.write(b"NOT JSON {")
            cache_path = Path(tmp.name)

        try:
            cache = FileCache(cache_path)
            # Should not crash, should return None and handle gracefully
            val = cache.get("us-east-1")
            assert val is None

            # Should be able to overwrite corrupt cache
            from vetch.sensing.cache import CachedIntensity
            cache.set("us-east-1", CachedIntensity(100, 1234, "live"))
            assert cache.get("us-east-1").intensity_gco2e_kwh == 100
        finally:
            if cache_path.exists():
                os.unlink(cache_path)

    def test_readonly_filesystem(self) -> None:
        """Verify fail-open when filesystem is read-only."""
        with patch("os.replace", side_effect=PermissionError("Read-only")):
            cache = FileCache(Path("/tmp/fake-cache.json"))
            from vetch.sensing.cache import CachedIntensity
            # Should not raise
            success = cache.set("us-east-1", CachedIntensity(100, 1234, "live"))
            assert success is False


class TestChaosRegistry:
    """Chaos tests for registry failures."""

    def test_missing_registry_file(self) -> None:
        """Verify fallback when registry files are missing."""
        from vetch.calculation import _reset_registries, calculate_energy

        _reset_registries()
        with patch("pathlib.Path.read_text", side_effect=FileNotFoundError()):
            # Should use fallback values
            energy, tier, uncertainty_pct, source, basis, known = calculate_energy(
                1000, 500, "gpt-4o"
            )
            assert source == "fallback"
            assert energy > 0
            assert known is False
        _reset_registries()


class TestChaosWrapper:
    """Chaos tests for wrapper/context failures."""

    def test_llm_call_works_when_vetch_emitter_crashes(self) -> None:
        """Verify LLM call proceeds even if emitter raises exception."""
        emitter = BufferedEmitter()
        set_test_emitter(emitter)

        try:
            # Mock emit to raise
            with patch("vetch.emitter.emit", side_effect=RuntimeError("Emitter crash")):
                with VetchContext() as ctx:
                    # Simulated LLM call
                    result = "LLM Success"

                # Context should have completed without raising the Emitter crash
                assert result == "LLM Success"
        finally:
            set_test_emitter(None)

    def test_invalid_energy_override_fail_open(self) -> None:
        """Verify fallback when user provides invalid energy_override."""
        # wh_per_1k_input is required, omitting it
        invalid_override = {"wh_per_1k_output": 1.5}

        with VetchContext(energy_override=invalid_override) as ctx:
            # Should fall back to registry
            assert ctx._energy_override is None

    def test_exception_not_suppressed(self) -> None:
        """User exceptions MUST NOT be suppressed."""
        import pytest

        class UserCodeError(Exception):
            pass

        with pytest.raises(UserCodeError):
            with VetchContext() as ctx:
                raise UserCodeError("User's business logic error")


class TestKillSwitch:
    """Tests for VETCH_DISABLED emergency kill switch."""

    def test_disabled_context_is_noop(self) -> None:
        """Disabled context should be a no-op."""
        emitter = BufferedEmitter()
        set_test_emitter(emitter)

        try:
            # Create explicitly disabled context
            with VetchContext(_disabled=True) as ctx:
                # Should complete without error
                pass

            # tracking_disabled should be True
            assert ctx.tracking_disabled is True
        finally:
            set_test_emitter(None)

    def test_disabled_context_skips_setup(self) -> None:
        """Disabled context should skip normal setup."""
        # Create explicitly disabled context
        ctx = VetchContext(_disabled=True)

        # Enter context - should be fast because setup is skipped
        with ctx:
            pass

        # tracking_disabled should be True
        assert ctx.tracking_disabled is True
        # _start_time should be None (setup skipped)
        assert ctx._start_time is None


class TestGracefulDegradation:
    """Tests for graceful degradation under adverse conditions."""

    def test_handles_zero_tokens(self) -> None:
        """Zero tokens should not crash."""
        from vetch.calculation import calculate_energy

        energy, tier, uncertainty_pct, source, basis, known = calculate_energy(0, 0, "gpt-4o")
        assert energy == 0.0

    def test_handles_very_large_token_counts(self) -> None:
        """Extremely large token counts should not overflow."""
        from vetch.calculation import calculate_energy

        # 1 billion tokens
        energy, tier, uncertainty_pct, source, basis, known = calculate_energy(
            1_000_000_000, 500_000_000, "gpt-4o"
        )
        assert energy > 0
        assert energy < float("inf")

    def test_handles_empty_model_name(self) -> None:
        """Empty model name should fall back gracefully."""
        from vetch.calculation import calculate_energy

        energy, tier, uncertainty_pct, source, basis, known = calculate_energy(100, 50, "")
        assert energy >= 0
        assert known is False
        assert source == "fallback"


class TestChaosNetwork:
    """Chaos tests for network failures."""

    def test_malformed_grid_api_response(self) -> None:
        """Verify fallback when API returns invalid JSON."""
        from unittest.mock import MagicMock

        mock_response = MagicMock()
        mock_response.read.return_value = b"NOT VALID JSON {"
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            intensity = get_carbon_intensity("us-east-1", api_key="fake")
            assert intensity.signal_quality == "blind"
            assert intensity.intensity_gco2e_kwh > 0

    def test_http_emitter_timeout(self) -> None:
        """HTTP emitter should fail silently on timeout."""
        import logging

        from vetch.emitter import HttpHandler

        handler = HttpHandler("http://localhost:9999/fake")
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg='{"test": true}', args=(), exc_info=None
        )
        # Should not raise - fails silently
        handler.emit(record)

    def test_http_emitter_connection_refused(self) -> None:
        """HTTP emitter should fail silently on connection refused."""
        import logging

        from vetch.emitter import HttpHandler

        handler = HttpHandler("http://127.0.0.1:1/unreachable")
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg='{"test": true}', args=(), exc_info=None
        )
        # Should not raise
        handler.emit(record)


class TestChaosStorage:
    """Chaos tests for SQLite storage failures."""

    def test_corrupt_sqlite_database(self) -> None:
        """Verify fail-open when SQLite database is corrupt."""
        import tempfile

        from vetch.storage import configure_storage, store_event

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            # Write garbage to simulate corrupt DB
            tmp.write(b"NOT A SQLITE DATABASE" * 100)
            db_path = Path(tmp.name)

        try:
            configure_storage(enabled=True, path=db_path)

            # Should not raise - fails silently
            event = {
                "event_id": "test-123",
                "timestamp": "2026-02-17T00:00:00Z",
                "model": "gpt-4o",
                "estimated_energy_wh": 0.1,
            }
            store_event(event)  # Should not raise
        finally:
            configure_storage(enabled=False)
            if db_path.exists():
                os.unlink(db_path)

    def test_readonly_database_directory(self) -> None:
        """Verify fail-open when DB directory is read-only."""
        from vetch.storage import configure_storage, store_event

        # Use a path that will fail
        with patch("pathlib.Path.mkdir", side_effect=PermissionError("Read-only")):
            with patch("pathlib.Path.exists", return_value=False):
                configure_storage(enabled=True, path=Path("/nonexistent/path/db.sqlite"))

                event = {"event_id": "test", "timestamp": "2026-02-17T00:00:00Z"}
                # Should not raise
                store_event(event)

        configure_storage(enabled=False)


class TestChaosConcurrency:
    """Chaos tests for concurrent access."""

    def test_concurrent_cache_writes(self) -> None:
        """Multiple threads writing to cache should not corrupt data."""
        import threading
        import time

        from vetch.sensing.cache import CachedIntensity

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            cache_path = Path(tmp.name)

        cache = FileCache(cache_path)
        errors = []

        def writer(region_id: int) -> None:
            try:
                for i in range(10):
                    cache.set(
                        f"region-{region_id}",
                        CachedIntensity(100 + i, time.time(), "live")
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should have been raised
        assert len(errors) == 0, f"Errors during concurrent writes: {errors}"

        # Cache should still be readable
        regions = cache.list_regions()
        assert len(regions) > 0

        # Cleanup
        if cache_path.exists():
            os.unlink(cache_path)
        lock_path = cache_path.with_suffix(".lock")
        if lock_path.exists():
            os.unlink(lock_path)

    def test_concurrent_context_managers(self) -> None:
        """Multiple VetchContexts in threads should not interfere."""
        import threading

        results = []
        errors = []

        def run_context(thread_id: int) -> None:
            try:
                with VetchContext(tags={"thread": str(thread_id)}) as ctx:
                    # Simulate work
                    _ = sum(range(1000))
                results.append(ctx.event.get("tags", {}).get("thread"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run_context, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors in concurrent contexts: {errors}"
        assert len(results) == 10


class TestChaosProviders:
    """Chaos tests for provider patching failures."""

    def test_openai_patch_failure_doesnt_crash(self) -> None:
        """If OpenAI patching fails, wrap() should still work."""
        # Mock the patch function to raise
        with patch(
            "vetch.providers.openai.patch_openai_client",
            side_effect=RuntimeError("Patch failed")
        ):
            with VetchContext() as ctx:
                # Should complete without error
                pass
            # Context should have completed
            assert ctx.event is not None

    def test_anthropic_patch_failure_doesnt_crash(self) -> None:
        """If Anthropic patching fails, wrap() should still work."""
        with patch(
            "vetch.providers.anthropic.patch_anthropic_client",
            side_effect=RuntimeError("Patch failed")
        ):
            with VetchContext() as ctx:
                pass
            assert ctx.event is not None

    def test_provider_import_failure_doesnt_crash(self) -> None:
        """If provider module import fails, wrap() should still work."""
        # This simulates the case where openai isn't installed
        import sys
        original = sys.modules.get("openai")
        sys.modules["openai"] = None  # type: ignore[assignment]

        try:
            with VetchContext() as ctx:
                pass
            assert ctx.event is not None
        finally:
            if original is not None:
                sys.modules["openai"] = original
            elif "openai" in sys.modules:
                del sys.modules["openai"]


class TestChaosPricing:
    """Chaos tests for pricing calculation failures."""

    def test_missing_pricing_registry(self) -> None:
        """Cost calculation should return 0 if pricing registry is missing."""
        from vetch.calculation import _reset_registries, calculate_cost

        _reset_registries()
        with patch("pathlib.Path.read_text", side_effect=FileNotFoundError()):
            cost, input_cost, output_cost, cache_write, cache_read, tier_info = calculate_cost(1000, 500, "gpt-4o")
            # Should return 0 rather than crash
            assert cost == 0.0
            # tier_info is "none" when registry fails to load
            assert tier_info == "none"
        _reset_registries()

    def test_malformed_pricing_json(self) -> None:
        """Cost calculation should handle malformed JSON gracefully."""
        from vetch.calculation import _reset_registries, calculate_cost

        _reset_registries()
        with patch("pathlib.Path.read_text", return_value="NOT JSON {"):
            cost, input_cost, output_cost, cache_write, cache_read, tier_info = calculate_cost(1000, 500, "gpt-4o")
            assert cost == 0.0
            assert tier_info == "none"
        _reset_registries()


class TestChaosOTel:
    """Chaos tests for OpenTelemetry integration failures."""

    def test_otel_not_installed_returns_false(self) -> None:
        """When OTel is not installed, attach_to_otel_span returns False."""
        from vetch.otel import attach_to_otel_span

        event = {"estimated_energy_wh": 0.1}
        # Should return False when OTel not in sys.modules
        result = attach_to_otel_span(event)
        # Since OTel is not actually installed in test env, returns False
        assert result is False

    def test_otel_function_never_crashes(self) -> None:
        """OTel function should never raise exceptions."""
        from vetch.otel import attach_to_otel_span

        # Various edge case events
        test_events = [
            {},  # Empty event
            {"estimated_energy_wh": None},  # None values
            {"model": "x" * 10000},  # Very long strings
            {"estimated_energy_wh": float("inf")},  # Infinity
        ]

        for event in test_events:
            # Should never raise, always return bool
            result = attach_to_otel_span(event)
            assert isinstance(result, bool)


class TestChaosEdgeCases:
    """Chaos tests for edge case inputs."""

    def test_negative_token_counts_clamped_to_zero(self) -> None:
        """Negative token counts are clamped to zero."""
        from vetch.calculation import calculate_energy

        # Negative tokens should be clamped to 0
        energy, tier, uncertainty_pct, source, basis, known = calculate_energy(-100, -50, "gpt-4o")
        assert energy == 0.0  # Clamped to zero

    def test_nan_in_grid_intensity_returns_zero(self) -> None:
        """NaN grid intensity returns zero carbon (defensive)."""
        from vetch.calculation import calculate_carbon

        # NaN should be handled defensively
        carbon, pue, pue_tier, pue_source = calculate_carbon(0.1, float("nan"))
        assert carbon == 0.0  # Returns 0 instead of NaN

    def test_inf_in_grid_intensity_capped(self) -> None:
        """Infinity in grid intensity is capped."""
        from vetch.calculation import calculate_carbon

        # Inf should be capped
        carbon, pue, pue_tier, pue_source = calculate_carbon(0.1, float("inf"))
        assert carbon > 0  # Returns a real value
        assert carbon < float("inf")  # Not infinity

    def test_inf_in_energy(self) -> None:
        """Infinity in energy produces infinity in carbon."""
        from vetch.calculation import calculate_carbon

        # Inf energy with valid grid intensity
        carbon, pue, pue_tier, pue_source = calculate_carbon(float("inf"), 100)
        assert carbon == float("inf")

    def test_unicode_in_tags(self) -> None:
        """Unicode characters in tags should work."""
        with VetchContext(tags={"team": "データ科学", "emoji": "🔋"}) as ctx:
            pass

        assert ctx.event["tags"]["team"] == "データ科学"
        assert ctx.event["tags"]["emoji"] == "🔋"

    def test_very_long_model_name(self) -> None:
        """Very long model names should not crash."""
        from vetch.calculation import calculate_energy

        long_name = "a" * 10000
        energy, tier, uncertainty_pct, source, basis, known = calculate_energy(100, 50, long_name)
        assert energy >= 0
        assert known is False

    def test_special_characters_in_region(self) -> None:
        """Special characters in region should be handled."""
        # Should not crash
        intensity = get_carbon_intensity("../../../etc/passwd")
        assert intensity is not None
