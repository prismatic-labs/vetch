"""Tests for dynamic remote registry fetching."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from vetch.registry.remote import (
    RemoteRegistryFetcher,
    freeze_registry,
    get_vetch_home,
    load_offline_registry,
    reset_remote_fetcher,
)


class TestGetVetchHome:
    """Tests for VETCH_HOME directory resolution."""

    def test_default_is_home_dot_vetch(self) -> None:
        """Default VETCH_HOME is ~/.vetch/."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VETCH_HOME", None)
            home = get_vetch_home()
            assert home == Path.home() / ".vetch"

    def test_respects_env_var(self) -> None:
        """VETCH_HOME env var overrides default."""
        with patch.dict(os.environ, {"VETCH_HOME": "/tmp/custom-vetch"}):
            home = get_vetch_home()
            assert home == Path("/tmp/custom-vetch")


class TestRemoteRegistryFetcher:
    """Tests for RemoteRegistryFetcher."""

    def test_is_disabled_by_default(self) -> None:
        """Remote registry is disabled by default (opt-in since 0.1.7)."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VETCH_REGISTRY_REMOTE", None)
            fetcher = RemoteRegistryFetcher()
            assert fetcher._is_enabled() is False

    def test_enabled_via_env_true(self) -> None:
        """Remote registry can be enabled via VETCH_REGISTRY_REMOTE=true."""
        with patch.dict(os.environ, {"VETCH_REGISTRY_REMOTE": "true"}):
            fetcher = RemoteRegistryFetcher()
            assert fetcher._is_enabled() is True

    def test_enabled_via_env_one(self) -> None:
        """Remote registry enabled via VETCH_REGISTRY_REMOTE=1."""
        with patch.dict(os.environ, {"VETCH_REGISTRY_REMOTE": "1"}):
            fetcher = RemoteRegistryFetcher()
            assert fetcher._is_enabled() is True

    def test_enabled_via_env_yes(self) -> None:
        """Remote registry enabled via VETCH_REGISTRY_REMOTE=yes."""
        with patch.dict(os.environ, {"VETCH_REGISTRY_REMOTE": "yes"}):
            fetcher = RemoteRegistryFetcher()
            assert fetcher._is_enabled() is True

    def test_disabled_via_env_false(self) -> None:
        """Remote registry stays disabled via VETCH_REGISTRY_REMOTE=false."""
        with patch.dict(os.environ, {"VETCH_REGISTRY_REMOTE": "false"}):
            fetcher = RemoteRegistryFetcher()
            assert fetcher._is_enabled() is False

    def test_disabled_via_env_zero(self) -> None:
        """Remote registry stays disabled via VETCH_REGISTRY_REMOTE=0."""
        with patch.dict(os.environ, {"VETCH_REGISTRY_REMOTE": "0"}):
            fetcher = RemoteRegistryFetcher()
            assert fetcher._is_enabled() is False

    def test_merge_registry_adds_new_entries(self) -> None:
        """Remote entries not in bundled are added."""
        fetcher = RemoteRegistryFetcher()

        bundled = {"model-a": {"wh_per_1k_input": 0.1}}
        remote = {"model-b": {"wh_per_1k_input": 0.2}}

        merged = fetcher._merge_registry(bundled, remote)
        assert "model-a" in merged
        assert "model-b" in merged

    def test_merge_registry_overrides_with_newer_version(self) -> None:
        """Remote overrides bundled when version is newer."""
        fetcher = RemoteRegistryFetcher()

        bundled = {"model-a": {"wh_per_1k_input": 0.1, "version": "1"}}
        remote = {"model-a": {"wh_per_1k_input": 0.2, "version": "2"}}

        merged = fetcher._merge_registry(bundled, remote)
        assert merged["model-a"]["wh_per_1k_input"] == 0.2

    def test_merge_registry_keeps_bundled_when_newer(self) -> None:
        """Bundled preserved when version is newer than remote."""
        fetcher = RemoteRegistryFetcher()

        bundled = {"model-a": {"wh_per_1k_input": 0.1, "version": "2"}}
        remote = {"model-a": {"wh_per_1k_input": 0.2, "version": "1"}}

        merged = fetcher._merge_registry(bundled, remote)
        assert merged["model-a"]["wh_per_1k_input"] == 0.1

    def test_merge_registry_skips_metadata_keys(self) -> None:
        """Keys starting with _ are skipped during merge."""
        fetcher = RemoteRegistryFetcher()

        bundled = {"model-a": {"wh_per_1k_input": 0.1}}
        remote = {"_comment": "metadata", "model-a": {"wh_per_1k_input": 0.2}}

        merged = fetcher._merge_registry(bundled, remote)
        assert "_comment" not in merged

    def test_get_energy_returns_bundled_when_no_cache(self) -> None:
        """get_energy returns bundled data when no cache exists."""
        fetcher = RemoteRegistryFetcher()

        bundled = {"model-a": {"wh_per_1k_input": 0.1}}
        result = fetcher.get_energy(bundled)
        assert result == bundled

    def test_get_energy_merges_with_cache(self) -> None:
        """get_energy merges cached remote data with bundled."""
        fetcher = RemoteRegistryFetcher()
        fetcher._energy_cache = {"model-b": {"wh_per_1k_input": 0.2}}

        bundled = {"model-a": {"wh_per_1k_input": 0.1}}
        result = fetcher.get_energy(bundled)
        assert "model-a" in result
        assert "model-b" in result

    def test_fetch_and_cache_disabled(self) -> None:
        """fetch_and_cache returns False when disabled."""
        with patch.dict(os.environ, {"VETCH_REGISTRY_REMOTE": "false"}):
            fetcher = RemoteRegistryFetcher()
            result = fetcher.fetch_and_cache()
            assert result is False

    def test_has_remote_data_initially_false(self) -> None:
        """has_remote_data is False before first fetch."""
        fetcher = RemoteRegistryFetcher()
        assert fetcher.has_remote_data is False

    def test_stop_cancels_timer(self) -> None:
        """stop() cancels background timer."""
        fetcher = RemoteRegistryFetcher()
        mock_timer = MagicMock()
        fetcher._timer = mock_timer
        fetcher.stop()
        assert fetcher._stopped is True
        mock_timer.cancel.assert_called_once()

    def test_fetch_json_handles_network_error(self) -> None:
        """_fetch_json returns None on network error."""
        fetcher = RemoteRegistryFetcher(
            base_url="https://nonexistent.invalid",
            timeout_seconds=0.5,
        )
        result = fetcher._fetch_json("energy.json")
        assert result is None

    def test_custom_base_url_via_env(self) -> None:
        """VETCH_REGISTRY_URL env var overrides base URL."""
        with patch.dict(
            os.environ, {"VETCH_REGISTRY_URL": "https://custom.example.com/registry"}
        ):
            fetcher = RemoteRegistryFetcher()
            assert "custom.example.com" in fetcher._base_url

    def test_custom_refresh_hours_via_env(self) -> None:
        """VETCH_REGISTRY_REFRESH_HOURS env var overrides interval."""
        with patch.dict(os.environ, {"VETCH_REGISTRY_REFRESH_HOURS": "12"}):
            fetcher = RemoteRegistryFetcher()
            assert fetcher._refresh_hours == 12.0


class TestOfflineRegistry:
    """Tests for offline registry loading."""

    def test_load_from_directory(self) -> None:
        """Load registry from offline directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            energy = {"test-model": {"wh_per_1k_input": 0.5}}
            Path(tmpdir, "energy.json").write_text(json.dumps(energy))

            result = load_offline_registry(tmpdir, "energy.json")
            assert result is not None
            assert "test-model" in result

    def test_returns_none_for_missing_file(self) -> None:
        """Return None when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_offline_registry(tmpdir, "nonexistent.json")
            assert result is None

    def test_returns_none_for_invalid_json(self) -> None:
        """Return None for malformed JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "bad.json").write_text("not json{{{")
            result = load_offline_registry(tmpdir, "bad.json")
            assert result is None

    def test_offline_mode_via_env(self) -> None:
        """VETCH_REGISTRY_PATH env var enables offline mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            energy = {
                "offline-model": {
                    "wh_per_1k_input": 0.1,
                    "wh_per_1k_output": 0.3,
                    "tier": 3,
                    "basis": "test",
                }
            }
            Path(tmpdir, "energy.json").write_text(json.dumps(energy))

            with patch.dict(
                os.environ,
                {"VETCH_REGISTRY_PATH": tmpdir, "VETCH_REGISTRY_REMOTE": "false"},
            ):
                from vetch.calculation import _reset_registries, resolve_model

                _reset_registries()
                try:
                    resolved, known = resolve_model("offline-model")
                    assert known is True
                    assert resolved == "offline-model"
                finally:
                    _reset_registries()


class TestFreezeRegistry:
    """Tests for registry freeze functionality."""

    def test_freeze_creates_file(self) -> None:
        """freeze_registry writes a JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "frozen.json"

            # Mock the fetcher to return test data
            with patch.object(
                RemoteRegistryFetcher,
                "_fetch_json",
                side_effect=[
                    {"test-model": {"wh_per_1k_input": 0.1}},
                    {"test-model": {"usd_per_1k_input": 0.01}},
                    {"test-alias": "test-model"},
                ],
            ):
                result = freeze_registry(output)

            assert result is True
            assert output.exists()

            data = json.loads(output.read_text())
            assert "energy" in data
            assert "pricing" in data
            assert "_frozen_at" in data

    def test_freeze_fails_on_network_error(self) -> None:
        """freeze_registry returns False on fetch failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "frozen.json"

            with patch.object(
                RemoteRegistryFetcher,
                "_fetch_json",
                return_value=None,
            ):
                result = freeze_registry(output)

            assert result is False


class TestResetRemoteFetcher:
    """Tests for global fetcher reset."""

    def test_reset_clears_global(self) -> None:
        """reset_remote_fetcher clears the global instance."""
        reset_remote_fetcher()
        # Should not raise
        reset_remote_fetcher()


class TestETagCaching:
    """Tests for ETag-based caching."""

    def test_etag_stored_on_fetch(self) -> None:
        """ETag from response is stored for next request."""
        fetcher = RemoteRegistryFetcher()

        mock_response = MagicMock()
        mock_response.read.return_value = b'{"test": "data"}'
        mock_response.headers = {"ETag": '"abc123"'}

        with patch("vetch.registry.remote.urlopen", return_value=mock_response):
            fetcher._fetch_json("energy.json")

        assert fetcher._etags.get("energy.json") == '"abc123"'
