"""Tests for CI mode functionality.

These tests verify:
- CI environment detection
- Stats accumulation
- Summary printing
"""

from __future__ import annotations

import os
from io import StringIO
from unittest.mock import patch

import pytest


class TestCIDetection:
    """Tests for CI environment detection."""

    def test_detects_github_actions(self) -> None:
        """Detect GitHub Actions environment."""
        # Import fresh to avoid cached state
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}, clear=True):
            # Need to reimport to pick up env var
            import importlib
            import vetch.ci
            importlib.reload(vetch.ci)

            assert vetch.ci.is_ci() is True

    def test_detects_generic_ci(self) -> None:
        """Detect generic CI=true environment."""
        with patch.dict(os.environ, {"CI": "true"}, clear=True):
            import importlib
            import vetch.ci
            importlib.reload(vetch.ci)

            assert vetch.ci.is_ci() is True

    def test_not_ci_when_unset(self) -> None:
        """Not CI when env vars unset."""
        with patch.dict(os.environ, {}, clear=True):
            import importlib
            import vetch.ci
            importlib.reload(vetch.ci)

            assert vetch.ci.is_ci() is False


class TestCIStatsTracking:
    """Tests for CI stats accumulation."""

    @pytest.fixture(autouse=True)
    def reset_stats(self) -> None:
        """Reset stats between tests."""
        import vetch.ci
        vetch.ci._CI_STATS = {
            "count": 0,
            "energy_wh": 0.0,
            "carbon_g": 0.0,
            "cost_usd": 0.0,
        }
        yield

    def test_track_event_accumulates(self) -> None:
        """track_ci_event accumulates stats."""
        import vetch.ci

        # Mock is_ci to return True
        with patch.object(vetch.ci, "is_ci", return_value=True):
            vetch.ci.track_ci_event({
                "estimated_energy_wh": 0.001,
                "estimated_carbon_g": 0.05,
                "estimated_cost_usd": 0.01,
            })

            assert vetch.ci._CI_STATS["count"] == 1
            assert vetch.ci._CI_STATS["energy_wh"] == 0.001
            assert vetch.ci._CI_STATS["carbon_g"] == 0.05
            assert vetch.ci._CI_STATS["cost_usd"] == 0.01

    def test_track_multiple_events(self) -> None:
        """Multiple events accumulate correctly."""
        import vetch.ci

        with patch.object(vetch.ci, "is_ci", return_value=True):
            for _ in range(3):
                vetch.ci.track_ci_event({
                    "estimated_energy_wh": 0.001,
                    "estimated_carbon_g": 0.05,
                    "estimated_cost_usd": 0.01,
                })

            assert vetch.ci._CI_STATS["count"] == 3
            assert abs(vetch.ci._CI_STATS["energy_wh"] - 0.003) < 1e-9
            assert abs(vetch.ci._CI_STATS["carbon_g"] - 0.15) < 1e-9
            assert abs(vetch.ci._CI_STATS["cost_usd"] - 0.03) < 1e-9

    def test_track_skipped_when_not_ci(self) -> None:
        """Tracking is skipped when not in CI."""
        import vetch.ci

        with patch.object(vetch.ci, "is_ci", return_value=False):
            vetch.ci.track_ci_event({
                "estimated_energy_wh": 0.001,
                "estimated_carbon_g": 0.05,
                "estimated_cost_usd": 0.01,
            })

            assert vetch.ci._CI_STATS["count"] == 0

    def test_handles_missing_fields(self) -> None:
        """Missing fields default to 0."""
        import vetch.ci

        with patch.object(vetch.ci, "is_ci", return_value=True):
            vetch.ci.track_ci_event({})

            assert vetch.ci._CI_STATS["count"] == 1
            assert vetch.ci._CI_STATS["energy_wh"] == 0.0

    def test_handles_none_fields(self) -> None:
        """None fields default to 0."""
        import vetch.ci

        with patch.object(vetch.ci, "is_ci", return_value=True):
            vetch.ci.track_ci_event({
                "estimated_energy_wh": None,
                "estimated_carbon_g": None,
                "estimated_cost_usd": None,
            })

            assert vetch.ci._CI_STATS["count"] == 1
            assert vetch.ci._CI_STATS["energy_wh"] == 0.0


class TestCISummaryPrinting:
    """Tests for CI summary output."""

    def test_summary_not_printed_when_empty(self, capsys) -> None:
        """No summary when no events tracked."""
        import vetch.ci

        vetch.ci._CI_STATS = {
            "count": 0,
            "energy_wh": 0.0,
            "carbon_g": 0.0,
            "cost_usd": 0.0,
        }

        vetch.ci._print_ci_summary()

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_summary_format(self, capsys) -> None:
        """Summary has expected format."""
        import vetch.ci

        vetch.ci._CI_STATS = {
            "count": 10,
            "energy_wh": 0.123,
            "carbon_g": 5.678,
            "cost_usd": 0.4567,
        }

        vetch.ci._print_ci_summary()

        captured = capsys.readouterr()
        assert "VETCH CI SUMMARY" in captured.out
        assert "Total Inferences: 10" in captured.out
        assert "0.123 Wh" in captured.out
        assert "5.678 g CO2e" in captured.out
        assert "$0.4567" in captured.out
        assert "Efficiency Check Complete" in captured.out
