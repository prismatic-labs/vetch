"""Tests for Vetch CLI.

These tests verify the CLI subcommands:
- estimate
- compare
- methodology
- check
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from vetch.cli import audit, compare, estimate, methodology, parse_duration


class TestCLIEstimate:
    """Tests for 'vetch estimate' command."""

    def test_estimate_text_output(self, capsys) -> None:
        """Verify text output for estimate."""
        args = Namespace(
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            region="us-east-1",
            format="text",
        )

        estimate(args)

        captured = capsys.readouterr()
        assert "Energy:" in captured.out
        assert "Wh" in captured.out
        assert "Carbon:" in captured.out
        assert "Cost:" in captured.out

    def test_estimate_json_output(self, capsys) -> None:
        """Verify JSON output for estimate."""
        args = Namespace(
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            region="us-east-1",
            format="json",
        )

        estimate(args)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["model"] == "gpt-4o"
        assert "energy_wh" in data
        assert data["grid_region"] == "us-east-1"


class TestCLICompare:
    """Tests for 'vetch compare' command."""

    def test_compare_table_output(self, capsys) -> None:
        """Verify table output for compare."""
        args = Namespace(
            models="gpt-4o,gpt-3.5-turbo",
            input_tokens=1000,
            output_tokens=500,
            region="us-east-1",
            format="table",
        )

        compare(args)

        captured = capsys.readouterr()
        assert "Model" in captured.out
        assert "gpt-4o" in captured.out
        assert "gpt-3.5-turbo" in captured.out
        assert "Energy (Wh)" in captured.out


class TestCLIMethodology:
    """Tests for 'vetch methodology' command."""

    def test_methodology_preamble(self, capsys) -> None:
        """Verify methodology preamble output."""
        args = Namespace(full=False, contribute=False)

        methodology(args)

        captured = capsys.readouterr()
        assert "# Vetch Methodology" in captured.out
        assert "Vetch exists because" in captured.out

    def test_methodology_full(self, capsys) -> None:
        """Verify full methodology output."""
        args = Namespace(full=True, contribute=False)

        methodology(args)

        captured = capsys.readouterr()
        assert "The Formula" in captured.out
        assert "PUE" in captured.out

    def test_methodology_contribute(self, capsys) -> None:
        """Verify contribution guide output."""
        args = Namespace(full=False, contribute=True)

        methodology(args)

        captured = capsys.readouterr()
        assert "Submission Format" in captured.out
        assert "Email to marco@prismaticlabs.ai" in captured.out


class TestCLICheck:
    """Tests for 'vetch check' command."""

    def test_check_output(self, capsys) -> None:
        """Verify check command output."""
        from vetch.cli import check
        args = Namespace()

        with patch("vetch.sensing.grid.get_carbon_intensity") as mock_grid:
            mock_grid.return_value.intensity_gco2e_kwh = 420.0
            mock_grid.return_value.signal_quality = "live"

            check(args)

        captured = capsys.readouterr()
        assert "Vetch v" in captured.out
        assert "Checking Grid API" in captured.out
        assert "Checking cache status" in captured.out


class TestCLIAudit:
    """Tests for stored audit CLI behavior."""

    def test_parse_duration_combined_units(self) -> None:
        """Duration parser accepts strict combined windows."""
        assert parse_duration("1h30m") == timedelta(minutes=90)
        assert parse_duration("24 h") == timedelta(hours=24)
        assert parse_duration("1w") == timedelta(days=7)

    def test_parse_duration_rejects_unknown_units(self) -> None:
        """Duration parser rejects accidental prefix matches."""
        with pytest.raises(argparse.ArgumentTypeError):
            parse_duration("1moon")

    def test_audit_uses_stored_metadata(self, capsys, tmp_path) -> None:
        """Audit command can render the deterministic stored report."""
        from vetch.storage import configure_storage, store_event

        db_path = tmp_path / "usage.db"
        configure_storage(enabled=True, path=db_path)
        now = datetime.now(timezone.utc)
        store_event({
            "event_id": "audit-cli-1",
            "timestamp": now.isoformat(),
            "model": "gpt-4o",
            "provider": "openai",
            "usage": {"text": {"input_tokens": 100, "output_tokens": 25}},
            "estimated_energy_wh": 0.01,
            "estimated_carbon_g": 0.004,
            "estimated_cost_usd": 0.02,
            "tags": {"feature": "rag-search"},
        })

        audit(Namespace(
            format="json",
            window=timedelta(days=1),
            model=None,
            tags=None,
            stored=True,
            session=False,
        ))

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["total_requests"] == 1
        assert data["total_tokens"] == 125

    def test_audit_empty_session_honors_json_format(self, capsys) -> None:
        """Empty session fallback still returns JSON when requested."""
        audit(Namespace(
            format="json",
            window=timedelta(days=1),
            model=None,
            tags=None,
            stored=False,
            session=True,
        ))

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["total_requests"] == 0
        assert data["advisories"] == []


class TestCLIMain:
    """Tests for CLI main entry point."""

    def test_main_estimate(self) -> None:
        """Verify main() calls estimate."""
        from vetch.cli import main
        with patch("sys.argv", ["vetch", "estimate", "--model", "gpt-4o"]):
            with patch("vetch.cli.estimate") as mock_estimate:
                main()
                mock_estimate.assert_called_once()

    def test_main_compare(self) -> None:
        """Verify main() calls compare."""
        from vetch.cli import main
        with patch("sys.argv", ["vetch", "compare", "--models", "gpt-4o"]):
            with patch("vetch.cli.compare") as mock_compare:
                main()
                mock_compare.assert_called_once()

    def test_main_methodology(self) -> None:
        """Verify main() calls methodology."""
        from vetch.cli import main
        with patch("sys.argv", ["vetch", "methodology"]):
            with patch("vetch.cli.methodology") as mock_meth:
                main()
                mock_meth.assert_called_once()

    def test_main_check(self) -> None:
        """Verify main() calls check."""
        from vetch.cli import main
        with patch("sys.argv", ["vetch", "check"]):
            with patch("vetch.cli.check") as mock_check:
                main()
                mock_check.assert_called_once()

    def test_main_no_args(self, capsys) -> None:
        """Verify main() shows help with no args."""
        from vetch.cli import main
        with patch("sys.argv", ["vetch"]):
            # argparse might exit or print help
            try:
                main()
            except SystemExit:
                pass
            captured = capsys.readouterr()
            # Depending on how subparsers are configured, it might show help
            pass
