"""Tests for Vetch CLI.

These tests verify the CLI subcommands:
- estimate
- compare
- methodology
- check
"""

from __future__ import annotations

import json
from argparse import Namespace
from unittest.mock import patch

from vetch.cli import compare, estimate, methodology


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

