"""Tests for new CLI commands: status, dashboard, registry."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from vetch.cli import main


class TestStatusCommand:
    """Tests for vetch status command."""

    def test_status_outputs_version(self, capsys: pytest.CaptureFixture[str]) -> None:
        """status command shows version."""
        with patch("sys.argv", ["vetch", "status"]):
            main()

        output = capsys.readouterr().out
        assert "Vetch v" in output

    def test_status_shows_registry_section(self, capsys: pytest.CaptureFixture[str]) -> None:
        """status command shows registry section."""
        with patch("sys.argv", ["vetch", "status"]):
            main()

        output = capsys.readouterr().out
        assert "Registry:" in output

    def test_status_shows_providers(self, capsys: pytest.CaptureFixture[str]) -> None:
        """status command shows providers section."""
        with patch("sys.argv", ["vetch", "status"]):
            main()

        output = capsys.readouterr().out
        assert "Providers:" in output

    def test_status_shows_config(self, capsys: pytest.CaptureFixture[str]) -> None:
        """status command shows config section."""
        with patch("sys.argv", ["vetch", "status"]):
            main()

        output = capsys.readouterr().out
        assert "Config:" in output
        assert "VETCH_REGION" in output

    def test_status_shows_budgets(self, capsys: pytest.CaptureFixture[str]) -> None:
        """status command shows budgets section."""
        with patch("sys.argv", ["vetch", "status"]):
            main()

        output = capsys.readouterr().out
        assert "Budgets:" in output


class TestDashboardCommand:
    """Tests for vetch dashboard command."""

    def test_dashboard_list(self, capsys: pytest.CaptureFixture[str]) -> None:
        """dashboard --list shows available templates."""
        with patch("sys.argv", ["vetch", "dashboard", "--list"]):
            main()

        output = capsys.readouterr().out
        assert "Available dashboard templates:" in output

    def test_dashboard_export_grafana_to_stdout(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """dashboard --export grafana outputs JSON."""
        with patch("sys.argv", ["vetch", "dashboard", "--export", "grafana"]):
            main()

        output = capsys.readouterr().out
        # Should be valid JSON (Grafana dashboard)
        data = json.loads(output)
        assert isinstance(data, dict)

    def test_dashboard_export_to_file(self) -> None:
        """dashboard --export grafana --output writes to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "dashboard.json"

            with patch(
                "sys.argv",
                ["vetch", "dashboard", "--export", "grafana", "--output", str(output_path)],
            ):
                main()

            assert output_path.exists()
            data = json.loads(output_path.read_text())
            assert isinstance(data, dict)

    def test_dashboard_unknown_type(self) -> None:
        """dashboard with unknown type exits with error."""
        with patch("sys.argv", ["vetch", "dashboard", "--export", "unknown"]):
            with pytest.raises(SystemExit):
                main()


class TestRegistryFreezeCommand:
    """Tests for vetch registry freeze command."""

    def test_registry_freeze_calls_freeze(self) -> None:
        """registry freeze calls freeze_registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "frozen.json"

            with (
                patch("sys.argv", ["vetch", "registry", "freeze", "--output", str(output)]),
                patch(
                    "vetch.registry.remote.freeze_registry", return_value=True
                ) as mock_freeze,
            ):
                main()

            mock_freeze.assert_called_once_with(str(output))

    def test_registry_freeze_error_exits(self) -> None:
        """registry freeze exits with error on failure."""
        with (
            patch("sys.argv", ["vetch", "registry", "freeze"]),
            patch("vetch.registry.remote.freeze_registry", return_value=False),
            pytest.raises(SystemExit),
        ):
            main()
