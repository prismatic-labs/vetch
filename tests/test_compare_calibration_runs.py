"""Behavioral checks for the calibration-run comparison helper."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _comparison_module():
    path = Path(__file__).parents[1] / "scripts" / "compare_calibration_runs.py"
    spec = importlib.util.spec_from_file_location("compare_calibration_runs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(model: str) -> dict:
    return {
        "identity": {
            "provider": "vllm",
            "model": model,
            "gpu": "h100-sxm-80gb",
            "serving_engine": "vllm",
            "precision": "bf16",
        },
        "provenance": {"fit": {}},
    }


def test_comparison_refuses_mixed_model_identities(tmp_path, monkeypatch, capsys):
    (tmp_path / "first.json").write_text(json.dumps(_record("model-a")))
    (tmp_path / "second.json").write_text(json.dumps(_record("model-b")))
    monkeypatch.setattr(
        sys, "argv", ["compare_calibration_runs.py", "--dir", str(tmp_path)]
    )

    assert _comparison_module().main() == 2
    output = capsys.readouterr().out
    assert "multiple calibration identities" in output
    assert "model-a" in output and "model-b" in output


def test_comparison_model_filter_selects_one_identity(tmp_path, monkeypatch, capsys):
    (tmp_path / "first.json").write_text(json.dumps(_record("model-a")))
    (tmp_path / "second.json").write_text(json.dumps(_record("model-b")))
    monkeypatch.setattr(
        sys,
        "argv",
        ["compare_calibration_runs.py", "--dir", str(tmp_path), "--model", "model-a"],
    )

    assert _comparison_module().main() == 0
    assert "1 run(s)" in capsys.readouterr().out
