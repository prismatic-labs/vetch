"""Parity tests: docs/calculator/index.html JS vs vetch.calculation SDK."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from vetch.calculation import (
    calculate_carbon,
    calculate_cost,
    calculate_energy,
    calculate_water,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CALCULATOR_HTML = REPO_ROOT / "docs/calculator/index.html"
GLOBAL_AVERAGES = REPO_ROOT / "src/vetch/sensing/global_averages.json"

REL_TOL = 1e-4
ABS_TOL = 1e-6

# Generated from SDK calls — do not hand-edit expected values.
GOLDEN_CASES: list[tuple[str, int, int, str]] = [
    ("gpt-4o", 500, 200, "global"),           # short input bucket
    ("gpt-4o", 800, 500, "global"),           # short input (<1000) despite high output
    ("gpt-4o", 1000, 500, "global"),          # medium input bucket
    ("gpt-4o", 6000, 500, "global"),          # long input bucket
    ("gpt-4", 6000, 500, "global"),           # long requested, medium fallback
    ("claude-3.7-sonnet", 1000, 500, "global"),
    ("gemini-2.5-pro", 1000, 500, "global"),
    ("gemini-2.5-pro", 300000, 1000, "global"),  # tiered threshold pricing
    ("llama-3.1-70b", 1000, 500, "global"),   # zero-cost open weight
    ("mixtral-8x7b", 1000, 500, "global"),
    ("deepseek-r1", 1000, 500, "global"),
    ("gpt-4.1-mini", 100, 50, "global"),
    ("gemini-2.5-pro", 1000, 500, "eu-west-2"),  # regional grid
]


def _grid_intensity(region_key: str) -> float:
    data = json.loads(GLOBAL_AVERAGES.read_text())
    if region_key == "global":
        return float(data["global"])
    return float(data["regions"][region_key])


def _sdk_metrics(model: str, input_tokens: int, output_tokens: int, region_key: str) -> dict:
    grid_gco2 = _grid_intensity(region_key)
    energy_wh, *_ = calculate_energy(input_tokens, output_tokens, model)
    carbon_g, pue, *_ = calculate_carbon(energy_wh, grid_gco2, model=model)
    cost_usd, *_ = calculate_cost(input_tokens, output_tokens, model)
    water_l = calculate_water(energy_wh, model=model)
    return {
        "energy_it_wh": energy_wh,
        "pue": pue,
        "carbon_g": carbon_g,
        "cost_usd": cost_usd,
        "water_ml": None if water_l is None else water_l * 1000.0,
    }


def _extract_calculator_script() -> str:
    html = CALCULATOR_HTML.read_text()
    scripts = re.findall(r"<script>(.*?)</script>", html, flags=re.DOTALL)
    calc_scripts = [s for s in scripts if "const MODELS" in s and "calcMetrics" in s]
    assert len(calc_scripts) == 1, "Expected exactly one calculator script block"
    return calc_scripts[0].strip()


def _js_metrics(
    model: str,
    input_tokens: int,
    output_tokens: int,
    grid_gco2: float,
    water_host: str = "auto",
    water_region: str = "global",
) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")

    script = _extract_calculator_script()
    runner = (
        script
        + "\nconst m = calcMetrics("
        + json.dumps(model)
        + ", "
        + str(input_tokens)
        + ", "
        + str(output_tokens)
        + ", "
        + str(grid_gco2)
        + ", "
        + json.dumps(water_host)
        + ", "
        + json.dumps(water_region)
        + ");\n"
        + "console.log(JSON.stringify(m));\n"
    )
    result = subprocess.run(
        [node, "-e", runner],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Node calculator failed:\nstdout={result.stdout}\nstderr={result.stderr}"
        )
    return json.loads(result.stdout.strip())


def _assert_close(actual: object, expected: object, label: str) -> None:
    if expected is None:
        assert actual is None, f"{label}: expected null, got {actual!r}"
        return
    if actual is None:
        raise AssertionError(f"{label}: expected {expected!r}, got null")
    actual_f = float(actual)  # type: ignore[arg-type]
    expected_f = float(expected)  # type: ignore[arg-type]
    if expected_f == 0.0:
        assert actual_f == pytest.approx(expected_f, abs=ABS_TOL), label
        return
    assert actual_f == pytest.approx(expected_f, rel=REL_TOL, abs=ABS_TOL), label


@pytest.mark.parametrize(
    "model,input_tokens,output_tokens,region_key",
    GOLDEN_CASES,
    ids=[f"{m}-{i}in-{o}out-{r}" for m, i, o, r in GOLDEN_CASES],
)
def test_calculator_matches_sdk(
    model: str,
    input_tokens: int,
    output_tokens: int,
    region_key: str,
) -> None:
    grid_gco2 = _grid_intensity(region_key)
    sdk = _sdk_metrics(model, input_tokens, output_tokens, region_key)
    js = _js_metrics(model, input_tokens, output_tokens, grid_gco2)

    for key in ("energy_it_wh", "pue", "carbon_g", "cost_usd", "water_ml"):
        _assert_close(js[key], sdk[key], f"{model} {key}")


def test_calculator_host_selector_reaches_aws_region() -> None:
    energy_wh, *_ = calculate_energy(1000, 500, "llama-3.1-70b")
    expected_l = calculate_water(
        energy_wh,
        model="llama-3.1-70b",
        provider_hint="aws",
        region="us-east-1",
    )
    js = _js_metrics(
        "llama-3.1-70b",
        1000,
        500,
        _grid_intensity("us-east-1"),
        water_host="aws",
        water_region="us-east-1",
    )

    assert expected_l is not None
    assert js["water_ml"] == pytest.approx(expected_l * 1000)
    assert js["water_accounting_kind"] == "withdrawal"
