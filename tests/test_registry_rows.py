"""Fixture tests for the v0.9.0 registry rows (Gemini 3.x, current Claude).

Guards the verified pricing and the proxied energy tiers added in B1, plus the
energy<->pricing parity invariant the CI freshness check enforces.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vetch.calculation import calculate_cost, calculate_energy, resolve_model_match

_REGISTRY = Path(__file__).resolve().parent.parent / "src" / "vetch" / "registry"


def _load(name: str) -> dict:
    return {k: v for k, v in json.loads((_REGISTRY / name).read_text()).items()
            if not k.startswith("_")}


# (model, expected usd_per_1k_input, expected usd_per_1k_output)
NEW_PRICED_ROWS = [
    ("gemini-3-flash", 0.0005, 0.003),
    ("gemini-3.5-flash", 0.0015, 0.009),
    ("gemini-3.1-flash-lite", 0.00025, 0.0015),
    ("gemini-3.1-pro", 0.002, 0.012),
    ("claude-sonnet-4-5", 0.003, 0.015),
    ("claude-sonnet-4-6", 0.003, 0.015),
    ("claude-opus-4-5", 0.005, 0.025),
    ("claude-opus-4-6", 0.005, 0.025),
    ("claude-opus-4-7", 0.005, 0.025),
    ("claude-opus-4-8", 0.005, 0.025),
]


class TestNewRegistryRows:
    """Each new model resolves exactly and carries verified pricing."""

    @pytest.mark.parametrize("model,exp_in,exp_out", NEW_PRICED_ROWS)
    def test_resolves_exact(self, model: str, exp_in: float, exp_out: float) -> None:
        match = resolve_model_match(model)
        assert match.name == model
        assert match.known is True
        assert match.precision == "exact"

    @pytest.mark.parametrize("model,exp_in,exp_out", NEW_PRICED_ROWS)
    def test_pricing_matches_verified(self, model: str, exp_in: float, exp_out: float) -> None:
        pricing = _load("pricing.json")
        assert model in pricing, f"{model} missing from pricing.json"
        assert pricing[model]["usd_per_1k_input"] == exp_in
        assert pricing[model]["usd_per_1k_output"] == exp_out

    @pytest.mark.parametrize("model,exp_in,exp_out", NEW_PRICED_ROWS)
    def test_energy_present_and_tier3(self, model: str, exp_in: float, exp_out: float) -> None:
        # All new rows are honest Tier-3 proxies (no empirical measurement yet).
        _, tier, _, source, _, known = calculate_energy(1000, 500, model)
        assert known is True
        assert source == "registry"
        assert tier == 3

    @pytest.mark.parametrize("model,exp_in,exp_out", NEW_PRICED_ROWS)
    def test_cost_uses_verified_price(self, model: str, exp_in: float, exp_out: float) -> None:
        # 1000 input + 500 output, under any tiered threshold -> exact list price.
        total, cost_in, cost_out, *_ = calculate_cost(1000, 500, model)
        assert cost_in == pytest.approx(exp_in)
        assert cost_out == pytest.approx(exp_out * 0.5)

    def test_gemini_3_1_pro_tiered_pricing(self) -> None:
        # 300k input crosses the 200k threshold -> input billed at 2x.
        _, cost_in, _, *_ = calculate_cost(300_000, 1000, "gemini-3.1-pro")
        assert cost_in == pytest.approx(300 * 0.002 * 2.0)

    def test_gemini_3_flash_preview_alias(self) -> None:
        match = resolve_model_match("gemini-3-flash-preview")
        assert match.name == "gemini-3-flash"
        assert match.precision == "alias"

    def test_claude_sonnet_4_6_dated_alias(self) -> None:
        match = resolve_model_match("claude-sonnet-4-6-latest")
        assert match.name == "claude-sonnet-4-6"
        assert match.precision == "alias"


class TestRegistryParity:
    """energy.json and pricing.json must stay in lockstep (CI also enforces)."""

    def test_energy_pricing_key_parity(self) -> None:
        energy = set(_load("energy.json"))
        pricing = set(_load("pricing.json"))
        assert energy == pricing, (
            f"energy-only: {sorted(energy - pricing)}; "
            f"pricing-only: {sorted(pricing - energy)}"
        )

    def test_alias_targets_exist(self) -> None:
        energy = _load("energy.json")
        aliases = _load("aliases.json")
        missing = sorted({v for v in aliases.values() if v not in energy})
        assert not missing, f"alias targets missing from energy.json: {missing}"
