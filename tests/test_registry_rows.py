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
    ("gpt-5.6-sol", 0.005, 0.03),
    ("gpt-5.6-terra", 0.002, 0.012),
    ("gpt-5.6-luna", 0.0002, 0.0012),
    ("gemini-3.6-flash", 0.0015, 0.0075),
    ("gemma-4-31b-it", 0.0, 0.0),  # self-hosted: no list price (cost via provider)
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

    @pytest.mark.parametrize("model", ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"])
    def test_gpt_5_6_reasoning_and_context_tier(self, model: str) -> None:
        from vetch.calculation import _is_reasoning_compute_model

        assert _is_reasoning_compute_model(model) is True
        # >272k input selects the long-context rates for the full request:
        # input at 2x and output at 1.5x.
        short = calculate_cost(100_000, 1000, model)
        long = calculate_cost(300_000, 1000, model)
        short_in = short[1] / 100_000
        long_in = long[1] / 300_000
        assert long_in == pytest.approx(short_in * 2.0)
        assert long[2] == pytest.approx(short[2] * 1.5)

    def test_gpt_5_6_long_context_cache_uses_long_input_rate(self) -> None:
        _, _, _, cache_write, cache_read, _ = calculate_cost(
            300_000,
            1000,
            "gpt-5.6-sol",
            cache_read_tokens=100_000,
            cache_creation_tokens=100_000,
        )
        assert cache_read == pytest.approx(0.1)  # 100k × $10/M × 0.1
        assert cache_write == pytest.approx(1.25)  # 100k × $10/M × 1.25

    def test_gpt_5_6_public_alias_routes_to_sol(self) -> None:
        match = resolve_model_match("gpt-5.6")
        assert match.name == "gpt-5.6-sol"
        assert match.precision == "alias"

    def test_gemini_3_6_flash_preview_alias(self) -> None:
        match = resolve_model_match("gemini-3.6-flash-preview")
        assert match.name == "gemini-3.6-flash"
        assert match.precision == "alias"

    def test_undisclosed_hosted_model_metadata_is_not_invented(self) -> None:
        energy = _load("energy.json")
        for model in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
            assert "quantization" not in energy[model]

        gemini = energy["gemini-3.6-flash"]
        for field in ("architecture", "total_params_b", "active_params_b", "quantization"):
            assert field not in gemini
        assert "discloses no Wh/token" in gemini["basis"]

    def test_gemma_zero_price_excludes_infrastructure_cost(self) -> None:
        pricing = _load("pricing.json")["gemma-4-31b-it"]
        assert pricing["cost_scope"] == "model_license_only"
        assert "infrastructure and electricity costs are excluded" in pricing["notes"]

    def test_gemma_4_31b_it_energy_is_own_row_not_cloud_proxy(self) -> None:
        # A self-hosted 30.7B dense model must use its own standardized proxy,
        # not the Gemini large-cloud proxy it fell back to without a row.
        gemma, *_ = calculate_energy(1000, 500, "gemma-4-31b-it")
        pro, *_ = calculate_energy(1000, 500, "gemini-3.1-pro")
        assert gemma < pro


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
