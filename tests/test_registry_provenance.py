"""Provenance invariants for coefficient registries.

The rule these encode: a coefficient that reaches a user either cites a source
or is absent. The WUE registry learned this the hard way — an unsourced 1.8
L/kWh default produced a water number for every call, including providers that
publish nothing. These tests keep that from returning, here and in PUE.

The guard itself lives in scripts/check_registry_provenance.py so CI can run it
without pytest; this module asserts the same invariants at unit level and pins
the behaviour that the guard exists to protect.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from vetch.calculation import (
    PROVIDER_PUE,
    PROVIDER_PUE_AS_OF,
    PROVIDER_PUE_SOURCES,
    PROVIDER_PUE_TIER,
    get_provider_pue,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_provenance_guard_passes() -> None:
    """The shipped registries satisfy the guard CI runs."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_registry_provenance.py")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_no_default_wue_constant() -> None:
    """Water has no default coefficient. Its spread is too wide for one."""
    import vetch.calculation as calc

    assert not hasattr(calc, "DEFAULT_WUE")
    assert not hasattr(calc, "PROVIDER_WUE")


@pytest.mark.parametrize("provider", sorted(PROVIDER_PUE))
def test_every_pue_entry_is_classified(provider: str) -> None:
    """No PUE value ships without a source, a date, and an attribution tier."""
    assert PROVIDER_PUE_SOURCES.get(provider)
    assert provider in PROVIDER_PUE_AS_OF
    assert PROVIDER_PUE_TIER.get(provider) in (1, 2)


@pytest.mark.parametrize("provider", sorted(PROVIDER_PUE))
def test_pue_tier_agrees_with_its_source_text(provider: str) -> None:
    """Tier 2 names its assumption; tier 1 does not carry one."""
    names_assumption = "assum" in PROVIDER_PUE_SOURCES[provider].lower()
    assert names_assumption == (PROVIDER_PUE_TIER[provider] == 2)


def test_undisclosed_hosting_is_tier_2_not_tier_1() -> None:
    """OpenAI and Anthropic do not publish where they serve.

    The PUE figure applied to them is real (Microsoft's, AWS's); the claim that
    it applies is an assumption. Water already refuses to guess the host, so
    carbon must not silently claim to know it.
    """
    for model in ("gpt-4o", "claude-sonnet-4-5"):
        _, tier, source = get_provider_pue(model)
        assert tier == 2, f"{model} claims a disclosed host"
        assert "assum" in source.lower(), f"{model} hides its assumption"


def test_operator_run_services_stay_tier_1() -> None:
    """Google serving Gemini is a disclosed attribution, not an assumption."""
    for model in ("gemini-2.5-pro",):
        _, tier, _ = get_provider_pue(model)
        assert tier == 1


def test_open_weight_model_name_does_not_imply_an_operator() -> None:
    """A Llama or Gemma checkpoint can be served anywhere.

    Resolving a datacenter operator from an open-weight model name is an
    assumption even when the resolved operator publishes its PUE.
    """
    for model in ("llama-3.1-70b", "mixtral-8x7b", "gemma-4-31b-it"):
        _, tier, source = get_provider_pue(model)
        assert tier == 2, f"{model} claims a disclosed host"
        assert "open-weight" in source


def test_explicit_provider_hint_is_trusted() -> None:
    """A caller naming its own operator is not an inference by Vetch."""
    _, tier, _ = get_provider_pue(model=None, provider_hint="aws")
    assert tier == 1


def test_unknown_model_falls_back_to_documented_default() -> None:
    """The default is tier 3 and says it resolved no operator."""
    pue, tier, source = get_provider_pue("some-unreleased-model")
    assert tier == 3
    assert pue == 1.2
    assert "no operator resolved" in source
