"""MCP tool handlers for Vetch.

All tools are fail-open: errors return an error dict, never crash the server.
Privacy-first: no tool accepts or returns prompt/completion content.
"""

from __future__ import annotations

import functools
import os
from typing import Any, Callable, TypeVar

import vetch.calculation as _calc
from vetch.advisory import generate_advisories
from vetch.calculation import (
    calculate_carbon,
    calculate_cost,
    calculate_energy,
    calculate_water_with_provenance,
)
from vetch.health import get_health_status
from vetch.sensing.grid import get_carbon_intensity, get_cleanest_region
from vetch.stats import get_session_stats
from vetch.training import contextualize_footprint

F = TypeVar("F", bound=Callable[..., dict[str, Any]])


def _safe(fn: F) -> F:
    """Wrap a tool handler so exceptions become error dicts."""
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            return {"error": str(exc), "tool": fn.__name__}
    return wrapper  # type: ignore[return-value]


@_safe
def vetch_estimate(
    model: str,
    input_tokens: int,
    output_tokens: int,
    region: str | None = None,
    provider_hint: str | None = None,
    wue_override: float | None = None,
    wue_boundary: str | None = None,
    water_accounting_kind: str | None = None,
    wue_uncertainty_pct: float | None = None,
) -> dict[str, Any]:
    """Estimate energy, carbon, water, and cost for a single LLM inference call."""
    region = region or os.environ.get("VETCH_REGION")

    # Energy
    energy_wh, tier, uncertainty_pct, source, basis, model_known = calculate_energy(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
    )

    # Confidence from energy tier
    confidence_map = {1: "high", 2: "medium", 3: "low"}
    confidence = confidence_map.get(tier, "low")

    # Grid intensity
    grid = get_carbon_intensity(region)

    # Carbon
    carbon_g, pue, pue_tier, pue_source = calculate_carbon(
        energy_wh=energy_wh,
        grid_intensity_gco2e_kwh=grid.intensity_gco2e_kwh,
        model=model,
        provider_hint=provider_hint,
    )

    # Water. The core calculator returns liters; MCP exposes milliliters for
    # readable per-call values while retaining liters for unit clarity.
    water = calculate_water_with_provenance(
        energy_wh=energy_wh,
        model=model,
        provider_hint=provider_hint,
        region=region,
        wue_override=wue_override,
        wue_boundary=wue_boundary,
        water_accounting_kind=water_accounting_kind,
        wue_uncertainty_pct=wue_uncertainty_pct,
    )
    water_l = water.water_l
    water_ml = None if water_l is None else water_l * 1000

    # Cost
    total_cost, input_cost, output_cost, cache_write, cache_read, billing_tier = (
        calculate_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
        )
    )

    # Training context
    training_context = contextualize_footprint(carbon_g)

    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "energy_wh": round(energy_wh, 6),
        "carbon_g": round(carbon_g, 6),
        "water_l": None if water_l is None else round(water_l, 8),
        "water_ml": None if water_ml is None else round(water_ml, 4),
        "water_tier": water.tier,
        "water_basis": water.basis,
        "water_boundary": water.boundary,
        "water_accounting_kind": water.accounting,
        "water_uncertainty_pct": water.uncertainty_pct,
        "water_warnings": list(water.warnings),
        "cost_usd": round(total_cost, 6),
        "cost_breakdown": {
            "input_usd": round(input_cost, 6),
            "output_usd": round(output_cost, 6),
            "billing_tier": billing_tier,
        },
        "confidence": confidence,
        "energy_tier": tier,
        "energy_uncertainty_pct": uncertainty_pct,
        "energy_source": source,
        "grid_intensity_gco2e_kwh": grid.intensity_gco2e_kwh,
        "signal_quality": grid.signal_quality,
        "pue": pue,
        "region": region,
        "provider_hint": provider_hint,
        "training_context": training_context,
    }


@_safe
def vetch_compare(
    models: list[str],
    input_tokens: int,
    output_tokens: int,
    region: str | None = None,
    sort_by: str = "cost_usd",
    provider_hints: dict[str, str] | None = None,
    wue_override: float | None = None,
    wue_boundary: str | None = None,
    water_accounting_kind: str | None = None,
    wue_uncertainty_pct: float | None = None,
) -> dict[str, Any]:
    """Compare energy, carbon, water, and cost across multiple models."""
    results = []
    for m in models:
        est = vetch_estimate(
            model=m,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            region=region,
            provider_hint=(provider_hints or {}).get(m),
            wue_override=wue_override,
            wue_boundary=wue_boundary,
            water_accounting_kind=water_accounting_kind,
            wue_uncertainty_pct=wue_uncertainty_pct,
        )
        results.append(est)

    # Filter out errors
    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    water_definitions = {
        (
            str(r["water_boundary"]),
            str(r["water_accounting_kind"]),
        )
        for r in valid
        if r.get("water_l") is not None
        and r.get("water_boundary") is not None
        and r.get("water_accounting_kind") is not None
    }
    water_unclassified = any(
        r.get("water_l") is not None
        and (
            r.get("water_boundary") is None
            or r.get("water_accounting_kind") is None
        )
        for r in valid
    )
    water_comparable = not water_unclassified and len(water_definitions) <= 1
    comparison_warning = None
    if not water_comparable:
        comparison_warning = (
            "Water values use incompatible accounting definitions and were not ranked."
        )

    effective_sort_by: str | None = sort_by
    if sort_by in {"water_l", "water_ml"} and not water_comparable:
        effective_sort_by = None
    if valid:
        if effective_sort_by is not None:
            valid.sort(
                key=lambda r: (
                    float("inf")
                    if r.get(effective_sort_by) is None
                    else r.get(effective_sort_by, float("inf"))
                )
            )

        cheapest = min(valid, key=lambda r: r.get("cost_usd", float("inf")))
        greenest = min(valid, key=lambda r: r.get("carbon_g", float("inf")))

        for r in valid:
            r["is_cheapest"] = r["model"] == cheapest["model"]
            r["is_greenest"] = r["model"] == greenest["model"]

    return {
        "comparisons": valid,
        "errors": errors,
        "sort_by": effective_sort_by,
        "requested_sort_by": sort_by,
        "water_comparable": water_comparable,
        "water_definitions": [
            {"boundary": boundary, "accounting_kind": accounting}
            for boundary, accounting in sorted(water_definitions)
        ],
        "comparison_warning": comparison_warning,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


@_safe
def vetch_session_stats() -> dict[str, Any]:
    """Return current session statistics with advisories and training context."""
    stats = get_session_stats()
    summary = stats.summary()
    advisories = generate_advisories(stats)

    # Training context for session carbon
    training_context = contextualize_footprint(stats.total_carbon_g)

    return {
        **summary,
        "total_energy_wh": stats.total_energy_wh,
        "total_carbon_g": stats.total_carbon_g,
        "total_water_ml": summary.get("total_water_ml"),
        "water_attributed_calls": summary.get("water_attributed_calls"),
        "water_unattributed_calls": summary.get("water_unattributed_calls"),
        "total_cost_usd": stats.total_cost_usd,
        "models_used": sorted(stats.models_used),
        "advisories": [
            {
                "code": a.code,
                "severity": a.severity,
                "title": a.title,
                "description": a.description,
            }
            for a in advisories
        ],
        "training_context": training_context,
    }


@_safe
def vetch_status() -> dict[str, Any]:
    """Return Vetch health status and budget info."""
    from vetch import __version__
    from vetch.budget import get_budget_status

    health = get_health_status()
    budget = get_budget_status()

    return {
        "version": __version__,
        "health": health,
        "budget": budget,
    }


@_safe
def vetch_grid_intensity(region: str) -> dict[str, Any]:
    """Return grid carbon intensity for a region."""
    grid = get_carbon_intensity(region)
    return {
        "region": region,
        "intensity_gco2e_kwh": grid.intensity_gco2e_kwh,
        "signal_quality": grid.signal_quality,
    }


@_safe
def vetch_cleanest_region(regions: list[str]) -> dict[str, Any]:
    """Find the cleanest (lowest carbon) region from a list of candidates."""
    cleanest, intensity = get_cleanest_region(regions)
    return {
        "cleanest_region": cleanest,
        "intensity_gco2e_kwh": intensity,
        "candidates": regions,
    }


@_safe
def vetch_registry_lookup(model: str) -> dict[str, Any]:
    """Look up energy coefficients and pricing for a model."""
    _calc._load_registry()

    energy_data = (_calc._ENERGY or {}).get(model)
    pricing_data = (_calc._PRICING or {}).get(model)

    if energy_data is None and pricing_data is None:
        return {"error": f"Model '{model}' not found in registry", "model": model}

    result: dict[str, Any] = {"model": model}
    if energy_data is not None:
        result["energy"] = energy_data
    if pricing_data is not None:
        result["pricing"] = pricing_data

    return result


@_safe
def vetch_check_budget() -> dict[str, Any]:
    """Check remaining budget across all configured budgets."""
    from vetch.budget import get_budget_detail

    budgets = get_budget_detail()
    if not budgets:
        return {
            "budgets": {},
            "message": (
                "No budgets configured. "
                "Set via VETCH_BUDGET_SESSION_COST_USD or vetch.budget.set_budget()."
            ),
        }
    return {"budgets": budgets}
