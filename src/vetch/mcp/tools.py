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
    calculate_water,
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
    )

    # Water. The core calculator returns liters; MCP exposes milliliters for
    # readable per-call values while retaining liters for unit clarity.
    water_l = calculate_water(
        energy_wh=energy_wh,
        model=model,
        region=region,
    )
    water_ml = water_l * 1000

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
        "water_l": round(water_l, 8),
        "water_ml": round(water_ml, 4),
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
        "training_context": training_context,
    }


@_safe
def vetch_compare(
    models: list[str],
    input_tokens: int,
    output_tokens: int,
    region: str | None = None,
    sort_by: str = "cost_usd",
) -> dict[str, Any]:
    """Compare energy, carbon, water, and cost across multiple models."""
    results = []
    for m in models:
        est = vetch_estimate(
            model=m,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            region=region,
        )
        results.append(est)

    # Filter out errors
    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    if valid:
        valid.sort(key=lambda r: r.get(sort_by, float("inf")))

        cheapest = min(valid, key=lambda r: r.get("cost_usd", float("inf")))
        greenest = min(valid, key=lambda r: r.get("carbon_g", float("inf")))

        for r in valid:
            r["is_cheapest"] = r["model"] == cheapest["model"]
            r["is_greenest"] = r["model"] == greenest["model"]

    return {
        "comparisons": valid,
        "errors": errors,
        "sort_by": sort_by,
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
        "total_water_ml": stats.total_water_ml,
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
