"""MCP resource handlers for Vetch."""

from __future__ import annotations

import os
from typing import Any

import vetch.calculation as _calc
from vetch import __version__


def list_models() -> list[str]:
    """Return all model names from the energy registry."""
    _calc._load_registry()
    return sorted((_calc._ENERGY or {}).keys())


def get_energy_data(model: str) -> dict[str, Any]:
    """Return energy coefficients for a model."""
    _calc._load_registry()
    data = (_calc._ENERGY or {}).get(model)
    if data is None:
        return {"error": f"Model '{model}' not found in energy registry"}
    return {"model": model, **data}


def get_pricing_data(model: str) -> dict[str, Any]:
    """Return pricing data for a model."""
    _calc._load_registry()
    data = (_calc._PRICING or {}).get(model)
    if data is None:
        return {"error": f"Model '{model}' not found in pricing registry"}
    return {"model": model, **data}


def get_config() -> dict[str, Any]:
    """Return current Vetch configuration."""
    return {
        "region": os.environ.get("VETCH_REGION", "(not set)"),
        "output": os.environ.get("VETCH_OUTPUT", "stderr"),
        "default_pue": float(os.environ.get("VETCH_DEFAULT_PUE", "1.2")),
        "cache_mode": os.environ.get("VETCH_CACHE_MODE", "file"),
    }


def get_version() -> str:
    """Return Vetch version string."""
    return __version__
