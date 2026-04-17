"""Training emissions context for inference footprint comparison.

Provides reference training emissions data (from Stanford AI Index 2026 /
Epoch AI) and a function to express inference carbon footprints as a
fraction of known training costs.

Data sources (in priority order):
1. ``VETCH_TRAINING_REGISTRY_PATH`` env var — custom / air-gapped file
2. Bundled ``registry/training_emissions.json``

The registry is lazy-loaded and cached for the process lifetime.  Call
``_reset_training_data()`` to force a reload (for test isolation or
after updating the env var at runtime).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TRAINING_PATH = Path(__file__).parent / "registry" / "training_emissions.json"
_TRAINING_DATA: dict[str, Any] | None = None


def _load_training_data() -> dict[str, Any]:
    """Load training emissions, respecting env-var override."""
    global _TRAINING_DATA
    if _TRAINING_DATA is not None:
        return _TRAINING_DATA

    # 1. Check for user-supplied override
    override_path = os.environ.get("VETCH_TRAINING_REGISTRY_PATH")
    if override_path:
        try:
            data = json.loads(Path(override_path).read_text())
            logger.debug("Loaded training emissions from %s", override_path)
            _TRAINING_DATA = data
            return _TRAINING_DATA
        except Exception as exc:
            logger.warning(
                "Failed to load training emissions from %s: %s — "
                "falling back to bundled data",
                override_path,
                exc,
            )

    # 2. Bundled default
    _TRAINING_DATA = json.loads(_TRAINING_PATH.read_text())
    return _TRAINING_DATA


def _reset_training_data() -> None:
    """Reset cached data (for test isolation)."""
    global _TRAINING_DATA
    _TRAINING_DATA = None


def get_training_emissions() -> dict[str, dict[str, Any]]:
    """Return training emissions registry (model -> {co2e_tonnes, year}).

    Filters out metadata keys (starting with _).
    """
    data = _load_training_data()
    return {k: v for k, v in data.items() if not k.startswith("_")}


def contextualize_footprint(carbon_g: float) -> list[dict[str, Any]]:
    """Compare an inference carbon footprint against known training costs.

    Args:
        carbon_g: Inference carbon footprint in grams CO2e.

    Returns:
        List of dicts sorted by training cost descending, each with:
        - model: training model name
        - training_co2e_tonnes: training emissions
        - fraction: inference as fraction of training
        - description: human-readable comparison string
    """
    data = get_training_emissions()
    carbon_tonnes = carbon_g / 1_000_000  # grams -> tonnes

    comparisons: list[dict[str, Any]] = []
    for model, info in sorted(
        data.items(), key=lambda x: x[1]["co2e_tonnes"], reverse=True
    ):
        training_tonnes = info["co2e_tonnes"]
        fraction = carbon_tonnes / training_tonnes if training_tonnes > 0 else 0.0
        pct = fraction * 100

        if pct < 0.001:
            desc = f"equivalent to {pct:.6f}% of {model} training"
        elif pct < 1:
            desc = f"equivalent to {pct:.4f}% of {model} training"
        else:
            desc = f"equivalent to {pct:.2f}% of {model} training"

        comparisons.append({
            "model": model,
            "training_co2e_tonnes": training_tonnes,
            "fraction": round(fraction, 10),
            "description": desc,
        })

    return comparisons
