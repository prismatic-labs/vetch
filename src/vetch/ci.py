"""CI detection and automatic summary reporting.

If Vetch detects it is running in a CI environment (GitHub Actions, CI=true),
it will automatically aggregate stats and print a summary to stdout at exit.

Note: This module is EXPERIMENTAL and may change in future versions.
"""

from __future__ import annotations

import os
import atexit
import sys
import warnings
from typing import Any

# P2: Mark as experimental
warnings.warn(
    "vetch.ci is experimental and may change in future versions",
    FutureWarning,
    stacklevel=2,
)

# Global counters for CI summary
_CI_STATS = {
    "count": 0,
    "energy_wh": 0.0,
    "carbon_g": 0.0,
    "cost_usd": 0.0,
}

def is_ci() -> bool:
    """Check if running in a CI environment."""
    return os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true"

def track_ci_event(event: dict[str, Any]) -> None:
    """Update CI stats with a new event."""
    if not is_ci():
        return
    
    _CI_STATS["count"] += 1
    _CI_STATS["energy_wh"] += event.get("estimated_energy_wh") or 0.0
    _CI_STATS["carbon_g"] += event.get("estimated_carbon_g") or 0.0
    _CI_STATS["cost_usd"] += event.get("estimated_cost_usd") or 0.0

def _print_ci_summary() -> None:
    """Print ASCII summary table to stdout."""
    if _CI_STATS["count"] == 0:
        return

    print("\n" + "=" * 40)
    print("VETCH CI SUMMARY (ALPHA)")
    print("-" * 40)
    print(f"Total Inferences: {_CI_STATS['count']}")
    print(f"Total Energy:     {_CI_STATS['energy_wh']:.3f} Wh")
    print(f"Total Carbon:     {_CI_STATS['carbon_g']:.3f} g CO2e")
    print(f"Estimated Cost:   ${_CI_STATS['cost_usd']:.4f}")
    print("-" * 40)
    print("Efficiency Check Complete.")
    print("=" * 40 + "\n")

# Register cleanup handler
if is_ci():
    atexit.register(_print_ci_summary)
