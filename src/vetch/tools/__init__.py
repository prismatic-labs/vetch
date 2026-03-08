"""Vetch Tools - Utilities for energy calibration, validation, and analysis.

This package provides user-facing utilities for working with energy data:

- calibration: Reverse-engineer per-token energy from aggregate measurements
- validation: Validate energy/pricing data in the registry (future)
- reporting: Generate carbon reports (future)
- analysis: Analyze historical trends (future)

Example:
    >>> from vetch.tools.calibration import calibrate_from_measurement
    >>> result = calibrate_from_measurement(
    ...     measurement_wh=0.24,
    ...     pue=1.10,
    ...     median_tokens=800,
    ... )
    >>> print(f"Input: {result.energy_per_1k_input:.3f} Wh/1k")
"""

from __future__ import annotations

__all__ = ["calibration"]
