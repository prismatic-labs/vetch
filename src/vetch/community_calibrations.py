"""Bundled community Apple Silicon calibration records (data/calibrations.json)."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from importlib import resources
from typing import Any

from vetch.calibrate import CalibrationResult, calibration_model_variants

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_community_records() -> list[dict[str, Any]]:
    try:
        data_path = resources.files("vetch.data").joinpath("community_calibrations.json")
        raw = data_path.read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("community_calibrations.json is invalid JSON")
        return []
    if not isinstance(parsed, list):
        return []
    return [r for r in parsed if isinstance(r, dict)]


def _chip_family_hint() -> str | None:
    try:
        from vetch.calibrate_metal import get_hardware_info

        return get_hardware_info().chip_family
    except Exception:
        return None


def lookup_community_calibration(provider: str, model: str) -> CalibrationResult | None:
    """Best-effort community prior when no ~/.vetch file exists (Tier 0, higher uncertainty)."""
    records = _load_community_records()
    if not records:
        return None

    chip_family = _chip_family_hint()
    variants = set(calibration_model_variants(model))

    for record in records:
        if record.get("provider", "ollama") != provider:
            continue
        rec_model = record.get("model")
        if not isinstance(rec_model, str) or rec_model not in variants:
            continue
        hw = record.get("hardware") or {}
        rec_family = hw.get("chip_family") if isinstance(hw, dict) else None
        if chip_family and rec_family and rec_family != chip_family:
            continue
        try:
            return CalibrationResult(
                model=model,
                provider=provider,
                wh_per_1k_input=float(record["wh_per_1k_input"]),
                wh_per_1k_output=float(record["wh_per_1k_output"]),
                tier=0,
                samples=int(record.get("samples") or 0),
                gpu_name=(
                    hw.get("chip_raw") if isinstance(hw, dict) else None
                ),
                wh_per_image=record.get("wh_per_image"),
                visual_tokens_per_image=record.get("visual_tokens_per_image"),
                intercept_wh=record.get("intercept_wh"),
                active=True,
                origin="community",
            )
        except (KeyError, TypeError, ValueError):
            continue
    return None
