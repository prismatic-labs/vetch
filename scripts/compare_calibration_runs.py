#!/usr/bin/env python3
"""Compare repeated calibration runs of one identity.

A calibration is a measurement, and a single measurement is a claim rather than
a result. This prints every run of an identity side by side -- the current
record plus everything the archive kept when later runs overwrote it -- so a
threshold question can be answered from a spread rather than from one number.

    python3 scripts/compare_calibration_runs.py
    python3 scripts/compare_calibration_runs.py --dir ~/.vetch/calibrations
"""

from __future__ import annotations

import argparse
import datetime
import json
import statistics
from pathlib import Path
from typing import Any

from vetch.calibrate_v2 import (
    MAX_IDLE_UNCERTAINTY_ROW_SHARE_PCT,
    idle_uncertainty_row_share_pct,
)


def _load(path: Path) -> dict[str, Any] | None:
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return record if isinstance(record, dict) and record.get("provenance") else None


def _precision(fit: dict[str, Any], value: float | None, key: str) -> float | None:
    """Bootstrap CI half-width as a percentage of the estimate."""
    ci = fit.get(key)
    if not ci:
        return None
    mid = value if value is not None else (ci[0] + ci[1]) / 2
    if not mid:
        return None
    return 100 * ((ci[1] - ci[0]) / 2) / abs(mid)


def _num(value: Any, default: float = float("nan")) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def _identity_key(record: dict[str, Any]) -> tuple[Any, ...]:
    identity = record.get("identity") or {}
    host = record.get("host") or {}
    conditions = record.get("measurement_conditions") or {}
    serving_config = record.get("serving_config") or {}
    return tuple(
        [
            *(identity.get(key) for key in (
                "provider", "model", "gpu", "serving_engine", "precision",
                "visual_token_budget", "concurrency",
            )),
            *(host.get(key) for key in (
                "infrastructure_provider", "site", "tenancy", "gpu_form_factor",
                "instance_type",
            )),
            conditions.get("enforced_power_limit_w"),
            json.dumps(serving_config, sort_keys=True, separators=(",", ":")),
        ]
    )


def _identity_label(key: tuple[Any, ...]) -> str:
    (
        provider, model, gpu, engine, precision, visual_budget, concurrency,
        infra, site, tenancy, form_factor, instance_type, power_cap, serving,
    ) = key
    return (
        f"provider={provider!r}, model={model!r}, gpu={gpu!r}, "
        f"serving_engine={engine!r}, precision={precision!r}, "
        f"power_cap_w={power_cap!r}, host={infra!r}/{site!r}/{tenancy!r}, "
        f"instance_type={instance_type!r}, serving_config={serving!r}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="~/.vetch/calibrations",
                    help="calibration directory (default: ~/.vetch/calibrations)")
    ap.add_argument("--model", help="model identity to compare")
    ap.add_argument("--provider", help="provider identity to compare")
    args = ap.parse_args()

    root = Path(args.dir).expanduser()
    paths = sorted(
        list(root.glob("*.json")) + list((root / "archive").glob("*.json")),
        key=lambda p: p.stat().st_mtime,
    )
    runs = [(p, rec) for p in paths if (rec := _load(p)) is not None]
    if not runs:
        print(f"No calibration records found under {root}")
        return 1

    if args.model:
        wanted = args.model.casefold()
        runs = [
            (path, record) for path, record in runs
            if str((record.get("identity") or {}).get("model", "")).casefold() == wanted
        ]
    if args.provider:
        wanted = args.provider.casefold()
        runs = [
            (path, record) for path, record in runs
            if str((record.get("identity") or {}).get("provider", "")).casefold() == wanted
        ]
    if not runs:
        print("No calibration records matched the requested identity.")
        return 1

    identities = {_identity_key(record) for _path, record in runs}
    if len(identities) != 1:
        print(
            "Refusing to combine multiple calibration identities. "
            "Use --model and optionally --provider to select one:\n"
            + "\n".join(
                f"  - {_identity_label(key)}"
                for key in sorted(identities, key=str)
            )
        )
        return 2

    print(f"{len(runs)} run(s) under {root}\n")
    print(f"{'#':<3} {'when':<13} {'idle_b':<7} {'idle_a':<7} {'drift%':<7} "
          f"{'share%':<7} {'rows':<5} {'r2':<7} {'decode Wh/1k':<15} {'+-%':<6} "
          f"{'prefill+-%':<11} {'active'}")

    decode_vals: list[float] = []
    drifts: list[float] = []
    shares: list[float] = []
    for i, (path, rec) in enumerate(runs, 1):
        prov = rec["provenance"]
        fit = prov.get("fit") or {}
        coef = rec.get("coefficients") or {}
        decode = coef.get("wh_per_1k_output")
        decode_pct = _precision(fit, decode, "output_ci95")
        prefill_pct = _precision(fit, coef.get("wh_per_1k_input"), "input_ci95")
        drift = _num(prov.get("idle_drift_pct"))
        # The gate is the energy share, not the raw drift percentage: a
        # percentage of idle is scale-free and says nothing about whether the
        # coefficients are threatened. Drift stays as a displayed diagnostic.
        share = idle_uncertainty_row_share_pct(
            prov.get("raw_run_table") or [],
            prov.get("idle_watts_before"), prov.get("idle_watts_after"),
        )
        if share is not None:
            shares.append(share)
        stamp = datetime.datetime.fromtimestamp(
            path.stat().st_mtime
        ).strftime("%m-%d %H:%M")
        if decode is not None:
            decode_vals.append(float(decode))
        if drift == drift:  # not NaN
            drifts.append(drift)
        shown = f"{decode:.6f}" if decode is not None else "not identified"
        dp = f"{decode_pct:.1f}" if decode_pct is not None else "-"
        pp = f"{prefill_pct:.1f}" if prefill_pct is not None else "-"
        sh = f"{share:.2f}" if share is not None else "-"
        print(f"{i:<3} {stamp:<13} {_num(prov.get('idle_watts_before')):<7.3f} "
              f"{_num(prov.get('idle_watts_after')):<7.3f} {drift:<7.1f} "
              f"{sh:<7} "
              f"{len(prov.get('raw_run_table') or []):<5} "
              f"{_num(fit.get('r2')):<7.4f} {shown:<15} {dp:<6} {pp:<11} "
              f"{rec.get('active')}")

    print()
    if len(decode_vals) >= 2:
        spread = (max(decode_vals) - min(decode_vals)) / statistics.mean(decode_vals)
        print(f"decode identified in {len(decode_vals)}/{len(runs)} run(s); "
              f"spread across them: {100 * spread:.2f}%")
    if len(drifts) >= 2:
        print(f"idle drift (diagnostic only): min {min(drifts):.1f}%  "
              f"median {statistics.median(drifts):.1f}%  max {max(drifts):.1f}%")
    if shares:
        over = sum(1 for x in shares if x > MAX_IDLE_UNCERTAINTY_ROW_SHARE_PCT)
        print(f"idle uncertainty as a share of a typical row's energy: "
              f"min {min(shares):.2f}%  max {max(shares):.2f}%  "
              f"(gate is {MAX_IDLE_UNCERTAINTY_ROW_SHARE_PCT:.0f}%)")
        print(f"            {over}/{len(shares)} run(s) over the gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
