#!/usr/bin/env python3
"""Migrate schema-1 calibration records to schema 2.

Adds host / serving_config / measurement_conditions / energy_basis / node_power,
moves identity.instance_type onto host, and renames files to the measurement
slug. Does not invent values that were not recorded.

Optional ``--refit`` re-runs nested-model identifiability on raw_run_table
(batch-1 style records). Batched single-row tables are not nested-refit;
their clamped 0.0 input coefficient is nulled instead.

Usage:
    python scripts/migrate_calibration_records_v1_to_v2.py h100-calibration-records
    python scripts/migrate_calibration_records_v1_to_v2.py h100-calibration-records --refit
    python scripts/migrate_calibration_records_v1_to_v2.py path/to/file.json --in-place
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from vetch.calibration_store import (  # noqa: E402
    CALIB_RECORD_SCHEMA,
    CALIB_RECORD_SCHEMA_V1,
    apply_attestation_hashes,
    default_energy_basis,
    default_host,
    default_measurement_conditions,
    default_node_power,
    default_serving_config,
    identity_from_dict,
    infer_gpu_form_factor,
    measurement_slug,
)

BATCH1_INFRA = {
    "infrastructure_provider": "vast.ai",
    "provider_kind": "marketplace",
    "tenancy": "unknown",
}


def _is_batch1_h100(rec: dict[str, Any], path: Path) -> bool:
    ident = rec.get("identity") or {}
    gpu = str(ident.get("gpu") or rec.get("provenance", {}).get("gpu_canonical") or "")
    return "h100-sxm" in gpu.casefold() or "h100-sxm" in path.name.casefold()


def migrate_record(
    rec: dict[str, Any],
    *,
    path: Path | None = None,
    refit: bool = False,
    batch1_defaults: bool = False,
) -> dict[str, Any]:
    if rec.get("schema_version") == CALIB_RECORD_SCHEMA:
        return rec
    ident = identity_from_dict(rec["identity"])
    prov = dict(rec.get("provenance") or {})
    instance_type = ident.instance_type
    ident_dict = {
        "provider": ident.provider,
        "model": ident.model,
        "gpu": ident.gpu,
        "serving_engine": ident.serving_engine,
        "precision": ident.precision,
        "visual_token_budget": ident.visual_token_budget,
        "concurrency": ident.concurrency,
        "instance_type": None,
    }

    form = infer_gpu_form_factor(ident.gpu) or infer_gpu_form_factor(
        str(prov.get("gpu_canonical") or "")
    )
    infra_provider = None
    provider_kind = None
    tenancy = "unknown"
    if batch1_defaults and path is not None and _is_batch1_h100(rec, path):
        infra_provider = BATCH1_INFRA["infrastructure_provider"]
        provider_kind = BATCH1_INFRA["provider_kind"]
        tenancy = BATCH1_INFRA["tenancy"]
        if form is None:
            form = "sxm"

    host = default_host(
        infrastructure_provider=infra_provider,
        provider_kind=provider_kind,
        instance_type=instance_type,
        tenancy=tenancy,
        gpu_form_factor=form,
    )
    conditions = default_measurement_conditions(
        enforced_power_limit_w=prov.get("enforced_power_limit_w"),
        sm_clock_locked=prov.get("sm_clock_locked"),
        sm_clock_mhz=(prov.get("clocks") or {}).get("sm_clock_mhz"),
    )
    # Batch 1 cannot recover serving flags (chunked prefill, prefix cache).
    # Leave the block null rather than inferring from identity.serving_engine.
    if batch1_defaults and path is not None and _is_batch1_h100(rec, path):
        serving = default_serving_config()
    else:
        serving = default_serving_config(
            engine=ident.serving_engine,
            engine_version=prov.get("server_version"),
        )

    notes = list(prov.get("schema_migration") or []) if isinstance(
        prov.get("schema_migration"), list
    ) else []
    notes.append(
        "Host block reconstructed after the fact rather than captured at "
        "measurement time; weaker evidence than a contemporaneous record."
    )
    prov["schema_migration"] = notes
    prov.setdefault("pue_applied", False)
    prov.setdefault("grid_applied", False)

    coef = dict(rec.get("coefficients") or {})
    table = prov.get("raw_run_table") or []
    batched = str(prov.get("measurement_basis") or "").endswith("batched") or (
        isinstance(table, list) and len(table) < 3
    )

    energy_basis = default_energy_basis()
    if refit and not batched and isinstance(table, list) and len(table) >= 8:
        from vetch.calibrate_v2 import CALIBRATION_PROTOCOL_VERSION, fit_record, nested_fit_to_dict

        fit = fit_record(rec)
        coef["wh_per_1k_input"] = fit.wh_per_1k_input
        coef["wh_per_1k_output"] = fit.wh_per_1k_output
        coef["wh_per_image"] = fit.wh_per_image
        coef["intercept_wh"] = fit.intercept_wh
        coef["status"] = fit.status
        coef["null_reasons"] = fit.null_reasons
        coef["published_rung"] = fit.published_rung
        energy_basis = fit.energy_basis
        rec["active"] = fit.active
        rec["rejection_reasons"] = fit.rejection_reasons or None
        prov["methodology_version"] = CALIBRATION_PROTOCOL_VERSION
        prov["identifiability"] = nested_fit_to_dict(fit)
        fit_block = dict(prov.get("fit") or {})
        if "v1" not in fit_block:
            v1_keep = {
                k: fit_block[k]
                for k in (
                    "input_ci95", "output_ci95", "image_ci95", "intercept_ci95",
                )
                if k in fit_block
            }
            if v1_keep:
                fit_block["v1"] = v1_keep
        fit_block["residuals_structured"] = fit.residuals_structured
        fit_block["published_rung"] = fit.published_rung
        if fit.output_ci95:
            fit_block["output_ci95"] = list(fit.output_ci95)
        if fit.input_ci95:
            fit_block["input_ci95"] = list(fit.input_ci95)
        if fit.intercept_ci95:
            fit_block["intercept_ci95"] = list(fit.intercept_ci95)
        if fit.image_ci95:
            fit_block["image_ci95"] = list(fit.image_ci95)
        fit_block["r2"] = fit.r2
        prov["fit"] = fit_block
    else:
        # v2 forbids the clamp: a stored 0.0 input is unidentified.
        for key in ("wh_per_1k_input", "wh_per_1k_output", "wh_per_image", "intercept_wh"):
            if coef.get(key) == 0.0:
                coef[key] = None

    out: dict[str, Any] = dict(rec)
    out["schema_version"] = CALIB_RECORD_SCHEMA
    out["identity"] = ident_dict
    out["host"] = host
    out["measurement_conditions"] = conditions
    out["serving_config"] = serving
    out["energy_basis"] = energy_basis
    out["node_power"] = default_node_power()
    out["coefficients"] = coef
    out["provenance"] = prov

    identity_obj = identity_from_dict(ident_dict)
    profile_body = {
        "identity": ident_dict,
        "host_key": {
            "identity": ident_dict,
            "host": {
                "infrastructure_provider": host.get("infrastructure_provider"),
                "site": host.get("site"),
                "tenancy": host.get("tenancy"),
                "gpu_form_factor": host.get("gpu_form_factor"),
                "instance_type": host.get("instance_type"),
            },
            "enforced_power_limit_w": conditions.get("enforced_power_limit_w"),
            "serving_config": serving,
        },
        "coefficients": {
            k: coef.get(k)
            for k in (
                "wh_per_1k_input", "wh_per_1k_output", "wh_per_image",
                "visual_tokens_per_image", "intercept_wh",
            )
        },
    }
    apply_attestation_hashes(out, host_key=profile_body["host_key"])
    out["_new_stem"] = measurement_slug(identity_obj, host, conditions, serving)
    return out


def _iter_json(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*.json") if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Record file or directory")
    parser.add_argument("--refit", action="store_true", help="Re-fit nested models (needs NumPy)")
    parser.add_argument("--in-place", action="store_true", help="Rewrite files (default: dry-run)")
    parser.add_argument(
        "--batch1-defaults", action="store_true", default=True,
        help="Fill vast.ai marketplace host fields for H100-SXM batch-1 files",
    )
    args = parser.parse_args()

    files = _iter_json(args.path)
    if not files:
        print("No JSON files found", file=sys.stderr)
        return 1

    for path in files:
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"SKIP {path}: {e}")
            continue
        if not isinstance(rec, dict) or "identity" not in rec:
            continue
        if rec.get("schema_version") == CALIB_RECORD_SCHEMA:
            print(f"SKIP already schema 2: {path.name}")
            continue
        if rec.get("schema_version") not in (CALIB_RECORD_SCHEMA_V1, 1):
            continue
        migrated = migrate_record(
            rec, path=path, refit=args.refit, batch1_defaults=args.batch1_defaults,
        )
        new_stem = migrated.pop("_new_stem")
        dest = path.with_name(f"{new_stem}.json")
        print(f"{path.name} -> {dest.name}  active={migrated.get('active')}")
        if args.in_place:
            dest.write_text(json.dumps(migrated, indent=2) + "\n", encoding="utf-8")
            if dest.resolve() != path.resolve():
                path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
