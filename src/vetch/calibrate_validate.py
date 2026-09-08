"""Standalone calibration-record validator (no GPU, no NVIDIA driver).

JSON Schema checks types and required fields. Identifiability lives here, not
in the schema file. Keep this module off the runtime path: never import it from
``vetch.calculation``.
"""

from __future__ import annotations

import copy
import json
import logging
import math
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # TypeGuard landed in 3.10 and the floor is 3.9. `from __future__ import
    # annotations` makes every annotation a string, so the name is only ever
    # needed by the type checker, never at runtime.
    from typing import TypeGuard

from vetch.calibrate_v2 import (
    BATCHED_MIN_SAMPLES,
    IDLE_DRIFT_FLAG_PCT,
    IDLE_DRIFT_WARN_PCT,
    MAX_IDLE_UNCERTAINTY_ROW_SHARE_PCT,
    gate_divergence,
    idle_uncertainty_row_share_pct,
    terms_for_rung,
)
from vetch.calibration_invariants import errors_only, record_invariants

logger = logging.getLogger(__name__)

_SCHEMA_CACHE: dict[str, Any] | None = None


def load_record_schema() -> dict[str, Any]:
    """Published JSON Schema for calibration record v2."""
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        data_path = resources.files("vetch.data").joinpath("calibration_record.schema.json")
        _SCHEMA_CACHE = json.loads(data_path.read_text(encoding="utf-8"))
    return _SCHEMA_CACHE


def _type_ok(value: Any, declared: Any) -> bool:
    if declared is None:
        return True
    if isinstance(declared, list):
        return any(_type_ok(value, t) for t in declared)
    if declared == "null":
        return value is None
    if declared == "object":
        return isinstance(value, dict)
    if declared == "array":
        return isinstance(value, list)
    if declared == "string":
        return isinstance(value, str)
    if declared == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if declared == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if declared == "boolean":
        return isinstance(value, bool)
    return True


def _check_against_schema(
    value: Any,
    schema: Mapping[str, Any],
    defs: Mapping[str, Any],
    path: str,
    errors: list[str],
) -> None:
    if "$ref" in schema:
        ref = str(schema["$ref"])
        if ref.startswith("#/$defs/"):
            key = ref.split("/")[-1]
            _check_against_schema(value, defs[key], defs, path, errors)
            return
    types = schema.get("type")
    if types is not None and not _type_ok(value, types):
        errors.append(f"{path}: expected type {types}, got {type(value).__name__}")
        return
    if value is None:
        enum = schema.get("enum")
        if enum is not None and None not in enum:
            errors.append(f"{path}: null not allowed")
        return
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected {schema['const']!r}, got {value!r}")
    enum = schema.get("enum")
    if enum is not None and value not in enum:
        errors.append(f"{path}: {value!r} not in {enum}")
    type_decl = schema.get("type")
    is_object_schema = type_decl in ("object", None) or (
        isinstance(type_decl, list) and "object" in type_decl
    )
    if isinstance(value, dict) and is_object_schema:
        for key in schema.get("required") or []:
            if key not in value:
                errors.append(f"{path}.{key}: required field missing")
        props = schema.get("properties") or {}
        for key, sub in props.items():
            if key in value:
                _check_against_schema(value[key], sub, defs, f"{path}.{key}", errors)
    if isinstance(value, list) and schema.get("items"):
        for i, item in enumerate(value):
            _check_against_schema(
                value=item, schema=schema["items"], defs=defs,
                path=f"{path}[{i}]", errors=errors,
            )


def check_json_schema(record: Mapping[str, Any]) -> list[str]:
    """Type and required-field check against the published schema. No identifiability."""
    schema = load_record_schema()
    errors: list[str] = []
    defs = schema.get("$defs") or {}
    _check_against_schema(record, schema, defs, "$", errors)
    return errors


def _finite_number(value: Any) -> TypeGuard[float | int | str]:
    if value is None or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def batched_corroboration(record: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Does this record carry the by-products a real ``--batched`` run leaves?

    The batched lane skips nested-OLS identifiability, so entry to it cannot
    rest on a self-declared label. Keying off a substring of the free-text
    ``measurement_basis`` let a hand-edited record with an empty run table
    claim an identified decode coefficient. Every clause below is something a
    genuine run emits as corroborating evidence. These fields are not proof
    against a deliberate edit. All twelve shipped batched records satisfy the
    conjunction.

    Returns ``(qualifies, failures)``. A record that fails any clause falls
    through to the nested lane and is judged there.
    """
    failures: list[str] = []
    prov = record.get("provenance")
    if not isinstance(prov, Mapping):
        return False, ["no provenance block"]

    # Machine-written enum, not a substring search over free text.
    if prov.get("calibration_mode") != "batched":
        return False, ["provenance.calibration_mode != 'batched'"]

    identity = record.get("identity")
    concurrency = identity.get("concurrency") if isinstance(identity, Mapping) else None
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency < 1:
        failures.append(f"identity.concurrency={concurrency!r} is not a positive integer")

    fit = prov.get("amortization_fit")
    if not isinstance(fit, Mapping):
        failures.append("provenance.amortization_fit missing")
    elif not (_finite_number(fit.get("a")) and _finite_number(fit.get("b"))):
        failures.append("provenance.amortization_fit lacks finite numeric a and b")

    for key in ("samples", "requests_per_level"):
        val = prov.get(key)
        if not isinstance(val, int) or isinstance(val, bool) or val < BATCHED_MIN_SAMPLES:
            failures.append(f"provenance.{key}={val!r} < {BATCHED_MIN_SAMPLES}")

    table = prov.get("raw_run_table")
    if not isinstance(table, list) or len(table) != 1:
        n = len(table) if isinstance(table, list) else None
        failures.append(
            f"raw_run_table has {n} rows; a batched aggregate is exactly 1 "
            "(zero rows is no evidence, not a throughput aggregate)"
        )
    else:
        row = table[0] if isinstance(table[0], Mapping) else {}
        for key in ("energy_wh", "output_tokens"):
            val = row.get(key)
            if not _finite_number(val) or float(val) <= 0:
                failures.append(f"raw_run_table[0].{key}={val!r} is not positive")

    if not any(
        _finite_number(prov.get(k)) for k in ("idle_watts_before", "idle_watts_after")
    ):
        failures.append("no idle_watts_before/after")

    if not str(prov.get("energy_source") or "").strip():
        failures.append("provenance.energy_source missing")

    return (not failures), failures


def is_batched_measurement(record: Mapping[str, Any]) -> bool:
    """True when the record qualifies for the batched amortization lane."""
    qualifies, _failures = batched_corroboration(record)
    return qualifies


def idle_drift_flags(record: Mapping[str, Any]) -> tuple[float | None, list[str]]:
    """Idle drift as a visible flag, never a silent pass and never a reject.

    Drift undermines the idle subtraction the marginal basis rests on, and it
    is ungated in the shipped pipeline even though v1 rejected above 15%.
    Following the residuals_structured precedent it is surfaced, not fatal, so
    the concurrency evidence survives while its weakness stays legible.
    """
    prov = record.get("provenance")
    if not isinstance(prov, Mapping):
        return None, []
    raw = prov.get("idle_drift_pct")
    if not _finite_number(raw):
        return None, []
    drift = float(raw)
    # Judge on the energy share, not on the raw percentage. A percentage of idle
    # is scale-free and therefore says nothing about whether the coefficients are
    # threatened: a 1 W laptop idle and a 90 W H100 idle report wildly different
    # percentages for the same absolute wobble. `idle_drift_pct` is still
    # reported, because it is the number an operator can sanity-check by eye.
    share = idle_uncertainty_row_share_pct(
        prov.get("raw_run_table") or [],
        prov.get("idle_watts_before"),
        prov.get("idle_watts_after"),
    )
    if share is not None:
        if share > MAX_IDLE_UNCERTAINTY_ROW_SHARE_PCT:
            return drift, [
                f"idle drift contributes {share:.1f}% of a typical row's energy "
                f"(idle_drift_pct={drift:.2f}), over the "
                f"{MAX_IDLE_UNCERTAINTY_ROW_SHARE_PCT:.0f}% limit; the idle "
                "subtraction the marginal basis rests on is unreliable. Flag, "
                "not a reject: disclose it wherever this record is used."
            ]
        if share > MAX_IDLE_UNCERTAINTY_ROW_SHARE_PCT / 2:
            return drift, [
                f"idle drift contributes {share:.1f}% of a typical row's energy "
                f"(idle_drift_pct={drift:.2f}); idle subtraction is weak on this "
                "record"
            ]
        return drift, []
    # No run table to weigh it against: fall back to the raw percentage.
    if drift > IDLE_DRIFT_FLAG_PCT:
        return drift, [
            f"idle_drift_pct={drift:.2f} exceeds {IDLE_DRIFT_FLAG_PCT:.0f}% and "
            "the record carries no run table to weigh it against "
            "(idle subtraction is unreliable). Flag, not a reject."
        ]
    if drift > IDLE_DRIFT_WARN_PCT:
        return drift, [
            f"idle_drift_pct={drift:.2f} exceeds {IDLE_DRIFT_WARN_PCT:.0f}%; "
            "idle subtraction is weak on this record"
        ]
    return drift, []


def _identifiability_batched(
    record: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Validate a batched record from stored coefficients. No nested OLS, no NumPy."""
    warnings = [
        "measurement_basis is batched (amortization curve "
        "Wh/1k_out ≈ a/C + b). Nested OLS identifiability does not apply "
        "to a one-row throughput aggregate."
    ]
    raw_coef = record.get("coefficients")
    raw_prov = record.get("provenance")
    coef: dict[str, Any] = raw_coef if isinstance(raw_coef, dict) else {}
    prov: dict[str, Any] = raw_prov if isinstance(raw_prov, dict) else {}
    table = prov.get("raw_run_table") or []
    admitted = len(table) if isinstance(table, list) else 0

    def _positive(value: Any) -> TypeGuard[float | int | str]:
        if value is None or isinstance(value, bool):
            return False
        try:
            return float(value) > 0.0
        except (TypeError, ValueError):
            return False

    def _negative(value: Any) -> bool:
        if value is None or isinstance(value, bool):
            return False
        try:
            return float(value) < 0.0
        except (TypeError, ValueError):
            return False

    # The batched estimator identifies decode throughput only.
    out = coef.get("wh_per_1k_output")
    status = {
        "wh_per_1k_output": (
            "identified" if _positive(out)
            else ("rejected" if _negative(out) else "not_identified")
        ),
        "intercept_wh": "not_identified",
        "wh_per_1k_input": "not_identified",
        "wh_per_image": "not_identified",
    }
    null_reasons = {
        "intercept_wh": "batched estimator does not identify an intercept",
        "wh_per_1k_input": "batched estimator does not identify prefill",
        "wh_per_image": "batched estimator does not identify an image term",
    }
    if status["wh_per_1k_output"] == "not_identified":
        null_reasons["wh_per_1k_output"] = (
            "batched record has no positive wh_per_1k_output"
        )
    leftover = [
        key for key in ("wh_per_1k_input", "intercept_wh", "wh_per_image")
        if _positive(coef.get(key))
    ]
    if leftover:
        warnings.append(
            "stored "
            + ", ".join(leftover)
            + " on a batched record is leftover, not an identified axis"
        )

    rejection_reasons: list[str] = []
    if status["wh_per_1k_output"] == "rejected":
        rejection_reasons.append("wh_per_1k_output has unphysical sign")
    active = status["wh_per_1k_output"] == "identified" and not rejection_reasons
    ident = {
        "estimator": "amortization_1_over_c",
        "published_rung": "batched",
        "status": status,
        "null_reasons": null_reasons,
        "residuals_structured": False,
        "active": active,
        "rejection_reasons": rejection_reasons,
        "design_leverage": {},
        "energy_basis": None,
        "admitted_rows": admitted,
        "dropped_zero_raw": 0,
        "coefficients": {
            "wh_per_1k_input": None,
            "wh_per_1k_output": float(out) if _positive(out) else None,
            "wh_per_image": None,
            "intercept_wh": None,
        },
    }
    return ident, warnings


_COEFFICIENT_AXES = (
    "wh_per_1k_input", "wh_per_1k_output", "wh_per_image", "intercept_wh",
)
# Bootstrap resampling is seeded, so a refit of the same table is deterministic
# and an exact match is the norm. The tolerance absorbs float round-tripping
# through JSON, not genuine disagreement.
_COEFFICIENT_RTOL = 1e-6


def _restatements(
    record: Mapping[str, Any], ident: Mapping[str, Any] | None
) -> tuple[list[str], list[str]]:
    """Ways the record's stored claims disagree with a refit of its own table.

    ``active`` agreeing is not enough. A record can hold an identified status,
    the right published rung and a coefficient that is simply a different
    number from the one its run table produces, and every other check here
    passes it.

    The two directions are not the same failure. A record that publishes a
    number its data does not support is over-claiming, and that is fatal. A
    record that withholds a number its data would support is under-claiming --
    a stricter in-house gate, an axis the writer chose not to publish -- which
    is conservative, not dishonest, and is reported as a warning.

    Returns ``(errors, warnings)``.
    """
    if not ident or ident.get("estimator") != "nested_ols":
        # Only the nested lane re-derives coefficients from the run table. The
        # batched lane reads the stored decode figure back and reports every
        # other axis as not_identified, so comparing against it would flag a
        # legitimately stored `wh_per_image` as an over-claim. Batched records
        # are corroborated by `batched_corroboration` instead.
        return [], []
    errors: list[str] = []
    warnings: list[str] = []
    coef = record.get("coefficients")
    coef = coef if isinstance(coef, Mapping) else {}
    computed = ident.get("coefficients") or {}

    for axis in _COEFFICIENT_AXES:
        stored, fresh = coef.get(axis), computed.get(axis)
        if stored is None and fresh is None:
            continue
        if stored is None:
            warnings.append(
                f"{axis}: record publishes nothing but its run table supports "
                f"{fresh!r}; the record is more conservative than its data"
            )
            continue
        if fresh is None:
            errors.append(
                f"{axis}: record publishes {stored!r} but its run table does "
                "not support publishing this axis at all"
            )
            continue
        s_num, f_num = _num(stored), _num(fresh)
        if s_num is None or f_num is None or not _close(s_num, f_num):
            errors.append(
                f"{axis}: record publishes {stored!r} but its run table "
                f"supports {fresh!r}"
            )

    stored_status = coef.get("status")
    if isinstance(stored_status, Mapping):
        fresh_status = ident.get("status") or {}
        for axis in _COEFFICIENT_AXES:
            if axis not in stored_status:
                continue
            was, now = stored_status[axis], fresh_status.get(axis)
            if was == now:
                continue
            if was == "identified":
                errors.append(
                    f"{axis}: record claims status 'identified' but its run "
                    f"table supports {now!r}"
                )
            else:
                warnings.append(
                    f"{axis}: record stores status {was!r} but its run table "
                    f"supports {now!r}"
                )

    stored_rung = coef.get("published_rung")
    fresh_rung = ident.get("published_rung")
    if stored_rung is not None and stored_rung != fresh_rung:
        message = (
            f"published_rung: record stores {stored_rung!r} but its run table "
            f"supports {fresh_rung!r}"
        )
        stored_terms = terms_for_rung(stored_rung)
        fresh_terms = terms_for_rung(fresh_rung)
        # A malformed/unknown rung or two incomparable term sets is not a
        # conservative under-claim: it changes the regression model.
        if (
            not stored_terms
            or not fresh_terms
            or not (stored_terms < fresh_terms or stored_terms > fresh_terms)
            or stored_terms > fresh_terms
        ):
            errors.append(message)
        else:
            warnings.append(message)
    return errors, warnings


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=_COEFFICIENT_RTOL, abs_tol=1e-15)


def verify_attestation_hashes(record: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute ``profile_hash`` / ``content_hash`` from the record's own fields.

    Both are derivable from what the file already carries, so a record whose
    coefficients were edited after it was written no longer hashes to the
    digest it ships. This does not make a record trustworthy -- whoever edited
    it could recompute the hash too -- it makes an *un*edited one cheap to
    confirm, and catches the careless case for free. The real check is
    ``_restatements``, which re-derives the numbers from the run table.
    """
    stored = {
        key: record.get(key)
        for key in ("profile_hash", "content_hash")
        if isinstance(record.get(key), str)
    }
    if not stored:
        return {"checked": False, "reason": "record carries no attestation hashes"}
    try:
        from vetch.calibration_store import (
            _measurement_key_payload,
            apply_attestation_hashes,
            identity_from_dict,
        )

        identity = record.get("identity")
        identity = identity if isinstance(identity, Mapping) else {}
        # Use the canonical parser so early records using ``backend`` and
        # current records using ``serving_engine`` hash identically to their
        # writer. Constructing CalibrationIdentity directly silently dropped
        # the legacy alias and made valid records look tampered.
        idn = identity_from_dict(identity)
        probe = apply_attestation_hashes(
            copy.deepcopy(dict(record)),
            host_key=_measurement_key_payload(
                idn,
                record.get("host"),
                record.get("measurement_conditions"),
                record.get("serving_config"),
            ),
        )
    except Exception as exc:  # pragma: no cover - defensive
        return {"checked": False, "reason": f"could not recompute: {exc}"}

    mismatched = [key for key, value in stored.items() if probe.get(key) != value]
    return {"checked": True, "match": not mismatched, "mismatched": mismatched}


def validate_calibration_record(
    record: Mapping[str, Any],
    *,
    identifiability: bool = True,
) -> dict[str, Any]:
    """Validate a record. GPU-free.

    Returns a report dict: ``ok``, ``schema_errors``, ``identifiability`` (if
    requested), ``active``, ``status``, ``warnings``.
    """
    schema_version = record.get("schema_version")
    warnings: list[str] = []
    schema_errors: list[str] = []

    if schema_version == 1:
        warnings.append(
            "schema_version 1: v1 remains resolvable at inference with "
            "clamp-to-zero semantics. Migrate to v2; active=true on newly "
            "written records requires schema 2."
        )
        # v1 is not checked against the v2 JSON Schema.
    elif schema_version == 2:
        schema_errors = check_json_schema(record)
    else:
        schema_errors.append(
            f"schema_version={schema_version!r}: expected 1 (legacy) or 2"
        )

    drift, drift_warnings = idle_drift_flags(record)
    warnings.extend(drift_warnings)

    # Self-consistency comes before identifiability: a record that contradicts
    # itself cannot be trusted however good its statistics look.
    violations = record_invariants(record)
    invariant_errors = [str(v) for v in errors_only(violations)]
    warnings.extend(str(v) for v in violations if v.severity == "warning")

    ident: dict[str, Any] | None = None
    if identifiability:
        qualifies, batched_failures = batched_corroboration(record)
        if not qualifies and batched_failures and (
            (record.get("provenance") or {}).get("calibration_mode") == "batched"
        ):
            warnings.append(
                "record declares calibration_mode='batched' but lacks the "
                "by-products of a real batched run ("
                + "; ".join(batched_failures)
                + "); judged as a nested fit instead"
            )
        if qualifies:
            ident, extra = _identifiability_batched(record)
            warnings.extend(extra)
        else:
            from vetch.calibrate_v2 import fit_record

            fit = fit_record(record)
            stored_gate = (
                (record.get("provenance") or {}).get("fit") or {}
            ).get("gate")
            divergence = gate_divergence(stored_gate)
            if divergence:
                warnings.append(
                    "this record was judged under different gate parameters "
                    "than this build uses ("
                    + "; ".join(
                        f"{k}: recorded {v['recorded']}, current {v['current']}"
                        for k, v in sorted(divergence.items())
                    )
                    + "); statuses below are recomputed under the current gate"
                )
            elif stored_gate is None:
                warnings.append(
                    "record does not state the gate parameters it was judged "
                    "under; its stored statuses cannot be reproduced"
                )
            ident = {
                "estimator": "nested_ols",
                "published_rung": fit.published_rung,
                "status": dict(fit.status),
                "null_reasons": dict(fit.null_reasons),
                "residuals_structured": fit.residuals_structured,
                "active": fit.active,
                "rejection_reasons": list(fit.rejection_reasons),
                "design_leverage": fit.design_leverage,
                "energy_basis": fit.energy_basis,
                "admitted_rows": fit.admitted_rows,
                "dropped_zero_raw": fit.dropped_zero_raw,
                "dropped_malformed": fit.dropped_malformed,
                "dropped_under_floor": fit.dropped_under_floor,
                "coefficients": {
                    "wh_per_1k_input": fit.wh_per_1k_input,
                    "wh_per_1k_output": fit.wh_per_1k_output,
                    "wh_per_image": fit.wh_per_image,
                    "intercept_wh": fit.intercept_wh,
                },
                "unpublished_rejections": list(fit.unpublished_rejections),
            }
            for reason in fit.unpublished_rejections:
                warnings.append(
                    "impossible sign on an axis outside the published rung "
                    f"({reason}); the published coefficients are not read from "
                    "that fit, so the record stays active, but the measurement "
                    "is suspect"
                )
            if fit.dropped_under_floor:
                warnings.append(
                    f"{fit.dropped_under_floor} run-table row(s) dropped for "
                    "falling under the power-sampling duration floor; the grid "
                    "was mis-sized and those design points are lost"
                )
            if fit.dropped_malformed:
                warnings.append(
                    f"{fit.dropped_malformed} run-table row(s) dropped as "
                    "malformed (a field the fit reads is null, non-numeric or "
                    "non-finite)"
                )
            if fit.residuals_structured:
                warnings.append(
                    "residuals_structured=true on the published nested model "
                    "(misspecification flag; does not by itself reject an "
                    "identified decode slope)"
                )

    ok = not schema_errors and not invariant_errors
    if ident is not None and ident.get("rejection_reasons"):
        ok = False
    # The point of running against someone else's file is catching a record
    # whose stored claim outruns what its own data supports.
    active_mismatch = None
    if ident is not None and isinstance(record.get("active"), bool):
        if bool(record["active"]) != bool(ident.get("active")):
            active_mismatch = {
                "stored": bool(record["active"]),
                "computed": bool(ident.get("active")),
            }
            warnings.append(
                f"record stores active={record['active']} but this data "
                f"supports active={ident.get('active')}"
            )
            ok = False

    # Comparing only `active` let the headline numbers say anything they liked.
    # The refit above already computes what the run table supports; a stored
    # coefficient that disagrees with it is the exact failure this command is
    # advertised to catch, and doubling one used to pass clean.
    restatement_errors, restatement_warnings = _restatements(record, ident)
    warnings.extend(restatement_warnings)
    if restatement_errors:
        warnings.extend(restatement_errors)
        ok = False

    attestation = verify_attestation_hashes(record)
    if attestation.get("checked") and not attestation.get("match"):
        warnings.append(
            "attestation hash mismatch: "
            + ", ".join(sorted(attestation.get("mismatched", ())))
            + "; the record's content does not hash to the digest it carries"
        )
        ok = False

    return {
        "ok": ok,
        "schema_version": schema_version,
        "schema_errors": schema_errors,
        "invariant_errors": invariant_errors,
        "identifiability": ident,
        "warnings": warnings,
        "idle_drift_pct": drift,
        "active_mismatch": active_mismatch,
        "restatements": restatement_errors,
        "restatement_warnings": restatement_warnings,
        "attestation": attestation,
        "active": (ident or {}).get("active", record.get("active")),
    }


def validate_calibration_file(path: str | Path, *, identifiability: bool = True) -> dict[str, Any]:
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {
            "ok": False,
            "path": str(p),
            "schema_errors": ["file is not a JSON object"],
            "identifiability": None,
            "warnings": [],
            "active": False,
        }
    report = validate_calibration_record(raw, identifiability=identifiability)
    report["path"] = str(p)
    return report


def format_report(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    path = report.get("path")
    if path:
        lines.append(f"file: {path}")
    lines.append(f"schema_version: {report.get('schema_version')}")
    lines.append(f"ok: {report.get('ok')}")
    for err in report.get("schema_errors") or []:
        lines.append(f"schema error: {err}")
    for err in report.get("invariant_errors") or []:
        lines.append(f"invariant error: {err}")
    ident = report.get("identifiability")
    if ident:
        lines.append(f"published_rung: {ident.get('published_rung')}")
        if ident.get("estimator"):
            lines.append(f"estimator: {ident.get('estimator')}")
        lines.append(f"active: {ident.get('active')}")
        status = ident.get("status") or {}
        for axis, st in status.items():
            reason = (ident.get("null_reasons") or {}).get(axis, "")
            extra = f" ({reason})" if st != "identified" and reason else ""
            lines.append(f"  {axis}: {st}{extra}")
        coef = ident.get("coefficients") or {}
        lines.append(
            f"coefficients: in={coef.get('wh_per_1k_input')!r} "
            f"out={coef.get('wh_per_1k_output')!r} "
            f"intercept={coef.get('intercept_wh')!r} "
            f"image={coef.get('wh_per_image')!r}"
        )
        if ident.get("residuals_structured"):
            lines.append("residuals_structured: true (flag, not a reject)")
        for reason in ident.get("rejection_reasons") or []:
            lines.append(f"rejected: {reason}")
        dl = ident.get("design_leverage") or {}
        inp = dl.get("input") or {}
        if inp:
            lines.append(
                f"design_leverage.input: design={inp.get('design_leverage')} "
                f"fitted={inp.get('fitted_leverage')}"
            )
    for w in report.get("warnings") or []:
        lines.append(f"warning: {w}")
    return "\n".join(lines) + "\n"
