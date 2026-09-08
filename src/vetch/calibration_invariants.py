"""Internal-consistency checks for a calibration record.

Identifiability asks whether a design could resolve a coefficient. These
invariants ask a prior question: does the record contradict itself? Nothing in
v2 asked it, which is how ``achieved_concurrency = 0.0`` shipped in twelve
records that each requested a concurrency between 1 and 32, and how ``samples``
came to mean two different things depending on measurement mode.

One function, three call sites: the writer before saving, ``validate-calibration``
for records we did not produce, and a CI check over everything shipped. A record
that contradicts itself should never reach a consumer, regardless of how good
its statistics are.

Stdlib only. This runs on the no-GPU, no-NumPy path.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

Severity = Literal["error", "warning"]

# `samples` does NOT mean one thing. A batched writer sets it to the request
# count (`requests_per_level`); a nested writer sets it to the row count
# (`len(run_records)`). Those coincided until rows could aggregate repeats, and
# batch 2 broke the tie: 32 rows over 398 requests, `samples=32`.
#
# Rather than silently redefine a shipped field, the writer now also emits
# `requests_measured`, and this module checks each field against what it
# actually means in each mode. `samples` keeps its per-mode meaning; the
# ambiguity is resolved by the second field, not by pretending it never existed.
BATCHED_EXPECTED_ROWS = 1


def _requests_in_table(rows: Sequence[Any]) -> int:
    total = 0
    for raw in rows:
        row = _mapping(raw)
        count = _num(row.get("requests"))
        total += max(1, int(count)) if count is not None else 1
    return total


@dataclass(frozen=True)
class Violation:
    code: str
    message: str
    severity: Severity = "error"

    def __str__(self) -> str:
        return f"[{self.severity}] {self.code}: {self.message}"


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def is_batched(record: Mapping[str, Any]) -> bool:
    return _mapping(record.get("provenance")).get("calibration_mode") == "batched"


def _check_shape(record: Mapping[str, Any], out: list[Violation]) -> None:
    prov = _mapping(record.get("provenance"))
    rows = prov.get("raw_run_table")
    rows = rows if isinstance(rows, list) else []
    samples = _num(prov.get("samples"))

    if samples is None:
        out.append(Violation("samples_missing", "provenance.samples is absent"))
        return
    if not rows:
        # A record may legitimately omit its run table (size, privacy). That
        # limits what a consumer can re-derive, but it is not a contradiction.
        out.append(Violation(
            "raw_run_table_absent",
            f"samples={int(samples)} but no raw_run_table; the record cannot be "
            "re-fitted or checked for identifiability by a third party",
            severity="warning",
        ))
        return
    if is_batched(record):
        if len(rows) != BATCHED_EXPECTED_ROWS:
            out.append(Violation(
                "batched_row_count",
                f"batched record has {len(rows)} run-table rows, expected "
                f"{BATCHED_EXPECTED_ROWS} (one throughput aggregate)",
            ))
        if samples < 1:
            out.append(Violation("batched_samples", f"samples={samples:g} < 1"))
    elif len(rows) != int(samples):
        out.append(Violation(
            "samples_row_mismatch",
            f"samples={int(samples)} but raw_run_table has {len(rows)} rows; a "
            "nested record's samples is its row count",
        ))

    requests_measured = _num(prov.get("requests_measured"))
    if requests_measured is not None and rows:
        actual = _requests_in_table(rows)
        if int(requests_measured) != actual:
            out.append(Violation(
                "requests_measured_mismatch",
                f"provenance.requests_measured={int(requests_measured)} but the "
                f"run table's per-row `requests` sum to {actual}",
            ))
    elif rows and not is_batched(record) and _requests_in_table(rows) != len(rows):
        # Aggregated rows without the field: the reader cannot tell how many
        # requests produced these numbers, and `samples` will not tell them.
        out.append(Violation(
            "requests_measured_absent",
            f"raw_run_table aggregates {_requests_in_table(rows)} requests into "
            f"{len(rows)} rows but provenance.requests_measured is absent; "
            "`samples` is the row count and does not record the request count",
            severity="warning",
        ))


def _check_rows(record: Mapping[str, Any], out: list[Violation]) -> None:
    prov = _mapping(record.get("provenance"))
    rows = prov.get("raw_run_table")
    if not isinstance(rows, list):
        return
    for i, raw in enumerate(rows):
        row = _mapping(raw)
        gross, net = _num(row.get("raw_energy_wh")), _num(row.get("energy_wh"))
        if gross is not None and net is not None and net > gross + 1e-12:
            out.append(Violation(
                "net_exceeds_gross",
                f"row {i}: energy_wh={net:g} exceeds raw_energy_wh={gross:g}; "
                "idle subtraction cannot increase measured energy",
            ))
        duration = _num(row.get("duration_ms"))
        if duration is not None and duration <= 0:
            out.append(Violation("row_duration", f"row {i}: duration_ms={duration:g} <= 0"))
        for key in ("output_tokens", "text_tokens", "n_images"):
            value = _num(row.get(key))
            if value is not None and value < 0:
                out.append(Violation("row_negative", f"row {i}: {key}={value:g} < 0"))


def _check_concurrency(record: Mapping[str, Any], out: list[Violation]) -> None:
    prov = _mapping(record.get("provenance"))
    requested = _num(_mapping(record.get("identity")).get("concurrency"))
    achieved = _num(prov.get("achieved_concurrency"))
    if requested is None or achieved is None:
        return  # unknown is honest; only a stored value can contradict
    if achieved <= 0:
        out.append(Violation(
            "achieved_concurrency_zero",
            f"identity.concurrency={requested:g} but achieved_concurrency="
            f"{achieved:g}; a zero reading means it was sampled when no request "
            "was in flight. Record null for unknown rather than zero",
        ))
    elif achieved > requested * 1.5:
        out.append(Violation(
            "achieved_concurrency_high",
            f"achieved_concurrency={achieved:g} exceeds requested "
            f"{requested:g} by more than 50%",
            severity="warning",
        ))


def _check_coefficients(record: Mapping[str, Any], out: list[Violation]) -> None:
    coef = _mapping(record.get("coefficients"))
    status = _mapping(coef.get("status"))
    reasons = _mapping(coef.get("null_reasons"))
    axes = ("wh_per_1k_input", "wh_per_1k_output", "wh_per_image", "intercept_wh")

    for axis in axes:
        value, state = coef.get(axis), status.get(axis)
        if not status:
            continue
        if value is not None and state != "identified":
            out.append(Violation(
                "published_unidentified",
                f"{axis} carries a value but status is {state!r}; only an "
                "identified axis may publish a coefficient",
            ))
        if value is None and state != "identified" and axis not in reasons:
            out.append(Violation(
                "missing_null_reason",
                f"{axis} is null with status {state!r} but no null_reasons entry",
                severity="warning",
            ))
        if value is not None and (num := _num(value)) is not None and num < 0:
            out.append(Violation(
                "negative_coefficient",
                f"{axis}={num:g} is negative; energy coefficients cannot be",
            ))

    if record.get("active") is True and status:
        if not any(v == "identified" for v in status.values()):
            out.append(Violation(
                "active_without_identified",
                "active=true but no coefficient is identified",
            ))
        # "active=true implies no rejected axis anywhere" is the OLD rule and
        # is now wrong: an impossible sign on an axis outside the published
        # rung does not contaminate the published numbers, so such a record is
        # legitimately active. What must still hold is the narrower claim -- no
        # rejected axis may be a term of the rung the numbers were read from.
        from vetch.calibrate_v2 import terms_for_rung

        published = terms_for_rung(coef.get("published_rung"))
        contaminating = sorted(
            axis for axis in published if status.get(axis) == "rejected"
        )
        if contaminating:
            out.append(Violation(
                "active_with_rejected_published_term",
                f"active=true but {', '.join(contaminating)} is rejected and is "
                f"a term of the published rung {coef.get('published_rung')!r}; "
                "the published coefficients were read from a fit containing an "
                "impossible sign",
            ))


def _check_energy_basis(record: Mapping[str, Any], out: list[Violation]) -> None:
    basis = _mapping(record.get("energy_basis"))
    if not basis:
        return
    published = _num(_mapping(record.get("coefficients")).get("wh_per_1k_output"))
    marginal = _num(_mapping(basis.get("marginal")).get("wh_per_1k_output"))
    if published is not None and marginal is not None:
        if abs(published - marginal) > 1e-9:
            out.append(Violation(
                "energy_basis_mismatch",
                f"energy_basis.marginal.wh_per_1k_output={marginal:g} disagrees "
                f"with coefficients.wh_per_1k_output={published:g}",
            ))
    attributed = _mapping(basis.get("attributed"))
    levels = attributed.get("levels")
    if isinstance(levels, list) and levels:
        seen = [_num(_mapping(x).get("assumed_utilisation")) for x in levels]
        if sorted(v for v in seen if v is not None) != [0.2, 0.5, 0.8]:
            out.append(Violation(
                "attributed_levels",
                f"attributed utilisation levels {seen} are not the protocol's "
                "fixed 0.2/0.5/0.8; two labs would emit different columns",
                severity="warning",
            ))


def _check_idle(record: Mapping[str, Any], out: list[Violation]) -> None:
    prov = _mapping(record.get("provenance"))
    if prov.get("idle_subtracted") is not True:
        return
    before, after = _num(prov.get("idle_watts_before")), _num(prov.get("idle_watts_after"))
    if before is None and after is None:
        out.append(Violation(
            "idle_subtracted_without_idle",
            "idle_subtracted=true but neither idle_watts_before nor "
            "idle_watts_after is recorded",
        ))
        return
    for name, value in (("before", before), ("after", after)):
        if value is not None and value <= 0:
            # Implausible, not self-contradictory: the record is consistent, it
            # just describes a measurement that cannot support identifiability.
            # fit_nested_models already refuses to publish coefficients without
            # a positive idle figure, which is the right layer for that call.
            # This layer stays strictly about internal contradiction.
            out.append(Violation(
                "idle_nonpositive",
                f"idle_watts_{name}={value:g} <= 0 with idle_subtracted=true; "
                "no energy was actually subtracted",
                severity="warning",
            ))


def _check_identity(record: Mapping[str, Any], out: list[Violation]) -> None:
    identity = _mapping(record.get("identity"))
    if record.get("schema_version") == 2 and identity.get("instance_type") is not None:
        out.append(Violation(
            "identity_instance_type",
            "identity.instance_type must be null in schema 2; it moved to host",
        ))
    provider = identity.get("provider")
    host = _mapping(record.get("host")).get("infrastructure_provider")
    if provider and host and str(provider).casefold() == str(host).casefold():
        out.append(Violation(
            "provider_is_infrastructure",
            f"identity.provider={provider!r} duplicates "
            "host.infrastructure_provider; identity.provider is the serving "
            "stack label and must stay 'self-hosted' for local inference",
        ))


def _check_measurement_basis(record: Mapping[str, Any], out: list[Violation]) -> None:
    """The stated basis must match the source the numbers actually came from.

    A record claiming an energy-counter basis while its figures were produced by
    power-sampling integration misstates its own measurement quality, and the
    two paths differ in more than precision: only the sampling path carries a
    duration floor, so a consumer applying the wrong admission rule silently
    keeps or drops the wrong rows.
    """
    prov = _mapping(record.get("provenance"))
    source = str(prov.get("energy_source") or "").casefold()
    basis = str(prov.get("measurement_basis") or "").casefold()
    if not source or not basis:
        return
    counter_source = "totalenergy" in source.replace("_", "")
    sampling_source = "integration" in source or "sampl" in source
    if counter_source and "power" in basis:
        out.append(Violation(
            "basis_source_mismatch",
            f"energy_source={prov.get('energy_source')!r} is a hardware counter "
            f"but measurement_basis={prov.get('measurement_basis')!r} describes "
            "power sampling",
        ))
    elif sampling_source and "power" not in basis and "energy" in basis:
        out.append(Violation(
            "basis_source_mismatch",
            f"energy_source={prov.get('energy_source')!r} is power-sampling "
            f"integration but measurement_basis={prov.get('measurement_basis')!r} "
            "claims an energy-counter basis",
        ))


CHECKS = (
    _check_shape, _check_rows, _check_concurrency,
    _check_coefficients, _check_energy_basis, _check_idle, _check_identity,
    _check_measurement_basis,
)


def record_invariants(record: Mapping[str, Any]) -> list[Violation]:
    """Every way this record contradicts itself. Empty means coherent."""
    out: list[Violation] = []
    for check in CHECKS:
        check(record, out)
    return out


def errors_only(violations: Sequence[Violation]) -> list[Violation]:
    return [v for v in violations if v.severity == "error"]
