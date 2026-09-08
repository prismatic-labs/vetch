"""Calibration protocol v2: nested-model identifiability and dual energy basis.

This module is part of the calibration extra. Bootstrap intervals need NumPy.
Never import it from :mod:`vetch.calculation` (runtime path is stdlib-only).

Protocol constants are pinned here and in ``docs/calibration-protocol-v2.md``.
The leverage constants are defined here so all calibration paths use the same
identifiability rules.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

CALIBRATION_PROTOCOL_VERSION = "2.0"

# Post-fit gate: how precisely a coefficient must be determined to be worth
# publishing, as the bootstrap CI's half-width relative to the estimate. This
# is measured, not predicted, and it subsumes the older tests: a CI that
# includes zero has a relative half-width of at least 100% by construction, so
# any target below that already enforces "interval excludes zero".
MAX_RELATIVE_CI_HALF_WIDTH = 0.10
# Pre-run only. Predicting whether a design will work needs a noise prior
# because no residuals exist yet. Never use this to judge a fit that has
# already been measured: the standard error is right there in the interval.
LEVERAGE_MULTIPLE = 10.0
# Conservative fraction of idle energy treated as integration uncertainty.
# The energy-counter path has no sampling-gap hazard; this is a prior, not a
# duration floor.
IDLE_INTEGRATION_UNCERTAINTY = 0.02
# Prefill isolation arm (protocol §2.4). A design without this arm cannot
# identify the input coefficient, regardless of what a misspecified fit's
# interval says. Gemma 4 31B fp8 is the record that proves the interval-only
# rule is wrong.
PREFILL_ARM_MIN_INPUT_TOKENS = 1024
PREFILL_ARM_MAX_OUTPUT_TOKENS = 8
PREFILL_ARM_MIN_ROWS = 2
PREFILL_ARM_MIN_SPAN_TOKENS = 1024
# Image arm: both with-image and without-image cells must be populated. This is
# a presence rule only -- there is no energy-magnitude test on the image term,
# and there never was one; `image_design_leverage` counts rows. Whether the
# image term is worth publishing is then decided by the same realised-precision
# gate as every other axis.
IMAGE_ARM_MIN_ROWS_EACH = 2
# Attributed utilisation columns. Fixed so two labs emit the same levels.
ATTRIBUTED_UTILISATION_LEVELS: tuple[float, ...] = (0.2, 0.5, 0.8)
# Duration floor applies only on the power-sampling fallback, never on the
# NVML energy-counter path (Yang, Adamek & Armour arXiv:2312.02741 is about
# nvidia-smi sampling gaps, not nvmlDeviceGetTotalEnergyConsumption).
POWER_SAMPLING_DURATION_FLOOR_MS = 2000.0
BOOTSTRAP_RESAMPLES = 500
BOOTSTRAP_SEED = 42
# A rung whose design matrix is collinear cannot separate its own terms, no
# matter how tight the bootstrap interval looks: every resample inherits the
# same degeneracy, so the interval is narrow and meaningless. v1 gated this at
# 30 and v2 must not be laxer than the pipeline it replaces.
MAX_CONDITION_NUMBER = 30.0
# Idle drift is a measurement-validity signal, not an identifiability one: it
# undermines the idle subtraction that the marginal basis rests on. Following
# the precedent set for residuals_structured, it is a flag rather than a
# reject, so a drifting record stays usable and stays visibly weak.
IDLE_DRIFT_WARN_PCT = 10.0
IDLE_DRIFT_FLAG_PCT = 15.0
# What actually decides whether an unstable idle baseline matters: not how far
# idle moved, but how much that movement contributes to the rows being fitted.
#
# `idle_drift_pct` is scale-free, and that is exactly wrong. It was sized for a
# ~90 W H100 idle where a 0.3 W background excursion is 0.3%. Apple Silicon idles
# near 1 W, where the same excursion is 30% -- measured directly with
# `scripts/measure_idle_floor.py`: a machine doing nothing shows 33.6% worst-case
# disagreement between two sampling windows, so an idle laptop trips a 15% gate.
#
# This ratio is dimensionally correct and accelerator-agnostic:
#   |idle_after - idle_before| x row_duration / row_energy
# Measured across every shipped record: H100 0.32-2.13%, clean Apple Silicon
# runs 0.24-2.31%, and the one Apple Silicon run whose decode coefficient came
# out 19% wrong scores 40.2%. See kudzu/DIARY.md.
MAX_IDLE_UNCERTAINTY_ROW_SHARE_PCT = 10.0
# A batched record skips nested-OLS identifiability only. It never skips
# provenance sanity, so it must still look like something a real run produced.
BATCHED_MIN_SAMPLES = 8
# Likewise for explained variance. A rung that explains nothing is not a
# source of identified coefficients.
MIN_R2 = 0.85

STATUS_IDENTIFIED = "identified"
STATUS_NOT_IDENTIFIED = "not_identified"
STATUS_REJECTED = "rejected"

def _rung_name(labels: Sequence[str]) -> str:
    """Canonical rung name. The four chain members keep their historic strings."""
    terms = set(labels)
    if terms == {"output"}:
        return "output_only"
    ordered = ["output"] + [t for t in ("intercept", "input", "image") if t in terms]
    return "_".join(ordered)


RUNG_OUTPUT_ONLY = "output_only"
RUNG_OUTPUT_INTERCEPT = "output_intercept"
RUNG_OUTPUT_INTERCEPT_INPUT = "output_intercept_input"
RUNG_FULL = "output_intercept_input_image"


def terms_for_rung(rung_name: str | None) -> frozenset[str]:
    """Return the canonical coefficient terms represented by a rung name.

    Rung names are deliberately stable interchange strings. Comparing their
    underscore-token counts is not sufficient: ``output_intercept`` and
    ``output_input`` both contain two tokens but fit different models.
    """
    if not isinstance(rung_name, str) or not rung_name:
        return frozenset()
    axis_for_label = {
        "output": "wh_per_1k_output",
        "intercept": "intercept_wh",
        "input": "wh_per_1k_input",
        "image": "wh_per_image",
    }
    # `_rung_name` spells the single-term rung "output_only", so splitting on
    # "_" yields a token ("only") that names no axis. Without this inversion
    # the most common published rung in the whole record set decodes to the
    # empty set -- indistinguishable from a malformed name -- and a record
    # conservatively publishing output-only is reported as an over-claim.
    if rung_name == RUNG_OUTPUT_ONLY:
        return frozenset({axis_for_label["output"]})
    labels = set(rung_name.split("_"))
    if "output" not in labels or not labels <= set(axis_for_label):
        return frozenset()
    return frozenset(axis_for_label[label] for label in labels)

PROVIDER_KINDS = (
    "marketplace",
    "hyperscaler",
    "sovereign_cloud",
    "on_premise",
    "academic_hpc",
)
TENANCIES = ("dedicated_host", "shared_host", "unknown")
GPU_FORM_FACTORS = ("sxm", "pcie", "soc", "unknown")
PUE_BASES = ("annualised", "seasonal", "instantaneous", "design")
GRID_ACCOUNTING_CONVENTIONS = (
    "location_based_average",
    "location_based_marginal",
    "market_based",
)
WATER_BOUNDARIES = ("site", "source")
WATER_ACCOUNTING_KINDS = (
    "consumption",
    "withdrawal",
    "cooling_and_humidification_water",
)


def idle_uncertainty_row_share_pct(
    rows: Sequence[Mapping[str, Any]],
    idle_watts_before: Any,
    idle_watts_after: Any,
) -> float | None:
    """Idle-subtraction uncertainty as a percentage of a typical row's energy.

    The two idle windows disagree by some amount; that disagreement is the
    uncertainty in what gets subtracted from every row. Expressed against the
    energy actually measured, it says whether an unstable baseline threatens the
    coefficients -- which a percentage of idle itself cannot, because it does not
    know how large idle is relative to the workload.

    Duration and energy are both taken at window level so an aggregated row
    (several requests in one timing window) stays consistent on both sides.
    The median across rows resists a single anomalous window.

    Returns None when the record cannot support the calculation.
    """
    before, after = _finite_or_none(idle_watts_before), _finite_or_none(idle_watts_after)
    if before is None or after is None:
        return None
    delta_w = abs(after - before)
    shares: list[float] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        duration_ms = _finite_or_none(row.get("duration_ms"))
        energy_wh = _finite_or_none(row.get("energy_wh"))
        if not duration_ms or not energy_wh or duration_ms <= 0 or energy_wh <= 0:
            continue
        shares.append(100.0 * (delta_w * (duration_ms / 3_600_000.0)) / energy_wh)
    if not shares:
        return None
    shares.sort()
    mid = len(shares) // 2
    if len(shares) % 2:
        return shares[mid]
    return (shares[mid - 1] + shares[mid]) / 2.0


def _finite_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def gate_parameters() -> dict[str, Any]:
    """The thresholds a verdict was reached under.

    Recorded with every record so ``identified`` means the same thing when it
    is read back. Without this, a threshold change silently reinterprets every
    record already in the wild, and an operator who disagrees with a value has
    no way to show the disagreement other than argument. With it they can
    re-run against their own numbers and point at the difference.

    These values are deliberately conservative starting points, not derived
    constants. They are expected to move as measurements accumulate, which is
    precisely why the record has to say which ones it was judged against.
    """
    return {
        "protocol_version": CALIBRATION_PROTOCOL_VERSION,
        "max_relative_ci_half_width": MAX_RELATIVE_CI_HALF_WIDTH,
        "max_condition_number": MAX_CONDITION_NUMBER,
        "max_idle_uncertainty_row_share_pct": MAX_IDLE_UNCERTAINTY_ROW_SHARE_PCT,
        "min_r2": MIN_R2,
        # Pre-run design planning only; these do not decide a published verdict.
        "design_leverage_multiple": LEVERAGE_MULTIPLE,
        "design_idle_integration_uncertainty": IDLE_INTEGRATION_UNCERTAINTY,
        "prefill_arm_min_input_tokens": PREFILL_ARM_MIN_INPUT_TOKENS,
        "prefill_arm_max_output_tokens": PREFILL_ARM_MAX_OUTPUT_TOKENS,
        "prefill_arm_min_rows": PREFILL_ARM_MIN_ROWS,
        "prefill_arm_min_span_tokens": PREFILL_ARM_MIN_SPAN_TOKENS,
        "image_arm_min_rows_each": IMAGE_ARM_MIN_ROWS_EACH,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def gate_divergence(stored: Mapping[str, Any] | None) -> dict[str, Any]:
    """Which gate parameters differ between a stored verdict and this build."""
    if not isinstance(stored, Mapping):
        return {}
    current = gate_parameters()
    return {
        key: {"recorded": stored.get(key), "current": value}
        for key, value in current.items()
        if key in stored and stored.get(key) != value
    }


@dataclass
class AxisFit:
    name: str
    coefficient: float | None
    ci95: tuple[float, float] | None
    status: str
    reason: str
    leverage: float | None
    has_design_leverage: bool


@dataclass
class NestedFitResult:
    """Published nested model plus per-axis status."""

    published_rung: str
    intercept_wh: float | None
    wh_per_1k_input: float | None
    wh_per_1k_output: float | None
    wh_per_image: float | None
    status: dict[str, str]
    null_reasons: dict[str, str]
    r2: float
    condition_number: float
    intercept_ci95: tuple[float, float] | None
    input_ci95: tuple[float, float] | None
    output_ci95: tuple[float, float] | None
    image_ci95: tuple[float, float] | None
    residuals_structured: bool
    active: bool
    rejection_reasons: list[str]
    axes: dict[str, AxisFit]
    design_leverage: dict[str, Any]
    energy_basis: dict[str, Any]
    admitted_rows: int
    dropped_zero_raw: int
    # Impossible-sign axes that sit OUTSIDE the published rung. They do not
    # contaminate the published numbers, so they do not deactivate the record,
    # but they are a measurement-quality signal and must stay visible. Their
    # per-axis status and reason also appear in `status` / `null_reasons`.
    unpublished_rejections: list[str] = field(default_factory=list)
    dropped_malformed: int = 0
    #: Rows dropped for falling under the power-sampling duration floor. Only
    #: ever non-zero on a sampling path, and always a design-sizing problem:
    #: the row aggregation exists to keep short points above this floor.
    dropped_under_floor: int = 0
    rungs: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Aliases so measurement_provenance_core can treat this like FitResult.
    @property
    def valid(self) -> bool:
        return self.active

    @property
    def invalid_reasons(self) -> list[str]:
        return list(self.rejection_reasons)


def require_numpy(extra: str = "calibrate") -> Any:
    """Import NumPy or raise a CLI-friendly error naming the right extra.

    The fit is shared across accelerators but the install path is not: CUDA
    users want vetch[calibrate], Apple Silicon users vetch[apple-silicon].
    Telling someone to install the wrong one is a dead end.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise RuntimeError(
            "Calibration identifiability requires NumPy. Install the "
            f"calibration extra: pip install 'vetch[{extra}]'"
        ) from e
    return np


def _ci95(vals: Sequence[float]) -> tuple[float, float] | None:
    if not vals:
        return None
    s = sorted(vals)
    lo = s[int(len(s) * 0.025)]
    hi = s[int(len(s) * 0.975)]
    return (float(lo), float(hi))




def detect_residuals_structured(
    runs: Sequence[Mapping[str, Any]],
    residuals: Sequence[float],
) -> bool:
    """True when residual means differ in sign across workload groups.

    This is a misspecification flag on the published linear model. It does not
    by itself discard an identified decode slope.
    """
    if len(runs) < 8 or len(residuals) != len(runs):
        return False

    def _group_key(run: Mapping[str, Any], factor: str) -> Any:
        if factor == "n_images":
            return 1 if _per_request(run, "n_images") > 0 else 0
        # Per request, not the row total. A row may aggregate N identical
        # requests, so a prefill point pinned at 8 output tokens carries a
        # total of 8*N and lands in the long-output group next to real decode
        # rows -- which is exactly the contrast this detector exists to draw.
        # On the batch-2 record that misgrouped 12 of 32 rows.
        return 0 if _per_request(run, "output_tokens") < 50 else 1

    for factor in ("n_images", "output_bucket"):
        groups: dict[Any, list[float]] = {}
        for run, residual in zip(runs, residuals):
            groups.setdefault(_group_key(run, factor), []).append(float(residual))
        if len(groups) < 2:
            continue
        means = [sum(vals) / len(vals) for vals in groups.values() if vals]
        if len(means) < 2 or max(means) - min(means) <= 0.0005:
            continue
        signs = {1 if m > 0 else -1 if m < 0 else 0 for m in means}
        if 0 in signs or len(signs) < 2:
            continue
        return True
    return False


def _finite_field(row: Mapping[str, Any], key: str) -> float | None:
    """A row's numeric field, or None when it is present but not a finite number.

    A missing field is a legitimate zero (``n_images`` on a text-only design).
    A field that is present and is null, a string, or NaN is corruption, and
    admitting it lets ``float(None)`` raise from inside the fit. ``fit_record``
    runs on third-party files via ``validate-calibration``, so a malformed
    record has to come back as a failed report, never as a traceback.
    """
    if key not in row:
        return 0.0
    value = row.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


# Fields the fit reads off every admitted row. `energy_wh` may legitimately be
# 0.0 (a net figure clamped when idle draw approached the run's total), so it
# is checked for finiteness, not positivity.
_REQUIRED_ROW_FIELDS = (
    "energy_wh", "output_tokens", "text_tokens", "n_images", "requests",
)


def admit_rows(
    table: Sequence[Mapping[str, Any]],
    *,
    energy_source: str | None = None,
) -> tuple[list[dict[str, Any]], int, int, int]:
    """Drop corrupt rows only. Do not apply a 30% idle gate or a 2s counter floor.

    A 30% idle-fraction rule empties four of the seven batch-1 records. A 2
    second floor empties three. Short rows are how decode was identified.

    The duration floor applies only when ``energy_source`` is a power-sampling
    fallback, where nvidia-smi sampling gaps are a real hazard.

    Returns ``(admitted, dropped_zero_raw, dropped_malformed, dropped_under_floor)``.
    """
    admitted: list[dict[str, Any]] = []
    dropped_zero_raw = 0
    dropped_malformed = 0
    dropped_under_floor = 0
    sampling = bool(
        energy_source and "power" in energy_source.casefold()
        and "counter" not in energy_source.casefold()
        and "energy_counter" not in energy_source.casefold()
    )
    for raw in table:
        if not isinstance(raw, Mapping):
            dropped_malformed += 1
            continue
        row = dict(raw)
        try:
            raw_wh = float(row.get("raw_energy_wh") or 0.0)
        except (TypeError, ValueError):
            dropped_zero_raw += 1
            continue
        if not math.isfinite(raw_wh) or raw_wh <= 0.0:
            dropped_zero_raw += 1
            continue
        if any(_finite_field(row, key) is None for key in _REQUIRED_ROW_FIELDS):
            dropped_malformed += 1
            continue
        # ``duration_ms`` is optional in older interchange records, but when a
        # writer supplies it every diagnostic path below reads it. Validate it
        # here so corrupt third-party rows are reported as dropped rather than
        # escaping later as ``float('bad')``.
        if "duration_ms" in row and _finite_field(row, "duration_ms") is None:
            dropped_malformed += 1
            continue
        if "requests" in row and (_finite_field(row, "requests") or 0.0) <= 0.0:
            # `requests` is a divisor. `_requests` already floors it at 1, but a
            # stored 0 means the writer did not know how many requests the
            # window held, which is not the same as one.
            dropped_malformed += 1
            continue
        if sampling:
            try:
                duration_ms = float(row.get("duration_ms") or 0.0)
            except (TypeError, ValueError):
                dropped_under_floor += 1
                continue
            if duration_ms < POWER_SAMPLING_DURATION_FLOOR_MS:
                # Counted, not silent. This is the row aggregation's whole
                # purpose: a prefill point under the floor is a design point
                # lost, and losing the short end collapses the arm's span. An
                # uncounted `continue` here meant a mis-sized grid degraded the
                # fit with nothing anywhere saying rows had gone missing.
                dropped_under_floor += 1
                continue
        admitted.append(row)
    return admitted, dropped_zero_raw, dropped_malformed, dropped_under_floor


def _idle_watts(provenance: Mapping[str, Any]) -> float | None:
    """Mean idle draw, or None when the record never recorded one.

    None is not zero. Without an idle figure there is no sigma, and without a
    sigma every leverage test silently passes -- which would bless a design
    that resolves nothing.
    """
    before = provenance.get("idle_watts_before")
    after = provenance.get("idle_watts_after")
    vals = [
        float(v) for v in (before, after)
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    if not vals:
        return None
    return sum(vals) / len(vals)


def idle_relative_uncertainty(provenance: Mapping[str, Any] | None) -> float:
    """Relative idle-subtraction uncertainty for one record.

    Two idle samples taken either side of a run disagree by some amount, and
    that disagreement is a direct observation of how unstable the baseline was.
    It is a *lower* bound on the true uncertainty, because two endpoints cannot
    see within-run drift, sensor error, or thermal transients during the runs
    themselves. So it raises the figure when idle was demonstrably unstable and
    never lowers it below the floor.

    Measured across the shipped records: nested runs disagree by 0.45-1.35%,
    batched runs by 3-15% (longer, hotter, more thermal drift). One constant
    cannot serve both.
    """
    floor = IDLE_INTEGRATION_UNCERTAINTY
    if not isinstance(provenance, Mapping):
        return floor
    before = provenance.get("idle_watts_before")
    after = provenance.get("idle_watts_after")
    values = [
        float(v) for v in (before, after)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0
    ]
    if len(values) < 2:
        return floor
    mean = sum(values) / len(values)
    if mean <= 0:
        return floor
    observed_half_spread = abs(values[1] - values[0]) / 2.0 / mean
    return max(floor, observed_half_spread)


def sigma_idle(
    rows: Sequence[Mapping[str, Any]],
    idle_watts: float | None,
    *,
    relative: float = IDLE_INTEGRATION_UNCERTAINTY,
) -> float:
    """Idle-subtraction uncertainty over exactly the rows given.

    Callers pass the rows that inform a particular axis. Averaging over the
    whole design charges a short-row axis for noise it never incurred: in the
    v2 grid the prefill arm runs at 200 ms while decode isolation runs to 17 s,
    so a global mean overstates prefill noise by ~20x and makes an adequate
    design look marginal.

    DIAGNOSTIC ONLY on the post-fit path. No published verdict is decided by
    this number -- realised precision replaced predicted noise in 283e725. It
    is reported so a reader can see the noise scale each axis was measured
    against, and it still gates the pre-run design planner, which has no
    residuals to work from.
    """
    if not rows or not idle_watts or idle_watts <= 0:
        return 0.0
    energies = [
        idle_watts * (float(r.get("duration_ms") or 0.0) / 3_600_000.0)
        for r in rows
    ]
    return relative * (sum(energies) / len(energies))


def _rows_informing_input(rows: Sequence[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]:
    """Prefill-isolation rows if the design has them, else the whole design."""
    # Per request, matching `_prefill_arm_count`. Reading row totals here made
    # an aggregated prefill row fail the output<=8 test and silently fall back
    # to the whole design, which is the noise scale this function exists to
    # avoid importing.
    arm = [
        r for r in rows
        if _per_request(r, "text_tokens") >= PREFILL_ARM_MIN_INPUT_TOKENS
        and _per_request(r, "output_tokens") <= PREFILL_ARM_MAX_OUTPUT_TOKENS
    ]
    return arm if len(arm) >= PREFILL_ARM_MIN_ROWS else rows


def _ols_with_bootstrap(
    x_cols: list[list[float]],
    y: list[float],
    np: Any,
) -> tuple[list[float], float, float, list[tuple[float, float] | None], list[float]]:
    X = np.column_stack(x_cols) if x_cols else np.ones((len(y), 0))
    yv = np.asarray(y, dtype=float)
    n, k = X.shape
    if k == 0 or n == 0:
        return [], 0.0, float("inf"), [], []
    coeffs, _, _, _ = np.linalg.lstsq(X, yv, rcond=None)
    yhat = X @ coeffs
    ss_res = float(np.sum((yv - yhat) ** 2))
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    if k >= 2:
        # Standardise EVERY column, not columns 1: . The v1 fit could slice
        # off column 0 because the intercept was structurally first and always
        # present; the v2 lattice makes the intercept optional, so on an
        # intercept-free rung column 0 is a real predictor and leaving it in
        # raw units makes the condition number a function of the unit the
        # predictor happens to be expressed in. Measured on the shipped
        # records: the batch-2 output_input rung reported 3.10 against a true
        # 1.40 (and 3084 had input been carried in raw tokens rather than
        # per-1k), and the vision output_image rung reported 26.77 against a
        # true 1.69 -- 11% below a reject gate, for a near-orthogonal design.
        # The constant intercept column is handled by the same near-zero guard
        # that already protected any other degenerate column.
        col_stds = X.std(axis=0)
        col_stds = np.where(col_stds < 1e-10, 1.0, col_stds)
        cond = float(np.linalg.cond(X / col_stds))
    else:
        cond = 1.0
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boots: list[Any] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        idx = rng.integers(0, n, size=n)
        try:
            cb, *_ = np.linalg.lstsq(X[idx], yv[idx], rcond=None)
            boots.append(cb)
        except Exception:
            continue
    cis: list[tuple[float, float] | None] = []
    boot_arr = np.asarray(boots) if boots else np.zeros((0, k))
    for j in range(k):
        cis.append(_ci95(boot_arr[:, j].tolist()) if len(boot_arr) else None)
    return [float(c) for c in coeffs], r2, cond, cis, [float(v) for v in (yv - yhat)]


def _requests(row: Mapping[str, Any]) -> float:
    """Requests aggregated into this row's timing window (at least 1)."""
    try:
        count = float(row.get("requests") or 1)
    except (TypeError, ValueError):
        return 1.0
    return count if count > 0 else 1.0


def _per_request(row: Mapping[str, Any], key: str) -> float:
    """A row's value for one request.

    A row may aggregate several identical requests into one timing window so
    that a short design point still occupies a measurable window. Totals are
    what the regression needs; the design point is what the arm checks describe,
    and that is per request.
    """
    try:
        total = float(row.get(key) or 0)
    except (TypeError, ValueError):
        return 0.0
    return total / _requests(row)


def _prefill_arm_count(rows: Sequence[Mapping[str, Any]]) -> int:
    n = 0
    for r in rows:
        text = _per_request(r, "text_tokens")
        out = _per_request(r, "output_tokens")
        if text >= PREFILL_ARM_MIN_INPUT_TOKENS and out <= PREFILL_ARM_MAX_OUTPUT_TOKENS:
            n += 1
    return n


def _token_span(rows: Sequence[Mapping[str, Any]], key: str) -> tuple[float, float, float]:
    """Span of the per-request design points, not of the aggregated totals."""
    vals = [_per_request(r, key) for r in rows]
    if not vals:
        return 0.0, 0.0, 0.0
    return min(vals), max(vals), max(vals) - min(vals)


def input_design_leverage(rows: Sequence[Mapping[str, Any]]) -> tuple[bool, str]:
    """Leverage on prefill is a design property. The interval does not decide it."""
    if not rows:
        return False, "no admitted rows"
    _lo, hi, span = _token_span(rows, "text_tokens")
    arm = _prefill_arm_count(rows)
    if hi < PREFILL_ARM_MIN_INPUT_TOKENS or span < PREFILL_ARM_MIN_SPAN_TOKENS:
        return (
            False,
            f"design lacked prefill isolation arm "
            f"(max input {hi:.0f} < {PREFILL_ARM_MIN_INPUT_TOKENS} or "
            f"span {span:.0f} < {PREFILL_ARM_MIN_SPAN_TOKENS})",
        )
    if arm < PREFILL_ARM_MIN_ROWS:
        return (
            False,
            f"design lacked prefill isolation arm "
            f"({arm} rows with input>={PREFILL_ARM_MIN_INPUT_TOKENS} and "
            f"output<={PREFILL_ARM_MAX_OUTPUT_TOKENS}; need {PREFILL_ARM_MIN_ROWS})",
        )
    return True, "prefill isolation arm present"


def image_design_leverage(rows: Sequence[Mapping[str, Any]]) -> tuple[bool, str]:
    # Per request, matching the regression's image column and the other arm
    # checks. Aggregation preserves the sign of `n_images`, so this reads the
    # same for every design the v2 grid emits; it diverges only for a row that
    # mixes image and text-only requests inside one timing window, which the
    # grid never produces and a third-party table might.
    with_img = sum(1 for r in rows if _per_request(r, "n_images") > 0)
    without = sum(1 for r in rows if _per_request(r, "n_images") <= 0)
    if with_img < IMAGE_ARM_MIN_ROWS_EACH or without < IMAGE_ARM_MIN_ROWS_EACH:
        return (
            False,
            f"design lacked image arm ({with_img} with-image rows, {without} without; "
            f"need {IMAGE_ARM_MIN_ROWS_EACH} each)",
        )
    return True, "image arm present"


def _relative_half_width(
    value: float | None, ci: tuple[float, float] | None
) -> float | None:
    """Bootstrap CI half-width as a fraction of the estimate."""
    if value is None or ci is None or value == 0:
        return None
    return ((ci[1] - ci[0]) / 2.0) / abs(value)


def _axis_status(
    *,
    name: str,
    value: float | None,
    ci: tuple[float, float] | None,
    has_design_leverage: bool,
    design_reason: str,
) -> AxisFit:
    """Decide an axis on what the data actually determined.

    A coefficient is published when the design could resolve it at all and the
    fit pinned it to within MAX_RELATIVE_CI_HALF_WIDTH. The bootstrap interval
    is the realised standard error and already accounts for design span, sample
    size, collinearity and which rows inform this axis, which is what
    (X'X)^-1_jj encodes. Judging a measured fit against a *predicted* noise
    level instead was the defect this replaces.
    """
    precision = _relative_half_width(value, ci)
    if not has_design_leverage:
        return AxisFit(
            name=name, coefficient=None, ci95=ci,
            status=STATUS_NOT_IDENTIFIED, reason=design_reason,
            leverage=precision, has_design_leverage=False,
        )
    if value is None or ci is None:
        return AxisFit(
            name=name, coefficient=None, ci95=ci,
            status=STATUS_NOT_IDENTIFIED, reason="no interval for this axis",
            leverage=None, has_design_leverage=True,
        )
    # Rejection requires CONFIDENCE that the sign is impossible, which means the
    # whole interval below zero. A slightly negative point estimate whose
    # interval spans zero is blind, not wrong, and condemning it would turn an
    # unidentified axis into a whole-record rejection -- the exact outcome this
    # protocol exists to avoid. Such an axis fails the precision test below
    # anyway, since an interval containing zero is at least 100% wide.
    if ci[1] < 0.0:
        return AxisFit(
            name=name, coefficient=None, ci95=ci,
            status=STATUS_REJECTED,
            reason=(
                f"entire interval below zero, so the sign is impossible "
                f"(value={value:.6g}, CI={list(ci)})"
            ),
            leverage=precision, has_design_leverage=True,
        )
    if precision is None or precision > MAX_RELATIVE_CI_HALF_WIDTH:
        shown = "undefined" if precision is None else f"{precision * 100:.0f}%"
        return AxisFit(
            name=name, coefficient=None, ci95=ci,
            status=STATUS_NOT_IDENTIFIED,
            reason=(
                f"coefficient determined only to +/-{shown}, worse than the "
                f"+/-{MAX_RELATIVE_CI_HALF_WIDTH * 100:.0f}% publication target"
            ),
            leverage=precision, has_design_leverage=True,
        )
    return AxisFit(
        name=name, coefficient=float(value), ci95=ci,
        status=STATUS_IDENTIFIED,
        reason=(
            f"determined to +/-{precision * 100:.1f}%, sign physical, "
            "design resolves this axis"
        ),
        leverage=precision, has_design_leverage=True,
    )


def _energy_leverage(span_predictor: float, slope: float | None, sigma: float) -> float | None:
    if slope is None or sigma <= 0:
        return None
    return abs(slope) * abs(span_predictor) / sigma


def attributed_levels(
    *,
    marginal_wh_per_1k_output: float | None,
    rows: Sequence[Mapping[str, Any]],
    idle_watts: float,
) -> list[dict[str, Any]]:
    """Idle amortised over measured tokens at the protocol utilisation levels.

    This is still not fleet overhead: provisioned-but-idle capacity, MIG
    partitions held for isolation, and autoscaling headroom are outside both
    bases. The coefficients are a floor, not an operational estimate.
    """
    levels: list[dict[str, Any]] = []
    if not rows or idle_watts <= 0 or marginal_wh_per_1k_output is None:
        return [
            {"assumed_utilisation": u, "wh_per_1k_output": None}
            for u in ATTRIBUTED_UTILISATION_LEVELS
        ]
    total_out = sum(float(r.get("output_tokens") or 0) for r in rows)
    total_h = sum(float(r.get("duration_ms") or 0.0) for r in rows) / 3_600_000.0
    if total_out <= 0 or total_h <= 0:
        return [
            {"assumed_utilisation": u, "wh_per_1k_output": None}
            for u in ATTRIBUTED_UTILISATION_LEVELS
        ]
    hours_per_1k = 1000.0 * total_h / total_out
    for u in ATTRIBUTED_UTILISATION_LEVELS:
        # Idle draw that would be attributed if the device spent fraction u busy.
        idle_wh_per_1k = idle_watts * hours_per_1k / u
        levels.append({
            "assumed_utilisation": u,
            "wh_per_1k_output": float(marginal_wh_per_1k_output + idle_wh_per_1k),
        })
    return levels


def fit_nested_models(
    table: Sequence[Mapping[str, Any]],
    *,
    idle_watts: float | None,
    energy_source: str | None = None,
    visual_tokens_per_image: int = 0,
    idle_relative: float = IDLE_INTEGRATION_UNCERTAINTY,
) -> NestedFitResult:
    """Fit nested OLS models and publish the largest identified rung.

    Rungs: output only; output+intercept; output+intercept+input;
    output+intercept+input+image. Prefer output+intercept over output only
    whenever the intercept is identified. A negative input coefficient on a
    design with no prefill arm is ``not_identified``, not a whole-record reject.
    ``residuals_structured`` is a misspecification flag, not a reject.
    """
    np = require_numpy()
    rows, dropped_zero_raw, dropped_malformed, dropped_under_floor = admit_rows(
        table, energy_source=energy_source
    )
    empty_status = {
        "wh_per_1k_output": STATUS_NOT_IDENTIFIED,
        "intercept_wh": STATUS_NOT_IDENTIFIED,
        "wh_per_1k_input": STATUS_NOT_IDENTIFIED,
        "wh_per_image": STATUS_NOT_IDENTIFIED,
    }
    def _barren(reason: str) -> NestedFitResult:
        """No axis can be identified. Publish nothing and say why."""
        return NestedFitResult(
            published_rung=RUNG_OUTPUT_ONLY,
            intercept_wh=None,
            wh_per_1k_input=None,
            wh_per_1k_output=None,
            wh_per_image=None,
            status=empty_status,
            null_reasons=dict.fromkeys(empty_status, reason),
            r2=0.0,
            condition_number=float("inf"),
            intercept_ci95=None,
            input_ci95=None,
            output_ci95=None,
            image_ci95=None,
            residuals_structured=False,
            active=False,
            rejection_reasons=[reason],
            axes={},
            design_leverage={},
            energy_basis=_empty_energy_basis(),
            admitted_rows=len(rows),
            dropped_zero_raw=dropped_zero_raw,
            dropped_malformed=dropped_malformed,
            dropped_under_floor=dropped_under_floor,
        )

    if len(rows) < 3:
        return _barren(f"admitted rows {len(rows)} < 3")
    # Missing idle is fatal for identifiability, never a free pass. With no
    # idle figure sigma is zero, every leverage ratio is undefined, and the
    # design gate would wave through a table with no span on any axis.
    if idle_watts is None:
        return _barren(
            "no idle measurement in provenance (idle_watts_before/after); "
            "design leverage cannot be predicted, so no axis is identifiable"
        )
    if idle_watts <= 0:
        return _barren(
            f"idle_watts={idle_watts!r} is not positive; design leverage "
            "cannot be predicted, so no axis is identifiable"
        )

    # Normalise to ONE request. A row may aggregate several identical requests
    # so a short design point still occupies a measurable window; the design
    # point is the per-request one, and dividing through restores it. Sizing
    # repeats to a fixed window makes TOTAL input nearly constant across the
    # prefill arm -- 98648, 93450, 92805 tokens for points meant to span 1k to
    # 16k -- which collapses the very span the arm exists to create.
    y = [float(r["energy_wh"]) / _requests(r) for r in rows]
    out_k = [_per_request(r, "output_tokens") / 1000.0 for r in rows]
    in_k = [
        max(0.0, _per_request(r, "text_tokens")
            - _per_request(r, "n_images") * visual_tokens_per_image) / 1000.0
        for r in rows
    ]
    n_img = [_per_request(r, "n_images") for r in rows]
    # Rows are per-request after normalisation, so the intercept is the usual
    # column of ones rather than a request count.
    ones = [1.0] * len(rows)
    has_image_feature = any(v > 0 for v in n_img)
    in_span_k = max(in_k) - min(in_k) if in_k else 0.0
    out_span_k = max(out_k) - min(out_k) if out_k else 0.0

    in_lev_ok, in_lev_reason = input_design_leverage(rows)
    img_lev_ok, img_lev_reason = image_design_leverage(rows)

    # The four-rung chain cannot express "decode and image identified, prefill
    # not", which is the state of every vision model measured without a prefill
    # arm: an unidentified input term would drag the image term down with it and
    # force a fallback to output-only, a model that is actively wrong when
    # images carry energy. Optional axes are independent, so enumerate the
    # lattice over them instead of a single chain. Names for the four chain
    # members are unchanged.
    column_of = {"intercept": ones, "input": in_k, "output": out_k, "image": n_img}
    optional = ["intercept", "input"] + (["image"] if has_image_feature else [])
    rungs_spec: list[tuple[str, list[str], list[list[float]]]] = []
    for mask in range(1 << len(optional)):
        chosen = {optional[i] for i in range(len(optional)) if mask & (1 << i)}
        labels: list[str] = [t for t in ("intercept", "input", "output", "image")
                             if t == "output" or t in chosen]
        rungs_spec.append(
            (_rung_name(labels), labels, [column_of[t] for t in labels])
        )

    fitted: dict[str, dict[str, Any]] = {}
    for rung_name, labels, cols in rungs_spec:
        coeffs, r2, cond, cis, resid = _ols_with_bootstrap(cols, y, np)
        by = {lab: coeffs[i] for i, lab in enumerate(labels)}
        ci_by = {lab: cis[i] for i, lab in enumerate(labels)}
        fitted[rung_name] = {
            "labels": labels,
            "coeffs": by,
            "ci": ci_by,
            "r2": r2,
            "condition_number": cond,
            "residuals": resid,
        }

    # A rung is usable only if its own design matrix is well conditioned and it
    # explains the data. Status for an axis is decided on the rung that first
    # contains it, so an unusable rung must not hand out an "identified".
    rung_quality: dict[str, dict[str, Any]] = {}
    for rung_name, body in fitted.items():
        cond = float(body["condition_number"])
        r2 = float(body["r2"])
        problems: list[str] = []
        if not math.isfinite(cond) or cond > MAX_CONDITION_NUMBER:
            problems.append(
                f"condition number {cond:.4g} > {MAX_CONDITION_NUMBER:.0f} "
                "(design is collinear; terms cannot be separated)"
            )
        if not math.isfinite(r2) or r2 < MIN_R2:
            problems.append(f"r2 {r2:.4g} < {MIN_R2}")
        rung_quality[rung_name] = {
            "ok": not problems,
            "condition_number": cond,
            "r2": r2,
            "problems": problems,
        }

    def _rung_reason(rung_name: str) -> str:
        return f"{rung_name} rung unusable: " + "; ".join(
            rung_quality[rung_name]["problems"]
        )

    AXIS_LABEL = {
        "wh_per_1k_output": "output",
        "intercept_wh": "intercept",
        "wh_per_1k_input": "input",
        "wh_per_image": "image",
    }

    # Design leverage is a property of the design, known before any fit, so
    # using it to seed the assessment model is not circular.
    design_leverage_ok = {
        "intercept": len(rows) >= 8,
        "input": in_lev_ok,
        "image": img_lev_ok if has_image_feature else False,
    }
    design_reason_of = {
        "intercept": "n < 8; intercept lacks leverage",
        "input": in_lev_reason,
        "image": img_lev_reason if has_image_feature
                 else "no image feature in the design",
    }

    def _best_rung_for(label: str, allowed: set[str]) -> str | None:
        """Largest usable rung containing this term and nothing outside ``allowed``.

        A term must be judged inside a model whose every other term the design
        actually resolves. Omitting a live term judges a coefficient on a fit
        that is wrong for unrelated reasons -- drop a real image term and the
        output-only fit goes to a negative r2, condemning a decode slope the
        data determines perfectly well. Including a term the design *cannot*
        resolve is just as bad, and batch 1 is the proof: judged inside the
        full model, an unresolvable prefill term drags Gemma's intercept
        negative and turns a blind axis into a whole-record rejection, which is
        the exact outcome v2 exists to stop.
        """
        candidates = [
            name for name, body in fitted.items()
            if label in body["labels"]
            and rung_quality[name]["ok"]
            and all(lbl == "output" or lbl in allowed for lbl in body["labels"])
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda n: (len(fitted[n]["labels"]), n))

    axis_rung: dict[str, str] = {}

    def _assess(axis: str, allowed: set[str]) -> AxisFit:
        label = AXIS_LABEL[axis]
        has_leverage = True if label == "output" else design_leverage_ok[label]
        design_reason = "output axis" if label == "output" else design_reason_of[label]
        rung = _best_rung_for(label, allowed)
        if rung is None:
            containing = [n for n, b in fitted.items() if label in b["labels"]]
            reason = (
                design_reason if not has_leverage
                else _rung_reason(
                    sorted(containing, key=lambda n: len(fitted[n]["labels"]))[-1]
                ) if containing else "no rung contains this term"
            )
            return AxisFit(
                name=axis, coefficient=None, ci95=None,
                status=STATUS_NOT_IDENTIFIED, reason=reason,
                leverage=None, has_design_leverage=False,
            )
        value = float(fitted[rung]["coeffs"][label])
        ci = fitted[rung]["ci"][label]
        # The intercept is a fixed cost, not a slope over a predictor, so its
        # leverage is sample size rather than |beta|/sigma. Using the point
        # estimate as its own span would be circular.
        result = _axis_status(
            name=axis, value=value, ci=ci,
            has_design_leverage=has_leverage,
            design_reason=design_reason,
        )
        axis_rung[axis] = rung
        return result

    # Prune blind terms one round at a time and re-assess the survivors inside
    # the smaller model. Only ``not_identified`` terms are pruned: a
    # ``rejected`` verdict reached inside a contaminated model is exactly the
    # verdict the contamination would produce, so it is not grounds for
    # removal until the model has been cleaned up.
    allowed = {lbl for lbl, ok in design_leverage_ok.items() if ok}
    assessed: dict[str, AxisFit] = {}
    # A pruned term keeps the verdict from the round that actually judged it.
    # Re-deriving it afterwards, once no admissible rung contains it, replaces a
    # real finding ("determined only to +/-40%") with a vacuous one.
    pruned_verdicts: dict[str, AxisFit] = {}
    for _round in range(len(design_leverage_ok) + 1):
        axis_rung.clear()
        assessed = {
            axis: _assess(axis, allowed)
            for axis in ("wh_per_1k_output", "intercept_wh",
                         "wh_per_1k_input", "wh_per_image")
        }
        for axis, fit in assessed.items():
            if AXIS_LABEL[axis] in allowed:
                pruned_verdicts[axis] = fit
        blind_axes = [
            (axis, fit) for axis, fit in assessed.items()
            if AXIS_LABEL[axis] in allowed and fit.status == STATUS_NOT_IDENTIFIED
        ]
        if not blind_axes:
            break
        # Prune ONE term per round, worst-determined first, and re-assess the
        # rest inside the smaller model. Removing every blind term at once
        # freezes each of their verdicts in the same contaminated fit, which is
        # the failure `_best_rung_for` warns about: an unresolvable prefill
        # term drags the intercept negative, and dropping both together means
        # the intercept is never judged in the clean model that would have
        # cleared it. Worst-first so the term doing the most damage goes first.
        ordered = sorted(
            blind_axes,
            key=lambda pair: (
                -(pair[1].leverage if pair[1].leverage is not None else float("inf")),
                pair[0],
            ),
        )
        # Never prune a term whose removal destroys the model. A term can be
        # real and load-bearing while still being too weakly determined to
        # publish: dropping the image term here is what produced the negative
        # r2 in the first place. Such a term stays in as a nuisance term that
        # absorbs its own variance, and is still never published. If the worst
        # term is load-bearing, try the next one rather than giving up.
        for axis, _fit in ordered:
            candidate = allowed - {AXIS_LABEL[axis]}
            if _best_rung_for("output", candidate) is not None:
                allowed = candidate
                break
        else:
            break

    def _final(axis: str) -> AxisFit:
        fit = assessed[axis]
        earlier = pruned_verdicts.get(axis)
        if earlier is not None and fit.ci95 is None and earlier.ci95 is not None:
            return earlier  # keep the round that could actually see this term
        return fit

    output_axis = _final("wh_per_1k_output")
    intercept_axis = _final("intercept_wh")
    input_axis = _final("wh_per_1k_input")
    image_axis = _final("wh_per_image")

    axes = {
        "wh_per_1k_output": output_axis,
        "intercept_wh": intercept_axis,
        "wh_per_1k_input": input_axis,
        "wh_per_image": image_axis,
    }

    # Largest usable rung whose every included term is identified. Preferring
    # more terms encodes the plan's rule that output+intercept beats
    # output-only whenever the intercept is identified, and the lattice lets a
    # vision run publish decode+image when prefill is unidentifiable.
    LABEL_AXIS = {v: k for k, v in AXIS_LABEL.items()}

    def _terms_of(rung_name: str) -> tuple[str, ...]:
        return tuple(LABEL_AXIS[lbl] for lbl in fitted[rung_name]["labels"])

    published_rung = RUNG_OUTPUT_ONLY
    published_terms: tuple[str, ...] = ("wh_per_1k_output",)
    chose = False
    for rung_name in sorted(
        fitted, key=lambda n: (len(fitted[n]["labels"]), n), reverse=True
    ):
        if not rung_quality[rung_name]["ok"]:
            continue
        terms = _terms_of(rung_name)
        if all(axes[t].status == STATUS_IDENTIFIED for t in terms):
            published_rung = rung_name
            published_terms = terms
            chose = True
            break
    if not chose:
        # No rung has every term identified. Fall back to the model the output
        # axis was actually judged in rather than to output-only, which may be
        # an unusable fit, and publish only the identified terms within it.
        fallback = axis_rung.get("wh_per_1k_output")
        if fallback is not None:
            published_rung = fallback
            published_terms = tuple(
                t for t in _terms_of(fallback)
                if axes[t].status == STATUS_IDENTIFIED
            )

    rejected = [ax for ax in axes.values() if ax.status == STATUS_REJECTED]
    # A rejected term invalidates the record only if it is a term of the rung
    # the published numbers came out of. Then the impossible sign contaminated
    # the fit those numbers were read from, and they cannot be trusted. A
    # rejection on some other rung -- a negative image term while decode and
    # intercept publish out of `output_intercept` -- says nothing about the
    # published coefficients, and killing the whole record for it is the
    # whole-record death v2 exists to stop.
    #
    # Note this is the rung's TERM SET, not `published_terms`. The latter is
    # the publishable subset (identified only), so a rejected axis can never be
    # in it and the rule would be vacuous.
    published_rung_terms = set(_terms_of(published_rung))
    fatal = [ax for ax in rejected if ax.name in published_rung_terms]
    rejection_reasons = [ax.reason for ax in fatal]
    unpublished_rejections = [
        f"{ax.name}: {ax.reason}" for ax in rejected if ax.name not in published_rung_terms
    ]
    identified_in_published = [
        t for t in published_terms if axes[t].status == STATUS_IDENTIFIED
    ]
    active = bool(identified_in_published) and not fatal

    published_fit = fitted[published_rung]
    published_resid = published_fit["residuals"]
    residuals_structured = detect_residuals_structured(rows, published_resid)

    # Axis name -> key inside a rung's coeffs/ci dict. Status is decided on
    # nested rungs; published numbers must come from the published rung or a
    # through-origin slope is paired with an intercept that slope already ate.
    _rung_key = dict(AXIS_LABEL)

    def _pub(name: str) -> float | None:
        # published_terms falls back to output-only when no rung qualifies, so
        # membership alone is not consent to publish: the axis must also have
        # come back identified.
        if name not in published_terms:
            return None
        if axes[name].status != STATUS_IDENTIFIED:
            return None
        val = published_fit["coeffs"].get(_rung_key[name])
        if val is None:
            return None
        return float(val)

    def _pub_ci(name: str) -> tuple[float, float] | None:
        """CI from the published rung, else from the rung the axis was judged on."""
        if name in published_terms:
            ci = published_fit["ci"].get(_rung_key[name])
            return None if ci is None else (float(ci[0]), float(ci[1]))
        return axes[name].ci95

    null_reasons = {
        ax.name: ax.reason
        for ax in axes.values()
        if ax.status != STATUS_IDENTIFIED
    }
    status = {ax.name: ax.status for ax in axes.values()}

    energy_basis = {
        "marginal": {
            "wh_per_1k_output": _pub("wh_per_1k_output"),
            "idle_subtracted": True,
        },
        "attributed": {
            "attribution_rule": "idle_amortised_over_measured_tokens",
            "levels": attributed_levels(
                marginal_wh_per_1k_output=_pub("wh_per_1k_output"),
                rows=rows,
                idle_watts=idle_watts,
            ),
        },
    }

    # Observed idle-subtraction noise, per axis, on the rows that inform it.
    # Reported, never gated: `idle_relative` reached this function and was
    # ignored, so the measurement behind it ("nested runs disagree by
    # 0.45-1.35%, batched by 3-15%") changed nothing anywhere. Publishing it
    # keeps the observation and states plainly that it decides nothing.
    input_rows = _rows_informing_input(rows)
    design_leverage = {
        "idle_noise": {
            "relative": idle_relative,
            "relative_source": (
                "observed idle_watts_before/after half-spread"
                if idle_relative > IDLE_INTEGRATION_UNCERTAINTY
                else f"floor ({IDLE_INTEGRATION_UNCERTAINTY})"
            ),
            "sigma_idle_wh": sigma_idle(rows, idle_watts, relative=idle_relative),
            "sigma_idle_input_wh": sigma_idle(
                input_rows, idle_watts, relative=idle_relative
            ),
            "gates_any_verdict": False,
        },
        "rung_quality": {
            k: {"ok": v["ok"], "condition_number": v["condition_number"],
                "r2": v["r2"], "problems": list(v["problems"])}
            for k, v in rung_quality.items()
        },
        "max_condition_number": MAX_CONDITION_NUMBER,
        "max_idle_uncertainty_row_share_pct": MAX_IDLE_UNCERTAINTY_ROW_SHARE_PCT,
        "min_r2": MIN_R2,
        "max_relative_ci_half_width": MAX_RELATIVE_CI_HALF_WIDTH,
        "axis_assessed_on_rung": dict(axis_rung),
        "output": {
            "relative_ci_half_width": output_axis.leverage, "span_k": out_span_k,
        },
        "input": {
            "design_leverage": in_lev_ok,
            "reason": in_lev_reason,
            "relative_ci_half_width": input_axis.leverage,
            "span_k": in_span_k,
            "prefill_arm_rows": _prefill_arm_count(rows),
        },
        "intercept": {
            "relative_ci_half_width": intercept_axis.leverage, "n": len(rows),
        },
        "image": {
            "design_leverage": img_lev_ok,
            "reason": img_lev_reason,
            "relative_ci_half_width": image_axis.leverage,
        },
    }

    return NestedFitResult(
        published_rung=published_rung,
        intercept_wh=_pub("intercept_wh"),
        wh_per_1k_input=_pub("wh_per_1k_input"),
        wh_per_1k_output=_pub("wh_per_1k_output"),
        wh_per_image=_pub("wh_per_image"),
        status=status,
        null_reasons=null_reasons,
        r2=float(published_fit["r2"]),
        condition_number=float(published_fit["condition_number"]),
        intercept_ci95=_pub_ci("intercept_wh"),
        input_ci95=_pub_ci("wh_per_1k_input"),
        output_ci95=_pub_ci("wh_per_1k_output"),
        image_ci95=_pub_ci("wh_per_image"),
        residuals_structured=residuals_structured,
        active=active,
        rejection_reasons=rejection_reasons,
        axes=axes,
        design_leverage=design_leverage,
        energy_basis=energy_basis,
        admitted_rows=len(rows),
        dropped_zero_raw=dropped_zero_raw,
        unpublished_rejections=unpublished_rejections,
        dropped_malformed=dropped_malformed,
        dropped_under_floor=dropped_under_floor,
        rungs={
            name: {
                "r2": body["r2"],
                "coeffs": body["coeffs"],
                "ci": {
                    k: (list(v) if v is not None else None)
                    for k, v in body["ci"].items()
                },
            }
            for name, body in fitted.items()
        },
    )


def _empty_energy_basis() -> dict[str, Any]:
    return {
        "marginal": {"wh_per_1k_output": None, "idle_subtracted": True},
        "attributed": {
            "attribution_rule": "idle_amortised_over_measured_tokens",
            "levels": [
                {"assumed_utilisation": u, "wh_per_1k_output": None}
                for u in ATTRIBUTED_UTILISATION_LEVELS
            ],
        },
    }


def fit_record(record: Mapping[str, Any]) -> NestedFitResult:
    """Re-fit a stored record's ``raw_run_table`` under protocol v2."""
    prov = record.get("provenance") or {}
    table = prov.get("raw_run_table") or []
    if not isinstance(table, list):
        table = []
    idle = _idle_watts(prov) if isinstance(prov, Mapping) else 0.0
    energy_source = prov.get("energy_source") if isinstance(prov, Mapping) else None
    vtok = 0
    coef = record.get("coefficients") or {}
    if isinstance(coef, Mapping) and coef.get("visual_tokens_per_image"):
        try:
            vtok = int(coef["visual_tokens_per_image"])
        except (TypeError, ValueError):
            vtok = 0
    return fit_nested_models(
        table,
        idle_watts=idle,
        idle_relative=idle_relative_uncertainty(prov if isinstance(prov, Mapping) else None),
        energy_source=str(energy_source) if energy_source else None,
        visual_tokens_per_image=vtok,
    )


def predict_design_leverage(
    *,
    idle_watts: float,
    planned_rows: Sequence[Mapping[str, Any]],
    expected_duration_ms: Sequence[float],
    prior_wh_per_1k_output: float,
    prior_wh_per_1k_input: float | None = None,
) -> dict[str, Any]:
    """Pre-run leverage prediction. Refuse an axis whose planned span is too small.

    ``prior_wh_per_1k_input`` defaults to 0.1 × the decode prior when unset.
    You do not have residuals before you spend GPU time.
    """
    if prior_wh_per_1k_input is None:
        prior_wh_per_1k_input = 0.1 * prior_wh_per_1k_output
    n = min(len(planned_rows), len(expected_duration_ms))
    if n == 0 or idle_watts <= 0:
        return {"ok": False, "reason": "no planned rows or idle watts"}
    idle_energies = [
        idle_watts * (float(expected_duration_ms[i]) / 3_600_000.0)
        for i in range(n)
    ]
    sigma = IDLE_INTEGRATION_UNCERTAINTY * (sum(idle_energies) / n)
    ins = [float(planned_rows[i].get("text_tokens") or 0) for i in range(n)]
    outs = [float(planned_rows[i].get("output_tokens") or 0) for i in range(n)]
    in_span_k = (max(ins) - min(ins)) / 1000.0
    out_span_k = (max(outs) - min(outs)) / 1000.0
    planned = [
        {"text_tokens": ins[i], "output_tokens": outs[i], "n_images": 0,
         "duration_ms": float(expected_duration_ms[i])}
        for i in range(n)
    ]
    in_ok, in_reason = input_design_leverage(planned)
    # The prefill axis is measured on the short arm rows, so its noise scale is
    # theirs. Using the whole design's mean would import the long decode rows'
    # idle energy into an axis that never ran that long.
    input_rows = _rows_informing_input(planned)
    sigma_input = IDLE_INTEGRATION_UNCERTAINTY * (
        sum(idle_watts * (float(r["duration_ms"]) / 3_600_000.0) for r in input_rows)
        / len(input_rows)
    )
    out_lev = _energy_leverage(out_span_k, prior_wh_per_1k_output, sigma) or 0.0
    in_lev = _energy_leverage(in_span_k, prior_wh_per_1k_input, sigma_input) or 0.0
    return {
        "ok": True,
        "sigma_idle_wh": sigma,
        "sigma_idle_input_wh": sigma_input,
        "output": {
            "leverage": out_lev,
            "adequate": out_lev >= LEVERAGE_MULTIPLE,
        },
        "input": {
            "design_leverage": in_ok,
            "reason": in_reason,
            "leverage": in_lev,
            "adequate": in_ok and in_lev >= LEVERAGE_MULTIPLE,
        },
        "leverage_multiple": LEVERAGE_MULTIPLE,
    }


# A prefill point must occupy a window long enough to measure. Prefill time is
# roughly linear in input length, so short prompts need more repeats than long
# ones to reach the same window. Sized against the observed H100 rate of about
# 30 input tokens per millisecond, then doubled for headroom.
PREFILL_WINDOW_TARGET_MS = 4000.0
PREFILL_TOKENS_PER_MS = 30.0
PREFILL_MAX_REPEATS = 64
# Requested prompt lengths are approximate; the realised token count varies with
# the tokenizer. Size design points below the server's limit so the longest
# prefill point is not rejected outright.
CONTEXT_SAFETY_FRACTION = 0.9


def prefill_repeats_for(
    approx_input_tokens: int, tokens_per_ms: float | None = None
) -> int:
    """Requests to aggregate into one timing window for a prefill point.

    ``tokens_per_ms`` is the accelerator's prefill throughput. The default is
    the observed H100 rate and is wrong by roughly two orders of magnitude for
    anything slower: applying it to a machine that prefills at 0.5 tok/ms asks
    for 64 repeats of a point that already fills the window on its own, turning
    one design point into minutes of runtime. Callers on other silicon should
    measure their rate and pass it rather than inherit this constant.
    """
    rate = tokens_per_ms if tokens_per_ms and tokens_per_ms > 0 else PREFILL_TOKENS_PER_MS
    per_request_ms = max(1.0, approx_input_tokens / rate)
    needed = math.ceil(PREFILL_WINDOW_TARGET_MS / per_request_ms)
    return max(1, min(PREFILL_MAX_REPEATS, needed))


def fits_context(
    approx_input_tokens: int, max_tokens: int, context_limit: int | None
) -> bool:
    """Whether a design point fits, with margin for tokenizer variation.

    Prompt length is requested in approximate tokens and the real count varies
    with the tokenizer, so a point sized exactly to the limit may or may not be
    accepted depending on the model. Leave room rather than lose the longest
    prefill point, which is the one carrying the arm's span.
    """
    if not context_limit or context_limit <= 0:
        return True
    return (approx_input_tokens + max_tokens) <= context_limit * CONTEXT_SAFETY_FRACTION


def v2_grid_design(
    *,
    vision: bool = False,
    quick: bool = False,
    context_limit: int | None = None,
    prefill_tokens_per_ms: float | None = None,
) -> list[dict[str, Any]]:
    """Unbalanced v2 design: prefill isolation, decode isolation, verification.

    Phase separation (TTFT-carved prefill vs decode integration) applies only
    at concurrency 1. This grid is the concurrency-1 arm. ``--batched`` is the
    shipped concurrency ladder and is not rebuilt here.
    """
    grid: list[dict[str, Any]] = []
    prefill_inputs: tuple[int, ...]
    decode_outputs: tuple[int, ...]
    verify: tuple[tuple[int, int], ...]
    if quick:
        prefill_inputs = (1024, 4096, 8192)
        decode_outputs = (512, 2048, 4096)
        verify = ((2048, 256), (512, 1024))
    else:
        prefill_inputs = (1024, 4096, 8192, 16384, 32768)
        decode_outputs = (512, 1024, 2048, 4096)
        verify = ((2048, 256), (1024, 1024), (4096, 128), (512, 2048))

    # Prefill isolation: long inputs, minimal outputs, replicated at corners.
    # These rows are short by construction -- pinning output at 8 tokens is what
    # isolates prefill -- and a single one can finish in under 100 ms. On the
    # power-sampling path that is below the duration floor, so the arm the whole
    # protocol is built around gets dropped. Repeating the same point inside one
    # timing window lengthens the measurement without moving the design point.
    prefill_inputs = tuple(
        n for n in prefill_inputs if fits_context(n, 8, context_limit)
    )
    for i, approx_in in enumerate(prefill_inputs):
        repeats = prefill_repeats_for(approx_in, prefill_tokens_per_ms)
        grid.append({
            "n_images": 0, "approx_text_tokens": approx_in, "max_tokens": 8,
            "replicate": False, "arm": "prefill", "requests": repeats,
        })
        if i in (0, len(prefill_inputs) - 1):
            grid.append({
                "n_images": 0, "approx_text_tokens": approx_in, "max_tokens": 8,
                "replicate": True, "arm": "prefill", "requests": repeats,
            })

    # Decode and verify points consume context too: a 4096-token generation
    # needs room for its own output. Filtering only the prefill arm left points
    # the server could not honour -- vLLM rejects them loudly, but Ollama
    # truncates and answers normally, so on the sampling path they became
    # silently wrong design points rather than dropped ones.
    decode_outputs = tuple(
        n for n in decode_outputs if fits_context(32, n, context_limit)
    )
    verify = tuple(
        (a, b) for a, b in verify if fits_context(a, b, context_limit)
    )

    # Decode isolation: minimal inputs, long outputs, replicated at corners.
    for i, max_out in enumerate(decode_outputs):
        grid.append({
            "n_images": 0, "approx_text_tokens": 32, "max_tokens": max_out,
            "replicate": False, "arm": "decode",
        })
        if i in (0, len(decode_outputs) - 1):
            grid.append({
                "n_images": 0, "approx_text_tokens": 32, "max_tokens": max_out,
                "replicate": True, "arm": "decode",
            })

    for approx_in, max_out in verify:
        grid.append({
            "n_images": 0, "approx_text_tokens": approx_in, "max_tokens": max_out,
            "replicate": False, "arm": "verify",
        })

    if vision:
        # Image arm on a subset, not the full cartesian product.
        for spec in list(grid):
            if spec["arm"] in ("decode", "verify") and not spec["replicate"]:
                img = dict(spec)
                img["n_images"] = 1
                grid.append(img)
    return grid


def nested_fit_to_dict(fit: NestedFitResult) -> dict[str, Any]:
    return {
        "published_rung": fit.published_rung,
        "status": dict(fit.status),
        "null_reasons": dict(fit.null_reasons),
        "residuals_structured": fit.residuals_structured,
        "design_leverage": fit.design_leverage,
        "admitted_rows": fit.admitted_rows,
        "dropped_zero_raw": fit.dropped_zero_raw,
        "dropped_malformed": fit.dropped_malformed,
        "dropped_under_floor": fit.dropped_under_floor,
    }
