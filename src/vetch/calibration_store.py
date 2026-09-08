"""Versioned, data-rich storage for hardware energy calibrations.

A calibration's coefficients are a function of far more than ``(provider, model)``
— they depend on the GPU, serving stack, precision, and model regime. Storing by
``(provider, model)`` alone (the legacy scheme in :mod:`vetch.calibrate`) makes two
calibrations of the same model on different GPUs collide. This module keys a
calibration by a full **identity** and records enough **provenance** to reproduce
it, share it, and reconcile it against a cloud carbon export.

Design:
- One JSON file per identity, keyed by a readable slug + a hash of the normalized
  identity (no collisions; human-greppable; trivially shareable).
- Overwriting the same identity archives the previous file under
  ``~/.vetch/calibrations/archive/`` (local history; not a multi-tenant registry).
- A single scoring resolver, backward-compatible with legacy flat files: at
  inference the caller typically has ``(provider, model)``, so an unambiguous match
  keeps its measured tier, while an ambiguous or cross-provider match is
  tier-capped (fail-loud) rather than silently trusted. Optional
  :class:`ResolveHints` (gpu / serving_engine / precision / power_cap_w) let an
  instrumented host claim ``exact`` when the serving stack and cap are known.
- Stdlib only at read time (json/pathlib/hashlib). Resolution validates every
  schema-2 record before exposing its coefficients -- schema, self-consistency
  and attestation hashes, all stdlib and microseconds. The NumPy refit that
  additionally catches a forgery whose hashes were recomputed is opt-in via
  ``VETCH_CALIB_REFIT_ON_RESOLVE``; see :func:`refit_on_resolve`.

The record schema is versioned by ``CALIB_RECORD_SCHEMA`` (integer 2). Schema 1
files remain resolvable at inference with their original clamp-to-zero
semantics, behind a warning. Newly written records are schema 2; ``active: true``
on a newly written record requires schema 2.

Storage (measurement) key and resolve key are different. The resolve key is
unchanged: ``self-hosted``, model, gpu, engine, precision, concurrency. The
storage key adds ``host`` (infrastructure provider, site, tenancy, form factor,
enforced power limit) plus a hash of ``serving_config``, so two providers or two
power caps never overwrite each other. Multiple measurement files under one
resolve key already tier-cap unless a hint selects one.

``identity.provider`` is the event label, always ``self-hosted`` for local
inference. Infrastructure owner belongs on ``host.infrastructure_provider``.

A new *optional* field is additive within a major; a new resolve-identity
dimension must be nullable and treated as a wildcard so older records still
resolve.

Identity JSON uses ``serving_engine`` (not ``backend``). Records written with the
older ``backend`` key are still readable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import stat
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vetch.calibrate import CalibrationResult

logger = logging.getLogger(__name__)

CALIB_RECORD_SCHEMA = 2
CALIB_RECORD_SCHEMA_V1 = 1
_V1_CLAMP_WARNING = (
    "Loaded a schema_version=1 calibration with clamp-to-zero coefficient "
    "semantics. Migrate to v2; a null coefficient means unidentified, never zero."
)

# Provenance keys that change across otherwise-identical measurement *profiles*
# (thermal idle drift, bootstrap noise, clock probes). Excluded from content_hash
# so a re-run under the same conditions can share an attestation hash; wall-clock
# identity is still in record["timestamp"] / measured_at.
_CONTENT_HASH_EXCLUDE = frozenset({
    "timestamp",
    "measured_at",
    "measured_by",
    "idle_watts_before",
    "idle_watts_after",
    "idle_drift_pct",
    "raw_run_table",
    "fit",
    "clocks",
    "compute_process_count_at_idle",
})

# Energy domain constants: NVML total-energy is GPU-board (core + HBM).
GPU_BOARD_INCLUDES = ["gpu_core", "hbm"]
GPU_BOARD_EXCLUDES = [
    "host_cpu", "host_dram", "nvlink_switch", "nic", "storage", "psu_loss", "cooling",
]
APPLE_SOC_INCLUDES = ["cpu", "gpu", "ane"]
APPLE_SOC_EXCLUDES = ["display", "psu_loss", "cooling"]

# --- Data-driven tables (bundled JSON; env can extend provider class) --------

_gpu_aliases_cache: dict[str, str] | None = None
_self_hosted_cache: frozenset[str] | None = None


def _load_json_resource(name: str) -> dict[str, Any]:
    try:
        from importlib import resources

        data_path = resources.files("vetch.data").joinpath(name)
        return json.loads(data_path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    except Exception as e:  # noqa: BLE001 — fail-open to empty / defaults
        logger.debug("Could not load vetch.data/%s: %s", name, e)
        return {}


def gpu_aliases() -> dict[str, str]:
    """Raw-name (casefolded) → canonical GPU key table."""
    global _gpu_aliases_cache
    if _gpu_aliases_cache is None:
        raw = _load_json_resource("gpu_aliases.json")
        _gpu_aliases_cache = {
            str(k).casefold(): str(v) for k, v in raw.items() if isinstance(v, str)
        }
    return _gpu_aliases_cache


# Cloud / vendor labels that must never enter the self-hosted equivalence class
# via env extension (wrong energy on real API traffic).
_CLOUD_PROVIDER_BLOCKLIST = frozenset({
    "openai", "anthropic", "bedrock", "azure", "azure-openai", "vertexai",
    "google", "gemini", "cohere", "mistral", "groq", "together", "fireworks",
    "aws", "gcp", "amazon", "microsoft",
})


def is_cloud_provider(provider: str | None) -> bool:
    """True if ``provider`` is a hosted cloud/API vendor (case-insensitive).

    Calibrating against a cloud provider, or resolving a local calibration as an
    exact measurement for cloud traffic, would attach one deployment's hardware
    numbers to a completely different (metered, undisclosed) stack. Both are
    refused/capped via this check.
    """
    if not provider:
        return False
    return provider.casefold() in _CLOUD_PROVIDER_BLOCKLIST


def self_hosted_providers() -> frozenset[str]:
    """Provider labels that may cross-reuse calibrations (always tier-capped).

    Bundled class ``self_hosted`` in ``provider_equivalence.json``, extended by
    ``VETCH_SELF_HOSTED_PROVIDERS`` (comma-separated). Cloud vendors (including
    ``openai``) are never admitted, even if listed in env.
    """
    global _self_hosted_cache
    if _self_hosted_cache is not None:
        return _self_hosted_cache

    data = _load_json_resource("provider_equivalence.json")
    base = data.get("self_hosted") or []
    names = {str(x).casefold() for x in base if x}
    extra = os.environ.get("VETCH_SELF_HOSTED_PROVIDERS", "")
    if extra.strip():
        for p in extra.split(","):
            label = p.strip().casefold()
            if not label:
                continue
            if label in _CLOUD_PROVIDER_BLOCKLIST:
                logger.warning(
                    "Ignoring cloud provider %r in VETCH_SELF_HOSTED_PROVIDERS "
                    "(never admitted to cross-reuse class).",
                    label,
                )
                continue
            names.add(label)
    names -= _CLOUD_PROVIDER_BLOCKLIST
    _self_hosted_cache = frozenset(names)
    return _self_hosted_cache


def hints_trusted() -> bool:
    """True when env hints may restore ``exact`` (operator opt-in).

    Without ``VETCH_CALIB_HINTS_TRUSTED``, hints may disambiguate but never claim
    measured Tier-0 ``exact`` — env is an untrusted attestation channel.
    """
    v = os.environ.get("VETCH_CALIB_HINTS_TRUSTED", "").strip().casefold()
    return v in ("1", "true", "yes", "on")


def _as_bool(value: Any, *, default: bool) -> bool:
    """Parse JSON-ish bools; unparseable values fail closed (False)."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip().casefold()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off", ""):
            return False
        return False
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return False
    return False


def _clear_policy_caches() -> None:
    """Test helper: drop cached GPU / provider tables."""
    global _gpu_aliases_cache, _self_hosted_cache
    _gpu_aliases_cache = None
    _self_hosted_cache = None


def canonical_gpu(raw: str | None) -> tuple[str | None, bool]:
    """Map a raw NVML / system_profiler GPU name to a canonical key.

    Returns ``(canonical_key, is_known)``. ``is_known`` is False when the name
    wasn't in the curated table and had to be normalized heuristically — callers
    should treat that as lower confidence.
    """
    if not raw:
        return None, False
    norm = re.sub(r"\s+", " ", raw.strip().casefold())
    aliases = gpu_aliases()
    if norm in aliases:
        return aliases[norm], True
    # Apple Silicon chips from system_profiler (e.g. "Apple M3 Max").
    m = re.match(r"apple (m\d+)(?:\s+(\w+))?$", norm)
    if m:
        family, tier = m.group(1), (m.group(2) or "base")
        return f"apple-{family}-{tier}", True
    # Heuristic fallback: strip vendor, collapse separators. Not authoritative.
    guess = re.sub(r"[^a-z0-9]+", "-", norm.replace("nvidia", "").replace("tesla", ""))
    return guess.strip("-") or None, False


# --- Identity ---------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationIdentity:
    """The tuple that determines an energy coefficient.

    ``serving_engine`` is the stack that served the model (vllm / ollama / …),
    not the HTTP API shape. ``visual_token_budget`` and ``concurrency`` are
    nullable forward-compat slots: null means "unknown / wildcard" so older
    records still resolve.

    ``instance_type`` is deprecated on identity (schema 2 moves it to
    ``host.instance_type``). It is still parsed from v1 files so those records
    resolve. Do not write both.
    """

    provider: str
    model: str
    gpu: str | None = None
    serving_engine: str | None = None
    precision: str | None = None
    instance_type: str | None = None  # deprecated; host.instance_type in schema 2
    visual_token_budget: int | None = None
    concurrency: int | None = None  # serving concurrency (batch=1 legacy = None)


@dataclass(frozen=True)
class ResolveHints:
    """Optional stack dims known at inference time.

    When provided, resolve prefers candidates matching all non-None fields and
    may keep ``exact`` even if other identities exist for the same model. A
    power-capped candidate is never exact without a matching ``power_cap_w``.
    """

    gpu: str | None = None
    serving_engine: str | None = None
    precision: str | None = None
    concurrency: int | None = None
    power_cap_w: float | None = None

    def as_cache_key(self) -> tuple[Any, ...]:
        return (
            self.gpu, self.serving_engine, self.precision, self.concurrency,
            self.power_cap_w,
        )

    def any_set(self) -> bool:
        return any(
            (v is not None and str(v).strip())
            for v in (self.gpu, self.serving_engine, self.precision)
        ) or self.concurrency is not None or self.power_cap_w is not None


def hints_from_env() -> ResolveHints | None:
    """Build hints from ``VETCH_CALIB_*`` environment variables."""
    gpu = os.environ.get("VETCH_CALIB_GPU") or None
    engine = os.environ.get("VETCH_CALIB_SERVING_ENGINE") or None
    precision = os.environ.get("VETCH_CALIB_PRECISION") or None
    conc_raw = os.environ.get("VETCH_CALIB_CONCURRENCY")
    concurrency: int | None = None
    if conc_raw is not None and str(conc_raw).strip():
        try:
            concurrency = int(str(conc_raw).strip())
        except ValueError:
            logger.warning(
                "Ignoring invalid VETCH_CALIB_CONCURRENCY=%r (expected int).", conc_raw,
            )
    cap_raw = os.environ.get("VETCH_CALIB_POWER_CAP_W")
    power_cap_w: float | None = None
    if cap_raw is not None and str(cap_raw).strip():
        try:
            power_cap_w = float(str(cap_raw).strip())
            if not math.isfinite(power_cap_w) or power_cap_w <= 0:
                raise ValueError
        except ValueError:
            logger.warning(
                "Ignoring invalid VETCH_CALIB_POWER_CAP_W=%r (expected positive number).",
                cap_raw,
            )
            power_cap_w = None
    hints = ResolveHints(
        gpu=gpu, serving_engine=engine, precision=precision,
        concurrency=concurrency, power_cap_w=power_cap_w,
    )
    return hints if hints.any_set() else None


def _slug_part(s: str | int | None) -> str:
    if s is None:
        return "na"
    s = str(s).strip().casefold()
    s = re.sub(r"[^a-z0-9.]+", "-", s)
    return s.strip("-.") or "na"


def _identity_canonical_json(identity: CalibrationIdentity) -> str:
    """Order-stable canonical JSON of the identity, for hashing and equality."""
    d = {
        k: (v.casefold() if isinstance(v, str) else v)
        for k, v in asdict(identity).items()
    }
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def identity_slug(identity: CalibrationIdentity) -> str:
    """Deterministic, human-readable, collision-free filename stem."""
    parts: list[Any] = [
        identity.provider, identity.model, identity.gpu,
        identity.serving_engine, identity.precision,
    ]
    # Non-null forward-compat dims appear in the greppable stem too.
    if identity.instance_type is not None:
        parts.append(identity.instance_type)
    if identity.visual_token_budget is not None:
        parts.append(identity.visual_token_budget)
    if identity.concurrency is not None:
        parts.append(f"c{identity.concurrency}")
    stem = "__".join(_slug_part(x) for x in parts)
    h = hashlib.sha256(_identity_canonical_json(identity).encode()).hexdigest()[:8]
    stem = stem[:180].rstrip("-.")
    return f"{stem}-{h}"


def _sha256_payload(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def identity_from_dict(raw: Mapping[str, Any]) -> CalibrationIdentity:
    """Parse identity JSON, accepting legacy ``backend`` as ``serving_engine``."""
    engine = raw.get("serving_engine")
    if engine is None:
        engine = raw.get("backend")  # v1 early records
    return CalibrationIdentity(
        provider=str(raw["provider"]),
        model=str(raw["model"]),
        gpu=raw.get("gpu"),
        serving_engine=engine,
        precision=raw.get("precision"),
        instance_type=raw.get("instance_type"),
        visual_token_budget=raw.get("visual_token_budget"),
        concurrency=_coerce_optional_int(raw.get("concurrency")),
    )


def _coerce_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    if n != n:  # NaN
        return None
    return n


def infer_gpu_form_factor(gpu: str | None) -> str | None:
    """sxm / pcie / soc from a canonical GPU key. None if it cannot be read off the name."""
    if not gpu:
        return None
    g = gpu.casefold()
    if "sxm" in g:
        return "sxm"
    if "pcie" in g or "pci-e" in g:
        return "pcie"
    if g.startswith("apple-") or "apple" in g:
        return "soc"
    return None


def default_facility() -> dict[str, Any]:
    return {
        "pue": None,
        "pue_source": None,
        "pue_basis": None,
        "wue_l_per_kwh": None,
        "wue_source": None,
        "water_boundary": None,
        "water_accounting_kind": None,
        "grid_intensity_gco2_per_kwh": None,
        "grid_intensity_source": None,
        "grid_accounting_convention": None,
    }


def default_host(
    *,
    infrastructure_provider: str | None = None,
    provider_kind: str | None = None,
    instance_type: str | None = None,
    site: str | None = None,
    region: str | None = None,
    grid_zone: str | None = None,
    country: str | None = None,
    host_cpu: str | None = None,
    host_cpu_cores: int | None = None,
    gpu_form_factor: str | None = None,
    gpu_count_in_host: int | None = None,
    tenancy: str | None = "unknown",
    facility: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fac = dict(facility) if facility else default_facility()
    return {
        "infrastructure_provider": infrastructure_provider,
        "provider_kind": provider_kind,
        "instance_type": instance_type,
        "site": site,
        "region": region,
        "grid_zone": grid_zone,
        "country": country,
        "host_cpu": host_cpu,
        "host_cpu_cores": host_cpu_cores,
        "gpu_form_factor": gpu_form_factor,
        "gpu_count_in_host": gpu_count_in_host,
        "tenancy": tenancy,
        "facility": fac,
    }


def default_serving_config(
    *,
    engine: str | None = None,
    engine_version: str | None = None,
    prefix_caching: bool | None = None,
    speculative_decoding: bool | None = None,
    tensor_parallel_size: int | None = None,
    flags_raw: str | None = None,
    engine_specific: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "engine": engine,
        "engine_version": engine_version,
        "prefix_caching": prefix_caching,
        "speculative_decoding": speculative_decoding,
        "tensor_parallel_size": tensor_parallel_size,
        "flags_raw": flags_raw,
        "engine_specific": dict(engine_specific) if engine_specific else {},
    }


def default_measurement_conditions(
    *,
    enforced_power_limit_w: float | None = None,
    sm_clock_locked: bool | None = None,
    sm_clock_mhz: int | None = None,
) -> dict[str, Any]:
    return {
        "enforced_power_limit_w": enforced_power_limit_w,
        "sm_clock_locked": sm_clock_locked,
        "sm_clock_mhz": sm_clock_mhz,
    }


def default_node_power() -> dict[str, Any]:
    return {
        "available": False,
        "source": None,
        "board_wh": None,
        "node_wh": None,
        "node_to_board_ratio": None,
        "sampling_hz": None,
    }


def default_energy_basis() -> dict[str, Any]:
    return {
        "marginal": {"wh_per_1k_output": None, "idle_subtracted": True},
        "attributed": {
            "attribution_rule": "idle_amortised_over_measured_tokens",
            "levels": [
                {"assumed_utilisation": 0.2, "wh_per_1k_output": None},
                {"assumed_utilisation": 0.5, "wh_per_1k_output": None},
                {"assumed_utilisation": 0.8, "wh_per_1k_output": None},
            ],
        },
    }


def _identity_for_storage(identity: CalibrationIdentity) -> CalibrationIdentity:
    """Drop deprecated identity.instance_type so v2 does not carry both."""
    if identity.instance_type is None:
        return identity
    return replace(identity, instance_type=None)


def _measurement_key_payload(
    identity: CalibrationIdentity,
    host: Mapping[str, Any] | None,
    conditions: Mapping[str, Any] | None,
    serving_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    idn = _identity_for_storage(identity)
    host = host or {}
    conditions = conditions or {}
    serving = serving_config or {}
    return {
        "identity": {
            "provider": (idn.provider or "").casefold(),
            "model": (idn.model or "").casefold(),
            "gpu": (idn.gpu.casefold() if isinstance(idn.gpu, str) else idn.gpu),
            "serving_engine": (
                idn.serving_engine.casefold()
                if isinstance(idn.serving_engine, str) else idn.serving_engine
            ),
            "precision": (
                idn.precision.casefold() if isinstance(idn.precision, str) else idn.precision
            ),
            "visual_token_budget": idn.visual_token_budget,
            "concurrency": idn.concurrency,
        },
        "host": {
            "infrastructure_provider": host.get("infrastructure_provider"),
            "site": host.get("site"),
            "tenancy": host.get("tenancy"),
            "gpu_form_factor": host.get("gpu_form_factor"),
            "instance_type": host.get("instance_type"),
        },
        "enforced_power_limit_w": conditions.get("enforced_power_limit_w"),
        "serving_config": serving,
    }


def measurement_fingerprint(
    identity: CalibrationIdentity,
    host: Mapping[str, Any] | None,
    conditions: Mapping[str, Any] | None,
    serving_config: Mapping[str, Any] | None,
) -> tuple[Any, ...]:
    """Dims that distinguish two measurements under one resolve key."""
    host = host or {}
    conditions = conditions or {}
    serving_hash = _sha256_payload(serving_config or {})
    return (
        host.get("infrastructure_provider"),
        host.get("site"),
        host.get("tenancy"),
        host.get("gpu_form_factor"),
        host.get("instance_type"),
        conditions.get("enforced_power_limit_w"),
        serving_hash,
    )


def measurement_slug(
    identity: CalibrationIdentity,
    host: Mapping[str, Any] | None = None,
    conditions: Mapping[str, Any] | None = None,
    serving_config: Mapping[str, Any] | None = None,
) -> str:
    """Filename stem for a schema-2 record (measurement key, not resolve key)."""
    idn = _identity_for_storage(identity)
    host = host or {}
    conditions = conditions or {}
    parts: list[Any] = [
        idn.provider, idn.model, idn.gpu, idn.serving_engine, idn.precision,
    ]
    if idn.visual_token_budget is not None:
        parts.append(idn.visual_token_budget)
    if idn.concurrency is not None:
        parts.append(f"c{idn.concurrency}")
    infra = host.get("infrastructure_provider")
    if infra:
        parts.append(infra)
    site = host.get("site")
    if site:
        parts.append(site)
    limit = conditions.get("enforced_power_limit_w")
    if limit is not None:
        parts.append(f"pl{limit}")
    stem = "__".join(_slug_part(x) for x in parts)
    h = hashlib.sha256(
        json.dumps(
            _measurement_key_payload(idn, host, conditions, serving_config),
            sort_keys=True, separators=(",", ":"),
        ).encode()
    ).hexdigest()[:8]
    stem = stem[:180].rstrip("-.")
    return f"{stem}-{h}"


def record_stem(record: Mapping[str, Any]) -> str | None:
    """Expected filename stem for a stored record, or None if identity is missing."""
    try:
        idn = identity_from_dict(record["identity"])
    except (KeyError, TypeError, ValueError):
        return None
    if record.get("schema_version") == CALIB_RECORD_SCHEMA_V1:
        return identity_slug(idn)
    return measurement_slug(
        idn,
        record.get("host"),
        record.get("measurement_conditions"),
        record.get("serving_config"),
    )


# --- Record build / parse ---------------------------------------------------


_HASHED_COEFFICIENT_KEYS = (
    "wh_per_1k_input", "wh_per_1k_output", "wh_per_image",
    "visual_tokens_per_image", "intercept_wh",
)


def _requests_in_table(table: Sequence[Any]) -> int:
    """Requests issued across a run table, counting each row's aggregation."""
    total = 0
    for row in table:
        if not isinstance(row, Mapping):
            total += 1
            continue
        try:
            count = int(float(row.get("requests") or 1))
        except (TypeError, ValueError):
            count = 1
        total += max(1, count)
    return total


def apply_attestation_hashes(
    record: dict[str, Any], *, host_key: Mapping[str, Any]
) -> dict[str, Any]:
    """Set ``profile_hash`` and ``content_hash`` in place, and return the record.

    The only place these are computed. Duplicating the payload construction is
    how a writer and a migration drift apart and silently stop agreeing about
    what two records having "the same hash" means.
    """
    coefficients = record.get("coefficients") or {}
    provenance = record.get("provenance") or {}
    profile_body = {
        "identity": record.get("identity"),
        "host_key": dict(host_key),
        "coefficients": {k: coefficients.get(k) for k in _HASHED_COEFFICIENT_KEYS},
    }
    record["profile_hash"] = _sha256_payload(profile_body)
    stable_prov = {
        k: v for k, v in provenance.items() if k not in _CONTENT_HASH_EXCLUDE
    }
    record["content_hash"] = _sha256_payload(
        {**profile_body, "provenance": stable_prov}
    )
    return record


def build_record(
    result: CalibrationResult,
    identity: CalibrationIdentity,
    provenance: dict[str, Any],
    timestamp: float,
    *,
    host: Mapping[str, Any] | None = None,
    serving_config: Mapping[str, Any] | None = None,
    measurement_conditions: Mapping[str, Any] | None = None,
    energy_basis: Mapping[str, Any] | None = None,
    node_power: Mapping[str, Any] | None = None,
    coefficient_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a schema-2 calibration record (dict ready for JSON).

    ``identity.instance_type`` is moved onto ``host`` when present so the two
    fields are never both populated. Newly written records are schema 2;
    ``active: true`` requires schema 2.
    """
    host_block = default_host()
    if host:
        host_block.update(dict(host))
        if "facility" in host and host["facility"]:
            fac = default_facility()
            fac.update(dict(host["facility"]))
            host_block["facility"] = fac
    if identity.instance_type and not host_block.get("instance_type"):
        host_block["instance_type"] = identity.instance_type
    stored_identity = _identity_for_storage(identity)
    conditions = default_measurement_conditions()
    if measurement_conditions:
        conditions.update(dict(measurement_conditions))
    serving = default_serving_config()
    if serving_config:
        serving.update(dict(serving_config))
        if serving_config.get("engine_specific"):
            serving["engine_specific"] = dict(serving_config["engine_specific"])
    node = default_node_power()
    if node_power:
        node.update(dict(node_power))
        if node.get("available") and node.get("node_to_board_ratio") is not None:
            if node.get("board_wh") is None or node.get("node_wh") is None:
                # Never fill a ratio from the literature when no meter was present.
                node["node_to_board_ratio"] = None
    basis = dict(energy_basis) if energy_basis else default_energy_basis()

    coefficients: dict[str, Any] = {
        "wh_per_1k_input": result.wh_per_1k_input,
        "wh_per_1k_output": result.wh_per_1k_output,
        "wh_per_image": result.wh_per_image,
        "visual_tokens_per_image": result.visual_tokens_per_image,
        "intercept_wh": result.intercept_wh,
    }
    if coefficient_meta:
        for key in ("status", "null_reasons", "published_rung"):
            if key in coefficient_meta:
                coefficients[key] = coefficient_meta[key]
    if result.active and (
        result.wh_per_1k_input is None
        and result.wh_per_1k_output is None
        and result.wh_per_image is None
        and result.intercept_wh is None
    ):
        result = replace(result, active=False)

    prov = dict(provenance)
    prov.setdefault("tier", result.tier)
    prov.setdefault("samples", result.samples)
    # Row aggregation made `samples` ambiguous a third time. A nested writer
    # sets it to the row count, a batched writer to the request count, and once
    # a row can hold several identical requests the two stop being the same
    # number: batch 2 is 32 rows over 398 requests. Record the request count
    # explicitly rather than leave a reader to guess which one `samples` meant.
    table = prov.get("raw_run_table")
    if isinstance(table, list) and table:
        prov.setdefault("requests_measured", _requests_in_table(table))
    prov.setdefault("timestamp", timestamp)
    prov.setdefault("pue_applied", False)
    prov.setdefault("grid_applied", False)
    identity_dict = asdict(stored_identity)
    record: dict[str, Any] = {
        "schema_version": CALIB_RECORD_SCHEMA,
        "identity": identity_dict,
        "host": host_block,
        "measurement_conditions": conditions,
        "serving_config": serving,
        "energy_basis": basis,
        "node_power": node,
        "gpu_raw": result.gpu_name,
        "coefficients": coefficients,
        "provenance": prov,
        "active": bool(result.active),
        "rejection_reasons": result.rejection_reasons,
        "timestamp": timestamp,
    }
    return apply_attestation_hashes(
        record,
        host_key=_measurement_key_payload(
            stored_identity, host_block, conditions, serving
        ),
    )


def _gate_parameters() -> dict[str, Any] | None:
    """Identifiability gate constants, or None when the extra is unavailable."""
    try:
        from vetch.calibrate_v2 import gate_parameters
    except (ImportError, RuntimeError):
        return None
    return gate_parameters()


def measurement_provenance_core(
    *,
    samples: int,
    energy_source: str,
    measurement_basis: str,
    energy_domain: str,
    energy_domain_includes: Sequence[str],
    energy_domain_excludes: Sequence[str],
    idle_watts_before: float,
    idle_watts_after: float,
    idle_drift_pct: float,
    fit: Any,
    fit_engine: str,
    ci_method: str,
    run_records: Sequence[Mapping[str, Any]],
    gpu_name: str | None,
    gpu_canonical: str | None,
    gpu_known: bool,
    serving_engine: str | None,
    server_version: str | None = None,
    image_set: str | None = None,
    image_resolution_px: int | None = None,
    model_supports_images: bool | None = None,
    visual_tokens_assumed: int | None = None,
    visual_tokens_assumed_source: str | None = None,
    vetch_version: str | None = None,
    methodology_version: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Shared provenance keys for CUDA / Apple writers (domain extras via ``extra``)."""
    core: dict[str, Any] = {
        "tier": 0,
        "samples": samples,
        "energy_source": energy_source,
        "measurement_basis": measurement_basis,
        "energy_domain": energy_domain,
        "energy_domain_includes": list(energy_domain_includes),
        "energy_domain_excludes": list(energy_domain_excludes),
        "not_wall_power": True,
        "pue_applied": False,
        "grid_applied": False,
        "idle_subtracted": True,
        "idle_subtraction_method": "pre_post_average",
        "idle_watts_before": round(idle_watts_before, 3),
        "idle_watts_after": round(idle_watts_after, 3),
        "idle_drift_pct": round(idle_drift_pct, 3),
        "concurrency": 1,
        "batch_size": 1,
        "fit": {
            "r2": round(fit.r2, 4),
            "condition_number": (
                round(fit.condition_number, 2)
                if fit.condition_number != float("inf") else None
            ),
            "input_ci95": list(fit.input_ci95) if fit.input_ci95 is not None else None,
            "output_ci95": list(fit.output_ci95) if fit.output_ci95 is not None else None,
            "image_ci95": list(fit.image_ci95) if fit.image_ci95 is not None else None,
            "intercept_ci95": (
                list(_intercept_ci)
                if (_intercept_ci := getattr(fit, "intercept_ci95", None)) is not None
                else None
            ),
            "residuals_structured": fit.residuals_structured,
            "published_rung": getattr(fit, "published_rung", None),
            # The thresholds this verdict was reached under, so "identified"
            # still means the same thing when the record is read back after the
            # gate has moved.
            "gate": _gate_parameters(),
        },
        "fit_engine": fit_engine,
        "ci_method": ci_method,
        "raw_run_table": [
            {
                "n_images": r["n_images"], "text_tokens": r["text_tokens"],
                "output_tokens": r["output_tokens"],
                # Requests aggregated into this row's timing window. Without it
                # a consumer cannot recover the per-request design point, and
                # the fit silently treats an aggregate as a single request.
                "requests": r.get("requests", 1),
                "energy_wh": r["energy_wh"], "raw_energy_wh": r["raw_energy_wh"],
                "duration_ms": r["duration_ms"], "replicate": r["replicate"],
            }
            for r in run_records
        ],
        "gpu_name": gpu_name,
        "gpu_canonical": gpu_canonical,
        "gpu_known": gpu_known,
        "serving_engine": serving_engine,
        "server_version": server_version,
        "image_set": image_set,
        "image_resolution_px": image_resolution_px,
        "model_supports_images": model_supports_images,
        "visual_tokens_assumed": visual_tokens_assumed,
        "visual_tokens_assumed_source": visual_tokens_assumed_source,
        "vetch_version": vetch_version,
        "methodology_version": methodology_version,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "measured_by": None,
    }
    if extra:
        core.update(dict(extra))
    return core


def commit_calibration(
    result: CalibrationResult,
    identity: CalibrationIdentity,
    provenance: dict[str, Any],
    *,
    timestamp: float | None = None,
    host: Mapping[str, Any] | None = None,
    serving_config: Mapping[str, Any] | None = None,
    measurement_conditions: Mapping[str, Any] | None = None,
    energy_basis: Mapping[str, Any] | None = None,
    node_power: Mapping[str, Any] | None = None,
    coefficient_meta: Mapping[str, Any] | None = None,
) -> Path:
    """Build and atomically write a schema-2 record; return the active path."""
    ts = time.time() if timestamp is None else timestamp
    return write_record(build_record(
        result, identity, provenance, timestamp=ts,
        host=host,
        serving_config=serving_config,
        measurement_conditions=measurement_conditions,
        energy_basis=energy_basis,
        node_power=node_power,
        coefficient_meta=coefficient_meta,
    ))


# In-process index of top-level v1 records. Fingerprint includes per-file
# mtime/size so in-place rewrites invalidate the cache (dir mtime alone does not).
_index_fingerprint: tuple[Any, ...] | None = None
_index_records: list[tuple[Path, dict[str, Any]]] = []


def _clear_store_index() -> None:
    """Drop the in-process v1 directory index (call after writes)."""
    global _index_fingerprint, _index_records
    _index_fingerprint = None
    _index_records = []


def _dir_fingerprint(calib_dir: Path) -> tuple[Any, ...]:
    entries: list[tuple[str, float, int]] = []
    for p in sorted(calib_dir.glob("*.json")):
        try:
            st = p.lstat()
            if not stat.S_ISREG(st.st_mode):
                continue  # skip symlinks / non-regular
            entries.append((p.name, st.st_mtime, st.st_size))
        except OSError:
            continue
    try:
        d_mtime = calib_dir.stat().st_mtime
    except OSError:
        d_mtime = 0.0
    return (d_mtime, tuple(entries))


def _v1_index() -> list[tuple[Path, dict[str, Any]]]:
    """Cached list of (path, record) for top-level v1 JSON files."""
    global _index_fingerprint, _index_records
    from vetch.calibrate import CALIBRATION_DIR

    if not CALIBRATION_DIR.exists():
        _clear_store_index()
        return []
    fp = _dir_fingerprint(CALIBRATION_DIR)
    if _index_fingerprint == fp:
        return _index_records

    records: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(CALIBRATION_DIR.glob("*.json")):
        try:
            st = path.lstat()
            if not stat.S_ISREG(st.st_mode):
                logger.warning("Skipping non-regular calibration path %s", path)
                continue
        except OSError:
            continue
        rec = _read_json(path)
        if not rec or "identity" not in rec:
            continue
        schema = rec.get("schema_version")
        if schema not in (CALIB_RECORD_SCHEMA, CALIB_RECORD_SCHEMA_V1):
            continue
        try:
            identity_from_dict(rec["identity"])
        except (KeyError, TypeError, ValueError):
            continue
        expected = record_stem(rec)
        if expected is None or path.stem != expected:
            logger.warning(
                "Calibration file %s stem does not match identity slug %s; skipping.",
                path.name, expected,
            )
            continue
        records.append((path, rec))
    _index_fingerprint = fp
    _index_records = records
    return records


def write_record(record: dict[str, Any]) -> Path:
    """Write a calibration record to ``~/.vetch/calibrations/<slug>.json``.

    If a file already exists for this identity, it is moved to
    ``archive/<slug>.<UTC-timestamp>.<pid>.json`` before the new record is written.
    """
    from vetch.calibrate import CALIBRATION_DIR
    from vetch.calibration_invariants import errors_only, record_invariants

    # A record that contradicts itself must not reach disk. Catching this at
    # write time is the difference between one bad run and a bad record shipped
    # in every file a campaign produces.
    incoherent = errors_only(record_invariants(record))
    if incoherent:
        raise ValueError(
            "refusing to write a self-contradictory calibration record:\n  "
            + "\n  ".join(str(v) for v in incoherent)
        )

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(CALIBRATION_DIR, 0o700)
    except OSError:
        pass

    identity = identity_from_dict(record["identity"])
    record = dict(record)
    schema = record.get("schema_version", CALIB_RECORD_SCHEMA)
    if schema == CALIB_RECORD_SCHEMA_V1:
        record["identity"] = asdict(identity)
        path = CALIBRATION_DIR / f"{identity_slug(identity)}.json"
    else:
        stored = _identity_for_storage(identity)
        record["identity"] = asdict(stored)
        stem = measurement_slug(
            stored,
            record.get("host"),
            record.get("measurement_conditions"),
            record.get("serving_config"),
        )
        path = CALIBRATION_DIR / f"{stem}.json"

    if path.exists():
        archive_dir = CALIBRATION_DIR / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(archive_dir, 0o700)
        except OSError:
            pass
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        archived = archive_dir / f"{path.stem}.{stamp}.{os.getpid()}.json"
        # Collision within the same second: add monotonic ns.
        if archived.exists():
            archived = archive_dir / (
                f"{path.stem}.{stamp}.{os.getpid()}.{time.time_ns()}.json"
            )
        try:
            os.replace(path, archived)
            # The archive is the history of a calibration campaign -- the only
            # way to compare runs after one overwrites another. Under sudo it
            # is created root-owned 0700, which locks the operator out of their
            # own previous records and makes the directory look empty rather
            # than forbidden.
            _chown_to_invoking_user(archived)
            _chown_to_invoking_user(archive_dir)
        except OSError as e:
            logger.warning("Could not archive previous calibration %s: %s", path, e)

    tmp = path.with_suffix(f".json.tmp.{os.getpid()}")
    payload = json.dumps(record, indent=2)
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(payload)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    try:
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    # 0600 plus root ownership would leave this unreadable by the user whose
    # directory it is. Do the directory too: it is created 0700 on first run,
    # and a root-owned directory locks out every future record as well.
    _chown_to_invoking_user(path)
    _chown_to_invoking_user(CALIBRATION_DIR)

    _clear_store_index()
    try:
        from vetch.calculation import _clear_calibration_cache
        _clear_calibration_cache()
    except Exception:  # noqa: BLE001 — cache invalidation is best-effort
        pass
    return path


def record_to_result(record: dict[str, Any]) -> CalibrationResult | None:
    """Rehydrate a CalibrationResult from a v1 or v2 record, or None if malformed."""
    from vetch.calibrate import CalibrationResult

    try:
        schema = record.get("schema_version")
        if schema not in (CALIB_RECORD_SCHEMA, CALIB_RECORD_SCHEMA_V1):
            return None
        idn = identity_from_dict(record["identity"])
        coef = record["coefficients"]
        prov = record.get("provenance", {})
        active = _as_bool(record.get("active"), default=True)
        if schema == CALIB_RECORD_SCHEMA_V1:
            logger.warning(_V1_CLAMP_WARNING)
            wh_in: float | None = float(coef["wh_per_1k_input"])
            wh_out: float | None = float(coef["wh_per_1k_output"])
        else:
            # v2: 0.0 is the clamp lie. Treat as unidentified (null).
            wh_in = _v2_coeff(coef.get("wh_per_1k_input"))
            wh_out = _v2_coeff(coef.get("wh_per_1k_output"))
        return CalibrationResult(
            model=idn.model,
            provider=idn.provider,
            wh_per_1k_input=wh_in,
            wh_per_1k_output=wh_out,
            tier=int(prov.get("tier", 0)),
            samples=int(prov.get("samples", 0)),
            gpu_name=record.get("gpu_raw"),
            wh_per_image=_v2_coeff(coef.get("wh_per_image")) if schema != CALIB_RECORD_SCHEMA_V1
            else coef.get("wh_per_image"),
            visual_tokens_per_image=coef.get("visual_tokens_per_image"),
            intercept_wh=_v2_coeff(coef.get("intercept_wh")) if schema != CALIB_RECORD_SCHEMA_V1
            else coef.get("intercept_wh"),
            active=active,
            rejection_reasons=record.get("rejection_reasons"),
            serving_engine=idn.serving_engine,
            backend=idn.serving_engine,  # deprecated alias
            precision=idn.precision,
        )
    except (KeyError, ValueError, TypeError):
        return None


def _v2_coeff(value: Any) -> float | None:
    """Parse a v2 coefficient. null and non-positive mean unidentified, never zero."""
    if value is None:
        return None
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    if n != n or n <= 0.0:
        return None
    return n


# --- Resolution -------------------------------------------------------------


@dataclass
class _Candidate:
    result: CalibrationResult
    provider: str
    model: str
    gpu: str | None
    serving_engine: str | None
    precision: str | None
    instance_type: str | None
    visual_token_budget: int | None
    concurrency: int | None
    tier: int
    timestamp: float
    origin: str  # "local" | "community" | "legacy"
    gpu_known: bool = True
    measurement_fingerprint: tuple[Any, ...] | None = None
    power_cap_w: float | None = None

    def identity_tuple(self) -> tuple[Any, ...]:
        """Dims that distinguish an energy coefficient (for ambiguity checks).

        Includes the stored model string, concurrency, and forward-compat identity
        slots so ``moondream`` vs ``moondream:latest`` (or C=1 vs C=32) never
        collapse into a false ``exact`` Tier 0. Schema-2 measurement fingerprint
        (host + power cap + serving_config) distinguishes two files under one
        resolve key so they tier-cap rather than silently picking one host.
        """
        return (
            self.model.casefold(),
            self.gpu,
            self.serving_engine,
            self.precision,
            self.instance_type,
            self.visual_token_budget,
            self.concurrency,
            self.measurement_fingerprint,
        )

    def matches_hints(self, hints: ResolveHints) -> bool:
        if hints.gpu is not None:
            want = hints.gpu.casefold()
            cand = (self.gpu or "").casefold()
            raw = (self.result.gpu_name or "").casefold()
            canon, _ = canonical_gpu(hints.gpu)
            if want not in (cand, raw) and (canon or "").casefold() != cand:
                return False
        if hints.serving_engine is not None:
            if (self.serving_engine or "").casefold() != hints.serving_engine.casefold():
                return False
        if hints.precision is not None:
            if (self.precision or "").casefold() != hints.precision.casefold():
                return False
        if hints.concurrency is not None:
            # A batch=1 grid record has concurrency=None but is single-stream, so a
            # concurrency=1 hint should match it (not just batched C=1 records).
            effective = self.concurrency if self.concurrency is not None else 1
            if effective != hints.concurrency:
                return False
        if hints.power_cap_w is not None:
            if self.power_cap_w is None or not math.isclose(
                self.power_cap_w, hints.power_cap_w, rel_tol=1e-6, abs_tol=0.01
            ):
                return False
        return True


def _chown_to_invoking_user(path: Path) -> None:
    """Hand a root-written file back to the user who invoked sudo.

    Apple Silicon calibration needs root for powermetrics, and records are
    written 0600 by design. Under sudo that combination produces a root-owned,
    owner-only file sitting in the invoking user's own calibration directory:
    written to exactly the right place, and unreadable by the account that will
    later try to resolve it. `_read_json` then swallows the PermissionError and
    the calibration is silently absent.

    Best effort. Failing to chown is not a reason to lose the record.
    """
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if not uid:
        return
    try:
        os.chown(path, int(uid), int(gid) if gid else -1)
    except (OSError, ValueError) as e:  # noqa: BLE001 -- never lose the record
        logger.warning(
            "Wrote %s as root but could not hand it back to uid %s (%s). "
            "Run: sudo chown %s %s",
            path, uid, e, uid, path,
        )


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        st = path.lstat()
        if not stat.S_ISREG(st.st_mode):
            return None
        return json.loads(path.read_text())  # type: ignore[no-any-return]
    except PermissionError:
        # Distinct from "absent". A record we can see but cannot read is almost
        # always a root-written file from `sudo vetch calibrate-apple-silicon`,
        # and silence here is what makes that look like the calibration simply
        # never attached.
        logger.warning(
            "Calibration record %s exists but is not readable by this user "
            "(owner mismatch). If it was written under sudo, run: "
            "sudo chown $(id -u) %s", path, path,
        )
        return None
    except (OSError, json.JSONDecodeError):
        return None


_REFIT_ON_RESOLVE_ENV = "VETCH_CALIB_REFIT_ON_RESOLVE"


def refit_on_resolve() -> bool:
    """Whether resolution re-derives coefficients from each record's run table.

    Off by default, and the default is a considered trade rather than a
    shortcut. Resolution runs on the inference path. The refit is the full
    nested fit -- eight rungs by 500 bootstrap resamples -- which measures at
    roughly 35 ms per record and lands on the first call of every process,
    scaling with the number of local calibrations. That is a cold-start cost
    on a path budgeted at 5 ms, paid on every serverless invocation.

    What stays on unconditionally is the cheap half, and it is the half that
    catches the realistic failure: schema, self-consistency and the attestation
    hashes, all microseconds. A record edited by hand, truncated, or left over
    from an older schema is rejected by those alone.

    The refit adds exactly one thing -- catching an editor who changed a
    coefficient *and* recomputed the hashes but left the run table intact. That
    is a real attack on a shared calibration directory, so the check exists and
    is one env var away; it is simply not worth a cold start to every user who
    owns the directory their own calibrations sit in. ``vetch
    validate-calibration`` always runs the full check.
    """
    value = os.environ.get(_REFIT_ON_RESOLVE_ENV, "").strip().casefold()
    return value in {"1", "true", "yes", "on"}


def _runtime_record_valid(record: Mapping[str, Any]) -> bool:
    """Fail closed on invalid schema-2 records before exposing coefficients.

    Always: schema, invariants and attestation hashes (stdlib, microseconds).
    Only under ``VETCH_CALIB_REFIT_ON_RESOLVE``: the NumPy refit. See
    :func:`refit_on_resolve` for why the expensive half is opt-in.
    """
    if record.get("schema_version") != CALIB_RECORD_SCHEMA:
        return True
    provenance = record.get("provenance")
    table = provenance.get("raw_run_table") if isinstance(provenance, Mapping) else None
    full = isinstance(table, list) and bool(table) and refit_on_resolve()
    try:
        if full:
            try:
                import numpy  # noqa: F401
            except ImportError:
                logger.warning(
                    "%s is set but NumPy is not installed; falling back to "
                    "schema, invariant and hash checks.", _REFIT_ON_RESOLVE_ENV,
                )
                full = False
        from vetch.calibrate_validate import validate_calibration_record

        report = validate_calibration_record(record, identifiability=full)
    except Exception as exc:  # noqa: BLE001 -- never make inference traceback
        logger.warning("Skipping calibration record that could not be validated: %s", exc)
        return False
    if not report.get("ok"):
        problems: list[str] = []
        for key in ("schema_errors", "invariant_errors", "restatements"):
            problems.extend(str(item) for item in (report.get(key) or []))
        logger.warning(
            "Skipping invalid calibration record for %s: %s",
            (record.get("identity") or {}).get("model", "unknown"),
            "; ".join(problems) or "validation failed",
        )
        return False
    return True


def _gather_candidates(model: str) -> list[_Candidate]:
    from vetch.calibrate import (
        CALIBRATION_DIR,
        _load_calibration_file,
        _safe_model_filename,
        calibration_model_variants,
    )

    variants = list(calibration_model_variants(model))
    variant_cf = {v.casefold() for v in variants}
    v1_cands: list[_Candidate] = []
    legacy_cands: list[_Candidate] = []
    seen_paths: set[Path] = set()

    if not CALIBRATION_DIR.exists():
        return []

    for path, rec in _v1_index():
        try:
            idn = identity_from_dict(rec["identity"])
        except (KeyError, TypeError, ValueError):
            continue
        if idn.model.casefold() not in variant_cf:
            continue
        seen_paths.add(path)
        if not _runtime_record_valid(rec):
            continue
        res = record_to_result(rec)
        if res is None or not res.active:
            continue
        prov = rec.get("provenance") or {}
        # Missing gpu_known ⇒ unknown (fail closed). Only explicit true is trusted.
        gpu_known = _as_bool(prov.get("gpu_known"), default=False)
        fp = None
        if rec.get("schema_version") == CALIB_RECORD_SCHEMA:
            fp = measurement_fingerprint(
                idn,
                rec.get("host"),
                rec.get("measurement_conditions"),
                rec.get("serving_config"),
            )
        v1_cands.append(_Candidate(
            result=res,
            provider=idn.provider,
            model=idn.model,
            gpu=idn.gpu,
            serving_engine=idn.serving_engine,
            precision=idn.precision,
            instance_type=idn.instance_type,
            visual_token_budget=idn.visual_token_budget,
            concurrency=idn.concurrency,
            tier=res.tier,
            timestamp=float(rec.get("timestamp", 0.0)),
            origin="local",
            gpu_known=gpu_known,
            measurement_fingerprint=fp,
            power_cap_w=(
                _coerce_optional_float(
                    (rec.get("measurement_conditions") or {}).get("enforced_power_limit_w")
                )
                if rec.get("schema_version") == CALIB_RECORD_SCHEMA else None
            ),
        ))

    # Legacy flats: include unless a same-provider v1 record already covers this
    # model (v1 supersedes legacy for that provider only — not globally).
    v1_providers = {c.provider.casefold() for c in v1_cands}
    for variant in variants:
        suffix = f"_{_safe_model_filename(variant)}.json"
        for path in CALIBRATION_DIR.glob(f"*{suffix}"):
            if path in seen_paths:
                continue
            try:
                st = path.lstat()
                if not stat.S_ISREG(st.st_mode):
                    continue
            except OSError:
                continue
            legacy_rec = _read_json(path)
            if not legacy_rec or legacy_rec.get("schema_version"):
                continue
            other_provider = path.name[: -len(suffix)]
            if not other_provider:
                continue
            if other_provider.casefold() in v1_providers:
                continue  # same-provider v1 wins; keep other providers' legacy
            res = _load_calibration_file(other_provider, variant)
            if res is None or not res.active:
                continue
            gpu_key, gpu_known = canonical_gpu(res.gpu_name)
            legacy_cands.append(_Candidate(
                result=res,
                provider=other_provider,
                model=variant,
                gpu=gpu_key,
                serving_engine=None,
                precision=None,
                instance_type=None,
                visual_token_budget=None,
                concurrency=None,
                tier=res.tier,
                timestamp=float(legacy_rec.get("timestamp", 0.0)),
                origin="legacy",
                gpu_known=gpu_known if res.gpu_name else False,
            ))

    return v1_cands + legacy_cands


def _apply_score(
    res: CalibrationResult,
    cand: _Candidate,
    confidence: str,
    *,
    tier_floor: int = 0,
) -> CalibrationResult:
    """Return a new result with confidence/tier applied (never mutate the input)."""
    tier = max(res.tier, tier_floor)
    conf = confidence
    # Ambiguous provider label: never claim measured exact for cloud-tagged events
    # or cloud-keyed calibrations (real metered API vs an OpenAI-compatible local
    # server sharing the label). Applies to every hosted vendor, not just openai.
    if is_cloud_provider(cand.provider) and conf == "exact":
        conf = "curated"
        tier = max(tier, 1)
    if not cand.gpu_known:
        if conf == "exact":
            conf = "proxy"
        tier = max(tier, 1)
        logger.debug(
            "Calibration GPU %r was heuristically canonicalized; capped to tier %d (%s).",
            cand.gpu, tier, conf,
        )
    return replace(res, energy_confidence=conf, tier=tier)


def _pick_scored(
    pool: list[_Candidate],
    hints: ResolveHints | None,
    *,
    same_provider: bool,
    event_provider: str,
) -> CalibrationResult | None:
    """Score a same-provider or cross-provider candidate pool."""
    if not pool:
        return None

    working = pool
    hints_disambiguated = False
    if hints is not None and hints.any_set():
        matched = [c for c in pool if c.matches_hints(hints)]
        if not matched:
            # Fail-loud: operator asserted stack dims that match nothing.
            logger.warning(
                "Calibration hints %s matched no candidate for provider=%s; "
                "refusing to attach a calibration.",
                hints, event_provider,
            )
            return None
        if len({c.identity_tuple() for c in matched}) < len({c.identity_tuple() for c in pool}):
            hints_disambiguated = True
        working = matched

    working.sort(key=lambda c: (c.tier, -c.timestamp))
    best = working[0]
    distinct = {c.identity_tuple() for c in working}
    # A power-capped measurement is not an exact match unless the runtime
    # supplied the same cap. The cap is part of the measurement key, but the
    # event usually cannot know it; silently calling the sole capped record
    # exact would attach a potentially different operating point.
    cap_unmatched = best.power_cap_w is not None and not (
        hints is not None and hints.power_cap_w is not None
        and math.isclose(best.power_cap_w, hints.power_cap_w, rel_tol=1e-6, abs_tol=0.01)
    )

    if same_provider:
        if len(distinct) <= 1:
            # Env hints that narrowed a multi-identity pool are untrusted unless
            # VETCH_CALIB_HINTS_TRUSTED is set — otherwise curated, not exact.
            if hints_disambiguated and not hints_trusted():
                return _apply_score(best.result, best, "curated", tier_floor=1)
            # A cloud event provider never stays exact (even if the calibration is
            # self-hosted but keyed under a cloud label like openai/anthropic).
            if is_cloud_provider(event_provider):
                return _apply_score(best.result, best, "curated", tier_floor=1)
            if cap_unmatched:
                return _apply_score(best.result, best, "curated", tier_floor=1)
            return _apply_score(best.result, best, "exact", tier_floor=0)
        logger.debug(
            "Calibration ambiguous across identities %s; capped.",
            sorted(map(str, distinct)),
        )
        return _apply_score(best.result, best, "proxy", tier_floor=1)

    conf = "curated" if len(distinct) <= 1 else "proxy"
    return _apply_score(best.result, best, conf, tier_floor=1)


def resolve(
    provider: str,
    model: str,
    hints: ResolveHints | None = None,
) -> CalibrationResult | None:
    """Best local calibration for (provider, model), or None.

    At inference the caller often has no hardware/stack hints, so honesty is
    enforced by ambiguity over the full identity (model, gpu, serving_engine,
    precision, and non-null forward-compat dims):

    - a single same-provider identity keeps its measured tier -> ``exact``;
    - multiple same-provider identities are ambiguous -> ``proxy``;
    - hints that uniquely select among many restore ``exact`` only when
      ``VETCH_CALIB_HINTS_TRUSTED`` is set; otherwise ``curated``;
    - hints that match nothing -> no calibration (fail-loud);
    - cross-provider reuse only among :func:`self_hosted_providers` and always
      tier-capped;
    - ``openai`` never stays ``exact`` Tier 0;
    - a heuristically canonicalized (unknown) GPU never stays ``exact`` Tier 0.

    Returns a **new** :class:`~vetch.calibrate.CalibrationResult` with
    ``energy_confidence`` set — never mutates a cached object.
    """
    cands = _gather_candidates(model)
    if not cands:
        return None

    provider_cf = provider.casefold()
    same = [c for c in cands if c.provider.casefold() == provider_cf]

    if same:
        return _pick_scored(same, hints, same_provider=True, event_provider=provider)

    local = self_hosted_providers()
    if provider_cf not in local:
        return None
    cross = [c for c in cands if c.provider.casefold() in local]
    if not cross:
        return None
    res = _pick_scored(cross, hints, same_provider=False, event_provider=provider)
    if res is not None:
        logger.debug(
            "Calibration for %s/%s reused from local-equivalent provider (%s, tier %d).",
            provider, model, res.energy_confidence, res.tier,
        )
    return res
