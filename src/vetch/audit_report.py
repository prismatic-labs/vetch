"""Deterministic inference waste audit reports.

This module turns locally stored Vetch metadata into a consulting-ready
audit report. It is deliberately rules-based: no prompts, completions, or
LLM calls are required to generate the report.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from vetch.advisory import Advisory, generate_advisories, get_advisory_spec
from vetch.config import get_advisory_threshold
from vetch.stats import SessionStats
from vetch.storage import UsageSummary, query_daily_usage, query_events, query_usage

DEFAULT_TAG_KEYS = ("feature", "customer", "workflow", "team", "service", "env")
PREMIUM_WORKFLOW_TAG_KEYS = ("workflow", "feature", "service", "route", "operation")
MIN_FINDING_SCOPE_EVENTS = 2
LOW_VOLUME_EVENT_WARNING_THRESHOLD = 20
MIN_TAGGED_FRACTION = 0.5
MIN_COSTED_FRACTION = 0.8
MONTHLY_PROJECTION_DAYS = 30
MAX_FINDINGS = 25
MAX_BREAKDOWN_ROWS = 30
MAX_RENDERED_BREAKDOWN_ROWS = 15

PREMIUM_CODE = "PREMIUM-001"
PREMIUM_MIN_CALLS = 50
PREMIUM_MIN_PREMIUM_SHARE = 0.70
PREMIUM_MAX_INPUT_TOKEN_CV = 0.25
PREMIUM_MAX_OUTPUT_TOKEN_CV = 0.35
PREMIUM_MIN_AVG_OUTPUT_TOKENS = 10
PREMIUM_MAX_RETRY_RATE = 0.05
PREMIUM_MAX_TOOL_CALL_RATE = 0.10
PREMIUM_MAX_TRUNCATION_RATE = 0.02
PREMIUM_MIN_CANDIDATE_DISCOUNT = 0.50


@dataclass
class AuditDataQuality:
    total_events: int
    tagged_events: int
    tagged_fraction: float
    costed_events: int
    costed_fraction: float
    models: list[str]
    providers: list[str]
    methodology_versions: list[str]
    warnings: list[str]


@dataclass
class AuditBreakdownRow:
    dimension: str
    value: str
    requests: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    energy_wh: float
    carbon_g: float


@dataclass
class AuditFinding:
    code: str
    severity: str
    title: str
    scope: str
    description: str
    request_count: int
    evidence: dict[str, Any]
    confidence: str
    observed_avoidable_cost_usd: float | None
    projected_monthly_avoidable_cost_usd: float | None
    recommended_action: str
    automation_guidance: str
    security_signal: bool
    security_refs: tuple[str, ...]


@dataclass
class AuditReport:
    start_time: str
    end_time: str
    window_days: float
    total_requests: int
    total_tokens: int
    total_cost_usd: float
    total_energy_wh: float
    total_carbon_g: float
    observed_avoidable_cost_usd: float
    projected_monthly_avoidable_cost_usd: float
    realized_cache_savings_usd: float
    realized_cache_energy_savings_wh: float
    realized_cache_carbon_savings_g: float
    projected_monthly_cache_savings_usd: float
    circuit_breaker_interventions: int
    intervention_cost_at_risk_usd: float
    data_quality: AuditDataQuality
    findings: list[AuditFinding]
    breakdowns: list[AuditBreakdownRow]
    methodology_notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_audit_report(
    start: datetime,
    end: datetime,
    model: str | None = None,
    tags: dict[str, str] | None = None,
    tag_keys: tuple[str, ...] = DEFAULT_TAG_KEYS,
) -> AuditReport:
    """Build an audit report from locally stored events."""
    events = query_events(start=start, end=end, model=model, tags=tags)
    aggregate_summary = query_daily_usage(
        start=start,
        end=end,
        dimensions=tag_keys,
        model=model,
        tag_filter=tags,
    )
    use_aggregates = (
        aggregate_summary.total_requests > 0
        and aggregate_summary.total_requests > len(events)
    )
    window_seconds = max((end - start).total_seconds(), 1.0)
    window_days = window_seconds / 86400
    # Floor at 1 day for the projection denominator — sub-day windows produce
    # astronomical monthly projections that are not actionable.
    projection_days = max(window_days, 1.0)

    totals = (
        _summarize_usage_summary(aggregate_summary)
        if use_aggregates
        else _summarize_events(events)
    )
    data_quality = _data_quality(events)
    if use_aggregates:
        data_quality.warnings.append(
            "Executive totals use daily aggregates because some raw events are "
            "unavailable or compacted; findings are limited to retained raw events."
        )
    breakdowns = (
        _build_breakdowns_from_usage_summary(aggregate_summary, tag_keys)
        if use_aggregates
        else _build_breakdowns(events, tag_keys)
    )
    findings = _build_findings(events, window_days, tag_keys)

    stall_findings = [finding for finding in findings if finding.code == "STALL-001"]
    session_stall_findings = [
        finding for finding in stall_findings if finding.scope.startswith("session:")
    ]
    observed_basis = session_stall_findings or [
        finding for finding in stall_findings if finding.scope == "all"
    ]
    observed_avoidable = sum(
        finding.observed_avoidable_cost_usd or 0.0
        for finding in observed_basis
    )
    projected_monthly = (
        (observed_avoidable / projection_days) * MONTHLY_PROJECTION_DAYS
        if observed_avoidable
        else 0.0
    )

    methodology_versions = data_quality.methodology_versions or ["not recorded"]

    # Savings & interventions — query_usage reads from events + interventions tables
    savings_summary = query_usage(start=start, end=end, model=model, tags=tags)
    realized_cache_savings = savings_summary.total_cache_cost_saving_usd
    realized_cache_energy = savings_summary.total_cache_energy_saving_wh
    realized_cache_carbon = savings_summary.total_cache_carbon_saving_g
    projected_monthly_cache = (
        (realized_cache_savings / projection_days) * MONTHLY_PROJECTION_DAYS
        if realized_cache_savings
        else 0.0
    )

    return AuditReport(
        start_time=start.isoformat(),
        end_time=end.isoformat(),
        window_days=round(window_days, 4),
        total_requests=int(totals["requests"]),
        total_tokens=int(totals["input_tokens"] + totals["output_tokens"]),
        total_cost_usd=round(float(totals["cost_usd"]), 6),
        total_energy_wh=round(float(totals["energy_wh"]), 6),
        total_carbon_g=round(float(totals["carbon_g"]), 6),
        observed_avoidable_cost_usd=round(observed_avoidable, 6),
        projected_monthly_avoidable_cost_usd=round(projected_monthly, 6),
        realized_cache_savings_usd=round(realized_cache_savings, 6),
        realized_cache_energy_savings_wh=round(realized_cache_energy, 6),
        realized_cache_carbon_savings_g=round(realized_cache_carbon, 6),
        projected_monthly_cache_savings_usd=round(projected_monthly_cache, 6),
        circuit_breaker_interventions=savings_summary.total_circuit_breaker_interventions,
        intervention_cost_at_risk_usd=round(savings_summary.total_intervention_cost_at_risk_usd, 6),
        data_quality=data_quality,
        findings=findings,
        breakdowns=breakdowns,
        methodology_notes=[
            "This audit is deterministic and uses stored Vetch metadata only.",
            "Prompt and completion text are not required or inspected.",
            "Avoidable cost is only quantified where Vetch has direct observed evidence.",
            "Executive totals may use durable daily aggregates when raw events "
            "have been compacted.",
            "Cache and RAG findings are intentionally qualitative in this first pass.",
            "BABBLE-001 is a metadata-only proxy for unusually long generation; "
            "it does not inspect response content.",
            "PREMIUM-001 queues model-rightsizing candidates for eval; it does "
            "not prove a cheaper model is acceptable.",
            "Methodology versions observed: " + ", ".join(methodology_versions) + ".",
            "Energy and carbon figures are estimates from Vetch model and grid methodology.",
        ],
    )


def format_audit_report(report: AuditReport, output_format: str = "text") -> str:
    """Format an audit report as text, Markdown, or JSON."""
    if output_format == "json":
        return json.dumps(report.to_dict(), indent=2)
    if output_format == "markdown":
        return _format_markdown(report)
    return _format_text(report)


def _build_findings(
    events: list[dict[str, Any]],
    window_days: float,
    tag_keys: tuple[str, ...],
) -> list[AuditFinding]:
    scope_stats: dict[str, SessionStats] = {"all": SessionStats()}
    scope_counts: dict[str, int] = {"all": 0}
    for event in events:
        scopes = ["all"]
        session_id = event.get("session_id")
        if session_id:
            scopes.append(f"session:{session_id}")

        tags = event.get("tags") or {}
        if isinstance(tags, dict):
            for key in tag_keys:
                value = tags.get(key)
                if value:
                    scopes.append(f"{key}:{value}")

        for scope in scopes:
            scope_stats.setdefault(scope, SessionStats()).update(event)
            scope_counts[scope] = scope_counts.get(scope, 0) + 1

    findings: list[AuditFinding] = []
    seen: set[tuple[str, str]] = set()
    for scope, stats in scope_stats.items():
        if scope_counts.get(scope, 0) < MIN_FINDING_SCOPE_EVENTS:
            continue
        for advisory in generate_advisories(stats):
            dedupe_key = (scope, advisory.code)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            findings.append(_finding_from_advisory(advisory, stats, scope, window_days))

    findings.extend(_build_premium_findings(events))

    findings.sort(
        key=lambda f: (
            _severity_rank(f.severity),
            f.observed_avoidable_cost_usd or 0.0,
            _scope_rank(f.scope),
        ),
        reverse=True,
    )

    # Post-sort dedup: when multiple scopes produce the same advisory with
    # the same request count (e.g. scope="all" and scope="session:X" in a
    # single-session audit), keep only the most specific scope.
    # In multi-session audits, different session-scoped findings for the same
    # code will have different request counts and are preserved.
    seen_code_counts: set[tuple[str, int]] = set()
    seen_scoped_code_counts: set[tuple[str, str, int]] = set()
    deduped: list[AuditFinding] = []
    for f in findings:
        if f.code == PREMIUM_CODE:
            scoped_key = (f.code, f.scope, f.request_count or 0)
            if scoped_key in seen_scoped_code_counts:
                continue
            seen_scoped_code_counts.add(scoped_key)
            deduped.append(f)
            continue

        dedup_key = (f.code, f.request_count or 0)
        if dedup_key in seen_code_counts:
            continue
        seen_code_counts.add(dedup_key)
        deduped.append(f)

    return deduped[:MAX_FINDINGS]


def _build_premium_findings(events: list[dict[str, Any]]) -> list[AuditFinding]:
    """Find workflow-level model rightsizing candidates.

    This is deliberately an aggregate audit finding, not a runtime advisory.
    It queues stable premium-model workflows for evaluation; it does not claim
    the workflow is safe to downgrade or reroute automatically.
    """
    min_calls = int(get_advisory_threshold(PREMIUM_CODE, "min_calls", PREMIUM_MIN_CALLS))
    min_premium_share = get_advisory_threshold(
        PREMIUM_CODE, "min_premium_share", PREMIUM_MIN_PREMIUM_SHARE
    )
    max_input_cv = get_advisory_threshold(
        PREMIUM_CODE, "max_input_token_cv", PREMIUM_MAX_INPUT_TOKEN_CV
    )
    max_output_cv = get_advisory_threshold(
        PREMIUM_CODE, "max_output_token_cv", PREMIUM_MAX_OUTPUT_TOKEN_CV
    )
    min_avg_output = get_advisory_threshold(
        PREMIUM_CODE, "min_avg_output_tokens", PREMIUM_MIN_AVG_OUTPUT_TOKENS
    )
    max_retry_rate = get_advisory_threshold(
        PREMIUM_CODE, "max_retry_rate", PREMIUM_MAX_RETRY_RATE
    )
    max_tool_rate = get_advisory_threshold(
        PREMIUM_CODE, "max_tool_call_rate", PREMIUM_MAX_TOOL_CALL_RATE
    )
    max_truncation_rate = get_advisory_threshold(
        PREMIUM_CODE, "max_truncation_rate", PREMIUM_MAX_TRUNCATION_RATE
    )
    min_candidate_discount = get_advisory_threshold(
        PREMIUM_CODE,
        "min_candidate_discount",
        PREMIUM_MIN_CANDIDATE_DISCOUNT,
    )

    groups: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        identity = _workflow_identity(event)
        if identity is None:
            continue
        groups.setdefault(identity, []).append(event)

    findings: list[AuditFinding] = []
    for workflow, workflow_events in groups.items():
        requests = len(workflow_events)
        if requests < min_calls:
            continue

        input_tokens = [_event_input_tokens(event) for event in workflow_events]
        output_tokens = [_event_output_tokens(event) for event in workflow_events]
        input_cv = _coefficient_of_variation(input_tokens)
        output_cv = _coefficient_of_variation(output_tokens)
        avg_output_tokens = _average(output_tokens)
        if input_cv > max_input_cv or output_cv > max_output_cv:
            continue
        if avg_output_tokens < min_avg_output:
            continue

        retry_rate = _retry_event_rate(workflow_events)
        tool_call_rate = _tool_call_event_rate(workflow_events)
        truncation_rate = _truncation_rate(workflow_events)
        if (
            retry_rate > max_retry_rate
            or tool_call_rate > max_tool_rate
            or truncation_rate > max_truncation_rate
        ):
            continue

        model_stats = _premium_model_stats(workflow_events, min_candidate_discount)
        if model_stats is None:
            continue
        premium_share = float(model_stats["premium_share"])
        if premium_share < min_premium_share:
            continue

        confidence = _premium_confidence(
            requests=requests,
            premium_share=premium_share,
            input_cv=input_cv,
            output_cv=output_cv,
            retry_rate=retry_rate,
            tool_call_rate=tool_call_rate,
            truncation_rate=truncation_rate,
        )
        evidence = {
            "workflow_identity": workflow,
            "requests": requests,
            "premium_model": model_stats["premium_model"],
            "model_cost_class": "premium",
            "premium_share": round(premium_share, 4),
            "premium_cost_share": round(float(model_stats["premium_cost_share"]), 4),
            "input_token_cv": round(input_cv, 4),
            "output_token_cv": round(output_cv, 4),
            "avg_input_tokens": round(_average(input_tokens), 2),
            "avg_output_tokens": round(avg_output_tokens, 2),
            "retry_event_rate": round(retry_rate, 4),
            "tool_call_event_rate": round(tool_call_rate, 4),
            "truncation_rate": round(truncation_rate, 4),
            "candidate_models": model_stats["candidate_models"],
            "candidate_discount_threshold": round(min_candidate_discount, 4),
            "interpretation": (
                "Stable premium-model traffic should enter an eval queue. "
                "This finding is not a downgrade decision."
            ),
        }
        findings.append(
            AuditFinding(
                code=PREMIUM_CODE,
                severity="INFO",
                title="Large Model Rightsizing Candidate",
                scope=f"workflow:{workflow}",
                description=(
                    "This stable workflow is mostly running on a large, premium "
                    "model. Vetch cannot decide whether a smaller model is good "
                    "enough, but this is a good candidate for shadow evaluation."
                ),
                request_count=requests,
                evidence=evidence,
                confidence=confidence,
                observed_avoidable_cost_usd=None,
                projected_monthly_avoidable_cost_usd=None,
                recommended_action=(
                    "Run an eval against a standard or smaller candidate before "
                    "changing production routing."
                ),
                automation_guidance=(
                    "Do not auto-reroute from this finding alone; use it to queue "
                    "an offline or shadow evaluation."
                ),
                security_signal=False,
                security_refs=(),
            )
        )

    return findings


def _workflow_identity(event: dict[str, Any]) -> str | None:
    tags = event.get("tags") or {}
    if not isinstance(tags, dict):
        return None
    for key in PREMIUM_WORKFLOW_TAG_KEYS:
        value = tags.get(key)
        if value:
            return f"{key}:{value}"
    return None


def _premium_model_stats(
    events: list[dict[str, Any]],
    min_candidate_discount: float,
) -> dict[str, Any] | None:
    totals_by_model: dict[str, dict[str, float]] = {}
    total_cost = 0.0
    for event in events:
        model = str(event.get("model") or "")
        if not model:
            continue
        cost = float(event.get("estimated_cost_usd") or 0.0)
        total_cost += cost
        bucket = totals_by_model.setdefault(model, {"requests": 0.0, "cost": 0.0})
        bucket["requests"] += 1.0
        bucket["cost"] += cost

    candidates: list[dict[str, Any]] = []
    premium_requests = 0.0
    premium_cost = 0.0
    for model, totals in totals_by_model.items():
        model_info = _model_price_info(model)
        if model_info is None or model_info["model_cost_class"] != "premium":
            continue
        cheaper = _cheaper_candidate_models(
            model_info,
            min_candidate_discount=min_candidate_discount,
        )
        if not cheaper:
            continue
        premium_requests += totals["requests"]
        premium_cost += totals["cost"]
        candidates.append({
            "model": model,
            "requests": int(totals["requests"]),
            "cost": totals["cost"],
            "price": float(model_info["price"]),
            "candidate_models": cheaper,
        })

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item["requests"], item["cost"]), reverse=True)
    primary = candidates[0]
    request_count = len(events)
    premium_share = premium_requests / request_count if request_count else 0.0
    premium_cost_share = premium_cost / total_cost if total_cost > 0 else premium_share
    return {
        "premium_model": primary["model"],
        "premium_share": premium_share,
        "premium_cost_share": premium_cost_share,
        "candidate_models": primary["candidate_models"],
    }


def _model_price_info(model: str) -> dict[str, Any] | None:
    pricing = _pricing_registry()
    if not pricing:
        return None
    resolved = _resolve_pricing_model(model, pricing)
    if resolved is None:
        return None
    price = _weighted_model_price(pricing[resolved])
    provider = _model_provider(resolved)
    if price is None or provider == "unknown":
        return None
    return {
        "model": resolved,
        "provider": provider,
        "price": price,
        "model_cost_class": _model_cost_class(provider, price, pricing),
    }


def _pricing_registry() -> dict[str, dict[str, Any]]:
    from vetch import calculation

    calculation._load_registry()
    return calculation._PRICING or {}


def _resolve_pricing_model(
    model: str,
    pricing: dict[str, dict[str, Any]],
) -> str | None:
    if model in pricing:
        return model

    from vetch.calculation import resolve_model

    resolved, known = resolve_model(model)
    if known and resolved in pricing:
        return resolved

    parts = model.split("-")
    for i in range(len(parts) - 1, 0, -1):
        prefix = "-".join(parts[:i])
        if prefix in pricing:
            return prefix
    return None


def _weighted_model_price(entry: dict[str, Any]) -> float | None:
    try:
        rate_in = float(entry["usd_per_1k_input"])
        rate_out = float(entry["usd_per_1k_output"])
    except (KeyError, TypeError, ValueError):
        return None
    if rate_in == 0 and rate_out == 0:
        return 0.0
    # Output tokens are often the limiting cost lever. Give them more weight
    # without tying this audit signal to a content-specific workload.
    return (rate_in + (2 * rate_out)) / 3


def _model_provider(model: str) -> str:
    lower = model.lower()
    if lower.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if lower.startswith("claude"):
        return "anthropic"
    if lower.startswith("gemini"):
        return "google"
    if lower.startswith("deepseek"):
        return "deepseek"
    if lower.startswith(("llama", "mixtral")):
        return "local"
    return "unknown"


def _model_cost_class(provider: str, price: float, pricing: dict[str, dict[str, Any]]) -> str:
    if price == 0:
        return "local"
    same_provider_prices = sorted(
        candidate_price
        for candidate_model, entry in pricing.items()
        if _model_provider(candidate_model) == provider
        for candidate_price in [_weighted_model_price(entry)]
        if candidate_price is not None and candidate_price > 0
    )
    if not same_provider_prices:
        return "unknown"

    cheapest = same_provider_prices[0]
    if price >= cheapest * 3:
        return "premium"
    if price <= cheapest * 1.5:
        return "economy"
    return "standard"


def _cheaper_candidate_models(
    model_info: dict[str, Any],
    min_candidate_discount: float,
) -> list[str]:
    pricing = _pricing_registry()
    provider = str(model_info["provider"])
    current_model = str(model_info["model"])
    current_price = float(model_info["price"])
    max_candidate_price = current_price * min_candidate_discount
    candidates: list[tuple[float, str]] = []
    for candidate, entry in pricing.items():
        if candidate == current_model or _model_provider(candidate) != provider:
            continue
        price = _weighted_model_price(entry)
        if price is None or price <= 0 or price > max_candidate_price:
            continue
        candidates.append((price, candidate))
    candidates.sort()
    return [candidate for _, candidate in candidates[:5]]


def _premium_confidence(
    *,
    requests: int,
    premium_share: float,
    input_cv: float,
    output_cv: float,
    retry_rate: float,
    tool_call_rate: float,
    truncation_rate: float,
) -> str:
    if (
        requests >= 200
        and premium_share >= 0.9
        and input_cv <= 0.15
        and output_cv <= 0.20
        and retry_rate == 0
        and tool_call_rate == 0
        and truncation_rate == 0
    ):
        return "HIGH"
    if requests >= 100 and premium_share >= 0.8:
        return "MEDIUM"
    return "LOW"


def _event_input_tokens(event: dict[str, Any]) -> int:
    usage = event.get("usage", {}) or {}
    text = usage.get("text", {}) or {}
    return int(text.get("input_tokens") or event.get("input_tokens") or 0)


def _event_output_tokens(event: dict[str, Any]) -> int:
    usage = event.get("usage", {}) or {}
    text = usage.get("text", {}) or {}
    return int(text.get("output_tokens") or event.get("output_tokens") or 0)


def _coefficient_of_variation(values: list[int]) -> float:
    if not values:
        return 0.0
    mean = _average(values)
    if mean == 0:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / mean


def _average(values: list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _retry_event_rate(events: list[dict[str, Any]]) -> float:
    if not events:
        return 0.0
    retry_events = sum(1 for event in events if int(event.get("retry_count") or 0) > 0)
    return retry_events / len(events)


def _tool_call_event_rate(events: list[dict[str, Any]]) -> float:
    if not events:
        return 0.0
    tool_events = sum(
        1 for event in events if int(event.get("tool_call_count") or 0) > 0
    )
    return tool_events / len(events)


def _truncation_rate(events: list[dict[str, Any]]) -> float:
    if not events:
        return 0.0
    truncated = 0
    for event in events:
        finish_reason = str(event.get("finish_reason") or "").lower()
        if finish_reason in {"length", "max_tokens", "max_output_tokens"}:
            truncated += 1
    return truncated / len(events)


def _finding_from_advisory(
    advisory: Advisory,
    stats: SessionStats,
    scope: str,
    window_days: float,
) -> AuditFinding:
    observed = advisory.potential_savings_usd
    projected = (
        (observed / window_days) * MONTHLY_PROJECTION_DAYS
        if observed
        else None
    )
    spec = get_advisory_spec(advisory.code)

    return AuditFinding(
        code=advisory.code,
        severity=advisory.severity,
        title=advisory.title,
        scope=scope,
        description=advisory.description,
        request_count=advisory.request_count,
        evidence=spec.evidence(stats),
        confidence=spec.confidence(stats),
        observed_avoidable_cost_usd=round(observed, 6) if observed else None,
        projected_monthly_avoidable_cost_usd=round(projected, 6) if projected else None,
        recommended_action=spec.recommended_action,
        automation_guidance=spec.automation_guidance,
        security_signal=advisory.security_signal,
        security_refs=advisory.security_refs,
    )


def _summarize_events(events: list[dict[str, Any]]) -> dict[str, float]:
    summary = _empty_summary()
    for event in events:
        _add_event_to_summary(summary, event)
    return summary


def _summarize_usage_summary(summary: UsageSummary) -> dict[str, float]:
    return {
        "requests": float(summary.total_requests),
        "input_tokens": float(summary.total_input_tokens),
        "output_tokens": float(summary.total_output_tokens),
        "cost_usd": float(summary.total_cost_usd),
        "energy_wh": float(summary.total_energy_wh),
        "carbon_g": float(summary.total_carbon_g),
    }


def _empty_summary() -> dict[str, float]:
    return {
        "requests": 0.0,
        "input_tokens": 0.0,
        "output_tokens": 0.0,
        "cost_usd": 0.0,
        "energy_wh": 0.0,
        "carbon_g": 0.0,
    }


def _add_event_to_summary(summary: dict[str, float], event: dict[str, Any]) -> None:
    summary["requests"] += 1.0
    usage = event.get("usage", {}) or {}
    text = usage.get("text", {}) or {}
    summary["input_tokens"] += float(text.get("input_tokens") or 0)
    summary["output_tokens"] += float(text.get("output_tokens") or 0)
    summary["cost_usd"] += float(event.get("estimated_cost_usd") or 0)
    summary["energy_wh"] += float(event.get("estimated_energy_wh") or 0)
    summary["carbon_g"] += float(event.get("estimated_carbon_g") or 0)


def _data_quality(events: list[dict[str, Any]]) -> AuditDataQuality:
    total = len(events)
    tagged = sum(1 for event in events if event.get("tags"))
    costed = sum(1 for event in events if event.get("estimated_cost_usd") is not None)
    models = sorted({str(event.get("model")) for event in events if event.get("model")})
    providers = sorted({
        str(event.get("provider")) for event in events if event.get("provider")
    })
    methodology_versions = sorted({
        str(event.get("methodology_version"))
        for event in events
        if event.get("methodology_version")
    })

    tagged_fraction = tagged / total if total else 0.0
    costed_fraction = costed / total if total else 0.0

    warnings: list[str] = []
    if total == 0:
        warnings.append("No stored events were found for this audit window.")
    elif total < LOW_VOLUME_EVENT_WARNING_THRESHOLD:
        warnings.append("Fewer than 20 events: useful for smoke tests, weak for decisions.")
    if total and tagged_fraction < MIN_TAGGED_FRACTION:
        warnings.append("Less than half of events have tags; attribution will be limited.")
    if total and costed_fraction < MIN_COSTED_FRACTION:
        warnings.append("Some events lack cost estimates; spend totals may be incomplete.")
    if not models and total:
        warnings.append("No model names found on events; model breakdown is unavailable.")

    return AuditDataQuality(
        total_events=total,
        tagged_events=tagged,
        tagged_fraction=round(tagged_fraction, 4),
        costed_events=costed,
        costed_fraction=round(costed_fraction, 4),
        models=models,
        providers=providers,
        methodology_versions=methodology_versions,
        warnings=warnings,
    )


def _build_breakdowns(
    events: list[dict[str, Any]],
    tag_keys: tuple[str, ...],
) -> list[AuditBreakdownRow]:
    summaries: dict[tuple[str, str], dict[str, float]] = {}
    for event in events:
        model = event.get("model")
        if model:
            _add_event_to_summary(
                summaries.setdefault(("model", str(model)), _empty_summary()),
                event,
            )

        tags = event.get("tags") or {}
        if isinstance(tags, dict):
            for key in tag_keys:
                value = tags.get(key)
                if value:
                    _add_event_to_summary(
                        summaries.setdefault((key, str(value)), _empty_summary()),
                        event,
                    )

    rows = [
        _breakdown_row(dimension, value, totals)
        for (dimension, value), totals in summaries.items()
    ]

    rows.sort(key=lambda row: row.cost_usd, reverse=True)
    return rows[:MAX_BREAKDOWN_ROWS]


def _build_breakdowns_from_usage_summary(
    summary: UsageSummary,
    tag_keys: tuple[str, ...],
) -> list[AuditBreakdownRow]:
    rows: list[AuditBreakdownRow] = []
    for model, data in summary.by_model.items():
        rows.append(_breakdown_row("model", model, _summary_from_breakdown_data(data)))

    for tag_key in tag_keys:
        tag_values = summary.by_tag.get(tag_key) or {}
        for tag_value, data in tag_values.items():
            rows.append(
                _breakdown_row(
                    tag_key,
                    str(tag_value),
                    _summary_from_breakdown_data(data),
                )
            )

    rows.sort(key=lambda row: row.cost_usd, reverse=True)
    return rows[:MAX_BREAKDOWN_ROWS]


def _summary_from_breakdown_data(data: dict[str, Any]) -> dict[str, float]:
    return {
        "requests": float(data.get("requests") or 0),
        "input_tokens": float(data.get("input_tokens") or 0),
        "output_tokens": float(data.get("output_tokens") or 0),
        "cost_usd": float(data.get("cost_usd") or 0),
        "energy_wh": float(data.get("energy_wh") or 0),
        "carbon_g": float(data.get("carbon_g") or 0),
    }


def _breakdown_row(
    dimension: str,
    value: str,
    totals: dict[str, float],
) -> AuditBreakdownRow:
    return AuditBreakdownRow(
        dimension=dimension,
        value=value,
        requests=int(totals["requests"]),
        input_tokens=int(totals["input_tokens"]),
        output_tokens=int(totals["output_tokens"]),
        cost_usd=round(float(totals["cost_usd"]), 6),
        energy_wh=round(float(totals["energy_wh"]), 6),
        carbon_g=round(float(totals["carbon_g"]), 6),
    )


def _severity_rank(severity: str) -> int:
    return {"INFO": 1, "WARNING": 2, "CRITICAL": 3}.get(severity, 0)


def _scope_rank(scope: str) -> int:
    if scope.startswith("session:"):
        return 3
    if ":" in scope:
        return 2
    return 1


def _format_money(value: float | None) -> str:
    if value is None:
        return "not estimated"
    return f"${value:,.2f}"


def _format_text(report: AuditReport) -> str:
    lines = [
        "Vetch Inference Waste Audit",
        "=" * 50,
        f"Period: {report.start_time} to {report.end_time}",
        f"Requests: {report.total_requests:,}",
        f"Tokens: {report.total_tokens:,}",
        f"Cost: ${report.total_cost_usd:,.2f}",
        f"Energy: {report.total_energy_wh:,.2f} Wh",
        f"Carbon: {report.total_carbon_g:,.2f} gCO2e",
        f"Observed avoidable cost: ${report.observed_avoidable_cost_usd:,.2f}",
        "Projected monthly avoidable cost: "
        f"${report.projected_monthly_avoidable_cost_usd:,.2f}",
        "",
        "Data quality",
        "-" * 30,
        f"Tagged events: {report.data_quality.tagged_events:,}/"
        f"{report.data_quality.total_events:,} "
        f"({report.data_quality.tagged_fraction:.0%})",
        f"Costed events: {report.data_quality.costed_events:,}/"
        f"{report.data_quality.total_events:,} "
        f"({report.data_quality.costed_fraction:.0%})",
        "Methodology versions: "
        f"{', '.join(report.data_quality.methodology_versions) or 'not recorded'}",
    ]
    for warning in report.data_quality.warnings:
        lines.append(f"WARNING: {warning}")

    lines.extend([
        "",
        "Savings & Interventions",
        "-" * 30,
        "Realized cache savings",
        f"  Cost saved via caching:       ${report.realized_cache_savings_usd:,.2f}",
        f"  Energy saved via caching:     {report.realized_cache_energy_savings_wh:,.2f} Wh",
        f"  Carbon saved via caching:     {report.realized_cache_carbon_savings_g:,.2f} gCO2e",
        (
            "  Monthly run-rate:             "
            f"${report.projected_monthly_cache_savings_usd:,.2f} / month"
        ),
        "",
        "Circuit breaker interventions",
        f"  Interventions:                {report.circuit_breaker_interventions}",
        f"  Cost at risk interrupted:     ${report.intervention_cost_at_risk_usd:,.2f}",
        "",
    ])

    lines.extend(["Findings", "-" * 30])
    if not report.findings:
        lines.append("No waste advisories found for this window.")
    for finding in report.findings:
        security_suffix = " [security signal]" if finding.security_signal else ""
        lines.extend([
            f"[{finding.severity}] {finding.code}{security_suffix} - {finding.title}",
            f"Scope: {finding.scope}",
            f"Confidence: {finding.confidence}",
            f"Description: {finding.description}",
            f"Observed avoidable cost: "
            f"{_format_money(finding.observed_avoidable_cost_usd)}",
            f"Projected monthly avoidable cost: "
            f"{_format_money(finding.projected_monthly_avoidable_cost_usd)}",
            f"Action: {finding.recommended_action}",
            f"Automation: {finding.automation_guidance}",
        ])
        if finding.security_refs:
            lines.append(f"Security refs: {', '.join(finding.security_refs)}")
        lines.append("")

    if report.breakdowns:
        lines.extend(["Top breakdowns", "-" * 30])
        for row in report.breakdowns[:MAX_RENDERED_BREAKDOWN_ROWS]:
            lines.append(
                f"{row.dimension}:{row.value} - {row.requests:,} requests, "
                f"${row.cost_usd:,.2f}, {row.input_tokens + row.output_tokens:,} tokens"
            )

    return "\n".join(lines)


def _format_markdown(report: AuditReport) -> str:
    lines = [
        "# Vetch Inference Waste Audit",
        "",
        f"**Period:** `{report.start_time}` to `{report.end_time}`",
        "",
        "## Executive Summary",
        "",
        f"- Requests: **{report.total_requests:,}**",
        f"- Tokens: **{report.total_tokens:,}**",
        f"- Cost: **${report.total_cost_usd:,.2f}**",
        f"- Energy: **{report.total_energy_wh:,.2f} Wh**",
        f"- Carbon: **{report.total_carbon_g:,.2f} gCO2e**",
        f"- Observed avoidable cost: **${report.observed_avoidable_cost_usd:,.2f}**",
        "- Projected monthly avoidable cost: "
        f"**${report.projected_monthly_avoidable_cost_usd:,.2f}**",
        "",
        "## Data Quality",
        "",
        f"- Tagged events: {report.data_quality.tagged_events:,}/"
        f"{report.data_quality.total_events:,} "
        f"({report.data_quality.tagged_fraction:.0%})",
        f"- Costed events: {report.data_quality.costed_events:,}/"
        f"{report.data_quality.total_events:,} "
        f"({report.data_quality.costed_fraction:.0%})",
        f"- Models: {', '.join(report.data_quality.models) or 'not available'}",
        f"- Providers: {', '.join(report.data_quality.providers) or 'not available'}",
        "- Methodology versions: "
        f"{', '.join(report.data_quality.methodology_versions) or 'not recorded'}",
    ]
    if report.data_quality.warnings:
        lines.extend(["", "### Warnings", ""])
        for warning in report.data_quality.warnings:
            lines.append(f"- {warning}")

    lines.extend([
        "",
        "## Savings & Interventions",
        "",
        "**Realized cache savings** (actual, measurable)",
        "",
        f"- Cost saved via caching: **${report.realized_cache_savings_usd:,.2f}**",
        f"- Energy saved via caching: **{report.realized_cache_energy_savings_wh:,.2f} Wh**",
        f"- Carbon saved via caching: **{report.realized_cache_carbon_savings_g:,.2f} gCO2e**",
        (
            "- Monthly run-rate: "
            f"**${report.projected_monthly_cache_savings_usd:,.2f} / month**"
        ),
        "",
        (
            "**Circuit breaker interventions** "
            "(cost protected — reported separately, not guaranteed savings)"
        ),
        "",
        f"- Interventions: **{report.circuit_breaker_interventions}**",
        f"- Cost at risk interrupted: **${report.intervention_cost_at_risk_usd:,.2f}**",
        "",
        "## Findings",
        "",
    ])
    if not report.findings:
        lines.append("No waste advisories found for this window.")
    else:
        for finding in report.findings:
            title_suffix = " 🔒" if finding.security_signal else ""
            lines.extend([
                f"### {finding.code}{title_suffix}: {finding.title}",
                "",
                f"- Severity: **{finding.severity}**",
                f"- Scope: `{finding.scope}`",
                f"- Confidence: **{finding.confidence}**",
                f"- Description: {finding.description}",
                f"- Observed avoidable cost: "
                f"**{_format_money(finding.observed_avoidable_cost_usd)}**",
                f"- Projected monthly avoidable cost: "
                f"**{_format_money(finding.projected_monthly_avoidable_cost_usd)}**",
                f"- Recommended action: {finding.recommended_action}",
                f"- Automation guidance: {finding.automation_guidance}",
            ])
            if finding.security_refs:
                lines.append(f"- Security refs: {', '.join(finding.security_refs)}")
            lines.extend([
                "",
                "<details><summary>Evidence</summary>",
                "",
                "```json",
                json.dumps(finding.evidence, indent=2, sort_keys=True),
                "```",
                "",
                "</details>",
                "",
            ])

    if report.breakdowns:
        lines.extend([
            "## Top Breakdowns",
            "",
            "| Dimension | Value | Requests | Tokens | Cost | Energy | Carbon |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in report.breakdowns[:MAX_RENDERED_BREAKDOWN_ROWS]:
            tokens = row.input_tokens + row.output_tokens
            lines.append(
                f"| {row.dimension} | {row.value} | {row.requests:,} | "
                f"{tokens:,} | ${row.cost_usd:,.2f} | {row.energy_wh:,.2f} Wh | "
                f"{row.carbon_g:,.2f} gCO2e |"
            )

    lines.extend(["", "## Methodology Notes", ""])
    for note in report.methodology_notes:
        lines.append(f"- {note}")

    return "\n".join(lines)
