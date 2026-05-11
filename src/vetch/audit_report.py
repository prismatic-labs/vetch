"""Deterministic inference waste audit reports.

This module turns locally stored Vetch metadata into a consulting-ready
audit report. It is deliberately rules-based: no prompts, completions, or
LLM calls are required to generate the report.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from vetch.advisory import Advisory, generate_advisories, get_advisory_spec
from vetch.stats import SessionStats
from vetch.storage import UsageSummary, query_daily_usage, query_events

DEFAULT_TAG_KEYS = ("feature", "customer", "workflow", "team", "service", "env")
MIN_FINDING_SCOPE_EVENTS = 2
LOW_VOLUME_EVENT_WARNING_THRESHOLD = 20
MIN_TAGGED_FRACTION = 0.5
MIN_COSTED_FRACTION = 0.8
MONTHLY_PROJECTION_DAYS = 30
MAX_FINDINGS = 25
MAX_BREAKDOWN_ROWS = 30
MAX_RENDERED_BREAKDOWN_ROWS = 15


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
    evidence: dict[str, Any]
    confidence: str
    observed_avoidable_cost_usd: float | None
    projected_monthly_avoidable_cost_usd: float | None
    recommended_action: str
    automation_guidance: str


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
    aggregate_summary = (
        query_daily_usage(start=start, end=end, dimensions=tag_keys)
        if model is None and tags is None
        else UsageSummary(start, end)
    )
    use_aggregates = (
        aggregate_summary.total_requests > 0
        and aggregate_summary.total_requests > len(events)
    )
    window_seconds = max((end - start).total_seconds(), 1.0)
    window_days = window_seconds / 86400

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
        (observed_avoidable / window_days) * MONTHLY_PROJECTION_DAYS
        if observed_avoidable
        else 0.0
    )

    methodology_versions = data_quality.methodology_versions or ["not recorded"]

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

    findings.sort(
        key=lambda f: (
            _severity_rank(f.severity),
            f.observed_avoidable_cost_usd or 0.0,
            _scope_rank(f.scope),
        ),
        reverse=True,
    )
    return findings[:MAX_FINDINGS]


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
        evidence=spec.evidence(stats),
        confidence=spec.confidence(stats),
        observed_avoidable_cost_usd=round(observed, 6) if observed else None,
        projected_monthly_avoidable_cost_usd=round(projected, 6) if projected else None,
        recommended_action=spec.recommended_action,
        automation_guidance=spec.automation_guidance,
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

    lines.extend(["", "Findings", "-" * 30])
    if not report.findings:
        lines.append("No waste advisories found for this window.")
    for finding in report.findings:
        lines.extend([
            f"[{finding.severity}] {finding.code} - {finding.title}",
            f"Scope: {finding.scope}",
            f"Confidence: {finding.confidence}",
            f"Observed avoidable cost: "
            f"{_format_money(finding.observed_avoidable_cost_usd)}",
            f"Projected monthly avoidable cost: "
            f"{_format_money(finding.projected_monthly_avoidable_cost_usd)}",
            f"Action: {finding.recommended_action}",
            f"Automation: {finding.automation_guidance}",
            "",
        ])

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

    lines.extend(["", "## Findings", ""])
    if not report.findings:
        lines.append("No waste advisories found for this window.")
    else:
        for finding in report.findings:
            lines.extend([
                f"### {finding.code}: {finding.title}",
                "",
                f"- Severity: **{finding.severity}**",
                f"- Scope: `{finding.scope}`",
                f"- Confidence: **{finding.confidence}**",
                f"- Observed avoidable cost: "
                f"**{_format_money(finding.observed_avoidable_cost_usd)}**",
                f"- Projected monthly avoidable cost: "
                f"**{_format_money(finding.projected_monthly_avoidable_cost_usd)}**",
                f"- Recommended action: {finding.recommended_action}",
                f"- Automation guidance: {finding.automation_guidance}",
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
