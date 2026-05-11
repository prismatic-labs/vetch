"""Advisory engine for identifying inefficiency patterns.

Analyzes session or historical data to find:
- Redundant system prompts (Prompt Caching opportunity)
- High energy/low complexity usage
- Potential model downgrades
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

from vetch.stats import STALL_FRACTION_TRIGGER, SessionStats

BABBLE_AVG_OUTPUT_TOKENS_TRIGGER = 1500
BABBLE_HIGH_AVG_OUTPUT_TOKENS_TRIGGER = 3000
BABBLE_MIN_REQUESTS = 10


class Advisory(NamedTuple):
    code: str
    severity: str  # "INFO", "WARNING", "CRITICAL"
    title: str
    description: str
    potential_savings_usd: float | None = None
    request_count: int = 0


@dataclass(frozen=True)
class AdvisorySpec:
    """Human-facing metadata and evidence extraction for an advisory code."""

    recommended_action: str
    automation_guidance: str
    evidence: Callable[[SessionStats], dict[str, Any]]
    confidence: Callable[[SessionStats], str]


def _summary_evidence(*keys: str) -> Callable[[SessionStats], dict[str, Any]]:
    def evidence(stats: SessionStats) -> dict[str, Any]:
        summary = stats.summary()
        return {key: summary.get(key) for key in keys}

    return evidence


def _stall_confidence(stats: SessionStats) -> str:
    summary = stats.summary()
    window_size = int(summary.get("recent_window_size") or 0)
    low_fraction = float(summary.get("recent_low_output_fraction") or 0.0)
    input_similarity = float(summary.get("recent_input_similarity") or 0.0)
    if window_size >= 20 and low_fraction >= 0.9 and input_similarity >= 0.8:
        return "high"
    if window_size >= 10 and low_fraction >= STALL_FRACTION_TRIGGER:
        return "medium"
    return "low"


def _cache_confidence(stats: SessionStats) -> str:
    if stats.total_requests >= 50:
        return "medium"
    return "low"


def _rag_confidence(stats: SessionStats) -> str:
    summary = stats.summary()
    ratio = float(summary.get("average_input_output_ratio") or 0.0)
    if stats.total_requests >= 20 and ratio > 100:
        return "medium"
    return "low"


def _babble_confidence(stats: SessionStats) -> str:
    summary = stats.summary()
    avg_output_tokens = float(summary.get("recent_avg_output_tokens") or 0.0)
    if (
        stats.total_requests >= 20
        and avg_output_tokens >= BABBLE_HIGH_AVG_OUTPUT_TOKENS_TRIGGER
    ):
        return "high"
    if (
        stats.total_requests >= BABBLE_MIN_REQUESTS
        and avg_output_tokens >= BABBLE_AVG_OUTPUT_TOKENS_TRIGGER
    ):
        return "medium"
    return "low"


_ADVISORY_SPECS: dict[str, AdvisorySpec] = {
    "STALL-001": AdvisorySpec(
        recommended_action=(
            "Inspect the affected session, then consider warn -> kill or reroute "
            "only after confirming the low-output loop is not expected behaviour."
        ),
        automation_guidance=(
            "Safe candidate for automatic intervention after workflow-specific evals."
        ),
        evidence=_summary_evidence(
            "total_requests",
            "recent_window_size",
            "recent_low_output_count",
            "recent_low_output_fraction",
            "recent_input_similarity",
            "recent_stalled_cost_usd",
        ),
        confidence=_stall_confidence,
    ),
    "CACHE-001": AdvisorySpec(
        recommended_action=(
            "Review repeated prompt structure and enable provider prompt caching "
            "where supported."
        ),
        automation_guidance="Advisory only; cache configuration is application-specific.",
        evidence=lambda stats: {
            "total_requests": stats.total_requests,
            "repeated_input_token_counts": {
                str(token_count): count
                for token_count, count in stats.input_token_counts.items()
                if count > 1
            },
        },
        confidence=_cache_confidence,
    ),
    "RAG-001": AdvisorySpec(
        recommended_action=(
            "Review retrieval chunk count, duplicate chunks, relevance threshold, "
            "and whether retrieved context is cited or used."
        ),
        automation_guidance=(
            "Warn-only; RAG-heavy workflows need task-aware review before automation."
        ),
        evidence=_summary_evidence(
            "total_requests",
            "total_input_tokens",
            "total_output_tokens",
            "average_input_output_ratio",
        ),
        confidence=_rag_confidence,
    ),
    "BABBLE-001": AdvisorySpec(
        recommended_action=(
            "Review whether long generations are expected for this workflow. If not, "
            "tighten max_tokens, stop conditions, or response format constraints."
        ),
        automation_guidance=(
            "Warn-only until task-specific quality checks confirm shorter responses "
            "preserve usefulness."
        ),
        evidence=_summary_evidence(
            "total_requests",
            "recent_window_size",
            "recent_avg_output_tokens",
            "total_output_tokens",
        ),
        confidence=_babble_confidence,
    ),
}


def get_advisory_spec(code: str) -> AdvisorySpec:
    """Return consulting metadata for an advisory code."""
    return _ADVISORY_SPECS.get(
        code,
        AdvisorySpec(
            recommended_action="Review the advisory evidence before changing production policy.",
            automation_guidance="No automation guidance is registered for this advisory.",
            evidence=lambda stats: stats.summary(),
            confidence=lambda stats: "low",
        ),
    )


def generate_advisories(stats: SessionStats) -> list[Advisory]:
    """Generate advisories based on session statistics."""
    advisories = []

    # 1. Prompt Caching Opportunity
    # Logic: If many requests have identical input token counts, they likely share a system prompt.
    if stats.total_requests > 5:
        repeated_inputs = 0
        for count in stats.input_token_counts.values():
            if count > 1:
                repeated_inputs += count

        repetition_rate = repeated_inputs / stats.total_requests
        if repetition_rate > 0.5:
            advisories.append(Advisory(
                code="CACHE-001",
                severity="WARNING",
                title="High Input Repetition Detected",
                description=(
                    f"{repetition_rate:.0%} of requests share identical input token counts. "
                    "This suggests a static system prompt. "
                    "Enable Prompt Caching (Anthropic/DeepSeek) to reduce input costs by up to 90%."
                )
            ))

    # 2. Input/Output Imbalance (RAG Pattern)
    summary = stats.summary()
    ratio = summary.get("average_input_output_ratio")
    if ratio and ratio > 50:
        advisories.append(Advisory(
            code="RAG-001",
            severity="INFO",
            title="Heavy RAG Pattern Detected",
            description=(
                f"Average Input:Output ratio is {ratio:.1f}:1. "
                "Ensure your retriever is optimized. "
                "Consider compressing context or using a smaller model for the reading phase."
            )
        ))

    # 3. Agentic Stall Detection
    # Fires when ≥80 % of the recent window produces <5 output tokens AND
    # the inputs are repetitive (suggests the model is stuck on the same
    # prompt, not legitimately sending small tool pings).
    recent_low = summary.get("recent_low_output_count", 0)
    low_fraction = summary.get("recent_low_output_fraction", 0.0)
    window_size = summary.get("recent_window_size", 0)
    input_similarity = summary.get("recent_input_similarity", 0.0)
    stalled_cost = summary.get("recent_stalled_cost_usd", 0.0)

    if (
        stats.total_requests > 10
        and window_size >= 10
        and low_fraction >= STALL_FRACTION_TRIGGER
        and input_similarity >= 0.5
    ):
        severity = "CRITICAL" if stalled_cost > 5.0 else "WARNING"
        advisories.append(Advisory(
            code="STALL-001",
            severity=severity,
            title="Agentic Stall Detected",
            description=(
                f"{recent_low} of last {window_size} calls produced <5 output tokens "
                f"({low_fraction:.0%} of window) with {input_similarity:.0%} input "
                "similarity. This suggests a stalled agentic loop — the model may be "
                "stuck on repeated tool failures or empty completions. "
                f"Estimated wasted cost: ${stalled_cost:.2f}."
            ),
            potential_savings_usd=stalled_cost,
            request_count=recent_low,
        ))

    # 4. Babbling / Excessive Generation Proxy
    # Metadata-only first pass: unusually high recent average output length.
    # This does not inspect output content, so it is intentionally warn-only.
    avg_output_tokens = float(summary.get("recent_avg_output_tokens") or 0.0)
    if (
        stats.total_requests >= BABBLE_MIN_REQUESTS
        and avg_output_tokens >= BABBLE_AVG_OUTPUT_TOKENS_TRIGGER
    ):
        advisories.append(Advisory(
            code="BABBLE-001",
            severity="WARNING",
            title="Unusually Long Generation Detected",
            description=(
                f"Recent calls average {avg_output_tokens:,.0f} output tokens. "
                "If this workflow does not require long-form generation, the model "
                "may be producing redundant text. Review max_tokens, stop conditions, "
                "and response format constraints."
            ),
            request_count=int(window_size),
        ))

    return advisories


def format_advisories(advisories: list[Advisory], format: str = "text") -> str:
    """Format advisories for display."""
    if not advisories:
        return "No advisories found."

    if format == "json":
        import json
        return json.dumps([a._asdict() for a in advisories], indent=2)

    lines = ["\n🛡️  Vetch Advisories"]
    lines.append("=" * 50)

    for adv in advisories:
        icon = "🔴" if adv.severity == "CRITICAL" else "🟡" if adv.severity == "WARNING" else "🔵"
        lines.append(f"{icon} [{adv.code}] {adv.title}")
        lines.append(f"   {adv.description}")
        if adv.potential_savings_usd:
            lines.append(f"   Potential Savings: ${adv.potential_savings_usd:.2f}")
        lines.append("")

    return "\n".join(lines)
