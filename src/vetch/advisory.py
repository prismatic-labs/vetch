"""Advisory engine for identifying inefficiency patterns.

Analyzes session or historical data to find:
- Redundant system prompts (Prompt Caching opportunity)
- High energy/low complexity usage
- Potential model downgrades
"""

from __future__ import annotations

from typing import NamedTuple

from vetch.stats import STALL_FRACTION_TRIGGER, SessionStats


class Advisory(NamedTuple):
    code: str
    severity: str  # "INFO", "WARNING", "CRITICAL"
    title: str
    description: str
    potential_savings_usd: float | None = None
    request_count: int = 0


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
