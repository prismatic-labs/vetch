"""Advisory engine for identifying inefficiency patterns.

Analyzes session or historical data to find:
- Redundant system prompts (Prompt Caching opportunity)
- High energy/low complexity usage
- Potential model downgrades
"""

from __future__ import annotations

from typing import Any, NamedTuple

from vetch.stats import SessionStats


class Advisory(NamedTuple):
    code: str
    severity: str  # "INFO", "WARNING", "CRITICAL"
    title: str
    description: str
    potential_savings_usd: float | None = None


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
