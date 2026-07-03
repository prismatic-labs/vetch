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

from vetch.stats import (
    EMPTY_VISIBLE_OUTPUT_TOKEN_THRESHOLD,
    STALL_FRACTION_TRIGGER,
    STALL_LOW_OUTPUT_THRESHOLD,
    SessionStats,
)

TRUNC_MIN_WINDOW = 5
TRUNC_FRACTION_TRIGGER = 0.5
TRUNC_MIN_COUNT = 3
BABBLE_AVG_OUTPUT_TOKENS_TRIGGER = 1500
BABBLE_HIGH_AVG_OUTPUT_TOKENS_TRIGGER = 3000
BABBLE_MIN_REQUESTS = 10
ZOMBIE_INPUT_SIMILARITY_TRIGGER = 0.80
ZOMBIE_OUTPUT_CV_TRIGGER = 0.15
ZOMBIE_MIN_WINDOW = 5
CONTEXT_SNOWBALL_MIN_WINDOW = 8
CONTEXT_SNOWBALL_GROWTH_TRIGGER = 3.0
CONTEXT_SNOWBALL_INCREASE_TRIGGER = 0.70
CONTEXT_SNOWBALL_RATIO_TRIGGER = 4.0
EMPTY_VISIBLE_MIN_WINDOW = 5
EMPTY_VISIBLE_FRACTION_TRIGGER = 0.50
ERROR_MIN_WINDOW = 5
ERROR_CONSECUTIVE_TRIGGER = 3
ERROR_FRACTION_TRIGGER = 0.40
CACHE2_MIN_CALLS = 6
CACHE2_REPETITION_TRIGGER = 0.50
CACHE2_MISS_FRACTION_TRIGGER = 0.80
STREAM1_MIN_WINDOW = 5
STREAM1_INCOMPLETE_FRACTION_TRIGGER = 0.30
REASONING1_MIN_CALLS = 5
REASONING1_MISSING_FRACTION_TRIGGER = 0.40


class Advisory(NamedTuple):
    code: str
    severity: str  # "INFO", "WARNING", "CRITICAL"
    title: str
    description: str
    potential_savings_usd: float | None = None
    request_count: int = 0
    security_signal: bool = False
    security_refs: tuple[str, ...] = ()


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


def _threshold(stats: SessionStats, code: str, key: str, default: float) -> float:
    return stats.advisory_threshold(code, key, default)


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
    medium_trigger = _threshold(
        stats,
        "BABBLE-001",
        "min_avg_output_tokens",
        BABBLE_AVG_OUTPUT_TOKENS_TRIGGER,
    )
    high_trigger = _threshold(
        stats,
        "BABBLE-001",
        "high_avg_output_tokens",
        BABBLE_HIGH_AVG_OUTPUT_TOKENS_TRIGGER,
    )
    if (
        stats.total_requests >= 20
        and avg_output_tokens >= high_trigger
    ):
        return "high"
    if (
        stats.total_requests >= BABBLE_MIN_REQUESTS
        and avg_output_tokens >= medium_trigger
    ):
        return "medium"
    return "low"


def _trunc_confidence(stats: SessionStats) -> str:
    summary = stats.summary()
    count = int(summary.get("recent_max_tokens_finish_count") or 0)
    fraction = float(summary.get("recent_max_tokens_finish_fraction") or 0.0)
    min_count = int(_threshold(stats, "TRUNC-001", "min_count", TRUNC_MIN_COUNT))
    trigger = _threshold(
        stats,
        "TRUNC-001",
        "fraction_trigger",
        TRUNC_FRACTION_TRIGGER,
    )
    if count >= 8 and fraction >= 0.8:
        return "high"
    if count >= min_count and fraction >= trigger:
        return "medium"
    return "low"


def _zombie_confidence(stats: SessionStats) -> str:
    summary = stats.summary()
    window_size = int(summary.get("recent_window_size") or 0)
    input_similarity = float(summary.get("recent_input_similarity") or 0.0)
    output_cv = float(summary.get("recent_output_token_cv") or 0.0)
    min_window = int(_threshold(stats, "ZOMBIE-001", "min_window", ZOMBIE_MIN_WINDOW))
    input_trigger = _threshold(
        stats,
        "ZOMBIE-001",
        "input_similarity",
        ZOMBIE_INPUT_SIMILARITY_TRIGGER,
    )
    output_cv_trigger = _threshold(
        stats,
        "ZOMBIE-001",
        "output_cv",
        ZOMBIE_OUTPUT_CV_TRIGGER,
    )
    if (
        window_size >= 10
        and input_similarity >= 0.9
        and output_cv <= 0.08
    ):
        return "high"
    if (
        window_size >= min_window
        and input_similarity >= input_trigger
        and output_cv <= output_cv_trigger
    ):
        return "medium"
    return "low"


def _context_snowball_confidence(stats: SessionStats) -> str:
    summary = stats.summary()
    growth = float(summary.get("recent_input_growth_ratio") or 0.0)
    increase_fraction = float(summary.get("recent_input_increase_fraction") or 0.0)
    ratio = float(summary.get("average_input_output_ratio") or 0.0)
    growth_trigger = _threshold(
        stats,
        "CTX-001",
        "growth_trigger",
        CONTEXT_SNOWBALL_GROWTH_TRIGGER,
    )
    increase_trigger = _threshold(
        stats,
        "CTX-001",
        "increase_trigger",
        CONTEXT_SNOWBALL_INCREASE_TRIGGER,
    )
    ratio_trigger = _threshold(
        stats,
        "CTX-001",
        "ratio_trigger",
        CONTEXT_SNOWBALL_RATIO_TRIGGER,
    )
    if growth >= 8.0 and increase_fraction >= 0.9 and ratio >= 6.0:
        return "high"
    if (
        growth >= growth_trigger
        and increase_fraction >= increase_trigger
        and ratio >= ratio_trigger
    ):
        return "medium"
    return "low"


def _empty_visible_confidence(stats: SessionStats) -> str:
    summary = stats.summary()
    empty_fraction = float(summary.get("recent_empty_visible_output_fraction") or 0.0)
    empty_count = int(summary.get("recent_empty_visible_output_count") or 0)
    avg_output_tokens = float(summary.get("recent_avg_output_tokens") or 0.0)
    cap_hit_fraction = float(summary.get("recent_output_cap_hit_fraction") or 0.0)
    cap_hit_count = int(summary.get("recent_output_cap_hit_count") or 0)
    fraction_trigger = _threshold(
        stats,
        "EMPTY-001",
        "fraction_trigger",
        EMPTY_VISIBLE_FRACTION_TRIGGER,
    )
    if (
        empty_count >= 5
        and empty_fraction >= 0.8
        and cap_hit_count >= 5
        and cap_hit_fraction >= 0.8
    ):
        return "high"
    if empty_count >= 10 and empty_fraction >= 0.8 and avg_output_tokens >= 100:
        return "high"
    if empty_count >= 4 and empty_fraction >= fraction_trigger:
        return "medium"
    return "low"


def _cache2_confidence(stats: SessionStats) -> str:
    summary = stats.summary()
    miss_fraction = float(summary.get("recent_cache_miss_fraction") or 0.0)
    window_size = int(summary.get("recent_window_size") or 0)
    if miss_fraction >= 0.95 and window_size >= CACHE2_MIN_CALLS:
        return "high"
    if miss_fraction >= CACHE2_MISS_FRACTION_TRIGGER and window_size >= CACHE2_MIN_CALLS:
        return "medium"
    return "low"


def _stream1_confidence(stats: SessionStats) -> str:
    summary = stats.summary()
    stream_count = int(summary.get("recent_stream_count") or 0)
    incomplete_count = int(summary.get("recent_stream_incomplete_count") or 0)
    if stream_count < STREAM1_MIN_WINDOW:
        return "low"
    incomplete_fraction = incomplete_count / stream_count if stream_count else 0.0
    if incomplete_fraction >= 0.6:
        return "high"
    if incomplete_fraction >= STREAM1_INCOMPLETE_FRACTION_TRIGGER:
        return "medium"
    return "low"


def _reasoning1_confidence(stats: SessionStats) -> str:
    summary = stats.summary()
    reasoning_model_count = int(summary.get("recent_reasoning_model_count") or 0)
    missing_count = int(summary.get("recent_reasoning_missing_count") or 0)
    if reasoning_model_count < REASONING1_MIN_CALLS:
        return "low"
    missing_fraction = missing_count / reasoning_model_count if reasoning_model_count else 0.0
    if missing_fraction >= 0.8:
        return "high"
    if missing_fraction >= REASONING1_MISSING_FRACTION_TRIGGER:
        return "medium"
    return "low"


def _error_confidence(stats: SessionStats) -> str:
    summary = stats.summary()
    consecutive = int(summary.get("recent_consecutive_errors") or 0)
    fraction = float(summary.get("recent_error_fraction") or 0.0)
    window_size = int(summary.get("recent_window_size") or 0)
    if consecutive >= 5 or (fraction >= 0.6 and window_size >= ERROR_MIN_WINDOW):
        return "high"
    if consecutive >= ERROR_CONSECUTIVE_TRIGGER or (
        fraction >= ERROR_FRACTION_TRIGGER and window_size >= ERROR_MIN_WINDOW
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
    "ZOMBIE-001": AdvisorySpec(
        recommended_action=(
            "Inspect whether the workflow already reached a terminal state. If so, "
            "add an explicit completion guard, tool result sentinel, or max-step stop."
        ),
        automation_guidance=(
            "Advisory only until task-specific evals prove that repeated normal-length "
            "responses are wasteful for this workflow."
        ),
        evidence=_summary_evidence(
            "total_requests",
            "recent_window_size",
            "recent_input_similarity",
            "recent_avg_output_tokens",
            "recent_output_token_cv",
        ),
        confidence=_zombie_confidence,
    ),
    "CTX-001": AdvisorySpec(
        recommended_action=(
            "Inspect the agent loop for unbounded conversation history growth. "
            "Trim prior failed turns, summarize tool history, or stop after repeated "
            "tool-contract failures."
        ),
        automation_guidance=(
            "Warn-only. Automatic truncation or stop conditions need workflow-specific "
            "quality checks."
        ),
        evidence=_summary_evidence(
            "total_requests",
            "average_input_output_ratio",
            "recent_window_size",
            "recent_input_growth_ratio",
            "recent_input_increase_fraction",
            "recent_input_growth_tokens",
        ),
        confidence=_context_snowball_confidence,
    ),
    "EMPTY-001": AdvisorySpec(
        recommended_action=(
            "Check whether the provider/model is producing hidden reasoning, malformed "
            "content, or hitting the output cap before a visible answer. Consider a "
            "shorter cap, a non-reasoning model, or a response-format guard."
        ),
        automation_guidance=(
            "Advisory only. Empty visible output can be legitimate for some reasoning "
            "or tool-call workflows."
        ),
        evidence=_summary_evidence(
            "total_requests",
            "recent_window_size",
            "recent_avg_output_tokens",
            "recent_empty_visible_output_count",
            "recent_empty_visible_output_fraction",
            "recent_visible_output_count_window",
            "recent_output_cap_hit_count",
            "recent_output_cap_hit_fraction",
            "recent_output_cap_count_window",
        ),
        confidence=_empty_visible_confidence,
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
    "TRUNC-001": AdvisorySpec(
        recommended_action=(
            "Increase max_tokens, add a stop sequence to encourage earlier completion, "
            "or use a response format that keeps answers within the current cap."
        ),
        automation_guidance=(
            "Advisory only. Some workflows intentionally cap output length; "
            "confirm truncation is unintended before adjusting."
        ),
        evidence=_summary_evidence(
            "total_requests",
            "recent_window_size",
            "recent_max_tokens_finish_count",
            "recent_max_tokens_finish_fraction",
        ),
        confidence=_trunc_confidence,
    ),
    "ERROR-001": AdvisorySpec(
        recommended_action=(
            "Check provider status, API key validity, and prompt safety filters. "
            "Add structured error handling and exponential back-off. "
            "If errors are intermittent, consider a fallback model or provider."
        ),
        automation_guidance=(
            "Warn-only. Do not automatically kill on error rate alone — "
            "some errors are transient. Combine with retry count and latency signals."
        ),
        evidence=_summary_evidence(
            "total_requests",
            "recent_window_size",
            "recent_error_count",
            "recent_error_fraction",
            "recent_consecutive_errors",
        ),
        confidence=_error_confidence,
    ),
    "CACHE-002": AdvisorySpec(
        recommended_action=(
            "Enable Prompt Caching (Anthropic/OpenAI/DeepSeek) and verify that your "
            "system prompt is structured to be cache-eligible. Check that the SDK or "
            "provider is not prepending dynamic content before the static prefix."
        ),
        automation_guidance=(
            "Advisory only. Caching requires API-level opt-in; do not auto-enable "
            "without verifying cost expectations and latency trade-offs."
        ),
        evidence=_summary_evidence(
            "total_requests",
            "recent_window_size",
            "recent_cache_miss_count",
            "recent_cache_miss_fraction",
        ),
        confidence=_cache2_confidence,
    ),
    "STREAM-001": AdvisorySpec(
        recommended_action=(
            "Check for client-side disconnects, timeouts shorter than generation time, "
            "or upstream proxy cutoffs. Incomplete streams are billable but yield no "
            "useful output. Consider raising the client timeout or using non-streaming "
            "for short outputs."
        ),
        automation_guidance=(
            "Warn-only. Incomplete streams may be legitimate user interrupts. "
            "Only automate after confirming they are not expected behavior."
        ),
        evidence=_summary_evidence(
            "total_requests",
            "recent_stream_count",
            "recent_stream_incomplete_count",
        ),
        confidence=_stream1_confidence,
    ),
    "REASONING-001": AdvisorySpec(
        recommended_action=(
            "Verify that extended thinking / reasoning is enabled in the request "
            "(e.g., `thinking: {type: 'enabled', budget_tokens: N}` for Anthropic, "
            "or that the model variant actually performs chain-of-thought reasoning). "
            "If reasoning is intentionally disabled, switch to a non-reasoning model "
            "to avoid paying the reasoning-model premium."
        ),
        automation_guidance=(
            "Advisory only. Some workflows intentionally call reasoning models "
            "without extended thinking. Confirm intent before acting."
        ),
        evidence=_summary_evidence(
            "total_requests",
            "recent_reasoning_model_count",
            "recent_reasoning_missing_count",
        ),
        confidence=_reasoning1_confidence,
    ),
    "TOOL-DEAD-001": AdvisorySpec(
        recommended_action=(
            "Remove unused tools from the agent manifest or route them behind "
            "conditional registration so dead schemas are not re-transmitted every turn."
        ),
        automation_guidance=(
            "Report-only. Tool lists are application-specific; confirm tools are "
            "genuinely unused before removing them."
        ),
        evidence=_summary_evidence(
            "total_requests",
            "function_tools_never_called",
            "wasted_tool_schema_tokens",
            "wasted_tool_schema_cost_per_request_usd",
            "wasted_tool_schema_session_cost_usd",
            "dead_tool_offer_request_count",
        ),
        confidence=lambda stats: "medium",
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
        if repetition_rate > _threshold(stats, "CACHE-001", "repetition_rate", 0.5):
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
    if ratio and ratio > _threshold(stats, "RAG-001", "ratio_trigger", 50):
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
    # Fires when ≥80 % of the recent window produces below the low-output
    # threshold AND the inputs are repetitive. Override low_output_threshold
    # for classification pipelines that legitimately return short responses.
    recent_low = summary.get("recent_low_output_count", 0)
    low_fraction = summary.get("recent_low_output_fraction", 0.0)
    window_size = summary.get("recent_window_size", 0)
    input_similarity = summary.get("recent_input_similarity", 0.0)
    stalled_cost = summary.get("recent_stalled_cost_usd", 0.0)
    stall_low_threshold = int(
        summary.get("recent_low_output_threshold")
        or _threshold(
            stats,
            "STALL-001",
            "low_output_threshold",
            STALL_LOW_OUTPUT_THRESHOLD,
        )
    )

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
                f"{recent_low} of last {window_size} calls produced "
                f"<{stall_low_threshold} output tokens "
                f"({low_fraction:.0%} of window) with {input_similarity:.0%} input "
                "similarity. This suggests a stalled agentic loop - the model may be "
                "stuck on repeated tool failures or empty completions. "
                f"Estimated wasted cost: ${stalled_cost:.2f}. "
                "Security signal: repetitive low-output loops can be consistent with "
                "prompt-injection-driven agent loops or unbounded consumption."
            ),
            potential_savings_usd=stalled_cost,
            request_count=recent_low,
            security_signal=True,
            security_refs=("OWASP-LLM01", "OWASP-LLM10"),
        ))

    # 4. Zombie / Post-completion Drift
    # Distinct from STALL-001: output is not low. The model is still producing
    # normal-length text, but a stable repeated pattern suggests the task may
    # have completed while inference continued.
    # Limitation: uses input token-count identity as the similarity proxy, so it
    # only fires on strict replay loops (same prompt length). Accumulating-context
    # drift (growing inputs) will not satisfy input_similarity and should be
    # caught by CTX-001 instead. Tool-call-rate tracking is the planned signal
    # for post-completion loops with varying context.
    avg_output_tokens = float(summary.get("recent_avg_output_tokens") or 0.0)
    output_cv = float(summary.get("recent_output_token_cv") or 0.0)
    zombie_min_window = int(
        _threshold(stats, "ZOMBIE-001", "min_window", ZOMBIE_MIN_WINDOW)
    )
    zombie_input_sim = _threshold(
        stats,
        "ZOMBIE-001",
        "input_similarity",
        ZOMBIE_INPUT_SIMILARITY_TRIGGER,
    )
    zombie_output_cv = _threshold(
        stats,
        "ZOMBIE-001",
        "output_cv",
        ZOMBIE_OUTPUT_CV_TRIGGER,
    )
    if (
        window_size >= zombie_min_window
        and input_similarity >= zombie_input_sim
        and avg_output_tokens > stall_low_threshold
        and output_cv <= zombie_output_cv
    ):
        advisories.append(Advisory(
            code="ZOMBIE-001",
            severity="WARNING",
            title="Post-completion drift detected",
            description=(
                f"Last {window_size} calls used highly similar inputs "
                f"({input_similarity:.0%}) and generated normal-length outputs "
                f"averaging {avg_output_tokens:,.0f} tokens with low length "
                f"variation (CV {output_cv:.2f}). The model is replaying a nearly "
                "identical prompt and producing consistent-length responses - a "
                "strict replay loop. Note: this advisory uses input token-count "
                "identity as its similarity signal; it does not fire on "
                "accumulating-context drift. Check CTX-001 if input tokens are growing. "
                "Security signal: continued inference after likely completion can "
                "support review for excessive agency and unbounded consumption."
            ),
            request_count=int(window_size),
            security_signal=True,
            security_refs=("OWASP-LLM06", "OWASP-LLM10"),
        ))

    # 5. Context Snowball / Prefill-heavy Drift
    # The prompt grows every turn while the workflow still fails to terminate.
    # This is common after tool failures: each apology/error gets fed back in,
    # making the next request more expensive even if the task is not progressing.
    input_growth_ratio = float(summary.get("recent_input_growth_ratio") or 0.0)
    input_increase_fraction = float(
        summary.get("recent_input_increase_fraction") or 0.0
    )
    input_growth_tokens = float(summary.get("recent_input_growth_tokens") or 0.0)
    ctx_min_window = int(
        _threshold(stats, "CTX-001", "min_window", CONTEXT_SNOWBALL_MIN_WINDOW)
    )
    ctx_growth = _threshold(
        stats,
        "CTX-001",
        "growth_trigger",
        CONTEXT_SNOWBALL_GROWTH_TRIGGER,
    )
    ctx_increase = _threshold(
        stats,
        "CTX-001",
        "increase_trigger",
        CONTEXT_SNOWBALL_INCREASE_TRIGGER,
    )
    ctx_ratio = _threshold(
        stats,
        "CTX-001",
        "ratio_trigger",
        CONTEXT_SNOWBALL_RATIO_TRIGGER,
    )
    if (
        window_size >= ctx_min_window
        and input_growth_ratio >= ctx_growth
        and input_increase_fraction >= ctx_increase
        and ratio
        and ratio >= ctx_ratio
    ):
        advisories.append(Advisory(
            code="CTX-001",
            severity="WARNING",
            title="Context snowball detected",
            description=(
                f"Recent input tokens grew by {input_growth_ratio:.1f}x "
                f"(~{input_growth_tokens:,.0f} tokens) while the session kept "
                f"an average input:output ratio of {ratio:.1f}:1. This suggests "
                "conversation history or failed tool context is accumulating without "
                "proportional useful output. Security signal: context growth after "
                "tool or retrieval failures can support review for prompt injection "
                "or unbounded consumption."
            ),
            request_count=int(window_size),
            security_signal=True,
            security_refs=("OWASP-LLM01", "OWASP-LLM10"),
        ))

    # 6. Invisible / Empty Visible Output Burn
    # Privacy-safe diagnostic: Vetch stores character counts only. This catches
    # calls that consume output tokens while returning little or no visible text.
    empty_visible_count = int(summary.get("recent_empty_visible_output_count") or 0)
    empty_visible_fraction = float(
        summary.get("recent_empty_visible_output_fraction") or 0.0
    )
    visible_count_window = int(summary.get("recent_visible_output_count_window") or 0)
    output_cap_hit_count = int(summary.get("recent_output_cap_hit_count") or 0)
    empty_min_window = int(
        _threshold(stats, "EMPTY-001", "min_window", EMPTY_VISIBLE_MIN_WINDOW)
    )
    empty_fraction = _threshold(
        stats,
        "EMPTY-001",
        "fraction_trigger",
        EMPTY_VISIBLE_FRACTION_TRIGGER,
    )
    if (
        visible_count_window >= empty_min_window
        and empty_visible_count >= 4
        and empty_visible_fraction >= empty_fraction
        and avg_output_tokens >= EMPTY_VISIBLE_OUTPUT_TOKEN_THRESHOLD
    ):
        cap_sentence = (
            f" {output_cap_hit_count} of those calls appear to have hit the requested "
            "output cap."
            if output_cap_hit_count
            else ""
        )
        advisories.append(Advisory(
            code="EMPTY-001",
            severity="WARNING",
            title="Invisible output burn detected",
            description=(
                f"{empty_visible_count} of the last {visible_count_window} calls "
                "consumed output tokens while returning almost no visible text."
                f"{cap_sentence} This can happen with hidden reasoning, malformed "
                "responses, or output caps that are too short for the model to "
                "produce a final answer. "
                "Security signal: repeated empty visible output can support review "
                "for improper output handling or unbounded consumption. Vetch does "
                "not inspect content and does not infer data disclosure."
            ),
            request_count=empty_visible_count,
            security_signal=True,
            security_refs=("OWASP-LLM05", "OWASP-LLM10"),
        ))

    # 7. Repeated Truncation (finish_reason="max_tokens")
    # Fires when the provider reports max_tokens on a large fraction of recent
    # calls, suggesting responses are being cut off. Distinct from EMPTY-001
    # (which fires on empty visible output) — TRUNC-001 fires even when the
    # model is producing content that is structurally or semantically incomplete.
    max_tokens_finish_count = int(summary.get("recent_max_tokens_finish_count") or 0)
    max_tokens_finish_fraction = float(
        summary.get("recent_max_tokens_finish_fraction") or 0.0
    )
    trunc_min_window = int(_threshold(stats, "TRUNC-001", "min_window", TRUNC_MIN_WINDOW))
    trunc_min_count = int(_threshold(stats, "TRUNC-001", "min_count", TRUNC_MIN_COUNT))
    trunc_fraction = _threshold(
        stats,
        "TRUNC-001",
        "fraction_trigger",
        TRUNC_FRACTION_TRIGGER,
    )
    if (
        window_size >= trunc_min_window
        and max_tokens_finish_count >= trunc_min_count
        and max_tokens_finish_fraction >= trunc_fraction
    ):
        advisories.append(Advisory(
            code="TRUNC-001",
            severity="WARNING",
            title="Repeated response truncation detected",
            description=(
                f"{max_tokens_finish_count} of the last {window_size} calls ended "
                f"with finish_reason=max_tokens ({max_tokens_finish_fraction:.0%} of "
                "window). Responses are being cut off before completion. This can "
                "cause structurally incomplete tool calls, JSON, or answers. "
                "Security signal: systematic truncation can support review for "
                "improper output handling or unbounded consumption, including "
                "prompt-injection padding that exhausts context."
            ),
            request_count=max_tokens_finish_count,
            security_signal=True,
            security_refs=("OWASP-LLM01", "OWASP-LLM05", "OWASP-LLM10"),
        ))

    # 8. Babbling / Excessive Generation Proxy
    # Metadata-only first pass: unusually high recent average output length.
    # This does not inspect output content, so it is intentionally warn-only.
    babble_trigger = _threshold(
        stats,
        "BABBLE-001",
        "min_avg_output_tokens",
        BABBLE_AVG_OUTPUT_TOKENS_TRIGGER,
    )
    if (
        stats.total_requests >= BABBLE_MIN_REQUESTS
        and avg_output_tokens >= babble_trigger
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

    # 9. Error Storm
    # Fires when recent calls show a high error rate or consecutive errors,
    # indicating a broken prompt, exhausted quota, or provider outage.
    recent_error_count = int(summary.get("recent_error_count") or 0)
    recent_error_fraction = float(summary.get("recent_error_fraction") or 0.0)
    consecutive_errors = int(summary.get("recent_consecutive_errors") or 0)
    error_fraction_trigger = _threshold(
        stats, "ERROR-001", "fraction_trigger", ERROR_FRACTION_TRIGGER
    )
    error_consecutive_trigger = int(
        _threshold(stats, "ERROR-001", "consecutive_trigger", ERROR_CONSECUTIVE_TRIGGER)
    )
    if window_size >= ERROR_MIN_WINDOW and (
        consecutive_errors >= error_consecutive_trigger
        or (recent_error_count > 0 and recent_error_fraction >= error_fraction_trigger)
    ):
        severity = "CRITICAL" if consecutive_errors >= 5 else "WARNING"
        if consecutive_errors >= error_consecutive_trigger:
            desc_trigger = (
                f"Last {consecutive_errors} calls in a row returned errors."
            )
        else:
            desc_trigger = (
                f"{recent_error_count} of the last {window_size} calls returned errors "
                f"({recent_error_fraction:.0%} of window)."
            )
        advisories.append(Advisory(
            code="ERROR-001",
            severity=severity,
            title="High inference error rate detected",
            description=(
                f"{desc_trigger} "
                "Repeated errors waste spend on failed calls and can mask stall or "
                "zombie patterns. Check provider status, API key validity, content "
                "filters, and request format. "
                "Security signal: systematic errors can support review for "
                "prompt-injection attempts that trigger safety filters or "
                "malformed payloads designed to exhaust quota."
            ),
            request_count=recent_error_count,
            security_signal=True,
            security_refs=("OWASP-LLM04", "OWASP-LLM10"),
        ))

    # 10. Cache opportunity ignored (CACHE-002)
    # Fires when input-token repetition is high (same as CACHE-001 trigger) AND
    # most recent calls have no cache_read_tokens — meaning caching is available
    # but not being used. CACHE-001 fires on the opportunity; CACHE-002 fires when
    # the opportunity is clearly being missed in production.
    if stats.total_requests >= CACHE2_MIN_CALLS:
        # Reuse repetition_rate from section 1 above
        repeated_inputs2 = sum(c for c in stats.input_token_counts.values() if c > 1)
        repetition_rate2 = repeated_inputs2 / stats.total_requests
        cache_miss_count = int(summary.get("recent_cache_miss_count") or 0)
        cache_miss_fraction = float(summary.get("recent_cache_miss_fraction") or 0.0)
        cache2_rep_trigger = _threshold(
            stats, "CACHE-002", "repetition_rate", CACHE2_REPETITION_TRIGGER
        )
        cache2_miss_trigger = _threshold(
            stats, "CACHE-002", "miss_fraction", CACHE2_MISS_FRACTION_TRIGGER
        )
        if (
            window_size >= CACHE2_MIN_CALLS
            and repetition_rate2 >= cache2_rep_trigger
            and cache_miss_fraction >= cache2_miss_trigger
        ):
            advisories.append(Advisory(
                code="CACHE-002",
                severity="WARNING",
                title="Prompt caching opportunity not used",
                description=(
                    f"{repetition_rate2:.0%} of requests share identical input token counts, "
                    f"but {cache_miss_fraction:.0%} of recent calls had no cache reads. "
                    "The repetition pattern suggests a static system prompt that could be "
                    "cached, but caching does not appear to be active. "
                    "Enable Prompt Caching to reduce input costs by up to 90%."
                ),
                request_count=cache_miss_count,
            ))

    # 11. Incomplete stream burn (STREAM-001)
    # Fires when a significant fraction of streaming calls do not complete.
    stream_count = int(summary.get("recent_stream_count") or 0)
    stream_incomplete_count = int(summary.get("recent_stream_incomplete_count") or 0)
    stream1_min_window = int(
        _threshold(stats, "STREAM-001", "min_window", STREAM1_MIN_WINDOW)
    )
    stream1_fraction_trigger = _threshold(
        stats, "STREAM-001", "fraction_trigger", STREAM1_INCOMPLETE_FRACTION_TRIGGER
    )
    if stream_count >= stream1_min_window and stream_incomplete_count > 0:
        incomplete_fraction = stream_incomplete_count / stream_count
        if incomplete_fraction >= stream1_fraction_trigger:
            advisories.append(Advisory(
                code="STREAM-001",
                severity="WARNING",
                title="High rate of incomplete streams",
                description=(
                    f"{stream_incomplete_count} of {stream_count} recent streaming calls "
                    f"did not complete ({incomplete_fraction:.0%}). Incomplete streams "
                    "are billed for tokens generated so far, wasting spend with no "
                    "usable output. Check for client timeouts, dropped connections, "
                    "or upstream proxy cutoffs shorter than generation time."
                ),
                request_count=stream_incomplete_count,
            ))

    # 12. Reasoning model called without reasoning tokens (REASONING-001)
    # Fires when calls go to reasoning-capable models but usage.reasoning is absent.
    # Indicates either reasoning is disabled or the model is being used at full
    # premium cost without the test-time compute that justifies it.
    reasoning_model_count = int(summary.get("recent_reasoning_model_count") or 0)
    reasoning_missing_count = int(summary.get("recent_reasoning_missing_count") or 0)
    reasoning1_min_calls = int(
        _threshold(stats, "REASONING-001", "min_calls", REASONING1_MIN_CALLS)
    )
    reasoning1_fraction = _threshold(
        stats, "REASONING-001", "missing_fraction", REASONING1_MISSING_FRACTION_TRIGGER
    )
    if (
        reasoning_model_count >= reasoning1_min_calls
        and reasoning_missing_count > 0
        and reasoning_missing_count / reasoning_model_count >= reasoning1_fraction
    ):
        advisories.append(Advisory(
            code="REASONING-001",
            severity="INFO",
            title="Reasoning model without reasoning tokens",
            description=(
                f"{reasoning_missing_count} of {reasoning_model_count} recent calls to "
                "a reasoning-capable model returned no reasoning tokens. You may be "
                "paying the reasoning-model premium without using extended thinking. "
                "Either enable reasoning (set budget_tokens > 0) or switch to a "
                "non-reasoning variant to reduce cost."
            ),
            request_count=reasoning_missing_count,
        ))

    # 15. Dead function tools (offered but never invoked)
    never_called = summary.get("function_tools_never_called") or []
    wasted_cost = float(summary.get("wasted_tool_schema_session_cost_usd") or 0.0)
    wasted_per_request = float(summary.get("wasted_tool_schema_cost_per_request_usd") or 0.0)
    retransmit_count = int(summary.get("dead_tool_offer_request_count") or 0)
    min_requests = int(_threshold(stats, "TOOL-DEAD-001", "min_requests", 10))
    min_offered = int(_threshold(stats, "TOOL-DEAD-001", "min_offered_tools", 1))
    if (
        retransmit_count >= min_requests
        and len(never_called) >= min_offered
        and wasted_cost > 0
    ):
        advisories.append(Advisory(
            code="TOOL-DEAD-001",
            severity="INFO",
            title="Dead Function Tools Detected",
            description=(
                f"{len(never_called)} function tool(s) were offered but never invoked "
                f"this session ({', '.join(never_called[:5])}"
                f"{'...' if len(never_called) > 5 else ''}). "
                f"Estimated cache-aware session schema cost: ${wasted_cost:.4f} "
                f"(${wasted_per_request:.4f}/request × {retransmit_count} requests; "
                "directional estimate)."
            ),
            potential_savings_usd=wasted_cost,
            request_count=retransmit_count,
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
        sec = " 🔒" if adv.security_signal else ""
        lines.append(f"{icon} [{adv.code}]{sec} {adv.title}")
        lines.append(f"   {adv.description}")
        if adv.security_refs:
            lines.append(f"   Security refs: {', '.join(adv.security_refs)}")
        if adv.potential_savings_usd:
            lines.append(f"   Potential Savings: ${adv.potential_savings_usd:.2f}")
        lines.append("")

    return "\n".join(lines)
