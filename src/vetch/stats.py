"""Session statistics for real-time analysis.

Tracks aggregated metrics for the current process to support
advisories and summaries without querying the database.

SessionStats is a self-contained dataclass: every instance carries its
own rolling windows, token counts, and cost accumulators.  The global
``_session_stats`` singleton is a convenience for single-process CLI
usage.  For multi-user / web contexts, each ``vetch.Session`` now
carries its own ``SessionStats`` instance (see ``session.py``).
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, NamedTuple

# Rolling window size for recent output token tracking (stall detection)
_RECENT_WINDOW = 20

# Stall detection: fire when this fraction of the window is below the
# output-token threshold.  80 % means 16 of the last 20 calls.
STALL_LOW_OUTPUT_THRESHOLD = 5       # tokens
STALL_FRACTION_TRIGGER = 0.80        # 80 % of window


class _RecentCall(NamedTuple):
    """Single entry in the stall-detection rolling window."""

    output_tokens: int
    input_tokens: int
    cost_usd: float


@dataclass
class SessionStats:
    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_energy_wh: float = 0.0
    total_carbon_g: float = 0.0
    total_water_ml: float = 0.0
    total_cost_usd: float = 0.0

    # For pattern detection
    input_token_counts: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    models_used: set[str] = field(default_factory=set)

    # Rolling window for stall detection (bounded, memory-safe).
    # Each entry stores (output_tokens, input_tokens, cost_usd) so we
    # can compute per-call wasted cost and detect input similarity.
    recent_calls: deque[_RecentCall] = field(
        default_factory=lambda: deque(maxlen=_RECENT_WINDOW)
    )

    # Backward-compat shim: ``recent_output_tokens`` is derived from
    # ``recent_calls`` so existing callers keep working.
    @property
    def recent_output_tokens(self) -> list[int]:
        return [c.output_tokens for c in self.recent_calls]

    def update(self, event: Mapping[str, Any]) -> None:
        self.total_requests += 1

        usage = event.get("usage", {}) or {}
        text = usage.get("text", {}) or {}
        in_tok = text.get("input_tokens", 0)
        out_tok = text.get("output_tokens", 0)
        call_cost = event.get("estimated_cost_usd") or 0.0

        self.total_input_tokens += in_tok
        self.total_output_tokens += out_tok
        self.total_energy_wh += (event.get("estimated_energy_wh") or 0.0)
        self.total_carbon_g += (event.get("estimated_carbon_g") or 0.0)
        self.total_cost_usd += call_cost

        # Water: wrapper emits liters, MCP tools emit ml — accept both.
        water_ml = event.get("estimated_water_ml") or 0.0
        water_l = event.get("estimated_water_l") or 0.0
        self.total_water_ml += water_ml + water_l * 1000

        # Track for advisory
        if in_tok > 0:
            self.input_token_counts[in_tok] += 1

        # Rolling window for stall detection
        self.recent_calls.append(_RecentCall(out_tok, in_tok, call_cost))

        model = event.get("model")
        if model:
            self.models_used.add(model)

    def summary(self) -> dict[str, Any]:
        # If output is 0 but input exists, ratio is effectively infinite (input)
        if self.total_output_tokens == 0:
            ratio = float(self.total_input_tokens) if self.total_input_tokens > 0 else 0.0
        else:
            ratio = self.total_input_tokens / self.total_output_tokens

        # Stall detection metrics
        recent = list(self.recent_calls)
        window_size = len(recent)
        out_tokens = [c.output_tokens for c in recent]
        recent_avg = sum(out_tokens) / window_size if window_size else 0.0
        recent_low_count = sum(
            1 for c in recent if c.output_tokens < STALL_LOW_OUTPUT_THRESHOLD
        )
        recent_low_fraction = recent_low_count / window_size if window_size else 0.0

        # Wasted cost: sum of per-call costs for low-output calls only
        stalled_cost = sum(
            c.cost_usd for c in recent
            if c.output_tokens < STALL_LOW_OUTPUT_THRESHOLD
        )

        # Input similarity: fraction of recent calls sharing the most
        # common input token count (suggests stuck on same prompt).
        input_counts: dict[int, int] = defaultdict(int)
        for c in recent:
            if c.input_tokens > 0:
                input_counts[c.input_tokens] += 1
        max_same_input = max(input_counts.values()) if input_counts else 0
        input_similarity = max_same_input / window_size if window_size else 0.0

        return {
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_water_ml": self.total_water_ml,
            "average_input_output_ratio": ratio,
            "recent_avg_output_tokens": recent_avg,
            "recent_low_output_count": recent_low_count,
            "recent_low_output_fraction": recent_low_fraction,
            "recent_window_size": window_size,
            "recent_stalled_cost_usd": round(stalled_cost, 4),
            "recent_input_similarity": round(input_similarity, 4),
        }


# Global singleton — appropriate for single-process CLI / SDK usage.
# For multi-user contexts, use ``vetch.Session`` which carries its own
# ``SessionStats`` instance.
_session_stats = SessionStats()


def get_session_stats() -> SessionStats:
    return _session_stats


def track_session_event(event: dict[str, Any]) -> None:
    """Update global session stats."""
    _session_stats.update(event)
