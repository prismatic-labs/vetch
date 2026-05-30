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

import threading
from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from math import sqrt
from typing import Any, NamedTuple

# Rolling window size for recent output token tracking (stall detection)
_RECENT_WINDOW = 20

# Stall detection: fire when this fraction of the window is below the
# output-token threshold.  80 % means 16 of the last 20 calls.
STALL_LOW_OUTPUT_THRESHOLD = 5       # tokens
STALL_FRACTION_TRIGGER = 0.80        # 80 % of window
EMPTY_VISIBLE_OUTPUT_TOKEN_THRESHOLD = 20


class _RecentCall(NamedTuple):
    """Single entry in the stall-detection rolling window."""

    output_tokens: int
    input_tokens: int
    cost_usd: float
    visible_output_chars: int | None = None
    finish_reason: str | None = None
    requested_max_tokens: int | None = None
    error: bool = False
    cache_read_tokens: int = 0
    is_stream: bool = False
    complete: bool = True
    is_reasoning_model: bool = False
    has_reasoning_tokens: bool = False


def _count_trailing_errors(calls: Sequence[_RecentCall]) -> int:
    """Return the number of consecutive error calls at the tail of the window."""
    count = 0
    for call in reversed(calls):
        if call.error:
            count += 1
        else:
            break
    return count


@dataclass
class SessionStats:
    advisory_thresholds: dict[str, dict[str, float]] | None = None
    fire_advisory_hooks: bool = False
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

    def __post_init__(self) -> None:
        # Not a dataclass field — excluded from __init__, __eq__, repr.
        if self.advisory_thresholds is not None:
            self.advisory_thresholds = {
                code: dict(overrides)
                for code, overrides in self.advisory_thresholds.items()
            }
        object.__setattr__(self, "_lock", threading.Lock())
        object.__setattr__(self, "_summary_cache", None)
        object.__setattr__(self, "_summary_dirty", True)

    def advisory_threshold(self, code: str, key: str, default: float) -> float:
        """Return a threshold override scoped to this stats object if present."""
        if self.advisory_thresholds:
            scoped = self.advisory_thresholds.get(code, {})
            if key in scoped:
                return float(scoped[key])
        from vetch.config import get_advisory_threshold

        return get_advisory_threshold(code, key, default)

    # Backward-compat shim: ``recent_output_tokens`` is derived from
    # ``recent_calls`` so existing callers keep working.
    @property
    def recent_output_tokens(self) -> list[int]:
        return [c.output_tokens for c in self.recent_calls]

    def update(self, event: Mapping[str, Any]) -> None:
        with self._lock:  # type: ignore[attr-defined]
            self._update_locked(event)
            object.__setattr__(self, "_summary_dirty", True)
        if self.fire_advisory_hooks:
            _fire_advisory_hooks(self)

    def _update_locked(self, event: Mapping[str, Any]) -> None:
        self.total_requests += 1

        usage = event.get("usage", {}) or {}
        text = usage.get("text", {}) or {}
        in_tok = text.get("input_tokens", 0)
        out_tok = text.get("output_tokens", 0)
        call_cost = event.get("estimated_cost_usd") or 0.0
        visible_chars = event.get("visible_output_chars")
        if visible_chars is None:
            visible_chars = event.get("accumulated_chars")
        if not isinstance(visible_chars, int):
            visible_chars = None
        raw_finish_reason = event.get("finish_reason")
        finish_reason = raw_finish_reason if isinstance(raw_finish_reason, str) else None
        raw_requested_max_tokens = event.get("requested_max_tokens")
        requested_max_tokens = (
            raw_requested_max_tokens
            if isinstance(raw_requested_max_tokens, int)
            and raw_requested_max_tokens > 0
            else None
        )
        is_error = bool(event.get("error"))
        raw_crt = event.get("cache_read_tokens")
        cache_read_tokens = int(raw_crt) if isinstance(raw_crt, int) else 0
        is_stream = bool(event.get("is_stream"))
        complete = bool(event.get("complete", True))

        # Reasoning model detection — lazy import to avoid circular dependency.
        model_name = str(event.get("model") or "")
        is_reasoning_model = False
        has_reasoning_tokens = False
        if model_name:
            try:
                from vetch.calculation import _is_reasoning_compute_model
                is_reasoning_model = _is_reasoning_compute_model(model_name)
            except Exception:
                pass
        reasoning_usage = (usage.get("reasoning") or {}) if isinstance(usage, dict) else {}
        if isinstance(reasoning_usage, dict):
            has_reasoning_tokens = int(reasoning_usage.get("output_tokens") or 0) > 0

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

        # Rolling window for stall / error / stream / cache / reasoning detection
        self.recent_calls.append(
            _RecentCall(
                out_tok,
                in_tok,
                call_cost,
                visible_chars,
                finish_reason,
                requested_max_tokens,
                is_error,
                cache_read_tokens,
                is_stream,
                complete,
                is_reasoning_model,
                has_reasoning_tokens,
            ),
        )

        model = event.get("model")
        if model:
            self.models_used.add(model)

    def summary(self) -> dict[str, Any]:
        with self._lock:  # type: ignore[attr-defined]
            if not self._summary_dirty and self._summary_cache is not None:  # type: ignore[attr-defined]
                return dict(self._summary_cache)  # type: ignore[arg-type, attr-defined]
            result = self._compute_summary()
            object.__setattr__(self, "_summary_cache", result)
            object.__setattr__(self, "_summary_dirty", False)
        return dict(result)

    def _compute_summary(self) -> dict[str, Any]:
        # If output is 0 but input exists, ratio is effectively infinite (input)
        if self.total_output_tokens == 0:
            ratio = float(self.total_input_tokens) if self.total_input_tokens > 0 else 0.0
        else:
            ratio = self.total_input_tokens / self.total_output_tokens

        # Stall detection metrics — snapshot the deque under the lock to avoid
        # RuntimeError from concurrent mutation during iteration.
        recent = list(self.recent_calls)
        window_size = len(recent)
        out_tokens = [c.output_tokens for c in recent]
        recent_avg = sum(out_tokens) / window_size if window_size else 0.0
        if window_size and recent_avg > 0:
            output_variance = sum((tokens - recent_avg) ** 2 for tokens in out_tokens)
            recent_output_cv = sqrt(output_variance / window_size) / recent_avg
        else:
            recent_output_cv = 0.0
        stall_low_output_threshold = int(
            self.advisory_threshold(
                "STALL-001",
                "low_output_threshold",
                STALL_LOW_OUTPUT_THRESHOLD,
            )
        )
        recent_low_count = sum(
            1 for c in recent if c.output_tokens <= stall_low_output_threshold
        )
        recent_low_fraction = recent_low_count / window_size if window_size else 0.0

        # Wasted cost: sum of per-call costs for low-output calls only
        stalled_cost = sum(
            c.cost_usd for c in recent
            if c.output_tokens <= stall_low_output_threshold
        )

        # Input similarity: fraction of recent calls sharing the most
        # common input token count (suggests stuck on the same prompt).
        in_tokens = [c.input_tokens for c in recent if c.input_tokens > 0]
        input_counts: dict[int, int] = defaultdict(int)
        for tokens in in_tokens:
            input_counts[tokens] += 1

        max_same_input = max(input_counts.values()) if input_counts else 0
        input_similarity = max_same_input / window_size if window_size else 0.0

        if len(in_tokens) >= 2:
            transitions = zip(in_tokens, in_tokens[1:])
            increases = sum(1 for before, after in transitions if after > before)
            input_increase_fraction = increases / (len(in_tokens) - 1)
            edge = min(3, len(in_tokens))
            first_avg = sum(in_tokens[:edge]) / edge
            last_avg = sum(in_tokens[-edge:]) / edge
            input_growth_ratio = last_avg / first_avg if first_avg > 0 else 0.0
            input_growth_tokens = last_avg - first_avg
        else:
            input_increase_fraction = 0.0
            input_growth_ratio = 0.0
            input_growth_tokens = 0.0

        calls_with_visible_counts = [
            c for c in recent if c.visible_output_chars is not None
        ]
        empty_visible_output_count = sum(
            1 for c in calls_with_visible_counts
            if (
                c.output_tokens >= EMPTY_VISIBLE_OUTPUT_TOKEN_THRESHOLD
                and (c.visible_output_chars or 0) <= 2
                and c.finish_reason not in {"tool_calls", "tool_use"}
            )
        )
        visible_count_window = len(calls_with_visible_counts)
        empty_visible_output_fraction = (
            empty_visible_output_count / visible_count_window
            if visible_count_window
            else 0.0
        )
        calls_with_requested_caps = [
            c for c in recent if c.requested_max_tokens is not None
        ]
        output_cap_hit_count = sum(
            1 for c in calls_with_requested_caps
            if c.output_tokens >= (c.requested_max_tokens or 0)
        )
        output_cap_window = len(calls_with_requested_caps)
        output_cap_hit_fraction = (
            output_cap_hit_count / output_cap_window
            if output_cap_window
            else 0.0
        )

        # Truncation via finish_reason: provider-reported max_tokens hits.
        # Complements output_cap_hit_count (which requires requested_max_tokens
        # to be set); this works even when the user didn't set an explicit cap.
        max_tokens_finish_count = sum(
            1 for c in recent if c.finish_reason in {"max_tokens", "length"}
        )
        max_tokens_finish_fraction = (
            max_tokens_finish_count / window_size if window_size else 0.0
        )

        return {
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_water_ml": self.total_water_ml,
            "average_input_output_ratio": ratio,
            "recent_avg_output_tokens": recent_avg,
            "recent_output_token_cv": round(recent_output_cv, 4),
            "recent_low_output_threshold": stall_low_output_threshold,
            "recent_low_output_count": recent_low_count,
            "recent_low_output_fraction": recent_low_fraction,
            "recent_window_size": window_size,
            "recent_stalled_cost_usd": round(stalled_cost, 4),
            "recent_input_similarity": round(input_similarity, 4),
            "recent_input_growth_ratio": round(input_growth_ratio, 4),
            "recent_input_increase_fraction": round(input_increase_fraction, 4),
            "recent_input_growth_tokens": round(input_growth_tokens, 2),
            "recent_empty_visible_output_count": empty_visible_output_count,
            "recent_empty_visible_output_fraction": round(
                empty_visible_output_fraction, 4,
            ),
            "recent_visible_output_count_window": visible_count_window,
            "recent_output_cap_hit_count": output_cap_hit_count,
            "recent_output_cap_hit_fraction": round(output_cap_hit_fraction, 4),
            "recent_output_cap_count_window": output_cap_window,
            "recent_max_tokens_finish_count": max_tokens_finish_count,
            "recent_max_tokens_finish_fraction": round(max_tokens_finish_fraction, 4),
            "recent_error_count": sum(1 for c in recent if c.error),
            "recent_error_fraction": round(
                sum(1 for c in recent if c.error) / window_size if window_size else 0.0,
                4,
            ),
            "recent_consecutive_errors": _count_trailing_errors(recent),
            # Cache miss detection: calls where no cache tokens were read
            "recent_cache_miss_count": sum(1 for c in recent if c.cache_read_tokens == 0),
            "recent_cache_miss_fraction": round(
                sum(1 for c in recent if c.cache_read_tokens == 0) / window_size
                if window_size else 0.0,
                4,
            ),
            # Incomplete stream detection
            "recent_stream_count": sum(1 for c in recent if c.is_stream),
            "recent_stream_incomplete_count": sum(
                1 for c in recent if c.is_stream and not c.complete
            ),
            # Reasoning model with no reasoning tokens
            "recent_reasoning_model_count": sum(1 for c in recent if c.is_reasoning_model),
            "recent_reasoning_missing_count": sum(
                1 for c in recent
                if c.is_reasoning_model and not c.has_reasoning_tokens and c.output_tokens > 20
            ),
        }


# Global singleton — appropriate for single-process CLI / SDK usage.
# For multi-user contexts, use ``vetch.Session`` which carries its own
# ``SessionStats`` instance.
_session_stats = SessionStats(fire_advisory_hooks=True)

# Advisory push hooks — called after every update() on the global singleton.
# Receive a non-empty list of Advisory objects (from vetch.advisory). Hooks are
# polled every _ADVISORY_HOOK_INTERVAL calls to avoid computing advisories on
# every single inference.
_advisory_hooks: list[Callable[..., None]] = []
_advisory_hooks_lock = threading.Lock()
_ADVISORY_HOOK_INTERVAL = 10


def _fire_advisory_hooks(stats: SessionStats) -> None:
    with _advisory_hooks_lock:
        hooks = list(_advisory_hooks)
    if not hooks or stats.total_requests % _ADVISORY_HOOK_INTERVAL != 0:
        return
    try:
        from vetch.advisory import generate_advisories  # lazy — avoids circular import
        advisories = generate_advisories(stats)
        if not advisories:
            return
        for hook in hooks:
            hook(advisories)
    except Exception:
        pass  # hooks must never crash the calling thread


def on_advisory(callback: Callable[..., None]) -> Callable[..., None]:
    """Register a callback invoked when advisories fire in auto-instrumented mode.

    The callback receives a list of ``Advisory`` namedtuples. It is called every
    ``_ADVISORY_HOOK_INTERVAL`` updates on the global singleton (not per-call).
    For ``Session``-scoped stats, call ``generate_advisories(session.stats)``
    directly.

    Example::

        @vetch.on_advisory
        def handle(advisories):
            for adv in advisories:
                logging.warning("[%s] %s", adv.code, adv.title)
    """
    with _advisory_hooks_lock:
        _advisory_hooks.append(callback)
    return callback  # allow use as a decorator


def _reset_advisory_hooks() -> None:
    """Clear registered advisory hooks. Test-only."""
    with _advisory_hooks_lock:
        _advisory_hooks.clear()


def get_session_stats() -> SessionStats:
    return _session_stats


def _reset_session_stats() -> None:
    """Reset the global singleton. Test-only."""
    global _session_stats
    _session_stats = SessionStats(fire_advisory_hooks=True)
    _reset_advisory_hooks()


def track_session_event(event: dict[str, Any]) -> None:
    """Update global session stats."""
    _session_stats.update(event)
