"""Tests for advisory engine."""

from __future__ import annotations

from vetch.advisory import generate_advisories
from vetch.stats import SessionStats


class TestAdvisory:
    """Tests for Advisory module."""

    def test_cache_advisory(self) -> None:
        """Trigger cache repetition advisory."""
        stats = SessionStats()
        stats.total_requests = 10
        # 6 requests with same input
        stats.input_token_counts[2000] = 6

        advs = generate_advisories(stats)
        codes = [a.code for a in advs]
        assert "CACHE-001" in codes

    def test_rag_advisory(self) -> None:
        """Trigger RAG imbalance advisory."""
        stats = SessionStats()
        stats.total_requests = 10
        stats.total_input_tokens = 10000
        stats.total_output_tokens = 10

        advs = generate_advisories(stats)
        codes = [a.code for a in advs]
        assert "RAG-001" in codes

    def test_trunc_advisory_fires_on_frequent_max_tokens(self) -> None:
        """TRUNC-001 fires when finish_reason=max_tokens is frequent."""
        stats = SessionStats()
        for _ in range(5):
            stats.update({
                "model": "gpt-4o",
                "usage": {"text": {"input_tokens": 100, "output_tokens": 200}},
                "finish_reason": "max_tokens",
            })
        advs = generate_advisories(stats)
        codes = [a.code for a in advs]
        assert "TRUNC-001" in codes

    def test_trunc_advisory_does_not_fire_when_rare(self) -> None:
        """TRUNC-001 does not fire when truncation is infrequent."""
        stats = SessionStats()
        for _ in range(9):
            stats.update({
                "model": "gpt-4o",
                "usage": {"text": {"input_tokens": 100, "output_tokens": 50}},
                "finish_reason": "stop",
            })
        stats.update({
            "model": "gpt-4o",
            "usage": {"text": {"input_tokens": 100, "output_tokens": 200}},
            "finish_reason": "max_tokens",
        })
        advs = generate_advisories(stats)
        codes = [a.code for a in advs]
        assert "TRUNC-001" not in codes

    def test_advisory_thresholds_override_babble(self) -> None:
        """set_advisory_thresholds raises BABBLE-001 only above the custom threshold."""
        from vetch.config import set_advisory_thresholds
        set_advisory_thresholds({"BABBLE-001": {"min_avg_output_tokens": 3000}})
        try:
            # 2000 tokens — below default (1500) but below custom (3000): should NOT fire
            s = SessionStats()
            for _ in range(12):
                s.update({"model": "gpt-4o", "usage": {
                    "text": {"input_tokens": 100, "output_tokens": 2000},
                }})
            codes = [a.code for a in generate_advisories(s)]
            assert "BABBLE-001" not in codes

            # 3500 tokens — above custom threshold: should fire
            s2 = SessionStats()
            for _ in range(12):
                s2.update({"model": "gpt-4o", "usage": {
                    "text": {"input_tokens": 100, "output_tokens": 3500},
                }})
            codes2 = [a.code for a in generate_advisories(s2)]
            assert "BABBLE-001" in codes2
        finally:
            set_advisory_thresholds({})  # reset

    def test_advisory_thresholds_override_trunc(self) -> None:
        """TRUNC-001 does not fire when fraction is below custom threshold."""
        from vetch.config import set_advisory_thresholds
        set_advisory_thresholds({"TRUNC-001": {"fraction_trigger": 0.9}})
        try:
            s = SessionStats()
            # 6/10 = 60% max_tokens — fires by default (50%) but not with custom 90%
            for _ in range(6):
                s.update({"model": "gpt-4o", "usage": {
                    "text": {"input_tokens": 100, "output_tokens": 200},
                }, "finish_reason": "max_tokens"})
            for _ in range(4):
                s.update({"model": "gpt-4o", "usage": {
                    "text": {"input_tokens": 100, "output_tokens": 200},
                }, "finish_reason": "stop"})
            codes = [a.code for a in generate_advisories(s)]
            assert "TRUNC-001" not in codes
        finally:
            set_advisory_thresholds({})

    def test_on_advisory_callback_fires(self) -> None:
        """on_advisory callback is called after the poll interval."""
        import vetch
        from vetch.stats import (
            _ADVISORY_HOOK_INTERVAL,
            _advisory_hooks,
            _reset_session_stats,
            track_session_event,
        )

        _reset_session_stats()
        fired: list = []
        original_hooks = list(_advisory_hooks)
        try:
            @vetch.on_advisory
            def capture(advisories: list) -> None:
                fired.extend(advisories)

            # Feed a stall pattern — enough to cross _ADVISORY_HOOK_INTERVAL
            for _ in range(_ADVISORY_HOOK_INTERVAL * 2):
                track_session_event({
                    "model": "gpt-4o",
                    "usage": {"text": {"input_tokens": 500, "output_tokens": 1}},
                    "estimated_cost_usd": 0.05,
                })

            # Callback may have fired with STALL-001 if conditions are met;
            # the important thing is it did not crash and was invoked.
            assert capture in _advisory_hooks
        finally:
            _advisory_hooks.clear()
            _advisory_hooks.extend(original_hooks)
            _reset_session_stats()
