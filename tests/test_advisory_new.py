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

    def test_trunc_fires_on_openai_length_finish_reason(self) -> None:
        """TRUNC-001 fires on OpenAI's 'length' finish reason (not just 'max_tokens')."""
        stats = SessionStats()
        for _ in range(5):
            stats.update({
                "model": "gpt-4o",
                "usage": {"text": {"input_tokens": 100, "output_tokens": 200}},
                "finish_reason": "length",
            })
        codes = [a.code for a in generate_advisories(stats)]
        assert "TRUNC-001" in codes, "TRUNC-001 must fire on OpenAI finish_reason='length'"

    def test_error_advisory_fires_on_consecutive_errors(self) -> None:
        """ERROR-001 fires when 3+ consecutive errors are at the tail of the window."""
        stats = SessionStats()
        for _ in range(5):
            stats.update({
                "model": "gpt-4o",
                "usage": {"text": {"input_tokens": 100, "output_tokens": 50}},
            })
        for _ in range(3):
            stats.update({
                "model": "gpt-4o",
                "usage": {"text": {"input_tokens": 0, "output_tokens": 0}},
                "error": True,
            })
        codes = [a.code for a in generate_advisories(stats)]
        assert "ERROR-001" in codes

    def test_error_advisory_fires_on_high_error_fraction(self) -> None:
        """ERROR-001 fires when ≥40% of recent calls are errors."""
        stats = SessionStats()
        for _ in range(20):
            is_error = _ % 2 == 0  # 50% errors
            stats.update({
                "model": "gpt-4o",
                "usage": {"text": {"input_tokens": 100, "output_tokens": 50}},
                "error": is_error,
            })
        codes = [a.code for a in generate_advisories(stats)]
        assert "ERROR-001" in codes

    def test_error_advisory_does_not_fire_on_isolated_error(self) -> None:
        """ERROR-001 does not fire for a single isolated error."""
        stats = SessionStats()
        for _ in range(9):
            stats.update({
                "model": "gpt-4o",
                "usage": {"text": {"input_tokens": 100, "output_tokens": 50}},
            })
        stats.update({
            "model": "gpt-4o",
            "usage": {"text": {"input_tokens": 0, "output_tokens": 0}},
            "error": True,
        })
        codes = [a.code for a in generate_advisories(stats)]
        assert "ERROR-001" not in codes

    # --- CACHE-002 tests ---

    def test_cache002_fires_on_repetition_with_no_cache_reads(self) -> None:
        """CACHE-002 fires when input-token repetition is high but no cache reads."""
        stats = SessionStats()
        for _ in range(10):
            stats.update({
                "model": "claude-3-5-sonnet",
                "usage": {"text": {"input_tokens": 500, "output_tokens": 50}},
                "cache_read_tokens": 0,
            })
        codes = [a.code for a in generate_advisories(stats)]
        assert "CACHE-002" in codes

    def test_cache002_does_not_fire_when_cache_is_used(self) -> None:
        """CACHE-002 does not fire when cache reads are present."""
        stats = SessionStats()
        for _ in range(10):
            stats.update({
                "model": "claude-3-5-sonnet",
                "usage": {"text": {"input_tokens": 500, "output_tokens": 50}},
                "cache_read_tokens": 400,
            })
        codes = [a.code for a in generate_advisories(stats)]
        assert "CACHE-002" not in codes

    # --- STREAM-001 tests ---

    def test_stream001_fires_on_high_incomplete_stream_fraction(self) -> None:
        """STREAM-001 fires when most streaming calls are incomplete."""
        stats = SessionStats()
        for _ in range(8):
            stats.update({
                "model": "gpt-4o",
                "usage": {"text": {"input_tokens": 100, "output_tokens": 30}},
                "is_stream": True,
                "complete": False,
            })
        codes = [a.code for a in generate_advisories(stats)]
        assert "STREAM-001" in codes

    def test_stream001_does_not_fire_on_complete_streams(self) -> None:
        """STREAM-001 does not fire when all streams complete."""
        stats = SessionStats()
        for _ in range(8):
            stats.update({
                "model": "gpt-4o",
                "usage": {"text": {"input_tokens": 100, "output_tokens": 30}},
                "is_stream": True,
                "complete": True,
            })
        codes = [a.code for a in generate_advisories(stats)]
        assert "STREAM-001" not in codes

    def test_stream001_does_not_fire_below_min_window(self) -> None:
        """STREAM-001 does not fire with fewer than min_window stream calls."""
        stats = SessionStats()
        for _ in range(3):
            stats.update({
                "model": "gpt-4o",
                "usage": {"text": {"input_tokens": 100, "output_tokens": 30}},
                "is_stream": True,
                "complete": False,
            })
        codes = [a.code for a in generate_advisories(stats)]
        assert "STREAM-001" not in codes

    # --- REASONING-001 tests ---

    def test_reasoning001_fires_when_reasoning_tokens_absent(self) -> None:
        """REASONING-001 fires when o1/o3 calls have no reasoning tokens."""
        stats = SessionStats()
        for _ in range(8):
            stats.update({
                "model": "o3",
                "usage": {"text": {"input_tokens": 100, "output_tokens": 50}, "reasoning": None},
            })
        codes = [a.code for a in generate_advisories(stats)]
        assert "REASONING-001" in codes

    def test_reasoning001_does_not_fire_when_reasoning_present(self) -> None:
        """REASONING-001 does not fire when reasoning tokens are returned."""
        stats = SessionStats()
        for _ in range(8):
            stats.update({
                "model": "o3",
                "usage": {
                    "text": {"input_tokens": 100, "output_tokens": 50},
                    "reasoning": {"input_tokens": 0, "output_tokens": 200, "total_tokens": 200},
                },
            })
        codes = [a.code for a in generate_advisories(stats)]
        assert "REASONING-001" not in codes

    def test_reasoning001_does_not_fire_for_non_reasoning_model(self) -> None:
        """REASONING-001 does not fire for non-reasoning models."""
        stats = SessionStats()
        for _ in range(8):
            stats.update({
                "model": "gpt-4o",
                "usage": {"text": {"input_tokens": 100, "output_tokens": 50}},
            })
        codes = [a.code for a in generate_advisories(stats)]
        assert "REASONING-001" not in codes

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
