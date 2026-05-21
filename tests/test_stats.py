"""Tests for session statistics.

These tests verify that session-level metrics are correctly aggregated.
"""

from __future__ import annotations

import pytest

from vetch.stats import (
    _ADVISORY_HOOK_INTERVAL,
    SessionStats,
    _reset_session_stats,
    on_advisory,
    track_session_event,
)


class TestSessionStats:
    """Tests for SessionStats class."""

    def test_stats_initialization(self) -> None:
        """Verify initial state of stats."""
        stats = SessionStats()
        assert stats.total_requests == 0
        assert stats.total_input_tokens == 0
        assert stats.total_output_tokens == 0
        assert stats.total_energy_wh == 0.0
        assert stats.total_carbon_g == 0.0
        assert stats.total_cost_usd == 0.0
        assert len(stats.input_token_counts) == 0
        assert len(stats.models_used) == 0

    def test_stats_update(self) -> None:
        """Verify updating stats with an event."""
        stats = SessionStats()
        event = {
            "model": "gpt-4o",
            "usage": {"text": {"input_tokens": 100, "output_tokens": 50}},
            "estimated_energy_wh": 0.1,
            "estimated_carbon_g": 0.05,
            "estimated_cost_usd": 0.01,
        }

        stats.update(event)

        assert stats.total_requests == 1
        assert stats.total_input_tokens == 100
        assert stats.total_output_tokens == 50
        assert stats.total_energy_wh == 0.1
        assert stats.total_carbon_g == 0.05
        assert stats.total_cost_usd == 0.01
        assert stats.input_token_counts[100] == 1
        assert "gpt-4o" in stats.models_used

    def test_stats_multiple_updates(self) -> None:
        """Verify multiple updates aggregate correctly."""
        stats = SessionStats()

        for _ in range(3):
            event = {
                "model": "gpt-4o",
                "usage": {"text": {"input_tokens": 100, "output_tokens": 50}},
                "estimated_energy_wh": 0.1,
                "estimated_carbon_g": 0.05,
                "estimated_cost_usd": 0.01,
            }
            stats.update(event)

        assert stats.total_requests == 3
        assert stats.total_input_tokens == 300
        assert stats.total_output_tokens == 150
        assert stats.total_energy_wh == pytest.approx(0.3)
        assert stats.total_cost_usd == pytest.approx(0.03)

    def test_models_used_tracking(self) -> None:
        """Verify tracking of unique models."""
        stats = SessionStats()

        stats.update(
            {"model": "gpt-4o", "usage": {"text": {"input_tokens": 100, "output_tokens": 50}}}
        )
        stats.update(
            {
                "model": "claude-3-opus",
                "usage": {"text": {"input_tokens": 100, "output_tokens": 50}},
            }
        )
        stats.update(
            {"model": "gpt-4o", "usage": {"text": {"input_tokens": 100, "output_tokens": 50}}}
        )

        assert "gpt-4o" in stats.models_used
        assert "claude-3-opus" in stats.models_used
        assert len(stats.models_used) == 2

    def test_input_token_counts_tracking(self) -> None:
        """Verify tracking of input token counts for patterns."""
        stats = SessionStats()

        # Same input count multiple times (static prompt pattern)
        for _ in range(5):
            stats.update(
                {"model": "gpt-4o", "usage": {"text": {"input_tokens": 2000, "output_tokens": 100}}}
            )

        stats.update(
            {"model": "gpt-4o", "usage": {"text": {"input_tokens": 500, "output_tokens": 50}}}
        )

        assert stats.input_token_counts[2000] == 5
        assert stats.input_token_counts[500] == 1

    def test_summary_calculation(self) -> None:
        """Verify session summary data."""
        stats = SessionStats()
        stats.total_requests = 10
        stats.total_input_tokens = 1000
        stats.total_output_tokens = 200

        summary = stats.summary()
        assert summary["total_requests"] == 10
        assert summary["total_input_tokens"] == 1000
        assert summary["total_output_tokens"] == 200
        assert summary["average_input_output_ratio"] == 5.0

    def test_recent_output_tokens_populated(self) -> None:
        """Verify rolling window is populated on update."""
        stats = SessionStats()
        stats.update({
            "model": "gpt-4o",
            "usage": {"text": {"input_tokens": 100, "output_tokens": 42}},
        })
        assert stats.recent_output_tokens == [42]

    def test_recent_calls_bounded(self) -> None:
        """Verify rolling window doesn't exceed maxlen=20."""
        stats = SessionStats()
        for i in range(30):
            stats.update({
                "model": "gpt-4o",
                "usage": {"text": {"input_tokens": 100, "output_tokens": i}},
            })
        assert len(stats.recent_calls) == 20
        # Should contain the last 20 values (10..29)
        assert stats.recent_output_tokens == list(range(10, 30))

    def test_summary_includes_recent_fields(self) -> None:
        """Verify summary includes stall detection metrics."""
        stats = SessionStats()
        for _ in range(5):
            stats.update({
                "model": "gpt-4o",
                "usage": {"text": {"input_tokens": 100, "output_tokens": 0}},
                "estimated_cost_usd": 0.10,
            })
        summary = stats.summary()
        assert "recent_avg_output_tokens" in summary
        assert "recent_low_output_count" in summary
        assert "recent_low_output_fraction" in summary
        assert "recent_window_size" in summary
        assert "recent_stalled_cost_usd" in summary
        assert "recent_input_similarity" in summary
        assert "recent_output_token_cv" in summary
        assert "recent_input_growth_ratio" in summary
        assert "recent_input_increase_fraction" in summary
        assert "recent_empty_visible_output_count" in summary
        assert "recent_output_cap_hit_count" in summary
        assert "recent_output_cap_hit_fraction" in summary
        assert "recent_output_cap_count_window" in summary
        assert summary["recent_avg_output_tokens"] == 0.0
        assert summary["recent_output_token_cv"] == 0.0
        assert summary["recent_low_output_count"] == 5
        assert summary["recent_low_output_fraction"] == 1.0
        assert summary["recent_window_size"] == 5
        assert summary["recent_stalled_cost_usd"] == pytest.approx(0.5)
        # All 5 calls had input_tokens=100, so similarity = 1.0
        assert summary["recent_input_similarity"] == 1.0

    def test_input_growth_summary(self) -> None:
        """Verify summary captures monotonic context growth."""
        stats = SessionStats()
        for input_tokens in [100, 200, 300, 400, 500, 600]:
            stats.update({
                "model": "gpt-4o",
                "usage": {
                    "text": {"input_tokens": input_tokens, "output_tokens": 50}
                },
            })

        summary = stats.summary()
        assert summary["recent_input_increase_fraction"] == 1.0
        assert summary["recent_input_growth_ratio"] == pytest.approx(2.5)
        assert summary["recent_input_growth_tokens"] == pytest.approx(300.0)

    def test_empty_visible_output_summary(self) -> None:
        """Verify empty-visible output burn metrics are counted."""
        stats = SessionStats()
        for _ in range(4):
            stats.update({
                "model": "qwen3:8b",
                "usage": {"text": {"input_tokens": 200, "output_tokens": 160}},
                "visible_output_chars": 0,
                "requested_max_tokens": 160,
            })
        stats.update({
            "model": "qwen3:8b",
            "usage": {"text": {"input_tokens": 200, "output_tokens": 20}},
            "visible_output_chars": 100,
        })

        summary = stats.summary()
        assert summary["recent_empty_visible_output_count"] == 4
        assert summary["recent_empty_visible_output_fraction"] == pytest.approx(0.8)
        assert summary["recent_visible_output_count_window"] == 5
        assert summary["recent_output_cap_hit_count"] == 4
        assert summary["recent_output_cap_hit_fraction"] == pytest.approx(1.0)
        assert summary["recent_output_cap_count_window"] == 4

    def test_stalled_cost_only_counts_low_output(self) -> None:
        """Verify wasted cost sums only low-output calls, not all calls."""
        stats = SessionStats()
        # 3 normal calls at $1.00 each
        for _ in range(3):
            stats.update({
                "model": "gpt-4o",
                "usage": {"text": {"input_tokens": 500, "output_tokens": 200}},
                "estimated_cost_usd": 1.00,
            })
        # 2 stalled calls at $0.50 each
        for _ in range(2):
            stats.update({
                "model": "gpt-4o",
                "usage": {"text": {"input_tokens": 500, "output_tokens": 0}},
                "estimated_cost_usd": 0.50,
            })
        summary = stats.summary()
        assert summary["recent_stalled_cost_usd"] == pytest.approx(1.0)

    def test_input_similarity_diverse_inputs(self) -> None:
        """Verify input similarity is low when inputs vary."""
        stats = SessionStats()
        for i in range(10):
            stats.update({
                "model": "gpt-4o",
                "usage": {"text": {"input_tokens": 100 + i, "output_tokens": 50}},
            })
        summary = stats.summary()
        # All different input token counts → similarity = 1/10 = 0.1
        assert summary["recent_input_similarity"] == pytest.approx(0.1)

    def test_recent_output_token_cv(self) -> None:
        """Verify output token coefficient of variation for recent calls."""
        stats = SessionStats()
        for output_tokens in [95, 100, 105, 100, 100]:
            stats.update({
                "model": "gpt-4o",
                "usage": {
                    "text": {"input_tokens": 500, "output_tokens": output_tokens}
                },
            })

        summary = stats.summary()
        assert summary["recent_avg_output_tokens"] == 100
        assert summary["recent_output_token_cv"] == pytest.approx(0.0316)

    def test_water_ml_initialized_to_zero(self) -> None:
        """Verify initial water is zero."""
        stats = SessionStats()
        assert stats.total_water_ml == 0.0

    def test_water_ml_accumulated_from_ml(self) -> None:
        """Verify water accumulation from estimated_water_ml (MCP-style events)."""
        stats = SessionStats()
        stats.update({
            "model": "gpt-4o",
            "usage": {"text": {"input_tokens": 100, "output_tokens": 50}},
            "estimated_water_ml": 1.5,
        })
        stats.update({
            "model": "gpt-4o",
            "usage": {"text": {"input_tokens": 100, "output_tokens": 50}},
            "estimated_water_ml": 2.5,
        })
        assert stats.total_water_ml == pytest.approx(4.0)

    def test_water_ml_accumulated_from_liters(self) -> None:
        """Verify water accumulation from estimated_water_l (wrapper-style events)."""
        stats = SessionStats()
        stats.update({
            "model": "gpt-4o",
            "usage": {"text": {"input_tokens": 100, "output_tokens": 50}},
            "estimated_water_l": 0.003,
        })
        assert stats.total_water_ml == pytest.approx(3.0)

    def test_water_ml_in_summary(self) -> None:
        """Verify summary includes total_water_ml."""
        stats = SessionStats()
        stats.update({
            "model": "gpt-4o",
            "usage": {"text": {"input_tokens": 100, "output_tokens": 50}},
            "estimated_water_ml": 5.0,
        })
        summary = stats.summary()
        assert summary["total_water_ml"] == pytest.approx(5.0)

    def test_tool_use_finish_reason_excluded_from_empty_count(self) -> None:
        """tool_use finish_reason must not count as empty visible output (EMPTY-001 guard)."""
        stats = SessionStats()
        # 4 calls with tool_use stop — visible chars = 0 but should NOT count as empty
        for _ in range(4):
            stats.update({
                "model": "claude-3-haiku",
                "usage": {"text": {"input_tokens": 200, "output_tokens": 50}},
                "visible_output_chars": 0,
                "finish_reason": "tool_use",
            })
        summary = stats.summary()
        assert summary["recent_empty_visible_output_count"] == 0

    def test_tool_calls_finish_reason_excluded_from_empty_count(self) -> None:
        """tool_calls (OpenAI) finish_reason must not count as empty visible output."""
        stats = SessionStats()
        for _ in range(4):
            stats.update({
                "model": "gpt-4o",
                "usage": {"text": {"input_tokens": 200, "output_tokens": 50}},
                "visible_output_chars": 0,
                "finish_reason": "tool_calls",
            })
        summary = stats.summary()
        assert summary["recent_empty_visible_output_count"] == 0

    def test_max_tokens_finish_reason_counted(self) -> None:
        """finish_reason=max_tokens is counted in the truncation stat."""
        stats = SessionStats()
        for _ in range(5):
            stats.update({
                "model": "gpt-4o",
                "usage": {"text": {"input_tokens": 100, "output_tokens": 200}},
                "finish_reason": "max_tokens",
            })
        stats.update({
            "model": "gpt-4o",
            "usage": {"text": {"input_tokens": 100, "output_tokens": 50}},
            "finish_reason": "stop",
        })
        summary = stats.summary()
        assert summary["recent_max_tokens_finish_count"] == 5
        assert summary["recent_max_tokens_finish_fraction"] == pytest.approx(5 / 6, rel=1e-3)

    def test_max_tokens_finish_zero_when_none(self) -> None:
        """No max_tokens finish_reasons means count=0."""
        stats = SessionStats()
        for _ in range(5):
            stats.update({
                "model": "gpt-4o",
                "usage": {"text": {"input_tokens": 100, "output_tokens": 50}},
            })
        summary = stats.summary()
        assert summary["recent_max_tokens_finish_count"] == 0
        assert summary["recent_max_tokens_finish_fraction"] == 0.0

    def test_update_is_thread_safe(self) -> None:
        """Concurrent updates must not corrupt counts or crash with RuntimeError."""
        import threading
        stats = SessionStats()
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(50):
                    stats.update({
                        "model": "gpt-4o",
                        "usage": {"text": {"input_tokens": 100, "output_tokens": 20}},
                        "estimated_cost_usd": 0.001,
                    })
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        assert stats.total_requests == 400
        assert stats.total_input_tokens == 40000

    def test_summary_is_cached_until_update(self) -> None:
        """summary() returns the same object if no update has occurred."""
        stats = SessionStats()
        stats.update({"model": "gpt-4o", "usage": {
            "text": {"input_tokens": 100, "output_tokens": 20},
        }})
        s1 = stats.summary()
        s2 = stats.summary()
        # Both calls return equal dicts; second one comes from cache
        assert s1 == s2
        # After an update the cache is invalidated
        stats.update({"model": "gpt-4o", "usage": {
            "text": {"input_tokens": 200, "output_tokens": 40},
        }})
        s3 = stats.summary()
        assert s3["total_requests"] == 2


class TestAdvisoryHooks:
    """Tests for advisory push hooks on the global singleton."""

    def teardown_method(self) -> None:
        _reset_session_stats()

    def test_on_advisory_does_not_fire_empty_lists(self) -> None:
        _reset_session_stats()
        fired: list[object] = []
        on_advisory(lambda advisories: fired.append(advisories))

        for index in range(_ADVISORY_HOOK_INTERVAL):
            track_session_event({
                "model": "gpt-4o",
                "usage": {
                    "text": {"input_tokens": 100 + index, "output_tokens": 100}
                },
            })

        assert fired == []

    def test_on_advisory_fires_when_signal_exists(self) -> None:
        _reset_session_stats()
        fired: list[list[object]] = []
        on_advisory(lambda advisories: fired.append(advisories))

        for _ in range(_ADVISORY_HOOK_INTERVAL * 2):
            track_session_event({
                "model": "gpt-4o",
                "usage": {"text": {"input_tokens": 500, "output_tokens": 1}},
                "estimated_cost_usd": 0.05,
            })

        assert fired
        codes = {advisory.code for batch in fired for advisory in batch}
        assert "STALL-001" in codes
