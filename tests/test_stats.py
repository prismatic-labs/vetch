"""Tests for session statistics.

These tests verify that session-level metrics are correctly aggregated.
"""

from __future__ import annotations

import pytest

from vetch.stats import SessionStats


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
        assert summary["recent_avg_output_tokens"] == 0.0
        assert summary["recent_low_output_count"] == 5
        assert summary["recent_low_output_fraction"] == 1.0
        assert summary["recent_window_size"] == 5
        assert summary["recent_stalled_cost_usd"] == pytest.approx(0.5)
        # All 5 calls had input_tokens=100, so similarity = 1.0
        assert summary["recent_input_similarity"] == 1.0

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
