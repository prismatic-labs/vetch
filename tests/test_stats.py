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
