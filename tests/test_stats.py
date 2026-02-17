"""Tests for session statistics.

These tests verify:
- SessionStats tracking
- Event updates
- Summary generation
"""

from __future__ import annotations

from vetch.stats import SessionStats, get_session_stats, track_session_event


class TestSessionStats:
    """Tests for SessionStats dataclass."""

    def test_initial_state(self) -> None:
        """New SessionStats has zero values."""
        stats = SessionStats()

        assert stats.total_requests == 0
        assert stats.total_input_tokens == 0
        assert stats.total_output_tokens == 0
        assert stats.total_energy_wh == 0.0
        assert stats.total_carbon_g == 0.0
        assert stats.total_cost_usd == 0.0

    def test_update_from_event(self) -> None:
        """Update stats from an event dict."""
        stats = SessionStats()

        event = {
            "model": "gpt-4o",
            "usage": {
                "text": {
                    "input_tokens": 1000,
                    "output_tokens": 200,
                }
            },
            "estimated_energy_wh": 0.005,
            "estimated_carbon_g": 0.25,
            "estimated_cost_usd": 0.02,
        }

        stats.update(event)

        assert stats.total_requests == 1
        assert stats.total_input_tokens == 1000
        assert stats.total_output_tokens == 200
        assert stats.total_energy_wh == 0.005
        assert stats.total_carbon_g == 0.25
        assert stats.total_cost_usd == 0.02

    def test_update_multiple_events(self) -> None:
        """Accumulate stats from multiple events."""
        stats = SessionStats()

        for i in range(3):
            event = {
                "model": "gpt-4o",
                "usage": {"text": {"input_tokens": 100, "output_tokens": 50}},
                "estimated_energy_wh": 0.001,
                "estimated_carbon_g": 0.05,
                "estimated_cost_usd": 0.01,
            }
            stats.update(event)

        assert stats.total_requests == 3
        assert stats.total_input_tokens == 300
        assert stats.total_output_tokens == 150
        assert stats.total_energy_wh == 0.003
        assert stats.total_cost_usd == 0.03

    def test_track_models_used(self) -> None:
        """Track which models have been used."""
        stats = SessionStats()

        stats.update({"model": "gpt-4o", "usage": {"text": {"input_tokens": 100, "output_tokens": 50}}})
        stats.update({"model": "claude-3-opus", "usage": {"text": {"input_tokens": 100, "output_tokens": 50}}})
        stats.update({"model": "gpt-4o", "usage": {"text": {"input_tokens": 100, "output_tokens": 50}}})

        assert "gpt-4o" in stats.models_used
        assert "claude-3-opus" in stats.models_used
        assert len(stats.models_used) == 2

    def test_track_input_token_counts(self) -> None:
        """Track frequency of input token counts for pattern detection."""
        stats = SessionStats()

        # Same input count multiple times (static prompt pattern)
        for _ in range(5):
            stats.update({"model": "gpt-4o", "usage": {"text": {"input_tokens": 2000, "output_tokens": 100}}})

        stats.update({"model": "gpt-4o", "usage": {"text": {"input_tokens": 500, "output_tokens": 50}}})

        assert stats.input_token_counts[2000] == 5
        assert stats.input_token_counts[500] == 1

    def test_summary(self) -> None:
        """Generate summary dict."""
        stats = SessionStats()

        stats.update({
            "model": "gpt-4o",
            "usage": {"text": {"input_tokens": 1000, "output_tokens": 200}},
        })

        summary = stats.summary()

        assert summary["total_requests"] == 1
        assert summary["total_input_tokens"] == 1000
        assert summary["total_output_tokens"] == 200
        assert "average_input_output_ratio" in summary

    def test_summary_empty(self) -> None:
        """Summary handles empty stats."""
        stats = SessionStats()
        summary = stats.summary()

        assert summary["total_requests"] == 0
        assert summary["average_input_output_ratio"] == 0.0

    def test_handles_missing_usage(self) -> None:
        """Handle events with missing or None usage."""
        stats = SessionStats()

        stats.update({"model": "gpt-4o", "usage": None})
        stats.update({"model": "gpt-4o"})
        stats.update({"model": "gpt-4o", "usage": {}})

        # Should not crash, but also not count as requests
        assert stats.total_requests == 3  # Still counts requests
        assert stats.total_input_tokens == 0


class TestGlobalSessionStats:
    """Tests for global session stats singleton."""

    def test_get_session_stats(self) -> None:
        """get_session_stats returns SessionStats instance."""
        stats = get_session_stats()
        assert isinstance(stats, SessionStats)

    def test_singleton(self) -> None:
        """Same instance returned each time."""
        stats1 = get_session_stats()
        stats2 = get_session_stats()
        assert stats1 is stats2


class TestTrackSessionEvent:
    """Tests for track_session_event helper."""

    def test_track_session_event(self) -> None:
        """track_session_event updates global stats."""
        # Get current state
        stats = get_session_stats()
        initial_requests = stats.total_requests

        event = {
            "model": "gpt-4o",
            "usage": {"text": {"input_tokens": 500, "output_tokens": 100}},
            "estimated_cost_usd": 0.005,
        }

        track_session_event(event)

        assert stats.total_requests == initial_requests + 1
