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
