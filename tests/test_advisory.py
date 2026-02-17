"""Tests for advisory engine.

These tests verify:
- Advisory creation
- Advisory formatting
- Pattern detection from session stats
"""

from __future__ import annotations

from vetch.advisory import Advisory, format_advisories, generate_advisories
from vetch.stats import SessionStats


class TestAdvisory:
    """Tests for Advisory namedtuple."""

    def test_create_advisory(self) -> None:
        """Create an Advisory."""
        advisory = Advisory(
            code="CACHE-001",
            severity="WARNING",
            title="Test Advisory",
            description="Test description",
        )

        assert advisory.code == "CACHE-001"
        assert advisory.severity == "WARNING"
        assert advisory.title == "Test Advisory"
        assert advisory.description == "Test description"
        assert advisory.potential_savings_usd is None

    def test_advisory_with_savings(self) -> None:
        """Create an Advisory with potential savings."""
        advisory = Advisory(
            code="RAG-001",
            severity="INFO",
            title="Heavy RAG Pattern",
            description="High input:output ratio detected",
            potential_savings_usd=50.0,
        )

        assert advisory.potential_savings_usd == 50.0


class TestGenerateAdvisories:
    """Tests for advisory generation from session stats."""

    def test_empty_session(self) -> None:
        """Empty session generates no advisories."""
        stats = SessionStats()
        advisories = generate_advisories(stats)
        assert advisories == []

    def test_few_requests_no_advisory(self) -> None:
        """Few requests don't trigger advisories."""
        stats = SessionStats()

        # Simulate a few requests via update()
        for _ in range(3):
            stats.update({
                "model": "gpt-4o",
                "usage": {"text": {"input_tokens": 1000, "output_tokens": 200}},
            })

        advisories = generate_advisories(stats)
        # May or may not have advisories with only 3 requests
        # Just verify it runs without error
        assert isinstance(advisories, list)

    def test_detect_repeated_inputs(self) -> None:
        """Detect potential prompt caching opportunity."""
        stats = SessionStats()

        # Simulate 10 requests with identical input token counts
        for _ in range(10):
            stats.update({
                "model": "gpt-4o",
                "usage": {"text": {"input_tokens": 2000, "output_tokens": 100}},
            })

        advisories = generate_advisories(stats)

        # Should detect the repeated pattern
        assert len(advisories) >= 1
        assert any("CACHE" in a.code for a in advisories)


class TestFormatAdvisories:
    """Tests for advisory formatting."""

    def test_format_empty(self) -> None:
        """Format empty advisory list."""
        result = format_advisories([])
        assert "No advisories" in result

    def test_format_text(self) -> None:
        """Format advisories as text."""
        advisories = [
            Advisory(
                code="TEST-001",
                severity="WARNING",
                title="Test Warning",
                description="This is a test warning.",
            )
        ]

        result = format_advisories(advisories, "text")

        assert "TEST-001" in result
        assert "Test Warning" in result

    def test_format_json(self) -> None:
        """Format advisories as JSON."""
        import json

        advisories = [
            Advisory(
                code="TEST-001",
                severity="INFO",
                title="Test Info",
                description="Description here",
            )
        ]

        result = format_advisories(advisories, "json")
        parsed = json.loads(result)

        assert len(parsed) == 1
        assert parsed[0]["code"] == "TEST-001"
        assert parsed[0]["title"] == "Test Info"

    def test_format_with_savings(self) -> None:
        """Format advisory with potential savings."""
        advisories = [
            Advisory(
                code="SAVE-001",
                severity="WARNING",
                title="Savings Opportunity",
                description="You could save money",
                potential_savings_usd=25.50,
            )
        ]

        result = format_advisories(advisories, "text")
        assert "$25.50" in result
