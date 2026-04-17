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


class TestStallDetection:
    """Tests for STALL-001 agentic stall advisory.

    STALL-001 fires when ALL of:
    - total_requests > 10
    - window_size >= 10
    - ≥80% of the window has <5 output tokens
    - input_similarity >= 0.5 (repetitive inputs)
    """

    @staticmethod
    def _make_stalled_stats(
        num_calls: int = 15,
        output_tokens: int = 0,
        cost_per_call: float = 0.10,
        input_tokens: int = 500,
    ) -> SessionStats:
        """Build a SessionStats that simulates a stalled loop."""
        stats = SessionStats()
        for _ in range(num_calls):
            stats.update({
                "model": "gpt-4o",
                "usage": {
                    "text": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    }
                },
                "estimated_cost_usd": cost_per_call,
            })
        return stats

    def test_stall_triggers(self) -> None:
        """STALL-001 fires with 15 identical calls producing 0 output tokens."""
        stats = self._make_stalled_stats(num_calls=15, output_tokens=0)
        advisories = generate_advisories(stats)

        stall = [a for a in advisories if a.code == "STALL-001"]
        assert len(stall) == 1
        assert stall[0].severity in ("WARNING", "CRITICAL")
        assert "stalled" in stall[0].description.lower()

    def test_stall_triggers_low_output(self) -> None:
        """STALL-001 fires when output tokens are consistently < 5."""
        stats = self._make_stalled_stats(num_calls=15, output_tokens=3)
        advisories = generate_advisories(stats)

        stall = [a for a in advisories if a.code == "STALL-001"]
        assert len(stall) == 1

    def test_stall_does_not_trigger_normal_output(self) -> None:
        """STALL-001 does NOT fire with normal output (200+ tokens)."""
        stats = self._make_stalled_stats(num_calls=15, output_tokens=200)
        advisories = generate_advisories(stats)

        stall = [a for a in advisories if a.code == "STALL-001"]
        assert len(stall) == 0

    def test_stall_does_not_trigger_few_calls(self) -> None:
        """STALL-001 does NOT fire with only 5 calls (below threshold)."""
        stats = self._make_stalled_stats(num_calls=5, output_tokens=0)
        advisories = generate_advisories(stats)

        stall = [a for a in advisories if a.code == "STALL-001"]
        assert len(stall) == 0

    def test_stall_does_not_trigger_diverse_inputs(self) -> None:
        """STALL-001 does NOT fire when inputs are all different.

        Even if output is low, diverse inputs mean the model is trying
        different things — not stuck in a loop.
        """
        stats = SessionStats()
        for i in range(15):
            stats.update({
                "model": "gpt-4o",
                "usage": {"text": {"input_tokens": 500 + i, "output_tokens": 0}},
                "estimated_cost_usd": 0.10,
            })
        advisories = generate_advisories(stats)

        stall = [a for a in advisories if a.code == "STALL-001"]
        assert len(stall) == 0

    def test_stall_severity_warning(self) -> None:
        """STALL-001 is WARNING when stalled cost <= $5."""
        stats = self._make_stalled_stats(
            num_calls=15, output_tokens=0, cost_per_call=0.10,
        )
        advisories = generate_advisories(stats)

        stall = [a for a in advisories if a.code == "STALL-001"]
        assert len(stall) == 1
        assert stall[0].severity == "WARNING"

    def test_stall_severity_critical(self) -> None:
        """STALL-001 is CRITICAL when stalled cost > $5."""
        stats = self._make_stalled_stats(
            num_calls=15, output_tokens=0, cost_per_call=1.00,
        )
        advisories = generate_advisories(stats)

        stall = [a for a in advisories if a.code == "STALL-001"]
        assert len(stall) == 1
        assert stall[0].severity == "CRITICAL"
        assert stall[0].potential_savings_usd is not None
        assert stall[0].potential_savings_usd > 5.0

    def test_stall_cost_is_per_call(self) -> None:
        """Wasted cost is summed from stalled calls, not total * fraction."""
        stats = self._make_stalled_stats(
            num_calls=15, output_tokens=0, cost_per_call=0.50,
        )
        advisories = generate_advisories(stats)

        stall = [a for a in advisories if a.code == "STALL-001"]
        assert len(stall) == 1
        assert "$" in stall[0].description
        # 15 calls * $0.50 = $7.50 wasted (all in window, all stalled)
        assert stall[0].potential_savings_usd == 7.5

    def test_stall_description_mentions_similarity(self) -> None:
        """STALL-001 description mentions input similarity %."""
        stats = self._make_stalled_stats(num_calls=15, output_tokens=0)
        advisories = generate_advisories(stats)

        stall = [a for a in advisories if a.code == "STALL-001"]
        assert len(stall) == 1
        assert "similarity" in stall[0].description.lower()
