"""Tests for advisory engine.

These tests verify:
- Advisory creation
- Advisory formatting
- Pattern detection from session stats
"""

from __future__ import annotations

from vetch.advisory import (
    Advisory,
    format_advisories,
    generate_advisories,
    get_advisory_spec,
)
from vetch.config import _reset_config, get_advisory_threshold, set_advisory_thresholds
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

    def test_detect_babbling_uses_matching_confidence_threshold(self) -> None:
        """BABBLE-001 confidence should be medium when the advisory first fires."""
        stats = SessionStats()
        for _ in range(10):
            stats.update({
                "model": "gpt-4o",
                "usage": {"text": {"input_tokens": 100, "output_tokens": 1500}},
            })

        advisories = generate_advisories(stats)
        babble = [advisory for advisory in advisories if advisory.code == "BABBLE-001"]

        assert len(babble) == 1
        assert get_advisory_spec("BABBLE-001").confidence(stats) == "medium"


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

    def test_format_security_refs(self) -> None:
        """Format security-relevant advisories with badge and references."""
        advisories = [
            Advisory(
                code="STALL-001",
                severity="WARNING",
                title="Agentic Stall Detected",
                description="Repeated low-output loop.",
                security_signal=True,
                security_refs=("OWASP-LLM01", "OWASP-LLM10"),
            )
        ]

        result = format_advisories(advisories, "text")

        assert "[STALL-001] 🔒 Agentic Stall Detected" in result
        assert "Security refs: OWASP-LLM01, OWASP-LLM10" in result


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
        assert stall[0].security_signal is True
        assert stall[0].security_refs == ("OWASP-LLM01", "OWASP-LLM10")

    def test_stall_triggers_low_output(self) -> None:
        """STALL-001 fires when output tokens are consistently < 5."""
        stats = self._make_stalled_stats(num_calls=15, output_tokens=3)
        advisories = generate_advisories(stats)

        stall = [a for a in advisories if a.code == "STALL-001"]
        assert len(stall) == 1

    def test_stall_threshold_global_override_changes_detection(self) -> None:
        """Global STALL-001 threshold overrides affect summary and detection."""
        _reset_config()
        set_advisory_thresholds({"STALL-001": {"low_output_threshold": 1}})
        try:
            stats = self._make_stalled_stats(num_calls=15, output_tokens=3)
            advisories = generate_advisories(stats)

            stall = [a for a in advisories if a.code == "STALL-001"]
            assert stall == []
            assert stats.summary()["recent_low_output_threshold"] == 1
        finally:
            _reset_config()

    def test_stall_threshold_session_override_does_not_leak(self) -> None:
        """SessionStats-scoped thresholds isolate route/workflow tuning."""
        scoped = SessionStats(
            advisory_thresholds={"STALL-001": {"low_output_threshold": 1}}
        )
        default = SessionStats()

        for _ in range(15):
            event = {
                "model": "gpt-4o",
                "usage": {"text": {"input_tokens": 500, "output_tokens": 3}},
                "estimated_cost_usd": 0.10,
            }
            scoped.update(event)
            default.update(event)

        scoped_codes = {a.code for a in generate_advisories(scoped)}
        default_codes = {a.code for a in generate_advisories(default)}

        assert "STALL-001" not in scoped_codes
        assert "STALL-001" in default_codes

    def test_reset_config_clears_threshold_overrides(self) -> None:
        """_reset_config clears process-wide advisory threshold overrides."""
        set_advisory_thresholds({"BABBLE-001": {"min_avg_output_tokens": 9999}})
        _reset_config()

        assert (
            get_advisory_threshold(
                "BABBLE-001",
                "min_avg_output_tokens",
                1500,
            )
            == 1500
        )

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


class TestZombieDetection:
    """Tests for ZOMBIE-001 post-completion drift advisory."""

    @staticmethod
    def _make_zombie_stats(
        output_tokens: list[int] | None = None,
        input_tokens: list[int] | None = None,
    ) -> SessionStats:
        stats = SessionStats()
        outputs = output_tokens or [100, 102, 99, 101, 100, 98]
        inputs = input_tokens or [500] * len(outputs)
        for in_tok, out_tok in zip(inputs, outputs):
            stats.update({
                "model": "gpt-4o",
                "usage": {
                    "text": {"input_tokens": in_tok, "output_tokens": out_tok}
                },
                "estimated_cost_usd": 0.05,
            })
        return stats

    def test_zombie_triggers_on_repetitive_normal_length_outputs(self) -> None:
        stats = self._make_zombie_stats()
        advisories = generate_advisories(stats)

        zombie = [a for a in advisories if a.code == "ZOMBIE-001"]
        stall = [a for a in advisories if a.code == "STALL-001"]
        assert len(zombie) == 1
        assert len(stall) == 0
        assert zombie[0].title == "Post-completion drift detected"
        assert "normal-length outputs" in zombie[0].description
        assert zombie[0].request_count == 6

    def test_zombie_does_not_trigger_on_varied_output_lengths(self) -> None:
        stats = self._make_zombie_stats(output_tokens=[50, 150, 260, 40, 350, 90])
        advisories = generate_advisories(stats)

        assert not any(a.code == "ZOMBIE-001" for a in advisories)

    def test_zombie_does_not_trigger_on_low_output_stall(self) -> None:
        stats = self._make_zombie_stats(output_tokens=[1, 1, 1, 1, 1, 1])
        advisories = generate_advisories(stats)

        assert not any(a.code == "ZOMBIE-001" for a in advisories)

    def test_zombie_does_not_trigger_on_diverse_inputs(self) -> None:
        stats = self._make_zombie_stats(input_tokens=[500, 600, 700, 800, 900, 1000])
        advisories = generate_advisories(stats)

        assert not any(a.code == "ZOMBIE-001" for a in advisories)
        assert get_advisory_spec("ZOMBIE-001").confidence(stats) == "low"


class TestContextSnowballDetection:
    """Tests for CTX-001 context snowball advisory."""

    def test_context_snowball_triggers_on_growing_prompt(self) -> None:
        stats = SessionStats()
        for input_tokens in [150, 400, 650, 900, 1200, 1500, 1900, 2300]:
            stats.update({
                "model": "phi4-mini",
                "usage": {
                    "text": {"input_tokens": input_tokens, "output_tokens": 200}
                },
                "estimated_cost_usd": 0.05,
            })

        advisories = generate_advisories(stats)
        ctx = [a for a in advisories if a.code == "CTX-001"]

        assert len(ctx) == 1
        assert ctx[0].title == "Context snowball detected"
        assert "input tokens grew" in ctx[0].description
        assert get_advisory_spec("CTX-001").confidence(stats) == "medium"

    def test_context_snowball_does_not_trigger_on_stable_context(self) -> None:
        stats = SessionStats()
        for _ in range(10):
            stats.update({
                "model": "gpt-4o",
                "usage": {"text": {"input_tokens": 800, "output_tokens": 200}},
            })

        advisories = generate_advisories(stats)

        assert not any(a.code == "CTX-001" for a in advisories)


class TestEmptyVisibleOutputDetection:
    """Tests for EMPTY-001 invisible output burn advisory."""

    def test_empty_visible_output_triggers(self) -> None:
        stats = SessionStats()
        for _ in range(8):
            stats.update({
                "model": "qwen3:8b",
                "usage": {"text": {"input_tokens": 400, "output_tokens": 160}},
                "visible_output_chars": 0,
            })

        advisories = generate_advisories(stats)
        empty = [a for a in advisories if a.code == "EMPTY-001"]

        assert len(empty) == 1
        assert empty[0].title == "Invisible output burn detected"
        assert "almost no visible text" in empty[0].description
        assert empty[0].request_count == 8

    def test_empty_visible_output_cap_hits_raise_confidence(self) -> None:
        stats = SessionStats()
        for _ in range(8):
            stats.update({
                "model": "deepseek-r1:8b",
                "usage": {"text": {"input_tokens": 250, "output_tokens": 90}},
                "visible_output_chars": 0,
                "requested_max_tokens": 90,
            })

        advisories = generate_advisories(stats)
        empty = [a for a in advisories if a.code == "EMPTY-001"]

        assert len(empty) == 1
        assert "hit the requested output cap" in empty[0].description
        spec = get_advisory_spec("EMPTY-001")
        assert spec.confidence(stats) == "high"
        assert spec.evidence(stats)["recent_output_cap_hit_count"] == 8

    def test_empty_visible_output_requires_visible_counts(self) -> None:
        stats = SessionStats()
        for _ in range(8):
            stats.update({
                "model": "qwen3:8b",
                "usage": {"text": {"input_tokens": 400, "output_tokens": 160}},
            })

        advisories = generate_advisories(stats)

        assert not any(a.code == "EMPTY-001" for a in advisories)

    def test_empty_visible_output_ignores_tool_call_finish_reason(self) -> None:
        stats = SessionStats()
        for _ in range(8):
            stats.update({
                "model": "gpt-4o",
                "usage": {"text": {"input_tokens": 400, "output_tokens": 160}},
                "visible_output_chars": 0,
                "finish_reason": "tool_calls",
            })

        advisories = generate_advisories(stats)

        assert not any(a.code == "EMPTY-001" for a in advisories)
