"""Tests for ZOMBIE-001 long-output repetition detection."""

from __future__ import annotations

from vetch.advisory import generate_advisories
from vetch.stats import SessionStats


class TestStallLongOutput:
    """Tests for ZOMBIE-001 post-completion drift advisory."""

    @staticmethod
    def _make_repetitive_stats(
        num_calls: int = 15,
        output_tokens: int = 200,
        input_tokens: int = 500,
        cost_per_call: float = 0.10,
    ) -> SessionStats:
        """Build a SessionStats that simulates a long-output repetitive loop."""
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

    def test_zombie_001_triggers(self) -> None:
        """ZOMBIE-001 fires with repeated normal-length output."""
        stats = self._make_repetitive_stats(num_calls=15, output_tokens=200)
        advisories = generate_advisories(stats)

        zombie = [a for a in advisories if a.code == "ZOMBIE-001"]
        assert len(zombie) == 1
        assert zombie[0].title == "Post-completion drift detected"
        assert "replay loop" in zombie[0].description.lower()

    def test_zombie_001_does_not_trigger_diverse_output(self) -> None:
        """ZOMBIE-001 does NOT fire when output token counts vary."""
        stats = SessionStats()
        for output_tokens in [50, 250, 80, 400, 120, 310, 60, 280, 90, 500]:
            stats.update({
                "model": "gpt-4o",
                "usage": {
                    "text": {"input_tokens": 500, "output_tokens": output_tokens}
                },
                "estimated_cost_usd": 0.10,
            })
        advisories = generate_advisories(stats)

        zombie = [a for a in advisories if a.code == "ZOMBIE-001"]
        assert len(zombie) == 0

    def test_zombie_001_does_not_overlap_with_stall_001(self) -> None:
        """ZOMBIE-001 does NOT fire if STALL-001 already fired (low output)."""
        stats = self._make_repetitive_stats(num_calls=15, output_tokens=0)
        advisories = generate_advisories(stats)

        stall_001 = [a for a in advisories if a.code == "STALL-001"]
        zombie_001 = [a for a in advisories if a.code == "ZOMBIE-001"]

        assert len(stall_001) == 1
        assert len(zombie_001) == 0
