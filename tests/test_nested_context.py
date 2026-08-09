"""Tests for nested tracking contexts.

Verifies that nested wrap() calls:
1. Inherit region from the outer context
2. Merge tags (inner overrides outer)
3. Emit independent events
"""

from __future__ import annotations

from vetch.emitter import BufferedEmitter, set_test_emitter
from vetch.wrapper import VetchContext


class TestNestedContext:
    """Tests for nesting VetchContext."""

    def test_region_inheritance(self) -> None:
        """Inner context inherits region if not specified."""
        with VetchContext(region="us-east-1") as outer:
            with VetchContext() as inner:
                assert inner.region == "us-east-1"
            assert outer.region == "us-east-1"

    def test_tag_merging(self) -> None:
        """Inner tags are merged with outer tags."""
        with VetchContext(tags={"outer": "v1", "shared": "v1"}) as outer:
            with VetchContext(tags={"inner": "v2", "shared": "v2"}) as inner:
                assert inner.tags == {"outer": "v1", "inner": "v2", "shared": "v2"}
            assert outer.tags == {"outer": "v1", "shared": "v1"}

    def test_independent_events(self) -> None:
        """Each context emits its own event."""
        emitter = BufferedEmitter()
        set_test_emitter(emitter)

        from vetch.context import get_active_context

        _usage = {"text": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}
        try:
            with VetchContext(region="us-east-1", tags={"layer": "outer"}):
                with VetchContext(region="us-east-1", tags={"layer": "inner"}):
                    get_active_context().capture(model="gpt-4o", provider="openai", usage=_usage)
                get_active_context().capture(model="gpt-4o", provider="openai", usage=_usage)

            # Should have 2 events
            assert len(emitter) == 2

            # Events are emitted on __exit__, so inner is first
            assert emitter.events[0]["tags"] == {"layer": "inner"}
            assert emitter.events[1]["tags"] == {"layer": "outer"}
        finally:
            set_test_emitter(None)
