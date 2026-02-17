"""Tests for context module."""

from vetch.context import (
    CapturedCall,
    TrackingContext,
    get_active_context,
    has_active_context,
)


class TestTrackingContext:
    """Tests for TrackingContext."""

    def test_context_manager_basic(self) -> None:
        """TrackingContext can be used as context manager."""
        with TrackingContext() as ctx:
            assert ctx is not None

    def test_stores_region(self) -> None:
        """TrackingContext stores region."""
        with TrackingContext(region="us-east-1") as ctx:
            assert ctx.region == "us-east-1"

    def test_stores_tags(self) -> None:
        """TrackingContext stores tags."""
        tags = {"team": "ml"}
        with TrackingContext(tags=tags) as ctx:
            assert ctx.tags == tags

    def test_active_context_set_during_with(self) -> None:
        """Active context is set during with block."""
        assert has_active_context() is False
        with TrackingContext() as ctx:
            assert has_active_context() is True
            assert get_active_context() is ctx
        assert has_active_context() is False

    def test_nested_contexts(self) -> None:
        """Nested contexts restore correctly."""
        with TrackingContext(region="outer") as outer:
            assert get_active_context() is outer
            with TrackingContext(region="inner") as inner:
                assert get_active_context() is inner
                assert inner.parent is outer
            assert get_active_context() is outer
        assert get_active_context() is None

    def test_nested_inherits_region(self) -> None:
        """Inner context inherits region from outer if not specified."""
        with TrackingContext(region="us-west-2"), TrackingContext() as inner:
            assert inner.region == "us-west-2"

    def test_nested_override_region(self) -> None:
        """Inner context can override region."""
        with TrackingContext(region="us-west-2"):
            with TrackingContext(region="eu-west-1") as inner:
                assert inner.region == "eu-west-1"

    def test_nested_merges_tags(self) -> None:
        """Inner context merges tags with outer."""
        with TrackingContext(tags={"team": "ml", "env": "prod"}):
            with TrackingContext(tags={"env": "dev", "feature": "x"}) as inner:
                # Inner should have merged tags with inner overriding
                assert inner.tags == {"team": "ml", "env": "dev", "feature": "x"}


class TestCapturedCall:
    """Tests for CapturedCall dataclass."""

    def test_minimal_capture(self) -> None:
        """CapturedCall works with minimal data."""
        call = CapturedCall(model="gpt-4", provider="openai")
        assert call.model == "gpt-4"
        assert call.provider == "openai"
        assert call.usage is None
        assert call.is_stream is False
        assert call.complete is True
        assert call.error is False

    def test_capture_method(self) -> None:
        """TrackingContext.capture creates CapturedCall."""
        with TrackingContext() as ctx:
            ctx.capture(
                model="gpt-4o",
                provider="openai",
                usage={"text": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}},
            )
            assert ctx.captured_call is not None
            assert ctx.captured_call.model == "gpt-4o"
            assert ctx.captured_call.usage is not None

    def test_capture_stream_metadata(self) -> None:
        """CapturedCall can store stream metadata."""
        call = CapturedCall(
            model="gpt-4",
            provider="openai",
            is_stream=True,
            accumulated_chars=5000,
            complete=True,
        )
        assert call.is_stream is True
        assert call.accumulated_chars == 5000

    def test_capture_error_metadata(self) -> None:
        """CapturedCall can store error metadata."""
        call = CapturedCall(
            model="gpt-4",
            provider="openai",
            error=True,
            error_type="RateLimitError",
            complete=False,
        )
        assert call.error is True
        assert call.error_type == "RateLimitError"
        assert call.complete is False


class TestGetActiveContext:
    """Tests for get_active_context function."""

    def test_returns_none_outside_context(self) -> None:
        """Returns None when not in a context."""
        assert get_active_context() is None

    def test_returns_context_inside(self) -> None:
        """Returns the active context when inside."""
        with TrackingContext() as ctx:
            assert get_active_context() is ctx


class TestHasActiveContext:
    """Tests for has_active_context function."""

    def test_false_outside_context(self) -> None:
        """Returns False when not in a context."""
        assert has_active_context() is False

    def test_true_inside_context(self) -> None:
        """Returns True when inside a context."""
        with TrackingContext():
            assert has_active_context() is True
