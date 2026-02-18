"""Tests for wrapper module."""

import pytest

from vetch import wrap
from vetch.wrapper import VetchContext


class TestVetchContext:
    """Tests for VetchContext class."""

    def test_context_manager_basic(self) -> None:
        """Context manager can be used in with statement."""
        with VetchContext() as ctx:
            assert ctx is not None
            assert isinstance(ctx, VetchContext)

    def test_context_stores_region(self) -> None:
        """Context stores region parameter."""
        with VetchContext(region="us-east-1") as ctx:
            assert ctx.region == "us-east-1"

    def test_context_stores_tags(self) -> None:
        """Context stores tags parameter."""
        tags = {"team": "ml", "env": "prod"}
        with VetchContext(tags=tags) as ctx:
            assert ctx.tags == tags

    def test_event_available_after_exit(self) -> None:
        """Event is available after context exits."""
        with VetchContext() as ctx:
            pass

        assert ctx.event is not None
        assert ctx.event["schema_version"] == "1"

    def test_event_has_required_fields(self) -> None:
        """Event contains all required fields."""
        with VetchContext(region="eu-west-1") as ctx:
            pass

        event = ctx.event
        assert event is not None
        assert "event_id" in event
        assert "timestamp" in event
        assert "vetch_version" in event
        assert "signal_quality" in event
        assert event["region"] == "eu-west-1"

    def test_event_has_tags(self) -> None:
        """Event includes tags from context."""
        tags = {"team": "data"}
        with VetchContext(tags=tags) as ctx:
            pass

        assert ctx.event is not None
        assert ctx.event["tags"] == tags

    def test_tracking_not_disabled_by_default(self) -> None:
        """Tracking is enabled by default."""
        with VetchContext() as ctx:
            pass

        assert ctx.tracking_disabled is False

    def test_event_marks_error_on_exception(self) -> None:
        """Event marks error when exception occurs."""
        with pytest.raises(ValueError), VetchContext() as ctx:
            raise ValueError("test error")

        assert ctx.event is not None
        assert ctx.event["error"] is True
        assert ctx.event["error_type"] == "ValueError"
        assert ctx.event["complete"] is False

    def test_event_marks_complete_on_success(self) -> None:
        """Event marks complete on successful exit."""
        with VetchContext() as ctx:
            pass

        assert ctx.event is not None
        assert ctx.event["error"] is False
        assert ctx.event["complete"] is True

    def test_exception_not_suppressed(self) -> None:
        """Exceptions are not suppressed by context manager."""
        with pytest.raises(RuntimeError, match="test"), VetchContext():
            raise RuntimeError("test")

    def test_latency_recorded(self) -> None:
        """Latency is recorded in event."""
        import time

        with VetchContext() as ctx:
            time.sleep(0.01)  # 10ms

        assert ctx.event is not None
        assert ctx.event["latency_ms"] is not None
        assert ctx.event["latency_ms"] >= 10  # At least 10ms


class TestWrapFunction:
    """Tests for wrap() convenience function."""

    def test_wrap_returns_context(self) -> None:
        """wrap() returns a VetchContext."""
        with wrap() as ctx:
            assert isinstance(ctx, VetchContext)

    def test_wrap_passes_parameters(self) -> None:
        """wrap() passes parameters to VetchContext."""
        with wrap(region="ap-south-1", tags={"k": "v"}) as ctx:
            assert ctx.region == "ap-south-1"
            assert ctx.tags == {"k": "v"}


class TestEnergyOverrideValidation:
    """Tests for energy_override parameter validation."""

    def test_valid_override_accepted(self) -> None:
        """Valid energy override is accepted."""
        override = {
            "wh_per_1k_input": 0.8,
            "wh_per_1k_output": 2.4,
            "tier": 2,
            "source": "test",
        }
        ctx = VetchContext(energy_override=override)
        assert ctx._energy_override is not None
        assert ctx._energy_override["wh_per_1k_input"] == 0.8

    def test_invalid_override_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Invalid energy override logs warning."""
        import logging

        with caplog.at_level(logging.WARNING):
            ctx = VetchContext(energy_override={"invalid": "data"})

        assert ctx._energy_override is None
        assert "Invalid energy_override" in caplog.text

    def test_none_override_accepted(self) -> None:
        """None energy_override is accepted (no validation)."""
        ctx = VetchContext(energy_override=None)
        assert ctx._energy_override is None
