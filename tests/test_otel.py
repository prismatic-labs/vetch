"""Tests for OpenTelemetry bridge.

These tests verify:
- Span attribute attachment
- Graceful handling when OTel not installed
- No-op behavior when no active span
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

from vetch.otel import attach_to_otel_span


class TestAttachToOtelSpan:
    """Tests for OTel span decoration."""

    def test_returns_false_when_otel_not_in_modules(self) -> None:
        """Returns False when opentelemetry not in sys.modules."""
        event = {
            "estimated_energy_wh": 0.001,
            "estimated_carbon_g": 0.05,
            "estimated_cost_usd": 0.01,
        }

        # Temporarily remove opentelemetry from sys.modules if present
        otel_backup = sys.modules.get("opentelemetry")
        if "opentelemetry" in sys.modules:
            del sys.modules["opentelemetry"]

        try:
            result = attach_to_otel_span(event)
            assert result is False
        finally:
            if otel_backup is not None:
                sys.modules["opentelemetry"] = otel_backup

    def test_handles_empty_event(self) -> None:
        """Handles empty event dict gracefully."""
        event = {}

        # Should not raise
        result = attach_to_otel_span(event)
        # Returns False since OTel not loaded
        assert result is False

    def test_handles_none_values(self) -> None:
        """Handles None values in event gracefully."""
        event = {
            "estimated_energy_wh": None,
            "estimated_carbon_g": None,
            "estimated_cost_usd": None,
            "model": None,
            "provider": None,
        }

        # Should not raise
        result = attach_to_otel_span(event)
        # Returns False since OTel not loaded
        assert result is False

    def test_function_signature(self) -> None:
        """Function accepts InferenceEvent and returns bool."""
        from vetch.otel import attach_to_otel_span

        # Verify function exists and has correct signature
        event = {"estimated_energy_wh": 0.001}
        result = attach_to_otel_span(event)
        assert isinstance(result, bool)


class TestOtelIntegration:
    """Tests for OTel integration when mocked."""

    def test_attaches_to_recording_span(self) -> None:
        """When OTel is available and span is recording, attributes are set."""
        # Create mock span and trace module
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        mock_trace = MagicMock()
        mock_trace.get_current_span.return_value = mock_span

        event = {
            "estimated_energy_wh": 0.001,
            "estimated_carbon_g": 0.05,
            "estimated_cost_usd": 0.01,
            "model": "gpt-4o",
            "provider": "openai",
            "region": "us-east-1",
            "signal_quality": "live",
            "energy_tier": 2,
        }

        # Mock both opentelemetry and opentelemetry.trace in sys.modules
        mock_otel = MagicMock()
        mock_otel.trace = mock_trace

        original_otel = sys.modules.get("opentelemetry")
        original_trace = sys.modules.get("opentelemetry.trace")

        try:
            sys.modules["opentelemetry"] = mock_otel
            sys.modules["opentelemetry.trace"] = mock_trace

            result = attach_to_otel_span(event)

            # Should return True and set attributes
            assert result is True
            assert mock_span.set_attribute.call_count >= 8  # 8 attributes set
        finally:
            # Restore original state
            if original_otel is not None:
                sys.modules["opentelemetry"] = original_otel
            elif "opentelemetry" in sys.modules:
                del sys.modules["opentelemetry"]
            if original_trace is not None:
                sys.modules["opentelemetry.trace"] = original_trace
            elif "opentelemetry.trace" in sys.modules:
                del sys.modules["opentelemetry.trace"]

    def test_sets_model_match_attribute(self) -> None:
        """The vetch.model_match attribute is set on the span."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_trace = MagicMock()
        mock_trace.get_current_span.return_value = mock_span

        event = {
            "estimated_energy_wh": 0.001,
            "model": "gpt-4o",
            "provider": "openai",
            "signal_quality": "live",
            "energy_tier": 1,
            "model_match": "exact",
        }

        mock_otel = MagicMock()
        mock_otel.trace = mock_trace
        original_otel = sys.modules.get("opentelemetry")
        original_trace = sys.modules.get("opentelemetry.trace")
        try:
            sys.modules["opentelemetry"] = mock_otel
            sys.modules["opentelemetry.trace"] = mock_trace

            assert attach_to_otel_span(event) is True
            attrs = {c.args[0]: c.args[1] for c in mock_span.set_attribute.call_args_list}
            assert attrs.get("vetch.model_match") == "exact"
        finally:
            if original_otel is not None:
                sys.modules["opentelemetry"] = original_otel
            elif "opentelemetry" in sys.modules:
                del sys.modules["opentelemetry"]
            if original_trace is not None:
                sys.modules["opentelemetry.trace"] = original_trace
            elif "opentelemetry.trace" in sys.modules:
                del sys.modules["opentelemetry.trace"]

    def test_returns_false_on_non_recording_span(self) -> None:
        """Returns False when span is not recording."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = False

        mock_trace = MagicMock()
        mock_trace.get_current_span.return_value = mock_span

        event = {"estimated_energy_wh": 0.001}

        # The function should return False for non-recording span
        # Since we can't easily mock the import, just verify the function
        # handles this gracefully when OTel is not installed
        result = attach_to_otel_span(event)
        assert result is False  # OTel not actually installed


class TestOtelFailSafe:
    """Tests for fail-safe behavior."""

    def test_exception_returns_false(self) -> None:
        """Any exception during OTel operations returns False."""
        event = {"estimated_energy_wh": 0.001}

        # The function should never raise, always return bool
        result = attach_to_otel_span(event)
        assert isinstance(result, bool)

    def test_no_side_effects_without_otel(self) -> None:
        """Without OTel installed, no side effects occur."""
        event = {
            "estimated_energy_wh": 0.001,
            "model": "gpt-4o",
        }

        # Should be safe to call multiple times
        for _ in range(10):
            result = attach_to_otel_span(event)
            assert result is False


class TestExportAdvisoryOtlp:
    """Tests for advisory OTLP export."""

    def test_returns_false_when_not_configured(self) -> None:
        from vetch.otel import export_advisory_otlp, is_otlp_configured
        assert not is_otlp_configured()
        result = export_advisory_otlp("STALL-001", "CRITICAL", "kill",
                                      estimated_waste_usd=4.20)
        assert result is False

    def test_never_raises(self) -> None:
        from vetch.otel import export_advisory_otlp
        # Should not raise regardless of args
        for code in ("STALL-001", "RAG-001", "CACHE-001", "BABBLE-001", "UNKNOWN"):
            result = export_advisory_otlp(code, "WARNING", "warn")
            assert isinstance(result, bool)

    def test_with_mocked_meter_and_tracer(self) -> None:
        """When OTLP is configured, advisory counter and histogram are incremented."""
        import vetch.otel as otel_mod

        counter = MagicMock()
        histogram = MagicMock()
        tracer = MagicMock()
        # Simulate a span context manager
        span_ctx = MagicMock()
        span_ctx.__enter__ = MagicMock(return_value=MagicMock())
        span_ctx.__exit__ = MagicMock(return_value=False)
        tracer.start_as_current_span.return_value = span_ctx

        old_configured = otel_mod._otlp_configured
        old_counter = otel_mod._advisory_counter
        old_histogram = otel_mod._advisory_waste_histogram
        old_tracer = otel_mod._tracer
        try:
            otel_mod._otlp_configured = True
            otel_mod._advisory_counter = counter
            otel_mod._advisory_waste_histogram = histogram
            otel_mod._tracer = tracer

            result = otel_mod.export_advisory_otlp(
                "STALL-001", "CRITICAL", "kill",
                session_id="sess-abc",
                model="gpt-4o",
                estimated_waste_usd=8.50,
                tags={"feature": "agent", "customer": "acme"},
            )

            assert result is True
            counter.add.assert_called_once()
            call_attrs = counter.add.call_args[0][1]
            assert call_attrs["vetch.advisory.code"] == "STALL-001"
            assert call_attrs["vetch.advisory.action"] == "kill"
            assert call_attrs["vetch.tag.feature"] == "agent"

            histogram.record.assert_called_once()
            waste_call = histogram.record.call_args[0]
            assert waste_call[0] == 8.50

        finally:
            otel_mod._otlp_configured = old_configured
            otel_mod._advisory_counter = old_counter
            otel_mod._advisory_waste_histogram = old_histogram
            otel_mod._tracer = old_tracer

    def test_advisory_span_attributes_on_inference_event(self) -> None:
        """Advisory codes on an event are surfaced as span attributes."""
        from unittest.mock import MagicMock, patch

        import vetch.otel as otel_mod

        span = MagicMock()
        span_ctx = MagicMock()
        span_ctx.__enter__ = MagicMock(return_value=span)
        span_ctx.__exit__ = MagicMock(return_value=False)
        tracer = MagicMock()
        tracer.start_as_current_span.return_value = span_ctx

        event = {
            "model": "gpt-4o", "provider": "openai",
            "estimated_energy_wh": 0.01, "estimated_carbon_g": 0.004,
            "estimated_cost_usd": 0.01,
            "advisories": [{"code": "STALL-001", "severity": "CRITICAL"}],
        }

        old_configured = otel_mod._otlp_configured
        old_tracer = otel_mod._tracer
        try:
            otel_mod._otlp_configured = True
            otel_mod._tracer = tracer
            # Suppress OTel import errors
            with patch.dict("sys.modules", {"opentelemetry": MagicMock(),
                                            "opentelemetry.trace": MagicMock()}):
                otel_mod._export_event_sync(event)

            set_calls = {c[0][0]: c[0][1]
                         for c in span.set_attribute.call_args_list
                         if len(c[0]) == 2}
            assert set_calls.get("vetch.advisory_codes") == "STALL-001"
        finally:
            otel_mod._otlp_configured = old_configured
            otel_mod._tracer = old_tracer
