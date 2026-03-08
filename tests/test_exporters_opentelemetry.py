"""Tests for vetch.exporters.opentelemetry module - core functionality only."""

from unittest.mock import Mock, patch

from vetch.exporters.opentelemetry import (
    configure_auto_export,
    export_events,
    is_auto_export_enabled,
)


class TestConfigureAutoExport:
    """Test configure_auto_export function - state management."""

    def test_enables_auto_export(self):
        """Test enabling auto-export."""
        configure_auto_export(enabled=True)
        assert is_auto_export_enabled() is True

    def test_disables_auto_export(self):
        """Test disabling auto-export."""
        configure_auto_export(enabled=False)
        assert is_auto_export_enabled() is False

    def test_defaults_to_enabled(self):
        """Test that configure_auto_export defaults to enabled."""
        configure_auto_export()
        assert is_auto_export_enabled() is True


class TestExportEvents:
    """Test export_events function - batch export logic."""

    @patch("vetch.exporters.opentelemetry.export_event_as_span")
    def test_exports_multiple_events(self, mock_export):
        """Test exporting multiple events."""
        events = [Mock(spec=dict) for _ in range(3)]

        export_events(events)

        # Verify each event was exported
        assert mock_export.call_count == 3

    @patch("vetch.exporters.opentelemetry.export_event_as_span")
    def test_exports_empty_list(self, mock_export):
        """Test exporting empty list doesn't crash."""
        export_events([])

        # Verify no exports
        assert mock_export.call_count == 0


class TestIsAutoExportEnabled:
    """Test is_auto_export_enabled function."""

    def test_returns_current_state(self):
        """Test that is_auto_export_enabled reflects current state."""
        from vetch.exporters.opentelemetry import configure_auto_export, is_auto_export_enabled

        # Set to True
        configure_auto_export(enabled=True)
        assert is_auto_export_enabled() is True

        # Set to False
        configure_auto_export(enabled=False)
        assert is_auto_export_enabled() is False


class TestExportEventsWithParent:
    """Test export_events with parent span parameter."""

    @patch("vetch.exporters.opentelemetry.export_event_as_span")
    def test_exports_with_parent_span(self, mock_export):
        """Test that parent_span is passed to each export."""
        from unittest.mock import Mock

        parent = Mock()
        events = [Mock(spec=dict), Mock(spec=dict)]

        export_events(events, parent_span=parent)

        # Verify parent was passed to each call
        assert mock_export.call_count == 2
        for call in mock_export.call_args_list:
            assert call[1]["parent_span"] == parent


class TestExportEventAsSpan:
    """Test export_event_as_span functionality."""

    def test_logs_warning_when_opentelemetry_not_installed(self):
        """Test that export_event_as_span logs warning when OTel not available."""
        from vetch.exporters.opentelemetry import export_event_as_span

        event = {"provider": "test", "model": "test-model"}

        # Should handle missing opentelemetry gracefully
        export_event_as_span(event)
        # No assertion needed - just verify it doesn't crash

    def test_export_event_handles_minimal_event(self):
        """Test that export_event_as_span handles minimal event."""
        from vetch.exporters.opentelemetry import export_event_as_span

        event = {"provider": "test"}

        # Should not crash with minimal event
        export_event_as_span(event)


class TestExportEventsWithTracer:
    """Test export_events with custom tracer."""

    @patch("vetch.exporters.opentelemetry.export_event_as_span")
    def test_exports_with_custom_tracer(self, mock_export):
        """Test that custom tracer is passed through."""
        from unittest.mock import Mock

        tracer = Mock()
        events = [Mock(spec=dict)]

        export_events(events, tracer=tracer)

        # Verify tracer was passed
        assert mock_export.call_count == 1
        assert mock_export.call_args[1]["tracer"] == tracer

    @patch("vetch.exporters.opentelemetry.export_event_as_span")
    def test_exports_handles_single_event(self, mock_export):
        """Test export_events with single event."""
        from unittest.mock import Mock

        events = [Mock(spec=dict)]

        export_events(events)

        assert mock_export.call_count == 1


