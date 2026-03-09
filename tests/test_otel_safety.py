"""Tests for otel.py safety systems - OTLP export, background worker, backpressure.

These tests verify the critical safety systems that prevent Vetch from
crashing production servers:
- Background worker thread lifecycle and graceful shutdown
- Queue backpressure and dropped event handling
- Error rate limiting (circuit breaker for logging)
- OTLP configuration and error handling
"""

from __future__ import annotations

import queue
import threading
import time
from unittest.mock import MagicMock, Mock, patch

from vetch import otel


class TestOTLPConfiguration:
    """Test OTLP configuration and setup."""

    def setup_method(self):
        """Reset module state before each test."""
        # Reset global state
        otel._otlp_configured = False
        otel._tracer = None
        otel._meter = None
        otel._energy_histogram = None
        otel._carbon_histogram = None
        otel._cost_histogram = None
        otel._request_counter = None

    @patch("vetch.otel.logger")
    def test_configure_without_opentelemetry_installed(self, mock_logger):
        """configure_otlp_export handles missing OpenTelemetry SDK gracefully."""
        with patch.dict("sys.modules", {"opentelemetry": None}):
            with patch("vetch.otel.os.environ.get", return_value="http://localhost:4317"):
                # Should fail gracefully with ImportError
                result = otel.configure_otlp_export()

                assert result is False
                assert not otel.is_otlp_configured()
                # Should log warning about missing SDK
                mock_logger.warning.assert_called()
                assert "OpenTelemetry SDK not installed" in str(mock_logger.warning.call_args)

    @patch("vetch.otel._start_export_worker")
    @patch("vetch.otel.logger")
    def test_configure_without_endpoint(self, mock_logger, mock_start_worker):
        """configure_otlp_export returns False when no endpoint specified."""
        # Mock OpenTelemetry imports to get past ImportError
        with patch.dict("sys.modules", {
            "opentelemetry": MagicMock(),
            "opentelemetry.metrics": MagicMock(),
            "opentelemetry.trace": MagicMock(),
            "opentelemetry.exporter.otlp.proto.grpc.metric_exporter": MagicMock(),
            "opentelemetry.exporter.otlp.proto.grpc.trace_exporter": MagicMock(),
            "opentelemetry.sdk.metrics": MagicMock(),
            "opentelemetry.sdk.metrics.export": MagicMock(),
            "opentelemetry.sdk.resources": MagicMock(),
            "opentelemetry.sdk.trace": MagicMock(),
            "opentelemetry.sdk.trace.export": MagicMock(),
        }):
            with patch("vetch.otel.os.environ.get", return_value=None):
                result = otel.configure_otlp_export()

                assert result is False
                assert not otel.is_otlp_configured()
                mock_logger.warning.assert_called_with(
                    "No OTLP endpoint specified. Set OTEL_EXPORTER_OTLP_ENDPOINT."
                )

    @patch("vetch.otel._start_export_worker")
    @patch("vetch.otel.logger")
    def test_configure_with_endpoint_from_env(self, mock_logger, mock_start_worker):
        """configure_otlp_export reads endpoint from environment variable."""
        # Mock OpenTelemetry imports
        with patch.dict("sys.modules", {
            "opentelemetry": MagicMock(),
            "opentelemetry.metrics": MagicMock(),
            "opentelemetry.trace": MagicMock(),
            "opentelemetry.exporter.otlp.proto.grpc.metric_exporter": MagicMock(),
            "opentelemetry.exporter.otlp.proto.grpc.trace_exporter": MagicMock(),
            "opentelemetry.sdk.metrics": MagicMock(),
            "opentelemetry.sdk.metrics.export": MagicMock(),
            "opentelemetry.sdk.resources": MagicMock(),
            "opentelemetry.sdk.trace": MagicMock(),
            "opentelemetry.sdk.trace.export": MagicMock(),
        }):
            with patch("vetch.otel.os.environ.get") as mock_env:
                mock_env.side_effect = lambda key, default="": {
                    "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4317",
                    "OTEL_EXPORTER_OTLP_HEADERS": "",
                }.get(key, default)

                result = otel.configure_otlp_export()

                # Should succeed
                assert result is True
                assert otel.is_otlp_configured()
                mock_start_worker.assert_called_once()
                mock_logger.info.assert_called()

    @patch("vetch.otel._start_export_worker")
    def test_configure_parses_headers_from_env(self, mock_start_worker):
        """configure_otlp_export parses OTEL_EXPORTER_OTLP_HEADERS."""
        with patch.dict("sys.modules", {
            "opentelemetry": MagicMock(),
            "opentelemetry.metrics": MagicMock(),
            "opentelemetry.trace": MagicMock(),
            "opentelemetry.exporter.otlp.proto.grpc.metric_exporter": MagicMock(),
            "opentelemetry.exporter.otlp.proto.grpc.trace_exporter": MagicMock(),
            "opentelemetry.sdk.metrics": MagicMock(),
            "opentelemetry.sdk.metrics.export": MagicMock(),
            "opentelemetry.sdk.resources": MagicMock(),
            "opentelemetry.sdk.trace": MagicMock(),
            "opentelemetry.sdk.trace.export": MagicMock(),
        }):
            with patch("vetch.otel.os.environ.get") as mock_env:
                mock_env.side_effect = lambda key, default="": {
                    "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4317",
                    "OTEL_EXPORTER_OTLP_HEADERS": "x-api-key=secret123,x-tenant=prod",
                }.get(key, default)

                result = otel.configure_otlp_export()

                # Should succeed and parse headers
                assert result is True


class TestBackgroundWorkerLifecycle:
    """Test background worker thread lifecycle."""

    def setup_method(self):
        """Reset state before each test."""
        otel._otlp_configured = False
        otel._export_thread = None
        otel._shutdown_event.clear()
        # Clear queue
        while not otel._export_queue.empty():
            try:
                otel._export_queue.get_nowait()
                otel._export_queue.task_done()
            except queue.Empty:
                break

    @patch("vetch.otel._export_event_sync")
    def test_start_worker_creates_thread(self, mock_export):
        """_start_export_worker creates and starts a daemon thread."""
        otel._start_export_worker()

        assert otel._export_thread is not None
        assert otel._export_thread.is_alive()
        assert otel._export_thread.daemon is True
        assert otel._export_thread.name == "vetch-otlp-export"

        # Cleanup
        otel._shutdown_export_worker()

    @patch("vetch.otel._export_event_sync")
    def test_start_worker_is_idempotent(self, mock_export):
        """_start_export_worker doesn't create multiple threads."""
        otel._start_export_worker()
        first_thread = otel._export_thread

        otel._start_export_worker()
        second_thread = otel._export_thread

        # Should be the same thread
        assert first_thread is second_thread

        # Cleanup
        otel._shutdown_export_worker()

    @patch("vetch.otel._export_event_sync")
    def test_shutdown_worker_drains_queue(self, mock_export):
        """_shutdown_export_worker drains queue before stopping."""
        otel._otlp_configured = True
        otel._start_export_worker()

        # Add events to queue
        event1 = {"model": "gpt-4o", "provider": "openai"}
        event2 = {"model": "claude-3", "provider": "anthropic"}
        otel._export_queue.put(event1)
        otel._export_queue.put(event2)

        # Shutdown
        otel._shutdown_export_worker()

        # Queue should be drained (or nearly so)
        assert otel._export_queue.qsize() <= 1  # Sentinel may still be there

        # Worker should be stopped
        assert otel._export_thread is None or not otel._export_thread.is_alive()

    @patch("vetch.otel._export_event_sync")
    def test_shutdown_worker_handles_timeout(self, mock_export):
        """_shutdown_export_worker has a 5-second timeout to prevent hanging."""
        # Make export slow to simulate hung OTLP collector
        mock_export.side_effect = lambda event: time.sleep(0.1)

        otel._otlp_configured = True
        otel._start_export_worker()

        # Fill queue with events
        for i in range(10):
            otel._export_queue.put({"model": f"model-{i}"})

        # Shutdown should complete within reasonable time (< 10 seconds)
        start = time.time()
        otel._shutdown_export_worker()
        duration = time.time() - start

        # Should timeout and not wait forever
        assert duration < 10.0


class TestQueueBackpressure:
    """Test queue backpressure and dropped event handling."""

    def setup_method(self):
        """Reset state before each test."""
        otel._otlp_configured = True
        otel._dropped_events_count = 0
        otel._last_drop_warning = 0.0
        # Clear queue
        while not otel._export_queue.empty():
            try:
                otel._export_queue.get_nowait()
                otel._export_queue.task_done()
            except queue.Empty:
                break

    def teardown_method(self):
        """Cleanup after test."""
        otel._otlp_configured = False
        otel._dropped_events_count = 0

    @patch("vetch.otel.logger")
    def test_export_event_queues_successfully(self, mock_logger):
        """export_event_otlp queues event when queue not full."""
        event = {"model": "gpt-4o", "provider": "openai"}

        result = otel.export_event_otlp(event)

        assert result is True
        assert otel._export_queue.qsize() == 1
        mock_logger.warning.assert_not_called()

    @patch("vetch.otel.logger")
    def test_export_event_drops_when_queue_full(self, mock_logger):
        """export_event_otlp drops event when queue is full."""
        # Fill queue to capacity
        for i in range(otel._export_queue.maxsize):
            otel._export_queue.put({"model": f"model-{i}"})

        # Next event should be dropped
        event = {"model": "dropped", "provider": "test"}
        result = otel.export_event_otlp(event)

        assert result is False
        assert otel._dropped_events_count == 1

        # Cleanup
        while not otel._export_queue.empty():
            try:
                otel._export_queue.get_nowait()
                otel._export_queue.task_done()
            except queue.Empty:
                break

    @patch("vetch.otel.logger")
    @patch("time.monotonic")
    def test_dropped_events_rate_limited_logging(self, mock_monotonic, mock_logger):
        """Dropped events are logged with rate limiting."""
        # Mock monotonic time - start at a reasonable value
        mock_monotonic.return_value = 1000.0

        # Fill queue
        for i in range(otel._export_queue.maxsize):
            otel._export_queue.put({"model": f"model-{i}"})

        # Drop 1st event - should warn (1 % 1000 == 1 AND time diff > 60)
        event = {"model": "dropped"}
        otel.export_event_otlp(event)
        assert mock_logger.warning.call_count == 1

        # Drop more events without advancing time
        for _i in range(999):
            otel.export_event_otlp(event)

        # Should not warn again until 1000th drop
        # But 1000 % 1000 == 0, so next warning is at 1001
        assert mock_logger.warning.call_count == 1

        # Drop 1001st event with time advanced (> 60 seconds)
        mock_monotonic.return_value = 1061.0  # 61 seconds later
        otel.export_event_otlp(event)
        assert mock_logger.warning.call_count == 2

        # Cleanup
        while not otel._export_queue.empty():
            try:
                otel._export_queue.get_nowait()
                otel._export_queue.task_done()
            except queue.Empty:
                break

    def test_export_event_returns_false_when_not_configured(self):
        """export_event_otlp returns False when OTLP not configured."""
        otel._otlp_configured = False

        event = {"model": "gpt-4o"}
        result = otel.export_event_otlp(event)

        assert result is False
        assert otel._export_queue.qsize() == 0  # Not queued


class TestErrorRateLimiting:
    """Test error rate limiting (circuit breaker for logging)."""

    def setup_method(self):
        """Reset state before each test."""
        otel._last_error_log_time = 0.0
        otel._error_count_since_log = 0

    @patch("vetch.otel.logger")
    @patch("time.time")
    def test_log_error_rate_limited_first_error(self, mock_time, mock_logger):
        """First error is logged when enough time has passed."""
        # Start at a time where the interval has elapsed
        mock_time.return_value = 1000.0

        error = Exception("OTLP export failed")
        otel._log_error_rate_limited(error)

        assert mock_logger.warning.call_count == 1
        assert "OTLP export error:" in str(mock_logger.warning.call_args)

    @patch("vetch.otel.logger")
    @patch("time.time")
    def test_log_error_rate_limited_suppresses_rapid_errors(self, mock_time, mock_logger):
        """Rapid errors within 5 minutes are suppressed."""
        # Start at a time where the interval has elapsed
        mock_time.return_value = 1000.0

        # First error - should log
        otel._log_error_rate_limited(Exception("Error 1"))
        assert mock_logger.warning.call_count == 1

        # More errors within 5 minutes (299 seconds later)
        mock_time.return_value = 1299.0
        for i in range(10):
            otel._log_error_rate_limited(Exception(f"Error {i+2}"))

        # Should still only have logged once
        assert mock_logger.warning.call_count == 1

    @patch("vetch.otel.logger")
    @patch("time.time")
    def test_log_error_rate_limited_logs_after_interval(self, mock_time, mock_logger):
        """Errors after 5 minutes are logged with count."""
        # Start at a reasonable time
        mock_time.return_value = 1000.0

        # First error - should log
        otel._log_error_rate_limited(Exception("Error 1"))
        assert mock_logger.warning.call_count == 1

        # More errors within interval
        mock_time.return_value = 1100.0
        for i in range(5):
            otel._log_error_rate_limited(Exception(f"Error {i+2}"))

        # Should still only have 1 log
        assert mock_logger.warning.call_count == 1

        # Error after 5 minutes (300 seconds from first log)
        mock_time.return_value = 1301.0
        otel._log_error_rate_limited(Exception("Error 7"))

        # Should log again with count
        assert mock_logger.warning.call_count == 2
        assert "6 errors" in str(mock_logger.warning.call_args)

    @patch("vetch.otel.logger")
    @patch("time.time")
    def test_log_error_rate_limited_thread_safe(self, mock_time, mock_logger):
        """Error rate limiting is thread-safe."""
        mock_time.return_value = 0.0

        errors = []

        def log_errors():
            for i in range(10):
                try:
                    otel._log_error_rate_limited(Exception(f"Error {i}"))
                except Exception as e:
                    errors.append(e)

        # Run from multiple threads
        threads = [threading.Thread(target=log_errors) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not raise any errors
        assert len(errors) == 0
        # Should have rate-limited (not 50 logs)
        assert mock_logger.warning.call_count < 10


class TestStatsAPI:
    """Test get_otlp_stats() observability API."""

    def setup_method(self):
        """Reset state before each test."""
        otel._otlp_configured = False
        otel._dropped_events_count = 0
        while not otel._export_queue.empty():
            try:
                otel._export_queue.get_nowait()
                otel._export_queue.task_done()
            except queue.Empty:
                break

    def test_get_otlp_stats_structure(self):
        """get_otlp_stats returns correct structure."""
        stats = otel.get_otlp_stats()

        assert "queue_size" in stats
        assert "queue_current" in stats
        assert "dropped_events" in stats
        assert "configured" in stats

    def test_get_otlp_stats_reflects_queue_size(self):
        """get_otlp_stats reflects configured queue size."""
        stats = otel.get_otlp_stats()

        assert stats["queue_size"] == otel._export_queue.maxsize

    def test_get_otlp_stats_reflects_current_queue(self):
        """get_otlp_stats reflects current queue depth."""
        # Add some events
        for i in range(5):
            otel._export_queue.put({"model": f"model-{i}"})

        stats = otel.get_otlp_stats()

        assert stats["queue_current"] == 5

        # Cleanup
        while not otel._export_queue.empty():
            try:
                otel._export_queue.get_nowait()
                otel._export_queue.task_done()
            except queue.Empty:
                break

    def test_get_otlp_stats_reflects_dropped_events(self):
        """get_otlp_stats reflects dropped event count."""
        otel._dropped_events_count = 42

        stats = otel.get_otlp_stats()

        assert stats["dropped_events"] == 42

    def test_get_otlp_stats_reflects_configured_state(self):
        """get_otlp_stats reflects configured state."""
        otel._otlp_configured = False
        stats = otel.get_otlp_stats()
        assert stats["configured"] is False

        otel._otlp_configured = True
        stats = otel.get_otlp_stats()
        assert stats["configured"] is True


class TestExportEventSync:
    """Test _export_event_sync internal function."""

    def setup_method(self):
        """Reset state before each test."""
        otel._otlp_configured = False
        otel._tracer = None

    def test_export_event_sync_returns_false_when_not_configured(self):
        """_export_event_sync returns False when not configured."""
        event = {"model": "gpt-4o", "provider": "openai"}

        result = otel._export_event_sync(event)

        assert result is False

    @patch("vetch.otel.logger")
    def test_export_event_sync_handles_exceptions(self, mock_logger):
        """_export_event_sync handles exceptions gracefully."""
        # Configure but with invalid tracer
        otel._otlp_configured = True
        otel._tracer = Mock()
        otel._tracer.start_as_current_span.side_effect = Exception("OTLP error")

        event = {"model": "gpt-4o"}

        result = otel._export_event_sync(event)

        # Should return False and log debug
        assert result is False
        mock_logger.debug.assert_called()


class TestAutoConfiguration:
    """Test auto-configuration from environment variables."""

    @patch("vetch.otel.configure_otlp_export")
    @patch("vetch.otel.os.environ.get")
    def test_auto_configure_when_enabled(self, mock_env, mock_configure):
        """_auto_configure calls configure_otlp_export when VETCH_OTEL_EXPORT=true."""
        mock_env.side_effect = lambda key, default="": {
            "VETCH_OTEL_EXPORT": "true",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4317",
            "VETCH_OTEL_SERVICE_NAME": "my-service",
        }.get(key, default)

        otel._auto_configure()

        mock_configure.assert_called_once_with(service_name="my-service")

    @patch("vetch.otel.configure_otlp_export")
    @patch("vetch.otel.os.environ.get")
    def test_auto_configure_when_disabled(self, mock_env, mock_configure):
        """_auto_configure does not call configure_otlp_export when disabled."""
        mock_env.side_effect = lambda key, default="": {
            "VETCH_OTEL_EXPORT": "false",
        }.get(key, default)

        otel._auto_configure()

        mock_configure.assert_not_called()
