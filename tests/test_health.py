"""Integration tests for health check utilities.

Focus on critical paths: k8s probes work, endpoints don't crash.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestHealthStatus:
    """Test health status reporting."""

    def test_get_health_status_basic(self):
        """Test get_health_status returns valid structure."""
        from vetch.health import get_health_status

        health = get_health_status()

        # Critical: health check returns required fields for k8s probes
        assert "status" in health
        assert "components" in health
        assert "issues" in health
        assert "timestamp" in health
        assert health["status"] in ("healthy", "degraded", "unhealthy")

    def test_health_check_timestamp_format(self):
        """Test health check timestamp is ISO8601 format."""
        from vetch.health import get_health_status

        health = get_health_status()

        # Timestamp should end with Z (UTC)
        assert health["timestamp"].endswith("Z")
        # Should be ISO format (contains T separator)
        assert "T" in health["timestamp"]


    def test_health_check_with_circuit_breaker_open(self):
        """Test health status when circuit breaker is open."""
        from vetch.health import get_health_status

        mock_fetcher = MagicMock()
        mock_fetcher._circuit_open_until = 9999999999.0

        with patch("vetch.registry.remote.get_remote_fetcher", return_value=mock_fetcher):
            with patch("time.monotonic", return_value=0.0):
                health = get_health_status()

                assert "circuit_breaker" in health["components"]
                assert health["components"]["circuit_breaker"]["status"] == "open"
                assert not health["components"]["circuit_breaker"]["healthy"]
                assert "circuit_breaker_open" in health["issues"]

    def test_health_check_with_high_otlp_drops(self):
        """Test health status with high OTLP drop rate."""
        from vetch.health import get_health_status

        mock_stats = {
            "queue_current": 10,
            "queue_size": 100,
            "dropped_events": 1500,
        }

        with patch("vetch.otel.get_otlp_stats", return_value=mock_stats):
            health = get_health_status()

            assert "otlp_queue" in health["components"]
            assert health["components"]["otlp_queue"]["status"] == "degraded"
            assert not health["components"]["otlp_queue"]["healthy"]
            assert "otlp_high_drop_rate" in health["issues"]

    def test_health_check_with_high_queue_utilization(self):
        """Test health status with high queue utilization."""
        from vetch.health import get_health_status

        mock_stats = {
            "queue_current": 95,
            "queue_size": 100,
            "dropped_events": 10,
        }

        with patch("vetch.otel.get_otlp_stats", return_value=mock_stats):
            health = get_health_status()

            assert "otlp_queue" in health["components"]
            assert health["components"]["otlp_queue"]["status"] == "degraded"
            assert "otlp_queue_near_full" in health["issues"]

    def test_health_check_with_unknown_models(self):
        """Test health status with high unknown model count."""
        from vetch.health import get_health_status

        mock_stats = {
            "model_unknown": 150,
            "missing_required_tags": 0,
        }

        with patch("vetch.wrapper.get_tracking_stats", return_value=mock_stats):
            health = get_health_status()

            assert "tracking" in health["components"]
            assert health["components"]["tracking"]["status"] == "degraded"
            assert "high_unknown_model_rate" in health["issues"]

    def test_health_check_with_compliance_violations(self):
        """Test health status with compliance violations."""
        from vetch.health import get_health_status

        mock_stats = {
            "model_unknown": 5,
            "missing_required_tags": 15,
        }

        with patch("vetch.wrapper.get_tracking_stats", return_value=mock_stats):
            health = get_health_status()

            assert "tracking" in health["components"]
            assert health["components"]["tracking"]["status"] == "unhealthy"
            assert not health["components"]["tracking"]["healthy"]
            assert "compliance_violations" in health["issues"]

    def test_health_check_handles_registry_exception(self):
        """Test health check handles registry fetch exception gracefully."""
        from vetch.health import get_health_status

        with patch(
            "vetch.registry.remote.get_remote_fetcher",
            side_effect=Exception("Registry error"),
        ):
            health = get_health_status()

            assert "circuit_breaker" in health["components"]
            assert health["components"]["circuit_breaker"]["status"] == "unknown"
            assert health["components"]["circuit_breaker"]["healthy"]

    def test_health_check_handles_otlp_exception(self):
        """Test health check handles OTLP stats exception gracefully."""
        from vetch.health import get_health_status

        with patch("vetch.otel.get_otlp_stats", side_effect=Exception("OTLP error")):
            health = get_health_status()

            assert "otlp_queue" in health["components"]
            assert health["components"]["otlp_queue"]["status"] == "unknown"
            assert health["components"]["otlp_queue"]["healthy"]

    def test_health_check_handles_tracking_exception(self):
        """Test health check handles tracking stats exception gracefully."""
        from vetch.health import get_health_status

        with patch(
            "vetch.wrapper.get_tracking_stats", side_effect=Exception("Tracking error")
        ):
            health = get_health_status()

            assert "tracking" in health["components"]
            assert health["components"]["tracking"]["status"] == "unknown"
            assert health["components"]["tracking"]["healthy"]

    def test_health_overall_status_degraded(self):
        """Test overall status is degraded when any component degraded."""
        from vetch.health import get_health_status

        mock_stats = {
            "queue_current": 95,
            "queue_size": 100,
            "dropped_events": 10,
        }

        with patch("vetch.otel.get_otlp_stats", return_value=mock_stats):
            health = get_health_status()

            assert health["status"] == "degraded"

    def test_health_overall_status_unhealthy(self):
        """Test overall status is unhealthy when any component unhealthy."""
        from vetch.health import get_health_status

        mock_stats = {
            "model_unknown": 5,
            "missing_required_tags": 15,
        }

        with patch("vetch.wrapper.get_tracking_stats", return_value=mock_stats):
            health = get_health_status()

            assert health["status"] == "unhealthy"


class TestHealthEndpoints:
    """Test health check endpoint handlers."""

    def test_create_health_endpoint_returns_handlers(self):
        """Test create_health_endpoint returns two handlers."""
        from vetch.health import create_health_endpoint

        flask_handler, fastapi_handler = create_health_endpoint()

        # Critical: both handlers are callable
        assert callable(flask_handler)
        assert callable(fastapi_handler)

    def test_flask_handler_without_flask_installed(self):
        """Test Flask handler returns error when Flask not installed."""
        from vetch.health import create_health_endpoint

        flask_handler, _ = create_health_endpoint()

        # Without Flask, should return error tuple
        result = flask_handler()
        assert isinstance(result, tuple)
        assert result[0] == {"error": "Flask not installed"}
        assert result[1] == 500

    @pytest.mark.asyncio
    async def test_fastapi_handler_structure(self):
        """Test FastAPI handler doesn't crash when called."""
        from vetch.health import create_health_endpoint

        _, fastapi_handler = create_health_endpoint()

        # Critical: handler is async callable
        assert callable(fastapi_handler)

        # Calling it should not crash (will return error if FastAPI not installed)
        result = await fastapi_handler()
        assert result is not None
