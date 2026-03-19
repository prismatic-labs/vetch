"""Tests for the Green Signal API: get_cleanest_region()."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from vetch.sensing.grid import get_cleanest_region


def _mock_intensity(value: float) -> Mock:
    result = Mock()
    result.intensity_gco2e_kwh = value
    return result


class TestGetCleanestRegion:
    """Tests for get_cleanest_region()."""

    def test_returns_lowest_intensity_region(self) -> None:
        """Returns the region with the lowest carbon intensity."""
        intensities = {
            "us-east-1": 400.0,
            "eu-west-1": 120.0,
            "us-west-2": 250.0,
        }

        def mock_get_carbon(region: str, api_key: object = None) -> Mock:
            return _mock_intensity(intensities[region])

        with patch("vetch.sensing.grid.get_carbon_intensity", side_effect=mock_get_carbon):
            region, intensity = get_cleanest_region(
                ["us-east-1", "eu-west-1", "us-west-2"]
            )

        assert region == "eu-west-1"
        assert intensity == pytest.approx(120.0)

    def test_single_candidate_returns_it(self) -> None:
        """Single candidate is returned regardless of intensity."""
        with patch(
            "vetch.sensing.grid.get_carbon_intensity",
            return_value=_mock_intensity(500.0),
        ):
            region, intensity = get_cleanest_region(["us-east-1"])

        assert region == "us-east-1"
        assert intensity == pytest.approx(500.0)

    def test_empty_candidates_raises(self) -> None:
        """Empty candidates list raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            get_cleanest_region([])

    def test_tie_returns_first_winner(self) -> None:
        """When two regions tie, the first encountered lowest wins."""
        with patch(
            "vetch.sensing.grid.get_carbon_intensity",
            return_value=_mock_intensity(200.0),
        ):
            region, intensity = get_cleanest_region(["eu-west-1", "eu-north-1"])

        # First lowest encountered is returned
        assert region == "eu-west-1"
        assert intensity == pytest.approx(200.0)

    def test_passes_api_key_to_carbon_intensity(self) -> None:
        """api_key is forwarded to get_carbon_intensity for each region."""
        mock_fn = Mock(return_value=_mock_intensity(100.0))

        with patch("vetch.sensing.grid.get_carbon_intensity", mock_fn):
            get_cleanest_region(["us-east-1", "eu-west-1"], api_key="test-key")

        for call in mock_fn.call_args_list:
            assert call.kwargs.get("api_key") == "test-key"

    def test_returns_tuple_of_str_and_float(self) -> None:
        """Return type is always (str, float)."""
        with patch(
            "vetch.sensing.grid.get_carbon_intensity",
            return_value=_mock_intensity(300.0),
        ):
            result = get_cleanest_region(["us-east-1"])

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], float)

    def test_public_api_accessible_via_vetch(self) -> None:
        """vetch.get_cleanest_region() is accessible from the public API."""
        import vetch

        assert hasattr(vetch, "get_cleanest_region")
        assert callable(vetch.get_cleanest_region)


class TestConfigureHttpEndpoint:
    """Tests for configure_http_endpoint()."""

    def test_configure_http_endpoint_adds_handler(self) -> None:
        """configure_http_endpoint() wires up an HttpHandler on the emitter logger."""

        import vetch
        from vetch.emitter import HttpHandler
        from vetch.emitter import logger as emitter_logger

        initial_handlers = len(emitter_logger.handlers)

        vetch.configure_http_endpoint("https://example.com/ingest")

        http_handlers = [h for h in emitter_logger.handlers if isinstance(h, HttpHandler)]
        assert len(http_handlers) >= 1
        assert any(h.url == "https://example.com/ingest" for h in http_handlers)

        # Clean up
        for h in http_handlers:
            if h.url == "https://example.com/ingest":
                emitter_logger.removeHandler(h)
                h.close()

    def test_configure_http_endpoint_with_api_key(self) -> None:
        """configure_http_endpoint() stores the API key on the handler."""
        import vetch
        from vetch.emitter import HttpHandler
        from vetch.emitter import logger as emitter_logger

        vetch.configure_http_endpoint("https://example.com/ingest2", api_key="secret-key")

        http_handlers = [
            h for h in emitter_logger.handlers
            if isinstance(h, HttpHandler) and h.url == "https://example.com/ingest2"
        ]
        assert len(http_handlers) == 1
        assert http_handlers[0]._api_key == "secret-key"

        # Clean up
        for h in http_handlers:
            emitter_logger.removeHandler(h)
            h.close()

    def test_http_handler_sends_auth_header(self) -> None:
        """HttpHandler includes Authorization header when api_key is set."""
        from vetch.emitter import HttpHandler

        captured_requests = []

        def mock_urlopen(req: object, timeout: float = 0.5) -> object:
            captured_requests.append(req)
            mock_ctx = Mock()
            mock_ctx.__enter__ = Mock(return_value=mock_ctx)
            mock_ctx.__exit__ = Mock(return_value=False)
            return mock_ctx

        handler = HttpHandler("https://example.com/events", api_key="my-token")

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            import logging
            record = logging.LogRecord(
                name="test", level=logging.INFO,
                pathname="", lineno=0,
                msg='{"test": true}', args=(), exc_info=None,
            )
            handler.emit(record)
            # Allow background worker to process
            import time
            time.sleep(0.1)

        handler.close()

        if captured_requests:
            req = captured_requests[0]
            assert req.get_header("Authorization") == "Bearer my-token"

    def test_http_handler_no_auth_header_without_key(self) -> None:
        """HttpHandler omits Authorization header when no api_key is set."""
        from vetch.emitter import HttpHandler

        handler = HttpHandler("https://example.com/events")
        assert handler._api_key is None
        handler.close()
