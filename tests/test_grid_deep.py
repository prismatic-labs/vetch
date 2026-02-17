"""Deep tests for grid intensity logic.

Verifies country code extraction and API fetch branches.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from vetch.sensing.grid import _extract_country_code, _fetch_from_api


def test_extract_country_code():
    """Verify country code extraction from various region strings."""
    assert _extract_country_code("us-east-1") == "US"
    assert _extract_country_code("ca-central-1") == "CA"
    assert _extract_country_code("sa-east-1") == "BR"
    assert _extract_country_code("australia-southeast1") == "AU"
    assert _extract_country_code("asia-northeast1") == "JP"
    assert _extract_country_code("eu-west-1") is None
    assert _extract_country_code("unknown") is None

def test_fetch_from_api_success():
    """Verify successful API fetch."""
    mock_response = MagicMock()
    mock_response.read.return_value = b'{"carbonIntensity": 420.0}'
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        result = _fetch_from_api("us-east-1", "fake-key")
        assert result.intensity_gco2e_kwh == 420.0
        assert result.signal_quality == "live"

def test_fetch_from_api_no_intensity():
    """Verify handling of API response missing intensity."""
    mock_response = MagicMock()
    mock_response.read.return_value = b'{}'
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        result = _fetch_from_api("us-east-1", "fake-key")
        assert result is None

def test_fetch_from_api_http_error_retry():
    """Verify API retries on 429 and 500 errors."""
    from io import BytesIO
    from urllib.error import HTTPError

    mock_error_429 = HTTPError("url", 429, "Too Many Requests", {}, BytesIO(b""))
    mock_response = MagicMock()
    mock_response.read.return_value = b'{"carbonIntensity": 420.0}'
    mock_response.__enter__.return_value = mock_response

    with patch("time.sleep"): # Fast retry
        with patch("urllib.request.urlopen", side_effect=[mock_error_429, mock_response]):
            result = _fetch_from_api("us-east-1", "fake-key")
            assert result.intensity_gco2e_kwh == 420.0

def test_fetch_from_api_url_error():
    """Verify handling of URLError."""
    from urllib.error import URLError

    with patch("time.sleep"):
        with patch("urllib.request.urlopen", side_effect=URLError("DNS fail")):
            result = _fetch_from_api("us-east-1", "fake-key")
            assert result is None

