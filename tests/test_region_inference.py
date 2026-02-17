"""Tests for region inference logic.

Verifies that regions are correctly inferred from:
1. Environment variables (VETCH_REGION, AWS_REGION, etc.)
2. Provider-specific URLs (Azure OpenAI, Vertex AI)
3. Fallback priority order
"""

from __future__ import annotations

import os
from unittest.mock import patch

from vetch.providers.openai import infer_region_from_base_url
from vetch.providers.vertexai import infer_region_from_endpoint
from vetch.wrapper import _infer_region


class TestEnvironmentRegionInference:
    """Tests for inferring region from environment variables."""

    def test_infer_vetch_region_priority(self) -> None:
        """VETCH_REGION should have highest priority."""
        env = {
            "VETCH_REGION": "us-east-1",
            "AWS_REGION": "us-west-2",
            "GOOGLE_CLOUD_REGION": "europe-west1",
        }
        with patch.dict(os.environ, env, clear=True):
            region, warning = _infer_region()
            assert region == "us-east-1"
            assert warning is None  # No warning for explicit env var

    def test_infer_aws_region(self) -> None:
        """Infer from AWS environment variables."""
        with patch.dict(os.environ, {"AWS_REGION": "us-west-2"}, clear=True):
            region, warning = _infer_region()
            assert region == "us-west-2"
            assert warning is None

        with patch.dict(os.environ, {"AWS_DEFAULT_REGION": "us-west-2"}, clear=True):
            region, warning = _infer_region()
            assert region == "us-west-2"
            assert warning is None

    def test_infer_gcp_region(self) -> None:
        """Infer from Google Cloud environment variables."""
        with patch.dict(os.environ, {"GOOGLE_CLOUD_REGION": "asia-northeast1"}, clear=True):
            region, warning = _infer_region()
            assert region == "asia-northeast1"
            assert warning is None

    def test_infer_azure_region(self) -> None:
        """Infer from Azure environment variables."""
        with patch.dict(os.environ, {"AZURE_REGION": "uk-south"}, clear=True):
            region, warning = _infer_region()
            assert region == "uk-south"
            assert warning is None

    def test_infer_none_available(self) -> None:
        """Return None if no environment variables are set and timezone fails."""
        # Note: With timezone heuristic, this may return a region based on local TZ
        # We test that at minimum it doesn't crash
        with patch.dict(os.environ, {}, clear=True):
            region, warning = _infer_region()
            # Either None or a timezone-inferred region with warning
            if region is not None:
                assert warning is not None  # Timezone inference includes warning
                assert "inferred from timezone" in warning
            else:
                assert warning is None

    def test_timezone_inference_includes_warning(self) -> None:
        """Timezone-based inference should include accuracy warning."""
        # Force timezone inference by clearing all env vars
        with patch.dict(os.environ, {}, clear=True):
            # Reset the warning flag for this test
            import vetch.wrapper
            original = vetch.wrapper._timezone_warning_issued
            vetch.wrapper._timezone_warning_issued = False
            try:
                region, warning = _infer_region()
                if region is not None:
                    # If we got a region from timezone, warning should be set
                    assert warning is not None
                    assert "Accuracy ~30%" in warning
                    assert "VETCH_REGION" in warning
            finally:
                vetch.wrapper._timezone_warning_issued = original


class TestProviderUrlInference:
    """Tests for inferring region from provider base URLs."""

    def test_openai_azure_subdomain(self) -> None:
        """Infer Azure OpenAI region from subdomain."""
        url = "https://eastus.api.cognitive.microsoft.com/openai/deployments/..."
        assert infer_region_from_base_url(url) == "eastus"

    def test_openai_azure_resource_name(self) -> None:
        """Infer Azure OpenAI region from resource-style URL."""
        url = "https://my-resource.openai.azure.com/"
        assert infer_region_from_base_url(url) == "my-resource"

    def test_openai_standard_url(self) -> None:
        """Standard OpenAI URL should return None."""
        url = "https://api.openai.com/v1"
        assert infer_region_from_base_url(url) is None

    def test_vertex_ai_endpoint(self) -> None:
        """Infer Vertex AI region from endpoint."""
        endpoint = "us-central1-aiplatform.googleapis.com"
        assert infer_region_from_endpoint(endpoint) == "us-central1"

    def test_vertex_ai_europe(self) -> None:
        """Infer Vertex AI European region."""
        endpoint = "europe-west4-aiplatform.googleapis.com"
        assert infer_region_from_endpoint(endpoint) == "europe-west4"
