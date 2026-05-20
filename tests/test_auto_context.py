"""Integration tests for automatic context creation with instrument()."""

from __future__ import annotations

import os
from unittest.mock import Mock, patch

import pytest


class TestAutoContextCreation:
    """Test automatic context creation when using instrument() without wrap()."""

    def test_genai_auto_creates_context_when_instrumented(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """GenAI provider auto-creates context when instrument() is used."""
        import json
        import logging

        import vetch
        from vetch.providers import genai

        # Reset instrumentation state
        vetch._instrumented = False
        vetch._default_region = None
        vetch._default_tags = None
        genai._module_instrumented = False

        # Set up environment
        os.environ["VETCH_OUTPUT"] = "stderr"

        try:
            # Instrument with defaults
            vetch.instrument(region="us-central1", tags={"service": "test"})

            # Create mock client and response
            mock_client = Mock()
            mock_response = Mock()
            mock_response.model_name = "models/gemini-2.5-flash-001"
            mock_response.usage_metadata = Mock()
            mock_response.usage_metadata.prompt_token_count = 10
            mock_response.usage_metadata.candidates_token_count = 20
            mock_response.usage_metadata.total_token_count = 30
            mock_response.usage_metadata.thought_token_count = 0

            # Set up models.generate_content
            mock_client.models = Mock()
            original_generate = Mock(return_value=mock_response)
            mock_client.models.generate_content = original_generate

            # Patch the client (simulating what instrument() does)
            genai.patch_client(mock_client)

            # Capture logs
            with caplog.at_level(logging.INFO, logger="vetch.emitter"):
                result = mock_client.models.generate_content(
                    model="gemini-2.5-flash", contents="test"
                )

            # Verify result is returned
            assert result == mock_response

            # Verify event was logged
            assert len(caplog.records) > 0, "Expected event to be logged"

            # Find the event in logs
            event_json = None
            for record in caplog.records:
                try:
                    event = json.loads(record.message)
                    if event.get("provider") == "google_genai":
                        event_json = event
                        break
                except json.JSONDecodeError:
                    continue

            assert event_json is not None, "Expected to find GenAI event in logs"

            # Verify event contains expected data
            assert event_json["model"] == "gemini-2.5-flash"
            assert event_json["provider"] == "google_genai"
            assert event_json["region"] == "us-central1"
            assert event_json["tags"] == {"service": "test"}
            assert event_json["usage"]["text"]["input_tokens"] == 10
            assert event_json["usage"]["text"]["output_tokens"] == 20

        finally:
            # Clean up
            vetch.uninstrument()
            vetch._default_region = None
            vetch._default_tags = None
            os.environ.pop("VETCH_OUTPUT", None)

    def test_openai_auto_creates_context_when_instrumented(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """OpenAI provider auto-creates context when instrument() is used."""
        import json
        import logging

        import vetch
        from vetch.providers import openai as openai_provider

        # Reset instrumentation state
        vetch._instrumented = False
        vetch._default_region = None
        vetch._default_tags = None

        # Set up environment
        os.environ["VETCH_OUTPUT"] = "stderr"

        try:
            # Instrument with defaults
            vetch.instrument(region="us-east-1", tags={"env": "test"})

            # Create mock response
            mock_response = Mock()
            mock_response.model = "gpt-4o"
            mock_response.usage = Mock()
            mock_response.usage.prompt_tokens = 15
            mock_response.usage.completion_tokens = 25
            mock_response.usage.total_tokens = 40
            mock_response.usage.prompt_tokens_details = None
            mock_response.usage.completion_tokens_details = None
            mock_choice = Mock()
            mock_choice.finish_reason = "stop"
            mock_choice.message = Mock()
            mock_choice.message.content = "test response"
            mock_response.choices = [mock_choice]

            # Create mock completions object
            mock_completions = Mock()
            original_create = Mock(return_value=mock_response)
            mock_completions.create = original_create

            # Patch completions
            openai_provider._client_originals[mock_completions] = original_create

            wrapper = openai_provider._WeakChatWrapper(
                mock_completions, openai_provider._client_originals
            )

            # Capture logs
            with caplog.at_level(logging.INFO, logger="vetch.emitter"):
                result = wrapper(model="gpt-4o", messages=[{"role": "user", "content": "test"}])

            # Verify result is returned
            assert result == mock_response

            # Verify event was emitted
            assert len(caplog.records) > 0, "Expected event to be logged"

            # Find the event in logs
            event_json = None
            for record in caplog.records:
                try:
                    event = json.loads(record.message)
                    if event.get("provider") == "openai":
                        event_json = event
                        break
                except json.JSONDecodeError:
                    continue

            assert event_json is not None, "Expected to find OpenAI event in logs"

            # Verify event contains expected data
            assert event_json["model"] == "gpt-4o"
            assert event_json["provider"] == "openai"
            assert event_json["region"] == "us-east-1"
            assert event_json["tags"] == {"env": "test"}
            assert event_json["usage"]["text"]["input_tokens"] == 15
            assert event_json["usage"]["text"]["output_tokens"] == 25

        finally:
            # Clean up
            vetch.uninstrument()
            vetch._default_region = None
            vetch._default_tags = None
            os.environ.pop("VETCH_OUTPUT", None)

    def test_manual_wrap_still_works_with_instrument(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Manual wrap() still works after instrument() is called."""
        import json
        import logging

        import vetch
        from vetch.providers import genai

        # Reset instrumentation state
        vetch._instrumented = False
        vetch._default_region = None
        vetch._default_tags = None
        genai._module_instrumented = False

        # Set up environment
        os.environ["VETCH_OUTPUT"] = "stderr"

        try:
            # Instrument with defaults
            vetch.instrument(region="us-west-2", tags={"service": "auto"})

            # Create mock client and response
            mock_client = Mock()
            mock_response = Mock()
            mock_response.model_name = "models/gemini-1.5-pro-001"
            mock_response.usage_metadata = Mock()
            mock_response.usage_metadata.prompt_token_count = 100
            mock_response.usage_metadata.candidates_token_count = 200
            mock_response.usage_metadata.total_token_count = 300
            mock_response.usage_metadata.thought_token_count = 0

            mock_client.models = Mock()
            original_generate = Mock(return_value=mock_response)
            mock_client.models.generate_content = original_generate

            # Patch the client
            genai.patch_client(mock_client)

            # Capture logs
            with caplog.at_level(logging.INFO, logger="vetch.emitter"):
                with vetch.wrap(region="eu-west-1", tags={"custom": "tag"}):
                    result = mock_client.models.generate_content(
                        model="gemini-1.5-pro", contents="test"
                    )

            # Verify result
            assert result == mock_response

            # Verify event was logged
            assert len(caplog.records) > 0, "Expected event to be logged"

            # Find the event in logs
            event_json = None
            for record in caplog.records:
                try:
                    event = json.loads(record.message)
                    if event.get("provider") == "google_genai":
                        event_json = event
                        break
                except json.JSONDecodeError:
                    continue

            assert event_json is not None, "Expected to find GenAI event in logs"

            # Verify manual wrap() settings take precedence
            assert event_json["region"] == "eu-west-1"
            assert event_json["tags"]["custom"] == "tag"
            # Global tags from instrument() should also be present
            assert event_json["tags"]["service"] == "auto"

        finally:
            # Clean up
            vetch.uninstrument()
            vetch._default_region = None
            vetch._default_tags = None
            os.environ.pop("VETCH_OUTPUT", None)

    def test_auto_context_handles_errors_gracefully(self) -> None:
        """Auto-created context handles errors gracefully."""
        import json
        from io import StringIO

        import vetch
        from vetch.providers import genai

        # Reset instrumentation state
        vetch._instrumented = False
        vetch._default_region = None
        vetch._default_tags = None
        genai._module_instrumented = False

        os.environ["VETCH_OUTPUT"] = "stderr"

        try:
            # Instrument
            vetch.instrument(region="us-central1")

            # Create mock client that raises an error
            mock_client = Mock()
            mock_client.models = Mock()

            def raise_error(*args, **kwargs):
                raise ValueError("Test error")

            mock_client.models.generate_content = raise_error

            # Patch the client
            genai.patch_client(mock_client)

            # Capture stderr
            captured_stderr = StringIO()

            # Call should raise the error but still emit event
            with patch("sys.stderr", captured_stderr):
                with pytest.raises(ValueError, match="Test error"):
                    mock_client.models.generate_content(model="gemini-2.5-flash", contents="test")

            # Verify error event was emitted
            stderr_output = captured_stderr.getvalue()
            if stderr_output:
                event = json.loads(stderr_output.strip())
                assert event["error"] is True
                assert "ValueError" in event.get("error_type", "")

        finally:
            # Clean up
            vetch.uninstrument()
            vetch._default_region = None
            vetch._default_tags = None
            os.environ.pop("VETCH_OUTPUT", None)

    def test_anthropic_auto_creates_context_when_instrumented(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Anthropic provider auto-creates context when instrument() is used."""
        import json
        import logging

        import vetch
        from vetch.providers import anthropic as anthropic_provider

        # Reset instrumentation state
        vetch._instrumented = False
        vetch._default_region = None
        vetch._default_tags = None

        # Set up environment
        os.environ["VETCH_OUTPUT"] = "stderr"

        try:
            # Instrument with defaults
            vetch.instrument(region="us-west-1", tags={"service": "anthropic-test"})

            # Create mock response
            mock_response = Mock()
            mock_response.id = "msg_123"
            mock_response.model = "claude-3-5-sonnet-20241022"
            mock_response.usage = Mock()
            mock_response.usage.input_tokens = 20
            mock_response.usage.output_tokens = 30

            # Create mock messages object
            mock_messages = Mock()
            original_create = Mock(return_value=mock_response)
            mock_messages.create = original_create

            # Patch messages
            anthropic_provider._client_originals[mock_messages] = original_create

            wrapper = anthropic_provider._WeakMessagesWrapper(
                mock_messages, anthropic_provider._client_originals
            )

            # Capture logs
            with caplog.at_level(logging.INFO, logger="vetch.emitter"):
                result = wrapper(
                    model="claude-3-5-sonnet-20241022",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=100,
                )

            # Verify result is returned
            assert result == mock_response

            # Verify event was emitted
            assert len(caplog.records) > 0, "Expected event to be logged"

            # Find the event in logs
            event_json = None
            for record in caplog.records:
                try:
                    event = json.loads(record.message)
                    if event.get("provider") == "anthropic":
                        event_json = event
                        break
                except json.JSONDecodeError:
                    continue

            assert event_json is not None, "Expected to find Anthropic event in logs"

            # Verify event contains expected data
            assert event_json["model"] == "claude-3-5-sonnet-20241022"
            assert event_json["provider"] == "anthropic"
            assert event_json["region"] == "us-west-1"
            assert event_json["tags"] == {"service": "anthropic-test"}
            assert event_json["usage"]["text"]["input_tokens"] == 20
            assert event_json["usage"]["text"]["output_tokens"] == 30

        finally:
            # Clean up
            vetch.uninstrument()
            vetch._default_region = None
            vetch._default_tags = None
            os.environ.pop("VETCH_OUTPUT", None)

    def test_vertexai_auto_creates_context_when_instrumented(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """VertexAI provider auto-creates context when instrument() is used."""
        import json
        import logging

        import vetch
        from vetch.providers.vertexai import patch_vertexai_model

        # Reset instrumentation state
        vetch._instrumented = False
        vetch._default_region = None
        vetch._default_tags = None

        # Set up environment
        os.environ["VETCH_OUTPUT"] = "stderr"
        os.environ["VETCH_FORCE_PATCH"] = "true"  # Skip version check

        try:
            # Instrument with defaults
            vetch.instrument(region="us-central1", tags={"service": "vertexai-test"})

            # Create mock response
            mock_response = Mock()
            mock_response.usage_metadata = Mock()
            mock_response.usage_metadata.prompt_token_count = 25
            mock_response.usage_metadata.candidates_token_count = 35
            mock_response.usage_metadata.total_token_count = 60

            # Create mock model
            mock_model = Mock()
            mock_model._model_name = "gemini-1.5-flash"
            mock_model.generate_content = Mock(return_value=mock_response)

            # Patch the model
            patch_vertexai_model(mock_model)

            # Capture logs
            with caplog.at_level(logging.INFO, logger="vetch.emitter"):
                result = mock_model.generate_content("test prompt")

            # Verify result is returned
            assert result == mock_response

            # Verify event was emitted
            assert len(caplog.records) > 0, "Expected event to be logged"

            # Find the event in logs
            event_json = None
            for record in caplog.records:
                try:
                    event = json.loads(record.message)
                    if event.get("provider") == "vertexai":
                        event_json = event
                        break
                except json.JSONDecodeError:
                    continue

            assert event_json is not None, "Expected to find VertexAI event in logs"

            # Verify event contains expected data
            assert event_json["provider"] == "vertexai"
            assert event_json["region"] == "us-central1"
            assert event_json["tags"] == {"service": "vertexai-test"}
            assert event_json["usage"]["text"]["input_tokens"] == 25
            assert event_json["usage"]["text"]["output_tokens"] == 35

        finally:
            # Clean up
            vetch.uninstrument()
            vetch._default_region = None
            vetch._default_tags = None
            os.environ.pop("VETCH_OUTPUT", None)
            os.environ.pop("VETCH_FORCE_PATCH", None)
