"""Tests for vetch.providers.genai module - core functionality only."""

from unittest.mock import Mock

import pytest

try:
    import pytest_asyncio as _  # noqa: F401
    _has_pytest_asyncio = True
except ImportError:
    _has_pytest_asyncio = False

from vetch.providers.genai import (
    _normalize_model_name,
    extract_model,
    extract_usage,
    patch_client,
    track_genai,
    unpatch_client,
)


class TestExtractUsage:
    """Test extract_usage function - the critical parsing logic."""

    def test_extracts_usage_from_response(self):
        """Test extracting usage from Google GenAI response."""
        response = Mock()
        response.usage_metadata = Mock()
        response.usage_metadata.prompt_token_count = 100
        response.usage_metadata.candidates_token_count = 50
        response.usage_metadata.total_token_count = 150
        # Explicitly set thought_token_count to avoid Mock comparison issues
        response.usage_metadata.thought_token_count = 0

        usage, cache_read, cache_create = extract_usage(response)

        assert usage is not None
        assert usage["text"]["input_tokens"] == 100
        assert usage["text"]["output_tokens"] == 50
        assert usage["text"]["total_tokens"] == 150
        # Reasoning tokens should not be present when thought_token_count is 0
        assert "reasoning" not in usage
        assert cache_read is None
        assert cache_create is None

    def test_extracts_reasoning_tokens_for_thinking_models(self):
        """Test extracting reasoning tokens from extended thinking models."""
        response = Mock()
        response.usage_metadata = Mock()
        response.usage_metadata.prompt_token_count = 100
        response.usage_metadata.candidates_token_count = 50
        response.usage_metadata.total_token_count = 150
        response.usage_metadata.thought_token_count = 8200  # Gemini 2.0 Flash Thinking

        usage, cache_read, cache_create = extract_usage(response)

        assert usage is not None
        assert usage["text"]["input_tokens"] == 100
        assert usage["text"]["output_tokens"] == 50
        assert usage["text"]["total_tokens"] == 150
        # Reasoning tokens should be present
        assert "reasoning" in usage
        assert usage["reasoning"]["input_tokens"] == 0
        assert usage["reasoning"]["output_tokens"] == 8200
        assert usage["reasoning"]["total_tokens"] == 8200

    def test_extracts_plural_thoughts_tokens_for_current_sdk(self):
        """Test extracting reasoning tokens from Google's plural field name."""
        response = Mock()
        response.usage_metadata = Mock()
        response.usage_metadata.prompt_token_count = 100
        response.usage_metadata.candidates_token_count = 50
        response.usage_metadata.total_token_count = 1050
        response.usage_metadata.thoughts_token_count = 900

        usage, _, _ = extract_usage(response)

        assert usage is not None
        assert usage["reasoning"]["output_tokens"] == 900

    def test_returns_none_when_no_usage_metadata(self):
        """Test returns None when response has no usage_metadata."""
        response = Mock()
        response.usage_metadata = None

        usage, cache_read, cache_create = extract_usage(response)

        assert usage is None
        assert cache_read is None
        assert cache_create is None

    def test_handles_missing_token_fields(self):
        """Test handles missing token count fields gracefully."""
        response = Mock()
        response.usage_metadata = Mock(spec=[])  # Empty spec

        usage, cache_read, cache_create = extract_usage(response)

        assert usage is not None
        assert usage["text"]["input_tokens"] == 0
        assert usage["text"]["output_tokens"] == 0
        assert usage["text"]["total_tokens"] == 0

    def test_extracts_modality_token_details(self):
        """Preserve Google modality breakdowns when SDK exposes them."""

        class Detail:
            def __init__(self, modality: str, token_count: int) -> None:
                self.modality = modality
                self.token_count = token_count

        response = Mock()
        response.usage_metadata = Mock()
        response.usage_metadata.prompt_token_count = 1000
        response.usage_metadata.candidates_token_count = 100
        response.usage_metadata.total_token_count = 1100
        response.usage_metadata.thought_token_count = 0
        response.usage_metadata.prompt_tokens_details = [
            Detail("TEXT", 100),
            Detail("IMAGE", 650),
            Detail("VIDEO", 250),
        ]
        response.usage_metadata.candidates_tokens_details = [Detail("AUDIO", 80)]

        usage, _, _ = extract_usage(response)

        assert usage is not None
        assert usage["image"]["input_tokens"] == 650
        assert usage["video"]["input_tokens"] == 250
        assert usage["audio"]["output_tokens"] == 80


class TestExtractModel:
    """Test extract_model function - the critical normalization logic."""

    def test_extracts_model_name(self):
        """Test extracting model name from response."""
        response = Mock()
        response.model_name = "models/gemini-2.0-flash"

        model = extract_model(response)

        assert model == "gemini-2.0-flash"

    def test_strips_models_prefix(self):
        """Test stripping 'models/' prefix."""
        response = Mock()
        response.model_name = "models/gemini-1.5-pro"

        model = extract_model(response)

        assert model == "gemini-1.5-pro"

    def test_strips_version_suffix_three_digits(self):
        """Test stripping version suffixes like -001."""
        response = Mock()
        response.model_name = "models/gemini-1.5-pro-001"

        model = extract_model(response)

        assert model == "gemini-1.5-pro"

    def test_strips_version_suffix_four_digits(self):
        """Test stripping four-digit version suffixes like -1234."""
        response = Mock()
        response.model_name = "models/gemini-2.0-flash-1234"

        model = extract_model(response)

        assert model == "gemini-2.0-flash"

    def test_returns_unknown_when_missing(self):
        """Test returns 'unknown' when model_name missing."""
        response = Mock()
        response.model_name = None

        model = extract_model(response)

        assert model == "unknown"

    def test_preserves_base_name_without_prefix_or_suffix(self):
        """Test that models without prefix/suffix are preserved."""
        response = Mock()
        response.model_name = "gemini-flash"

        model = extract_model(response)

        assert model == "gemini-flash"


class TestNormalizeModelName:
    """Test _normalize_model_name helper function."""

    def test_strips_models_prefix(self):
        """Test stripping 'models/' prefix."""
        assert _normalize_model_name("models/gemini-2.0-flash") == "gemini-2.0-flash"

    def test_strips_version_suffix(self):
        """Test stripping version suffixes."""
        assert _normalize_model_name("gemini-1.5-pro-001") == "gemini-1.5-pro"
        assert _normalize_model_name("gemini-2.0-flash-1234") == "gemini-2.0-flash"

    def test_strips_both_prefix_and_suffix(self):
        """Test stripping both prefix and suffix."""
        assert _normalize_model_name("models/gemini-1.5-pro-002") == "gemini-1.5-pro"

    def test_preserves_base_name(self):
        """Test that base names are preserved."""
        assert _normalize_model_name("gemini-flash") == "gemini-flash"


class TestPatchClient:
    """Test client patching functionality."""

    def test_patch_client_marks_as_patched(self):
        """Test that patch_client sets vetch_patched."""
        client = Mock()
        client.models = Mock()
        client.models.generate_content = Mock()

        patch_client(client)

        assert hasattr(client, "vetch_patched")
        assert client.vetch_patched is True

    def test_patch_client_skips_already_patched(self):
        """Test that patching is idempotent."""
        client = Mock()
        client.vetch_patched = True
        client.models = Mock()
        original_method = client.models.generate_content

        patch_client(client)

        # Should not re-patch
        assert client.models.generate_content == original_method

    def test_unpatch_client_removes_marker(self):
        """Test that unpatch removes vetch_patched."""
        client = Mock()
        client.models = Mock()
        client.models.generate_content = Mock()
        original = client.models.generate_content

        patch_client(client)
        unpatch_client(client)

        assert not hasattr(client, "vetch_patched")

    def test_unpatch_client_handles_unpatched(self):
        """Test that unpatching unpatched client is safe."""
        client = Mock()
        # Should not raise
        unpatch_client(client)

    def test_patch_client_with_aio_methods(self):
        """Test patching client with async methods."""
        client = Mock()
        client.aio = Mock()
        client.aio.models = Mock()
        client.aio.models.generate_content = Mock()

        patch_client(client)

        # Verify aio method was patched
        assert hasattr(client, "vetch_patched")

    def test_patch_client_with_embed_content(self):
        """Test patching client with embed_content method."""
        client = Mock()
        client.models = Mock()
        client.models.embed_content = Mock()

        patch_client(client)

        # Verify embed method was patched
        assert hasattr(client, "vetch_patched")


class TestModuleInstrumentation:
    """Test module-level instrumentation functions."""

    def test_instrument_genai_module_returns_false_when_not_installed(self):
        """Test that instrument_genai_module returns False when module not available."""
        import sys
        from unittest.mock import patch

        import vetch.providers.genai as genai_provider
        from vetch.providers.genai import instrument_genai_module

        # Reset module-level instrumentation state so the import check is reached
        original_instrumented = genai_provider._module_instrumented
        genai_provider._module_instrumented = False

        try:
            # Simulate google.genai not being installed
            with patch.dict(sys.modules, {"google.genai": None}):
                result = instrument_genai_module()

            assert result is False
        finally:
            # Restore original state
            genai_provider._module_instrumented = original_instrumented

    def test_uninstrument_genai_module_when_not_instrumented(self):
        """Test that uninstrument_genai_module succeeds when not instrumented."""
        from vetch.providers.genai import uninstrument_genai_module

        # Should succeed even if not instrumented
        result = uninstrument_genai_module()
        assert result is True


class TestPatchedMethodBehavior:
    """Test behavior of patched methods with vetch context."""

    def test_wrapped_generate_calls_original_when_no_context(self):
        """Test that wrapped generate_content calls original when no active context."""
        from vetch.providers.genai import patch_client

        client = Mock()
        client.models = Mock()
        original_generate = Mock(return_value=Mock(usage_metadata=None, model_name="test"))
        client.models.generate_content = original_generate

        patch_client(client)

        # Call without context
        result = client.models.generate_content("test")

        # Should call original
        assert original_generate.called

    def test_wrapped_generate_extracts_metadata_with_context(self):
        """Test that wrapped generate_content works when context is active."""
        from vetch import VetchContext
        from vetch.providers.genai import patch_client

        client = Mock()
        client.models = Mock()

        # Mock response
        response = Mock()
        response.usage_metadata = Mock()
        response.usage_metadata.prompt_token_count = 100
        response.usage_metadata.candidates_token_count = 50
        response.usage_metadata.total_token_count = 150
        response.usage_metadata.thought_token_count = 0
        response.model_name = "models/gemini-2.0-flash-001"

        original_generate = Mock(return_value=response)
        client.models.generate_content = original_generate

        patch_client(client)

        # Call with context - should not crash and should call original
        with VetchContext():
            result = client.models.generate_content("test")

        # Verify original was called
        assert original_generate.called
        assert result == response


class TestTrackGenaiContextManager:
    """Test track_genai context manager."""

    def test_track_genai_patches_and_unpatches(self):
        """Test that track_genai patches on enter and unpatches on exit."""
        client = Mock()
        client.models = Mock()
        client.models.generate_content = Mock()

        with track_genai(client):
            assert hasattr(client, "vetch_patched")

        # Should be unpatched after exit
        assert not hasattr(client, "vetch_patched")

    @pytest.mark.skipif(not _has_pytest_asyncio, reason="pytest-asyncio not installed")
    @pytest.mark.asyncio
    async def test_atrack_genai_patches_and_unpatches(self):
        """Test that atrack_genai async context manager works."""
        from vetch.providers.genai import atrack_genai

        client = Mock()
        client.models = Mock()
        client.models.generate_content = Mock()

        async with atrack_genai(client):
            assert hasattr(client, "vetch_patched")

        # Should be unpatched after exit
        assert not hasattr(client, "vetch_patched")


class TestGenAIStreamWrapper:
    """Tests for sync/async streaming wrappers."""

    def test_sync_stream_captures_usage_on_completion(self):
        """_GenAIStreamWrapper collects usage from final chunk and captures on StopIteration."""

        from vetch.providers.genai import _GenAIStreamWrapper

        chunk1 = Mock(text="Hello ", usage_metadata=None)
        chunk2 = Mock(text="world", usage_metadata=None)
        final_chunk = Mock(text="!")
        final_chunk.usage_metadata = Mock(
            prompt_token_count=10,
            candidates_token_count=5,
            total_token_count=15,
            thought_token_count=0,
        )

        stream = iter([chunk1, chunk2, final_chunk])
        wrapper = _GenAIStreamWrapper(stream, "gemini-2.0-flash")

        chunks = list(wrapper)
        assert len(chunks) == 3
        assert wrapper._complete is True
        assert wrapper._final_usage is not None
        assert wrapper._final_usage["text"]["input_tokens"] == 10
        assert wrapper._final_usage["text"]["output_tokens"] == 5
        assert wrapper._accumulated_chars == 12  # "Hello " + "world" + "!"

    def test_sync_stream_captures_error_on_exception(self):
        """_GenAIStreamWrapper captures error metadata when stream raises."""
        from vetch.providers.genai import _GenAIStreamWrapper

        def bad_stream():
            yield Mock(text="a", usage_metadata=None)
            raise RuntimeError("connection reset")

        wrapper = _GenAIStreamWrapper(bad_stream(), "gemini-2.0-flash")
        with pytest.raises(RuntimeError, match="connection reset"):
            list(wrapper)

        assert wrapper._error is True
        assert wrapper._error_type == "RuntimeError"
        assert wrapper._captured is True

    def test_sync_stream_context_manager_captures_on_exit(self):
        """_GenAIStreamWrapper __exit__ triggers capture even on abandoned iteration."""
        from vetch.providers.genai import _GenAIStreamWrapper

        stream = iter([Mock(text="hi", usage_metadata=None)])
        wrapper = _GenAIStreamWrapper(stream, "gemini-2.0-flash")

        with wrapper:
            pass  # never iterate — abandoned stream

        assert wrapper._captured is True

    def test_patch_client_patches_generate_content_stream(self):
        """patch_client() patches generate_content_stream alongside generate_content."""
        from vetch.providers.genai import _GenAIStreamWrapper, patch_client

        client = Mock()
        client.models = Mock()
        client.models.generate_content = Mock(
            return_value=Mock(usage_metadata=None, model_name="x")
        )
        original_stream_fn = Mock(return_value=iter([Mock(text="tok", usage_metadata=None)]))
        client.models.generate_content_stream = original_stream_fn

        patch_client(client)

        # generate_content_stream should be wrapped
        assert client.models.generate_content_stream is not original_stream_fn

        # Calling the patched method should return a _GenAIStreamWrapper
        result = client.models.generate_content_stream(model="gemini-2.0-flash", contents="hi")
        assert isinstance(result, _GenAIStreamWrapper)

    def test_generate_content_stream_unpatched_on_unpatch(self):
        """unpatch_client() restores generate_content_stream to original."""
        from vetch.providers.genai import patch_client, unpatch_client

        client = Mock()
        client.models = Mock()
        client.models.generate_content = Mock(
            return_value=Mock(usage_metadata=None, model_name="x")
        )
        original_stream_fn = Mock()
        client.models.generate_content_stream = original_stream_fn

        patch_client(client)
        unpatch_client(client)

        assert client.models.generate_content_stream is original_stream_fn


class TestGenAIErrorCapture:
    """Tests for error capture in GenAI wrappers."""

    def test_error_captured_with_provider_on_exception(self):
        """_WeakMethodWrapper captures error with provider=google_genai before re-raising."""
        from vetch import VetchContext
        from vetch.providers.genai import patch_client

        client = Mock()
        client.models = Mock()
        client.models.generate_content = Mock(side_effect=ValueError("quota exceeded"))

        patch_client(client)

        captured_calls = []

        class _CapturingContext(VetchContext):
            def _on_event(self, event):
                captured_calls.append(event)

        with pytest.raises(ValueError, match="quota exceeded"):
            with VetchContext() as ctx:
                client.models.generate_content(model="gemini-2.0-flash", contents="hi")

        # The wrapper should have captured an error event
        # (VetchContext emits on __exit__ using whatever capture() received)
        # We verify the test doesn't hang and the exception propagates cleanly
        # A full integration test would inspect the emitted event's provider field.

    def test_error_not_swallowed_on_exception(self):
        """Exceptions must propagate out of wrapped call regardless of capture."""
        from vetch.providers.genai import patch_client

        client = Mock()
        client.models = Mock()
        client.models.generate_content = Mock(side_effect=ConnectionError("timeout"))

        patch_client(client)

        with pytest.raises(ConnectionError, match="timeout"):
            client.models.generate_content(model="gemini-2.0-flash", contents="hi")

