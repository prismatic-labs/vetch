"""Integration tests for LangChain callback handler.

Focus on critical paths: doesn't crash user code, basic functionality works.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestLangChainIntegration:
    """Test LangChain integration critical paths."""

    def test_callback_handler_creation(self):
        """Test handler can be created without crashing."""
        from vetch.integrations.langchain import VetchCallbackHandler

        handler = VetchCallbackHandler(region="us-east-1", tags={"test": "integration"})
        assert handler.region == "us-east-1"
        assert handler.tags == {"test": "integration"}
        assert handler.total_cost == 0.0
        assert handler.total_energy_wh == 0.0
        assert handler.call_count == 0

    def test_on_llm_error_doesnt_crash(self):
        """Test that errors don't crash the user's code."""
        from vetch.integrations.langchain import VetchCallbackHandler

        handler = VetchCallbackHandler(region="us-east-1")

        # This should not raise
        handler.on_llm_error(Exception("Test error"), run_id="test-123")

    def test_reset_clears_metrics(self):
        """Test reset() clears aggregated metrics."""
        from vetch.integrations.langchain import VetchCallbackHandler

        handler = VetchCallbackHandler(region="us-east-1")

        # Set some metrics
        handler.total_cost = 10.0
        handler.total_energy_wh = 100.0
        handler.call_count = 5

        # Reset should clear
        handler.reset()

        assert handler.total_cost == 0.0
        assert handler.total_energy_wh == 0.0
        assert handler.call_count == 0
        assert len(handler.events) == 0

    def test_handler_has_vetch_available(self):
        """Test handler detects vetch availability."""
        from vetch.integrations.langchain import VetchCallbackHandler

        handler = VetchCallbackHandler(region="us-east-1")

        # Should have _wrap attribute
        assert hasattr(handler, "_wrap")
        assert handler._vetch_available in (True, False)


class TestLangChainRealisticScenarios:
    """Test realistic production scenarios with LangChain."""

    def test_tracks_successful_llm_call(self):
        """Handler tracks metrics from successful LLM call with usage data."""

        from vetch.integrations.langchain import VetchCallbackHandler

        handler = VetchCallbackHandler(region="us-east-1", tags={"chain": "qa"})

        # Simulate LangChain LLMResult from OpenAI call
        mock_response = MagicMock()
        mock_response.generations = [[MagicMock()]]  # Non-empty generations
        mock_response.llm_output = {
            "model_name": "gpt-4",
            "token_usage": {
                "prompt_tokens": 50,
                "completion_tokens": 100,
                "total_tokens": 150,
            },
        }

        # Call the callback
        handler.on_llm_end(mock_response)

        # Should track the call
        assert handler.call_count > 0

    def test_skips_llm_call_without_usage(self):
        """Handler gracefully skips calls without token usage."""

        from vetch.integrations.langchain import VetchCallbackHandler

        handler = VetchCallbackHandler(region="us-east-1")

        # Simulate response without usage data
        mock_response = MagicMock()
        mock_response.generations = [[MagicMock()]]
        mock_response.llm_output = {}  # No token_usage

        # Should not crash
        handler.on_llm_end(mock_response)

    def test_aggregates_multiple_llm_calls(self):
        """Handler aggregates metrics across multiple LLM calls in a chain."""

        from vetch.integrations.langchain import VetchCallbackHandler

        handler = VetchCallbackHandler(region="us-west-2")

        # Simulate 3 LLM calls in a chain
        for _ in range(3):
            mock_response = MagicMock()
            mock_response.generations = [[MagicMock()]]
            mock_response.llm_output = {
                "model_name": "gpt-3.5-turbo",
                "token_usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
            }
            handler.on_llm_end(mock_response)

        # Should aggregate across all calls
        assert handler.call_count == 3

    def test_detects_different_providers(self):
        """Handler detects different LLM providers from model names."""

        from vetch.integrations.langchain import VetchCallbackHandler

        handler = VetchCallbackHandler(region="us-east-1")

        # Test OpenAI detection
        openai_response = MagicMock()
        openai_response.generations = [[MagicMock()]]
        openai_response.llm_output = {
            "model_name": "gpt-4",
            "token_usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }
        handler.on_llm_end(openai_response)

        # Test Anthropic detection
        anthropic_response = MagicMock()
        anthropic_response.generations = [[MagicMock()]]
        anthropic_response.llm_output = {
            "model_name": "claude-3-sonnet",
            "token_usage": {"prompt_tokens": 15, "completion_tokens": 25},
        }
        handler.on_llm_end(anthropic_response)

        # Should have tracked both calls
        assert handler.call_count == 2

    def test_bounded_event_storage_prevents_oom(self):
        """Event storage is bounded to prevent memory exhaustion."""

        from vetch.integrations.langchain import VetchCallbackHandler

        handler = VetchCallbackHandler(region="us-east-1")

        # Simulate many LLM calls (more than maxlen=1000)
        for _ in range(1500):
            mock_response = MagicMock()
            mock_response.generations = [[MagicMock()]]
            mock_response.llm_output = {
                "model_name": "gpt-4",
                "token_usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
            handler.on_llm_end(mock_response)

        # Events should be bounded (maxlen=1000)
        assert len(handler.events) <= 1000
        # But all calls should be counted
        assert handler.call_count == 1500


class TestLangChainErrorHandling:
    """Test error handling in production scenarios."""

    def test_handles_empty_generations_gracefully(self):
        """Handler gracefully handles LLM response with empty generations."""

        from vetch.integrations.langchain import VetchCallbackHandler

        handler = VetchCallbackHandler(region="us-east-1")

        # Simulate response with empty generations (API error scenario)
        mock_response = MagicMock()
        mock_response.generations = []  # Empty!
        mock_response.llm_output = {"token_usage": {"prompt_tokens": 10}}

        # Should not crash
        handler.on_llm_end(mock_response)
        # Should not have incremented call count
        assert handler.call_count == 0

    def test_detects_gemini_vertexai_models(self):
        """Handler correctly detects Google Gemini/VertexAI models."""

        from vetch.integrations.langchain import VetchCallbackHandler

        handler = VetchCallbackHandler(region="us-central1")

        # Test Gemini model detection
        gemini_response = MagicMock()
        gemini_response.generations = [[MagicMock()]]
        gemini_response.llm_output = {
            "model_name": "gemini-pro",
            "token_usage": {"prompt_tokens": 20, "completion_tokens": 30},
        }
        handler.on_llm_end(gemini_response)

        assert handler.call_count == 1

    def test_tracking_failure_doesnt_crash_user_code(self):
        """Tracking failures don't crash user's LangChain pipeline."""

        from vetch.integrations.langchain import VetchCallbackHandler

        handler = VetchCallbackHandler(region="us-east-1")

        # Mock vetch.wrap to raise an exception
        with patch.object(handler, "_wrap") as mock_wrap:
            mock_wrap.side_effect = Exception("Vetch internal error")

            mock_response = MagicMock()
            mock_response.generations = [[MagicMock()]]
            mock_response.llm_output = {
                "model_name": "gpt-4",
                "token_usage": {"prompt_tokens": 10, "completion_tokens": 20},
            }

            # Should not crash user's code
            handler.on_llm_end(mock_response)

    def test_on_llm_start_no_op(self):
        """on_llm_start is a no-op (tracking happens in on_llm_end)."""
        from vetch.integrations.langchain import VetchCallbackHandler

        handler = VetchCallbackHandler(region="us-east-1")

        # Should not crash, just passes through
        handler.on_llm_start(
            serialized={"name": "OpenAI"}, prompts=["Test prompt"], run_id="test-123"
        )


class TestLangChainTask1Fixes:
    """Task 1: real base subclass, usage_metadata extraction, provider heuristic."""

    def _buffered(self):
        from vetch.emitter import BufferedEmitter, set_test_emitter

        buf = BufferedEmitter()
        set_test_emitter(buf)
        return buf

    def test_handler_is_base_callback_handler(self):
        pytest.importorskip("langchain_core")
        from langchain_core.callbacks.base import BaseCallbackHandler

        from vetch.integrations.langchain import VetchCallbackHandler

        assert isinstance(VetchCallbackHandler(), BaseCallbackHandler)

    def test_chat_model_dispatch_does_not_raise(self):
        # Repro A: chat-model dispatch must not leak AttributeError('raise_error').
        pytest.importorskip("langchain_core")
        from langchain_core.callbacks import CallbackManager
        from langchain_core.messages import HumanMessage

        from vetch.integrations.langchain import VetchCallbackHandler

        cm = CallbackManager(handlers=[VetchCallbackHandler(region="europe-west9")])
        cm.on_chat_model_start({"name": "fake"}, [[HumanMessage(content="hi")]])

    def test_gemini_shaped_result_captured(self):
        # Repro B: usage_metadata on the message must be captured.
        pytest.importorskip("langchain_core")
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, LLMResult

        from vetch.emitter import set_test_emitter
        from vetch.integrations.langchain import VetchCallbackHandler

        self._buffered()
        try:
            h = VetchCallbackHandler()
            msg = AIMessage(
                content="x",
                usage_metadata={"input_tokens": 900, "output_tokens": 250, "total_tokens": 1150},
            )
            h.on_llm_end(
                LLMResult(
                    generations=[[ChatGeneration(message=msg)]],
                    llm_output={"prompt_feedback": {}},
                )
            )
            assert h.call_count == 1
            text = h.events[-1]["usage"]["text"]
            assert text["input_tokens"] == 900
            assert text["output_tokens"] == 250
        finally:
            set_test_emitter(None)

    def test_openai_shaped_result_still_captured(self):
        # Regression: legacy llm_output["token_usage"] path still works.
        pytest.importorskip("langchain_core")
        from langchain_core.outputs import Generation, LLMResult

        from vetch.emitter import set_test_emitter
        from vetch.integrations.langchain import VetchCallbackHandler

        self._buffered()
        try:
            h = VetchCallbackHandler()
            # Plain Generation has no .message -> forces the legacy token_usage path.
            h.on_llm_end(
                LLMResult(
                    generations=[[Generation(text="hi")]],
                    llm_output={
                        "model_name": "gpt-4",
                        "token_usage": {
                            "prompt_tokens": 50,
                            "completion_tokens": 100,
                            "total_tokens": 150,
                        },
                    },
                )
            )
            assert h.call_count == 1
            text = h.events[-1]["usage"]["text"]
            assert text["input_tokens"] == 50
            assert text["output_tokens"] == 100
        finally:
            set_test_emitter(None)

    def test_allowlist_applies_via_handler(self):
        pytest.importorskip("langchain_core")
        from langchain_core.outputs import Generation, LLMResult

        import vetch
        import vetch.config as _cfg
        from vetch.emitter import set_test_emitter
        from vetch.integrations.langchain import VetchCallbackHandler

        buf = self._buffered()
        prev_allowlist = _cfg._tag_allowlist
        try:
            vetch.set_tag_allowlist(["feature"])
            h = VetchCallbackHandler(tags={"feature": "x", "secret_key": "v"})
            h.on_llm_end(
                LLMResult(
                    generations=[[Generation(text="hi")]],
                    llm_output={
                        "model_name": "gpt-4",
                        "token_usage": {"prompt_tokens": 10, "completion_tokens": 20},
                    },
                )
            )
            assert h.call_count == 1
            tags = buf.events[-1].get("tags") or {}
            assert tags.get("feature") == "x"
            assert "secret_key" not in tags
            assert "v" not in tags.values()
        finally:
            _cfg._tag_allowlist = prev_allowlist
            set_test_emitter(None)
