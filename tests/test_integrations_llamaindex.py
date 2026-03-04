"""Integration tests for LlamaIndex callback handler.

Focus on critical paths: event handling, error resilience.
"""

from unittest.mock import MagicMock


class TestLlamaIndexIntegration:
    """Test LlamaIndex integration critical paths."""

    def test_callback_handler_creation(self):
        """Test handler can be created without crashing."""
        from vetch.integrations.llamaindex import VetchCallbackHandler

        handler = VetchCallbackHandler(region="us-west-2", tags={"env": "test"})
        assert handler.region == "us-west-2"
        assert handler.tags == {"env": "test"}
        assert handler.total_cost == 0.0
        assert handler.total_energy_wh == 0.0
        assert handler.call_count == 0

    def test_on_event_start_returns_event_id(self):
        """Test on_event_start returns event_id without crashing."""
        from vetch.integrations.llamaindex import VetchCallbackHandler

        handler = VetchCallbackHandler(region="us-east-1")

        # Mock CBEventType
        mock_event_type = MagicMock()
        event_id = handler.on_event_start(
            event_type=mock_event_type,
            payload={"test": "data"},
            event_id="test-event-123",
        )

        assert event_id == "test-event-123"

    def test_non_llm_event_ignored(self):
        """Test non-LLM events are ignored gracefully."""
        from vetch.integrations.llamaindex import VetchCallbackHandler

        handler = VetchCallbackHandler(region="us-east-1")

        # Mock non-LLM event
        mock_event_type = MagicMock()
        mock_event_type.name = "EMBEDDING"

        # Should not crash or track
        handler.on_event_end(
            event_type=mock_event_type, payload={}, event_id="test-123"
        )

        assert handler.call_count == 0

    def test_reset_clears_metrics(self):
        """Test reset() clears aggregated metrics."""
        from vetch.integrations.llamaindex import VetchCallbackHandler

        handler = VetchCallbackHandler(region="us-east-1")

        # Set metrics
        handler.total_cost = 5.0
        handler.total_energy_wh = 50.0
        handler.call_count = 3

        # Reset
        handler.reset()

        assert handler.total_cost == 0.0
        assert handler.total_energy_wh == 0.0
        assert handler.call_count == 0
        assert len(handler.events) == 0

    def test_start_trace_noop(self):
        """Test start_trace doesn't crash (no-op for compatibility)."""
        from vetch.integrations.llamaindex import VetchCallbackHandler

        handler = VetchCallbackHandler(region="us-east-1")

        # Should not crash
        handler.start_trace(trace_id="test-trace-123")

    def test_end_trace_noop(self):
        """Test end_trace doesn't crash (no-op for compatibility)."""
        from vetch.integrations.llamaindex import VetchCallbackHandler

        handler = VetchCallbackHandler(region="us-east-1")

        # Should not crash
        handler.end_trace(trace_id="test-trace-123", trace_map={})

    def test_handler_has_vetch_available(self):
        """Test handler detects vetch availability."""
        from vetch.integrations.llamaindex import VetchCallbackHandler

        handler = VetchCallbackHandler(region="us-east-1")

        # Should have _wrap and _vetch_available attributes
        assert hasattr(handler, "_wrap")
        assert hasattr(handler, "_vetch_available")
        assert handler._vetch_available in (True, False)
