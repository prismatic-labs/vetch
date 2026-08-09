"""Tests for graceful degradation on patch failure.

These tests verify fail-open behavior:
- LLM calls proceed even if patching fails
- Events are emitted with tracking_disabled=True
- No exceptions leak to user code
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from vetch.emitter import BufferedEmitter, set_test_emitter
from vetch.wrapper import VetchContext


class TestPatchFailureGracefulDegradation:
    """Tests for graceful degradation when patching fails."""

    def test_context_works_without_sdk(self) -> None:
        """Context manager works even without any SDK installed."""
        emitter = BufferedEmitter()
        set_test_emitter(emitter)

        try:
            with VetchContext(region="us-east-1") as ctx:
                # Simulate user code — no LLM call is intercepted here
                result = 1 + 1

            # Context should complete without error
            assert result == 2
            # ctx.event stays populated for introspection even when nothing ran...
            assert ctx.event is not None
            assert ctx.event["region"] == "us-east-1"

            # ...but an empty wrap() that intercepted no call emits nothing, so it
            # cannot pollute the aggregation stream (empty-wrap-no-emit contract).
            assert len(emitter) == 0
        finally:
            set_test_emitter(None)

    def test_tracking_disabled_on_setup_failure(self) -> None:
        """Tracking is disabled but context works if setup fails."""
        emitter = BufferedEmitter()
        set_test_emitter(emitter)

        try:
            # Mock _setup_patches to raise
            with patch.object(
                VetchContext, "_setup_patches", side_effect=RuntimeError("Setup failed")
            ):
                with VetchContext() as ctx:
                    result = "success"

                assert result == "success"
                assert ctx.tracking_disabled is True
                assert ctx.event is not None
                # Event still emitted even with tracking disabled
                assert len(emitter) == 1
                assert emitter.events[0]["tracking_disabled"] is True
        finally:
            set_test_emitter(None)

    def test_user_exception_not_suppressed(self) -> None:
        """User exceptions are never suppressed by Vetch."""
        emitter = BufferedEmitter()
        set_test_emitter(emitter)

        try:
            try:
                with VetchContext() as ctx:
                    raise ValueError("User error")
            except ValueError as e:
                assert str(e) == "User error"

            # Event should still be emitted
            assert ctx.event is not None
            assert ctx.event["error"] is True
            assert ctx.event["error_type"] == "ValueError"
        finally:
            set_test_emitter(None)

    def test_emission_failure_does_not_suppress_exception(self) -> None:
        """Emission failure doesn't affect user exception handling."""
        emitter = BufferedEmitter()
        set_test_emitter(emitter)

        try:
            # Make emit fail
            emitter.emit = MagicMock(side_effect=RuntimeError("Emit failed"))

            try:
                with VetchContext():
                    raise ValueError("User error")
            except ValueError as e:
                assert str(e) == "User error"

            # User exception was raised correctly despite emit failure
        finally:
            set_test_emitter(None)

    def test_cleanup_failure_silent(self) -> None:
        """Cleanup failures are silently ignored."""
        emitter = BufferedEmitter()
        set_test_emitter(emitter)

        try:
            with patch.object(
                VetchContext, "_cleanup_patches", side_effect=RuntimeError("Cleanup failed")
            ):
                with VetchContext() as ctx:
                    result = "success"

                # Should complete without error
                assert result == "success"
                assert ctx.event is not None
        finally:
            set_test_emitter(None)


class TestFailOpenBehavior:
    """Tests verifying fail-open principle."""

    def test_all_vetch_errors_caught(self) -> None:
        """All internal Vetch errors are caught and logged."""
        emitter = BufferedEmitter()
        set_test_emitter(emitter)

        try:
            # Create a context where everything fails
            with patch.object(VetchContext, "_setup_patches", side_effect=Exception("Setup")):
                with VetchContext() as ctx:
                    # User code should execute normally
                    x = 42

                assert x == 42
                assert ctx.tracking_disabled is True
        finally:
            set_test_emitter(None)

    def test_llm_call_proceeds_on_failure(self) -> None:
        """Simulated LLM call proceeds even if Vetch fails."""
        emitter = BufferedEmitter()
        set_test_emitter(emitter)

        # Mock an LLM client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello!"))]
        mock_client.chat.completions.create.return_value = mock_response

        try:
            with patch.object(
                VetchContext, "_setup_patches", side_effect=Exception("Patch failed")
            ):
                with VetchContext() as ctx:
                    # LLM call should work
                    response = mock_client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": "Hi"}],
                    )
                    content = response.choices[0].message.content

                assert content == "Hello!"
                assert ctx.tracking_disabled is True
        finally:
            set_test_emitter(None)


class TestEventOnFailure:
    """Tests for event content when failures occur."""

    def test_event_has_error_info(self) -> None:
        """Event contains error information when user code raises."""
        emitter = BufferedEmitter()
        set_test_emitter(emitter)

        try:
            try:
                with VetchContext(region="eu-west-1", tags={"env": "test"}) as ctx:
                    raise RuntimeError("Something went wrong")
            except RuntimeError:
                pass

            event = ctx.event
            assert event is not None
            assert event["error"] is True
            assert event["error_type"] == "RuntimeError"
            assert event["complete"] is False
            assert event["region"] == "eu-west-1"
            assert event["tags"] == {"env": "test"}
        finally:
            set_test_emitter(None)

    def test_event_has_latency_even_on_error(self) -> None:
        """Latency is recorded even when error occurs."""
        emitter = BufferedEmitter()
        set_test_emitter(emitter)

        try:
            import time

            try:
                with VetchContext() as ctx:
                    time.sleep(0.01)  # 10ms
                    raise ValueError("Error")
            except ValueError:
                pass

            assert ctx.event is not None
            assert ctx.event["latency_ms"] is not None
            assert ctx.event["latency_ms"] >= 10
        finally:
            set_test_emitter(None)
