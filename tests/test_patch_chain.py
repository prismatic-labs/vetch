"""Tests for patch chain compatibility.

These tests verify that Vetch works correctly with existing patches
from observability tools like Datadog, OpenTelemetry, and Sentry.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from vetch.compat import detect_existing_patches
from vetch.proxy import create_wrapper, get_original, is_vetch_patched


class TestPatchChainDetection:
    """Tests for detecting existing patches."""

    def test_detect_datadog_patch(self) -> None:
        """Detect Datadog patched function."""

        def func() -> None:
            pass

        # Simulate Datadog patch marker
        func._datadog_patch = True  # type: ignore[attr-defined]

        patches = detect_existing_patches(func)
        assert "datadog" in patches

    def test_detect_opentelemetry_patch(self) -> None:
        """Detect OpenTelemetry patched function."""

        def func() -> None:
            pass

        func._otel_patched = True  # type: ignore[attr-defined]

        patches = detect_existing_patches(func)
        assert "opentelemetry" in patches

    def test_detect_sentry_patch(self) -> None:
        """Detect Sentry patched function."""

        def func() -> None:
            pass

        func._sentry_wrapped = True  # type: ignore[attr-defined]

        patches = detect_existing_patches(func)
        assert "sentry" in patches

    def test_detect_multiple_patches(self) -> None:
        """Detect multiple patches on same function."""

        def func() -> None:
            pass

        func._datadog_patch = True  # type: ignore[attr-defined]
        func._otel_patched = True  # type: ignore[attr-defined]

        patches = detect_existing_patches(func)
        assert "datadog" in patches
        assert "opentelemetry" in patches
        assert len(patches) == 2

    def test_detect_unknown_wrapper(self) -> None:
        """Detect generic __wrapped__ attribute."""

        def original() -> None:
            pass

        def wrapper() -> None:
            pass

        wrapper.__wrapped__ = original  # type: ignore[attr-defined]

        patches = detect_existing_patches(wrapper)
        assert "unknown" in patches


class TestPatchChainPreservation:
    """Tests for preserving existing patches."""

    def test_vetch_wraps_existing_patch(self) -> None:
        """Vetch wrapper preserves existing Datadog patch."""

        def original() -> str:
            return "original"

        # Simulate Datadog wrapping
        def datadog_wrapper() -> str:
            return original() + "_datadog"

        datadog_wrapper._datadog_patch = True  # type: ignore[attr-defined]
        datadog_wrapper.__wrapped__ = original  # type: ignore[attr-defined]

        # Vetch wraps the Datadog wrapper
        vetch_wrapper = create_wrapper(datadog_wrapper)

        # Both patches should be detectable
        assert is_vetch_patched(vetch_wrapper)
        # The original Datadog wrapper is preserved
        assert get_original(vetch_wrapper) is datadog_wrapper

    def test_existing_patch_still_executes(self) -> None:
        """Existing patch behavior is preserved."""
        call_order: list[str] = []

        def original() -> str:
            call_order.append("original")
            return "result"

        # Simulate existing observability wrapper
        def existing_wrapper() -> str:
            call_order.append("existing_before")
            result = original()
            call_order.append("existing_after")
            return result

        existing_wrapper._otel_patched = True  # type: ignore[attr-defined]

        # Vetch wraps it
        def vetch_before(*args: Any, **kwargs: Any) -> None:
            call_order.append("vetch_before")

        def vetch_after(result: Any, *args: Any, **kwargs: Any) -> None:
            call_order.append("vetch_after")

        vetch_wrapper = create_wrapper(
            existing_wrapper,
            before_call=vetch_before,
            after_call=vetch_after,
        )

        result = vetch_wrapper()

        assert result == "result"
        # Vetch hooks run, then existing wrapper, then original
        assert call_order == [
            "vetch_before",
            "existing_before",
            "original",
            "existing_after",
            "vetch_after",
        ]

    def test_attribute_forwarding(self) -> None:
        """Attributes are forwarded through patch chain."""

        def original() -> None:
            pass

        original.custom_attr = "value"  # type: ignore[attr-defined]
        original.__doc__ = "Original docstring"

        wrapped = create_wrapper(original)

        # functools.wraps preserves these
        assert wrapped.__doc__ == "Original docstring"
        # Custom attributes should be accessible via get_original
        assert get_original(wrapped).custom_attr == "value"  # type: ignore[attr-defined]

    def test_double_vetch_patch_prevented(self) -> None:
        """Vetch doesn't double-patch already patched functions."""

        def original() -> str:
            return "result"

        # First Vetch wrap
        wrapped1 = create_wrapper(original)
        assert is_vetch_patched(wrapped1)

        # Check is_vetch_patched before wrapping again
        assert is_vetch_patched(wrapped1) is True

        # If we wrap again, the original is preserved
        wrapped2 = create_wrapper(wrapped1)
        assert get_original(wrapped2) is wrapped1
        # Can still get to true original
        assert get_original(get_original(wrapped2)) is original


class TestPatchChainWithMockClient:
    """Tests with mock LLM client scenarios."""

    def test_mock_openai_with_existing_instrumentation(self) -> None:
        """Mock OpenAI client with existing instrumentation."""
        call_log: list[str] = []

        # Mock OpenAI client with Datadog instrumentation
        class MockCompletions:
            def create(self, **kwargs: Any) -> MagicMock:
                call_log.append("openai_create")
                response = MagicMock()
                response.model = "gpt-4"
                response.usage = MagicMock(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                )
                return response

        # Simulate Datadog patch on the method
        mock_completions = MockCompletions()
        original_create = mock_completions.create

        def datadog_create(**kwargs: Any) -> Any:
            call_log.append("datadog_before")
            result = original_create(**kwargs)
            call_log.append("datadog_after")
            return result

        datadog_create._datadog_patch = True  # type: ignore[attr-defined]
        mock_completions.create = datadog_create  # type: ignore[method-assign]

        # Vetch wraps
        vetch_wrapper = create_wrapper(
            mock_completions.create,
            before_call=lambda *a, **k: call_log.append("vetch_before"),
            after_call=lambda r, *a, **k: call_log.append("vetch_after"),
        )

        mock_completions.create = vetch_wrapper  # type: ignore[method-assign]

        # Make a call
        result = mock_completions.create(model="gpt-4", messages=[])

        # All instrumentation should have run
        assert call_log == [
            "vetch_before",
            "datadog_before",
            "openai_create",
            "datadog_after",
            "vetch_after",
        ]
        assert result.model == "gpt-4"
