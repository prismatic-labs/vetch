"""Tests for Ollama provider instrumentation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from vetch.context import TrackingContext
from vetch.providers.ollama import _capture_after, _count_images, _usage_from_response


class TestOllamaHelpers:
    def test_count_images_list(self) -> None:
        assert _count_images({"images": ["a", "b"]}) == 2

    def test_count_single_image(self) -> None:
        assert _count_images({"image": "data"}) == 1

    def test_usage_includes_image_count(self) -> None:
        response = MagicMock(prompt_eval_count=100, eval_count=10)
        usage = _usage_from_response(response, n_images=1)
        assert usage["image"] is not None
        assert usage["image"]["image_count"] == 1  # type: ignore[index]

    def test_capture_sets_provider(self) -> None:
        ctx = TrackingContext()
        with ctx:
            response = MagicMock(prompt_eval_count=5, eval_count=2, response="hi")
            _capture_after("llama3", response, {"images": ["x"]})
            assert ctx.captured_call is not None
            assert ctx.captured_call.provider == "ollama"
            assert ctx.captured_call.usage is not None
            assert ctx.captured_call.usage["image"]["image_count"] == 1  # type: ignore[index]


class TestOllamaInstrument:
    def test_instrument_noop_without_ollama(self) -> None:
        from vetch.providers.ollama import instrument_ollama_module

        with patch.dict("sys.modules", {"ollama": None}):
            assert instrument_ollama_module() is False
