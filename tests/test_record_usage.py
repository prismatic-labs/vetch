"""Tests for record_usage() and provider_hint.

record_usage is the escape hatch for calls Vetch does not intercept (e.g. a
self-hosted model over raw HTTP): it runs the same calculation and emit path as
an instrumented call. provider_hint overrides the provider inferred from the
model name, so a self-hosted model is not billed at cloud rates.
"""

from __future__ import annotations

from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

import vetch
from vetch.emitter import BufferedEmitter, set_test_emitter


@pytest.fixture(autouse=True)
def _emitter():
    emitter = BufferedEmitter()
    set_test_emitter(emitter)
    yield emitter
    set_test_emitter(None)


class TestRecordUsageStandalone:
    def test_emits_one_event_with_usage(self, _emitter):
        ev = vetch.record_usage("gpt-4o", 1000, 200, region="us-east-1")
        assert len(_emitter.events) == 1
        assert ev is _emitter.events[0]
        text = ev["usage"]["text"]
        assert (text["input_tokens"], text["output_tokens"], text["total_tokens"]) == (
            1000,
            200,
            1200,
        )
        # Real calc ran: known OpenAI model has a cost and energy.
        assert ev["estimated_cost_usd"] and ev["estimated_cost_usd"] > 0
        assert ev["estimated_energy_wh"] and ev["estimated_energy_wh"] > 0

    def test_provider_inferred_from_model_when_no_hint(self, _emitter):
        vetch.record_usage("gpt-4o", 10, 10)
        assert _emitter.events[0]["provider"] == "openai"

    def test_reasoning_tokens_surfaced_and_counted(self, _emitter):
        ev = vetch.record_usage("gpt-4o", 100, 50, reasoning_tokens=30)
        assert ev["usage"]["reasoning"]["output_tokens"] == 30
        assert ev["usage"]["text"]["output_tokens"] == 50  # visible unchanged
        assert ev["usage"]["text"]["total_tokens"] == 180  # 100 + 50 + 30

    def test_negative_tokens_clamped(self, _emitter):
        ev = vetch.record_usage("gpt-4o", -5, -5)
        assert ev["usage"]["text"]["input_tokens"] == 0
        assert ev["usage"]["text"]["output_tokens"] == 0

    def test_tags_and_region_applied(self, _emitter):
        ev = vetch.record_usage("gpt-4o", 10, 10, region="eu-west-1", tags={"team": "ml"})
        assert ev["region"] == "eu-west-1"
        assert ev["tags"] == {"team": "ml"}

    def test_emit_false_returns_event_without_emitting(self, _emitter):
        ev = vetch.record_usage("gpt-4o", 10, 10, emit=False)
        assert ev is not None
        assert len(_emitter.events) == 0

    def test_two_calls_emit_two_events(self, _emitter):
        vetch.record_usage("gpt-4o", 10, 10)
        vetch.record_usage("gpt-4o", 20, 20)
        assert len(_emitter.events) == 2


class TestProviderHintSelfHosted:
    def test_self_hosted_zeroes_cost_keeps_energy(self, _emitter):
        ev = vetch.record_usage(
            "gemma-4-31b-it", 1000, 200, provider_hint="self-hosted", region="us-east-1"
        )
        assert ev["provider"] == "self-hosted"
        assert ev["billing_tier"] == "self-hosted"
        assert ev["estimated_cost_usd"] == 0.0
        # energy/carbon still computed
        assert ev["estimated_energy_wh"] and ev["estimated_energy_wh"] > 0

    def test_without_hint_gemma_infers_google_but_not_cloud_billed(self, _emitter):
        # gemma-4-31b-it is a first-class self-hosted row ($0 price),
        # so even without the hint the name infers google but is NOT billed at
        # cloud rates (the registry fix). provider_hint still adds the explicit
        # self-hosted billing tier (see test_self_hosted_zeroes_cost_keeps_energy).
        ev = vetch.record_usage("gemma-4-31b-it", 1000, 200, region="us-east-1")
        assert ev["provider"] == "google"
        assert ev["estimated_cost_usd"] == 0.0

    def test_openai_compatible_leaves_cost_unknown(self, _emitter):
        ev = vetch.record_usage(
            "gemma-4-31b-it", 1000, 200, provider_hint="openai-compatible"
        )
        assert ev["provider"] == "openai-compatible"
        assert ev["estimated_cost_usd"] is None  # unknown, not wrong


class TestCacheReadEnergy:
    def test_cache_read_marks_hit_and_reduces_energy(self, _emitter):
        full = vetch.record_usage("gpt-4o", 1000, 100, region="us-east-1")
        cached = vetch.record_usage(
            "gpt-4o", 1000, 100, region="us-east-1", cache_read_tokens=900
        )
        assert cached["cache_hit"] is True
        assert full["cache_hit"] is False
        # 900 of 1000 input tokens at the cache-read energy factor -> less energy.
        assert cached["estimated_energy_wh"] < full["estimated_energy_wh"]


class TestRecordUsageInsideWrap:
    def test_emits_own_event_and_returns_it(self, _emitter):
        # record_usage owns its event regardless of an active wrap(); it inherits
        # the wrap's region/tags. The wrap emits its own (empty) event on exit.
        with vetch.wrap(region="eu-west-1", tags={"team": "ml"}):
            ret = vetch.record_usage("gpt-4o", 500, 100)
        assert isinstance(ret, dict)  # always returns the event
        manual = next(e for e in _emitter.events if e["model"] == "gpt-4o")
        assert manual["region"] == "eu-west-1"
        assert manual["tags"] == {"team": "ml"}

    def test_no_silent_coalescing(self, _emitter):
        # Regression for the earlier capture-into-parent design that dropped
        # N-1 events. N manual records must each emit (+ 1 for the wrap itself).
        with vetch.wrap(region="us-east-1"):
            for _ in range(5):
                vetch.record_usage("gpt-4o", 100, 100)
        manual = [e for e in _emitter.events if e["model"] == "gpt-4o"]
        assert len(manual) == 5

    def test_does_not_patch_sdk_clients(self, _emitter):
        # record_usage must not patch the global OpenAI client (no live call).
        pytest.importorskip("openai")
        import openai

        client = openai.OpenAI(api_key="test-key")
        vetch.record_usage("gpt-4o", 10, 10)
        assert vetch.is_client_instrumented(client) is False


class TestWrapProviderHint:
    def test_overrides_captured_provider(self, _emitter):
        from vetch.context import get_active_context

        with vetch.wrap(provider_hint="self-hosted", region="us-east-1"):
            ctx = get_active_context()
            ctx.capture(
                model="gemma-4-31b-it",
                provider="google",  # what the SDK/name would infer
                usage={"text": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
                complete=True,
            )
        ev = _emitter.events[0]
        assert ev["provider"] == "self-hosted"
        assert ev["estimated_cost_usd"] == 0.0


class TestLatency:
    def test_no_fabricated_latency_by_default(self, _emitter):
        # Manual events must not report emit-overhead as inference latency.
        ev = vetch.record_usage("gpt-4o", 100, 100)
        assert ev["latency_ms"] is None

    def test_caller_supplied_duration_passes_through(self, _emitter):
        ev = vetch.record_usage("gpt-4o", 100, 100, duration_ms=1234.5)
        assert ev["latency_ms"] == 1234.5


class TestProviderHintValidation:
    def test_unrecognized_hint_warns_fail_loud(self, _emitter):
        ev = vetch.record_usage("gemma-4-31b-it", 100, 100, provider_hint="selfhosted")
        warnings = ev.get("vetch_warnings") or []
        assert any("Unrecognised provider_hint" in w for w in warnings)

    def test_case_and_whitespace_normalized(self, _emitter):
        ev = vetch.record_usage("gemma-4-31b-it", 100, 100, provider_hint="  Self-Hosted ")
        assert ev["provider"] == "self-hosted"
        assert ev["estimated_cost_usd"] == 0.0
        # A correctly-spelled hint produces no provider warning.
        warnings = ev.get("vetch_warnings") or []
        assert not any("Unrecognised provider_hint" in w for w in warnings)

    def test_known_hint_no_warning(self, _emitter):
        ev = vetch.record_usage("gpt-4o", 100, 100, provider_hint="openai-compatible")
        warnings = ev.get("vetch_warnings") or []
        assert not any("Unrecognised provider_hint" in w for w in warnings)


class TestPriceMultiplier:
    def test_multiplier_scales_cost(self, _emitter):
        full = vetch.record_usage("gpt-4o", 1000, 200)
        disc = vetch.record_usage("gpt-4o", 1000, 200, price_multiplier=0.5)
        assert disc["estimated_cost_usd"] == pytest.approx(full["estimated_cost_usd"] / 2)


class TestInstrumentProviderHintDefault:
    def test_default_applies_to_auto_instrumented_calls(self, _emitter):
        pytest.importorskip("openai")
        import openai

        from vetch.providers.openai import patch_openai_client, uninstrument_openai_module

        try:
            vetch.instrument(provider_hint="self-hosted", region="us-east-1")
            client = openai.OpenAI(api_key="test-key")
            client.chat.completions.create = MagicMock(
                return_value=NS(
                    model="gemma-4-31b-it",
                    usage=NS(
                        prompt_tokens=10,
                        completion_tokens=5,
                        total_tokens=15,
                        prompt_tokens_details=None,
                        completion_tokens_details=None,
                    ),
                    choices=[NS(message=NS(content="hi", tool_calls=None),
                                finish_reason="stop", delta=NS(content=None))],
                )
            )
            patch_openai_client(client)
            # No wrap(): auto-context should pick up the instrument() default hint.
            client.chat.completions.create(model="gemma-4-31b-it", messages=[])
            ev = _emitter.events[-1]
            assert ev["provider"] == "self-hosted"
            assert ev["estimated_cost_usd"] == 0.0
        finally:
            # Restore global state so other tests are unaffected.
            vetch.uninstrument()
            uninstrument_openai_module()
            import vetch as _v

            _v._default_provider_hint = None
            _v._default_region = None
            _v._instrumented = False


class TestSchemaParity:
    def test_record_usage_event_schema_matches_instrumented(self, _emitter):
        """record_usage output must be schema-identical to an instrumented call."""
        pytest.importorskip("openai")
        import openai

        from vetch.providers.openai import patch_openai_client

        # Instrumented event
        client = openai.OpenAI(api_key="test-key")
        client.chat.completions.create = MagicMock(
            return_value=NS(
                model="gpt-4o",
                usage=NS(
                    prompt_tokens=1000,
                    completion_tokens=200,
                    total_tokens=1200,
                    prompt_tokens_details=None,
                    completion_tokens_details=None,
                ),
                choices=[NS(message=NS(content="hi", tool_calls=None), finish_reason="stop",
                            delta=NS(content=None))],
            )
        )
        patch_openai_client(client)
        with vetch.wrap(region="us-east-1"):
            client.chat.completions.create(model="gpt-4o", messages=[])
        instrumented = _emitter.events[-1]

        # Manual event
        manual = vetch.record_usage("gpt-4o", 1000, 200, region="us-east-1")

        # Same schema (key set), even if values differ.
        assert set(manual.keys()) == set(instrumented.keys())
        assert manual["schema_version"] == instrumented["schema_version"]


class TestRecordUsageVisual:
    def test_video_modality_splits_like_intercepted(self, _emitter):
        from vetch import calculation as calc

        calc._load_registry()
        assert calc._ENERGY is not None
        key = "synthetic-vlm-record-usage"
        calc._ENERGY[key] = {
            "tier": 1,
            "wh_per_1k_input": 1.0,
            "wh_per_1k_output": 2.0,
            "wh_per_visual_unit": 0.5,
            "visual_tokens_per_unit": 100,
            "basis": "synthetic",
        }
        try:
            intercepted = calc.prepare_inference_metrics(
                model=key,
                provider="self-hosted",
                usage={
                    "text": {
                        "input_tokens": 1000,
                        "output_tokens": 50,
                        "total_tokens": 1050,
                    },
                    "video": {"input_tokens": 200, "visual_units": 2},
                },
                accumulated_chars=0,
                region=None,
                price_multiplier=1.0,
                energy_override=None,
                cache_read_tokens=0,
                cache_creation_tokens=None,
                existing_warnings=[],
            )
            manual = vetch.record_usage(
                key,
                1000,
                50,
                provider_hint="self-hosted",
                visual_input_tokens=200,
                visual_units=2,
                visual_modality="video",
                emit=False,
            )
            assert manual is not None
            assert manual["usage"].get("video", {}).get("input_tokens") == 200
            assert manual["energy_completeness"] == "complete"
            assert intercepted.energy_wh is not None
            assert abs(manual["estimated_energy_wh"] - intercepted.energy_wh) < 1e-9
        finally:
            calc._ENERGY.pop(key, None)

    def test_visual_without_coeff_is_text_only(self, _emitter):
        ev = vetch.record_usage(
            "gpt-4o",
            500,
            10,
            visual_input_tokens=300,
            visual_modality="audio",
            emit=False,
        )
        assert ev is not None
        assert ev["energy_completeness"] == "text_only"
        assert "audio" in (ev.get("usage") or {})
