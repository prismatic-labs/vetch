"""Tests for multimodal energy accounting and confidence-aware resolution.

Covers the two v0.10.x follow-ups:

1. Multimodal energy — the Responses extractor surfaces visual usage; the
   registry energy branch applies a visual coefficient when one is declared;
   and a visual call with no coefficient is flagged ``energy_completeness=
   "text_only"`` rather than emitted as a whole measurement.

2. Confidence-aware resolution — the match-confidence taxonomy, the session
   roll-up, and the opt-in strict reporting mode (quarantine / fail-loud).
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

import vetch
from vetch import calculation as calc
from vetch import config as vconfig
from vetch.exceptions import ConfidenceError, ConfigurationError
from vetch.schema import (
    confidence_class,
    event_confidence_class,
    meets_min_confidence,
)
from vetch.stats import (
    SessionStats,
    filter_events_by_confidence,
    require_confidence,
    rollup_confidence_from_events,
)

# ---------------------------------------------------------------------------
# 1. Multimodal energy
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_vlm_entry():
    """Inject a registry VLM row that declares a per-visual-unit coefficient."""
    calc._load_registry()
    assert calc._ENERGY is not None
    key = "synthetic-vlm-1"
    calc._ENERGY[key] = {
        "tier": 1,
        "wh_per_1k_input": 1.0,
        "wh_per_1k_output": 2.0,
        "wh_per_visual_unit": 0.5,       # Wh per normalized visual unit
        "visual_tokens_per_unit": 100,   # provider image tokens per unit
        "basis": "synthetic test row",
    }
    try:
        yield key
    finally:
        calc._ENERGY.pop(key, None)


class TestResponsesVisualUsage:
    def _response(self, image_tokens=0, image_count=0):
        details = NS(cached_tokens=0)
        if image_tokens:
            details.image_tokens = image_tokens
        if image_count:
            details.image_count = image_count
        usage = NS(
            input_tokens=1000,
            output_tokens=50,
            total_tokens=1050,
            input_tokens_details=details,
            output_tokens_details=NS(reasoning_tokens=0),
        )
        return NS(model="gpt-5.6-sol", usage=usage, status="completed", output_text="ok")

    def test_image_usage_populated_when_reported(self):
        from vetch.providers.openai import extract_responses_usage

        usage, _, _ = extract_responses_usage(self._response(image_tokens=800, image_count=4))
        assert usage["image"]["input_tokens"] == 800
        assert usage["image"]["image_count"] == 4

    def test_degrades_to_text_only_when_absent(self):
        from vetch.providers.openai import extract_responses_usage

        usage, _, _ = extract_responses_usage(self._response())
        assert "image" not in usage  # no fabricated visual quantity

    def test_image_count_defaults_zero_when_not_itemized(self):
        from vetch.providers.openai import extract_responses_usage

        usage, _, _ = extract_responses_usage(self._response(image_tokens=800))
        assert usage["image"]["input_tokens"] == 800
        assert usage["image"]["image_count"] == 0


class TestRegistryVisualEnergy:
    def test_visual_coefficient_adds_energy(self, synthetic_vlm_entry):
        base, *_ = calc.calculate_energy(1000, 50, synthetic_vlm_entry)
        assert abs(base - 1.1) < 1e-9  # 1.0 text-in + 0.1 text-out
        with_imgs, *_ = calc.calculate_energy(
            1000, 50, synthetic_vlm_entry, n_images=3
        )
        # 3 images → 3 visual units (100 tok/unit): 300 tokens move off the text
        # coefficient (−0.3 Wh) and are priced as visual (+1.5 Wh). Net +1.2.
        # text: (1000−300)/1000×1.0 = 0.7 ; +0.1 out ; +1.5 visual = 2.3 Wh
        assert abs(with_imgs - 2.3) < 1e-9

    def test_visual_tokens_subtracted_from_text(self, synthetic_vlm_entry):
        # 200 image tokens = 2 visual units at 100 tokens/unit. Those tokens are
        # removed from the text total (priced at the visual coeff instead).
        e, *_ = calc.calculate_energy(
            1000, 0, synthetic_vlm_entry, n_images=0, image_input_tokens=200
        )
        # text: (1000-200)/1000 * 1.0 = 0.8 ; visual: 2 * 0.5 = 1.0 ; total 1.8
        assert abs(e - 1.8) < 1e-9

    def test_visual_coefficient_present_helper(self, synthetic_vlm_entry):
        m = calc.resolve_model_match(synthetic_vlm_entry)
        assert calc.visual_coefficient_present(m, None) is True
        assert calc.visual_coefficient_present(calc.resolve_model_match("gpt-4o"), None) is False


class TestEnergyCompleteness:
    def test_text_only_flag_for_uncounted_visual(self):
        usage = {
            "text": {"input_tokens": 1000, "output_tokens": 50, "total_tokens": 1050},
            "image": {"input_tokens": 800, "image_count": 4},
        }
        m = calc.prepare_inference_metrics(
            model="gpt-4o",
            provider="openai",
            usage=usage,
            accumulated_chars=0,
            region=None,
            price_multiplier=1.0,
            energy_override=None,
            cache_read_tokens=0,
            cache_creation_tokens=None,
            existing_warnings=[],
            n_images=4,
        )
        assert m.energy_completeness == "text_only"
        assert any("text portion" in w for w in m.warnings)

    def test_video_only_routes_to_visual_coefficient(self, synthetic_vlm_entry):
        usage = {
            "text": {"input_tokens": 1000, "output_tokens": 0, "total_tokens": 1000},
            "video": {"input_tokens": 200},
        }
        m = calc.prepare_inference_metrics(
            model=synthetic_vlm_entry,
            provider="openai",
            usage=usage,
            accumulated_chars=0,
            region=None,
            price_multiplier=1.0,
            energy_override=None,
            cache_read_tokens=0,
            cache_creation_tokens=None,
            existing_warnings=[],
            n_images=0,
        )
        assert m.energy_completeness == "complete"
        # Same math as image_input_tokens=200: text 0.8 + visual 1.0 = 1.8
        assert m.energy_wh is not None and abs(m.energy_wh - 1.8) < 1e-9

    def test_audio_only_routes_to_visual_coefficient(self, synthetic_vlm_entry):
        usage = {
            "text": {"input_tokens": 1000, "output_tokens": 0, "total_tokens": 1000},
            "audio": {"input_tokens": 100},
        }
        m = calc.prepare_inference_metrics(
            model=synthetic_vlm_entry,
            provider="openai",
            usage=usage,
            accumulated_chars=0,
            region=None,
            price_multiplier=1.0,
            energy_override=None,
            cache_read_tokens=0,
            cache_creation_tokens=None,
            existing_warnings=[],
        )
        assert m.energy_completeness == "complete"
        # 100 audio tokens = 1 unit @ 100 tok/unit → text 0.9 + visual 0.5 = 1.4
        assert m.energy_wh is not None and abs(m.energy_wh - 1.4) < 1e-9

    def test_mixed_modalities_sum_units(self, synthetic_vlm_entry):
        usage = {
            "text": {"input_tokens": 1000, "output_tokens": 0, "total_tokens": 1000},
            "image": {"input_tokens": 100, "image_count": 1},
            "video": {"input_tokens": 100},
            "audio": {"input_tokens": 100},
        }
        # 300 media tokens = 3 units; text 0.7 + visual 1.5 = 2.2
        e, *_ = calc.calculate_energy(
            1000, 0, synthetic_vlm_entry,
            n_images=calc.media_units_from_usage(usage),
            image_input_tokens=calc.media_input_tokens_from_usage(usage),
        )
        assert abs(e - 2.2) < 1e-9
        assert calc.media_input_tokens_from_usage(usage) == 300
        assert calc.media_units_from_usage(usage) == 3.0

    def test_media_without_coefficient_is_text_only(self):
        usage = {
            "text": {"input_tokens": 500, "output_tokens": 10, "total_tokens": 510},
            "video": {"input_tokens": 400},
        }
        m = calc.prepare_inference_metrics(
            model="gpt-4o",
            provider="openai",
            usage=usage,
            accumulated_chars=0,
            region=None,
            price_multiplier=1.0,
            energy_override=None,
            cache_read_tokens=0,
            cache_creation_tokens=None,
            existing_warnings=[],
        )
        assert m.energy_completeness == "text_only"
        assert any("image/audio/video" in w for w in m.warnings)

    def test_no_double_count_media_tokens(self, synthetic_vlm_entry):
        # Media tokens must leave the text coefficient (not priced twice).
        with_media, *_ = calc.calculate_energy(
            1000, 0, synthetic_vlm_entry, n_images=0, image_input_tokens=200
        )
        text_only, *_ = calc.calculate_energy(800, 0, synthetic_vlm_entry)
        # with_media = 0.8 text + 1.0 visual = 1.8; text_only alone = 0.8
        assert abs(with_media - (text_only + 1.0)) < 1e-9

    def test_complete_when_no_visual_input(self):
        usage = {"text": {"input_tokens": 1000, "output_tokens": 50, "total_tokens": 1050}}
        m = calc.prepare_inference_metrics(
            model="gpt-4o",
            provider="openai",
            usage=usage,
            accumulated_chars=0,
            region=None,
            price_multiplier=1.0,
            energy_override=None,
            cache_read_tokens=0,
            cache_creation_tokens=None,
            existing_warnings=[],
            n_images=0,
        )
        assert m.energy_completeness == "complete"

    def test_complete_when_visual_coefficient_present(self, synthetic_vlm_entry):
        usage = {
            "text": {"input_tokens": 1000, "output_tokens": 50, "total_tokens": 1050},
            "image": {"input_tokens": 200, "image_count": 2},
        }
        m = calc.prepare_inference_metrics(
            model=synthetic_vlm_entry,
            provider="openai",
            usage=usage,
            accumulated_chars=0,
            region=None,
            price_multiplier=1.0,
            energy_override=None,
            cache_read_tokens=0,
            cache_creation_tokens=None,
            existing_warnings=[],
            n_images=2,
        )
        assert m.energy_completeness == "complete"


# ---------------------------------------------------------------------------
# 2. Confidence-aware resolution
# ---------------------------------------------------------------------------


class TestConfidenceTaxonomy:
    def test_class_mapping(self):
        assert confidence_class("exact") == "exact"
        assert confidence_class("alias") == "curated"
        assert confidence_class("prefix") == "proxy"
        assert confidence_class("family") == "proxy"
        assert confidence_class("fallback") == "none"
        assert confidence_class(None) == "none"
        assert confidence_class("bogus") == "none"

    def test_min_confidence_ordering(self):
        assert meets_min_confidence("exact", "curated") is True
        assert meets_min_confidence("alias", "curated") is True
        assert meets_min_confidence("prefix", "curated") is False
        assert meets_min_confidence("fallback", "proxy") is False
        assert meets_min_confidence("prefix", "proxy") is True

    def test_event_confidence_floors_on_calibration_match(self):
        # Exact registry + proxy calibration → proxy overall.
        assert event_confidence_class(_event("exact", calibration_match="proxy")) == "proxy"
        assert event_confidence_class(_event("exact", calibration_match="curated")) == "curated"
        assert event_confidence_class(_event("exact")) == "exact"
        assert event_confidence_class(_event("family", calibration_match="exact")) == "proxy"


def _event(model_match, energy=1.0, cost=0.01, completeness="complete",
           calibration_match=None):
    ev = {
        "model": "m",
        "model_match": model_match,
        "estimated_energy_wh": energy,
        "estimated_cost_usd": cost,
        "estimated_carbon_g": energy * 0.1,
        "energy_completeness": completeness,
        "usage": {"text": {"input_tokens": 10, "output_tokens": 5}},
    }
    if calibration_match is not None:
        ev["calibration_match"] = calibration_match
    return ev


class TestConfidenceRollup:
    def test_session_summary_has_confidence_block(self):
        stats = SessionStats(fire_advisory_hooks=False)
        stats.update(_event("exact", energy=3.0, cost=0.03))
        stats.update(_event("alias", energy=1.0, cost=0.01))
        stats.update(_event("family", energy=6.0, cost=0.06))
        conf = stats.summary()["confidence"]
        # High-confidence = exact + curated = 4 of 10 Wh.
        assert conf["high_confidence_energy_fraction"] == 0.4
        assert conf["by_class"]["proxy"]["energy_wh"] == 6.0
        assert conf["by_class"]["exact"]["count"] == 1

    def test_text_only_rollup(self):
        stats = SessionStats(fire_advisory_hooks=False)
        stats.update(_event("exact", energy=2.0, completeness="complete"))
        stats.update(_event("exact", energy=8.0, completeness="text_only"))
        conf = stats.summary()["confidence"]
        assert conf["energy_text_only_wh"] == 8.0
        assert conf["energy_text_only_count"] == 1
        assert conf["energy_text_only_fraction"] == 0.8

    def test_rollup_from_events_helper(self):
        events = [_event("exact"), _event("prefix"), _event("fallback")]
        conf = rollup_confidence_from_events(events)
        assert conf["by_class"]["exact"]["count"] == 1
        assert conf["by_class"]["proxy"]["count"] == 1
        assert conf["by_class"]["none"]["count"] == 1

    def test_rollup_respects_calibration_match_floor(self):
        stats = SessionStats(fire_advisory_hooks=False)
        stats.update(_event("exact", energy=5.0, calibration_match="proxy"))
        conf = stats.summary()["confidence"]
        assert conf["by_class"]["proxy"]["energy_wh"] == 5.0
        assert conf["by_class"]["exact"]["count"] == 0


class TestStrictMode:
    def teardown_method(self):
        vconfig._reset_config()

    def test_filter_quarantines_below_floor(self):
        events = [_event("exact"), _event("alias"), _event("prefix"), _event("fallback")]
        included, quarantined = filter_events_by_confidence(events, min_confidence="curated")
        assert len(included) == 2  # exact + alias
        assert len(quarantined) == 2  # prefix + fallback

    def test_filter_quarantines_calibration_proxy(self):
        events = [
            _event("exact", calibration_match="exact"),
            _event("exact", calibration_match="proxy"),
        ]
        included, quarantined = filter_events_by_confidence(events, min_confidence="curated")
        assert len(included) == 1
        assert len(quarantined) == 1

    def test_no_floor_includes_everything(self):
        events = [_event("prefix"), _event("fallback")]
        included, quarantined = filter_events_by_confidence(events, min_confidence=None)
        assert len(included) == 2 and quarantined == []

    def test_require_confidence_raises(self):
        events = [_event("exact"), _event("family")]
        with pytest.raises(ConfidenceError) as exc:
            require_confidence(events, min_confidence="curated")
        assert exc.value.min_confidence == "curated"

    def test_require_confidence_passes_when_all_meet_floor(self):
        events = [_event("exact"), _event("alias")]
        out = require_confidence(events, min_confidence="curated")
        assert len(out) == 2

    def test_config_floor_used_when_arg_omitted(self):
        vetch.set_min_match_confidence("exact")
        events = [_event("exact"), _event("alias")]
        included, quarantined = filter_events_by_confidence(events)
        assert len(included) == 1  # only exact meets an "exact" floor
        assert len(quarantined) == 1

    def test_session_stats_surfaces_below_min_confidence(self):
        """Setting the floor alone quarantines in the rollup (fail-open at emit)."""
        vetch.set_min_match_confidence("curated")
        stats = SessionStats(fire_advisory_hooks=False)
        stats.update(_event("exact", energy=2.0, cost=0.02))
        stats.update(_event("alias", energy=1.0, cost=0.01))
        stats.update(_event("family", energy=5.0, cost=0.05))  # proxy — below curated
        conf = stats.summary()["confidence"]
        assert conf["min_match_confidence"] == "curated"
        assert conf["below_min_confidence_count"] == 1
        assert conf["below_min_confidence_energy_wh"] == 5.0
        assert conf["below_min_confidence_cost_usd"] == 0.05
        # Totals still include the below-floor event (fail-open).
        assert stats.total_energy_wh == 8.0

    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("VETCH_MIN_MATCH_CONFIDENCE", "curated")
        assert vconfig.get_min_match_confidence() == "curated"

    def test_invalid_level_rejected(self):
        with pytest.raises(ConfigurationError):
            vetch.set_min_match_confidence("super-exact")
