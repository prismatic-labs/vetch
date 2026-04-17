"""Tests for training emissions contextualizer.

These tests verify:
- Loading training emissions registry (filtering metadata keys)
- Contextualize footprint returns correct fractions and descriptions
- Edge cases: zero carbon, exact match to known training cost
- Env-var override for custom training data
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from vetch.training import (
    _reset_training_data,
    contextualize_footprint,
    get_training_emissions,
)


@pytest.fixture(autouse=True)
def _isolate_training_cache() -> None:
    """Reset cached training data between tests."""
    _reset_training_data()


class TestGetTrainingEmissions:
    """Tests for loading and filtering the training registry."""

    def test_loads_all_models(self) -> None:
        """Registry contains all 5 frontier models."""
        data = get_training_emissions()
        assert len(data) == 5

    def test_filters_metadata_keys(self) -> None:
        """Keys starting with _ are excluded."""
        data = get_training_emissions()
        assert not any(k.startswith("_") for k in data)

    def test_model_names(self) -> None:
        """All expected models are present."""
        data = get_training_emissions()
        expected = {"Grok 4", "Grok 3", "GPT-4", "Llama 3.1 405B", "DeepSeek v3"}
        assert set(data.keys()) == expected

    def test_model_has_required_fields(self) -> None:
        """Each model entry has co2e_tonnes and year."""
        data = get_training_emissions()
        for model, info in data.items():
            assert "co2e_tonnes" in info, f"{model} missing co2e_tonnes"
            assert "year" in info, f"{model} missing year"
            assert isinstance(info["co2e_tonnes"], (int, float))
            assert isinstance(info["year"], int)


class TestContextualizeFootprint:
    """Tests for contextualize_footprint()."""

    def test_zero_carbon(self) -> None:
        """Zero carbon produces zero fractions for all models."""
        results = contextualize_footprint(0.0)
        assert len(results) == 5
        for r in results:
            assert r["fraction"] == 0.0

    def test_exact_match_gpt4(self) -> None:
        """5,184 tonnes (in grams) gives fraction=1.0 for GPT-4."""
        # GPT-4 training = 5184 tonnes = 5,184,000,000 grams
        carbon_g = 5_184 * 1_000_000  # tonnes -> grams
        results = contextualize_footprint(carbon_g)

        gpt4 = next(r for r in results if r["model"] == "GPT-4")
        assert gpt4["fraction"] == pytest.approx(1.0)
        assert gpt4["training_co2e_tonnes"] == 5184

    def test_sorted_by_training_cost_descending(self) -> None:
        """Results are sorted by training cost, largest first."""
        results = contextualize_footprint(100.0)
        training_costs = [r["training_co2e_tonnes"] for r in results]
        assert training_costs == sorted(training_costs, reverse=True)

    def test_small_inference_fraction(self) -> None:
        """A small inference footprint (1.6g) produces tiny fractions."""
        # Claude 4 Opus ~1.6g per query
        results = contextualize_footprint(1.6)
        for r in results:
            assert r["fraction"] < 1e-6  # way less than 1 millionth

    def test_descriptions_are_strings(self) -> None:
        """All results include a human-readable description."""
        results = contextualize_footprint(100.0)
        for r in results:
            assert isinstance(r["description"], str)
            assert "%" in r["description"]
            assert r["model"] in r["description"]

    def test_result_keys(self) -> None:
        """Each result dict has the expected keys."""
        results = contextualize_footprint(1.0)
        expected_keys = {"model", "training_co2e_tonnes", "fraction", "description"}
        for r in results:
            assert set(r.keys()) == expected_keys

    def test_grok4_is_first(self) -> None:
        """Grok 4 (largest training cost) appears first."""
        results = contextualize_footprint(1.0)
        assert results[0]["model"] == "Grok 4"
        assert results[0]["training_co2e_tonnes"] == 72816

    def test_deepseek_is_last(self) -> None:
        """DeepSeek v3 (smallest training cost) appears last."""
        results = contextualize_footprint(1.0)
        assert results[-1]["model"] == "DeepSeek v3"
        assert results[-1]["training_co2e_tonnes"] == 597


class TestTrainingRegistryOverride:
    """Tests for VETCH_TRAINING_REGISTRY_PATH env var."""

    def test_env_var_override(self, tmp_path: Path) -> None:
        """Custom training data loaded from env var."""
        custom = {
            "_source": "test",
            "TestModel": {"co2e_tonnes": 42, "year": 2026},
        }
        custom_path = tmp_path / "custom_training.json"
        custom_path.write_text(json.dumps(custom))

        os.environ["VETCH_TRAINING_REGISTRY_PATH"] = str(custom_path)
        try:
            _reset_training_data()
            data = get_training_emissions()
            assert "TestModel" in data
            assert data["TestModel"]["co2e_tonnes"] == 42
            assert len(data) == 1  # only TestModel, metadata filtered
        finally:
            del os.environ["VETCH_TRAINING_REGISTRY_PATH"]

    def test_bad_env_var_falls_back(self, tmp_path: Path) -> None:
        """Invalid override path falls back to bundled data."""
        os.environ["VETCH_TRAINING_REGISTRY_PATH"] = "/nonexistent/path.json"
        try:
            _reset_training_data()
            data = get_training_emissions()
            # Should fall back to bundled 5 models
            assert len(data) == 5
        finally:
            del os.environ["VETCH_TRAINING_REGISTRY_PATH"]
