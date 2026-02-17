"""Tests for energy, carbon, and cost calculation.

These tests verify:
- Model alias resolution
- Energy calculation (Wh)
- Carbon calculation (gCO2e)
- Cost calculation (USD)
- Conservative fallbacks for unknown models
- Energy overrides
- Token estimation (with and without tiktoken)
"""

from __future__ import annotations

from vetch.calculation import (
    calculate_carbon,
    calculate_cost,
    calculate_energy,
    estimate_tokens,
    resolve_model,
)


class TestModelResolution:
    """Tests for model name resolution and aliases."""

    def test_resolve_exact_match(self) -> None:
        """Resolve model when it exists exactly in registry."""
        resolved, known = resolve_model("gpt-4o")
        assert resolved == "gpt-4o"
        assert known is True

    def test_resolve_alias(self) -> None:
        """Resolve model via explicit alias."""
        resolved, known = resolve_model("gpt-4-0613")
        assert resolved == "gpt-4"
        assert known is True

    def test_resolve_prefix(self) -> None:
        """Resolve model via prefix matching."""
        resolved, known = resolve_model("gpt-4o-2024-05-13")
        assert resolved == "gpt-4o"
        assert known is True

    def test_resolve_unknown(self) -> None:
        """Handle completely unknown model."""
        resolved, known = resolve_model("mysterious-llm-v1")
        assert resolved == "mysterious-llm-v1"
        assert known is False


class TestEnergyCalculation:
    """Tests for energy consumption calculation."""

    def test_calculate_energy_known_model(self) -> None:
        """Calculate energy for a known model (gpt-4o: 0.3 in, 0.8 out)."""
        # (1000 * 0.3 + 500 * 0.8) / 1000 = (300 + 400) / 1000 = 0.7 Wh
        energy, tier, source, basis, known = calculate_energy(1000, 500, "gpt-4o")

        assert energy == 0.7
        assert tier == 3
        assert source == "registry"
        assert known is True
        assert basis is not None  # Has basis text

    def test_calculate_energy_unknown_model(self) -> None:
        """Calculate energy for unknown model using conservative fallback."""
        energy, tier, source, basis, known = calculate_energy(1000, 500, "unknown-model")

        # Fallback uses 1.5 in, 4.5 out
        # (1000 * 1.5 + 500 * 4.5) / 1000 = (1500 + 2250) / 1000 = 3.75 Wh
        assert energy == 3.75
        assert tier == 3
        assert source == "fallback"
        assert known is False

    def test_calculate_energy_override(self) -> None:
        """Use user-provided energy values."""
        override = {
            "wh_per_1k_input": 2.0,
            "wh_per_1k_output": 6.0,
            "tier": 2,
            "source": "benchmarks",
            "basis": "Custom measurement",
        }
        # (1000 * 2.0 + 500 * 6.0) / 1000 = (2000 + 3000) / 1000 = 5.0 Wh
        energy, tier, source, basis, known = calculate_energy(1000, 500, "gpt-4o", override)

        assert energy == 5.0
        assert tier == 2
        assert source == "override"
        assert basis == "Custom measurement"


class TestCarbonCalculation:
    """Tests for carbon emissions calculation."""

    def test_calculate_carbon(self) -> None:
        """Calculate carbon from energy and grid intensity."""
        import pytest
        # energy_wh * pue * grid_intensity / 1000
        # 2.0 Wh * 1.1 PUE * 400 gCO2e/kWh / 1000 = 0.88g
        carbon = calculate_carbon(2.0, 400.0, pue=1.1)
        assert carbon == pytest.approx(0.88)


class TestCostCalculation:
    """Tests for cost estimation."""

    def test_calculate_cost_known_model(self) -> None:
        """Calculate cost for a known model (gpt-4o: $0.005 in, $0.015 out)."""
        import pytest
        # (1000 * 0.005 + 500 * 0.015) / 1000 = (5.0 + 7.5) / 1000 = $0.0125
        total, cost_in, cost_out, tier = calculate_cost(1000, 500, "gpt-4o")

        assert total == pytest.approx(0.0125)
        assert cost_in == pytest.approx(0.005)
        assert cost_out == pytest.approx(0.0075)
        assert tier == "list"

    def test_calculate_cost_unknown_model(self) -> None:
        """Return zero cost for unknown models."""
        total, cost_in, cost_out, tier = calculate_cost(1000, 500, "unknown-model")

        assert total == 0.0
        assert tier == "none"


class TestTokenEstimation:
    """Tests for token count estimation."""

    def test_empty_text(self) -> None:
        """Empty text returns 0 tokens."""
        assert estimate_tokens("") == 0
        assert estimate_tokens(None) == 0

    def test_english_prose(self) -> None:
        """English prose uses ~4 chars/token heuristic or tiktoken."""
        text = "Hello, this is a simple test of the token estimation function."
        tokens = estimate_tokens(text)
        # Should be reasonable (10-20 tokens for this text)
        assert 5 <= tokens <= 25

    def test_cjk_text(self) -> None:
        """CJK text uses adjusted ratio (~1.5 chars/token)."""
        # Chinese text: "这是一个测试" (This is a test)
        text = "这是一个测试中文文本的功能"
        tokens = estimate_tokens(text)
        # CJK characters should yield more tokens per character
        # Without tiktoken: ~12 chars / 1.5 = ~8 tokens
        # With tiktoken: varies but typically similar
        assert tokens >= 5

    def test_code_like_text(self) -> None:
        """Code uses adjusted ratio (~3 chars/token)."""
        code = "def foo(): return {'a': [1, 2], 'b': (x + y) * z}"
        tokens = estimate_tokens(code)
        # Code-like content with many symbols
        assert tokens >= 5

    def test_with_model_hint(self) -> None:
        """Model hint can improve accuracy (when tiktoken available)."""
        text = "The quick brown fox jumps over the lazy dog."
        tokens_generic = estimate_tokens(text)
        tokens_gpt4 = estimate_tokens(text, model="gpt-4o")
        # Both should return reasonable values
        assert tokens_generic >= 5
        assert tokens_gpt4 >= 5

    def test_minimum_one_token(self) -> None:
        """Minimum of 1 token for non-empty text."""
        assert estimate_tokens("a") >= 1
        assert estimate_tokens("ab") >= 1
        assert estimate_tokens("abc") >= 1


class TestTokenEstimationWithTiktoken:
    """Tests specifically for tiktoken integration."""

    def test_tiktoken_available(self) -> None:
        """Check if tiktoken is available (informational)."""
        try:
            import tiktoken
            available = True
        except ImportError:
            available = False

        # If tiktoken is available, estimate_tokens should use it
        text = "Hello world"
        tokens = estimate_tokens(text, model="gpt-4o")
        if available:
            # tiktoken for "Hello world" gives exactly 2 tokens
            assert tokens == 2
        else:
            # Heuristic: 11 chars / 4 = 2-3 tokens
            assert tokens >= 2

    def test_tiktoken_fallback_on_unknown_model(self) -> None:
        """Unknown models should fall back to cl100k_base or heuristic."""
        text = "Testing with an unknown model name"
        tokens = estimate_tokens(text, model="totally-unknown-model-xyz")
        # Should still return reasonable estimate
        assert 5 <= tokens <= 15
