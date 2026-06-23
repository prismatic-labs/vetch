"""Self-hosted / OpenAI-compatible routing tests (B3).

A non-OpenAI endpoint must never be billed OpenAI's per-token rates: local hosts
report cost 0, third-party compatible hosts report cost unknown (None), and only
genuine OpenAI/Azure hosts use the list-price path.
"""

from __future__ import annotations

import pytest

from vetch.calculation import prepare_inference_metrics
from vetch.providers.openai import _infer_openai_provider


def _usage(in_tok: int = 1000, out_tok: int = 500) -> dict:
    return {"text": {"input_tokens": in_tok, "output_tokens": out_tok,
                     "total_tokens": in_tok + out_tok}}


def _metrics(provider: str, model: str = "gpt-4o"):
    return prepare_inference_metrics(
        model=model,
        provider=provider,
        usage=_usage(),
        accumulated_chars=0,
        region=None,
        price_multiplier=1.0,
        energy_override=None,
        cache_read_tokens=None,
        cache_creation_tokens=None,
        existing_warnings=[],
    )


class TestBaseUrlClassification:
    @pytest.mark.parametrize("base_url,expected", [
        (None, "openai"),
        ("https://api.openai.com/v1", "openai"),
        ("https://my-resource.openai.azure.com/", "openai"),
        ("https://eastus.api.cognitive.microsoft.com/", "openai"),
        ("http://localhost:11434/v1", "ollama"),
        ("http://localhost:8000/v1", "self-hosted"),
        ("http://127.0.0.1:8000", "self-hosted"),
        ("http://10.0.0.5:8000/v1", "self-hosted"),
        ("http://192.168.1.9:1234/v1", "self-hosted"),
        ("http://172.16.0.3:8000", "self-hosted"),
        ("https://openrouter.ai/api/v1", "openai-compatible"),
        ("https://api.together.xyz/v1", "openai-compatible"),
        ("https://my-vllm.example.com/v1", "openai-compatible"),
    ])
    def test_classification(self, base_url, expected) -> None:
        assert _infer_openai_provider(base_url) == expected

    def test_vetch_self_hosted_env(self, monkeypatch) -> None:
        monkeypatch.setenv("VETCH_SELF_HOSTED", "true")
        assert _infer_openai_provider("https://some-public-host.example/v1") == "self-hosted"


class TestCostRouting:
    def test_official_openai_is_list_priced(self) -> None:
        m = _metrics("openai", "gpt-4o")
        assert m.cost_usd is not None
        assert m.cost_usd > 0

    def test_self_hosted_cost_is_zero(self) -> None:
        # Even with a model that HAS OpenAI list pricing, self-hosted bills nothing.
        m = _metrics("self-hosted", "gpt-4o")
        assert m.cost_usd == 0.0
        assert m.billing_tier == "self-hosted"
        # Energy is still estimated.
        assert m.energy_wh is not None and m.energy_wh > 0

    def test_openai_compatible_cost_unknown(self) -> None:
        m = _metrics("openai-compatible", "gpt-4o")
        assert m.cost_usd is None
        assert m.billing_tier == "unknown"
        assert m.energy_wh is not None and m.energy_wh > 0

    def test_ollama_cost_is_zero(self) -> None:
        m = _metrics("ollama", "llama-3-70b")
        assert m.cost_usd == 0.0

    def test_non_openai_never_inherits_openai_price(self) -> None:
        openai_cost = _metrics("openai", "gpt-4o").cost_usd
        for provider in ("self-hosted", "openai-compatible", "ollama"):
            cost = _metrics(provider, "gpt-4o").cost_usd
            assert cost != openai_cost
