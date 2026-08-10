"""Regression tests for the post-v0.10.5 review fixes (honesty-critical paths).

Covers:
- the cloud-provider blocklist (not just openai) at the resolver, so local
  coefficients never resolve as an exact Tier-0 measurement for cloud traffic;
- the batched calibration path generating unique prompts/images per request, so
  vLLM prefix caching and vision-encoder caching cannot understate energy.
"""

from __future__ import annotations

import vetch.calibrate_cuda as cc
from vetch import calibration_store as store
from vetch.calibrate import CalibrationResult
from vetch.calibration_store import CalibrationIdentity, build_record


def test_is_cloud_provider_covers_all_vendors():
    for p in ["openai", "anthropic", "bedrock", "azure", "azure-openai",
              "vertexai", "google", "gemini", "cohere", "mistral", "groq",
              "together", "fireworks", "aws", "gcp", "AzUrE"]:
        assert store.is_cloud_provider(p) is True, p
    for p in ["vllm", "self-hosted", "ollama", "sglang", "", None]:
        assert store.is_cloud_provider(p) is False, p


def _mk(provider):
    idn = CalibrationIdentity(provider=provider, model="some/model",
                              gpu="h100-sxm-80gb", serving_engine="vllm",
                              precision="bf16")
    res = CalibrationResult(model="some/model", provider=provider,
                            wh_per_1k_input=0.3, wh_per_1k_output=1.7, tier=0,
                            samples=20, gpu_name="NVIDIA H100 80GB HBM3",
                            wh_per_image=0.9, visual_tokens_per_image=280,
                            intercept_wh=0.0006, active=True)
    return idn, res


def test_cloud_keyed_record_never_resolves_exact(monkeypatch, tmp_path):
    # A record mistakenly keyed under a cloud provider must never resolve as an
    # exact Tier-0 measurement for that cloud's real (metered) API traffic.
    monkeypatch.setattr("vetch.calibrate.CALIBRATION_DIR", tmp_path)
    idn, res = _mk("anthropic")
    rec = build_record(res, idn, {"gpu_known": True}, timestamp=100.0)
    store.write_record(rec)
    out = store.resolve("anthropic", "some/model")
    # Either refused outright (None) or attached but capped below exact.
    if out is not None:
        assert out.energy_confidence != "exact"
        assert out.tier >= 1


def test_batched_block_uses_unique_prompts_and_images():
    seen_prompts: list[str] = []
    seen_images: list[str | None] = []

    def _fake_generate(base_url, model, prompt, image_b64, out_tokens, **kwargs):
        seen_prompts.append(prompt)
        seen_images.append(image_b64)
        return (10, out_tokens)

    # Text-only block: every request gets a distinct prompt, no images.
    cc._run_concurrent_block(
        generate=_fake_generate, base_url="x", model="m", concurrency=2,
        n_requests=8, in_words=32, out_tokens=4, fixed_output=True,
        prompt_seed_base=1000,
    )
    assert len(set(seen_prompts)) == 8
    assert all(i is None for i in seen_images)

    # Image block: distinct prompts AND distinct images.
    seen_prompts.clear()
    seen_images.clear()
    cc._run_concurrent_block(
        generate=_fake_generate, base_url="x", model="m", concurrency=2,
        n_requests=6, in_words=32, out_tokens=4, fixed_output=True,
        prompt_seed_base=5000, image_seed_base=9000,
    )
    assert len(set(seen_prompts)) == 6
    assert all(i is not None for i in seen_images)
    assert len(set(seen_images)) == 6
