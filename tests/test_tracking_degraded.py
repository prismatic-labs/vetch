"""tracking_degraded recalibration (v0.9.0).

The flag means "Vetch had to compensate for degraded tracking inputs": it fires
for unknown models, prefix/family proxies, estimated usage, and missing usage,
while a healthy call (exact match, real usage) stays clean. Before v0.9.0 the
threshold (2.5) was above the score's maximum (2.0), so the flag never fired.
"""

from __future__ import annotations

from vetch.calculation import prepare_inference_metrics


def _metrics(model: str, usage: object, region: str | None = "us-east-1"):
    return prepare_inference_metrics(
        model=model,
        provider="openai",
        usage=usage,
        accumulated_chars=0,
        region=region,
        price_multiplier=1.0,
        energy_override=None,
        cache_read_tokens=None,
        cache_creation_tokens=None,
        existing_warnings=[],
    )


_REAL_USAGE = {"text": {"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500}}


class TestTrackingDegraded:
    def test_healthy_exact_call_is_not_degraded(self) -> None:
        assert _metrics("gpt-4o", _REAL_USAGE).tracking_degraded is False

    def test_honest_tier3_known_model_is_not_degraded(self) -> None:
        # A known Tier-3 model (working as designed) should not be flagged.
        assert _metrics("gemini-2.5-pro", _REAL_USAGE).tracking_degraded is False

    def test_family_proxy_is_degraded(self) -> None:
        m = _metrics("gemini-9-ultra", _REAL_USAGE)
        assert m.model_match == "family"
        assert m.tracking_degraded is True

    def test_prefix_proxy_is_degraded(self) -> None:
        m = _metrics("gpt-4o-frontier-2099", _REAL_USAGE)
        assert m.model_match == "prefix"
        assert m.tracking_degraded is True

    def test_unknown_model_is_degraded(self) -> None:
        assert _metrics("totally-unknown-xyz", _REAL_USAGE).tracking_degraded is True

    def test_missing_usage_is_degraded(self) -> None:
        # Parity with the JS SDK, which flags missing usage directly.
        assert _metrics("gpt-4o", None).tracking_degraded is True
