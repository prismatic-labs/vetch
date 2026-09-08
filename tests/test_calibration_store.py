"""Tests for the versioned, data-rich calibration store (calibration_store)."""

from __future__ import annotations

from pathlib import Path

from vetch import calibration_store as store
from vetch.calibrate import CalibrationResult
from vetch.calibration_store import (
    CalibrationIdentity,
    ResolveHints,
    _measurement_key_payload,
    apply_attestation_hashes,
    build_record,
    canonical_gpu,
    identity_from_dict,
    identity_slug,
)

# --- GPU canonicalization ---------------------------------------------------


def test_canonical_gpu_known_preserves_form_factor():
    assert canonical_gpu("NVIDIA H100 80GB HBM3") == ("h100-sxm-80gb", True)
    assert canonical_gpu("NVIDIA H100 PCIe") == ("h100-pcie-80gb", True)
    assert canonical_gpu("NVIDIA A100-SXM4-40GB") == ("a100-sxm-40gb", True)


def test_canonical_gpu_unknown_is_flagged():
    key, known = canonical_gpu("NVIDIA Superchip 9000")
    assert known is False and key and "superchip" in key


def test_canonical_gpu_none():
    assert canonical_gpu(None) == (None, False)


# --- Identity slug ----------------------------------------------------------


def _idn(**kw):
    base = dict(provider="vllm", model="google/gemma-4-31B-it", gpu="h100-sxm-80gb",
               serving_engine="vllm", precision="bf16")
    base.update(kw)
    return CalibrationIdentity(**base)


def test_slug_deterministic_and_readable():
    s1 = identity_slug(_idn())
    s2 = identity_slug(_idn())
    assert s1 == s2
    assert "gemma" in s1 and "h100-sxm-80gb" in s1 and s1.split("-")[-1].isalnum()


def test_slug_distinguishes_gpus_and_precision():
    assert identity_slug(_idn(gpu="a100-sxm-80gb")) != identity_slug(_idn(gpu="h100-sxm-80gb"))
    assert identity_slug(_idn(precision="fp8-e4m3")) != identity_slug(_idn(precision="bf16"))


# --- Record round-trip ------------------------------------------------------


def _result(**kw):
    base = dict(model="google/gemma-4-31B-it", provider="vllm",
                wh_per_1k_input=0.31, wh_per_1k_output=1.75, tier=0, samples=22,
                gpu_name="NVIDIA H100 80GB HBM3", wh_per_image=0.9,
                visual_tokens_per_image=280, intercept_wh=0.0006, active=True)
    base.update(kw)
    return CalibrationResult(**base)


def test_build_and_parse_round_trip():
    idn = _idn()
    rec = build_record(_result(), idn, {"energy_domain": "gpu_board"}, timestamp=123.0)
    assert rec["schema_version"] == 2
    assert rec["host"]["tenancy"] == "unknown"
    assert rec["node_power"]["available"] is False
    assert rec["content_hash"].startswith("sha256:")
    back = store.record_to_result(rec)
    assert back is not None
    assert back.wh_per_1k_input == 0.31
    assert back.wh_per_image == 0.9
    assert back.visual_tokens_per_image == 280
    assert back.serving_engine == "vllm" and back.precision == "bf16"
    assert back.gpu_name == "NVIDIA H100 80GB HBM3"  # raw preserved


# --- Resolution -------------------------------------------------------------


def _write(monkeypatch, tmp_path, identity, result, ts, provenance=None):
    monkeypatch.setattr("vetch.calibrate.CALIBRATION_DIR", tmp_path)
    # Writers always set gpu_known; default True here so tests of the happy path
    # aren't tripped by the fail-closed default on missing keys.
    prov = {"gpu_known": True}
    if provenance:
        prov.update(provenance)
    rec = build_record(result, identity, prov, timestamp=ts)
    store.write_record(rec)


def test_resolve_single_v1_keeps_tier_exact(monkeypatch, tmp_path):
    _write(monkeypatch, tmp_path, _idn(), _result(tier=0), ts=100.0)
    res = store.resolve("vllm", "google/gemma-4-31B-it")
    assert res is not None
    assert res.tier == 0                      # unambiguous -> measured tier kept
    assert res.energy_confidence == "exact"


def test_resolve_multi_gpu_is_ambiguous_and_capped(monkeypatch, tmp_path):
    # Same provider+model, two different GPUs -> can't tell which serves now.
    _write(monkeypatch, tmp_path, _idn(gpu="h100-sxm-80gb"), _result(tier=0), ts=100.0)
    _write(monkeypatch, tmp_path, _idn(gpu="a100-sxm-80gb"), _result(tier=0), ts=200.0)
    res = store.resolve("vllm", "google/gemma-4-31B-it")
    assert res is not None
    assert res.tier >= 1                      # capped: honest uncertainty
    assert res.energy_confidence == "proxy"


def test_resolve_cross_provider_is_capped(monkeypatch, tmp_path):
    _write(monkeypatch, tmp_path, _idn(provider="vllm"), _result(provider="vllm", tier=0), ts=100.0)
    # A self-hosted-labelled event reuses a cross-labelled self-hosted calibration:
    # curated (not exact), and always tier-capped (never measured Tier 0).
    res = store.resolve("self-hosted", "google/gemma-4-31B-it")
    assert res is not None
    assert res.energy_confidence == "curated"
    assert res.tier >= 1


def test_resolve_rejects_non_local_provider(monkeypatch, tmp_path):
    """A local calibration must NOT attach to a real cloud provider's event with
    a colliding model name — including the ambiguous "openai" label (a real
    OpenAI API call must not silently pick up a local coefficient)."""
    _write(monkeypatch, tmp_path, _idn(provider="vllm"), _result(provider="vllm"), ts=100.0)
    assert store.resolve("anthropic", "google/gemma-4-31B-it") is None
    assert store.resolve("bedrock", "google/gemma-4-31B-it") is None
    assert store.resolve("openai", "google/gemma-4-31B-it") is None  # ambiguous label excluded


def test_resolve_provider_casefold(monkeypatch, tmp_path):
    """A calibration saved with mixed-case --provider still resolves for a
    lowercase event (slugs casefold provider, so matching must too)."""
    _write(monkeypatch, tmp_path, _idn(provider="Self-Hosted"),
           _result(provider="Self-Hosted"), ts=100.0)
    res = store.resolve("self-hosted", "google/gemma-4-31B-it")
    assert res is not None and res.energy_confidence == "exact"


def test_resolve_openai_same_provider_never_exact(monkeypatch, tmp_path):
    """openai-keyed calibrations may resolve but never as measured exact Tier 0."""
    _write(monkeypatch, tmp_path, _idn(provider="openai"), _result(provider="openai"), ts=100.0)
    res = store.resolve("openai", "google/gemma-4-31B-it")
    assert res is not None
    assert res.energy_confidence == "curated"
    assert res.tier >= 1


def test_resolve_inactive_only_returns_none(monkeypatch, tmp_path):
    _write(monkeypatch, tmp_path, _idn(), _result(active=False), ts=100.0)
    assert store.resolve("vllm", "google/gemma-4-31B-it") is None


def test_resolve_multi_precision_same_gpu_is_ambiguous(monkeypatch, tmp_path):
    """bf16 AND fp8 on the same GPU are distinct identities -> ambiguous, not
    exact (would otherwise apply one stack's coefficients as hardware-exact)."""
    _write(monkeypatch, tmp_path, _idn(precision="bf16"), _result(tier=0), ts=100.0)
    _write(monkeypatch, tmp_path, _idn(precision="fp8-e4m3"), _result(tier=0), ts=200.0)
    res = store.resolve("vllm", "google/gemma-4-31B-it")
    assert res is not None
    assert res.energy_confidence == "proxy"
    assert res.tier >= 1


def test_resolve_case_insensitive_model(monkeypatch, tmp_path):
    """HF repo ids vary in case; a differently-cased event still resolves."""
    _write(monkeypatch, tmp_path, _idn(model="Google/Gemma-4-31B-it"),
           _result(model="Google/Gemma-4-31B-it"), ts=100.0)
    res = store.resolve("vllm", "google/gemma-4-31b-it")
    assert res is not None and res.energy_confidence == "exact"


def test_resolve_v1_supersedes_legacy(monkeypatch, tmp_path):
    """A legacy flat file and a v1 record for the same model must not mix (which
    corrupts the ambiguity check): v1 wins and stays exact."""
    import vetch.calibrate as cal

    monkeypatch.setattr(cal, "CALIBRATION_DIR", tmp_path)
    cal.save_calibration(cal.CalibrationResult(  # legacy flat file
        model="m", provider="vllm", wh_per_1k_input=9.9, wh_per_1k_output=9.9,
        tier=0, samples=20, gpu_name="NVIDIA A100-SXM4-80GB",
    ))
    _write(monkeypatch, tmp_path, _idn(model="m", gpu="h100-sxm-80gb"),
           _result(model="m", wh_per_1k_input=0.31), ts=100.0)
    res = store.resolve("vllm", "m")
    assert res is not None
    assert res.wh_per_1k_input == 0.31        # v1 wins, not the legacy 9.9
    assert res.energy_confidence == "exact"   # not corrupted to proxy by the legacy None-gpu


def test_slug_length_bounded_for_long_ids():
    idn = _idn(model="some-org/" + "a" * 300 + ":latest", precision="gguf:q4_k_m")
    slug = identity_slug(idn)
    assert len(slug) + len(".json") <= 255


def test_content_hash_stable_across_timestamps():
    idn = _idn()
    r1 = build_record(_result(), idn, {
        "energy_domain": "gpu_board", "measured_at": "t1",
        "idle_watts_before": 80.0, "raw_run_table": [{"n": 1}],
    }, timestamp=100.0)
    r2 = build_record(_result(), idn, {
        "energy_domain": "gpu_board", "measured_at": "t2",
        "idle_watts_before": 95.0, "raw_run_table": [{"n": 2}],
    }, timestamp=999.0)
    assert r1["content_hash"] == r2["content_hash"]  # run-noise fields excluded
    assert r1["profile_hash"] == r2["profile_hash"]  # identity+coeffs only
    assert r1["timestamp"] != r2["timestamp"]


def test_profile_hash_changes_with_coefficients():
    idn = _idn()
    r1 = build_record(_result(wh_per_1k_input=0.31), idn, {}, timestamp=1.0)
    r2 = build_record(_result(wh_per_1k_input=0.99), idn, {}, timestamp=1.0)
    assert r1["profile_hash"] != r2["profile_hash"]


def test_write_record_archives_previous(monkeypatch, tmp_path):
    monkeypatch.setattr("vetch.calibrate.CALIBRATION_DIR", tmp_path)
    idn = _idn()
    store.write_record(build_record(_result(wh_per_1k_input=0.1), idn, {}, timestamp=1.0))
    store.write_record(build_record(_result(wh_per_1k_input=0.2), idn, {}, timestamp=2.0))
    active = list(tmp_path.glob("*.json"))
    archived = list((tmp_path / "archive").glob("*.json"))
    assert len(active) == 1
    assert len(archived) == 1
    import json
    assert json.loads(active[0].read_text())["coefficients"]["wh_per_1k_input"] == 0.2
    # Archive must not participate in resolve.
    res = store.resolve("vllm", "google/gemma-4-31B-it")
    assert res is not None and res.wh_per_1k_input == 0.2


def test_resolve_unknown_gpu_is_capped(monkeypatch, tmp_path):
    """Heuristic GPU canonicalization must not stay exact Tier 0."""
    idn = _idn(gpu="mystery-accel-9000")
    rec = build_record(_result(tier=0), idn, {"gpu_known": False}, timestamp=100.0)
    monkeypatch.setattr("vetch.calibrate.CALIBRATION_DIR", tmp_path)
    store.write_record(rec)
    res = store.resolve("vllm", "google/gemma-4-31B-it")
    assert res is not None
    assert res.energy_confidence == "proxy"
    assert res.tier >= 1


def test_canonical_apple_silicon_known():
    assert canonical_gpu("Apple M3 Max") == ("apple-m3-max", True)
    assert canonical_gpu("Apple M2") == ("apple-m2-base", True)


def test_resolve_cross_provider_multi_gpu_is_proxy(monkeypatch, tmp_path):
    # Cross-provider AND ambiguous across GPUs -> proxy + tier capped.
    _write(monkeypatch, tmp_path, _idn(provider="vllm", gpu="h100-sxm-80gb"),
           _result(provider="vllm", tier=0), ts=100.0)
    _write(monkeypatch, tmp_path, _idn(provider="vllm", gpu="a100-sxm-80gb"),
           _result(provider="vllm", tier=0), ts=200.0)
    res = store.resolve("self-hosted", "google/gemma-4-31B-it")
    assert res is not None
    assert res.energy_confidence == "proxy"
    assert res.tier >= 1


def test_resolve_prefers_active(monkeypatch, tmp_path):
    _write(monkeypatch, tmp_path, _idn(gpu="h100-sxm-80gb"),
           _result(tier=0, active=False, wh_per_1k_input=9.9), ts=300.0)
    _write(monkeypatch, tmp_path, _idn(gpu="h100-sxm-80gb", precision="fp8-e4m3"),
           _result(tier=0, active=True, wh_per_1k_input=0.31), ts=100.0)
    res = store.resolve("vllm", "google/gemma-4-31B-it")
    assert res is not None and res.wh_per_1k_input == 0.31  # inactive skipped


def test_resolve_none_when_absent(monkeypatch, tmp_path):
    monkeypatch.setattr("vetch.calibrate.CALIBRATION_DIR", tmp_path)
    assert store.resolve("vllm", "does-not-exist") is None


def test_resolve_reads_legacy_flat_file(monkeypatch, tmp_path):
    import vetch.calibrate as cal

    monkeypatch.setattr(cal, "CALIBRATION_DIR", tmp_path)
    # Legacy flat file written by the old save path.
    cal.save_calibration(cal.CalibrationResult(
        model="llama3.1:8b", provider="ollama",
        wh_per_1k_input=0.12, wh_per_1k_output=0.36, tier=0, samples=20,
    ))
    res = store.resolve("ollama", "llama3.1:8b")
    assert res is not None and res.wh_per_1k_input == 0.12


# --- Inference-path labeling (prepare_inference_metrics) ---------------------


def test_prepare_metrics_exact_is_local_calibration(monkeypatch, tmp_path):
    from vetch.calculation import _clear_calibration_cache, prepare_inference_metrics

    monkeypatch.setattr("vetch.calibrate.CALIBRATION_DIR", tmp_path)
    store._clear_store_index()
    _clear_calibration_cache()
    _write(
        monkeypatch, tmp_path,
        _idn(provider="self-hosted", model="gemma-local-test"),
        _result(provider="self-hosted", model="gemma-local-test", tier=0,
                wh_per_1k_input=0.42, wh_per_1k_output=1.1),
        ts=100.0,
    )
    m = prepare_inference_metrics(
        model="gemma-local-test",
        provider="self-hosted",
        usage={"text": {"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500}},
        accumulated_chars=0,
        region=None,
        price_multiplier=1.0,
        energy_override=None,
        cache_read_tokens=None,
        cache_creation_tokens=None,
        existing_warnings=[],
    )
    assert m.energy_source == "local_calibration"
    assert m.calibration_match == "exact"
    assert m.energy_tier == 0
    assert "exact" in (m.energy_basis or "")


def test_prepare_metrics_cross_provider_is_reused(monkeypatch, tmp_path):
    from vetch.calculation import _clear_calibration_cache, prepare_inference_metrics

    monkeypatch.setattr("vetch.calibrate.CALIBRATION_DIR", tmp_path)
    store._clear_store_index()
    _clear_calibration_cache()
    _write(
        monkeypatch, tmp_path,
        _idn(provider="vllm", model="gemma-reuse-test"),
        _result(provider="vllm", model="gemma-reuse-test", tier=0,
                wh_per_1k_input=0.31, wh_per_1k_output=0.9),
        ts=100.0,
    )
    m = prepare_inference_metrics(
        model="gemma-reuse-test",
        provider="self-hosted",
        usage={"text": {"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500}},
        accumulated_chars=0,
        region=None,
        price_multiplier=1.0,
        energy_override=None,
        cache_read_tokens=None,
        cache_creation_tokens=None,
        existing_warnings=[],
    )
    assert m.energy_source == "reused_calibration"
    assert m.calibration_match == "curated"
    assert m.energy_tier >= 1
    assert "curated" in (m.energy_basis or "")


def test_legacy_backend_key_still_reads(monkeypatch, tmp_path):
    """Records written with identity.backend (early v1) still resolve."""
    from vetch.calibration_store import record_stem
    monkeypatch.setattr("vetch.calibrate.CALIBRATION_DIR", tmp_path)
    idn = _idn()
    rec = build_record(_result(), idn, {"gpu_known": True}, timestamp=1.0)
    # Simulate an on-disk record that still uses the old key name.
    rec["identity"] = {
        "provider": "vllm", "model": "google/gemma-4-31B-it",
        "gpu": "h100-sxm-80gb", "backend": "vllm", "precision": "bf16",
        "instance_type": None, "visual_token_budget": None,
    }
    # The alias is supported, but changing the identity also changes the
    # attestation payload. Simulate a genuinely written legacy record rather
    # than an edited record with stale hashes.
    parsed = identity_from_dict(rec["identity"])
    apply_attestation_hashes(
        rec,
        host_key=_measurement_key_payload(
            parsed, rec.get("host"), rec.get("measurement_conditions"),
            rec.get("serving_config"),
        ),
    )
    path = tmp_path / f"{record_stem(rec)}.json"
    import json
    path.write_text(json.dumps(rec))
    store._clear_store_index()
    res = store.resolve("vllm", "google/gemma-4-31B-it")
    assert res is not None
    assert res.serving_engine == "vllm"
    assert res.energy_confidence == "exact"


def test_resolve_hints_disambiguate_precision(monkeypatch, tmp_path):
    from vetch.calibration_store import ResolveHints

    store._clear_policy_caches()
    monkeypatch.delenv("VETCH_CALIB_HINTS_TRUSTED", raising=False)
    _write(monkeypatch, tmp_path, _idn(precision="bf16"),
           _result(wh_per_1k_input=0.31, tier=0), ts=100.0)
    _write(monkeypatch, tmp_path, _idn(precision="fp8-e4m3"),
           _result(wh_per_1k_input=0.22, tier=0), ts=200.0)
    # Blind resolve is ambiguous → proxy.
    blind = store.resolve("vllm", "google/gemma-4-31B-it")
    assert blind is not None and blind.energy_confidence == "proxy"
    # Untrusted hints pick FP8 → curated (env is not attestation).
    hinted = store.resolve(
        "vllm", "google/gemma-4-31B-it",
        hints=ResolveHints(precision="fp8-e4m3"),
    )
    assert hinted is not None
    assert hinted.energy_confidence == "curated"
    assert hinted.wh_per_1k_input == 0.22
    assert hinted.tier >= 1
    # Opt-in trust restores exact.
    monkeypatch.setenv("VETCH_CALIB_HINTS_TRUSTED", "1")
    trusted = store.resolve(
        "vllm", "google/gemma-4-31B-it",
        hints=ResolveHints(precision="fp8-e4m3"),
    )
    assert trusted is not None
    assert trusted.energy_confidence == "exact"
    assert trusted.tier == 0


def test_self_hosted_providers_env_extend(monkeypatch):
    store._clear_policy_caches()
    monkeypatch.setenv("VETCH_SELF_HOSTED_PROVIDERS", "lmstudio,openai,anthropic")
    names = store.self_hosted_providers()
    assert "lmstudio" in names
    assert "openai" not in names
    assert "anthropic" not in names  # cloud blocklist
    store._clear_policy_caches()


def test_slug_includes_instance_type():
    a = identity_slug(_idn(instance_type="p5.48xlarge"))
    b = identity_slug(_idn())
    assert "p5" in a
    assert a != b



# --- Adversarial honesty regressions ----------------------------------------


def test_adversarial_model_variant_is_proxy(monkeypatch, tmp_path):
    """moondream vs moondream:latest must not collapse to false exact."""
    _write(monkeypatch, tmp_path,
           _idn(model="moondream"), _result(model="moondream", wh_per_1k_input=0.11), ts=100.0)
    _write(monkeypatch, tmp_path,
           _idn(model="moondream:latest"),
           _result(model="moondream:latest", wh_per_1k_input=0.99), ts=200.0)
    res = store.resolve("vllm", "moondream")
    assert res is not None
    assert res.energy_confidence == "proxy"
    assert res.tier >= 1


def test_adversarial_instance_type_is_proxy(monkeypatch, tmp_path):
    _write(monkeypatch, tmp_path, _idn(instance_type="p5.48xlarge"),
           _result(wh_per_1k_input=0.31), ts=100.0)
    _write(monkeypatch, tmp_path, _idn(instance_type="p4d.24xlarge"),
           _result(wh_per_1k_input=0.77), ts=200.0)
    res = store.resolve("vllm", "google/gemma-4-31B-it")
    assert res is not None
    assert res.energy_confidence == "proxy"


def test_adversarial_hint_miss_refuses(monkeypatch, tmp_path):
    from vetch.calibration_store import ResolveHints
    _write(monkeypatch, tmp_path, _idn(), _result(wh_per_1k_input=0.31), ts=100.0)
    res = store.resolve(
        "vllm", "google/gemma-4-31B-it",
        hints=ResolveHints(gpu="h100-pcie-80gb"),
    )
    assert res is None


def test_adversarial_v1_does_not_suppress_other_provider_legacy(monkeypatch, tmp_path):
    import vetch.calibrate as cal
    monkeypatch.setattr(cal, "CALIBRATION_DIR", tmp_path)
    _write(monkeypatch, tmp_path, _idn(provider="vllm", model="gemma"),
           _result(provider="vllm", model="gemma", wh_per_1k_input=0.55), ts=100.0)
    cal.save_calibration(cal.CalibrationResult(
        model="gemma", provider="ollama", wh_per_1k_input=0.12, wh_per_1k_output=0.36,
        tier=0, samples=20, gpu_name="Apple M3 Max",
    ))
    store._clear_store_index()
    res = store.resolve("ollama", "gemma")
    assert res is not None
    assert res.wh_per_1k_input == 0.12
    assert res.energy_confidence == "exact"


def test_adversarial_missing_gpu_known_is_capped(monkeypatch, tmp_path):
    import json

    from vetch.calibration_store import record_stem
    monkeypatch.setattr("vetch.calibrate.CALIBRATION_DIR", tmp_path)
    idn = _idn(gpu="mystery-gpu")
    rec = build_record(_result(gpu_name="Mystery GPU"), idn, {}, timestamp=1.0)
    rec["provenance"].pop("gpu_known", None)
    (tmp_path / f"{record_stem(rec)}.json").write_text(json.dumps(rec))
    store._clear_store_index()
    res = store.resolve("vllm", "google/gemma-4-31B-it")
    assert res is not None
    assert res.energy_confidence == "proxy"
    assert res.tier >= 1


def test_adversarial_active_string_false_skipped(monkeypatch, tmp_path):
    import json

    from vetch.calibration_store import record_stem
    monkeypatch.setattr("vetch.calibrate.CALIBRATION_DIR", tmp_path)
    idn = _idn()
    rec = build_record(_result(active=False), idn, {"gpu_known": True}, timestamp=1.0)
    rec["active"] = "false"
    (tmp_path / f"{record_stem(rec)}.json").write_text(json.dumps(rec))
    store._clear_store_index()
    assert store.resolve("vllm", "google/gemma-4-31B-it") is None


# --- Concurrency identity / hints -------------------------------------------


def test_concurrency_in_slug_and_distinct():
    a = identity_slug(_idn(concurrency=1))
    b = identity_slug(_idn(concurrency=32))
    c = identity_slug(_idn())  # None
    assert "c1" in a and "c32" in b
    assert a != b != c


def test_resolve_concurrency_hint_selects(monkeypatch, tmp_path):
    from vetch.calibration_store import ResolveHints

    store._clear_policy_caches()
    monkeypatch.delenv("VETCH_CALIB_HINTS_TRUSTED", raising=False)
    _write(monkeypatch, tmp_path, _idn(concurrency=1),
           _result(wh_per_1k_output=0.54), ts=100.0)
    _write(monkeypatch, tmp_path, _idn(concurrency=32),
           _result(wh_per_1k_output=0.026), ts=200.0)
    blind = store.resolve("vllm", "google/gemma-4-31B-it")
    assert blind is not None and blind.energy_confidence == "proxy"
    hinted = store.resolve(
        "vllm", "google/gemma-4-31B-it",
        hints=ResolveHints(concurrency=32),
    )
    assert hinted is not None
    assert hinted.wh_per_1k_output == 0.026
    assert hinted.energy_confidence == "curated"  # untrusted hints
    monkeypatch.setenv("VETCH_CALIB_HINTS_TRUSTED", "1")
    trusted = store.resolve(
        "vllm", "google/gemma-4-31B-it",
        hints=ResolveHints(concurrency=32),
    )
    assert trusted is not None
    assert trusted.energy_confidence == "exact"
    assert trusted.tier == 0


def test_concurrency_none_back_compat(monkeypatch, tmp_path):
    """Legacy records without concurrency still resolve as a single identity."""
    _write(monkeypatch, tmp_path, _idn(), _result(wh_per_1k_output=0.5), ts=100.0)
    res = store.resolve("vllm", "google/gemma-4-31B-it")
    assert res is not None and res.energy_confidence == "exact"


def test_power_capped_record_is_not_exact_without_matching_hint(monkeypatch, tmp_path):
    monkeypatch.setattr("vetch.calibrate.CALIBRATION_DIR", tmp_path)
    rec = build_record(
        _result(), _idn(), {"gpu_known": True}, timestamp=1.0,
        measurement_conditions={"enforced_power_limit_w": 525.0},
    )
    store.write_record(rec)

    blind = store.resolve("vllm", "google/gemma-4-31B-it")
    assert blind is not None
    assert blind.energy_confidence == "curated"
    assert blind.tier >= 1

    matched = store.resolve(
        "vllm", "google/gemma-4-31B-it",
        hints=ResolveHints(power_cap_w=525.0),
    )
    assert matched is not None
    assert matched.energy_confidence == "exact"
    assert matched.tier == 0


def test_v1_record_still_resolves_with_clamp(monkeypatch, tmp_path):
    """Schema 1 stays resolvable; 0.0 input is the old clamp, not v2-null."""
    import json

    from vetch.calibration_store import identity_slug

    monkeypatch.setattr("vetch.calibrate.CALIBRATION_DIR", tmp_path)
    idn = _idn()
    rec = {
        "schema_version": 1,
        "identity": {
            "provider": "vllm",
            "model": "google/gemma-4-31B-it",
            "gpu": "h100-sxm-80gb",
            "serving_engine": "vllm",
            "precision": "bf16",
            "instance_type": None,
            "visual_token_budget": None,
            "concurrency": None,
        },
        "coefficients": {
            "wh_per_1k_input": 0.0,
            "wh_per_1k_output": 0.54,
        },
        "provenance": {"gpu_known": True, "samples": 11, "tier": 0},
        "active": True,
        "gpu_raw": "NVIDIA H100 80GB HBM3",
        "timestamp": 1.0,
    }
    (tmp_path / f"{identity_slug(idn)}.json").write_text(json.dumps(rec))
    store._clear_store_index()
    res = store.resolve("vllm", "google/gemma-4-31B-it")
    assert res is not None
    assert res.wh_per_1k_input == 0.0
    assert res.wh_per_1k_output == 0.54
    assert res.energy_confidence == "exact"


def test_two_hosts_under_one_resolve_key_tier_cap(monkeypatch, tmp_path):
    """Same resolve identity, two infrastructure owners: do not overwrite, cap tier."""
    from vetch.calibration_store import default_host

    _write(
        monkeypatch, tmp_path, _idn(),
        _result(wh_per_1k_output=0.54), ts=100.0,
        provenance={"gpu_known": True},
    )
    # Second write with a different host must be a distinct file.
    rec = build_record(
        _result(wh_per_1k_output=0.40),
        _idn(),
        {"gpu_known": True},
        timestamp=200.0,
        host=default_host(
            infrastructure_provider="other-cloud",
            provider_kind="hyperscaler",
            tenancy="unknown",
            gpu_form_factor="sxm",
        ),
    )
    monkeypatch.setattr("vetch.calibrate.CALIBRATION_DIR", tmp_path)
    store.write_record(rec)
    files = [p for p in tmp_path.glob("*.json")]
    assert len(files) == 2
    store._clear_store_index()
    res = store.resolve("vllm", "google/gemma-4-31B-it")
    assert res is not None
    assert res.energy_confidence == "proxy"
    assert res.tier >= 1


def test_hints_from_env_concurrency(monkeypatch):
    monkeypatch.setenv("VETCH_CALIB_CONCURRENCY", "16")
    monkeypatch.delenv("VETCH_CALIB_GPU", raising=False)
    monkeypatch.delenv("VETCH_CALIB_SERVING_ENGINE", raising=False)
    monkeypatch.delenv("VETCH_CALIB_PRECISION", raising=False)
    h = store.hints_from_env()
    assert h is not None and h.concurrency == 16


def _record_with_unattributed_input(repo: Path) -> Path:
    """A record whose input axis the design could not resolve.

    Selecting by filename broke once a second Qwen2.5-7B record existed: batch 2
    carries a prefill isolation arm and DOES identify that axis, which is the
    opposite of what these tests need. Select on the property instead.

    """
    import json as _json

    for path in sorted((repo / "h100-calibration-records").glob("*.json")):
        data = _json.loads(path.read_text())
        if not isinstance(data, dict) or not data.get("identity"):
            continue
        if data.get("coefficients", {}).get("wh_per_1k_input") is None:
            return path
    raise AssertionError("no record with an unattributed input axis")


def test_event_names_unattributed_axes(tmp_path, monkeypatch):
    """A null coefficient must be visible as a field, not only as prose.

    energy_completeness covers modality only, so a call can be "complete" while
    a whole coefficient axis went unpriced. Without a machine-readable signal a
    consumer silently under-reports long prompts.
    """
    import json
    import shutil
    from pathlib import Path

    import vetch

    repo = Path(__file__).resolve().parent.parent
    src = _record_with_unattributed_input(repo)
    record = json.loads(src.read_text())
    assert record["coefficients"]["wh_per_1k_input"] is None
    assert record["coefficients"]["wh_per_1k_output"] is not None

    shutil.copy(src, tmp_path / src.name)
    monkeypatch.setattr("vetch.calibrate.CALIBRATION_DIR", tmp_path)

    event = vetch.record_usage(
        model=record["identity"]["model"], provider_hint="self-hosted",
        input_tokens=100_000, output_tokens=10, emit=False,
    )
    assert event is not None
    data = dict(event)
    assert data["energy_source"] == "reused_calibration"
    assert data["energy_axes_unattributed"] == ["wh_per_1k_input"]
    assert any("input coefficient is unidentified" in w
               for w in (data.get("vetch_warnings") or []))


def test_unpriced_axis_makes_the_figure_a_lower_bound(tmp_path, monkeypatch):
    """An incomplete energy figure must not carry a symmetric interval.

    A visual portion with no coefficient and an unidentified coefficient axis
    are the same phenomenon: some energy was never priced. Either way the true
    value is >= the reported one by an unknown margin, so there is no p95.
    """
    import json
    import shutil
    from pathlib import Path

    import vetch

    repo = Path(__file__).resolve().parent.parent
    src = _record_with_unattributed_input(repo)
    shutil.copy(src, tmp_path / src.name)
    monkeypatch.setattr("vetch.calibrate.CALIBRATION_DIR", tmp_path)
    model = json.loads(src.read_text())["identity"]["model"]

    incomplete = dict(vetch.record_usage(
        model=model, provider_hint="self-hosted",
        input_tokens=100_000, output_tokens=10, emit=False))
    assert incomplete["energy_axes_unattributed"] == ["wh_per_1k_input"]
    assert incomplete["energy_estimate_kind"] == "lower_bound"
    assert incomplete["energy_p95_wh"] is None
    assert incomplete["carbon_p95_g"] is None
    # The priced portion still has a meaningful lower end.
    assert incomplete["energy_p5_wh"] is not None
    assert incomplete["energy_p5_wh"] < incomplete["estimated_energy_wh"]

    complete = dict(vetch.record_usage(
        model="gpt-4.1-mini", input_tokens=1000, output_tokens=100, emit=False))
    assert complete["energy_estimate_kind"] == "point"
    assert complete["energy_p95_wh"] > complete["estimated_energy_wh"]


def test_gate_parameters_are_recorded_and_divergence_is_reported():
    """A verdict must state the thresholds it was reached under."""
    import json
    from pathlib import Path

    from vetch.calibrate_v2 import gate_divergence, gate_parameters

    repo = Path(__file__).resolve().parent.parent
    record = json.loads(_record_with_unattributed_input(repo).read_text())
    stored = record["provenance"]["fit"]["gate"]
    assert stored["max_relative_ci_half_width"] == (
        gate_parameters()["max_relative_ci_half_width"])
    assert not gate_divergence(stored)

    moved = {**stored, "max_relative_ci_half_width": 0.5}
    assert gate_divergence(moved) == {
        "max_relative_ci_half_width": {
            "recorded": 0.5,
            "current": gate_parameters()["max_relative_ci_half_width"],
        }
    }


class TestRequestsMeasured:
    """`samples` means the row count for nested records and the request count
    for batched ones. Row aggregation made those different numbers, so the
    request count is now recorded explicitly rather than inferred."""

    def test_aggregated_table_records_request_count(self) -> None:
        from vetch.calibration_store import _requests_in_table

        table = [
            {"requests": 64}, {"requests": 64}, {"requests": 30},
            {"requests": 1}, {},  # a row with no `requests` is one request
        ]
        assert _requests_in_table(table) == 160

    def test_invariant_flags_aggregation_without_the_field(self) -> None:
        from vetch.calibration_invariants import record_invariants

        record = {
            "provenance": {
                "samples": 2,
                "raw_run_table": [
                    {"requests": 8, "energy_wh": 0.1, "raw_energy_wh": 0.2,
                     "duration_ms": 3000.0},
                    {"requests": 8, "energy_wh": 0.1, "raw_energy_wh": 0.2,
                     "duration_ms": 3000.0},
                ],
            },
        }
        codes = {v.code for v in record_invariants(record)}
        assert "requests_measured_absent" in codes

    def test_invariant_flags_a_wrong_request_count(self) -> None:
        from vetch.calibration_invariants import record_invariants

        record = {
            "provenance": {
                "samples": 2,
                "requests_measured": 99,
                "raw_run_table": [
                    {"requests": 8, "energy_wh": 0.1, "raw_energy_wh": 0.2,
                     "duration_ms": 3000.0},
                    {"requests": 8, "energy_wh": 0.1, "raw_energy_wh": 0.2,
                     "duration_ms": 3000.0},
                ],
            },
        }
        violations = [v for v in record_invariants(record)
                      if v.code == "requests_measured_mismatch"]
        assert violations and violations[0].severity == "error"

    def test_shipped_aggregated_record_carries_the_field(self) -> None:
        import json
        from pathlib import Path

        repo = Path(__file__).resolve().parent.parent
        for path in (repo / "h100-calibration-records").glob("*.json"):
            record = json.loads(path.read_text())
            prov = record.get("provenance") or {}
            rows = prov.get("raw_run_table") or []
            if any(float(r.get("requests") or 1) > 1 for r in rows):
                assert prov.get("requests_measured") is not None, path.name


class TestActiveWithRejectedPublishedTerm:
    """`active=true` may coexist with a rejected axis, but only outside the
    published rung. The old blanket rule would now contradict the fitter."""

    @staticmethod
    def _record(published_rung: str, status: dict) -> dict:
        return {
            "active": True,
            "coefficients": {
                "wh_per_1k_output": 0.5, "wh_per_1k_input": None,
                "wh_per_image": None, "intercept_wh": None,
                "published_rung": published_rung, "status": status,
                "null_reasons": {k: "reason" for k, v in status.items()
                                 if v != "identified"},
            },
        }

    def test_rejection_outside_published_rung_is_allowed(self) -> None:
        from vetch.calibration_invariants import record_invariants

        record = self._record("output_input", {
            "wh_per_1k_output": "identified", "wh_per_1k_input": "identified",
            "wh_per_image": "rejected", "intercept_wh": "not_identified",
        })
        codes = {v.code for v in record_invariants(record)}
        assert "active_with_rejected_published_term" not in codes

    def test_rejection_inside_published_rung_is_an_error(self) -> None:
        from vetch.calibration_invariants import errors_only, record_invariants

        record = self._record("output_image", {
            "wh_per_1k_output": "identified", "wh_per_image": "rejected",
            "wh_per_1k_input": "not_identified", "intercept_wh": "not_identified",
        })
        violations = [v for v in errors_only(record_invariants(record))
                      if v.code == "active_with_rejected_published_term"]
        assert violations, [str(v) for v in record_invariants(record)]

    def test_output_only_rung_is_understood(self) -> None:
        from vetch.calibration_invariants import record_invariants

        record = self._record("output_only", {
            "wh_per_1k_output": "rejected", "wh_per_1k_input": "not_identified",
            "wh_per_image": "not_identified", "intercept_wh": "not_identified",
        })
        codes = {v.code for v in record_invariants(record)}
        # output_only decodes to {wh_per_1k_output}; a rejected decode slope is
        # therefore inside the published rung.
        assert "active_with_rejected_published_term" in codes


class TestRefitOnResolveIsOptIn:
    """Resolution always runs the cheap integrity checks; the refit is opt-in.

    The refit is the full nested fit (8 rungs x 500 bootstrap resamples) and
    lands on the first call of a process, on a path budgeted at 5 ms.
    """

    @staticmethod
    def _record() -> dict:
        import json
        from pathlib import Path

        repo = Path(__file__).resolve().parent.parent
        path = next(
            p for p in (repo / "h100-calibration-records").glob("*.json")
            if json.loads(p.read_text()).get("provenance", {}).get("raw_run_table")
        )
        return json.loads(path.read_text())

    @staticmethod
    def _forge(record: dict) -> dict:
        """Double a coefficient, fix every mirror, recompute both hashes."""
        import copy

        from vetch.calibration_store import (
            CalibrationIdentity,
            _measurement_key_payload,
            apply_attestation_hashes,
        )

        forged = copy.deepcopy(record)
        doubled = forged["coefficients"]["wh_per_1k_output"] * 2
        forged["coefficients"]["wh_per_1k_output"] = doubled
        forged["energy_basis"]["marginal"]["wh_per_1k_output"] = doubled
        identity = forged["identity"]
        idn = CalibrationIdentity(**{
            key: identity.get(key) for key in (
                "provider", "model", "gpu", "serving_engine", "precision",
                "instance_type", "visual_token_budget", "concurrency",
            )
        })
        return apply_attestation_hashes(forged, host_key=_measurement_key_payload(
            idn, forged.get("host"), forged.get("measurement_conditions"),
            forged.get("serving_config"),
        ))

    def test_off_by_default(self, monkeypatch) -> None:
        from vetch.calibration_store import refit_on_resolve

        monkeypatch.delenv("VETCH_CALIB_REFIT_ON_RESOLVE", raising=False)
        assert refit_on_resolve() is False
        for value in ("1", "true", "yes", "on", "TRUE"):
            monkeypatch.setenv("VETCH_CALIB_REFIT_ON_RESOLVE", value)
            assert refit_on_resolve() is True, value
        for value in ("0", "false", "", "maybe"):
            monkeypatch.setenv("VETCH_CALIB_REFIT_ON_RESOLVE", value)
            assert refit_on_resolve() is False, value

    def test_cheap_checks_run_regardless(self, monkeypatch) -> None:
        import copy

        from vetch.calibration_store import _runtime_record_valid

        monkeypatch.delenv("VETCH_CALIB_REFIT_ON_RESOLVE", raising=False)
        record = self._record()
        assert _runtime_record_valid(record) is True

        # A naive edit leaves the attestation hashes stale.
        naive = copy.deepcopy(record)
        naive["coefficients"]["wh_per_1k_output"] *= 2
        assert _runtime_record_valid(naive) is False

    def test_refit_is_what_catches_a_complete_forgery(self, monkeypatch) -> None:
        from vetch.calibration_store import _runtime_record_valid

        forged = self._forge(self._record())

        monkeypatch.delenv("VETCH_CALIB_REFIT_ON_RESOLVE", raising=False)
        assert _runtime_record_valid(forged) is True, (
            "documented limit of the default path: a forgery consistent with "
            "its own hashes is only caught by the refit"
        )

        monkeypatch.setenv("VETCH_CALIB_REFIT_ON_RESOLVE", "1")
        assert _runtime_record_valid(forged) is False

    def test_cli_validator_always_catches_it(self) -> None:
        from vetch.calibrate_validate import validate_calibration_record

        report = validate_calibration_record(self._forge(self._record()))
        assert report["ok"] is False
        assert any("wh_per_1k_output" in r for r in report["restatements"])


class TestSudoWrittenRecordsStayReadable:
    """A root-written 0600 record in the user's own directory is unreadable.

    Apple Silicon calibration needs root for powermetrics and records are 0600
    by design; under sudo that lands a root-owned owner-only file exactly where
    the SDK looks, and `_read_json` swallowed the PermissionError so the
    calibration simply appeared never to attach.
    """

    def test_chown_is_a_noop_without_sudo(self, tmp_path, monkeypatch) -> None:
        from vetch.calibration_store import _chown_to_invoking_user

        monkeypatch.delenv("SUDO_UID", raising=False)
        target = tmp_path / "rec.json"
        target.write_text("{}")
        before = target.stat().st_uid
        _chown_to_invoking_user(target)  # must not raise
        assert target.stat().st_uid == before

    def test_chown_uses_sudo_uid(self, tmp_path, monkeypatch) -> None:
        import os

        from vetch.calibration_store import _chown_to_invoking_user

        called: dict = {}
        monkeypatch.setenv("SUDO_UID", "501")
        monkeypatch.setenv("SUDO_GID", "20")
        monkeypatch.setattr(
            os, "chown",
            lambda p, u, g: called.update(path=p, uid=u, gid=g),
        )
        target = tmp_path / "rec.json"
        target.write_text("{}")
        _chown_to_invoking_user(target)
        assert called == {"path": target, "uid": 501, "gid": 20}

    def test_chown_failure_does_not_raise(self, tmp_path, monkeypatch) -> None:
        import os

        from vetch.calibration_store import _chown_to_invoking_user

        monkeypatch.setenv("SUDO_UID", "501")

        def boom(*_a, **_k):
            raise PermissionError("nope")

        monkeypatch.setattr(os, "chown", boom)
        target = tmp_path / "rec.json"
        target.write_text("{}")
        _chown_to_invoking_user(target)  # losing the record would be worse

    def test_unreadable_record_warns_rather_than_vanishing(
        self, tmp_path, monkeypatch, caplog
    ) -> None:
        import logging
        from pathlib import Path

        from vetch.calibration_store import _read_json

        target = tmp_path / "rec.json"
        target.write_text('{"a": 1}')

        real_read = Path.read_text

        def denied(self, *a, **k):
            if self == target:
                raise PermissionError("owner mismatch")
            return real_read(self, *a, **k)

        monkeypatch.setattr(Path, "read_text", denied)
        with caplog.at_level(logging.WARNING):
            assert _read_json(target) is None
        assert any("not readable by this user" in r.message for r in caplog.records)


class TestArchiveStaysReadable:
    """The archive is the only record of a run once the next one overwrites it.

    Created root-owned 0700 under sudo, it locked the operator out of their own
    previous calibrations -- and `ls` reports a forbidden directory as "total 0",
    so it looked empty rather than inaccessible.
    """

    def test_archived_record_and_dir_are_handed_back(self, tmp_path, monkeypatch) -> None:
        import os

        import vetch.calibration_store as store

        chowned: list = []
        monkeypatch.setenv("SUDO_UID", "501")
        monkeypatch.setenv("SUDO_GID", "20")
        monkeypatch.setattr(os, "chown", lambda p, u, g: chowned.append(str(p)))

        archive = tmp_path / "archive"
        archive.mkdir()
        archived = archive / "old.json"
        archived.write_text("{}")
        store._chown_to_invoking_user(archived)
        store._chown_to_invoking_user(archive)
        assert str(archived) in chowned
        assert str(archive) in chowned

    def test_writer_chowns_active_and_archived_paths(self, tmp_path, monkeypatch) -> None:
        import os

        import vetch.calibration_store as store

        chowned: list[str] = []
        monkeypatch.setenv("SUDO_UID", "501")
        monkeypatch.setenv("SUDO_GID", "20")
        monkeypatch.setattr(os, "chown", lambda p, _u, _g: chowned.append(str(p)))
        monkeypatch.setattr("vetch.calibrate.CALIBRATION_DIR", tmp_path)

        idn = _idn()
        store.write_record(build_record(_result(), idn, {}, timestamp=1.0))
        store.write_record(build_record(_result(), idn, {}, timestamp=2.0))

        assert str(tmp_path / "archive") in chowned
        assert any(path.startswith(str(tmp_path / "archive")) for path in chowned)
        assert any(path.startswith(str(tmp_path)) for path in chowned)
