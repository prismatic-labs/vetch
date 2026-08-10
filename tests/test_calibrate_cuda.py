"""Tests for the NVIDIA GPU (CUDA) VLM calibration path.

No GPU is required: the NVML monitor and the Ollama driver are replaced with a
synthetic simulator driven by a virtual clock, so the full
grid -> measure -> fit -> CalibrationResult flow (including idle subtraction) is
exercised deterministically and the recovered coefficients are checked against
the injected ground truth. Runs on both the NumPy and pure-Python fit paths.
"""

from __future__ import annotations

import sys

import pytest

import vetch.calibrate_cuda as cc
from vetch.calibrate import GPUMonitor

# ---------------------------------------------------------------------------
# Energy meter
# ---------------------------------------------------------------------------


class _CounterMonitor:
    """Fake GPUMonitor with an advancing energy counter."""

    def __init__(self):
        self.mj = 0.0
        self.energy_ok = True

    def energy_counter_available(self):
        return True

    def get_total_energy_mj(self):
        return self.mj if self.energy_ok else None

    def get_power_w(self):
        return 300.0


def test_energy_meter_counter_mode_mj_to_wh():
    mon = _CounterMonitor()
    meter = cc._CudaEnergyMeter(mon)
    start = meter.mark()
    mon.mj += 3_600_000.0  # exactly 1 Wh in millijoules
    end = meter.mark()
    assert abs(meter.energy_between(start, end) - 1.0) < 1e-9


def test_counter_none_midrun_raises():
    """A counter that worked at detection but returns None mid-run must fail
    loud, not silently fall through to zero-energy integration."""
    mon = _CounterMonitor()
    meter = cc._CudaEnergyMeter(mon)
    meter.mark()
    mon.energy_ok = False
    with pytest.raises(RuntimeError, match="no reading mid-run"):
        meter.mark()


def test_counter_backwards_raises():
    """A monotonic counter going backwards (driver reset/wrap) is invalid data,
    not a plausible zero-energy sample."""
    mon = _CounterMonitor()
    meter = cc._CudaEnergyMeter(mon)
    start = meter.mark()
    mon.mj -= 1000.0  # reset/wrap
    end = meter.mark()
    with pytest.raises(RuntimeError, match="went backwards"):
        meter.energy_between(start, end)


def test_fallback_power_unreadable_fails_before_grid():
    """If neither the counter nor power reads work, fail at meter start rather
    than after running the paid grid on zero-watt samples."""
    class _Dead:
        def energy_counter_available(self):
            return False

        def get_power_w(self):
            raise RuntimeError("NVML_ERROR_NOT_SUPPORTED")

    with pytest.raises(RuntimeError, match="neither the NVML energy counter"):
        with cc._CudaEnergyMeter(_Dead()):
            pass


def test_integrate_short_window_uses_nearest_power():
    """Fewer than two interior samples -> hold nearest power across the window,
    not discard the run as zero."""
    class _PowerOnly:
        def energy_counter_available(self):
            return False

        def get_power_w(self):
            return 100.0

    meter = cc._CudaEnergyMeter(_PowerOnly())
    meter._samples = [(0.0, 200.0)]  # one sample only
    # 200 W held across a 3600s window (in ms) = 200 Wh.
    wh = meter._integrate(0.0, 3_600_000.0)
    assert abs(wh - 200.0) < 1e-6


def test_integrate_covers_window_edges():
    """Trapezoid plus the [t0, first] and [last, t1] edge segments."""
    class _PowerOnly:
        def energy_counter_available(self):
            return False

        def get_power_w(self):
            return 0.0

    meter = cc._CudaEnergyMeter(_PowerOnly())
    # Two interior samples at 100 W; window extends 3600s before/after them.
    meter._samples = [(3_600_000.0, 100.0), (7_200_000.0, 100.0)]
    wh = meter._integrate(0.0, 10_800_000.0)
    # 100 W held across the full 3h window = 300 Wh (edges included).
    assert abs(wh - 300.0) < 1e-6


# ---------------------------------------------------------------------------
# Net energy (idle subtraction)
# ---------------------------------------------------------------------------


def test_net_energy_subtracts_avg_idle():
    # 300 W gross for 3600s = 300 Wh; 80 W avg idle over same window = 80 Wh.
    net = cc._net_energy_wh(raw_wh=300.0, avg_idle_watts=80.0, duration_ms=3_600_000.0)
    assert abs(net - 220.0) < 1e-9


def test_net_energy_clamped_nonnegative():
    net = cc._net_energy_wh(raw_wh=0.001, avg_idle_watts=80.0, duration_ms=3_600_000.0)
    assert net == 0.0


# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------


def _good_fit():
    from vetch.calibrate_metal import FitResult

    return FitResult(
        intercept_wh=0.0005, wh_per_image=0.0008, wh_per_1k_input=0.0003,
        wh_per_1k_output=0.0018, r2=0.99, condition_number=5.0,
        input_ci95=(0.0002, 0.0004), output_ci95=(0.0016, 0.002),
        image_ci95=(0.0007, 0.0009), valid=True, invalid_reasons=[],
    )


def test_rejection_reasons_clean_run():
    assert cc._cuda_rejection_reasons(_good_fit(), samples=22, idle_drift_pct=2.0) == []


def test_fallback_is_not_a_rejection():
    """The power-sampling fallback must NOT invalidate an otherwise-good run."""
    assert cc._cuda_rejection_reasons(_good_fit(), samples=22, idle_drift_pct=2.0) == []


def test_rejection_reasons_flags_problems():
    reasons = cc._cuda_rejection_reasons(_good_fit(), samples=5, idle_drift_pct=20.0)
    assert any("samples=5" in r for r in reasons)
    assert any("idle_drift" in r for r in reasons)


# ---------------------------------------------------------------------------
# GPUMonitor energy counter
# ---------------------------------------------------------------------------


def test_get_total_energy_mj_returns_none_on_unsupported(monkeypatch):
    import types

    fake = types.ModuleType("pynvml")
    fake.nvmlDeviceGetTotalEnergyConsumption = lambda h: (_ for _ in ()).throw(
        RuntimeError("NVML_ERROR_NOT_SUPPORTED")
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    m = GPUMonitor()
    m._handle = object()
    assert m.get_total_energy_mj() is None
    assert m.energy_counter_available() is False


# ---------------------------------------------------------------------------
# Filename sanitization + provider-agnostic auto-load (calibrate.py)
# ---------------------------------------------------------------------------


def test_save_load_namespaced_model(monkeypatch, tmp_path):
    """A namespaced model (org/model:tag) must not try to write into a
    nonexistent subdirectory, and must round-trip."""
    import vetch.calibrate as cal

    monkeypatch.setattr(cal, "CALIBRATION_DIR", tmp_path)
    res = cal.CalibrationResult(
        model="org/model:tag", provider="ollama",
        wh_per_1k_input=0.1, wh_per_1k_output=0.2, tier=0, samples=20,
    )
    cal.save_calibration(res)
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1  # flat file, no subdir attempt
    loaded = cal.load_calibration("ollama", "org/model:tag")
    assert loaded is not None and loaded.wh_per_1k_input == 0.1


def test_provider_agnostic_autoload(monkeypatch, tmp_path):
    """A calibration saved under one self-hosted label loads for the same model
    under another self-hosted label (ollama <-> vllm/self-hosted)."""
    import vetch.calibrate as cal

    monkeypatch.setattr(cal, "CALIBRATION_DIR", tmp_path)
    res = cal.CalibrationResult(
        model="gemma-4-31b-it", provider="ollama",
        wh_per_1k_input=0.3, wh_per_1k_output=0.9, tier=0, samples=22,
    )
    cal.save_calibration(res)
    # Look it up under a different self-hosted label:
    loaded = cal.load_calibration("self-hosted", "gemma-4-31b-it")
    assert loaded is not None
    assert loaded.wh_per_1k_output == 0.9


def test_provider_fallback_prefers_active(monkeypatch, tmp_path):
    """Cross-provider reuse (within local-equivalent labels) skips an inactive
    sibling and returns the active one."""
    import vetch.calibrate as cal

    monkeypatch.setattr(cal, "CALIBRATION_DIR", tmp_path)
    cal.save_calibration(cal.CalibrationResult(  # local-equivalent, but inactive
        model="m", provider="ollama", wh_per_1k_input=1.0, wh_per_1k_output=1.0,
        tier=0, samples=20, active=False,
    ))
    cal.save_calibration(cal.CalibrationResult(  # local-equivalent, active
        model="m", provider="vllm", wh_per_1k_input=2.0, wh_per_1k_output=2.0,
        tier=0, samples=20, active=True,
    ))
    loaded = cal.load_calibration("self-hosted", "m")  # self-hosted reuses vllm
    assert loaded is not None
    assert loaded.provider == "vllm" and loaded.active is True


# ---------------------------------------------------------------------------
# End-to-end with a virtual-clock energy simulator (runs on both fit paths)
# ---------------------------------------------------------------------------


def test_calibrate_cuda_recovers_coefficients(monkeypatch, tmp_path):
    # Ground-truth active-energy model (Wh) plus a constant idle draw that the
    # code must subtract via the pre/post average.
    B0, BIMG, BIN, BOUT = 0.0006, 0.0009, 0.00031, 0.00175
    IDLE_W = 80.0
    PROBE_DELTA = 260  # model "includes" this many image tokens per image
    clock = {"s": 0.0}
    state = {"mj": 0.0}

    def _advance(seconds):
        clock["s"] += seconds
        state["mj"] += IDLE_W * seconds * 1000.0  # idle energy (W*s -> mJ)

    class FakeMonitor:
        def __init__(self, device_id=0):
            self.device_id = device_id

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_gpu_info(self):
            return {"name": "NVIDIA H100 80GB HBM3", "driver_version": "550.90",
                    "memory_total_mb": 81559}

        def energy_counter_available(self):
            return True

        def get_total_energy_mj(self):
            return state["mj"]

        def get_power_w(self):
            return IDLE_W

        def device_count(self):
            return 1

        def compute_process_count(self):
            return 1

        def get_power_limit_w(self):
            return 700.0

        def get_clocks(self):
            return {"sm_clock_mhz": 1900, "sm_max_clock_mhz": 1980,
                    "mem_clock_mhz": 2619, "applications_sm_clock_mhz": 1900}

    def fake_ollama(base_url, model, prompt, image_b64=None, max_tokens=32):
        n_img = 1 if image_b64 else 0
        text_only = max(4, len(prompt) // 4)
        # "includes" model: prompt_eval_count folds in the image tokens.
        text_tokens = text_only + (PROBE_DELTA if n_img else 0)
        out_tokens = max_tokens
        _advance(0.25)  # call duration: idle accrues over the window
        active_wh = B0 + BIMG * n_img + BIN * (text_only / 1000.0) + BOUT * (out_tokens / 1000.0)
        state["mj"] += active_wh * 3_600_000.0
        return text_tokens, out_tokens

    def fake_probe(gen, url, m, img):
        return {
            "image_token_accounting": "includes",  # nosec B105
            "delta_prompt_eval_count_with_image": PROBE_DELTA,
        }

    monkeypatch.setattr("time.sleep", lambda s: _advance(s))
    monkeypatch.setattr("time.monotonic", lambda: clock["s"])
    monkeypatch.setattr(cc, "assert_cuda", lambda: None)
    monkeypatch.setattr(cc, "_require_numpy", lambda: None)
    monkeypatch.setattr(cc, "_check_ollama", lambda url: "0.5.0")
    monkeypatch.setattr(cc, "GPUMonitor", FakeMonitor)
    monkeypatch.setattr(cc, "_ollama_generate", fake_ollama)
    monkeypatch.setattr(cc, "_load_image_set", lambda: (["poolimg"], "synthetic"))
    monkeypatch.setattr(cc, "_unique_image_b64", lambda seed, size=512: f"img-{seed}")
    monkeypatch.setattr(cc, "_warmup_image_b64", lambda vlm, pool: "warm" if vlm else None)
    monkeypatch.setattr(cc, "_probe_image_tokens", fake_probe)
    monkeypatch.setattr("vetch.calibrate.CALIBRATION_DIR", tmp_path)

    result = cc.calibrate_cuda(
        "gemma-4-31b-it", provider="self-hosted",
        precision="bf16", serving_engine="vllm", verbose=False,
    )

    assert result.tier == 0
    assert result.samples >= 8
    assert result.gpu_name == "NVIDIA H100 80GB HBM3"
    # Idle subtraction worked: coefficients recovered despite the 80 W idle draw.
    assert result.wh_per_image is not None
    assert abs(result.wh_per_image - BIMG) < BIMG * 0.15
    assert abs(result.wh_per_1k_input - BIN) < BIN * 0.15
    assert abs(result.wh_per_1k_output - BOUT) < BOUT * 0.15
    # #3: the saved visual-token count is the MEASURED probe delta, not 729.
    assert result.visual_tokens_per_image == PROBE_DELTA
    # A single data-rich v1 record was written and round-trips with identity+provenance.
    import json
    recs = list(tmp_path.glob("*.json"))
    assert len(recs) == 1
    rec = json.loads(recs[0].read_text())
    assert rec["schema_version"] == 1
    assert rec["identity"]["gpu"] == "h100-sxm-80gb"
    assert rec["identity"]["precision"] == "bf16"
    assert rec["identity"]["serving_engine"] == "vllm"
    assert rec["provenance"]["energy_domain"] == "gpu_board"
    assert rec["provenance"]["batch_size"] == 1
    assert rec["provenance"]["enforced_power_limit_w"] == 700.0
    assert len(rec["provenance"]["raw_run_table"]) == result.samples
    assert rec["content_hash"].startswith("sha256:")
    # Standardization fields for reproducibility / comparability.
    assert rec["provenance"]["grid_seed"] == cc._GRID_SEED
    assert rec["provenance"]["grid_design_id"].startswith("sym-")
    assert rec["provenance"]["sm_clock_locked"] is True
    assert rec["provenance"]["clocks"]["sm_max_clock_mhz"] == 1980
    assert rec["provenance"]["tensor_parallel_size"] is None


def _minimal_counter_monitor(state):
    class FakeMonitor:
        def __init__(self, device_id=0):
            self.device_id = device_id

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_gpu_info(self):
            return {"name": "NVIDIA H100 80GB HBM3", "driver_version": "550.90",
                    "memory_total_mb": 81559}

        def energy_counter_available(self):
            state["counter_checks"] = state.get("counter_checks", 0) + 1
            return True

        def get_total_energy_mj(self):
            return state["mj"]

        def get_power_w(self):
            return 0.0

        def device_count(self):
            return 1

        def compute_process_count(self):
            return 1

        def get_power_limit_w(self):
            return 700.0

        def get_clocks(self):
            return {"sm_clock_mhz": 1900, "sm_max_clock_mhz": 1980,
                    "mem_clock_mhz": 2619, "applications_sm_clock_mhz": 1900}

    return FakeMonitor


def _patch_common(monkeypatch, tmp_path, monitor_cls, fake_ollama, fake_probe):
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setattr(cc, "assert_cuda", lambda: None)
    monkeypatch.setattr(cc, "_require_numpy", lambda: None)
    monkeypatch.setattr(cc, "_check_ollama", lambda url: "0.5.0")
    monkeypatch.setattr(cc, "GPUMonitor", monitor_cls)
    monkeypatch.setattr(cc, "_ollama_generate", fake_ollama)
    monkeypatch.setattr(cc, "_load_image_set", lambda: (["poolimg"], "synthetic"))
    monkeypatch.setattr(cc, "_unique_image_b64", lambda seed, size=512: f"img-{seed}")
    monkeypatch.setattr(cc, "_warmup_image_b64", lambda vlm, pool: "warm" if vlm else None)
    monkeypatch.setattr(cc, "_probe_image_tokens", fake_probe)
    monkeypatch.setattr("vetch.calibrate.CALIBRATION_DIR", tmp_path)


def test_vlm_with_all_image_runs_failed_withholds_visual_term(monkeypatch, tmp_path):
    """#1: a VLM whose image runs all fail (slipping under the >half gate) must
    withhold wh_per_image (None, not 0) and be marked inactive — never present
    visual energy as complete."""
    state = {"mj": 0.0}

    def fake_ollama(base_url, model, prompt, image_b64=None, max_tokens=32):
        # Warmup (max_tokens=10) succeeds so the model presents as a VLM, but
        # every image run in the grid fails (e.g. image OOM on real workloads).
        if image_b64 is not None and max_tokens != 10:
            raise RuntimeError("image run OOM")
        state["mj"] += (0.001 + 0.0005 * max_tokens) * 3_600_000.0
        return max(4, len(prompt) // 4), max_tokens

    # Probe succeeds -> model_is_vlm=True, but every actual image run will fail.
    fake_probe = lambda gen, url, m, img: {  # noqa: E731
        "image_token_accounting": "excludes",  # nosec B105
        "delta_prompt_eval_count_with_image": 0,
    }
    _patch_common(monkeypatch, tmp_path, _minimal_counter_monitor(state), fake_ollama, fake_probe)

    result = cc.calibrate_cuda(
        "gemma-4-31b-it", provider="ollama", precision="bf16",
    )

    assert result.wh_per_image is None          # withheld, not 0
    assert result.visual_tokens_per_image is None
    assert result.active is False
    assert any("vlm_image_runs" in r for r in (result.rejection_reasons or []))


def test_energy_counter_availability_checked_once(monkeypatch, tmp_path):
    """#3: measurement mode is decided once (by the meter). calibrate_cuda must
    not independently re-query energy_counter_available()."""
    state = {"mj": 0.0, "counter_checks": 0}

    def fake_ollama(base_url, model, prompt, image_b64=None, max_tokens=32):
        state["mj"] += (0.001 + 0.0005 * max_tokens) * 3_600_000.0
        n_img = 1 if image_b64 else 0
        return max(4, len(prompt) // 4) + n_img, max_tokens

    fake_probe = lambda gen, url, m, img: {  # noqa: E731
        "image_token_accounting": "excludes",  # nosec B105
        "delta_prompt_eval_count_with_image": 0,
    }
    _patch_common(monkeypatch, tmp_path, _minimal_counter_monitor(state), fake_ollama, fake_probe)

    cc.calibrate_cuda("gemma-4-31b-it", provider="ollama", precision="bf16")

    # Exactly one availability check (the meter's), not a second in calibrate_cuda.
    assert state["counter_checks"] == 1


# ---------------------------------------------------------------------------
# vLLM / OpenAI-compatible backend
# ---------------------------------------------------------------------------


def test_openai_compat_generate_parses_usage(monkeypatch):
    """POST shape + usage parsing: image-first content, thinking disabled,
    usage.prompt_tokens/completion_tokens returned."""
    import json
    import urllib.request

    captured: dict = {}

    class FakeResp:
        def __init__(self, payload):
            self._p = payload

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return self._p

    def fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data)
        return FakeResp(
            json.dumps({"usage": {"prompt_tokens": 123, "completion_tokens": 45}}).encode()
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    pt, ct = cc._openai_compat_generate(
        "http://x/v1", "google/gemma-4-31B-it", "hello", image_b64="B64", max_tokens=7
    )
    assert (pt, ct) == (123, 45)
    assert captured["url"].endswith("/chat/completions")
    content = captured["body"]["messages"][0]["content"]
    assert content[0]["type"] == "image_url" and content[1]["type"] == "text"  # image first
    assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["body"]["max_tokens"] == 7
    assert captured["body"]["temperature"] == 0.0


def test_calibrate_cuda_openai_backend_dispatch(monkeypatch, tmp_path):
    """backend='openai' uses the OpenAI-compatible generate/health path and the
    vLLM default base URL."""
    state = {"mj": 0.0}
    used = {"gen": 0, "health": 0}

    def fake_gen(base_url, model, prompt, image_b64=None, max_tokens=32):
        used["gen"] += 1
        state["mj"] += (0.001 + 0.0005 * max_tokens) * 3_600_000.0
        n_img = 1 if image_b64 else 0
        return max(4, len(prompt) // 4) + n_img, max_tokens

    def fake_health(url):
        used["health"] += 1
        assert url.endswith("/v1")  # per-backend default resolved
        return "vllm-ok"

    fake_probe = lambda gen, url, m, img: {  # noqa: E731
        "image_token_accounting": "excludes",  # nosec B105
        "delta_prompt_eval_count_with_image": 0,
    }
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setattr(cc, "assert_cuda", lambda: None)
    monkeypatch.setattr(cc, "_require_numpy", lambda: None)
    monkeypatch.setattr(cc, "_check_openai_server", fake_health)
    monkeypatch.setattr(cc, "_openai_compat_generate", fake_gen)
    monkeypatch.setattr(cc, "GPUMonitor", _minimal_counter_monitor(state))
    monkeypatch.setattr(cc, "_load_image_set", lambda: (["p"], "synthetic"))
    monkeypatch.setattr(cc, "_unique_image_b64", lambda seed, size=512: f"img-{seed}")
    monkeypatch.setattr(cc, "_warmup_image_b64", lambda vlm, pool: "warm" if vlm else None)
    monkeypatch.setattr(cc, "_probe_image_tokens", fake_probe)
    monkeypatch.setattr("vetch.calibrate.CALIBRATION_DIR", tmp_path)

    result = cc.calibrate_cuda(
        "google/gemma-4-31B-it", provider="vllm", backend="openai",
        precision="bf16", serving_engine="vllm",
    )

    assert used["health"] == 1 and used["gen"] > 0
    assert result.tier == 0
    assert result.gpu_name == "NVIDIA H100 80GB HBM3"


def test_calibrate_cuda_requires_precision(monkeypatch, tmp_path):
    monkeypatch.setattr(cc, "assert_cuda", lambda: None)
    monkeypatch.setattr(cc, "_require_numpy", lambda: None)
    with pytest.raises(ValueError, match="precision is required"):
        cc.calibrate_cuda("m", precision=None)
    with pytest.raises(ValueError, match="precision is required"):
        cc.calibrate_cuda("m", precision="  ")


def test_calibrate_cuda_openai_requires_serving_engine(monkeypatch, tmp_path):
    monkeypatch.setattr(cc, "assert_cuda", lambda: None)
    monkeypatch.setattr(cc, "_require_numpy", lambda: None)
    with pytest.raises(ValueError, match="serving_engine is required"):
        cc.calibrate_cuda("m", backend="openai", precision="bf16")


# ---------------------------------------------------------------------------
# Batched / amortization curve
# ---------------------------------------------------------------------------


def test_fit_amortization_recovers_a_b():
    # Ground truth: Wh/1k_out = 0.52/C + 0.01
    points = [(c, 0.52 / c + 0.01) for c in (1, 2, 4, 8, 16, 32)]
    fit = cc._fit_amortization_curve(points)
    assert abs(fit["a"] - 0.52) < 1e-6
    assert abs(fit["b"] - 0.01) < 1e-6
    assert fit["r2"] > 0.999


def test_measure_concurrency_level_recovers_wh_per_1k(monkeypatch):
    """Synthetic counter + generate: injected a/C+b recovered within tolerance."""
    a, b = 0.50, 0.02
    idle_w = 80.0
    out_tokens_per_req = 64
    n_req = 16

    class _Mon:
        def __init__(self):
            self.mj = 0.0
            self._t0 = None

        def energy_counter_available(self):
            return True

        def get_total_energy_mj(self):
            # Advance energy with wall time: idle + decode power derived from a/C+b.
            import time as _t
            now = _t.monotonic()
            if self._t0 is None:
                self._t0 = now
                return self.mj
            dt = now - self._t0
            self._t0 = now
            # During generate blocks the test advances mj explicitly; idle measure
            # uses sleep — advance by idle only here.
            self.mj += idle_w * dt * 1000.0  # W * s * 1000 = mJ
            return self.mj

        def get_power_w(self):
            return idle_w + 100.0

        def get_clocks(self):
            return {"sm": 1000, "mem": 1000}

    mon = _Mon()
    meter = cc._CudaEnergyMeter(mon)

    # Patch measure idle to a constant to avoid sleep timing noise.
    monkeypatch.setattr(cc, "_measure_idle_watts", lambda *a, **k: idle_w)
    monkeypatch.setattr(cc, "_scrape_vllm_num_running", lambda url: None)

    def fake_generate(base_url, model, prompt, image_b64=None, max_tokens=32, **kw):
        # Simulate work duration proportional to tokens so energy_between spans time.
        import time as _t
        c = fake_generate.concurrency  # type: ignore[attr-defined]
        # Energy for this request: (a/c + b) Wh per 1k out * tokens
        wh = (a / c + b) * out_tokens_per_req / 1000.0
        # Convert to mJ and add on top of idle contribution during the same dt.
        # Sleep briefly so meter marks see distinct timestamps; inject energy
        # directly into the counter for the decode portion.
        _t.sleep(0.002)
        mon.mj += wh * 3_600_000.0
        return (96, out_tokens_per_req)

    for c in (1, 4, 16):
        fake_generate.concurrency = c  # type: ignore[attr-defined]
        # Net energy path: meter.energy_between includes idle*dt + our injected Wh.
        # _net_energy_wh subtracts idle*dt, leaving injected Wh.
        level = cc.measure_concurrency_level(
            meter=meter,
            monitor=mon,  # type: ignore[arg-type]
            generate=fake_generate,
            base_url="http://localhost:8000/v1",
            model="m",
            concurrency=c,
            requests_per_level=n_req,
            out_tokens=out_tokens_per_req,
            in_words=32,
            image_b64=None,
            fixed_output=True,
            idle_seconds=0.0,
            settle_seconds=0.0,
        )
        expected = a / c + b
        assert abs(level["wh_per_1k_out"] - expected) / expected < 0.15, (
            c, level["wh_per_1k_out"], expected
        )


def test_calibrate_cuda_batched_writes_curve(monkeypatch, tmp_path):
    """End-to-end batched path writes one record per C with amortization provenance."""
    import json

    import vetch.calibrate as cal

    monkeypatch.setattr(cal, "CALIBRATION_DIR", tmp_path)
    monkeypatch.setattr(cc, "assert_cuda", lambda: None)
    monkeypatch.setattr(cc, "_require_numpy", lambda: None)
    monkeypatch.setattr(cc, "_check_openai_server", lambda url: "fake")
    monkeypatch.setattr(cc, "canonical_gpu", lambda name: ("h100-sxm-80gb", True))

    a, b = 0.48, 0.015
    idle_w = 70.0

    class _MonCtx:
        def __init__(self, device_id=0):
            self.mj = 0.0
            self.device_id = device_id

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def energy_counter_available(self):
            return True

        def get_total_energy_mj(self):
            return self.mj

        def get_power_w(self):
            return idle_w + 50.0

        def get_power_limit_w(self):
            return 700.0

        def get_clocks(self):
            return {"sm": 1410, "mem": 1593}

        def get_gpu_info(self):
            return {
                "name": "NVIDIA H100 80GB HBM3",
                "memory_total_mb": 81559,
                "driver_version": "550.90",
            }

        def get_compute_process_count(self):
            return 1

    monkeypatch.setattr(cc, "GPUMonitor", _MonCtx)
    monkeypatch.setattr(cc, "_measure_idle_watts", lambda *a, **k: idle_w)
    monkeypatch.setattr(cc, "_scrape_vllm_num_running", lambda url: None)

    def fake_generate(base_url, model, prompt, image_b64=None, max_tokens=32, **kw):
        c = fake_generate.concurrency
        wh = (a / c + b) * 64 / 1000.0
        # Attach to the open meter via module-level hack: advance in measure by
        # patching energy_between — simpler: patch measure_concurrency_level.
        return (90, 64)

    # Drive measure_concurrency_level with known outputs.
    def fake_measure(**kwargs):
        c = kwargs["concurrency"]
        return {
            "concurrency": c,
            "wh_per_1k_out": a / c + b,
            "wh_per_image": None,
            "achieved_concurrency": float(c),
            "net_wh": 0.1,
            "gross_wh": 0.2,
            "idle_watts": idle_w,
            "output_tokens": 64 * kwargs["requests_per_level"],
            "prompt_tokens": 90 * kwargs["requests_per_level"],
            "wall_s": 1.0,
            "power_before_w": idle_w + 50,
            "power_after_w": idle_w + 50,
            "block_drift_pct": 1.0,
            "clocks_before": {},
            "clocks_after": {},
        }

    monkeypatch.setattr(cc, "measure_concurrency_level", fake_measure)
    monkeypatch.setattr(cc, "_load_image_set", lambda: ([], "none"))
    monkeypatch.setattr(
        cc, "_get_visual_tokens_per_image", lambda m: (0, "not_applicable")
    )

    results = cc.calibrate_cuda_batched(
        "google/gemma-4-31B-it",
        provider="self-hosted",
        backend="openai",
        serving_engine="vllm",
        precision="bf16",
        concurrencies=(1, 4, 32),
        requests_per_level=8,
        out_tokens=64,
        measure_images=False,
    )
    assert len(results) == 3
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 3
    recs = [json.loads(p.read_text()) for p in files]
    concs = sorted(r["identity"]["concurrency"] for r in recs)
    assert concs == [1, 4, 32]
    for r in recs:
        assert r["provenance"]["calibration_mode"] == "batched"
        assert "amortization_curve" in r["provenance"]
        fit = r["provenance"]["amortization_fit"]
        assert abs(fit["a"] - a) < 1e-6
        assert abs(fit["b"] - b) < 1e-6


def test_unique_media_generators_are_distinct_per_seed():
    from vetch.calibrate_metal import (
        _unique_audio_b64,
        _unique_media_b64,
        _unique_video_b64,
    )

    assert _unique_audio_b64(1) != _unique_audio_b64(2)
    assert _unique_video_b64(1) != _unique_video_b64(2)
    assert _unique_media_b64("audio", 7) == _unique_audio_b64(7)
    assert _unique_media_b64("video", 7) == _unique_video_b64(7)
    assert len(_unique_media_b64("image", 7)) > 0


def test_calibrate_cuda_rejects_audio_on_ollama_backend():
    import pytest

    from vetch import calibrate_cuda as cc

    with pytest.raises(ValueError, match="backend='openai'"):
        cc.calibrate_cuda(
            "m",
            precision="bf16",
            backend="ollama",
            modality="audio",
        )
