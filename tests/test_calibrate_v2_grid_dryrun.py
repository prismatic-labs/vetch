"""End-to-end dry run of ``--grid v2`` against a mocked GPU and server.

The v2 grid is the design that carries a prefill isolation arm, and no record
in this repository has one. Before spending rented-GPU time to get the first,
exercise the whole path here: a wiring bug found at $0 is cheaper than one
found at $3/hr with 4 machines available.

Ground truth is a linear model with prefill at 0.34% of decode, the ratio the
H100's bandwidth-to-compute figure implies, so the run also checks that the
design recovers a prefill term of a realistic size rather than a convenient one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

import vetch.calibrate_cuda as cc  # noqa: E402
from vetch.calibrate_v2 import v2_grid_design  # noqa: E402

IDLE_W = 124.0
B_OUT = 0.4863            # Wh per 1k output tokens (measured, Qwen2.5-7B/H100)
B_IN = B_OUT * 0.0034     # prefill, from bandwidth / compute
B0 = 0.004                # per-request fixed cost


def test_v2_grid_identifies_prefill_on_a_clean_mock(monkeypatch, tmp_path):
    clock = {"s": 0.0}
    state = {"mj": 0.0}
    calls: list[tuple[int, dict]] = []

    def advance(seconds: float) -> None:
        clock["s"] += seconds
        state["mj"] += IDLE_W * seconds * 1000.0

    class FakeMonitor:
        def __init__(self, device_id: int = 0) -> None:
            self.device_id = device_id

        def __enter__(self):
            return self

        def __exit__(self, *_a):
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

        def get_clocks(self):
            return {"sm_clock_mhz": 1980, "locked": False}

        def get_power_limit_w(self):
            return 700.0

        def compute_process_count(self):
            return 1

        def device_count(self):
            return 1

        def get_memory_mb(self):
            return 81559

    def fake_generate(base_url, model, prompt, image_b64=None, max_tokens=32, **_kw):
        calls.append((max_tokens, dict(_kw)))
        approx_in = max(4, len(prompt) // 4)
        advance(approx_in * 1.41e-5 + max_tokens * 4.2e-3)
        active_wh = B0 + B_IN * approx_in / 1000.0 + B_OUT * max_tokens / 1000.0
        state["mj"] += active_wh * 3_600_000.0
        return approx_in, max_tokens

    monkeypatch.setattr(cc, "assert_cuda", lambda: None)
    monkeypatch.setattr(cc, "GPUMonitor", FakeMonitor)
    monkeypatch.setattr(cc, "_require_numpy", lambda: None)
    monkeypatch.setattr(cc, "_check_openai_server", lambda url: "vllm/0.27.1")
    monkeypatch.setattr(cc, "_openai_compat_generate", fake_generate)
    monkeypatch.setattr(cc, "_load_image_set", lambda: (["img"], "synthetic"),
                        raising=False)
    monkeypatch.setattr(cc, "_unique_media_b64",
                        lambda modality, seed, **kw: f"m-{seed}", raising=False)
    monkeypatch.setattr(cc, "_warmup_image_b64",
                        lambda vlm, pool: None, raising=False)

    def no_vision(*_a, **_kw):
        # Text-only calibration is reached by the image probe failing, which is
        # what a non-VLM does. The v2 grid's text rows are what we care about.
        raise RuntimeError("model does not accept images")

    monkeypatch.setattr(cc, "_probe_image_tokens", no_vision)
    # One coherent fake clock: mark() reads time.monotonic() and sleeps advance
    # it, so idle power must come out at exactly IDLE_W rather than a divide-by-
    # near-zero artefact.
    monkeypatch.setattr(cc.time, "sleep", advance)
    monkeypatch.setattr(cc.time, "monotonic", lambda: clock["s"])
    monkeypatch.setattr("vetch.calibrate.CALIBRATION_DIR", tmp_path)

    grid = v2_grid_design(vision=False, quick=False)
    assert any(g["arm"] == "prefill" for g in grid)

    result = cc.calibrate_cuda(
        "Qwen/Qwen2.5-7B-Instruct", provider="self-hosted", precision="bf16",
        serving_engine="vllm", backend="openai", grid="v2", verbose=False,
    )

    # The point of the v2 design, and the thing no record has ever shown:
    # with a prefill isolation arm, the input axis becomes identifiable.
    assert result.wh_per_1k_output is not None
    assert result.wh_per_1k_output == pytest.approx(B_OUT, rel=0.15)
    assert result.wh_per_1k_input is not None, (
        "the v2 grid carries a prefill arm; if input is still unidentified the "
        "design cannot deliver the result the protocol is built around"
    )
    assert result.wh_per_1k_input == pytest.approx(B_IN, rel=0.5)
    assert result.active is True
    assert any(
        max_tokens == 4096
        and kwargs.get("min_tokens") == 4096
        and kwargs.get("ignore_eos") is True
        for max_tokens, kwargs in calls
    ), "CUDA v2 must pin the decode arm's output length on vLLM"

    written = list(Path(tmp_path).glob("*.json"))
    assert len(written) == 1
    import json
    record = json.loads(written[0].read_text())
    status = record["coefficients"]["status"]
    assert status["wh_per_1k_input"] == "identified"
    assert status["wh_per_1k_output"] == "identified"
    gate = record["provenance"]["fit"]["gate"]
    assert gate["max_relative_ci_half_width"] is not None


def test_prefill_arm_clears_the_power_sampling_floor():
    """A prefill point must occupy a measurable window.

    Pinning output at 8 tokens is what isolates prefill, so a single request can
    finish in under 100 ms -- below the duration floor that applies on the
    power-sampling path. On a rented H100 with the counter unavailable this
    dropped all 17 rows and the fit had nothing to work with.
    """
    from vetch.calibrate_v2 import (
        POWER_SAMPLING_DURATION_FLOOR_MS,
        PREFILL_TOKENS_PER_MS,
        prefill_repeats_for,
        v2_grid_design,
    )

    for spec in v2_grid_design(vision=False, quick=False):
        if spec["arm"] != "prefill":
            continue
        window_ms = spec["requests"] * spec["approx_text_tokens"] / PREFILL_TOKENS_PER_MS
        assert window_ms > POWER_SAMPLING_DURATION_FLOOR_MS, (
            f"{spec['approx_text_tokens']}-token prefill point projects a "
            f"{window_ms:.0f} ms window, under the {POWER_SAMPLING_DURATION_FLOOR_MS:.0f} ms floor"
        )
    assert prefill_repeats_for(1024) > prefill_repeats_for(32768)


def test_arm_detection_is_per_request_not_per_row():
    """Aggregated rows carry summed tokens; the arm describes the design point."""
    from vetch.calibrate_v2 import input_design_leverage

    aggregated = [
        {"text_tokens": 4096 * 30, "output_tokens": 8 * 30, "requests": 30, "n_images": 0},
        {"text_tokens": 32768 * 4, "output_tokens": 8 * 4, "requests": 4, "n_images": 0},
        {"text_tokens": 32 * 1, "output_tokens": 4096, "requests": 1, "n_images": 0},
    ]
    ok, why = input_design_leverage(aggregated)
    assert ok, why
    # Read as raw totals the same design looks like it has no arm at all.
    as_totals = [{**r, "requests": 1} for r in aggregated]
    assert not input_design_leverage(as_totals)[0]


def test_contended_gpu_aborts_before_any_grid_request(monkeypatch, tmp_path):
    """A contended box must cost one idle window, not a whole grid."""
    import pytest

    calls = {"generate": 0}

    class ContendedMonitor:
        """A neighbour saturating the card. Our own processes look normal."""

        def __init__(self, device_id: int = 0) -> None:
            self.device_id = device_id

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def get_gpu_info(self):
            return {"name": "NVIDIA H100 80GB HBM3", "driver_version": "595.84",
                    "memory_total_mb": 81559}

        def energy_counter_available(self):
            return True

        def get_total_energy_mj(self):
            return 1e9

        def get_power_w(self):
            return 698.0

        def get_utilization_pct(self):
            return 100          # the decisive signal

        def compute_process_count(self):
            return 1            # a container cannot see the neighbour

        def get_clocks(self):
            return {"sm_clock_mhz": 1980, "locked": False}

        def get_power_limit_w(self):
            return 700.0

        def device_count(self):
            return 1

    def counted_generate(*_a, max_tokens: int = 32, **_kw):
        calls["generate"] += 1
        return 100, max_tokens

    monkeypatch.setattr(cc, "assert_cuda", lambda: None)
    monkeypatch.setattr(cc, "_require_numpy", lambda: None)
    monkeypatch.setattr(cc, "GPUMonitor", ContendedMonitor)
    monkeypatch.setattr(cc, "_check_openai_server", lambda url: "vllm/0.28.0")
    monkeypatch.setattr(cc, "server_max_model_len", lambda url: 32768)
    monkeypatch.setattr(cc, "_openai_compat_generate", counted_generate)
    monkeypatch.setattr(cc, "_load_image_set", lambda: (["i"], "synthetic"),
                        raising=False)
    monkeypatch.setattr(cc, "_probe_image_tokens",
                        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("no vlm")))
    monkeypatch.setattr(cc.time, "sleep", lambda _s: None)
    monkeypatch.setattr("vetch.calibrate.CALIBRATION_DIR", tmp_path)

    with pytest.raises(RuntimeError, match="Idle baseline is not idle"):
        cc.calibrate_cuda(
            "Qwen/Qwen2.5-7B-Instruct", provider="self-hosted", precision="bf16",
            serving_engine="vllm", backend="openai", grid="v2", verbose=False,
        )

    # A warmup before the idle window is expected and cheap. What must not
    # happen is the grid: the v2 design issues ~199 requests, and the run that
    # motivated this check spent all of them before reporting the problem.
    grid_requests = sum(
        spec.get("requests", 1) for spec in v2_grid_design(vision=False, quick=False)
    )
    assert calls["generate"] < 5, (
        f"aborted after {calls['generate']} requests; the grid is {grid_requests}"
    )
    assert not list(tmp_path.glob("*.json")), "wrote a record from a bad baseline"


def test_aggregated_rows_are_normalised_to_one_request():
    """Aggregation is noise reduction, not a different design point.

    Sizing repeats to a fixed timing window makes TOTAL input nearly constant
    across the prefill arm. Measured on an H100: points meant to span 1k-16k
    tokens produced totals of 98648, 93450 and 92805, and the resulting design
    had a condition number of 58.7. Dividing through by the request count
    restores the per-request span and drops it to 4.5.
    """
    from vetch.calibrate_v2 import fit_nested_models

    # Per-request design points spanning 811..12331 input tokens, aggregated to
    # a roughly constant window, exactly as the shipped grid produces them.
    points = [(811, 64), (3115, 30), (6187, 15), (12331, 8)]
    rows = []
    for in_tok, reps in points:
        for _ in range(2):
            energy = reps * (0.004 + 0.0009 * in_tok / 1000.0 + 0.486 * 8 / 1000.0)
            rows.append({
                "requests": reps,
                "text_tokens": in_tok * reps, "output_tokens": 8 * reps,
                "n_images": 0, "energy_wh": energy, "raw_energy_wh": energy * 3,
                "duration_ms": 4000.0,
            })
    for out_tok in (512, 1024, 2048, 4096):
        energy = 0.004 + 0.0009 * 32 / 1000.0 + 0.486 * out_tok / 1000.0
        rows.append({
            "requests": 1, "text_tokens": 32, "output_tokens": out_tok,
            "n_images": 0, "energy_wh": energy, "raw_energy_wh": energy * 3,
            "duration_ms": out_tok * 4.2,
        })

    fit = fit_nested_models(rows, idle_watts=117.0)
    cond = fit.design_leverage["rung_quality"]["output_intercept_input"]["condition_number"]
    assert cond < 30, f"aggregated design is collinear (condition number {cond:.1f})"

    # Read as raw totals the same measurement collapses: the prefill arm's
    # totals cluster instead of spanning.
    totals = [{**r, "requests": 1} for r in rows]
    collapsed = fit_nested_models(totals, idle_watts=117.0)
    assert (
        collapsed.design_leverage["rung_quality"]["output_intercept_input"][
            "condition_number"] > cond
    )
