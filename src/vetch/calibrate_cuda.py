"""NVIDIA GPU VLM energy calibration (Tier 0).

The CUDA counterpart to :mod:`vetch.calibrate_metal`. It runs the same symmetric
``(n_images, text_tokens, output_tokens)`` grid and the same 4-parameter
least-squares fit, but measures energy with NVML instead of powermetrics:

- Primary: ``nvmlDeviceGetTotalEnergyConsumption`` — a monotonic hardware energy
  counter (Volta+). Read before/after each run for exact per-run energy, no
  sampling thread, no root.
- Fallback: integrate ``nvmlDeviceGetPowerUsage`` samples over the run window
  when the counter is unavailable (older GPUs / some virtualized passthrough).

Output is a :class:`vetch.calibrate.CalibrationResult` carrying ``wh_per_image``,
``visual_tokens_per_image``, and ``intercept_wh`` — the exact fields the energy
override path already consumes, so a saved calibration is auto-loaded by
``calculate_energy`` with no further wiring.

The grid, fit, Ollama driver, image generation, and image-token probe are
imported from :mod:`vetch.calibrate_metal` so the two hardware paths stay in
lockstep; only energy measurement and the environment preflight differ here.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Any

from vetch.calibrate import CalibrationResult, GPUMonitor
from vetch.calibrate_metal import (
    CALIBRATION_IMAGE_SET_SYNTHETIC,
    CALIBRATION_IMAGE_SIZE_PX,
    FitResult,
    _build_grid,
    _check_ollama,
    _fit,
    _get_visual_tokens_per_image,
    _load_image_set,
    _ollama_generate,
    _unique_image_b64,
    _unique_media_b64,
    _unique_prompt,
    _warmup_image_b64,
    grid_design_id,
)
from vetch.calibration_store import (
    GPU_BOARD_EXCLUDES,
    GPU_BOARD_INCLUDES,
    CalibrationIdentity,
    canonical_gpu,
    commit_calibration,
    is_cloud_provider,
    measurement_provenance_core,
)

OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"

# Fallback power-sampling cadence when the energy counter is unavailable.
_POWER_SAMPLE_INTERVAL_MS = 50

# Drift guard: reject if idle power moves more than this between the pre- and
# post-grid baselines (thermal throttling / contention invalidates the fit).
_MAX_IDLE_DRIFT_PCT = 15.0

# Minimum successful image runs before a visual coefficient is trusted. A full
# grid has ~11 image runs; below this the wh_per_image fit is unreliable, so the
# visual term is withheld rather than reported as a measured (possibly zero) value.
_MIN_IMAGE_RUNS = 4

# Fixed grid shuffle seed so run order (hence thermal ordering) is reproducible
# and comparable across calibrations of the same profile.
_GRID_SEED = 20240808


def is_cuda_available() -> bool:
    """True if NVML initializes and at least one device is present."""
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            return bool(pynvml.nvmlDeviceGetCount() > 0)
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return False


def assert_cuda() -> None:
    """Raise a helpful error if NVML/GPU is not usable."""
    try:
        import pynvml  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "NVIDIA calibration requires the NVML bindings. Install them on the "
            "GPU host: pip install nvidia-ml-py"
        ) from e
    if not is_cuda_available():
        raise RuntimeError(
            "No NVIDIA GPU visible to NVML. Confirm the driver is loaded "
            "(nvidia-smi) and that this is a GPU host, not a CPU-only box."
        )


def _require_numpy() -> None:
    try:
        import numpy  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "CUDA calibration requires NumPy for the least-squares fit and "
            "bootstrap CIs. Install it on the GPU host: pip install numpy"
        ) from e


VLLM_DEFAULT_BASE_URL = "http://localhost:8000/v1"


def _check_openai_server(base_url: str) -> str:
    """Return a status string for an OpenAI-compatible server, or raise."""
    import json as _json
    from urllib import request as _req
    from urllib.error import URLError

    url = base_url.rstrip("/") + "/models"
    try:
        with _req.urlopen(url, timeout=5) as resp:
            data = _json.loads(resp.read())
    except (URLError, OSError, ValueError) as e:
        raise RuntimeError(
            f"OpenAI-compatible server not reachable at {base_url} ({e}). "
            "Start vLLM, e.g.: python -m vllm.entrypoints.openai.api_server "
            "--model Qwen/Qwen2.5-32B-Instruct --dtype bfloat16."
        ) from e
    ids = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
    listed = ", ".join(str(i) for i in ids) if ids else "no models listed"
    return f"openai-compatible ({listed})"


def _openai_compat_generate(
    base_url: str,
    model: str,
    prompt: str,
    image_b64: str | None = None,
    max_tokens: int = 32,
    *,
    min_tokens: int | None = None,
    ignore_eos: bool = False,
    modality: str = "image",
) -> tuple[int, int]:
    """One OpenAI-compatible ``/chat/completions`` call (vLLM, Ollama ``/v1``…).

    Returns ``(prompt_tokens, completion_tokens)`` from the response ``usage``.
    Non-text media is sent as a base64 data URI placed BEFORE the text (Gemma
    expects image-first). ``modality`` selects the MIME / content-part shape
    (``image`` / ``audio`` / ``video``). Temperature is pinned to 0 and Gemma's
    thinking is disabled so output length is deterministic.

    For batched calibration, pass ``min_tokens=max_tokens`` and
    ``ignore_eos=True`` so output length is held constant (vLLM).
    """
    import json as _json
    from urllib import request as _req
    from urllib.error import URLError

    content: list[dict[str, Any]] = []
    if image_b64 is not None:
        mod = (modality or "image").lower()
        if mod == "audio":
            # OpenAI-compatible audio input part (vLLM multimodal extension).
            content.append({
                "type": "input_audio",
                "input_audio": {
                    "data": image_b64,
                    "format": "wav",
                },
            })
        elif mod == "video":
            content.append({
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{image_b64}"},
            })
        else:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            })
    content.append({"type": "text", "text": prompt})

    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
        # vLLM passes this through to apply_chat_template; disables Gemma thinking.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if min_tokens is not None:
        payload["min_tokens"] = int(min_tokens)
    if ignore_eos:
        payload["ignore_eos"] = True
    body = _json.dumps(payload).encode()
    url = base_url.rstrip("/") + "/chat/completions"
    req = _req.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with _req.urlopen(req, timeout=300) as resp:
            data = _json.loads(resp.read())
    except URLError as e:
        raise RuntimeError(f"OpenAI-compatible request failed: {e}") from e

    usage = data.get("usage") or {}
    return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)


def _probe_image_tokens(
    generate: Any, base_url: str, model: str, image_b64: str
) -> dict[str, Any]:
    """Backend-agnostic image-token probe (mirrors calibrate_metal's).

    Raises (via ``generate``) if the model rejects images, which the caller
    treats as a text-only model.
    """
    text_prompt = "What color is the sky?"
    text_only, _ = generate(base_url, model, text_prompt, None, 1)
    with_image, _ = generate(base_url, model, text_prompt, image_b64, 1)
    delta = with_image - text_only
    return {
        "image_token_accounting": "includes" if delta > 10 else "excludes",  # nosec B105
        "delta_prompt_eval_count_with_image": delta,
        "text_only_prompt_eval_count": text_only,
        "with_image_prompt_eval_count": with_image,
    }


class _CudaEnergyMeter:
    """Per-run energy over a window, from the NVML counter or power samples.

    ``mark()`` snapshots the current time and (counter mode) the cumulative
    energy; ``energy_between(start, end)`` returns Wh for that window. Counter
    mode is exact and needs no background thread. Power-sampling mode spins a
    lightweight sampler and integrates trapezoidally.
    """

    def __init__(self, monitor: GPUMonitor, sample_interval_ms: int = _POWER_SAMPLE_INTERVAL_MS):
        self.monitor = monitor
        self.sample_interval_ms = sample_interval_ms
        self.use_counter = monitor.energy_counter_available()
        self._samples: list[tuple[float, float]] = []  # (t_ms, watts)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._sampler_failed = False

    def __enter__(self) -> _CudaEnergyMeter:
        if not self.use_counter:
            # Validate power readability up front so a device that supports
            # neither the energy counter nor power reads fails BEFORE the paid
            # grid, not after producing a run of zero-watt samples.
            try:
                self.monitor.get_power_w()
            except Exception as e:
                raise RuntimeError(
                    "GPU exposes neither the NVML energy counter nor readable "
                    f"power ({e}); cannot measure energy on this device."
                ) from e
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            t = time.monotonic() * 1000
            try:
                w = self.monitor.get_power_w()
            except Exception:
                # Don't append zero-watt noise indefinitely: mark failed and stop
                # so energy_between() raises loudly instead of undercounting.
                self._sampler_failed = True
                self._stop.set()
                break
            self._samples.append((t, w))
            self._stop.wait(self.sample_interval_ms / 1000.0)

    def mark(self) -> tuple[float, float | None]:
        t = time.monotonic() * 1000
        if self.use_counter:
            e = self.monitor.get_total_energy_mj()
            if e is None:
                # Counter worked at detection but failed mid-run. Fail loud
                # rather than fall through to integration (no sampler exists in
                # counter mode, which would silently yield zero energy).
                raise RuntimeError(
                    "NVML energy counter returned no reading mid-run; aborting "
                    "rather than recording zero energy."
                )
            return (t, e)
        return (t, None)

    def energy_between(
        self, start: tuple[float, float | None], end: tuple[float, float | None]
    ) -> float:
        (t0, e0), (t1, e1) = start, end
        if self.use_counter:
            assert e0 is not None and e1 is not None  # mark() guarantees this
            delta = e1 - e0
            if delta < 0:
                # Monotonic counter went backwards: driver reset/wrap during the
                # window. The datum is corrupt; surface it instead of clamping a
                # reset to a plausible zero-energy sample.
                raise RuntimeError(
                    "NVML energy counter went backwards (driver reset/wrap) "
                    "during a run; measurement is invalid."
                )
            return delta / 3_600_000.0  # mJ -> Wh
        if self._sampler_failed:
            raise RuntimeError("GPU power sampler failed during the run.")
        return self._integrate(t0, t1)

    def _nearest_power(self, t0: float, t1: float) -> float | None:
        if not self._samples:
            return None
        mid = (t0 + t1) / 2.0
        return min(self._samples, key=lambda s: abs(s[0] - mid))[1]

    def _integrate(self, t0: float, t1: float) -> float:
        pts = [(t, w) for (t, w) in self._samples if t0 <= t <= t1]
        if len(pts) < 2:
            # Too few interior samples for a trapezoid (short call): approximate
            # with the nearest power reading held across the window rather than
            # discarding it as zero, which would undercount short inferences.
            nearest = self._nearest_power(t0, t1)
            if nearest is None:
                return 0.0
            return nearest * ((t1 - t0) / 3_600_000.0)
        wh = 0.0
        for (ta, wa), (tb, wb) in zip(pts, pts[1:]):
            dt_h = (tb - ta) / 3_600_000.0
            wh += (wa + wb) / 2.0 * dt_h
        # Cover the unsampled edges [t0, first] and [last, t1] with the nearest
        # edge sample so window boundaries aren't silently dropped.
        wh += pts[0][1] * ((pts[0][0] - t0) / 3_600_000.0)
        wh += pts[-1][1] * ((t1 - pts[-1][0]) / 3_600_000.0)
        return wh


def _net_energy_wh(raw_wh: float, avg_idle_watts: float, duration_ms: float) -> float:
    """Active energy of a run: gross minus the idle draw over the same window.

    Uses the pre/post idle average (passed in) so idle-power growth across the
    grid does not bias later runs. Clamped at 0 for runs shorter than the idle
    floor.
    """
    duration_h = duration_ms / 3_600_000.0
    return max(0.0, raw_wh - avg_idle_watts * duration_h)


def _measure_idle_watts(meter: _CudaEnergyMeter, seconds: float = 3.0) -> float:
    """Average GPU power over an idle window, in Watts."""
    start = meter.mark()
    time.sleep(seconds)
    end = meter.mark()
    energy_wh = meter.energy_between(start, end)
    duration_h = (end[0] - start[0]) / 3_600_000.0
    return energy_wh / duration_h if duration_h > 0 else 0.0


def _cuda_rejection_reasons(
    fit: FitResult,
    samples: int,
    idle_drift_pct: float,
) -> list[str]:
    """Reasons a run should not become the active local calibration.

    Note: the power-sampling fallback (energy counter unavailable) is NOT a
    rejection. It is a coarser but legitimate measurement path — hard-failing it
    would mean a full paid run on virtualized hardware could never install a
    calibration. It is recorded in the detail JSON (energy_source) and warned
    about instead.
    """
    import math

    reasons = list(fit.invalid_reasons)
    if not fit.valid and not reasons:
        reasons.append("valid=false in fit")
    if math.isfinite(fit.condition_number) and fit.condition_number > 30:
        reasons.append(
            f"condition_number={fit.condition_number:.1f} > 30 "
            "(workload shapes not distinct enough)"
        )
    if samples < 8:
        reasons.append(f"samples={samples} < 8 minimum")
    if idle_drift_pct > _MAX_IDLE_DRIFT_PCT:
        reasons.append(f"idle_drift_pct={idle_drift_pct:.1f} > {_MAX_IDLE_DRIFT_PCT}%")

    deduped: list[str] = []
    for reason in reasons:
        if reason not in deduped:
            deduped.append(reason)
    return deduped


def calibrate_cuda(
    model: str,
    provider: str = "self-hosted",
    base_url: str | None = None,
    device_id: int = 0,
    iterations: int = 1,
    verbose: bool = False,
    backend: str = "ollama",
    precision: str | None = None,
    serving_engine: str | None = None,
    modality: str = "image",
) -> CalibrationResult:
    """Measure NVIDIA GPU energy coefficients for a locally-served model.

    Args:
        model: Model name as the serving backend expects it (Ollama tag, or the
            HF repo id for vLLM, e.g. "Qwen/Qwen2.5-32B-Instruct").
        provider: Provider label written into the calibration identity. Default
            ``"self-hosted"`` so production events tagged with
            ``provider_hint="self-hosted"`` resolve as an exact Tier-0 match.
            Must match the label your events emit (or a label in the
            cross-provider self-hosted reuse set).
        base_url: Server base URL. Defaults per backend (Ollama root, or the vLLM
            OpenAI base ``.../v1``).
        device_id: NVML device index to measure.
        iterations: Grid iteration multiplier (1 = one pass of ~22 runs).
        verbose: Print per-run detail.
        backend: ``"ollama"`` (native /api) or ``"openai"`` (OpenAI-compatible
            /chat/completions — vLLM, SGLang, Ollama's /v1). Use ``"openai"`` to
            calibrate the BF16 weights as served by vLLM.
        precision: Required identity dimension (e.g. ``bf16``, ``fp8-e4m3``,
            ``gguf:q4_k_m``). Distinct precisions must not share a file; omitting
            this collapses BF16/FP8 into one silent overwrite.
        serving_engine: Serving stack label for the identity (``vllm``,
            ``sglang``, …). Required when ``backend="openai"`` so vLLM/SGLang
            do not collide under a generic ``openai`` backend key.
        modality: Non-text media for the visual grid (``image`` / ``audio`` /
            ``video``). Synthetic unique payloads per request. Audio/video
            require ``backend="openai"``.

    Returns:
        CalibrationResult with wh_per_1k_input/output, wh_per_image (VLMs),
        visual_tokens_per_image, and intercept_wh. Tier 0 (hardware-measured).
    """
    modality = (modality or "image").strip().lower()
    if modality not in ("image", "audio", "video"):
        raise ValueError(
            f"modality must be 'image', 'audio', or 'video'; got {modality!r}"
        )
    if modality != "image" and backend != "openai":
        raise ValueError(
            f"modality={modality!r} requires backend='openai' "
            "(Ollama native /api only accepts images)."
        )

    assert_cuda()
    _require_numpy()

    # Precision is part of the calibration identity. Refuse before the paid grid
    # so BF16 and FP8 cannot silently overwrite each other under precision=na.
    if not precision or not str(precision).strip():
        raise ValueError(
            "precision is required (e.g. 'bf16', 'fp8-e4m3', 'gguf:q4_k_m'). "
            "It is an identity dimension: omitting it lets distinct quantizations "
            "collide on one file and resolve as exact Tier 0 with the wrong coeffs."
        )
    precision = str(precision).strip()

    if backend == "openai":
        _raw_generate = _openai_compat_generate

        def _openai_generate_with_modality(
            base_url: str,
            model: str,
            prompt: str,
            image_b64: str | None = None,
            max_tokens: int = 32,
            **kwargs: Any,
        ) -> tuple[int, int]:
            try:
                return _raw_generate(
                    base_url, model, prompt, image_b64, max_tokens,
                    modality=modality, **kwargs,
                )
            except TypeError:
                # Test doubles / older generate() shims may omit modality=.
                return _raw_generate(
                    base_url, model, prompt, image_b64, max_tokens, **kwargs,
                )

        generate: Any = _openai_generate_with_modality
        health_check = _check_openai_server
        default_base = VLLM_DEFAULT_BASE_URL
        # OpenAI-compatible is an API shape, not a serving stack. Without an
        # explicit engine, vLLM and SGLang would share one identity.
        if not serving_engine or not str(serving_engine).strip():
            raise ValueError(
                "serving_engine is required when backend='openai' "
                "(e.g. 'vllm', 'sglang'). Pass --serving-engine so distinct "
                "stacks do not collide under a generic 'openai' backend key."
            )
        serving_engine = str(serving_engine).strip()
    elif backend == "ollama":
        generate = _ollama_generate
        health_check = _check_ollama
        default_base = OLLAMA_DEFAULT_BASE_URL
    else:
        raise ValueError(f"Unknown backend {backend!r}; expected 'ollama' or 'openai'.")
    if base_url is None:
        base_url = default_base
    server_version = health_check(base_url)

    with GPUMonitor(device_id) as monitor:
        gpu_info = monitor.get_gpu_info()

        engine = serving_engine or backend  # identity's serving engine
        print("Vetch NVIDIA GPU Calibration")
        print(f"  Model:    {model}")
        print(f"  GPU:      {gpu_info['name']} (device {device_id})")
        print(f"  Driver:   {gpu_info['driver_version']}")
        print(f"  Provider: {provider}  (must match production provider_hint for Tier 0)")
        print(f"  Backend:  {backend} / engine={engine} @ {base_url}  ({server_version})")
        print(f"  Precision:{precision}")
        print(f"  Modality: {modality}")

        # Multi-GPU hosts: we meter device_id, but the server chooses its own
        # GPU. If they differ, the coefficients are meaningless. We can't force
        # the server's placement from here, so warn loudly and record the risk.
        gpu_count = monitor.device_count()
        # Captured inside the monitor block (NVML is shut down on exit). An
        # unrecorded power cap makes the coefficient unreproducible.
        power_limit_w = monitor.get_power_limit_w()
        clocks = monitor.get_clocks()
        # Best-effort lock detection: an applications clock set below the max
        # indicates a manual clock lock (nvidia-smi -ac). -lgc locks aren't
        # visible here, so None = "couldn't determine", never a false "unlocked".
        _app = clocks.get("applications_sm_clock_mhz")
        _max = clocks.get("sm_max_clock_mhz")
        sm_clock_locked: bool | None = (
            True if (_app is not None and _max is not None and _app < _max) else None
        )
        if not sm_clock_locked:
            print(
                "NOTE: GPU clocks not detected as locked; boost clocks vary energy "
                "run-to-run. For a reproducible profile, lock them "
                "(nvidia-smi -lgc <MHz>). Clocks are recorded regardless.",
                file=sys.stderr,
            )
        gpu_canonical, gpu_known = canonical_gpu(gpu_info["name"])
        if is_cloud_provider(provider):
            raise ValueError(
                f"provider={provider!r} is a cloud/API vendor and is refused for "
                "Tier-0 calibration: it is ambiguous (real hosted API vs a local "
                "OpenAI-compatible server) and would attach local coefficients to "
                "cloud events with a colliding model name. Use --provider "
                "self-hosted (or vllm/sglang) and instrument(provider_hint=...)."
            )
        if gpu_count > 1:
            print(
                f"WARNING: {gpu_count} GPUs visible. This meters device {device_id} "
                f"only. Ensure the model server is pinned to the SAME GPU (start it "
                f"with CUDA_VISIBLE_DEVICES={device_id}) or the energy will not match "
                "the measured device.",
                file=sys.stderr,
            )

        visual_tokens, visual_tokens_source = _get_visual_tokens_per_image(model)
        image_pool, active_image_set = _load_image_set()
        print(f"  Images:   {active_image_set} ({len(image_pool)} for probe/warmup)")

        # Build grid + pre-generate unique per-run images (busts Ollama's image
        # KV cache; a reused image skips the vision encoder and zeroes the
        # per-image energy). All generation happens outside the timing windows.
        grid = _build_grid(seed=_GRID_SEED) * max(1, iterations)
        run_images: list[list[str]] = []
        for i, spec in enumerate(grid):
            run_seed = i + 100
            run_images.append(
                [
                    _unique_media_b64(modality, seed=run_seed + j * 1000)
                    for j in range(spec.n_images)
                ]
            )
        total_runs = len(grid)

        # Probe how the model accounts for image tokens (outside the meter).
        model_is_vlm = True
        token_probe: dict[str, Any] = {}
        fit_visual_tokens = 0
        print("Probing image-token accounting...")
        try:
            token_probe = _probe_image_tokens(generate, base_url, model, image_pool[0])
            print(f"  Image token accounting: {token_probe['image_token_accounting']}")
            if token_probe["image_token_accounting"] == "includes":
                fit_visual_tokens = max(0, token_probe["delta_prompt_eval_count_with_image"])
        except RuntimeError as probe_err:
            model_is_vlm = False
            token_probe = {
                "image_token_accounting": "unsupported",  # nosec B105 — LLM tokens
                "image_probe_error": str(probe_err),
            }
            print(
                f"  Model does not accept images ({probe_err}); running text-only "
                "calibration (wh_per_image not measured).",
                file=sys.stderr,
            )

        if not model_is_vlm:
            grid = [spec for spec in grid if spec.n_images == 0]
            run_images = [[] for _ in grid]
            total_runs = len(grid)

        with _CudaEnergyMeter(monitor) as meter:
            # Single source of truth for the measurement mode: the meter decides
            # once at construction. Re-querying monitor.energy_counter_available()
            # here could disagree with the meter on a transient NVML response and
            # then mislabel the saved energy_source metadata.
            energy_counter = meter.use_counter
            energy_desc = (
                "NVML total-energy counter (exact)"
                if energy_counter
                else "power-sampling integration (fallback)"
            )
            print(f"  Energy:   {energy_desc}")
            print()

            # Warm up (loads model + vision encoder, reaches operating temp).
            warmup_prompt = _unique_prompt(approx_tokens=20, seed=0)
            warmup_img = _warmup_image_b64(model_is_vlm, image_pool)
            print("Warming up...")
            generate(base_url, model, warmup_prompt, warmup_img, max_tokens=10)
            time.sleep(2.0)

            print("Measuring idle baseline (before)...")
            idle_watts_before = _measure_idle_watts(meter)
            # Contamination baseline: at idle, only the serving process should be
            # on the (whole-device) GPU. More than one compute process suggests a
            # co-tenant whose draw would inflate every reading.
            idle_proc_count = monitor.compute_process_count()

            print(f"Running calibration grid ({total_runs} runs)...")
            raw_records: list[dict[str, Any]] = []
            grid_failures = 0
            for i, (spec, images) in enumerate(zip(grid, run_images)):
                prompt = _unique_prompt(approx_tokens=spec.approx_text_tokens, seed=i + 100)
                try:
                    start = meter.mark()
                    text_tokens, output_tokens = generate(
                        base_url, model, prompt,
                        image_b64=images[0] if images else None,
                        max_tokens=spec.max_tokens,
                    )
                    end = meter.mark()
                    gross_wh = meter.energy_between(start, end)
                except RuntimeError as e:
                    # Covers Ollama failures AND per-run energy faults (counter
                    # returned None / went backwards): skip this run, keep going.
                    print(f"  Run {i+1}/{total_runs} FAILED: {e}", file=sys.stderr)
                    grid_failures += 1
                    continue

                if text_tokens == 0:
                    grid_failures += 1
                    continue

                # Defer net-energy to after the post-grid idle baseline so the
                # idle term can use the pre/post average (thermal drift over the
                # grid otherwise biases later runs high). Store gross now.
                raw_records.append({
                    "n_images": spec.n_images,
                    "text_tokens": text_tokens,
                    "output_tokens": output_tokens,
                    "raw_energy_wh": gross_wh,
                    "duration_ms": end[0] - start[0],
                    "replicate": spec.replicate,
                })
                if verbose:
                    print(
                        f"  [{i+1:2d}/{total_runs}] n_img={spec.n_images} "
                        f"in={text_tokens:4d} out={output_tokens:3d} "
                        f"gross={gross_wh*1000:.4f} mWh ({(end[0]-start[0]):.0f}ms)"
                    )
                else:
                    print(f"\r  Progress: {i+1}/{total_runs}", end="", flush=True)
                time.sleep(1.0)
            print()

            print("Measuring idle baseline (after)...")
            idle_watts_after = _measure_idle_watts(meter)

    # --- Quality gates -------------------------------------------------------
    if grid_failures > total_runs // 2:
        raise RuntimeError(
            f"Too many failed grid runs ({grid_failures}/{total_runs}). "
            "Check Ollama health, the model pull, and GPU memory."
        )

    idle_drift_pct = (
        abs(idle_watts_after - idle_watts_before) / max(idle_watts_before, 0.01) * 100
    )
    print(
        f"  Idle power before: {idle_watts_before:.1f} W  after: {idle_watts_after:.1f} W "
        f"(drift {idle_drift_pct:.1f}%)"
    )

    # Net energy per run using the pre/post idle AVERAGE, computed now that both
    # baselines exist. Using only the pre-grid baseline would let idle-power
    # growth over the grid bias later runs high.
    avg_idle_watts = (idle_watts_before + idle_watts_after) / 2.0
    for rec in raw_records:
        rec["energy_wh"] = _net_energy_wh(
            rec["raw_energy_wh"], avg_idle_watts, rec["duration_ms"]
        )

    # Drop runs whose gross energy is zero despite real duration: a measurement
    # gap, not a real datum, and it would bias the regression toward the origin.
    gap_runs = [r for r in raw_records if r["raw_energy_wh"] <= 0.0 and r["duration_ms"] > 500]
    if gap_runs:
        print(
            f"WARNING: dropping {len(gap_runs)} run(s) with zero gross energy "
            "despite >500ms duration (measurement gap).",
            file=sys.stderr,
        )
    run_records = [r for r in raw_records if r not in gap_runs]

    # Whole-device contamination: warn if more than the serving process shared
    # the GPU during measurement (co-tenant draw inflates every reading).
    contaminated = isinstance(idle_proc_count, int) and idle_proc_count > 1
    if contaminated:
        print(
            f"WARNING: {idle_proc_count} compute processes were on this GPU during "
            "measurement. Energy is whole-device, so a co-tenant inflates every "
            "reading. Use a dedicated (non-MIG-shared) GPU for a trustworthy result.",
            file=sys.stderr,
        )

    if len(run_records) < 8:
        raise RuntimeError(
            f"Too few successful runs ({len(run_records)}) to fit. "
            "Check Ollama is serving the model and the GPU has enough memory."
        )

    # Per-modality coverage. If this is a VLM but its image runs failed (e.g.
    # exactly half the grid — all image cells — dropped, slipping under the
    # >half gate), the fit has no image column and fit.wh_per_image is 0. Saving
    # that would falsely present visual energy as measured-and-complete. Require a
    # minimum of successful image runs before trusting a visual coefficient.
    image_runs = sum(1 for r in run_records if r["n_images"] > 0)
    vlm_measured = model_is_vlm and image_runs >= _MIN_IMAGE_RUNS

    print(f"Fitting energy model ({len(run_records)} runs)...")
    fit = _fit(run_records, visual_tokens_per_image=fit_visual_tokens)
    rejection_reasons = _cuda_rejection_reasons(
        fit=fit,
        samples=len(run_records),
        idle_drift_pct=idle_drift_pct,
    )
    if contaminated:
        rejection_reasons.append(
            f"gpu_shared: {idle_proc_count} compute processes on device during run"
        )
    if model_is_vlm and not vlm_measured:
        rejection_reasons.append(
            f"vlm_image_runs={image_runs} < {_MIN_IMAGE_RUNS} minimum: the visual "
            "term could not be measured (image runs failed); wh_per_image withheld"
        )
    calibration_valid = not rejection_reasons

    if not calibration_valid:
        print("\nWARNING: Calibration quality issues:")
        for reason in rejection_reasons:
            print(f"  - {reason}")
        print("  Detail JSON saved, but active calibration will not be installed.")

    if not energy_counter:
        print(
            "NOTE: energy measured via power-sampling integration (energy counter "
            "unavailable). Coarser than the hardware counter but still installed.",
            file=sys.stderr,
        )

    # Withhold the visual term entirely (None, never 0) unless it was actually
    # measured. A None wh_per_image makes calculate_energy flag visual calls as
    # energy_completeness="text_only" — honest — whereas a 0 would present them
    # as complete with no image cost.
    wh_per_image = fit.wh_per_image if vlm_measured else None
    # Save the MEASURED per-image visual-token count (probe delta), not the
    # assumed constant: calculate_energy uses this to subtract visual tokens from
    # text at inference time, so it must match what the model actually charges.
    # fit_visual_tokens is the probe delta for "includes" models, 0 for
    # "excludes" (image tokens not folded into the text count -> nothing to
    # subtract), and None applies when the visual term wasn't measured.
    visual_tokens_for_result = fit_visual_tokens if vlm_measured else None

    result = CalibrationResult(
        model=model,
        provider=provider,
        wh_per_1k_input=fit.wh_per_1k_input,
        wh_per_1k_output=fit.wh_per_1k_output,
        tier=0,
        samples=len(run_records),
        gpu_name=gpu_info["name"],
        wh_per_image=wh_per_image,
        visual_tokens_per_image=visual_tokens_for_result,
        intercept_wh=fit.intercept_wh,
        active=calibration_valid,
        rejection_reasons=rejection_reasons if rejection_reasons else None,
        serving_engine=engine,
        backend=engine,
        precision=precision,
    )

    # --- Data-rich v1 record (single artifact: coefficients + full provenance) ---
    try:
        import numpy  # noqa: F401
        fit_engine, ci_method = "numpy", "bootstrap_500"
    except ImportError:
        fit_engine, ci_method = "pure_python", "heuristic_pm20"

    from vetch import __version__ as _vetch_version
    from vetch.calculation import METHODOLOGY_VERSION

    def _rng(key: str) -> list[int]:
        vals = [r[key] for r in run_records]
        return [min(vals), max(vals)] if vals else [0, 0]

    identity = CalibrationIdentity(
        provider=provider,
        model=model,
        gpu=gpu_canonical,
        serving_engine=engine,
        precision=precision,
    )
    provenance = measurement_provenance_core(
        samples=len(run_records),
        energy_source=(
            "nvmlDeviceGetTotalEnergyConsumption" if energy_counter else "power_integration"
        ),
        measurement_basis="nvml_gpu_energy",
        energy_domain="gpu_board",
        energy_domain_includes=GPU_BOARD_INCLUDES,
        energy_domain_excludes=GPU_BOARD_EXCLUDES,
        idle_watts_before=idle_watts_before,
        idle_watts_after=idle_watts_after,
        idle_drift_pct=idle_drift_pct,
        fit=fit,
        fit_engine=fit_engine,
        ci_method=ci_method,
        run_records=run_records,
        gpu_name=gpu_info["name"],
        gpu_canonical=gpu_canonical,
        gpu_known=gpu_known,
        serving_engine=engine,
        server_version=server_version,
        image_set=CALIBRATION_IMAGE_SET_SYNTHETIC,
        image_resolution_px=CALIBRATION_IMAGE_SIZE_PX,
        model_supports_images=model_is_vlm,
        visual_tokens_assumed=visual_tokens,
        visual_tokens_assumed_source=visual_tokens_source,
        vetch_version=_vetch_version,
        methodology_version=METHODOLOGY_VERSION,
        extra={
            "samples_per_modality": {
                "text_only": len(run_records) - image_runs, "image": image_runs,
            },
            "tensor_parallel_size": None,
            "sampler_cadence_ms": None if energy_counter else _POWER_SAMPLE_INTERVAL_MS,
            "integration_method": "energy_counter" if energy_counter else "trapezoidal",
            "token_accounting_basis": (
                "prompt_tokens" if backend == "openai" else "prompt_eval_count"
            ),
            "image_token_accounting": token_probe.get("image_token_accounting"),
            "calibrated_range": {
                "text_tokens": _rng("text_tokens"),
                "output_tokens": _rng("output_tokens"),
                "n_images": _rng("n_images"),
            },
            "enforced_power_limit_w": power_limit_w,
            "clocks": clocks,
            "sm_clock_locked": sm_clock_locked,
            "grid_design_id": grid_design_id(),
            "grid_seed": _GRID_SEED,
            "memory_total_mb": gpu_info["memory_total_mb"],
            "driver_version": gpu_info["driver_version"],
            "device_id": device_id,
            "device_count": gpu_count,
            "compute_process_count_at_idle": idle_proc_count,
            "gpu_shared_warning": contaminated,
            "api_backend": backend,  # HTTP shape (ollama|openai), not serving_engine
            "model_revision": None,
        },
    )
    record_path = commit_calibration(result, identity, provenance)
    print(f"\nCalibration record written to {record_path}")
    if not calibration_valid:
        print("  (active=false: recorded for audit, will not auto-load)")

    return result


# ---------------------------------------------------------------------------
# Batched / concurrency-amortization calibration
# ---------------------------------------------------------------------------

# Thermal drift over a steady-state block (not pre/post idle) — batched load
# heats far more than batch=1; gate is looser but still fail-loud.
_MAX_BLOCK_DRIFT_PCT = 25.0

# Default concurrency sweep for production-representative curves.
_DEFAULT_CONCURRENCIES: tuple[int, ...] = (1, 4, 8, 16, 32)


def _fit_amortization_curve(
    points: list[tuple[int, float]],
) -> dict[str, float]:
    """Least-squares fit of Wh/1k_out(C) = a/C + b.

    Returns ``{a, b, r2}``. Needs at least two distinct concurrency points.
    """
    xs = [1.0 / float(c) for c, _ in points if c > 0]
    ys = [float(y) for c, y in points if c > 0]
    n = len(xs)
    if n < 2:
        y0 = ys[0] if ys else 0.0
        return {"a": 0.0, "b": y0, "r2": 0.0}

    # Normal equations for [1/C, 1] @ [a, b] = y
    s_xx = sum(x * x for x in xs)
    s_x = sum(xs)
    s_xy = sum(x * y for x, y in zip(xs, ys))
    s_y = sum(ys)
    det = s_xx * n - s_x * s_x
    if abs(det) < 1e-18:
        return {"a": 0.0, "b": s_y / n, "r2": 0.0}
    a = (s_xy * n - s_x * s_y) / det
    b = (s_xx * s_y - s_x * s_xy) / det
    y_hat = [a * x + b for x in xs]
    ss_res = sum((y - yh) ** 2 for y, yh in zip(ys, y_hat))
    y_mean = s_y / n
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-18 else 0.0
    return {"a": float(a), "b": float(b), "r2": float(r2)}


def _scrape_vllm_num_running(base_url: str) -> float | None:
    """Best-effort scrape of ``vllm:num_requests_running`` from /metrics."""
    import re as _re
    from urllib import request as _req
    from urllib.error import URLError

    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    url = root + "/metrics"
    try:
        with _req.urlopen(url, timeout=2) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except (URLError, OSError, TimeoutError):
        return None
    # Prefer unlabeled gauge; fall back to first matching sample.
    m = _re.search(
        r"^vllm:num_requests_running(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$",
        text,
        _re.MULTILINE,
    )
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _run_concurrent_block(
    *,
    generate: Any,
    base_url: str,
    model: str,
    concurrency: int,
    n_requests: int,
    in_words: int,
    out_tokens: int,
    fixed_output: bool,
    prompt_seed_base: int,
    image_seed_base: int | None = None,
) -> tuple[int, int, float]:
    """Fire ``n_requests`` at concurrency C; return (prompt_tok, out_tok, wall_s).

    Every request gets a UNIQUE prompt (nonce-first via ``_unique_prompt``) and,
    when imaging, a unique image. This is essential: vLLM prefix caching and the
    vision encoder cache would otherwise skip prefill/encode for all but the first
    request, understating measured energy by a large factor. ``prompt_seed_base``
    and ``image_seed_base`` offset the per-request seeds so separate blocks
    (warmup / text / image) and concurrency levels never reuse a prompt.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _one(i: int) -> tuple[int, int]:
        kwargs: dict[str, Any] = {}
        if fixed_output:
            kwargs["min_tokens"] = out_tokens
            kwargs["ignore_eos"] = True
        prompt = _unique_prompt(approx_tokens=in_words, seed=prompt_seed_base + i)
        image_b64 = (
            _unique_image_b64(seed=image_seed_base + i)
            if image_seed_base is not None
            else None
        )
        result = generate(
            base_url, model, prompt, image_b64, out_tokens, **kwargs,
        )
        return int(result[0]), int(result[1])

    t0 = time.monotonic()
    prompt_tok = 0
    out_tok = 0
    errors: list[BaseException] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futs = [pool.submit(_one, i) for i in range(n_requests)]
        for fut in as_completed(futs):
            try:
                p, o = fut.result()
                prompt_tok += int(p)
                out_tok += int(o)
            except BaseException as e:  # noqa: BLE001 — collect, fail after
                errors.append(e)
    wall_s = time.monotonic() - t0
    if errors:
        raise RuntimeError(
            f"{len(errors)}/{n_requests} concurrent requests failed "
            f"(first: {errors[0]})"
        )
    return prompt_tok, out_tok, wall_s


def measure_concurrency_level(
    *,
    meter: _CudaEnergyMeter,
    monitor: GPUMonitor,
    generate: Any,
    base_url: str,
    model: str,
    concurrency: int,
    requests_per_level: int,
    out_tokens: int,
    in_words: int,
    image_b64: str | None,
    fixed_output: bool,
    idle_seconds: float = 2.0,
    settle_seconds: float = 1.0,
) -> dict[str, Any]:
    """Measure Wh/1k_output (and optional wh_per_image) at concurrency C.

    Returns a dict with keys used by :func:`calibrate_cuda_batched` /
    amortization fit. Injectable for unit tests.
    """
    # Per-request unique prompts/images (defeat prefix + vision caches). Offset
    # seeds by concurrency and block so no prompt is reused across blocks or
    # concurrency levels. The image block uses its own prompt seeds (reusing the
    # text block's would hit the prefix cache) but the same prompt length, so the
    # image-minus-text delta isolates the visual cost in expectation.
    # The per-concurrency stride (1e8) must exceed the largest block offset plus
    # requests_per_level, or two concurrency levels could share a seed window and
    # silently reuse prompts across blocks.
    seed_c = _GRID_SEED + concurrency * 100_000_000
    warm_prompt_base = seed_c + 1_000_000
    text_prompt_base = seed_c + 2_000_000
    image_prompt_seed_base = seed_c + 3_000_000

    # Warm the pipe so the measured block is steady-state.
    warm_n = max(concurrency, min(concurrency * 2, requests_per_level))
    _run_concurrent_block(
        generate=generate, base_url=base_url, model=model,
        concurrency=concurrency, n_requests=warm_n, in_words=in_words,
        out_tokens=out_tokens, fixed_output=fixed_output,
        prompt_seed_base=warm_prompt_base,
    )
    if settle_seconds > 0:
        time.sleep(settle_seconds)

    idle_w = _measure_idle_watts(meter, seconds=idle_seconds)
    power_before = monitor.get_power_w()
    clocks_before = monitor.get_clocks() if hasattr(monitor, "get_clocks") else {}
    running_sample = _scrape_vllm_num_running(base_url)

    # Text-only steady-state block.
    start = meter.mark()
    p_tok, o_tok, wall_s = _run_concurrent_block(
        generate=generate, base_url=base_url, model=model,
        concurrency=concurrency, n_requests=requests_per_level, in_words=in_words,
        out_tokens=out_tokens, fixed_output=fixed_output,
        prompt_seed_base=text_prompt_base,
    )
    end = meter.mark()
    running_after = _scrape_vllm_num_running(base_url)
    gross_wh = meter.energy_between(start, end)
    net_wh = _net_energy_wh(gross_wh, idle_w, wall_s * 1000.0)
    wh_per_1k_out = (net_wh / o_tok * 1000.0) if o_tok > 0 else 0.0
    # Second idle sample after the block so idle drift is a real before/after
    # comparison (distinct from block_drift_pct, which is the block's own swing).
    idle_w_after = _measure_idle_watts(meter, seconds=idle_seconds)
    idle_drift_pct = abs(idle_w_after - idle_w) / max(idle_w, 0.01) * 100.0

    power_after = monitor.get_power_w()
    clocks_after = monitor.get_clocks() if hasattr(monitor, "get_clocks") else {}
    drift_pct = 0.0
    if power_before > 0:
        drift_pct = abs(power_after - power_before) / power_before * 100.0

    # Achieved concurrency from the vLLM metrics scrape only. We do NOT synthesize
    # a latency×throughput estimate: for a closed-loop ThreadPool that product is
    # algebraically equal to the requested concurrency, so it would report a
    # tautology as if it were measured. If metrics are unavailable, leave it None
    # (unknown) rather than assert the target was reached.
    achieved = running_after if running_after is not None else running_sample

    wh_per_image: float | None = None
    if image_b64 is not None:
        if settle_seconds > 0:
            time.sleep(settle_seconds)
        idle_w_img = _measure_idle_watts(meter, seconds=idle_seconds)
        start_i = meter.mark()
        _p_i, o_i, wall_i = _run_concurrent_block(
            generate=generate, base_url=base_url, model=model,
            concurrency=concurrency, n_requests=requests_per_level, in_words=in_words,
            out_tokens=out_tokens, fixed_output=fixed_output,
            prompt_seed_base=image_prompt_seed_base,
            image_seed_base=seed_c + 4_000_000,
        )
        end_i = meter.mark()
        gross_i = meter.energy_between(start_i, end_i)
        net_i = _net_energy_wh(gross_i, idle_w_img, wall_i * 1000.0)
        # Image-minus-text delta per request ≈ per-image cost. This assumes the
        # two blocks decoded the same number of output tokens. With fixed_output
        # (vLLM min_tokens+ignore_eos) that holds; with a backend that cannot pin
        # output length (e.g. ollama) the totals can diverge and the subtraction
        # would conflate decode-energy delta with image energy — so refuse the
        # estimate rather than report a contaminated number.
        tokens_matched = o_tok > 0 and abs(o_i - o_tok) <= 0.05 * o_tok
        if fixed_output or tokens_matched:
            wh_per_image = (net_i - net_wh) / float(requests_per_level)
        else:
            wh_per_image = None

    return {
        "concurrency": concurrency,
        "wh_per_1k_out": wh_per_1k_out,
        "wh_per_image": wh_per_image,
        "achieved_concurrency": achieved,
        "net_wh": net_wh,
        "gross_wh": gross_wh,
        "idle_watts": idle_w,
        "idle_watts_before": idle_w,
        "idle_watts_after": idle_w_after,
        "idle_drift_pct": idle_drift_pct,
        "output_tokens": o_tok,
        "prompt_tokens": p_tok,
        "wall_s": wall_s,
        "power_before_w": power_before,
        "power_after_w": power_after,
        "block_drift_pct": drift_pct,
        "clocks_before": clocks_before,
        "clocks_after": clocks_after,
    }


def calibrate_cuda_batched(
    model: str,
    *,
    concurrencies: tuple[int, ...] | list[int] = _DEFAULT_CONCURRENCIES,
    requests_per_level: int = 64,
    out_tokens: int = 64,
    in_words: int = 96,
    provider: str = "self-hosted",
    base_url: str | None = None,
    device_id: int = 0,
    backend: str = "openai",
    precision: str | None = None,
    serving_engine: str | None = None,
    verbose: bool = False,
    measure_images: bool = True,
) -> list[CalibrationResult]:
    """Measure per-output-token energy at real serving concurrency (amortization curve).

    For each concurrency C, drives a fixed workload at true concurrency C, meters
    whole-GPU energy over a steady-state window, subtracts idle, and records
    Wh/1k_output(C). Fits Wh/1k_out = a/C + b and writes one v1 record per C
    (identity.concurrency=C) with the full curve + fit in provenance.

    The batch=1 grid path (:func:`calibrate_cuda`) is unchanged — this is an
    additive production-representative mode.
    """
    assert_cuda()
    _require_numpy()

    if not precision or not str(precision).strip():
        raise ValueError(
            "precision is required (e.g. 'bf16', 'fp8-e4m3'). "
            "It is an identity dimension for batched records too."
        )
    precision = str(precision).strip()
    if is_cloud_provider(provider):
        raise ValueError(
            f"provider={provider!r} is a cloud/API vendor and is refused for "
            "Tier-0 calibration. Use provider='self-hosted' (or vllm)."
        )

    concs = sorted({int(c) for c in concurrencies if int(c) > 0})
    if not concs:
        raise ValueError("concurrencies must contain at least one positive int")

    if backend == "openai":
        generate: Any = _openai_compat_generate
        health_check = _check_openai_server
        default_base = VLLM_DEFAULT_BASE_URL
        if not serving_engine or not str(serving_engine).strip():
            raise ValueError(
                "serving_engine is required when backend='openai' (e.g. 'vllm')."
            )
        serving_engine = str(serving_engine).strip()
        fixed_output = True
    elif backend == "ollama":
        generate = _ollama_generate
        health_check = _check_ollama
        default_base = OLLAMA_DEFAULT_BASE_URL
        serving_engine = serving_engine or "ollama"
        fixed_output = False  # ollama has no ignore_eos/min_tokens pair
    else:
        raise ValueError(f"Unknown backend {backend!r}; expected 'ollama' or 'openai'.")

    if base_url is None:
        base_url = default_base
    server_version = health_check(base_url)
    engine = serving_engine or backend

    with GPUMonitor(device_id) as monitor:
        gpu_info = monitor.get_gpu_info()
        gpu_canonical, gpu_known = canonical_gpu(gpu_info["name"])
        print("Vetch NVIDIA GPU Batched Calibration")
        print(f"  Model:         {model}")
        print(f"  GPU:           {gpu_info['name']} (device {device_id})")
        print(f"  Provider:      {provider}")
        print(f"  Backend:       {backend} / engine={engine} @ {base_url}")
        print(f"  Precision:     {precision}")
        print(f"  Concurrencies: {concs}")
        print(f"  Requests/level:{requests_per_level}  out_tokens={out_tokens}")
        print(f"  Server:        {server_version}")

        # Whole-device contamination + multi-GPU guards (mirrors the batch=1 path).
        # The NVML board-energy counter sums ALL work on the metered device, so a
        # co-tenant process, or metering the wrong GPU, silently corrupts the curve.
        gpu_shared_warning = False
        try:
            proc_n = monitor.compute_process_count()
        except Exception:
            proc_n = None
        if proc_n is not None and proc_n > 1:
            gpu_shared_warning = True
            print(
                f"  WARNING: {proc_n} compute processes share device {device_id}; "
                "board-energy readings include co-tenant work and will overstate "
                "per-request energy. Calibrate on a dedicated GPU.",
                file=sys.stderr,
            )
        try:
            dev_n = monitor.device_count()
        except Exception:
            dev_n = None
        if dev_n is not None and dev_n > 1:
            print(
                f"  NOTE: {dev_n} GPUs visible; metering device {device_id} only. "
                "Ensure the server is pinned to this GPU (CUDA_VISIBLE_DEVICES).",
                file=sys.stderr,
            )

        image_b64: str | None = None
        visual_tokens: int | None = None
        visual_tokens_source = "not_applicable"
        if measure_images:
            try:
                image_pool, _ = _load_image_set()
                if image_pool:
                    image_b64 = image_pool[0]
                    visual_tokens, visual_tokens_source = _get_visual_tokens_per_image(model)
                    try:
                        _probe_image_tokens(generate, base_url, model, image_b64)
                    except Exception as e:
                        print(f"  Images:       skipped ({e})")
                        image_b64 = None
                        visual_tokens = None
                else:
                    print("  Images:       skipped (empty image pool)")
            except Exception as e:
                print(f"  Images:       unavailable ({e})")
                image_b64 = None

        levels: list[dict[str, Any]] = []
        with _CudaEnergyMeter(monitor) as meter:
            energy_counter = meter.use_counter
            print(
                f"  Energy:        "
                f"{'NVML total-energy counter' if energy_counter else 'power integration'}"
            )
            for c in concs:
                print(f"\nMeasuring concurrency={c} ...")
                level = measure_concurrency_level(
                    meter=meter,
                    monitor=monitor,
                    generate=generate,
                    base_url=base_url,
                    model=model,
                    concurrency=c,
                    requests_per_level=requests_per_level,
                    out_tokens=out_tokens,
                    in_words=in_words,
                    image_b64=image_b64,
                    fixed_output=fixed_output,
                )
                levels.append(level)
                if verbose:
                    print(
                        f"  C={c}: Wh/1k_out={level['wh_per_1k_out']:.6f} "
                        f"achieved≈{level['achieved_concurrency']} "
                        f"drift={level['block_drift_pct']:.1f}%"
                    )
                if level["block_drift_pct"] > _MAX_BLOCK_DRIFT_PCT:
                    print(
                        f"  WARNING: block power drift {level['block_drift_pct']:.1f}% "
                        f"> {_MAX_BLOCK_DRIFT_PCT}% at C={c}",
                        file=sys.stderr,
                    )
                # Cool-down between levels so thermal from C_n doesn't bias C_n+1.
                time.sleep(2.0)

        fit = _fit_amortization_curve(
            [(int(lv["concurrency"]), float(lv["wh_per_1k_out"])) for lv in levels]
        )
        print(
            f"\nAmortization fit: Wh/1k_out ≈ {fit['a']:.4f}/C + {fit['b']:.4f} "
            f"(R²={fit['r2']:.4f})"
        )

        curve = [
            {
                "concurrency": int(lv["concurrency"]),
                "wh_per_1k_out": round(float(lv["wh_per_1k_out"]), 6),
                "wh_per_image": (
                    None if lv["wh_per_image"] is None
                    else round(float(lv["wh_per_image"]), 6)
                ),
                "achieved_concurrency": lv["achieved_concurrency"],
                "output_tokens": lv["output_tokens"],
                "wall_s": round(float(lv["wall_s"]), 3),
                "block_drift_pct": round(float(lv["block_drift_pct"]), 2),
            }
            for lv in levels
        ]

        from vetch import __version__ as _vetch_version
        from vetch.calculation import METHODOLOGY_VERSION

        results: list[CalibrationResult] = []
        for lv in levels:
            c = int(lv["concurrency"])
            rejection: list[str] = []
            if lv["block_drift_pct"] > _MAX_BLOCK_DRIFT_PCT:
                rejection.append(
                    f"block_drift_pct={lv['block_drift_pct']:.1f} > {_MAX_BLOCK_DRIFT_PCT}%"
                )
            if lv["output_tokens"] <= 0:
                rejection.append("output_tokens=0")
            active = not rejection

            result = CalibrationResult(
                model=model,
                provider=provider,
                # Input energy is negligible at decode-dominated batched load;
                # report ~0 rather than a false-precision prefill coefficient.
                wh_per_1k_input=0.0,
                wh_per_1k_output=float(lv["wh_per_1k_out"]),
                tier=0,
                samples=int(requests_per_level),
                gpu_name=gpu_info["name"],
                wh_per_image=lv["wh_per_image"],
                visual_tokens_per_image=visual_tokens if lv["wh_per_image"] is not None else None,
                intercept_wh=0.0,
                active=active,
                rejection_reasons=rejection or None,
                serving_engine=engine,
                backend=engine,
                precision=precision,
            )
            identity = CalibrationIdentity(
                provider=provider,
                model=model,
                gpu=gpu_canonical,
                serving_engine=engine,
                precision=precision,
                concurrency=c,
            )
            provenance = measurement_provenance_core(
                samples=int(requests_per_level),
                energy_source=(
                    "nvmlDeviceGetTotalEnergyConsumption"
                    if energy_counter else "power_integration"
                ),
                measurement_basis="nvml_gpu_energy_batched",
                energy_domain="gpu_board",
                energy_domain_includes=GPU_BOARD_INCLUDES,
                energy_domain_excludes=GPU_BOARD_EXCLUDES,
                idle_watts_before=float(lv.get("idle_watts_before", lv["idle_watts"])),
                idle_watts_after=float(lv.get("idle_watts_after", lv["idle_watts"])),
                idle_drift_pct=float(
                    lv.get("idle_drift_pct", lv.get("block_drift_pct", 0.0))
                ),
                fit=type("F", (), {
                    "r2": fit["r2"],
                    "condition_number": float("inf"),
                    "input_ci95": (0.0, 0.0),
                    "output_ci95": (0.0, 0.0),
                    "image_ci95": (0.0, 0.0),
                    "residuals_structured": False,
                })(),
                fit_engine="amortization_1_over_c",
                ci_method="none",
                run_records=[{
                    "n_images": 0 if lv["wh_per_image"] is None else 1,
                    "text_tokens": lv["prompt_tokens"],
                    "output_tokens": lv["output_tokens"],
                    "energy_wh": lv["net_wh"],
                    "raw_energy_wh": lv["gross_wh"],
                    "duration_ms": lv["wall_s"] * 1000.0,
                    "replicate": 0,
                }],
                gpu_name=gpu_info["name"],
                gpu_canonical=gpu_canonical,
                gpu_known=gpu_known,
                serving_engine=engine,
                server_version=server_version,
                image_set=CALIBRATION_IMAGE_SET_SYNTHETIC if image_b64 else None,
                image_resolution_px=CALIBRATION_IMAGE_SIZE_PX if image_b64 else None,
                model_supports_images=image_b64 is not None,
                visual_tokens_assumed=visual_tokens,
                visual_tokens_assumed_source=visual_tokens_source,
                vetch_version=_vetch_version,
                methodology_version=METHODOLOGY_VERSION,
                extra={
                    "calibration_mode": "batched",
                    "concurrency": c,
                    "batch_size": c,  # continuous batching effective size ≈ C
                    "requests_per_level": requests_per_level,
                    "out_tokens_target": out_tokens,
                    "in_words": in_words,
                    "fixed_output_length": fixed_output,
                    "achieved_concurrency": lv["achieved_concurrency"],
                    "amortization_curve": curve,
                    "amortization_fit": fit,
                    "api_backend": backend,
                    "device_id": device_id,
                    "driver_version": gpu_info["driver_version"],
                    "memory_total_mb": gpu_info["memory_total_mb"],
                    "power_before_w": lv["power_before_w"],
                    "power_after_w": lv["power_after_w"],
                    "block_drift_pct": lv.get("block_drift_pct"),
                    "clocks_before": lv.get("clocks_before"),
                    "clocks_after": lv.get("clocks_after"),
                    "gpu_shared_warning": gpu_shared_warning,
                    "compute_process_count": proc_n,
                    "device_count": dev_n,
                },
            )
            path = commit_calibration(result, identity, provenance)
            print(f"  Record C={c}: {path}")
            results.append(result)

        return results
