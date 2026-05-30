"""Calibration module for hardware-level energy measurement.

This module provides tools to measure actual power draw during inference
to improve upon Tier 3 (Proxy) estimates.

Supports:
- NVIDIA GPUs via pynvml (pip install nvidia-ml-py3)
- Apple Silicon via powermetrics (vetch.calibrate_metal; requires sudo)
"""

from __future__ import annotations

import json
import logging
import math
import time
import warnings
from pathlib import Path
from typing import Any, Callable, NamedTuple

logger = logging.getLogger(__name__)

# Track if warning has been issued
_EXPERIMENTAL_WARNING_ISSUED = False


def _warn_experimental() -> None:
    """Issue experimental warning once."""
    global _EXPERIMENTAL_WARNING_ISSUED
    if not _EXPERIMENTAL_WARNING_ISSUED:
        _EXPERIMENTAL_WARNING_ISSUED = True
        warnings.warn(
            "vetch.calibrate is experimental. API may change. "
            "NVIDIA path requires: pip install nvidia-ml-py3",
            FutureWarning,
            stacklevel=3,
        )

# Directory for saved hardware calibrations
CALIBRATION_DIR = Path.home() / ".vetch" / "calibrations"


class CalibrationResult(NamedTuple):
    """Result of a model calibration run."""

    model: str
    provider: str
    wh_per_1k_input: float
    wh_per_1k_output: float
    tier: int
    samples: int
    gpu_name: str


def is_gpu_available() -> bool:
    """Check if NVIDIA GPU and management library are available."""
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        return True
    except (ImportError, Exception):
        return False


def get_gpu_error() -> str:
    """Get descriptive error why GPU is not available."""
    try:
        import pynvml
        pynvml.nvmlInit()
        return "Unknown error"
    except ImportError:
        return "pynvml not installed (pip install nvidia-ml-py3)"
    except Exception as e:
        return str(e)


class GPUMonitor:
    """Context manager for monitoring GPU power draw."""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._handle = None
        self._start_energy_mj = 0.0

    def __enter__(self) -> GPUMonitor:
        import pynvml
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        import pynvml
        pynvml.nvmlShutdown()

    def get_gpu_info(self) -> dict[str, Any]:
        import pynvml
        name = pynvml.nvmlDeviceGetName(self._handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        driver = pynvml.nvmlSystemGetDriverVersion()
        return {
            "name": name.decode("utf-8") if isinstance(name, bytes) else name,
            "memory_total_mb": mem.total // (1024 * 1024),
            "driver_version": driver.decode("utf-8") if isinstance(driver, bytes) else driver,
        }

    def get_power_w(self) -> float:
        """Get current power usage in Watts."""
        import pynvml
        # Returns milliwatts
        return float(pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0)


def _nvidia_calibration_rejection_reasons(
    *,
    iterations: int,
    total_wh: float,
    total_weighted_tokens: int,
    wh_per_1k_input: float,
    wh_per_1k_output: float,
    sample_wh_per_weighted_token: list[float],
) -> list[str]:
    """Return reasons a heuristic NVIDIA calibration should not be installed."""
    reasons: list[str] = []
    if iterations < 3:
        reasons.append(f"iterations={iterations} < 3 minimum")
    if total_weighted_tokens <= 0:
        reasons.append("workload produced no weighted tokens")
    if not math.isfinite(total_wh) or total_wh <= 0:
        reasons.append(f"total_wh={total_wh:.8f} is not positive")
    for name, value in (
        ("wh_per_1k_input", wh_per_1k_input),
        ("wh_per_1k_output", wh_per_1k_output),
    ):
        if not math.isfinite(value) or value <= 0:
            reasons.append(f"{name}={value:.8f} is not positive")

    usable_samples = [v for v in sample_wh_per_weighted_token if math.isfinite(v) and v > 0]
    if len(usable_samples) != len(sample_wh_per_weighted_token):
        reasons.append("one or more calibration samples had non-positive energy/token")
    if len(usable_samples) >= 3:
        mean = sum(usable_samples) / len(usable_samples)
        variance = sum((v - mean) ** 2 for v in usable_samples) / len(usable_samples)
        cv = math.sqrt(variance) / mean if mean > 0 else float("inf")
        if cv > 2.0:
            reasons.append(f"run-to-run energy variance too high (cv={cv:.2f})")

    return reasons


def calibrate_model(
    provider: str,
    model: str,
    workload: Callable[[], tuple[int, int]],
    iterations: int = 5
) -> CalibrationResult:
    """Measure power draw for a specific model/workload.

    Args:
        provider: Inference provider (e.g. 'ollama', 'vllm')
        model: Model name
        workload: Function that runs an LLM call and returns (input_tokens, output_tokens)
        iterations: Number of samples to take

    Returns:
        CalibrationResult
    """
    _warn_experimental()
    if not is_gpu_available():
        raise RuntimeError(f"GPU not available: {get_gpu_error()}")

    with GPUMonitor() as monitor:
        gpu_info = monitor.get_gpu_info()

        total_wh = 0.0
        total_in = 0
        total_out = 0
        sample_wh_per_weighted_token: list[float] = []

        # Warmup
        workload()

        for _ in range(iterations):
            start_time = time.perf_counter()

            # Simple sampling in a tight loop isn't perfect, but
            # good enough for Alpha. Better to use energy counters if supported.
            # We'll use a thread or async in Beta.
            in_tokens, out_tokens = workload()

            duration = time.perf_counter() - start_time
            # Get final sample (heuristic)
            power_w = monitor.get_power_w()

            # wh = Watts * Hours
            wh = power_w * (duration / 3600.0)
            weighted_tokens = in_tokens + (out_tokens * 3)
            sample_wh_per_weighted_token.append(
                wh / weighted_tokens if weighted_tokens > 0 else 0.0
            )

            total_wh += wh
            total_in += in_tokens
            total_out += out_tokens

        # Simple split based on our 1:3 ratio for now
        # Beta will use separate input/output intensive workloads
        total_weighted_tokens = total_in + (total_out * 3)
        wh_per_token = total_wh / total_weighted_tokens if total_weighted_tokens > 0 else 0.0
        wh_per_1k_input = wh_per_token * 1000
        wh_per_1k_output = wh_per_token * 3000
        rejection_reasons = _nvidia_calibration_rejection_reasons(
            iterations=iterations,
            total_wh=total_wh,
            total_weighted_tokens=total_weighted_tokens,
            wh_per_1k_input=wh_per_1k_input,
            wh_per_1k_output=wh_per_1k_output,
            sample_wh_per_weighted_token=sample_wh_per_weighted_token,
        )
        if rejection_reasons:
            joined = "; ".join(rejection_reasons)
            raise RuntimeError(f"Calibration quality issues detected; not saving: {joined}")

        res = CalibrationResult(
            model=model,
            provider=provider,
            wh_per_1k_input=wh_per_1k_input,
            wh_per_1k_output=wh_per_1k_output,
            tier=1,  # Tier 1: single-sample heuristic; Tier 0 requires energy-counter methodology
            samples=iterations,
            gpu_name=gpu_info["name"]
        )

        save_calibration(res)
        return res


def save_calibration(res: CalibrationResult) -> None:
    """Save a CalibrationResult to ~/.vetch/calibrations/."""
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    path = CALIBRATION_DIR / f"{res.provider}_{res.model.replace(':', '_')}.json"

    data: dict[str, Any] = {
        "wh_per_1k_input": res.wh_per_1k_input,
        "wh_per_1k_output": res.wh_per_1k_output,
        "tier": res.tier,
        "samples": res.samples,
        "basis": f"Hardware calibration on {res.gpu_name} ({res.samples} samples)",
        "timestamp": time.time(),
        "gpu_name": res.gpu_name,
    }

    path.write_text(json.dumps(data, indent=2))


# Keep private alias for backward compatibility with any direct callers
_save_calibration = save_calibration


def load_calibration(provider: str, model: str) -> CalibrationResult | None:
    """Load a saved CalibrationResult from ~/.vetch/calibrations/, or None if absent."""
    path = CALIBRATION_DIR / f"{provider}_{model.replace(':', '_')}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return CalibrationResult(
            model=model,
            provider=provider,
            wh_per_1k_input=float(data["wh_per_1k_input"]),
            wh_per_1k_output=float(data["wh_per_1k_output"]),
            tier=int(data.get("tier", 0)),
            samples=int(data.get("samples", 0)),
            gpu_name=data.get("gpu_name", ""),
        )
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


def format_calibration_result(res: CalibrationResult) -> str:
    lines = [
        f"Calibration Complete for {res.model} ({res.provider})",
        f"Hardware: {res.gpu_name}",
        "----------------------------------------",
        f"Energy (Input):  {res.wh_per_1k_input:.4f} Wh/1k tokens",
        f"Energy (Output): {res.wh_per_1k_output:.4f} Wh/1k tokens",
    ]
    tier_labels = {
        0: "Tier 0 (Measured)",
        1: "Tier 1 (Vendor)",
        2: "Tier 2 (Research)",
        3: "Tier 3 (Estimated)",
    }
    lines.append(f"Confidence:      {tier_labels.get(res.tier, f'Tier {res.tier}')}")
    return "\n".join(lines) + "\n"
