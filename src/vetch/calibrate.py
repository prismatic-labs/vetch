"""Calibration module for hardware-level energy measurement.

This module provides tools to measure actual GPU power draw during
inference to improve upon Tier 3 (Proxy) estimates.

Note: This module is EXPERIMENTAL and requires pynvml (nvidia-ml-py3).
API may change in future versions.
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any, Callable, NamedTuple

# P2: Mark as experimental
warnings.warn(
    "vetch.calibrate is experimental. API may change. Requires: pip install nvidia-ml-py3",
    FutureWarning,
    stacklevel=2,
)

logger = logging.getLogger(__name__)

# Cache for hardware calibrations
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
    if not is_gpu_available():
        raise RuntimeError(f"GPU not available: {get_gpu_error()}")

    with GPUMonitor() as monitor:
        gpu_info = monitor.get_gpu_info()

        total_wh = 0.0
        total_in = 0
        total_out = 0

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

            total_wh += wh
            total_in += in_tokens
            total_out += out_tokens

        # Simple split based on our 1:3 ratio for now
        # Beta will use separate input/output intensive workloads
        wh_per_token = total_wh / (total_in + (total_out * 3))

        res = CalibrationResult(
            model=model,
            provider=provider,
            wh_per_1k_input=(wh_per_token * 1000),
            wh_per_1k_output=(wh_per_token * 3000),
            tier=0,  # Tier 0: Measured
            samples=iterations,
            gpu_name=gpu_info["name"]
        )

        _save_calibration(res)
        return res


def _save_calibration(res: CalibrationResult) -> None:
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    path = CALIBRATION_DIR / f"{res.provider}_{res.model.replace(':', '_')}.json"

    data = {
        "wh_per_1k_input": res.wh_per_1k_input,
        "wh_per_1k_output": res.wh_per_1k_output,
        "tier": 0,
        "basis": f"Hardware calibration on {res.gpu_name} ({res.samples} samples)",
        "timestamp": time.time()
    }
    path.write_text(json.dumps(data, indent=2))


def format_calibration_result(res: CalibrationResult) -> str:
    return (
        f"Calibration Complete for {res.model} ({res.provider})\n"
        f"Hardware: {res.gpu_name}\n"
        f"----------------------------------------\n"
        f"Energy (Input):  {res.wh_per_1k_input:.4f} Wh/1k tokens\n"
        f"Energy (Output): {res.wh_per_1k_output:.4f} Wh/1k tokens\n"
        f"Confidence:      Tier 0 (Measured)\n"
    )
