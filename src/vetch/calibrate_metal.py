"""Apple Silicon energy calibration.

Measures SoC energy (CPU + GPU + ANE) during Ollama inference and produces
Tier 0 hardware-measured coefficients. Two samplers:

- ``ioreport`` (default when available): the ``Energy Model`` counters of the
  IOReport framework, a monotonic energy accumulator per subsystem. Counter
  basis, no root, per-domain split. See ``vetch.ioreport``.
- ``powermetrics``: trapezoidal integration of the ``Combined Power`` line.
  Power-sampling basis, requires root, kept as the fallback.

IMPORTANT: only the powermetrics sampler requires root.

Supports VLMs: in addition to wh_per_1k_input and wh_per_1k_output, it
measures wh_per_image — the energy cost of vision-encoder processing per image.

Usage (CLI, preferred):
    vetch calibrate-apple-silicon --model moondream --precision apple-native
    sudo vetch calibrate-apple-silicon --sampler powermetrics ...   # fallback

Usage (Python API):
    from vetch.calibrate_metal import calibrate_apple_silicon
    result, path = calibrate_apple_silicon(
        model="moondream:latest", precision="apple-native",
    )
"""

from __future__ import annotations

import base64
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple
from urllib import request as urllib_request
from urllib.error import URLError

from vetch import ioreport
from vetch.calibrate_v2 import (
    IDLE_DRIFT_FLAG_PCT,
    IDLE_DRIFT_WARN_PCT,
    MAX_IDLE_UNCERTAINTY_ROW_SHARE_PCT,
    idle_uncertainty_row_share_pct,
)

if TYPE_CHECKING:
    from vetch.calibrate import CalibrationResult
    from vetch.calibrate_v2 import NestedFitResult

OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"

# Standard image size for community calibration submissions.
#
# Why 378px: moondream's vision encoder natively operates at 378×378.
# Sending images at this resolution avoids a server-side resize step, making
# measurements directly comparable across machines without any resize variance.
# For tiling VLMs (LLaVA-1.6, Qwen-VL) this is the single-tile size — use
# larger images to measure multi-tile cost separately.
#
# Community contributors MUST use the default to be included in aggregated
# datasets. The detail JSON records image_resolution_px so submissions can be
# filtered by resolution if needed.
CALIBRATION_IMAGE_SIZE_PX = 378

# Image set identifiers — recorded in the detail JSON so the aggregation script
# can group comparable runs. Bump the version suffix if the set changes.
CALIBRATION_IMAGE_SET_SYNTHETIC  = "vetch_standard_v1"    # default: seeded noise
CALIBRATION_IMAGE_SET_WIKIMEDIA  = "vetch_wikimedia_v1"   # preferred: real images

# Default used when the Wikimedia set is not downloaded yet
CALIBRATION_IMAGE_SET = CALIBRATION_IMAGE_SET_SYNTHETIC

# Where the Wikimedia standard images are stored after download
CALIB_IMAGES_DIR = Path.home() / ".vetch" / "calib_images"

# ---------------------------------------------------------------------------
# Standard Wikimedia image set
#
# Eight diverse, public-domain images fetched via the Wikimedia FilePath API
# at exactly CALIBRATION_IMAGE_SIZE_PX wide.  The API handles resizing
# server-side so no local image library is needed.
#
# URL format: commons.wikimedia.org/w/index.php?title=Special:FilePath
#             &file=<encoded_filename>&width=<px>
#
# All images are public domain or CC0.  The `source` field is the Wikimedia
# Commons file page — paste it in your calibration issue for full provenance.
# ---------------------------------------------------------------------------
WIKIMEDIA_IMAGES: list[dict[str, str]] = [
    {
        "name": "mona_lisa.jpg",
        "file": "Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg",
        "description": "Mona Lisa — portrait painting, fine detail",
        "license": "Public Domain",
        "source": "https://commons.wikimedia.org/wiki/File:Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg",
    },
    {
        "name": "great_wave.jpg",
        "file": "Great_Wave_off_Kanagawa2.jpg",
        "description": "The Great Wave off Kanagawa — woodblock print, high contrast",
        "license": "Public Domain",
        "source": "https://commons.wikimedia.org/wiki/File:Great_Wave_off_Kanagawa2.jpg",
    },
    {
        "name": "earthrise.jpg",
        "file": "NASA-Apollo8-Dec24-Earthrise.jpg",
        "description": "Earthrise (Apollo 8) — landscape with Earth against black space",
        "license": "Public Domain (NASA)",
        "source": "https://commons.wikimedia.org/wiki/File:NASA-Apollo8-Dec24-Earthrise.jpg",
    },
    {
        "name": "sunflower.jpg",
        "file": "Sunflower_from_Silesia2.jpg",
        "description": "Sunflower macro — close-up natural detail",
        "license": "Public Domain",
        "source": "https://commons.wikimedia.org/wiki/File:Sunflower_from_Silesia2.jpg",
    },
    {
        "name": "milky_way.jpg",
        "file": "Milky_Way_Night_Sky_Black_Rock_Desert_Nevada.jpg",
        "description": "Milky Way night sky — low-light, stars",
        "license": "CC0",
        "source": "https://commons.wikimedia.org/wiki/File:Milky_Way_Night_Sky_Black_Rock_Desert_Nevada.jpg",
    },
    {
        "name": "colosseum.jpg",
        "file": "Colosseum_in_Rome%2C_Italy_-_April_2007.jpg",
        "description": "Colosseum — architecture, stone texture",
        "license": "CC BY-SA 3.0",
        "source": "https://commons.wikimedia.org/wiki/File:Colosseum_in_Rome,_Italy_-_April_2007.jpg",
    },
    {
        "name": "periodic_table.png",
        "file": "Simple_Periodic_Table_Chart-en.svg",
        "description": "Periodic table — text-dense structured diagram",
        "license": "Public Domain",
        "source": "https://commons.wikimedia.org/wiki/File:Simple_Periodic_Table_Chart-en.svg",
    },
    {
        "name": "mandrill.png",
        "file": "Mandrill_Zoo_of_Atlanta.jpg",
        "description": "Mandrill portrait — vivid colours, facial detail",
        "license": "CC BY 2.0",
        "source": "https://commons.wikimedia.org/wiki/File:Mandrill_Zoo_of_Atlanta.jpg",
    },
]

# ---------------------------------------------------------------------------
# Wikimedia image download and loading
# ---------------------------------------------------------------------------

def wikimedia_image_url(file_encoded: str, width: int = CALIBRATION_IMAGE_SIZE_PX) -> str:
    return (
        f"https://commons.wikimedia.org/w/index.php"
        f"?title=Special:FilePath&file={file_encoded}&width={width}"
    )


def download_calibration_images(
    target_dir: Path = CALIB_IMAGES_DIR,
    *,
    strict: bool = False,
) -> None:
    """Download the Wikimedia standard image set using curl.

    Safe to run multiple times — skips files that already exist.
    Does NOT need sudo; run this before calibrating.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    total = len(WIKIMEDIA_IMAGES)
    print(f"Downloading {total} standard calibration images to {target_dir}")
    for i, img in enumerate(WIKIMEDIA_IMAGES, 1):
        dest = target_dir / img["name"]
        if dest.exists():
            print(f"  [{i}/{total}] {img['name']} — already present, skipping")
            continue
        url = wikimedia_image_url(img["file"])
        print(f"  [{i}/{total}] {img['name']} ({img['description']})...")
        result = subprocess.run(
            ["curl", "-fsSL", "-o", str(dest), "-L", url],
            capture_output=True,
        )
        if result.returncode != 0:
            msg = (
                f"    WARNING: download failed (curl exit {result.returncode}). "
                f"Skipping — will use synthetic image for this slot."
            )
            print(msg)
            if strict:
                raise RuntimeError(f"Failed to download {img['name']} (strict mode)")
        else:
            size_kb = dest.stat().st_size // 1024
            print(f"    {size_kb} KB")
    downloaded = sum(1 for img in WIKIMEDIA_IMAGES if (target_dir / img["name"]).exists())
    print(f"\n{downloaded}/{total} images ready in {target_dir}")
    if strict and downloaded < total:
        raise RuntimeError(
            f"Only {downloaded}/{total} Wikimedia images available. "
            "Re-run with network access or without --strict-images."
        )


def _wikimedia_images_ready(target_dir: Path = CALIB_IMAGES_DIR) -> bool:
    """True if at least one Wikimedia standard image is present on disk."""
    return any((target_dir / img["name"]).exists() for img in WIKIMEDIA_IMAGES)


def _load_image_set(target_dir: Path = CALIB_IMAGES_DIR) -> tuple[list[str], str]:
    """Return (list_of_base64_strings, image_set_name).

    Uses Wikimedia images where downloaded; fills any gaps with synthetic
    noise so the pool always has exactly len(WIKIMEDIA_IMAGES) entries.
    Falls back entirely to synthetic if nothing has been downloaded.
    """
    images_b64: list[str] = []
    missing: list[str] = []

    for i, img in enumerate(WIKIMEDIA_IMAGES):
        path = target_dir / img["name"]
        if path.exists():
            images_b64.append(base64.b64encode(path.read_bytes()).decode())
        else:
            images_b64.append(_unique_image_b64(seed=1000 + i))
            missing.append(img["name"])

    if not missing:
        return images_b64, CALIBRATION_IMAGE_SET_WIKIMEDIA

    if missing and len(missing) < len(WIKIMEDIA_IMAGES):
        print(
            f"NOTE: {len(missing)} Wikimedia image(s) missing "
            f"({', '.join(missing)}); using synthetic noise for those probe/warmup slots.\n"
            f"      Re-run --fetch-images to complete the download (optional).",
            file=sys.stderr,
        )
        return images_b64, CALIBRATION_IMAGE_SET_WIKIMEDIA

    # Nothing downloaded at all — synthetic images are used for probe/warmup.
    # Grid runs always use unique seeded images regardless.
    print(
        "NOTE: Wikimedia standard images not found. Using synthetic images for probe/warmup.\n"
        "      Run --fetch-images to download them (optional; does not affect measurements).\n",
        file=sys.stderr,
    )
    return images_b64, CALIBRATION_IMAGE_SET_SYNTHETIC


# Documented visual token counts for known fixed-encoder VLMs
_KNOWN_VISUAL_TOKENS: dict[str, int] = {
    "moondream": 729,
    "moondream2": 729,
    "llava": 576,
    "bakllava": 576,
}


# ---------------------------------------------------------------------------
# Platform checks
# ---------------------------------------------------------------------------

def is_apple_silicon() -> bool:
    """Return True if running on Apple Silicon (M-series)."""
    if platform.system() != "Darwin":
        return False
    r = subprocess.run(
        ["sysctl", "-n", "hw.optional.arm64"],
        capture_output=True, text=True,
    )
    return r.returncode == 0 and r.stdout.strip() == "1"


def assert_apple_silicon() -> None:
    if not is_apple_silicon():
        print(
            "ERROR: calibrate-apple-silicon requires macOS on Apple Silicon (M1/M2/M3/M4/M5).\n"
            "Use 'vetch calibrate' for NVIDIA GPU calibration on other platforms.",
            file=sys.stderr,
        )
        sys.exit(1)


def assert_root() -> None:
    if os.geteuid() != 0:
        print(
            "ERROR: powermetrics requires root access.\n"
            "\n"
            "Why: powermetrics reads Apple PMU (power management unit) counters\n"
            "that expose per-subsystem power (CPU, GPU, ANE). Access is gated on\n"
            "root on all recent macOS versions.\n"
            "\n"
            "Re-run with sudo, preserving HOME so the record is written to\n"
            "your own calibration directory rather than root's:\n"
            "  sudo HOME=\"$HOME\" vetch calibrate-apple-silicon \\\n"
            "      --model llama3.2:3b --precision gguf:q4_k_m",
            file=sys.stderr,
        )
        sys.exit(1)


def warn_if_home_is_root() -> None:
    """Say where the record will land before spending the run, not after.

    ``sudo`` resets HOME to /var/root under the default macOS sudoers policy,
    and the calibration directory is derived from ``Path.home()``. Without this
    the run succeeds, writes a root-owned record somewhere the SDK will never
    look, and the operator finds out by the coefficient never attaching.
    """
    if os.geteuid() != 0:
        return
    sudo_user = os.environ.get("SUDO_USER")
    if not sudo_user:
        return
    from vetch.calibrate import CALIBRATION_DIR

    if sudo_user in CALIBRATION_DIR.parts:
        return
    print(
        f"WARNING: running under sudo as {sudo_user!r}, but HOME resolves to "
        f"{Path.home()}.\n"
        f"         The record will be written to {CALIBRATION_DIR}, which is\n"
        f"         root-owned and is NOT where inference looks for it.\n"
        f"         Cancel now and re-run as:\n"
        f'           sudo HOME="$HOME" vetch calibrate-apple-silicon ...',
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Hardware metadata
# ---------------------------------------------------------------------------

@dataclass
class HardwareInfo:
    chip_raw: str
    memory_gb: int
    macos_version: str
    chip_family: str = ""
    chip_tier: str = ""

    def __post_init__(self) -> None:
        # Normalize "Apple M3 Max" → family="M3", tier="Max"
        m = re.match(r"Apple (M\d+)\s*(\w+)?", self.chip_raw)
        if m:
            self.chip_family = m.group(1)
            self.chip_tier = m.group(2) or ""


def get_hardware_info() -> HardwareInfo:
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType", "-json"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        data = json.loads(out)
        hw = data["SPHardwareDataType"][0]
        chip_raw = hw.get("cpu_type", hw.get("chip_type", "Apple Silicon"))
        mem_str = hw.get("physical_memory", "0 GB")
        mem_gb = int(re.search(r"\d+", mem_str).group()) if re.search(r"\d+", mem_str) else 0  # type: ignore[union-attr]
        macos = platform.mac_ver()[0]
        return HardwareInfo(chip_raw=chip_raw, memory_gb=mem_gb, macos_version=macos)
    except Exception:
        return HardwareInfo(
            chip_raw="Apple Silicon",
            memory_gb=0,
            macos_version=platform.mac_ver()[0],
        )


# ---------------------------------------------------------------------------
# Power state checks
# ---------------------------------------------------------------------------

def assert_power_state_usable(
    power_state: Mapping[str, Any], *, allow_degraded: bool = False
) -> None:
    """Refuse a run whose record could never be installed. Checked up front.

    Being on battery or in Low Power Mode already makes the finished record
    non-installable via `_active_calibration_rejection_reasons`, and Low Power
    Mode additionally throttles the very silicon being measured. Detecting that
    at the end, after the full grid has run, spends the GPU time to produce a
    record nothing will use -- and the same conditions drive the idle drift that
    then flags the run anyway.

    ``on_ac_power`` of None means pmset could not be read. That is unknown, not
    bad, so it is allowed through: the check fails open.
    """
    blockers: list[str] = []
    if power_state.get("on_ac_power") is False:
        blockers.append("running on battery (plug in AC power)")
    if power_state.get("low_power_mode"):
        blockers.append("Low Power Mode is active (disable it in System Settings)")
    if not blockers:
        return
    if allow_degraded:
        print(
            "  Proceeding under --allow-degraded-power: "
            + "; ".join(blockers)
            + ".\n  The record will be written and flagged, but will NOT be "
            "installed as the active calibration.",
            file=sys.stderr,
        )
        return
    print(
        "ERROR: refusing to start; this run could not produce an installable "
        "calibration.\n\n  "
        + "\n  ".join(blockers)
        + "\n\nBoth conditions make the finished record non-installable, and "
        "Low Power Mode\nthrottles the silicon being measured. Fix them and "
        "re-run, or pass\n--allow-degraded-power to measure anyway and keep "
        "the result flagged.",
        file=sys.stderr,
    )
    sys.exit(2)


# Sustained GPU load on an Apple Silicon laptop can exceed what a small adapter
# supplies, and the machine then draws the difference from the battery while
# `pmset -g ps` still reports "AC Power". That is a different power and thermal
# regime from a fully-supplied run, it shifts as the battery drains, and it is a
# plausible source of idle drift between the start and end of a grid. Apple
# ships 70 W and larger adapters with the machines this path targets, so
# anything under that is worth disclosing on the record.
ADAPTER_WATTS_ADVISORY = 70

# Idle baselines are taken identically at both ends of a run: settle, then
# sample. Asymmetry between them shows up as "drift" that is really a protocol
# artifact, and it biases the mean that every row's idle subtraction uses.
IDLE_SETTLE_S = 2.0
IDLE_WINDOW_S = 3.0


def classify_idle_drift(
    idle_watts_before: float,
    idle_watts_after: float,
) -> tuple[float, str]:
    """Return the drift percentage and its non-fatal measurement signal."""
    drift_pct = (
        abs(idle_watts_after - idle_watts_before)
        / max(idle_watts_before, 0.01)
        * 100
    )
    if drift_pct > IDLE_DRIFT_FLAG_PCT:
        return drift_pct, "flag"
    if drift_pct > IDLE_DRIFT_WARN_PCT:
        return drift_pct, "warn"
    return drift_pct, "ok"


def fit_record_blocks(fit: Any) -> dict[str, Any]:
    """The record blocks that come from the fit rather than from the sampler.

    Keep fit-derived metadata together so the writer cannot omit the status,
    rung, or energy basis when it persists a calibration.
    """
    from vetch.calibrate_v2 import nested_fit_to_dict

    return {
        "coefficient_meta": nested_fit_to_dict(fit),
        "energy_basis": fit.energy_basis,
    }


def measure_idle_window(
    label: str,
    *,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[float, float]:
    """Settle, then record one idle sampling window. Returns (start_ms, end_ms).

    Both baselines go through here so they cannot drift apart. They did: the
    "after" window once had no settle at all, so a figure taken off a workload
    tail was compared against one taken after a pause. `idle_drift_pct` and the
    mean that every row's idle subtraction uses both inherited that asymmetry,
    and across four runs idle_after came in below idle_before three times --
    not the direction thermal drift predicts.

    `sleep` and `clock` are injectable so the symmetry is testable without a
    powermetrics session.
    """
    if label:
        print(f"Measuring idle baseline ({label})...")
    sleep(IDLE_SETTLE_S)
    start = clock() * 1000
    sleep(IDLE_WINDOW_S)
    return start, clock() * 1000


def check_power_state() -> dict[str, Any]:
    result: dict[str, Any] = {
        "on_ac_power": None,
        "low_power_mode": False,
        "high_power_mode": False,
        "adapter_watts": None,
        "battery_percent": None,
        "battery_charging": None,
    }
    try:
        batt = subprocess.check_output(
            ["pmset", "-g", "ps"], text=True, stderr=subprocess.DEVNULL
        )
        result["on_ac_power"] = "AC Power" in batt
        pct = re.search(r"(\d+)%", batt)
        if pct:
            result["battery_percent"] = int(pct.group(1))
        lowered = batt.lower()
        if "discharging" in lowered:
            result["battery_charging"] = "discharging"
        elif "not charging" in lowered:
            result["battery_charging"] = "not_charging"
        elif "charging" in lowered:
            result["battery_charging"] = "charging"
        if not result["on_ac_power"]:
            print(
                "WARNING: Running on battery. Energy measurements will be less accurate\n"
                "and may vary. Plug in AC power for best results.",
                file=sys.stderr,
            )
    except Exception:
        pass
    try:
        adapter = subprocess.check_output(
            ["pmset", "-g", "adapter"], text=True, stderr=subprocess.DEVNULL
        )
        watts = re.search(r"Wattage\s*=\s*(\d+)", adapter)
        if watts:
            result["adapter_watts"] = int(watts.group(1))
            if result["adapter_watts"] < ADAPTER_WATTS_ADVISORY:
                print(
                    f"NOTE: {result['adapter_watts']} W power adapter. Sustained GPU load "
                    f"can exceed this,\n      in which case the machine supplements from "
                    f"the battery while still\n      reporting AC power. That is a "
                    f"different power regime and can show up\n      as idle drift between "
                    f"the start and end of the run. Recorded on the\n      calibration "
                    f"record; a {ADAPTER_WATTS_ADVISORY} W or larger adapter is preferable.",
                    file=sys.stderr,
                )
    except Exception:
        pass
    try:
        pmset = subprocess.check_output(
            ["pmset", "-g"], text=True, stderr=subprocess.DEVNULL
        )
        if "lowpowermode" in pmset:
            m = re.search(r"lowpowermode\s+(\d+)", pmset)
            if m and m.group(1) == "1":
                result["low_power_mode"] = True
                print(
                    "WARNING: Low Power Mode is active. This throttles CPU/GPU and will\n"
                    "understate normal operating energy. Disable it for accurate calibration.",
                    file=sys.stderr,
                )
        if "highpowermode" in pmset:
            m = re.search(r"highpowermode\s+(\d+)", pmset)
            if m and m.group(1) == "1":
                result["high_power_mode"] = True
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

SAMPLER_IOREPORT = "ioreport"
SAMPLER_POWERMETRICS = "powermetrics"
SAMPLER_CHOICES = ("auto", SAMPLER_IOREPORT, SAMPLER_POWERMETRICS)

# `energy_source` drives the admission rule in `calibrate_v2.admit_rows`: a
# source naming power sampling gets the 2 s duration floor, a counter does
# not. `measurement_basis` is derived from the sampler so the two cannot
# disagree, as `calibrate_cuda._measurement_basis` does for NVML.
ENERGY_SOURCE_BY_SAMPLER = {
    SAMPLER_IOREPORT: "ioreport_energy_counter",
    SAMPLER_POWERMETRICS: "powermetrics",
}
MEASUREMENT_BASIS_BY_SAMPLER = {
    SAMPLER_IOREPORT: "ioreport_soc_energy",
    SAMPLER_POWERMETRICS: "powermetrics_estimated_soc_power",
}
INTEGRATION_METHOD_BY_SAMPLER = {
    SAMPLER_IOREPORT: "energy_counter",
    SAMPLER_POWERMETRICS: "trapezoidal",
}


def resolve_sampler(requested: str) -> str:
    """Pick the sampler: IOReport when it can be read, else powermetrics.

    ``auto`` falls back rather than failing, per the fail-open rule; an explicit
    ``ioreport`` request on a machine where it is unavailable is an error, so a
    record is never silently produced on a different basis from the one asked
    for.
    """
    if requested not in SAMPLER_CHOICES:
        raise ValueError(f"sampler must be one of {SAMPLER_CHOICES}, got {requested!r}")
    if requested == SAMPLER_POWERMETRICS:
        return SAMPLER_POWERMETRICS
    if ioreport.available():
        return SAMPLER_IOREPORT
    if requested == SAMPLER_IOREPORT:
        raise RuntimeError(
            "IOReport energy counters are not readable on this machine; "
            "re-run with --sampler powermetrics (requires sudo)."
        )
    return SAMPLER_POWERMETRICS


# ---------------------------------------------------------------------------
# powermetrics monitor
# ---------------------------------------------------------------------------

@dataclass
class PowerSample:
    mono_ms: float   # monotonic time in ms
    combined_watts: float


@dataclass
class MonitorInfo:
    format_used: str = "text"
    gpu_power_captured: bool = False
    sample_interval_ms: int = 100
    samplers: list[str] = field(default_factory=lambda: ["cpu_power", "gpu_power", "ane_power"])


_COMBINED_POWER_RE = re.compile(
    r"\b(?:"
    r"combined\s+power"
    r"|combined\s+package\s+power"
    r"|total\s+(?:soc\s+)?power"
    r"|soc\s+power"
    r"|cpu\s*\+\s*gpu\s*\+\s*ane\s+power"
    r")\b[^:\n]*:\s*([\d.]+)\s*(mW|W)\b",
    re.IGNORECASE,
)


def _parse_combined_power_watts(line: str) -> float | None:
    """Parse a combined SoC power line from powermetrics text output."""
    m = _COMBINED_POWER_RE.search(line)
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2).lower()
    return value / 1000.0 if unit == "mw" else value


class AppleSiliconMonitor:
    """Context manager that streams powermetrics samples in the background.

    Parses "Combined Power (CPU + GPU + ANE): NNNN mW" lines from the text
    output of powermetrics. Provides trapezoidal integration over any window.
    """

    def __init__(self, sample_interval_ms: int = 100) -> None:
        self._interval_ms = sample_interval_ms
        self._samples: list[PowerSample] = []
        self._reader_thread: Any = None
        self._stop_event: Any = None
        self.info = MonitorInfo(sample_interval_ms=sample_interval_ms)

    def _parse_line(self, line: str, mono_ms: float) -> None:
        """Parse a single powermetrics output line and append a sample if it matches.

        Not called by _collect_loop (which batches and timestamps backward), but
        used by unit tests to verify the parsing regex independently.
        """
        watts = _parse_combined_power_watts(line)
        if watts is not None:
            self._samples.append(PowerSample(mono_ms=mono_ms, combined_watts=watts))
            if not self.info.gpu_power_captured:
                self.info.gpu_power_captured = True
            return
        if not self.info.gpu_power_captured:
            if re.search(r"GPU Power:\s*[\d.]+\s*(?:mW|W)\b", line):
                self.info.gpu_power_captured = True

    def _collect_loop(self) -> None:
        """Collect power samples by running powermetrics in 3-second batches.

        Using subprocess.run() with a fixed -n count avoids the stdio buffering
        issue that occurs when powermetrics writes to a pipe in long-running mode:
        without a TTY, powermetrics uses full block buffering and data only arrives
        after the buffer fills or the process exits — too late for integrate() calls.

        Timestamps are anchored to t_after (when the subprocess returns) and
        assigned backward at interval_ms spacing so the last sample sits at t_after
        and all earlier samples are correctly placed even with variable startup time.
        """
        n_per_batch = max(1, 3000 // self._interval_ms)  # ~3 seconds of samples
        batch_timeout = n_per_batch * self._interval_ms / 1000 + 10

        while not self._stop_event.is_set():
            try:
                result = subprocess.run(
                    [
                        "powermetrics",
                        "--samplers", "cpu_power,gpu_power,ane_power",
                        "--sample-rate", str(self._interval_ms),
                        "-n", str(n_per_batch),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=batch_timeout,
                )
            except (subprocess.TimeoutExpired, OSError):
                continue
            t_after = time.monotonic() * 1000

            # Collect watts values in order, then assign timestamps backward from
            # t_after so the last sample sits at t_after and earlier samples are at
            # regular interval_ms spacing before it.
            parsed_watts: list[float] = []
            for line in result.stdout.splitlines():
                watts = _parse_combined_power_watts(line)
                if watts is not None:
                    parsed_watts.append(watts)
                elif not self.info.gpu_power_captured:
                    if re.search(r"GPU Power:\s*[\d.]+\s*(?:mW|W)\b", line):
                        self.info.gpu_power_captured = True

            n = len(parsed_watts)
            for i, watts in enumerate(parsed_watts):
                mono_ms = t_after - (n - 1 - i) * self._interval_ms
                self._samples.append(PowerSample(mono_ms=mono_ms, combined_watts=watts))
                if not self.info.gpu_power_captured:
                    self.info.gpu_power_captured = True

    def __enter__(self) -> AppleSiliconMonitor:
        import threading

        self._stop_event = threading.Event()
        self._reader_thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._reader_thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        if self._stop_event:
            self._stop_event.set()
        if self._reader_thread:
            # Wait for the current batch to finish (up to 3s measurement + startup margin)
            self._reader_thread.join(timeout=20)

    def integrate(self, t_start_mono_ms: float, t_end_mono_ms: float) -> float:
        """Trapezoidal integration of combined power over [t_start, t_end] in Wh.

        Includes the nearest samples just outside the window boundaries so that
        partial intervals at the edges are covered, not silently dropped.
        """
        all_samples = list(self._samples)  # snapshot to avoid race with reader thread
        if not all_samples:
            return 0.0

        # Find the last sample strictly before t_start (left anchor)
        left_anchor = None
        for s in reversed(all_samples):
            if s.mono_ms < t_start_mono_ms:
                left_anchor = s
                break

        # Find the first sample strictly after t_end (right anchor)
        right_anchor = None
        for s in all_samples:
            if s.mono_ms > t_end_mono_ms:
                right_anchor = s
                break

        # Window samples
        window = [s for s in all_samples if t_start_mono_ms <= s.mono_ms <= t_end_mono_ms]

        def _interp(s0: PowerSample, s1: PowerSample, t: float) -> PowerSample:
            frac = (t - s0.mono_ms) / (s1.mono_ms - s0.mono_ms)
            w = s0.combined_watts + frac * (s1.combined_watts - s0.combined_watts)
            return PowerSample(mono_ms=t, combined_watts=w)

        # For left boundary: interpolate from (left_anchor → first window sample) or
        # directly between anchors when window is empty.
        boundary_samples: list[PowerSample] = []
        right_first = window[0] if window else right_anchor
        if left_anchor is not None and right_first is not None:
            if right_first.mono_ms > left_anchor.mono_ms:
                boundary_samples = [_interp(left_anchor, right_first, t_start_mono_ms)]

        # For right boundary: interpolate from (last window sample → right_anchor).
        tail_samples: list[PowerSample] = []
        left_last = window[-1] if window else left_anchor
        if right_anchor is not None and left_last is not None:
            if right_anchor.mono_ms > left_last.mono_ms:
                tail_samples = [_interp(left_last, right_anchor, t_end_mono_ms)]

        effective = boundary_samples + window + tail_samples
        if len(effective) < 2:
            if not effective:
                return 0.0
            duration_h = (t_end_mono_ms - t_start_mono_ms) / 3_600_000.0
            return effective[0].combined_watts * duration_h

        wh = 0.0
        for i in range(1, len(effective)):
            dt_h = (effective[i].mono_ms - effective[i - 1].mono_ms) / 3_600_000.0
            avg_w = (effective[i].combined_watts + effective[i - 1].combined_watts) / 2.0
            wh += avg_w * dt_h
        return wh

    def mean_watts(self, t_start_mono_ms: float, t_end_mono_ms: float) -> float:
        """Mean watts over the window (for idle baseline measurement)."""
        samples_in_window = [
            s for s in self._samples
            if t_start_mono_ms <= s.mono_ms <= t_end_mono_ms
        ]
        if not samples_in_window:
            return 0.0
        return sum(s.combined_watts for s in samples_in_window) / len(samples_in_window)


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def _check_ollama(base_url: str) -> str:
    """Return ollama version string or raise RuntimeError.

    Verifies both the CLI (for version) and the HTTP server at base_url.
    """
    import json as _json
    from urllib import request as _urllib_request
    from urllib.error import URLError

    # Verify the HTTP server at base_url is reachable
    version_url = base_url.rstrip("/") + "/api/version"
    try:
        with _urllib_request.urlopen(version_url, timeout=5) as resp:
            data = _json.loads(resp.read())
            http_version = data.get("version", "")
    except (URLError, OSError, ValueError) as e:
        raise RuntimeError(
            f"Ollama server not reachable at {base_url}. "
            f"Start Ollama and ensure it is listening on that address. ({e})"
        ) from e

    # Also try CLI version as a secondary check (non-fatal if CLI absent)
    try:
        r = subprocess.run(
            ["ollama", "--version"], capture_output=True, text=True, timeout=5
        )
        cli_version = r.stdout.strip()
        return cli_version if cli_version else f"ollama/{http_version}"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return f"ollama/{http_version}"


def _ollama_generate(
    base_url: str,
    model: str,
    prompt: str,
    image_b64: str | None = None,
    max_tokens: int = 32,
    num_ctx: int | None = None,
) -> tuple[int, int]:
    """Run one Ollama generate call. Returns (prompt_eval_count, eval_count).

    ``num_ctx`` must be pinned for calibration. Ollama's default is 4096 and it
    truncates a longer prompt rather than refusing it, so an unpinned run would
    quietly measure a 4k prefill point while the design believed it asked for
    32k. Pass ONE value for the whole grid: changing ``num_ctx`` between calls
    makes Ollama evict and reload the model, which on unified memory costs
    several GB of churn per change and would land inside a timing window.
    """
    # repeat_penalty=1.0 is load-bearing, not a style choice. The default
    # penalty makes each further item of a long enumerated list less likely and
    # drives the model to EOS: measured on llama3.2, the same prompt at the same
    # cap of 2048 realised 416 tokens under the default sampler and 2048 with
    # the penalty neutralised. Sampling parameters do not change the energy of a
    # forward pass, so this holds output length without distorting what is being
    # measured. Recorded in provenance.extra["sampling"].
    options: dict[str, Any] = {"num_predict": max_tokens, "repeat_penalty": 1.0}
    if num_ctx is not None:
        options["num_ctx"] = int(num_ctx)
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }
    if image_b64 is not None:
        payload["images"] = [image_b64]

    body = json.dumps(payload).encode()
    req = urllib_request.Request(
        f"{base_url}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
    except URLError as e:
        raise RuntimeError(f"Ollama request failed: {e}") from e

    prompt_tokens = data.get("prompt_eval_count") or 0
    output_tokens = data.get("eval_count") or 0
    return int(prompt_tokens), int(output_tokens)


def _probe_image_token_accounting(
    base_url: str, model: str, image_b64: str
) -> dict[str, Any]:
    """Check whether Ollama includes image tokens in prompt_eval_count."""
    text_prompt = "What color is the sky?"
    text_only, _ = _ollama_generate(base_url, model, text_prompt, max_tokens=1)
    with_image, _ = _ollama_generate(base_url, model, text_prompt, image_b64, max_tokens=1)
    delta = with_image - text_only
    accounting = "includes" if delta > 10 else "excludes"
    return {
        "image_token_accounting": accounting,
        "delta_prompt_eval_count_with_image": delta,
        "text_only_prompt_eval_count": text_only,
        "with_image_prompt_eval_count": with_image,
    }


def _get_visual_tokens_per_image(model: str) -> tuple[int, str]:
    """Return (visual_token_count, source) for a VLM model."""
    base = model.split(":")[0].lower()
    for key, count in _KNOWN_VISUAL_TOKENS.items():
        if key in base:
            return count, "known_constant"
    return 729, "assumed_default"


# ---------------------------------------------------------------------------
# Unique prompts and images
# ---------------------------------------------------------------------------

def _unique_prompt(approx_tokens: int, seed: int, token_scale: float = 1.0) -> str:
    """Generate a unique prompt of approximately the right token count.

    ``token_scale`` corrects the words-per-token heuristic below against the
    model's real tokenizer. Measured on llama3.2 the heuristic under-delivers by
    about 22% -- a point asking for 1024 tokens realises 803 -- which is enough
    to drop the smallest prefill point under PREFILL_ARM_MIN_INPUT_TOKENS and
    silently shrink the arm. Pass the ratio from :func:`probe_prefill` so a
    design point means the token count it claims.
    """
    rng = random.Random(seed)
    nonce = f"[ref:{rng.randint(100000, 999999)}]"
    filler_words = [
        "the", "a", "an", "is", "was", "are", "were", "has", "had", "have",
        "will", "would", "could", "should", "may", "might", "must", "shall",
        "blue", "red", "green", "large", "small", "fast", "slow", "bright",
        "dark", "warm", "cold", "old", "new", "good", "bad", "long", "short",
        "river", "mountain", "forest", "ocean", "city", "town", "house", "road",
        "sun", "moon", "star", "cloud", "rain", "wind", "fire", "stone", "tree",
    ]
    # Rough: 1 token ≈ 0.75 words
    scale = token_scale if token_scale and token_scale > 0 else 1.0
    target_words = max(4, int(approx_tokens * 0.75 * scale))
    words = [rng.choice(filler_words) for _ in range(target_words - 4)]
    # The instruction has to hold output length, not just input length.
    # `num_predict` is a CAP, not a target: the model stops at EOS and the
    # decode arm collapses. Measured on llama3.2, "Describe in detail ...
    # Respond briefly." realised 118-183 tokens against caps of 512/2048/4096 --
    # a ~65-token span on the axis that identifies wh_per_1k_output. An
    # enumeration with no natural end reaches 512/2048 exactly and 2982 of 4096.
    # (Ollama accepts `ignore_eos` and silently ignores it, so the prompt is the
    # only lever; vLLM's min_tokens/ignore_eos do work, which is why the CUDA
    # path never needed this.)
    prompt = (
        f"{nonce} Using these terms: {' '.join(words)}. "
        "Write a numbered list of 2000 distinct short factual statements about "
        "them. Do not stop early, do not summarise, do not conclude."
    )
    return prompt


def _unique_image_b64(seed: int, size: int = CALIBRATION_IMAGE_SIZE_PX) -> str:
    """Generate a unique PNG image as a base64 string (no Pillow dependency)."""
    rng = random.Random(seed)

    # Minimal 1x1 PNG with a random color, embedded in a larger canvas comment
    # We use a simple approach: generate raw RGB bytes and encode as BMP-style
    # data URL, but since we need PNG format for Ollama, build a minimal PNG.

    def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        import struct
        import zlib
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        return length + chunk_type + data + crc

    import struct
    import zlib

    # Generate unique pixel data (size×size RGB)
    pixels: list[int] = []
    for _y in range(size):
        for _x in range(size):
            r = rng.randint(0, 255)
            g = rng.randint(0, 255)
            b = rng.randint(0, 255)
            pixels.extend([r, g, b])

    # Build raw image data: filter byte (0 = None) + row data
    raw_rows = bytearray()
    for y in range(size):
        raw_rows.append(0)  # filter byte
        for x in range(size):
            idx = (y * size + x) * 3
            raw_rows.extend(pixels[idx:idx + 3])

    compressed = zlib.compress(bytes(raw_rows), level=1)

    png = b"\x89PNG\r\n\x1a\n"  # PNG signature
    # IHDR chunk
    ihdr_data = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)  # 8-bit RGB
    png += _png_chunk(b"IHDR", ihdr_data)
    # IDAT chunk
    png += _png_chunk(b"IDAT", compressed)
    # IEND chunk
    png += _png_chunk(b"IEND", b"")

    return base64.b64encode(png).decode()


def _unique_audio_b64(seed: int, n_samples: int = 256) -> str:
    """Generate a unique minimal WAV (PCM16 mono 8 kHz) as base64.

    Synthetic only — hardware measurement validates server acceptance. Each seed
    produces distinct PCM so serving-side caches cannot collapse requests.
    """
    import struct

    rng = random.Random(seed)
    pcm = bytearray()
    for i in range(n_samples):
        # Deterministic unique samples from seed + index.
        sample = int(rng.randint(-16000, 16000)) ^ (i & 0xFF)
        sample = max(-32768, min(32767, sample))
        pcm += struct.pack("<h", sample)

    data_size = len(pcm)
    byte_rate = 8000 * 2  # mono 16-bit
    wav = bytearray()
    wav += b"RIFF"
    wav += struct.pack("<I", 36 + data_size)
    wav += b"WAVEfmt "
    wav += struct.pack("<IHHIIHH", 16, 1, 1, 8000, byte_rate, 2, 16)
    wav += b"data"
    wav += struct.pack("<I", data_size)
    wav += pcm
    return base64.b64encode(bytes(wav)).decode()


def _unique_video_b64(seed: int) -> str:
    """Generate a unique synthetic video-shaped payload as base64.

    Not a real codec bitstream — a seed-tagged container so each request is
    unique for calibration. Hardware runs validate whether the serving stack
    accepts the payload; unit tests only assert uniqueness and non-empty output.
    """
    rng = random.Random(seed)
    # Minimal ISO BMFF-ish 'ftyp' + 'mdat' with unique payload.
    brand = b"isom"
    ftyp = b"ftyp" + brand + (b"\x00" * 4) + brand
    ftyp_box = (8 + len(ftyp)).to_bytes(4, "big") + ftyp
    payload = bytes(rng.randint(0, 255) for _ in range(64))
    # Embed seed so uniqueness is obvious even if RNG collides across platforms.
    payload = seed.to_bytes(4, "big", signed=False) + payload
    mdat = b"mdat" + payload
    mdat_box = (8 + len(mdat)).to_bytes(4, "big") + mdat
    return base64.b64encode(ftyp_box + mdat_box).decode()


def _unique_media_b64(modality: str, seed: int) -> str:
    """Dispatch to the synthetic generator for ``image`` / ``audio`` / ``video``."""
    if modality == "audio":
        return _unique_audio_b64(seed)
    if modality == "video":
        return _unique_video_b64(seed)
    return _unique_image_b64(seed)


def _warmup_image_b64(model_is_vlm: bool, image_pool: list[str]) -> str | None:
    """Return the warm-up image only when the probed model supports images."""
    if not model_is_vlm:
        return None
    return image_pool[1 % len(image_pool)]


# ---------------------------------------------------------------------------
# Workload grid
# ---------------------------------------------------------------------------

@dataclass
class WorkloadSpec:
    n_images: int
    approx_text_tokens: int
    max_tokens: int
    replicate: bool = False  # True = anchor cell, run multiple times
    # Identical requests aggregated into ONE timing window. Prefill-isolation
    # points are short by construction (output pinned at 8 tokens) and a single
    # one can fall under the power-sampling duration floor. Repeating the point
    # lengthens the window without moving the design point; the row then records
    # summed tokens and summed energy. The fit normalises those totals back to
    # one request before building its design matrix, including the constant
    # intercept column.
    requests: int = 1


# Ceiling on the context Apple Silicon calibration will ask Ollama to allocate.
# The KV cache for a long context lives in the same unified memory as the model
# weights, so an unbounded ceiling turns a calibration run into an OOM on the
# 16 GB machines this path targets. 8192 keeps the prefill arm's span at 8x
# (1024 -> 8192, well past PREFILL_ARM_MIN_SPAN_TOKENS) at a KV cost measured in
# hundreds of MB rather than GB. Raise it deliberately, with headroom to spare.
METAL_MAX_CONTEXT_TOKENS = 8192
_METAL_CONTEXT_ENV = "VETCH_METAL_MAX_CONTEXT"


def metal_context_ceiling() -> int:
    """Operator ceiling on requested context, from the environment or default."""
    raw = os.environ.get(_METAL_CONTEXT_ENV, "").strip()
    if not raw:
        return METAL_MAX_CONTEXT_TOKENS
    try:
        value = int(raw)
    except ValueError:
        print(f"  Ignoring invalid {_METAL_CONTEXT_ENV}={raw!r} (expected an integer).",
              file=sys.stderr)
        return METAL_MAX_CONTEXT_TOKENS
    if value <= 0:
        print(f"  Ignoring non-positive {_METAL_CONTEXT_ENV}={raw!r}.", file=sys.stderr)
        return METAL_MAX_CONTEXT_TOKENS
    return value


def _ollama_context_length(base_url: str, model: str) -> int | None:
    """Architecture context length from ``/api/show``, or None when unknown.

    This is the *model's* limit, not Ollama's effective one. Ollama serves with
    ``num_ctx`` (default 4096) regardless of what the architecture supports, and
    silently truncates a longer prompt instead of refusing it -- so this figure is
    an upper bound for sizing, and the run must still pin ``num_ctx`` itself.
    """
    body = json.dumps({"model": model}).encode()
    req = urllib_request.Request(
        f"{base_url}/api/show",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except (URLError, OSError, ValueError) as e:
        # Fail open: an unknown limit means "use our ceiling", never "abort".
        print(f"  Could not read model context length ({e}); using ceiling.",
              file=sys.stderr)
        return None
    info = data.get("model_info")
    if not isinstance(info, dict):
        return None
    # Key is architecture-scoped, e.g. "llama.context_length", "gemma3.context_length".
    for key, value in info.items():
        if str(key).endswith(".context_length"):
            try:
                length = int(value)
            except (TypeError, ValueError):
                continue
            if length > 0:
                return length
    return None


# Overshoot when correcting a requested token count. The arm tests are ">=",
# and the tokenizer varies run to run, so aiming exactly at the threshold fails
# it about as often as it clears it.
TOKEN_TARGET_HEADROOM = 1.06


class PrefillProbe(NamedTuple):
    """What one timed prefill request tells us about this machine and model."""

    tokens_per_ms: float | None
    #: realised prompt_eval_count / requested tokens. <1 means the prompt
    #: generator under-delivers against this tokenizer.
    token_ratio: float | None

    @property
    def token_scale(self) -> float:
        """Multiplier that makes a requested token count come out right.

        Carries deliberate headroom. The arm thresholds are ``>=`` tests, and
        correcting to exactly the nominal count lands within tokenizer noise of
        them -- measured on llama3.2, an exactly-corrected 1024-token point
        realises 1014 and still fails ``>= 1024``. Overshooting slightly costs
        nothing: the row records the tokens actually processed, so the design
        point is whatever came back, and the context budget has far more than
        this much slack.
        """
        if not self.token_ratio or self.token_ratio <= 0:
            return 1.0
        return (1.0 / self.token_ratio) * TOKEN_TARGET_HEADROOM


def probe_prefill(
    base_url: str, model: str, num_ctx: int | None = None
) -> PrefillProbe:
    """Measure this machine's prefill throughput in tokens/ms, or None.

    The shared grid sizes its prefill repeats from a throughput prior, and the
    bundled prior is an H100 figure (30 tok/ms). Apple Silicon is one to two
    orders of magnitude slower, so inheriting it asks for 64 repeats of a point
    that already fills the timing window alone. One timed request costs a couple
    of seconds and replaces the guess with this machine's own number.

    The same request also reports how many tokens the prompt generator actually
    produced, which the design needs: a point that asks for 1024 tokens and
    realises 803 falls under the prefill arm threshold and quietly stops being
    an arm row.

    Runs outside every timing window. Fails open: None means "use the prior".
    """
    from vetch.calibrate_v2 import PREFILL_ARM_MIN_INPUT_TOKENS

    # Load the model FIRST and time the second request. The probe is otherwise
    # the first call of the process and pays model load plus KV allocation --
    # which grows with num_ctx, so a larger context budget makes the machine
    # look slower. Measured: this reported <=0.25 tok/ms against a true 0.86,
    # which sized every prefill point at requests=1, left those windows under
    # the 2 s power-sampling floor, and had `admit_rows` drop them.
    try:
        _ollama_generate(
            base_url, model, _unique_prompt(approx_tokens=64, seed=6),
            max_tokens=1, num_ctx=num_ctx,
        )
    except RuntimeError:
        pass  # a failed warm-up is not fatal; the timed call reports for itself

    prompt = _unique_prompt(approx_tokens=PREFILL_ARM_MIN_INPUT_TOKENS, seed=7)
    start = time.monotonic() * 1000
    try:
        measured, _ = _ollama_generate(
            base_url, model, prompt, max_tokens=8, num_ctx=num_ctx,
        )
    except RuntimeError as e:
        print(f"  Prefill rate probe failed ({e}); using the bundled prior.",
              file=sys.stderr)
        return PrefillProbe(None, None)
    elapsed_ms = (time.monotonic() * 1000) - start
    if elapsed_ms <= 0 or measured <= 0:
        return PrefillProbe(None, None)
    return PrefillProbe(
        tokens_per_ms=measured / elapsed_ms,
        token_ratio=measured / float(PREFILL_ARM_MIN_INPUT_TOKENS),
    )


def resolve_metal_context_limit(base_url: str, model: str) -> int:
    """Context budget for this run: the model's own limit, under our ceiling."""
    ceiling = metal_context_ceiling()
    architecture = _ollama_context_length(base_url, model)
    if architecture is None:
        return ceiling
    return min(architecture, ceiling)


def _grid_design(
    context_limit: int | None = None,
    quick: bool = False,
    prefill_tokens_per_ms: float | None = None,
) -> list[WorkloadSpec]:
    """Return the shared protocol-v2 workload design for Apple Silicon.

    Metal uses powermetrics sampling, so the shared v2 design's prefill arm is
    especially important: its ``requests`` value aggregates short calls into a
    measurable timing window while preserving the per-request design point.

    ``context_limit`` drops prefill points the server cannot actually process,
    exactly as the CUDA path does with the vLLM server's ``max_model_len``. The
    the servers handle over-length requests differently: vLLM rejects them,
    while Ollama truncates to ``num_ctx`` and answers normally. Without a limit
    here the four longest prefill points all
    collapse toward the same realised length and the arm loses the span it
    exists to create -- after paying the runtime to send them.
    """
    from vetch.calibrate_v2 import v2_grid_design

    return [
        WorkloadSpec(
            n_images=int(spec.get("n_images", 0)),
            approx_text_tokens=int(spec["approx_text_tokens"]),
            max_tokens=int(spec["max_tokens"]),
            replicate=bool(spec.get("replicate", False)),
            requests=max(1, int(spec.get("requests", 1))),
        )
        for spec in v2_grid_design(
            vision=True, quick=quick, context_limit=context_limit,
            prefill_tokens_per_ms=prefill_tokens_per_ms,
        )
    ]


def _legacy_grid_design() -> list[WorkloadSpec]:
    """The pre-v2 Metal/CUDA compatibility grid.

    Kept only for explicit ``grid='v1'`` CUDA callers. New Apple Silicon
    calibration always uses :func:`_grid_design`, which is the shared v2 grid.
    """
    grid: list[WorkloadSpec] = []
    for approx_in in [20, 128, 512]:
        for max_out in [5, 64, 256]:
            grid.append(WorkloadSpec(0, approx_in, max_out))
    for approx_in in [20, 128, 512]:
        for max_out in [5, 64, 256]:
            grid.append(WorkloadSpec(1, approx_in, max_out))
    grid.extend([
        WorkloadSpec(0, 128, 32, replicate=True),
        WorkloadSpec(0, 128, 32, replicate=True),
        WorkloadSpec(1, 128, 32, replicate=True),
        WorkloadSpec(1, 128, 32, replicate=True),
    ])
    return grid


def legacy_grid_design_id() -> str:
    """Stable id for the retained pre-v2 compatibility grid."""
    import hashlib

    specs = sorted(
        (s.n_images, s.approx_text_tokens, s.max_tokens, s.replicate)
        for s in _legacy_grid_design()
    )
    digest = hashlib.sha256(json.dumps(specs).encode()).hexdigest()[:8]
    return f"sym-{digest}"


def _build_legacy_grid(seed: int | None = None) -> list[WorkloadSpec]:
    """Build the retained pre-v2 compatibility grid."""
    grid = _legacy_grid_design()
    (random.Random(seed) if seed is not None else random).shuffle(grid)
    return grid


def grid_design_id() -> str:
    """Stable id for the grid DESIGN (its multiset of workload points).

    Two calibrations sharing this id used the same workload shape and are
    comparable; it changes automatically if the design points change.
    """
    # Keep the same identifier as calibrate-cuda so records disclose a shared
    # experiment rather than merely a shared estimator.
    return "v2-unbalanced"


def _build_grid(
    seed: int | None = None,
    context_limit: int | None = None,
    quick: bool = False,
    prefill_tokens_per_ms: float | None = None,
) -> list[WorkloadSpec]:
    """Return the grid, shuffled. A ``seed`` makes the run order reproducible
    (thermal ordering is then comparable across runs); None keeps the legacy
    non-deterministic shuffle for the Apple Silicon path.
    """
    grid = _grid_design(
        context_limit=context_limit, quick=quick,
        prefill_tokens_per_ms=prefill_tokens_per_ms,
    )
    (random.Random(seed) if seed is not None else random).shuffle(grid)
    return grid


# ---------------------------------------------------------------------------
# Active-calibration gating
#
# The fit itself lives in vetch.calibrate_v2 and is shared with the CUDA path.
# Identifiability is a property of the experimental design, not of the silicon,
# so a second implementation here would only let the two drift apart.
# ---------------------------------------------------------------------------

def _require_numpy_for_calibration() -> None:
    """Apple Silicon calibration needs NumPy for conditioning and bootstrap CIs."""
    from vetch.calibrate_v2 import require_numpy

    require_numpy("apple-silicon")


def _active_calibration_rejection_reasons(
    fit: NestedFitResult,
    power_state: dict[str, Any],
    gpu_power_captured: bool,
    idle_drift_pct: float,
    samples: int,
    idle_share_pct: float | None = None,
) -> list[str]:
    """Reasons a detail run should not become the active local calibration."""
    reasons = list(fit.invalid_reasons)
    if not fit.valid and not reasons:
        reasons.append("valid=false in fit")
    # inf means pure-Python fallback (no NumPy): _fit already skips the cond gate
    if math.isfinite(fit.condition_number) and fit.condition_number > 30:
        reasons.append(
            f"condition_number={fit.condition_number:.1f} > 30 "
            "(workload shapes not distinct enough)"
        )
    if samples < 8:
        reasons.append(f"samples={samples} < 8 minimum")
    if power_state.get("on_ac_power") is not True:
        reasons.append("not on AC power")
    if power_state.get("low_power_mode"):
        reasons.append("low_power_mode=true during calibration")
    if not gpu_power_captured:
        reasons.append("gpu_power_captured=false — GPU energy not measured")
    # Gate on the contribution to measured energy. The raw percentage rejected
    # three of four runs whose coefficients agree with each other to 0.5%.
    if idle_share_pct is not None:
        if idle_share_pct > MAX_IDLE_UNCERTAINTY_ROW_SHARE_PCT:
            reasons.append(
                f"idle drift contributes {idle_share_pct:.1f}% of a typical "
                f"row's energy > {MAX_IDLE_UNCERTAINTY_ROW_SHARE_PCT:.0f}% "
                f"(idle_drift_pct={idle_drift_pct:.1f})"
            )
    elif idle_drift_pct > IDLE_DRIFT_FLAG_PCT:
        reasons.append(
            f"idle_drift_pct={idle_drift_pct:.1f} > {IDLE_DRIFT_FLAG_PCT:.0f}% "
            "(no run table to weigh it against)"
        )

    deduped: list[str] = []
    for reason in reasons:
        if reason not in deduped:
            deduped.append(reason)
    return deduped


# ---------------------------------------------------------------------------
# Main calibration function
# ---------------------------------------------------------------------------

def calibrate_apple_silicon(
    model: str,
    provider: str = "self-hosted",
    base_url: str = OLLAMA_DEFAULT_BASE_URL,
    iterations: int = 1,
    verbose: bool = False,
    precision: str = "apple-native",
    serving_engine: str = "ollama",
    quick: bool = False,
    allow_degraded_power: bool = False,
    sampler: str = "auto",
) -> tuple[CalibrationResult, Path]:
    """Measure Apple Silicon energy coefficients for an Ollama model.

    Args:
        model: Ollama model name (e.g. "moondream:latest")
        provider: Provider label for the calibration identity (default
            ``"self-hosted"`` — match production ``provider_hint``).
        base_url: Ollama API base URL
        iterations: Grid iteration multiplier (1 = one pass of ~28 runs)
        verbose: Print per-run details
        precision: Identity dimension (default ``"apple-native"``; use a GGUF
            tag like ``gguf:q4_k_m`` when calibrating a quantized Ollama build).
        serving_engine: Serving stack label (default ``"ollama"``).
        quick: Use the reduced v2 grid. For smoke-testing the run end to end on
            new hardware, not for producing a coefficient: fewer design points
            means wider intervals, and axes will correctly come back
            ``not_identified``.

    Returns:
        ``(CalibrationResult, record_path)`` -- coefficients plus the schema-2
        calibration record path.
    """
    from vetch.calibrate import CalibrationResult
    from vetch.calibration_store import (
        APPLE_SOC_EXCLUDES,
        APPLE_SOC_INCLUDES,
        CalibrationIdentity,
        canonical_gpu,
        commit_calibration,
        is_cloud_provider,
        measurement_provenance_core,
    )

    # --- Preflight -----------------------------------------------------------
    assert_apple_silicon()
    _require_numpy_for_calibration()
    sampler = resolve_sampler(sampler)
    if sampler == SAMPLER_POWERMETRICS:
        assert_root()
        warn_if_home_is_root()
    energy_source = ENERGY_SOURCE_BY_SAMPLER[sampler]
    if is_cloud_provider(provider):
        raise ValueError(
            f"provider={provider!r} is a cloud/API vendor and is refused for Tier-0 "
            "calibration: it is ambiguous (real hosted API vs OpenAI-compatible "
            "local) and would attach local coefficients to cloud events. Use "
            "provider='self-hosted' or 'ollama'."
        )

    hw = get_hardware_info()
    power_state = check_power_state()
    assert_power_state_usable(power_state, allow_degraded=allow_degraded_power)
    ollama_version = _check_ollama(base_url)

    print("Vetch Apple Silicon Calibration")
    print(f"  Model:    {model}")
    print(f"  Chip:     {hw.chip_raw}")
    print(f"  Memory:   {hw.memory_gb} GB")
    print(f"  macOS:    {hw.macos_version}")
    print(f"  Ollama:   {ollama_version}")
    print(
        "  Power:    powermetrics estimated SoC draw (CPU+GPU+ANE), not wall-plug power."
    )
    print()

    visual_tokens, visual_tokens_source = _get_visual_tokens_per_image(model)

    # --- Load image pool BEFORE starting powermetrics ------------------------
    # Used for probe and warm-up; not used for grid runs (those use unique images).
    image_pool, active_image_set = _load_image_set()
    print(f"  Images:   {active_image_set} ({len(image_pool)} available for probe/warmup)")
    print()

    # --- Build grid and pre-generate unique images BEFORE the monitor --------
    # Unique-seeded images per run defeat Ollama's image embedding KV cache.
    # If the same image is reused, Ollama skips the vision encoder on subsequent
    # runs and returns nearly zero energy, biasing wh_per_image downward.
    # Image generation happens here (outside all timing windows) so it never
    # contaminates measurements.
    # Ask what this model can actually process, then hold it under the Metal
    # memory ceiling. Design points above the budget are dropped rather than
    # sent and silently truncated -- Ollama answers a too-long prompt normally,
    # so an unfiltered grid loses the prefill span without reporting anything.
    from vetch.calibrate_v2 import PREFILL_ARM_MIN_INPUT_TOKENS

    context_limit = resolve_metal_context_limit(base_url, model)
    # ONE num_ctx for probe, warm-up and every measured request. Changing it
    # makes Ollama evict and reload the model -- several GB of unified-memory
    # churn -- so pin the budget itself rather than a per-grid minimum, and pay
    # the single load before any timing window opens.
    grid_num_ctx = context_limit
    print(f"  Context:  budget {context_limit} tokens "
          f"(ceiling {metal_context_ceiling()}); num_ctx pinned at {grid_num_ctx}")
    probe = probe_prefill(base_url, model, num_ctx=grid_num_ctx)
    token_scale = probe.token_scale
    if probe.tokens_per_ms:
        print(f"  Prefill:  measured {probe.tokens_per_ms:.2f} tokens/ms; "
              f"tokenizer realises {probe.token_ratio:.2f} of requested tokens "
              f"(prompts scaled x{token_scale:.2f})")
    grid = _build_grid(
        context_limit=context_limit, quick=quick,
        prefill_tokens_per_ms=probe.tokens_per_ms,
    ) * max(1, iterations)
    total_runs = len(grid)
    total_requests = sum(spec.requests for spec in grid)
    longest = max((s.approx_text_tokens for s in grid), default=0)
    if longest < PREFILL_ARM_MIN_INPUT_TOKENS:
        print(
            f"  WARNING: longest prefill point is {longest} tokens, below the "
            f"{PREFILL_ARM_MIN_INPUT_TOKENS}-token arm threshold; "
            "wh_per_1k_input will not be identifiable from this run.",
            file=sys.stderr,
        )

    print(f"Pre-generating {sum(s.n_images * s.requests for s in grid)} unique calibration "
          f"image(s) for {total_runs} timing windows / {total_requests} requests "
          "(busts Ollama KV cache)...")
    run_images: list[list[list[str]]] = []
    for i, spec in enumerate(grid):
        request_images: list[list[str]] = []
        for request_index in range(spec.requests):
            run_seed = i * 1000 + request_index + 100
            request_images.append([
                _unique_image_b64(seed=run_seed + j * 1000)
                for j in range(spec.n_images)
            ])
        run_images.append(request_images)
    print()

    # --- Probe image-token accounting (outside monitor, uses pool image) -----
    # Must run before the monitor so probe API calls don't contaminate idle baseline.
    # If the model doesn't accept images (text-only), the probe fails gracefully and
    # the calibration degrades to a text-only 3-parameter fit (β0, β_in, β_out).
    model_is_vlm = True
    token_probe: dict[str, Any] = {}
    fit_visual_tokens = 0

    print("Probing image-token accounting...")
    try:
        token_probe = _probe_image_token_accounting(base_url, model, image_pool[0])
        print(f"  Image token accounting: {token_probe['image_token_accounting']}")
        delta = token_probe["delta_prompt_eval_count_with_image"]
        print(f"  Delta prompt_eval_count with image: {delta}")
        # Determine how many tokens to subtract per image in the design matrix.
        if token_probe["image_token_accounting"] == "includes":
            fit_visual_tokens = max(0, delta)
    except RuntimeError as probe_err:
        model_is_vlm = False
        token_probe = {
            "image_token_accounting": "unsupported",  # nosec B105 — LLM tokens, not auth
            "image_probe_error": str(probe_err),
        }
        print(
            f"  Model does not appear to support images ({probe_err})\n"
            "  Running text-only calibration (wh_per_image will not be measured).",
            file=sys.stderr,
        )
    print()

    # Filter grid to text-only runs when the model doesn't support images
    if not model_is_vlm:
        grid = [spec for spec in grid if spec.n_images == 0]
        run_images = [[[] for _ in range(spec.requests)] for spec in grid]
        total_runs = len(grid)
        total_requests = sum(spec.requests for spec in grid)

    # --- powermetrics monitor ------------------------------------------------
    # Declare time-window variables before the with block so they are accessible
    # for deferred energy computation after the monitor exits.
    idle_start = idle_end = 0.0
    idle_start2 = idle_end2 = 0.0

    monitor: AppleSiliconMonitor | ioreport.IOReportEnergyMonitor
    if sampler == SAMPLER_IOREPORT:
        print("Starting IOReport energy counters...")
        monitor = ioreport.IOReportEnergyMonitor(sample_interval_ms=100)
    else:
        print("Starting powermetrics... (this requires root)")
        monitor = AppleSiliconMonitor(sample_interval_ms=100)
    with monitor:
        # Wait briefly for monitor to settle and get first samples
        time.sleep(1.0)

        # Warm-up BEFORE idle baseline: loads model/vision encoder and brings system to
        # operating temperature. Idle power measured after warm-up represents the
        # thermal state during calibration runs, not the colder pre-load idle.
        warmup_mode = "image call to load vision encoder" if model_is_vlm else "text-only call"
        print(f"Warming up ({warmup_mode})...")
        warmup_prompt = _unique_prompt(approx_tokens=20, seed=0)
        warmup_img = _warmup_image_b64(model_is_vlm, image_pool)
        # Same num_ctx as the grid, so the warm-up loads the model at the size
        # the measured runs will use and no run pays a reload.
        _ollama_generate(base_url, model, warmup_prompt, warmup_img, max_tokens=10,
                         num_ctx=grid_num_ctx)
        # Idle baseline (after warm-up) — record window; energy computed post-exit
        idle_start, idle_end = measure_idle_window("post-warmup")

        # Run the workload grid — record windows only, no energy yet.
        # The batch-based monitor returns samples only when each subprocess.run()
        # completes (~4 s per batch), so integrate() would return 0.0 if called
        # inline. All energies are computed after __exit__ joins the thread.
        print(f"Running calibration grid ({total_runs} timing windows / "
              f"{sum(spec.requests for spec in grid)} requests)...")
        print()

        raw_records: list[dict[str, Any]] = []
        grid_failures = 0
        for i, (spec, request_images) in enumerate(zip(grid, run_images)):
            text_tokens_total = 0
            output_tokens_total = 0
            t_start = time.monotonic() * 1000
            failed = False
            for request_index, images in enumerate(request_images):
                prompt = _unique_prompt(
                    approx_tokens=spec.approx_text_tokens,
                    seed=i * 1000 + request_index + 100,
                    token_scale=token_scale,
                )
                try:
                    text_tokens, output_tokens = _ollama_generate(
                        base_url, model, prompt,
                        image_b64=images[0] if images else None,
                        max_tokens=spec.max_tokens,
                        num_ctx=grid_num_ctx,
                    )
                except RuntimeError as e:
                    print(
                        f"  Run {i+1}/{total_runs} request {request_index + 1}/"
                        f"{spec.requests} FAILED: {e}",
                        file=sys.stderr,
                    )
                    failed = True
                    break
                text_tokens_total += text_tokens
                output_tokens_total += output_tokens
            t_end = time.monotonic() * 1000

            if failed or text_tokens_total == 0:
                grid_failures += 1
                continue

            raw_records.append({
                "n_images": spec.n_images,
                "text_tokens": text_tokens_total,
                "output_tokens": output_tokens_total,
                "requests": spec.requests,
                "t_start": t_start,
                "t_end": t_end,
                "replicate": spec.replicate,
            })

            if verbose:
                print(
                    f"  [{i+1:2d}/{total_runs}] n_img={spec.n_images} "
                    f"in/request={text_tokens_total / spec.requests:4.0f} "
                    f"out/request={output_tokens_total / spec.requests:3.0f} "
                    f"({(t_end-t_start)/1000:.1f}s)  [energy pending]"
                )
            else:
                pct = (i + 1) * 100 // total_runs
                print(f"\r  Progress: {i+1}/{total_runs} ({pct}%)", end="", flush=True)

            time.sleep(2.0)

        print()  # newline after progress

        if grid_failures > total_runs // 2:
            raise RuntimeError(
                f"Too many failed grid runs ({grid_failures}/{total_runs}). "
                "Check Ollama health, model pull, and network before retrying."
            )
        if grid_failures > total_runs // 4:
            print(
                f"WARNING: {grid_failures}/{total_runs} grid runs failed — "
                "fit may be biased.",
                file=sys.stderr,
            )

        # Idle baseline (after the grid), through the same helper so it cannot
        # diverge from the "before" window.
        idle_start2, idle_end2 = measure_idle_window("post-run")

    # __exit__ joined the background thread — all batches have now completed
    # and monitor._samples contains the full session's power data.

    # --- Sanity check: did we get any power samples at all? ------------------
    if len(monitor._samples) == 0 and sampler == SAMPLER_IOREPORT:
        raise RuntimeError("IOReport produced no energy samples.")
    if len(monitor._samples) == 0:
        raise RuntimeError(
            "powermetrics produced no power samples. The 'Combined Power' line was\n"
            "not found in powermetrics output. This may mean:\n"
            "  - powermetrics is not outputting cpu_power/gpu_power samplers\n"
            "  - The output format changed in this macOS version\n"
            "Run 'sudo powermetrics --samplers cpu_power,gpu_power -n 1' and check output."
        )

    if not monitor.info.gpu_power_captured:
        print(
            "WARNING: GPU power readings not found in powermetrics output.\n"
            "         Energy estimates may be significantly understated.\n"
            "         Verify powermetrics is capturing GPU subsystem.",
            file=sys.stderr,
        )

    # --- Compute idle baselines and per-run energies (deferred) --------------
    # All batches have completed now that the monitor thread has been joined.
    idle_watts_before = monitor.mean_watts(idle_start, idle_end)
    idle_watts_after = monitor.mean_watts(idle_start2, idle_end2)
    print(f"  Idle power before: {idle_watts_before:.2f} W  after: {idle_watts_after:.2f} W")

    run_records: list[dict[str, Any]] = []
    if verbose:
        print("\nPer-run energies (computed from full sample set):")
    for raw in raw_records:
        energy_wh = monitor.integrate(raw["t_start"], raw["t_end"])
        duration_ms = raw["t_end"] - raw["t_start"]
        duration_h = duration_ms / 3_600_000.0
        avg_idle_watts = (idle_watts_before + idle_watts_after) / 2.0
        net_energy_wh = max(0.0, energy_wh - avg_idle_watts * duration_h)
        record: dict[str, Any] = {
            "n_images": raw["n_images"],
            "text_tokens": raw["text_tokens"],
            "output_tokens": raw["output_tokens"],
            "requests": raw["requests"],
            "energy_wh": net_energy_wh,
            "raw_energy_wh": energy_wh,  # pre-clamping, used for gap detection
            "duration_ms": duration_ms,
            "replicate": raw["replicate"],
        }
        if isinstance(monitor, ioreport.IOReportEnergyMonitor):
            # Gross per-domain energy for the same window. The fit uses the
            # combined `energy_wh`; the split is what makes an `apple_soc`
            # figure comparable with a `gpu_board` one after the fact.
            record["raw_energy_by_domain_wh"] = monitor.integrate_by_domain(
                raw["t_start"], raw["t_end"]
            )
        run_records.append(record)
        if verbose:
            ni = raw["n_images"]
            ti = raw["text_tokens"]
            ot = raw["output_tokens"]
            print(
                f"  n_img={ni} in={ti:4d} out={ot:3d} "
                f"E={net_energy_wh*1000:.4f} mWh  ({duration_ms:.0f}ms)"
            )

    # --- Gap detection -------------------------------------------------------
    # Runs that fall entirely within a powermetrics batch startup gap produce
    # exactly 0.0 from monitor.integrate() (no samples in window). These
    # contaminate the regression. Use raw_energy_wh (pre-idle-subtraction) for
    # the check — net_energy_wh can be 0.0 for legitimate runs when idle power
    # is close to the run's total power (clamped by max(0, ...)).
    gap_fallen = [r for r in run_records if r["raw_energy_wh"] == 0.0 and r["duration_ms"] > 500]
    if gap_fallen:
        print(
            f"WARNING: {len(gap_fallen)} run(s) returned 0 energy despite "
            f"taking >{min(r['duration_ms'] for r in gap_fallen):.0f}ms — "
            "likely fell in a powermetrics batch gap. Excluding from fit.",
            file=sys.stderr,
        )
        run_records = [r for r in run_records if r not in gap_fallen]

    # --- Drift check ---------------------------------------------------------
    # Flag, never abort. Protocol v2 is explicit that idle drift is a
    # measurement-validity signal rather than an identifiability one, and that
    # "a drifting record stays usable and stays visibly weak" -- the same
    # precedent as residuals_structured. Raising here destroyed the entire run
    # AFTER every measurement had been taken, discarding the GPU time and
    # writing nothing, while `_active_calibration_rejection_reasons` was already
    # set up to record the drift and withhold the active install. That graceful
    # path was unreachable for this one condition.
    idle_drift_pct, idle_drift_level = classify_idle_drift(
        idle_watts_before, idle_watts_after
    )
    if idle_drift_level == "flag":
        print(
            f"WARNING: Idle power drifted {idle_drift_pct:.1f}% between start and end.\n"
            f"         Start: {idle_watts_before:.2f} W  End: {idle_watts_after:.2f} W\n"
            f"         Thermal state changed; the idle subtraction every\n"
            f"         coefficient rests on is unreliable. The record will be\n"
            f"         written and marked, but not installed as the active\n"
            f"         calibration. Let the machine cool, check Low Power Mode\n"
            f"         is off and that you are on AC power, then re-run.",
            file=sys.stderr,
        )
    elif idle_drift_level == "warn":
        print(
            f"NOTE: idle power drifted {idle_drift_pct:.1f}% "
            f"({idle_watts_before:.2f} W -> {idle_watts_after:.2f} W).",
            file=sys.stderr,
        )

    # --- Fit -----------------------------------------------------------------
    if len(run_records) < 8:
        raise RuntimeError(
            f"Too few successful runs ({len(run_records)}) to fit the model. "
            "Check Ollama is running and the model is pulled."
        )

    print(f"Fitting energy model ({len(run_records)} runs)...")
    # One fitting path for every accelerator. Identifiability is a property of
    # the design, not the silicon. The shared implementation keeps coefficient
    # semantics consistent across accelerators and preserves uncertain axes as
    # unidentified rather than flooring them to zero.
    from vetch.calibrate_v2 import fit_nested_models, idle_relative_uncertainty

    idle_watts = (idle_watts_before + idle_watts_after) / 2.0
    fit = fit_nested_models(
        run_records,
        idle_watts=idle_watts if idle_watts > 0 else None,
        energy_source=energy_source,
        visual_tokens_per_image=fit_visual_tokens,
        idle_relative=idle_relative_uncertainty({
            "idle_watts_before": idle_watts_before,
            "idle_watts_after": idle_watts_after,
        }),
    )
    # How much the idle-window disagreement contributes to a typical row's
    # energy. This, not the raw drift percentage, is what decides whether an
    # unstable baseline threatens the coefficients.
    idle_share_pct = idle_uncertainty_row_share_pct(
        run_records, idle_watts_before, idle_watts_after
    )
    if idle_share_pct is not None:
        print(f"  Idle drift contributes {idle_share_pct:.1f}% of a typical "
              f"row's energy (limit {MAX_IDLE_UNCERTAINTY_ROW_SHARE_PCT:.0f}%)")
    rejection_reasons = _active_calibration_rejection_reasons(
        fit=fit,
        power_state=power_state,
        gpu_power_captured=monitor.info.gpu_power_captured,
        idle_drift_pct=idle_drift_pct,
        samples=len(run_records),
        idle_share_pct=idle_share_pct,
    )
    calibration_valid = not rejection_reasons

    # --- Validation output ---------------------------------------------------
    if not calibration_valid:
        print("\nWARNING: Calibration quality issues detected:")
        for reason in rejection_reasons:
            print(f"  - {reason}")
        print("  Detail JSON will be saved, but active calibration will not be installed.")

    # --- Build CalibrationResult ---------------------------------------------
    wh_per_image = fit.wh_per_image if model_is_vlm else None
    visual_tokens_for_result = visual_tokens if model_is_vlm else None
    visual_tokens_source_for_result = visual_tokens_source if model_is_vlm else "not_applicable"

    result = CalibrationResult(
        model=model,
        provider=provider,
        wh_per_1k_input=fit.wh_per_1k_input,
        wh_per_1k_output=fit.wh_per_1k_output,
        tier=0,
        samples=len(run_records),
        gpu_name=hw.chip_raw,
        wh_per_image=wh_per_image,
        visual_tokens_per_image=visual_tokens_for_result,
        intercept_wh=fit.intercept_wh,
        active=calibration_valid,
        rejection_reasons=rejection_reasons if rejection_reasons else None,
        serving_engine=serving_engine,
        backend=serving_engine,
        precision=precision,
    )

    # --- Data-rich v1 record (same store as CUDA; supersedes legacy flat files) ---
    try:
        import numpy  # noqa: F401
        fit_engine, ci_method = "numpy", "bootstrap_500"
    except ImportError:
        fit_engine, ci_method = "pure_python", "heuristic_pm20"

    from vetch import __version__ as _vetch_version
    from vetch.calculation import METHODOLOGY_VERSION

    gpu_canonical, gpu_known = canonical_gpu(hw.chip_raw)
    identity = CalibrationIdentity(
        provider=provider,
        model=model,
        gpu=gpu_canonical,
        serving_engine=serving_engine,
        precision=precision,
    )
    provenance = measurement_provenance_core(
        samples=len(run_records),
        energy_source=energy_source,
        measurement_basis=MEASUREMENT_BASIS_BY_SAMPLER[sampler],
        energy_domain="apple_soc",
        energy_domain_includes=APPLE_SOC_INCLUDES,
        energy_domain_excludes=APPLE_SOC_EXCLUDES,
        idle_watts_before=idle_watts_before,
        idle_watts_after=idle_watts_after,
        idle_drift_pct=idle_drift_pct,
        fit=fit,
        fit_engine=fit_engine,
        ci_method=ci_method,
        run_records=run_records,
        gpu_name=hw.chip_raw,
        gpu_canonical=gpu_canonical,
        gpu_known=gpu_known,
        serving_engine=serving_engine,
        server_version=ollama_version,
        image_set=CALIBRATION_IMAGE_SET_SYNTHETIC,
        image_resolution_px=CALIBRATION_IMAGE_SIZE_PX,
        model_supports_images=model_is_vlm,
        visual_tokens_assumed=visual_tokens_for_result,
        visual_tokens_assumed_source=visual_tokens_source_for_result,
        vetch_version=_vetch_version,
        methodology_version=METHODOLOGY_VERSION,
        extra={
            "sampler": sampler,
            "sampler_cadence_ms": monitor.info.sample_interval_ms,
            "integration_method": INTEGRATION_METHOD_BY_SAMPLER[sampler],
            "energy_channels": (
                monitor.info.channels
                if isinstance(monitor, ioreport.IOReportEnergyMonitor) else None
            ),
            "idle_uncertainty_row_share_pct": idle_share_pct,
            "chip_family": hw.chip_family,
            "chip_tier": hw.chip_tier,
            "memory_gb": hw.memory_gb,
            "macos_version": hw.macos_version,
            "api_backend": "ollama",
            "gpu_power_captured": monitor.info.gpu_power_captured,
            "power_state": power_state,
            # Sampler settings that shape the measured workload. Ollama honours
            # neither ignore_eos nor a minimum length, so output length is held
            # by the prompt shape plus a neutralised repetition penalty. Both
            # change WHAT was measured, so both belong in the record.
            "sampling": {
                "repeat_penalty": 1.0,
                "repeat_penalty_reason": (
                    "the default penalty drives a long enumeration to EOS; "
                    "measured 416 of 2048 requested tokens with it, 2048 without"
                ),
                "output_length_held_by": "prompt_shape",
                "ignore_eos_supported": False,
                "num_ctx": grid_num_ctx,
                "prompt_token_scale": token_scale,
            },
        },
    )
    # Carry the nested fit's verdict and energy basis onto the record, as the
    # CUDA writer does.
    record_path = commit_calibration(
        result, identity, provenance, **fit_record_blocks(fit)
    )
    print(f"\nCalibration record written to {record_path}")
    if not calibration_valid:
        print("  (active=false: recorded for audit, will not auto-load)")

    return result, record_path


def format_calibration_result_apple(result: CalibrationResult, detail_path: Path) -> str:
    status = (
        "  Status:             ACTIVE (saved to ~/.vetch/calibrations/)"
        if result.active
        else "  Status:             NOT ACTIVE (quality gates failed — recorded for audit only)"
    )
    wh_in = "null" if result.wh_per_1k_input is None else f"{result.wh_per_1k_input:.6f}"
    wh_out = "null" if result.wh_per_1k_output is None else f"{result.wh_per_1k_output:.6f}"
    lines = [
        "",
        "━" * 50,
        "  Vetch Apple Silicon Calibration Complete",
        "━" * 50,
        status,
        f"  Model:              {result.model}",
        f"  Hardware:           {result.gpu_name}",
        "  Tier:               0 (hardware-measured)",
        f"  Samples:            {result.samples}",
        "",
        f"  wh_per_1k_input:    {wh_in}  Wh / 1k text tokens",
        f"  wh_per_1k_output:   {wh_out}  Wh / 1k output tokens",
    ]
    if result.wh_per_image is not None:
        lines.append(f"  wh_per_image:       {result.wh_per_image:.6f}  Wh / image")
    lines += [
        "",
        "  Note: powermetrics measures estimated SoC power (CPU + GPU + ANE),",
        "  not wall-plug power. Coefficients are local-hardware specific.",
        "  intercept_wh is fixed energy per inference request (from the LS fit),",
        "  not amortized across tokens — expect higher Wh on tiny prompts.",
        "",
        "  Saved to: ~/.vetch/calibrations/  (schema-2 measurement-keyed record)",
        f"  Record:   {detail_path}",
        "",
        "  Measurement images: unique seeded synthetic per run (KV-cache busting)",
        "  VLM image basis: 378px single-tile image; reported image tokens scale",
        "  high-resolution image energy when providers expose them.",
        "  To share with the Vetch community:",
        f"    cat {detail_path}",
        "  Then open a GitHub issue at https://github.com/prismatic-labs/vetch",
        f"  with the title: [calib] {result.gpu_name} / {result.model}",
        "  Only submissions using the default image set are included in",
        "  aggregated community datasets.",
        "━" * 50,
        "",
    ]
    return "\n".join(lines)
