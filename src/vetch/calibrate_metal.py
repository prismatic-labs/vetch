"""Apple Silicon energy calibration using powermetrics.

Measures actual SoC power draw (CPU + GPU + ANE) during Ollama inference
via Apple's powermetrics tool. Produces Tier 0 hardware-measured coefficients.

IMPORTANT: powermetrics requires root. Run with sudo.

Supports VLMs: in addition to wh_per_1k_input and wh_per_1k_output, it
measures wh_per_image — the energy cost of vision-encoder processing per image.

Usage (CLI, preferred):
    sudo vetch calibrate-apple-silicon --model moondream --precision apple-native

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib import request as urllib_request
from urllib.error import URLError

if TYPE_CHECKING:
    from vetch.calibrate import CalibrationResult

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
            "Re-run with sudo:\n"
            "  sudo vetch calibrate-apple-silicon --model moondream --provider ollama",
            file=sys.stderr,
        )
        sys.exit(1)


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

def check_power_state() -> dict[str, Any]:
    result: dict[str, Any] = {
        "on_ac_power": None,
        "low_power_mode": False,
        "high_power_mode": False,
    }
    try:
        batt = subprocess.check_output(
            ["pmset", "-g", "ps"], text=True, stderr=subprocess.DEVNULL
        )
        result["on_ac_power"] = "AC Power" in batt
        if not result["on_ac_power"]:
            print(
                "WARNING: Running on battery. Energy measurements will be less accurate\n"
                "and may vary. Plug in AC power for best results.",
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
) -> tuple[int, int]:
    """Run one Ollama generate call. Returns (prompt_eval_count, eval_count)."""
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens},
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

def _unique_prompt(approx_tokens: int, seed: int) -> str:
    """Generate a unique prompt of approximately the right token count."""
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
    target_words = max(4, int(approx_tokens * 0.75))
    words = [rng.choice(filler_words) for _ in range(target_words - 4)]
    prompt = f"{nonce} Describe in detail: {' '.join(words)}. Respond briefly."
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


def _grid_design() -> list[WorkloadSpec]:
    """The deterministic (unshuffled) grid design.

    Design principles:
    - n_images=2 excluded: two sequential API calls add a second per-request
      overhead that leaks into wh_per_image, inflating it.
    - Text-only and single-image runs share the same (text, output) design
      points, making the n_images dimension orthogonal to in_k and out_k.
      This minimises the condition number after feature standardisation.
    - Replicates at a mid-range point give a noise estimate.
    """
    grid: list[WorkloadSpec] = []

    # Text-only (n_images=0) — 3×3 grid anchors β_in and β_out
    for approx_in in [20, 128, 512]:
        for max_out in [5, 64, 256]:
            grid.append(WorkloadSpec(n_images=0, approx_text_tokens=approx_in, max_tokens=max_out))

    # Single image — identical (text, output) grid, orthogonalises β_img
    for approx_in in [20, 128, 512]:
        for max_out in [5, 64, 256]:
            grid.append(WorkloadSpec(n_images=1, approx_text_tokens=approx_in, max_tokens=max_out))

    # Anchor replicates (×2 extra) for noise estimation
    grid.extend([
        WorkloadSpec(n_images=0, approx_text_tokens=128, max_tokens=32, replicate=True),
        WorkloadSpec(n_images=0, approx_text_tokens=128, max_tokens=32, replicate=True),
        WorkloadSpec(n_images=1, approx_text_tokens=128, max_tokens=32, replicate=True),
        WorkloadSpec(n_images=1, approx_text_tokens=128, max_tokens=32, replicate=True),
    ])
    return grid


def grid_design_id() -> str:
    """Stable id for the grid DESIGN (its multiset of workload points).

    Two calibrations sharing this id used the same workload shape and are
    comparable; it changes automatically if the design points change.
    """
    import hashlib

    specs = sorted(
        (s.n_images, s.approx_text_tokens, s.max_tokens, s.replicate)
        for s in _grid_design()
    )
    digest = hashlib.sha256(json.dumps(specs).encode()).hexdigest()[:8]
    return f"sym-{digest}"


def _build_grid(seed: int | None = None) -> list[WorkloadSpec]:
    """Return the grid, shuffled. A ``seed`` makes the run order reproducible
    (thermal ordering is then comparable across runs); None keeps the legacy
    non-deterministic shuffle for the Apple Silicon path.
    """
    grid = _grid_design()
    (random.Random(seed) if seed is not None else random).shuffle(grid)
    return grid


# ---------------------------------------------------------------------------
# Least-squares fit
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    intercept_wh: float
    wh_per_image: float
    wh_per_1k_input: float
    wh_per_1k_output: float
    r2: float
    condition_number: float
    input_ci95: tuple[float, float]
    output_ci95: tuple[float, float]
    image_ci95: tuple[float, float]
    valid: bool
    invalid_reasons: list[str]
    residuals_structured: bool = False


def _detect_residuals_structured(
    runs: list[dict[str, Any]],
    residuals: list[float],
) -> bool:
    """True when residual means differ in sign across workload groups."""
    if len(runs) < 8 or len(residuals) != len(runs):
        return False

    def _group_key(run: dict[str, Any], factor: str) -> Any:
        if factor == "n_images":
            return run["n_images"]
        return 0 if run["output_tokens"] < 50 else 1

    for factor in ("n_images", "output_bucket"):
        groups: dict[Any, list[float]] = {}
        for run, residual in zip(runs, residuals):
            groups.setdefault(_group_key(run, factor), []).append(residual)
        if len(groups) < 2:
            continue
        means = [sum(vals) / len(vals) for vals in groups.values() if vals]
        if len(means) < 2 or max(means) - min(means) <= 0.0005:
            continue
        signs = {1 if m > 0 else -1 if m < 0 else 0 for m in means}
        if 0 in signs or len(signs) < 2:
            continue
        return True
    return False


def _fit(runs: list[dict[str, Any]], visual_tokens_per_image: int = 0) -> FitResult:
    """Least-squares fit of E = β0 + β_img*n_img + β_in*(text_only/1k) + β_out*(out/1k).

    visual_tokens_per_image: if Ollama includes visual tokens in prompt_eval_count
    (as moondream does), pass the per-image token count so the design matrix uses
    text-only tokens rather than the correlated total. This decouples the n_images
    column from in_k and keeps the condition number low.
    """
    try:
        import numpy as np
        _numpy_available = True
    except ImportError:
        _numpy_available = False

    def _text_only_k(r: dict[str, Any]) -> float:
        return float(max(0.0, r["text_tokens"] - r["n_images"] * visual_tokens_per_image)) / 1000.0

    has_image_feature = any(r["n_images"] > 0 for r in runs)
    feature_names = ["intercept"]
    if has_image_feature:
        feature_names.append("image")
    feature_names.extend(["input", "output"])

    def _row(r: dict[str, Any]) -> list[float]:
        values = {
            "intercept": 1.0,
            "image": float(r["n_images"]),
            "input": _text_only_k(r),
            "output": r["output_tokens"] / 1000.0,
        }
        return [values[name] for name in feature_names]

    n = len(runs)
    if _numpy_available:
        import numpy as np  # type: ignore[import]

        X = np.array([_row(r) for r in runs])
        y = np.array([r["energy_wh"] for r in runs])

        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

        # Condition number on standardised X (not raw X).
        # Raw cond is inflated by unit differences between intercept (1.0) and
        # feature columns (values 0.001–0.5). Standardising non-intercept columns
        # makes the threshold of 30 meaningful and hardware-independent.
        col_stds = X[:, 1:].std(axis=0)
        col_stds[col_stds < 1e-10] = 1.0  # avoid division by zero for degenerate columns
        X_scaled = X.copy()
        X_scaled[:, 1:] /= col_stds
        cond = float(np.linalg.cond(X_scaled))

        coeff_by_name = {
            name: float(value)
            for name, value in zip(feature_names, coeffs)
        }
        b0 = coeff_by_name["intercept"]
        b_img = coeff_by_name.get("image", 0.0)
        b_in = coeff_by_name["input"]
        b_out = coeff_by_name["output"]

        y_pred = X @ coeffs
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Bootstrap CIs (500 resamples)
        rng = np.random.default_rng(42)
        boot_in, boot_out, boot_img = [], [], []
        for _ in range(500):
            idx = rng.integers(0, n, size=n)
            Xb, yb = X[idx], y[idx]
            try:
                cb, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
                boot_coeffs = {
                    name: float(value)
                    for name, value in zip(feature_names, cb)
                }
                if has_image_feature:
                    boot_img.append(boot_coeffs["image"])
                boot_in.append(boot_coeffs["input"])
                boot_out.append(boot_coeffs["output"])
            except Exception:
                pass

        def _ci95(vals: list[float]) -> tuple[float, float]:
            if not vals:
                return (0.0, 0.0)
            s = sorted(vals)
            lo = s[int(len(s) * 0.025)]
            hi = s[int(len(s) * 0.975)]
            return (lo, hi)

        input_ci = _ci95(boot_in)
        output_ci = _ci95(boot_out)
        image_ci = _ci95(boot_img) if has_image_feature else (0.0, 0.0)

    else:
        # Pure-Python normal equations.
        rows = [_row(r) for r in runs]
        y_vals = [r["energy_wh"] for r in runs]

        # XtX and Xty
        k = len(feature_names)
        XtX = [[0.0] * k for _ in range(k)]
        Xty = [0.0] * k
        for row, yi in zip(rows, y_vals):
            for i in range(k):
                Xty[i] += row[i] * yi
                for j in range(k):
                    XtX[i][j] += row[i] * row[j]

        # Solve via Gaussian elimination
        aug = [XtX[i][:] + [Xty[i]] for i in range(k)]
        for col in range(k):
            pivot = max(range(col, k), key=lambda r: abs(aug[r][col]))
            aug[col], aug[pivot] = aug[pivot], aug[col]
            if abs(aug[col][col]) < 1e-12:
                continue
            for row_i in range(k):
                if row_i != col:
                    factor = aug[row_i][col] / aug[col][col]
                    for j in range(k + 1):
                        aug[row_i][j] -= factor * aug[col][j]
        coeffs_list = [aug[i][k] / aug[i][i] if abs(aug[i][i]) > 1e-12 else 0.0 for i in range(k)]
        coeff_by_name = dict(zip(feature_names, coeffs_list))
        b0 = coeff_by_name["intercept"]
        b_img = coeff_by_name.get("image", 0.0)
        b_in = coeff_by_name["input"]
        b_out = coeff_by_name["output"]

        y_pred = [sum(r[j] * coeffs_list[j] for j in range(k)) for r in rows]
        y_mean = sum(y_vals) / len(y_vals)
        ss_res = sum((yi - ypi) ** 2 for yi, ypi in zip(y_vals, y_pred))
        ss_tot = sum((yi - y_mean) ** 2 for yi in y_vals)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        cond = float("inf")  # can't compute without numpy; condition check skipped
        input_ci = (b_in * 0.8, b_in * 1.2)
        output_ci = (b_out * 0.8, b_out * 1.2)
        image_ci = (b_img * 0.8, b_img * 1.2) if has_image_feature else (0.0, 0.0)

    # Validation
    reasons = []
    if r2 < 0.85:
        reasons.append(f"r2={r2:.3f} < 0.85")
    if math.isfinite(cond) and cond > 30:
        reasons.append(f"condition_number={cond:.1f} > 30 (workload shapes not distinct enough)")
    # Intercept: a small negative is a benign regression artifact (clamped to 0);
    # only a clearly-negative intercept signals a bad fit.
    if b0 < -1e-3:
        reasons.append(f"intercept_wh={b0:.8f} is negative (regression artifact)")
    # Output must be clearly positive — decode energy is the dominant, well-posed
    # term, so a negative here is a real anomaly.
    if b_out < 0:
        reasons.append(f"wh_per_1k_output={b_out:.6f} is negative")
    # Small terms (input/image) are handled by the CI-aware floor below, not by a
    # bare sign check, so a coefficient at the measurement floor is reported as 0
    # rather than rejecting the whole fit.

    residuals = [float(r["energy_wh"]) - float(p) for r, p in zip(runs, y_pred)]
    residuals_structured = _detect_residuals_structured(runs, residuals)
    if residuals_structured:
        reasons.append("residuals_structured=true — structured residual pattern detected")

    # Floor a small coefficient to 0 when its bootstrap 95% CI includes zero:
    # it is not distinguishable from the measurement floor, so reporting the point
    # estimate would be false precision. The CI stays in provenance as the loud
    # signal. A coefficient whose entire CI sits below zero is a real anomaly.
    def _floor_small(value: float, ci: tuple[float, float], name: str) -> float:
        lo, hi = ci
        if lo <= 0.0 <= hi:
            return 0.0
        if hi < 0.0:
            reasons.append(
                f"{name}={value:.6f} is negative (95% CI entirely below zero "
                f"[{lo:.6f}, {hi:.6f}]; regression artifact)"
            )
            return 0.0
        return max(0.0, value)

    return FitResult(
        intercept_wh=max(0.0, b0),
        wh_per_image=(
            _floor_small(b_img, image_ci, "wh_per_image")
            if has_image_feature
            else 0.0
        ),
        wh_per_1k_input=_floor_small(b_in, input_ci, "wh_per_1k_input"),
        wh_per_1k_output=max(0.0, b_out),
        r2=r2,
        condition_number=cond,
        input_ci95=input_ci,
        output_ci95=output_ci,
        image_ci95=image_ci,
        valid=len(reasons) == 0,
        invalid_reasons=reasons,
        residuals_structured=residuals_structured,
    )


def _require_numpy_for_calibration() -> None:
    """Apple Silicon calibration needs NumPy for conditioning and bootstrap CIs."""
    try:
        import numpy  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Apple Silicon calibration requires NumPy for a reliable least-squares fit. "
            "Install the optional extra before running calibration: "
            "pip install 'vetch[apple-silicon]'"
        ) from e


def _active_calibration_rejection_reasons(
    fit: FitResult,
    power_state: dict[str, Any],
    gpu_power_captured: bool,
    idle_drift_pct: float,
    samples: int,
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
    if idle_drift_pct > 15:
        reasons.append(f"idle_drift_pct={idle_drift_pct:.1f} > 15%")

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

    Returns:
        ``(CalibrationResult, record_path)`` — coefficients plus the v1 JSON path.
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
    assert_root()
    if is_cloud_provider(provider):
        raise ValueError(
            f"provider={provider!r} is a cloud/API vendor and is refused for Tier-0 "
            "calibration: it is ambiguous (real hosted API vs OpenAI-compatible "
            "local) and would attach local coefficients to cloud events. Use "
            "provider='self-hosted' or 'ollama'."
        )

    hw = get_hardware_info()
    power_state = check_power_state()
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
    grid = _build_grid() * max(1, iterations)
    total_runs = len(grid)

    print(f"Pre-generating {sum(s.n_images for s in grid)} unique calibration image(s) "
          f"for {total_runs} runs (busts Ollama KV cache)...")
    run_images: list[list[str]] = []
    for i, spec in enumerate(grid):
        run_seed = i + 100
        imgs = [_unique_image_b64(seed=run_seed + j * 1000) for j in range(spec.n_images)]
        run_images.append(imgs)
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
        run_images = [[] for _ in grid]

    # --- powermetrics monitor ------------------------------------------------
    # Declare time-window variables before the with block so they are accessible
    # for deferred energy computation after the monitor exits.
    idle_start = idle_end = 0.0
    idle_start2 = idle_end2 = 0.0

    print("Starting powermetrics... (this requires root)")
    with AppleSiliconMonitor(sample_interval_ms=100) as monitor:
        # Wait briefly for monitor to settle and get first samples
        time.sleep(1.0)

        # Warm-up BEFORE idle baseline: loads model/vision encoder and brings system to
        # operating temperature. Idle power measured after warm-up represents the
        # thermal state during calibration runs, not the colder pre-load idle.
        warmup_mode = "image call to load vision encoder" if model_is_vlm else "text-only call"
        print(f"Warming up ({warmup_mode})...")
        warmup_prompt = _unique_prompt(approx_tokens=20, seed=0)
        warmup_img = _warmup_image_b64(model_is_vlm, image_pool)
        _ollama_generate(base_url, model, warmup_prompt, warmup_img, max_tokens=10)
        time.sleep(2.0)

        # Idle baseline (after warm-up) — record window; energy computed post-exit
        print("Measuring idle baseline (post-warmup)...")
        idle_start = time.monotonic() * 1000
        time.sleep(3.0)
        idle_end = time.monotonic() * 1000

        # Run the workload grid — record windows only, no energy yet.
        # The batch-based monitor returns samples only when each subprocess.run()
        # completes (~4 s per batch), so integrate() would return 0.0 if called
        # inline. All energies are computed after __exit__ joins the thread.
        print(f"Running calibration grid ({total_runs} runs)...")
        print()

        raw_records: list[dict[str, Any]] = []
        grid_failures = 0
        for i, (spec, images) in enumerate(zip(grid, run_images)):
            run_seed = i + 100
            prompt = _unique_prompt(approx_tokens=spec.approx_text_tokens, seed=run_seed)

            t_start = time.monotonic() * 1000
            try:
                text_tokens, output_tokens = _ollama_generate(
                    base_url, model, prompt,
                    image_b64=images[0] if images else None,
                    max_tokens=spec.max_tokens,
                )
            except RuntimeError as e:
                print(f"  Run {i+1}/{total_runs} FAILED: {e}", file=sys.stderr)
                grid_failures += 1
                continue
            t_end = time.monotonic() * 1000

            if text_tokens == 0:
                grid_failures += 1
                continue

            raw_records.append({
                "n_images": spec.n_images,
                "text_tokens": text_tokens,
                "output_tokens": output_tokens,
                "t_start": t_start,
                "t_end": t_end,
                "replicate": spec.replicate,
            })

            if verbose:
                print(
                    f"  [{i+1:2d}/{total_runs}] n_img={spec.n_images} "
                    f"in={text_tokens:4d} out={output_tokens:3d} "
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

        # Idle baseline (after)
        idle_start2 = time.monotonic() * 1000
        time.sleep(3.0)
        idle_end2 = time.monotonic() * 1000

    # __exit__ joined the background thread — all batches have now completed
    # and monitor._samples contains the full session's power data.

    # --- Sanity check: did we get any power samples at all? ------------------
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
        run_records.append({
            "n_images": raw["n_images"],
            "text_tokens": raw["text_tokens"],
            "output_tokens": raw["output_tokens"],
            "energy_wh": net_energy_wh,
            "raw_energy_wh": energy_wh,  # pre-clamping, used for gap detection
            "duration_ms": duration_ms,
            "replicate": raw["replicate"],
        })
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
    idle_drift_pct = abs(idle_watts_after - idle_watts_before) / max(idle_watts_before, 0.01) * 100
    if idle_drift_pct > 15:
        print(
            f"WARNING: Idle power drifted {idle_drift_pct:.1f}% between start and end.\n"
            f"         Start: {idle_watts_before:.2f} W  End: {idle_watts_after:.2f} W\n"
            f"         Thermal state changed. Results may be unreliable.",
            file=sys.stderr,
        )
        raise RuntimeError(
            f"Idle power drift {idle_drift_pct:.1f}% exceeds 15% limit. "
            "Let the machine cool, disable Low Power Mode, and retry on AC power."
        )

    # --- Fit -----------------------------------------------------------------
    if len(run_records) < 8:
        raise RuntimeError(
            f"Too few successful runs ({len(run_records)}) to fit the model. "
            "Check Ollama is running and the model is pulled."
        )

    print(f"Fitting energy model ({len(run_records)} runs)...")
    fit = _fit(run_records, visual_tokens_per_image=fit_visual_tokens)
    rejection_reasons = _active_calibration_rejection_reasons(
        fit=fit,
        power_state=power_state,
        gpu_power_captured=monitor.info.gpu_power_captured,
        idle_drift_pct=idle_drift_pct,
        samples=len(run_records),
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
        energy_source="powermetrics",
        measurement_basis="powermetrics_estimated_soc_power",
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
            "sampler_cadence_ms": monitor.info.sample_interval_ms,
            "integration_method": "trapezoidal",
            "chip_family": hw.chip_family,
            "chip_tier": hw.chip_tier,
            "memory_gb": hw.memory_gb,
            "macos_version": hw.macos_version,
            "api_backend": "ollama",
            "gpu_power_captured": monitor.info.gpu_power_captured,
            "power_state": power_state,
        },
    )
    record_path = commit_calibration(result, identity, provenance)
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
        f"  wh_per_1k_input:    {result.wh_per_1k_input:.6f}  Wh / 1k text tokens",
        f"  wh_per_1k_output:   {result.wh_per_1k_output:.6f}  Wh / 1k output tokens",
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
        "  Saved to: ~/.vetch/calibrations/  (v1 identity-keyed record)",
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
