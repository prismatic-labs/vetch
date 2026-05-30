"""Tests for Apple Silicon calibration module (calibrate_metal.py).

All tests that require actual hardware (powermetrics, Ollama) are skipped
automatically in CI. The core logic — power parsing, trapezoidal integration,
PNG generation, and the least-squares fit — is tested without any external deps.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

class TestIsAppleSilicon:
    def test_returns_false_on_non_darwin(self) -> None:
        from vetch.calibrate_metal import is_apple_silicon
        with patch("platform.system", return_value="Linux"):
            assert is_apple_silicon() is False

    def test_returns_false_when_sysctl_reports_0(self) -> None:
        from vetch.calibrate_metal import is_apple_silicon
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "0\n"
        with patch("platform.system", return_value="Darwin"), \
             patch("subprocess.run", return_value=mock_result):
            assert is_apple_silicon() is False

    def test_returns_true_when_sysctl_reports_1(self) -> None:
        from vetch.calibrate_metal import is_apple_silicon
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "1\n"
        with patch("platform.system", return_value="Darwin"), \
             patch("subprocess.run", return_value=mock_result):
            assert is_apple_silicon() is True

    def test_returns_false_when_sysctl_fails(self) -> None:
        from vetch.calibrate_metal import is_apple_silicon
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        with patch("platform.system", return_value="Darwin"), \
             patch("subprocess.run", return_value=mock_result):
            assert is_apple_silicon() is False


# ---------------------------------------------------------------------------
# Hardware info
# ---------------------------------------------------------------------------

class TestHardwareInfo:
    def test_chip_family_parsing(self) -> None:
        from vetch.calibrate_metal import HardwareInfo
        hw = HardwareInfo(chip_raw="Apple M3 Max", memory_gb=36, macos_version="15.4.1")
        assert hw.chip_family == "M3"
        assert hw.chip_tier == "Max"

    def test_chip_family_base_model(self) -> None:
        from vetch.calibrate_metal import HardwareInfo
        hw = HardwareInfo(chip_raw="Apple M5", memory_gb=16, macos_version="15.0")
        assert hw.chip_family == "M5"
        assert hw.chip_tier == ""

    def test_unknown_chip_graceful(self) -> None:
        from vetch.calibrate_metal import HardwareInfo
        hw = HardwareInfo(chip_raw="Intel Core i9", memory_gb=64, macos_version="14.0")
        assert hw.chip_family == ""


# ---------------------------------------------------------------------------
# AppleSiliconMonitor — parsing and integration (no actual powermetrics)
# ---------------------------------------------------------------------------

class TestAppleSiliconMonitorParsing:
    def _make_monitor(self) -> Any:
        from vetch.calibrate_metal import AppleSiliconMonitor
        monitor = AppleSiliconMonitor.__new__(AppleSiliconMonitor)
        monitor._interval_ms = 100
        monitor._samples = []
        monitor._reader_thread = None
        monitor._stop_event = None
        from vetch.calibrate_metal import MonitorInfo
        monitor.info = MonitorInfo()
        return monitor

    def test_parse_combined_power_line_mw(self) -> None:
        monitor = self._make_monitor()
        monitor._parse_line("Combined Power (CPU + GPU + ANE): 4821 mW", mono_ms=1000.0)
        assert len(monitor._samples) == 1
        assert abs(monitor._samples[0].combined_watts - 4.821) < 1e-6
        assert monitor.info.gpu_power_captured is True

    def test_parse_combined_power_line_w(self) -> None:
        monitor = self._make_monitor()
        monitor._parse_line("Combined Power (CPU + GPU + ANE): 4.821 W", mono_ms=1000.0)
        assert len(monitor._samples) == 1
        assert abs(monitor._samples[0].combined_watts - 4.821) < 1e-6

    def test_parse_total_soc_power_line(self) -> None:
        monitor = self._make_monitor()
        monitor._parse_line("Total SoC Power: 4821 mW", mono_ms=1000.0)
        assert len(monitor._samples) == 1
        assert abs(monitor._samples[0].combined_watts - 4.821) < 1e-6

    def test_parse_cpu_gpu_ane_power_line(self) -> None:
        monitor = self._make_monitor()
        monitor._parse_line("CPU + GPU + ANE Power: 4.821 W", mono_ms=1000.0)
        assert len(monitor._samples) == 1
        assert abs(monitor._samples[0].combined_watts - 4.821) < 1e-6

    def test_parse_ignores_unrelated_lines(self) -> None:
        monitor = self._make_monitor()
        monitor._parse_line("CPU die temperature: 45.3 C", mono_ms=1000.0)
        assert len(monitor._samples) == 0

    def test_parse_mw_to_watts_conversion(self) -> None:
        monitor = self._make_monitor()
        monitor._parse_line("Combined Power (CPU + GPU + ANE): 10000 mW", mono_ms=500.0)
        assert abs(monitor._samples[0].combined_watts - 10.0) < 1e-9

    def test_trapezoidal_integration(self) -> None:
        from vetch.calibrate_metal import PowerSample
        monitor = self._make_monitor()
        # Constant 10 W over 1 second = 10/3600 Wh
        t0 = 0.0
        t1 = 1000.0  # 1 second in ms
        monitor._samples = [
            PowerSample(mono_ms=t0, combined_watts=10.0),
            PowerSample(mono_ms=500.0, combined_watts=10.0),
            PowerSample(mono_ms=t1, combined_watts=10.0),
        ]
        wh = monitor.integrate(t0, t1)
        expected = 10.0 / 3600.0
        assert abs(wh - expected) < 1e-9

    def test_trapezoidal_integration_varying_power(self) -> None:
        from vetch.calibrate_metal import PowerSample
        monitor = self._make_monitor()
        # Linear ramp: 0 W → 10 W over 1 second → average 5 W → 5/3600 Wh
        # Samples sit exactly at window boundaries so no interpolation needed
        monitor._samples = [
            PowerSample(mono_ms=0.0, combined_watts=0.0),
            PowerSample(mono_ms=1000.0, combined_watts=10.0),
        ]
        wh = monitor.integrate(0.0, 1000.0)
        expected = 5.0 / 3600.0
        assert abs(wh - expected) < 1e-9

    def test_trapezoidal_integration_with_edge_interpolation(self) -> None:
        from vetch.calibrate_metal import PowerSample
        monitor = self._make_monitor()
        # Samples at 0ms and 2000ms, but window is 500ms–1500ms.
        # Interpolated start power = 5W, interpolated end power = 15W,
        # average over 1s = 10W → 10/3600 Wh
        monitor._samples = [
            PowerSample(mono_ms=0.0, combined_watts=0.0),
            PowerSample(mono_ms=2000.0, combined_watts=20.0),
        ]
        wh = monitor.integrate(500.0, 1500.0)
        expected = 10.0 / 3600.0  # 1 second at average 10 W
        assert abs(wh - expected) < 1e-9

    def test_integrate_empty_window_returns_zero(self) -> None:
        monitor = self._make_monitor()
        wh = monitor.integrate(1000.0, 2000.0)
        assert wh == 0.0

    def test_mean_watts(self) -> None:
        from vetch.calibrate_metal import PowerSample
        monitor = self._make_monitor()
        monitor._samples = [
            PowerSample(mono_ms=0.0, combined_watts=4.0),
            PowerSample(mono_ms=500.0, combined_watts=6.0),
            PowerSample(mono_ms=1000.0, combined_watts=8.0),
        ]
        mean = monitor.mean_watts(0.0, 1000.0)
        assert abs(mean - 6.0) < 1e-9


# ---------------------------------------------------------------------------
# Unique prompt / image generation
# ---------------------------------------------------------------------------

class TestUniquePrompt:
    def test_different_seeds_produce_different_prompts(self) -> None:
        from vetch.calibrate_metal import _unique_prompt
        p1 = _unique_prompt(approx_tokens=50, seed=1)
        p2 = _unique_prompt(approx_tokens=50, seed=2)
        assert p1 != p2

    def test_same_seed_is_reproducible(self) -> None:
        from vetch.calibrate_metal import _unique_prompt
        p = _unique_prompt(approx_tokens=50, seed=42)
        assert p == _unique_prompt(approx_tokens=50, seed=42)

    def test_contains_nonce(self) -> None:
        from vetch.calibrate_metal import _unique_prompt
        p = _unique_prompt(approx_tokens=30, seed=7)
        assert "[ref:" in p


class TestUniqueImage:
    def test_produces_valid_base64_png(self) -> None:
        import base64

        from vetch.calibrate_metal import _unique_image_b64
        b64 = _unique_image_b64(seed=1, size=16)  # small size for test speed
        raw = base64.b64decode(b64)
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"

    def test_different_seeds_produce_different_images(self) -> None:
        from vetch.calibrate_metal import _unique_image_b64
        assert _unique_image_b64(seed=1, size=16) != _unique_image_b64(seed=2, size=16)

    def test_same_seed_is_reproducible(self) -> None:
        from vetch.calibrate_metal import _unique_image_b64
        assert _unique_image_b64(seed=99, size=16) == _unique_image_b64(seed=99, size=16)

    def test_default_size_is_standard(self) -> None:
        import inspect

        from vetch.calibrate_metal import CALIBRATION_IMAGE_SIZE_PX, _unique_image_b64
        sig = inspect.signature(_unique_image_b64)
        assert sig.parameters["size"].default == CALIBRATION_IMAGE_SIZE_PX


class TestWarmupImage:
    def test_text_only_model_gets_no_warmup_image(self) -> None:
        from vetch.calibrate_metal import _warmup_image_b64

        assert _warmup_image_b64(False, ["img0", "img1"]) is None

    def test_vlm_gets_second_pool_image(self) -> None:
        from vetch.calibrate_metal import _warmup_image_b64

        assert _warmup_image_b64(True, ["img0", "img1"]) == "img1"


class TestNumpyRequirement:
    def test_missing_numpy_fails_before_long_calibration_run(self) -> None:
        import builtins

        from vetch.calibrate_metal import _require_numpy_for_calibration

        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "numpy":
                raise ImportError("no numpy")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            try:
                _require_numpy_for_calibration()
            except RuntimeError as e:
                assert "vetch[apple-silicon]" in str(e)
            else:
                raise AssertionError("missing NumPy should fail calibration preflight")


# ---------------------------------------------------------------------------
# Visual token lookup
# ---------------------------------------------------------------------------

class TestVisualTokenLookup:
    def test_moondream_returns_729(self) -> None:
        from vetch.calibrate_metal import _get_visual_tokens_per_image
        tokens, source = _get_visual_tokens_per_image("moondream:latest")
        assert tokens == 729
        assert source == "known_constant"

    def test_moondream_without_tag(self) -> None:
        from vetch.calibrate_metal import _get_visual_tokens_per_image
        tokens, source = _get_visual_tokens_per_image("moondream")
        assert tokens == 729

    def test_unknown_model_returns_default(self) -> None:
        from vetch.calibrate_metal import _get_visual_tokens_per_image
        tokens, source = _get_visual_tokens_per_image("my-custom-vlm:latest")
        assert tokens == 729  # assumed default
        assert source == "assumed_default"


# ---------------------------------------------------------------------------
# Fit (least-squares model)
# ---------------------------------------------------------------------------

class TestFit:
    def _synthetic_runs(
        self,
        b0: float = 0.00002,
        b_img: float = 0.0005,
        b_in: float = 0.00030,
        b_out: float = 0.00180,
        noise: float = 0.0,
    ) -> list[dict]:
        """Generate synthetic run records from known coefficients."""
        import random
        rng = random.Random(42)
        runs = []
        for n_img in [0, 0, 0, 1, 1, 2]:
            for in_k in [0.020, 0.128, 0.512]:
                for out_k in [0.005, 0.064, 0.256]:
                    energy = b0 + b_img * n_img + b_in * in_k + b_out * out_k
                    if noise:
                        energy += rng.gauss(0, noise)
                    runs.append({
                        "n_images": n_img,
                        "text_tokens": int(in_k * 1000),
                        "output_tokens": int(out_k * 1000),
                        "energy_wh": max(0.0, energy),
                    })
        return runs

    def test_recovers_known_coefficients(self) -> None:
        from vetch.calibrate_metal import _fit
        runs = self._synthetic_runs()
        fit = _fit(runs)
        assert abs(fit.wh_per_1k_input - 0.00030) < 1e-6
        assert abs(fit.wh_per_1k_output - 0.00180) < 1e-6
        assert abs(fit.wh_per_image - 0.0005) < 1e-6
        assert fit.r2 > 0.999
        assert fit.valid is True

    def test_valid_false_on_negative_coefficient(self) -> None:
        from vetch.calibrate_metal import _fit
        # All text-only runs → can't distinguish image coefficient → will likely be negative or zero
        # Instead, explicitly test with data that produces a negative image coefficient
        runs = []
        # Craft runs where energy goes DOWN when images increase (pathological data)
        for n_img in [0, 1, 2]:
            for in_k in [0.050, 0.200]:
                for out_k in [0.010, 0.100]:
                    energy = 0.001 - 0.001 * n_img + 0.0003 * in_k + 0.0018 * out_k
                    runs.append({
                        "n_images": n_img,
                        "text_tokens": int(in_k * 1000),
                        "output_tokens": int(out_k * 1000),
                        "energy_wh": max(0.0, energy),
                    })
        fit = _fit(runs)
        assert fit.valid is False
        assert any("negative" in r for r in fit.invalid_reasons)

    def test_valid_false_on_low_r2(self) -> None:
        import random

        from vetch.calibrate_metal import _fit
        rng = random.Random(7)
        # Pure noise — no signal
        runs = [
            {
                "n_images": rng.randint(0, 2),
                "text_tokens": rng.randint(20, 500),
                "output_tokens": rng.randint(5, 250),
                "energy_wh": rng.uniform(0.0001, 0.005),
            }
            for _ in range(30)
        ]
        fit = _fit(runs)
        # Noisy data should produce low R² and fail
        assert fit.r2 < 0.85 or not fit.valid

    def test_text_only_fit_drops_image_dimension(self) -> None:
        import math

        from vetch.calibrate_metal import _active_calibration_rejection_reasons, _fit

        runs = []
        for in_k in [0.020, 0.128, 0.512]:
            for out_k in [0.005, 0.064, 0.256]:
                energy = 0.00002 + 0.00030 * in_k + 0.00180 * out_k
                runs.append({
                    "n_images": 0,
                    "text_tokens": int(in_k * 1000),
                    "output_tokens": int(out_k * 1000),
                    "energy_wh": energy,
                })

        fit = _fit(runs)

        assert fit.valid is True
        assert not math.isnan(fit.condition_number)  # inf ok: pure-Python fallback skips gate
        assert fit.wh_per_image == 0.0
        assert fit.image_ci95 == (0.0, 0.0)
        assert _active_calibration_rejection_reasons(
            fit=fit,
            power_state={"on_ac_power": True, "low_power_mode": False},
            gpu_power_captured=True,
            idle_drift_pct=5.0,
            samples=len(runs),
        ) == []

    def test_coefficients_non_negative_after_clamping(self) -> None:
        from vetch.calibrate_metal import _fit
        runs = self._synthetic_runs(b0=0.00002, b_img=0.0005, b_in=0.0003, b_out=0.0018)
        fit = _fit(runs)
        assert fit.wh_per_image >= 0.0
        assert fit.wh_per_1k_input >= 0.0
        assert fit.wh_per_1k_output >= 0.0

    def test_with_noise_still_recovers_roughly(self) -> None:
        from vetch.calibrate_metal import _fit
        runs = self._synthetic_runs(noise=0.00002)  # small Gaussian noise
        fit = _fit(runs)
        assert abs(fit.wh_per_1k_input - 0.00030) < 0.00010
        assert abs(fit.wh_per_1k_output - 0.00180) < 0.00050
        assert fit.r2 > 0.85


class TestActiveCalibrationQualityGate:
    def _fit_result(self, *, valid: bool = True, cond: float = 10.0) -> object:
        from vetch.calibrate_metal import FitResult

        return FitResult(
            intercept_wh=0.00001,
            wh_per_image=0.0005,
            wh_per_1k_input=0.0003,
            wh_per_1k_output=0.0018,
            r2=0.95,
            condition_number=cond,
            input_ci95=(0.0002, 0.0004),
            output_ci95=(0.0015, 0.0020),
            image_ci95=(0.0004, 0.0006),
            valid=valid,
            invalid_reasons=[] if valid else ["r2=0.100 < 0.85"],
        )

    def test_valid_clean_run_can_become_active(self) -> None:
        from vetch.calibrate_metal import _active_calibration_rejection_reasons

        reasons = _active_calibration_rejection_reasons(
            fit=self._fit_result(),
            power_state={"on_ac_power": True, "low_power_mode": False},
            gpu_power_captured=True,
            idle_drift_pct=5.0,
            samples=22,
        )

        assert reasons == []

    def test_invalid_fit_is_not_active(self) -> None:
        from vetch.calibrate_metal import _active_calibration_rejection_reasons

        reasons = _active_calibration_rejection_reasons(
            fit=self._fit_result(valid=False),
            power_state={"on_ac_power": True, "low_power_mode": False},
            gpu_power_captured=True,
            idle_drift_pct=5.0,
            samples=22,
        )

        assert any("r2" in reason for reason in reasons)

    def test_suspect_environment_is_not_active(self) -> None:
        from vetch.calibrate_metal import _active_calibration_rejection_reasons

        reasons = _active_calibration_rejection_reasons(
            fit=self._fit_result(),
            power_state={"on_ac_power": False, "low_power_mode": True},
            gpu_power_captured=False,
            idle_drift_pct=20.0,
            samples=7,
        )

        assert any("not on AC power" in reason for reason in reasons)
        assert any("low_power_mode" in reason for reason in reasons)
        assert any("gpu_power_captured" in reason for reason in reasons)
        assert any("idle_drift" in reason for reason in reasons)
        assert any("samples=7" in reason for reason in reasons)

    def test_infinite_condition_number_skips_cond_gate(self) -> None:
        """inf cond means pure-Python fallback (no NumPy); do not reject active install."""
        from vetch.calibrate_metal import _active_calibration_rejection_reasons

        reasons = _active_calibration_rejection_reasons(
            fit=self._fit_result(cond=float("inf")),
            power_state={"on_ac_power": True, "low_power_mode": False},
            gpu_power_captured=True,
            idle_drift_pct=5.0,
            samples=22,
        )

        assert not any("condition_number" in reason for reason in reasons)


# ---------------------------------------------------------------------------
# CalibrationResult VLM fields
# ---------------------------------------------------------------------------

class TestCalibrationResultVLMFields:
    def test_vlm_fields_default_none(self) -> None:
        from vetch.calibrate import CalibrationResult
        r = CalibrationResult(
            model="llama3.1:8b", provider="ollama",
            wh_per_1k_input=0.04, wh_per_1k_output=0.12,
            tier=0, samples=10, gpu_name="NVIDIA RTX 4090",
        )
        assert r.wh_per_image is None
        assert r.visual_tokens_per_image is None
        assert r.intercept_wh is None

    def test_vlm_fields_can_be_set(self) -> None:
        from vetch.calibrate import CalibrationResult
        r = CalibrationResult(
            model="moondream:latest", provider="ollama",
            wh_per_1k_input=0.00031, wh_per_1k_output=0.00185,
            tier=0, samples=28, gpu_name="Apple M5",
            wh_per_image=0.00054,
            visual_tokens_per_image=729,
            intercept_wh=0.000018,
        )
        assert abs(r.wh_per_image - 0.00054) < 1e-9  # type: ignore[operator]
        assert r.visual_tokens_per_image == 729


# ---------------------------------------------------------------------------
# save_calibration — VLM fields persisted
# ---------------------------------------------------------------------------

class TestSaveCalibrationVLM:
    def test_vlm_fields_written_to_json(self) -> None:
        import tempfile

        import vetch.calibrate as vcal
        from vetch.calibrate import CalibrationResult, save_calibration

        r = CalibrationResult(
            model="moondream:latest", provider="ollama",
            wh_per_1k_input=0.00031, wh_per_1k_output=0.00185,
            tier=0, samples=28, gpu_name="Apple M5",
            wh_per_image=0.00054,
            visual_tokens_per_image=729,
            intercept_wh=0.000018,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            original = vcal.CALIBRATION_DIR
            vcal.CALIBRATION_DIR = Path(tmpdir)
            try:
                save_calibration(r)
                path = Path(tmpdir) / "ollama_moondream_latest.json"
                assert path.exists()
                data = json.loads(path.read_text())
                assert abs(data["wh_per_image"] - 0.00054) < 1e-9
                assert data["visual_tokens_per_image"] == 729
                assert abs(data["intercept_wh"] - 0.000018) < 1e-9
            finally:
                vcal.CALIBRATION_DIR = original

    def test_vlm_fields_omitted_when_none(self) -> None:
        import tempfile

        import vetch.calibrate as vcal
        from vetch.calibrate import CalibrationResult, save_calibration

        r = CalibrationResult(
            model="llama3:8b", provider="ollama",
            wh_per_1k_input=0.04, wh_per_1k_output=0.12,
            tier=0, samples=5, gpu_name="Test GPU",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            original = vcal.CALIBRATION_DIR
            vcal.CALIBRATION_DIR = Path(tmpdir)
            try:
                save_calibration(r)
                data = json.loads((Path(tmpdir) / "ollama_llama3_8b.json").read_text())
                assert "wh_per_image" not in data
                assert "visual_tokens_per_image" not in data
            finally:
                vcal.CALIBRATION_DIR = original


# ---------------------------------------------------------------------------
# load_calibration
# ---------------------------------------------------------------------------

class TestLoadCalibration:
    def test_load_returns_none_when_file_absent(self) -> None:
        import vetch.calibrate as vcal
        from vetch.calibrate import load_calibration
        with tempfile.TemporaryDirectory() as tmpdir:
            original = vcal.CALIBRATION_DIR
            vcal.CALIBRATION_DIR = Path(tmpdir)
            try:
                result = load_calibration("ollama", "no-such-model")
                assert result is None
            finally:
                vcal.CALIBRATION_DIR = original

    def test_load_model_alias_without_tag(self) -> None:
        import vetch.calibrate as vcal
        from vetch.calibrate import CalibrationResult, load_calibration, save_calibration

        r = CalibrationResult(
            model="moondream:latest",
            provider="ollama",
            wh_per_1k_input=0.00031,
            wh_per_1k_output=0.00185,
            tier=0,
            samples=28,
            gpu_name="Apple M5",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            original = vcal.CALIBRATION_DIR
            vcal.CALIBRATION_DIR = Path(tmpdir)
            try:
                save_calibration(r)
                loaded = load_calibration("ollama", "moondream")
                assert loaded is not None
                assert loaded.wh_per_1k_input == r.wh_per_1k_input
            finally:
                vcal.CALIBRATION_DIR = original

    def test_roundtrip_with_vlm_fields(self) -> None:
        import vetch.calibrate as vcal
        from vetch.calibrate import CalibrationResult, load_calibration, save_calibration

        r = CalibrationResult(
            model="moondream:latest", provider="ollama",
            wh_per_1k_input=0.00031, wh_per_1k_output=0.00185,
            tier=0, samples=28, gpu_name="Apple M5",
            wh_per_image=0.00054,
            visual_tokens_per_image=729,
            intercept_wh=0.000018,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            original = vcal.CALIBRATION_DIR
            vcal.CALIBRATION_DIR = Path(tmpdir)
            try:
                save_calibration(r)
                loaded = load_calibration("ollama", "moondream:latest")
                assert loaded is not None
                assert abs(loaded.wh_per_image - 0.00054) < 1e-9  # type: ignore[operator]
                assert loaded.visual_tokens_per_image == 729
                assert loaded.wh_per_1k_input == r.wh_per_1k_input
                assert loaded.samples == 28
                assert abs(loaded.intercept_wh - 0.000018) < 1e-9  # type: ignore[operator]
            finally:
                vcal.CALIBRATION_DIR = original


# ---------------------------------------------------------------------------
# calculate_energy — n_images integration
# ---------------------------------------------------------------------------

class TestCalculateEnergyNImages:
    def test_cache_read_discount_applies_to_energy_override(self) -> None:
        from vetch.calculation import calculate_energy

        override = {
            "wh_per_1k_input": 1.0,
            "wh_per_1k_output": 3.0,
            "tier": 0,
        }
        uncached, *_ = calculate_energy(
            1000, 0, "local-model", energy_override=override, cache_read_tokens=0
        )
        cached, *_ = calculate_energy(
            1000, 0, "local-model", energy_override=override, cache_read_tokens=1000
        )

        assert uncached == 1.0
        assert 0.10 <= cached / uncached <= 0.25

    def test_visual_tokens_subtracted_from_text_coefficient(self) -> None:
        from vetch.calculation import calculate_energy

        override = {
            "wh_per_1k_input": 1.0,
            "wh_per_1k_output": 0.0,
            "wh_per_image": 0.5,
            "visual_tokens_per_image": 729,
            "tier": 0,
        }
        combined, *_ = calculate_energy(
            1000, 0, "moondream", energy_override=override, n_images=1
        )
        text_only, *_ = calculate_energy(
            271, 0, "moondream", energy_override=override, n_images=0
        )
        assert abs(combined - text_only - 0.5) < 1e-9

    def test_n_images_adds_image_energy(self) -> None:
        from vetch.calculation import calculate_energy
        override = {
            "wh_per_1k_input": 0.0003,
            "wh_per_1k_output": 0.0018,
            "wh_per_image": 0.0005,
            "tier": 0,
        }
        base_energy, *_ = calculate_energy(
            100, 50, "moondream", energy_override=override, n_images=0
        )
        vlm_energy, *_ = calculate_energy(
            100, 50, "moondream", energy_override=override, n_images=2
        )
        assert abs(vlm_energy - base_energy - 0.001) < 1e-9  # 2 × 0.0005

    def test_n_images_zero_no_change(self) -> None:
        from vetch.calculation import calculate_energy
        override = {
            "wh_per_1k_input": 0.0003,
            "wh_per_1k_output": 0.0018,
            "wh_per_image": 0.0005,
            "tier": 0,
        }
        e0, *_ = calculate_energy(100, 50, "moondream", energy_override=override, n_images=0)
        e_no_arg, *_ = calculate_energy(100, 50, "moondream", energy_override=override)
        assert e0 == e_no_arg

    def test_n_images_without_wh_per_image_no_change(self) -> None:
        from vetch.calculation import calculate_energy
        override = {
            "wh_per_1k_input": 0.0003,
            "wh_per_1k_output": 0.0018,
            "tier": 0,
        }
        e0, *_ = calculate_energy(100, 50, "moondream", energy_override=override, n_images=0)
        e1, *_ = calculate_energy(100, 50, "moondream", energy_override=override, n_images=3)
        assert e0 == e1  # no wh_per_image → no change

    def test_image_tokens_scale_energy_above_image_count(self) -> None:
        from vetch.calculation import calculate_energy
        override = {
            "wh_per_1k_input": 0.0003,
            "wh_per_1k_output": 0.0018,
            "wh_per_image": 0.0005,
            "visual_tokens_per_image": 450,
            "tier": 0,
        }
        base_energy, *_ = calculate_energy(
            2000, 50, "moondream", energy_override=override, n_images=0
        )
        high_res_energy, *_ = calculate_energy(
            2000,
            50,
            "moondream",
            energy_override=override,
            n_images=1,
            image_input_tokens=900,
        )
        # 900 visual tokens → 2 image units (900/450), text-only input = 1100 tokens
        text_only, *_ = calculate_energy(
            1100, 50, "moondream", energy_override=override, n_images=0
        )
        assert abs(high_res_energy - text_only - 0.001) < 1e-9

    def test_image_tokens_do_not_reduce_below_image_count(self) -> None:
        from vetch.calculation import calculate_energy
        override = {
            "wh_per_1k_input": 0.0003,
            "wh_per_1k_output": 0.0018,
            "wh_per_image": 0.0005,
            "visual_tokens_per_image": 450,
            "tier": 0,
        }
        two_image_energy, *_ = calculate_energy(
            2000,
            50,
            "moondream",
            energy_override=override,
            n_images=2,
            image_input_tokens=450,
        )
        text_only, *_ = calculate_energy(
            1100, 50, "moondream", energy_override=override, n_images=0
        )
        assert abs(two_image_energy - text_only - 0.001) < 1e-9
