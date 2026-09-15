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


class TestSharedV2Grid:
    def test_metal_uses_prefill_arm_and_request_aggregation(self) -> None:
        from vetch.calibrate_metal import _grid_design, grid_design_id

        grid = _grid_design()
        assert grid_design_id() == "v2-unbalanced"
        assert any(spec.approx_text_tokens >= 1024 for spec in grid)
        assert any(spec.requests > 1 for spec in grid)
        assert any(spec.approx_text_tokens >= 1024 and spec.max_tokens <= 8
                   for spec in grid)


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
                        # powermetrics is a sampling path, so the shared
                        # admission rule applies a duration floor. Apple Silicon
                        # decode is slow enough that real runs clear it; the
                        # fixture has to as well or every row is dropped.
                        "raw_energy_wh": max(0.0, energy) * 3.0,
                        "duration_ms": 2500.0 + out_k * 1000 * 20.0,
                        "energy_wh": max(0.0, energy),
                    })
        return runs

    # The fit is shared with the CUDA path (vetch.calibrate_v2). These tests
    # therefore assert v2 semantics rather than the v1 behaviour Apple Silicon
    # used to have on its own: coefficients the design cannot resolve come back
    # None with a reason, never floored to 0.0.

    def _fit(self, runs, **kw):
        from vetch.calibrate_v2 import fit_nested_models

        return fit_nested_models(
            runs, idle_watts=kw.pop("idle_watts", 5.0),
            energy_source="powermetrics", **kw,
        )

    def test_recovers_decode_and_withholds_unresolvable_prefill(self) -> None:
        """The synthetic grid tops out at 512 input tokens, so it has no prefill
        isolation arm. Decode is recoverable; prefill is not, and must be
        withheld with a reason rather than published or floored to zero."""
        fit = self._fit(self._synthetic_runs())
        assert fit.wh_per_1k_input is None
        assert "prefill isolation arm" in fit.null_reasons["wh_per_1k_input"]
        assert fit.wh_per_1k_output is not None
        assert fit.valid is True
        # r2 describes the PUBLISHED rung, which omits prefill, so it sits below
        # the full model's near-perfect fit on noiseless data.
        assert fit.r2 > 0.95
        assert fit.rungs["output_intercept_input_image"]["r2"] > 0.999

        # The estimator is exact: in the model containing every term, decode
        # comes back at the generator's 0.00180.
        full = fit.rungs["output_intercept_input_image"]["coeffs"]["output"]
        assert abs(full - 0.00180) < 1e-6

        # The PUBLISHED value is higher, because the rung that survives omits
        # prefill and the survivor absorbs it. That bias is the documented cost
        # of withholding a term, not an estimator error, and it is why the
        # record carries published_rung alongside the number.
        assert fit.published_rung == "output_image"
        assert fit.wh_per_1k_output > full

    def test_prefill_is_recovered_when_the_design_supports_it(self) -> None:
        """Same generator plus a prefill arm: the coefficient becomes publishable.

        This is the pair that shows the withholding above is a property of the
        design and not of the estimator.
        """
        runs = self._synthetic_runs()
        b0, b_in, b_out = 0.00002, 0.00030, 0.00180
        for in_tok in (2048, 4096, 8192, 16384):
            for _ in range(2):
                runs.append({
                    "n_images": 0, "text_tokens": in_tok, "output_tokens": 8,
                    "energy_wh": b0 + b_in * in_tok / 1000 + b_out * 8 / 1000,
                    "raw_energy_wh": 1.0,
                    # powermetrics is a sampling path, so a prefill point must
                    # still clear the duration floor. On real hardware that is
                    # what row aggregation is for.
                    "duration_ms": 2500.0, "requests": 1,
                })
        fit = self._fit(runs)
        assert fit.wh_per_1k_input is not None, fit.null_reasons
        assert abs(fit.wh_per_1k_input - b_in) < 5e-5

    def test_negative_coefficient_is_not_published(self) -> None:
        runs = self._synthetic_runs(b_in=-0.0005)
        fit = self._fit(runs)
        assert fit.wh_per_1k_input is None

    def test_low_r2_blocks_the_rung(self) -> None:
        import random

        rng = random.Random(7)
        runs = self._synthetic_runs()
        for r in runs:
            r["energy_wh"] = abs(rng.gauss(0.01, 0.01))
        fit = self._fit(runs)
        assert fit.wh_per_1k_output is None or fit.r2 < 0.85

    def test_text_only_fit_drops_image_dimension(self) -> None:
        runs = [r for r in self._synthetic_runs() if r["n_images"] == 0]
        fit = self._fit(runs)
        assert fit.wh_per_image is None
        assert fit.null_reasons["wh_per_image"] == "no image feature in the design"

    def test_no_coefficient_is_ever_floored_to_zero(self) -> None:
        """The v1 Apple Silicon path floored a coefficient whose interval spanned
        zero to 0.0, which reads downstream as 'this axis is free'."""
        fit = self._fit(self._synthetic_runs())
        for axis in ("wh_per_1k_input", "wh_per_1k_output", "wh_per_image",
                     "intercept_wh"):
            assert getattr(fit, axis) != 0.0

    def test_with_noise_still_recovers_decode(self) -> None:
        fit = self._fit(self._synthetic_runs(noise=0.00002))
        assert fit.wh_per_1k_output is not None
        assert abs(fit.wh_per_1k_output - 0.00180) < 0.00050
        assert fit.r2 > 0.85


class TestActiveCalibrationQualityGate:
    def _fit_result(self, *, valid: bool = True, cond: float = 10.0) -> object:
        """A fit stub in the shared shape, so the gate is tested against what
        the Metal path now actually produces."""
        from vetch.calibrate_v2 import NestedFitResult

        return NestedFitResult(
            published_rung="output_intercept_input_image",
            intercept_wh=0.00001, wh_per_1k_input=0.0003,
            wh_per_1k_output=0.0018, wh_per_image=0.0005,
            status={
                "wh_per_1k_output": "identified", "intercept_wh": "identified",
                "wh_per_1k_input": "identified", "wh_per_image": "identified",
            },
            null_reasons={}, r2=0.95, condition_number=cond,
            intercept_ci95=(0.0, 0.00002), input_ci95=(0.0002, 0.0004),
            output_ci95=(0.0015, 0.0020), image_ci95=(0.0004, 0.0006),
            residuals_structured=False, active=valid,
            rejection_reasons=[] if valid else ["r2=0.100 < 0.85"],
            axes={}, design_leverage={}, energy_basis={},
            admitted_rows=27, dropped_zero_raw=0,
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


class TestMetalContextLimit:
    """Ollama truncates an over-length prompt and answers normally.

    vLLM rejects one, so the CUDA path could omit a limit and still fail loudly.
    On the sampling path an unfiltered grid loses the prefill span silently,
    after paying the runtime to send prompts the server will not honour.
    """

    def test_ceiling_defaults_and_env_override(self, monkeypatch) -> None:
        from vetch.calibrate_metal import (
            METAL_MAX_CONTEXT_TOKENS,
            metal_context_ceiling,
        )

        monkeypatch.delenv("VETCH_METAL_MAX_CONTEXT", raising=False)
        assert metal_context_ceiling() == METAL_MAX_CONTEXT_TOKENS
        monkeypatch.setenv("VETCH_METAL_MAX_CONTEXT", "16384")
        assert metal_context_ceiling() == 16384
        for bad in ("nonsense", "-5", "0"):
            monkeypatch.setenv("VETCH_METAL_MAX_CONTEXT", bad)
            assert metal_context_ceiling() == METAL_MAX_CONTEXT_TOKENS, bad

    def test_every_point_fits_the_budget(self) -> None:
        from vetch.calibrate_metal import _grid_design

        for limit in (2048, 4096, 8192, 16384, 32768):
            grid = _grid_design(context_limit=limit)
            assert grid, limit
            for spec in grid:
                assert spec.approx_text_tokens + spec.max_tokens <= limit, (
                    limit, spec,
                )

    def test_default_ceiling_keeps_a_real_prefill_arm(self) -> None:
        from vetch.calibrate_metal import METAL_MAX_CONTEXT_TOKENS, _grid_design
        from vetch.calibrate_v2 import (
            PREFILL_ARM_MAX_OUTPUT_TOKENS,
            PREFILL_ARM_MIN_INPUT_TOKENS,
            PREFILL_ARM_MIN_ROWS,
            PREFILL_ARM_MIN_SPAN_TOKENS,
        )

        grid = _grid_design(context_limit=METAL_MAX_CONTEXT_TOKENS)
        arm = [
            s for s in grid
            if s.approx_text_tokens >= PREFILL_ARM_MIN_INPUT_TOKENS
            and s.max_tokens <= PREFILL_ARM_MAX_OUTPUT_TOKENS
        ]
        assert len(arm) >= PREFILL_ARM_MIN_ROWS
        inputs = [s.approx_text_tokens for s in arm]
        assert max(inputs) - min(inputs) >= PREFILL_ARM_MIN_SPAN_TOKENS
        # And the arm still aggregates repeats into a measurable window.
        assert all(s.requests > 1 for s in arm)

    def test_unfiltered_grid_would_exceed_a_default_ollama_context(self) -> None:
        from vetch.calibrate_metal import _grid_design

        unfiltered = _grid_design(context_limit=None)
        assert max(s.approx_text_tokens for s in unfiltered) == 32768
        # 4096 is Ollama's common default num_ctx: most of the arm would be
        # truncated rather than refused.
        over = [s for s in unfiltered if s.approx_text_tokens + s.max_tokens > 4096]
        assert len(over) >= 5

    def test_context_length_probe_fails_open(self, monkeypatch) -> None:
        from urllib.error import URLError

        import vetch.calibrate_metal as metal

        def boom(*_args, **_kwargs):
            raise URLError("connection refused")

        monkeypatch.setattr(metal.urllib_request, "urlopen", boom)
        monkeypatch.delenv("VETCH_METAL_MAX_CONTEXT", raising=False)
        assert metal._ollama_context_length("http://localhost:11434", "m") is None
        # An unreachable probe must fall back to the ceiling, never abort.
        assert metal.resolve_metal_context_limit("http://localhost:11434", "m") == (
            metal.METAL_MAX_CONTEXT_TOKENS
        )

    def test_model_limit_below_ceiling_wins(self, monkeypatch) -> None:
        import vetch.calibrate_metal as metal

        monkeypatch.setattr(
            metal, "_ollama_context_length", lambda *_a, **_k: 4096,
        )
        monkeypatch.delenv("VETCH_METAL_MAX_CONTEXT", raising=False)
        assert metal.resolve_metal_context_limit("u", "m") == 4096

    def test_generate_pins_num_ctx(self, monkeypatch) -> None:
        import json as _json

        import vetch.calibrate_metal as metal

        captured: dict = {}

        class _Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return b'{"prompt_eval_count": 10, "eval_count": 5}'

        def fake_urlopen(req, timeout=None):
            captured.update(_json.loads(req.data))
            return _Resp()

        monkeypatch.setattr(metal.urllib_request, "urlopen", fake_urlopen)
        metal._ollama_generate(
            "http://localhost:11434", "m", "prompt", max_tokens=8, num_ctx=8192,
        )
        assert captured["options"]["num_ctx"] == 8192
        assert captured["options"]["num_predict"] == 8


class TestPrefillRepeatSizing:
    """Repeats are sized from this machine's measured rate, not an H100 prior.

    `PREFILL_TOKENS_PER_MS = 30` is an H100 figure. Apple Silicon is one to two
    orders of magnitude slower, so inheriting it asks for 64 repeats of a point
    that already fills the 4 s window on its own.
    """

    def test_measured_rate_collapses_the_repeat_count(self) -> None:
        from vetch.calibrate_metal import _grid_design

        inherited = _grid_design(context_limit=8192)
        measured = _grid_design(context_limit=8192, prefill_tokens_per_ms=0.5)
        assert sum(s.requests for s in inherited) > 200
        assert sum(s.requests for s in measured) < 40
        # Same design points either way: only the aggregation depth moves.
        assert [s.approx_text_tokens for s in inherited] == [
            s.approx_text_tokens for s in measured
        ]

    def test_repeats_still_fill_the_window(self) -> None:
        from vetch.calibrate_v2 import (
            PREFILL_MAX_REPEATS,
            PREFILL_WINDOW_TARGET_MS,
            prefill_repeats_for,
        )

        for rate in (0.25, 0.5, 2.0, 30.0):
            for tokens in (1024, 4096):
                repeats = prefill_repeats_for(tokens, rate)
                window = repeats * (tokens / rate)
                if repeats == PREFILL_MAX_REPEATS:
                    # The cap binds before the target on very fast silicon;
                    # that is the cap doing its job, not a sizing failure.
                    continue
                assert window >= PREFILL_WINDOW_TARGET_MS * 0.99, (rate, tokens)

    def test_bad_rates_fall_back_to_the_prior(self) -> None:
        from vetch.calibrate_v2 import prefill_repeats_for

        baseline = prefill_repeats_for(1024)
        for bad in (None, 0.0, -1.0):
            assert prefill_repeats_for(1024, bad) == baseline, bad

    def test_probe_fails_open(self, monkeypatch) -> None:
        import vetch.calibrate_metal as metal

        def boom(*_a, **_k):
            raise RuntimeError("ollama down")

        monkeypatch.setattr(metal, "_ollama_generate", boom)
        probe = metal.probe_prefill("http://localhost:11434", "m")
        assert probe.tokens_per_ms is None
        assert probe.token_scale == 1.0  # falls back to the unscaled generator

    def test_probe_returns_tokens_per_ms(self, monkeypatch) -> None:
        import vetch.calibrate_metal as metal

        clock = iter([1000.0, 1002.0])  # seconds -> 2000 ms elapsed
        monkeypatch.setattr(metal.time, "monotonic", lambda: next(clock))
        monkeypatch.setattr(metal, "_ollama_generate", lambda *a, **k: (1024, 8))
        probe = metal.probe_prefill("http://localhost:11434", "m")
        assert probe.tokens_per_ms is not None
        assert abs(probe.tokens_per_ms - 1024 / 2000.0) < 1e-9
        assert abs(probe.token_ratio - 1.0) < 1e-9


class TestTokenTargetCorrection:
    """A design point must realise the token count it claims.

    Measured on llama3.2 via Ollama, the `1 token ~ 0.75 words` heuristic
    under-delivers by ~22%: a point asking for 1024 tokens realises 803, which
    is below PREFILL_ARM_MIN_INPUT_TOKENS and silently stops being an arm row.
    """

    def test_scale_inverts_the_ratio_with_headroom(self) -> None:
        from vetch.calibrate_metal import TOKEN_TARGET_HEADROOM, PrefillProbe

        probe = PrefillProbe(tokens_per_ms=0.5, token_ratio=0.784)
        assert abs(probe.token_scale - (1 / 0.784) * TOKEN_TARGET_HEADROOM) < 1e-9
        # Correcting to exactly nominal lands inside tokenizer noise of a ">="
        # threshold; 1024 * (1/0.784) realised 1014 on real hardware.
        assert 1024 * probe.token_scale * 0.784 > 1024

    def test_missing_ratio_leaves_the_generator_alone(self) -> None:
        from vetch.calibrate_metal import PrefillProbe

        for ratio in (None, 0.0, -1.0):
            assert PrefillProbe(tokens_per_ms=1.0, token_ratio=ratio).token_scale == 1.0

    def test_scale_lengthens_the_prompt(self) -> None:
        from vetch.calibrate_metal import _unique_prompt

        plain = _unique_prompt(1024, seed=1)
        scaled = _unique_prompt(1024, seed=1, token_scale=1.35)
        assert len(scaled) > len(plain) * 1.2
        # A bad scale must not shrink or empty the prompt.
        for bad in (0.0, -1.0):
            assert _unique_prompt(1024, seed=1, token_scale=bad) == plain


class TestIdleDriftIsFlaggedNotFatal:
    """Idle drift must not destroy a completed run.

    Protocol v2: drift is a measurement-validity signal, and "a drifting record
    stays usable and stays visibly weak". Raising after every measurement had
    been taken discarded the whole run and wrote nothing, while the rejection
    path that records the drift and withholds the active install already
    existed and was simply unreachable for this condition.
    """

    def test_metal_uses_the_protocol_thresholds(self) -> None:
        from vetch.calibrate_metal import IDLE_DRIFT_FLAG_PCT, IDLE_DRIFT_WARN_PCT
        from vetch.calibrate_v2 import (
            IDLE_DRIFT_FLAG_PCT as PROTOCOL_FLAG,
        )
        from vetch.calibrate_v2 import (
            IDLE_DRIFT_WARN_PCT as PROTOCOL_WARN,
        )

        assert IDLE_DRIFT_FLAG_PCT == PROTOCOL_FLAG
        assert IDLE_DRIFT_WARN_PCT == PROTOCOL_WARN

    def test_drift_withholds_the_active_install(self) -> None:
        from vetch.calibrate_metal import _active_calibration_rejection_reasons

        class _Fit:
            invalid_reasons: list[str] = []
            valid = True
            condition_number = 2.0

        reasons = _active_calibration_rejection_reasons(
            fit=_Fit(),  # type: ignore[arg-type]
            power_state={"on_ac_power": True, "low_power_mode": False},
            gpu_power_captured=True,
            idle_drift_pct=101.9,
            samples=11,
        )
        assert any("idle_drift_pct" in r for r in reasons)

    def test_no_drift_leaves_the_record_installable(self) -> None:
        from vetch.calibrate_metal import _active_calibration_rejection_reasons

        class _Fit:
            invalid_reasons: list[str] = []
            valid = True
            condition_number = 2.0

        reasons = _active_calibration_rejection_reasons(
            fit=_Fit(),  # type: ignore[arg-type]
            power_state={"on_ac_power": True, "low_power_mode": False},
            gpu_power_captured=True,
            idle_drift_pct=3.0,
            samples=11,
        )
        assert reasons == []

    def test_drift_is_classified_without_becoming_an_abort(self) -> None:
        from vetch.calibrate_metal import classify_idle_drift

        flagged_pct, flagged = classify_idle_drift(10.0, 12.0)
        warned_pct, warned = classify_idle_drift(10.0, 11.1)
        stable_pct, stable = classify_idle_drift(10.0, 10.3)

        assert flagged == "flag" and flagged_pct == 20.0
        assert warned == "warn" and abs(warned_pct - 11.0) < 1e-9
        assert stable == "ok" and abs(stable_pct - 3.0) < 1e-9


class TestPowerStatePreflight:
    """Refuse a doomed run in the first second, not after the full grid.

    Battery and Low Power Mode already make the finished record
    non-installable, and Low Power Mode throttles the silicon being measured.
    The old code printed both warnings up front and ran anyway.
    """

    def test_battery_and_low_power_are_refused(self) -> None:
        import pytest as _pytest

        from vetch.calibrate_metal import assert_power_state_usable

        for state in (
            {"on_ac_power": False, "low_power_mode": False},
            {"on_ac_power": True, "low_power_mode": True},
            {"on_ac_power": False, "low_power_mode": True},
        ):
            with _pytest.raises(SystemExit) as exc:
                assert_power_state_usable(state)
            assert exc.value.code == 2, state

    def test_good_state_proceeds(self) -> None:
        from vetch.calibrate_metal import assert_power_state_usable

        assert_power_state_usable({"on_ac_power": True, "low_power_mode": False})

    def test_unknown_power_state_fails_open(self) -> None:
        from vetch.calibrate_metal import assert_power_state_usable

        # pmset unreadable is unknown, not bad.
        assert_power_state_usable({"on_ac_power": None, "low_power_mode": False})

    def test_override_proceeds_but_stays_non_installable(self) -> None:
        from vetch.calibrate_metal import (
            _active_calibration_rejection_reasons,
            assert_power_state_usable,
        )

        state = {"on_ac_power": False, "low_power_mode": True}
        assert_power_state_usable(state, allow_degraded=True)  # must not exit

        class _Fit:
            invalid_reasons: list[str] = []
            valid = True
            condition_number = 2.0

        reasons = _active_calibration_rejection_reasons(
            fit=_Fit(),  # type: ignore[arg-type]
            power_state=state, gpu_power_captured=True,
            idle_drift_pct=3.0, samples=11,
        )
        assert any("AC power" in r for r in reasons)
        assert any("low_power_mode" in r for r in reasons)


class TestMetalRecordCarriesTheFitVerdict:
    """Sharing the estimator is not the same as sharing the record.

    The first real Apple Silicon v2 run published `wh_per_1k_output` with no
    `status`, no `null_reasons` and no `published_rung`, and an `energy_basis`
    of nulls that contradicted the coefficient beside it. Both blocks come from
    the fit and both were simply not passed to `commit_calibration`.
    """

    def test_nested_fit_to_dict_supplies_the_three_fields(self) -> None:
        from vetch.calibrate_v2 import fit_nested_models, nested_fit_to_dict

        rows = []
        for i in range(24):
            text, out = ((1024.0 * (1 + i % 4), 8.0) if i % 2 == 0
                         else (32.0, 256.0 * (1 + i % 4)))
            energy = 0.004 * text / 1000 + 0.5 * out / 1000
            rows.append({
                "text_tokens": text, "output_tokens": out, "n_images": 0,
                "energy_wh": energy, "raw_energy_wh": energy + 0.05,
                "duration_ms": 3000.0,
            })
        meta = nested_fit_to_dict(fit_nested_models(rows, idle_watts=100.0))
        for key in ("status", "null_reasons", "published_rung"):
            assert key in meta, key

    def test_invariants_skip_every_axis_check_without_status(self) -> None:
        """Why the omission was silent: an absent status map disables the checks."""
        from vetch.calibration_invariants import record_invariants

        unqualified = {
            "coefficients": {"wh_per_1k_output": 0.0449},  # no status at all
            "provenance": {"samples": 0},
        }
        codes = {v.code for v in record_invariants(unqualified)}
        assert "published_unidentified" not in codes


class TestAdapterDisclosure:
    """`on_ac_power: true` does not mean the adapter is supplying the load.

    Sustained GPU load can exceed a small adapter; the machine then draws the
    difference from the battery while pmset still reports AC Power. Different
    power and thermal regime, shifts as the battery drains, and a plausible
    source of idle drift across a grid.
    """

    @staticmethod
    def _patched(monkeypatch, ps: str, adapter: str):
        import vetch.calibrate_metal as metal

        def fake(cmd, **_k):
            return adapter if "adapter" in cmd else ps

        monkeypatch.setattr(metal.subprocess, "check_output", fake)
        return metal.check_power_state()

    def test_small_adapter_is_recorded(self, monkeypatch) -> None:
        state = self._patched(
            monkeypatch,
            "Now drawing from 'AC Power'\n -InternalBattery-0 85%; AC attached; not charging",
            " Wattage = 45W\n Current = 2240mA\n",
        )
        assert state["adapter_watts"] == 45
        assert state["battery_percent"] == 85
        assert state["battery_charging"] == "not_charging"
        assert state["on_ac_power"] is True

    def test_discharging_on_ac_is_captured(self, monkeypatch) -> None:
        state = self._patched(
            monkeypatch,
            "Now drawing from 'AC Power'\n -InternalBattery-0 61%; discharging",
            " Wattage = 45W\n",
        )
        assert state["on_ac_power"] is True
        assert state["battery_charging"] == "discharging"

    def test_missing_adapter_info_is_not_fatal(self, monkeypatch) -> None:
        state = self._patched(
            monkeypatch, "Now drawing from 'AC Power'\n", "no wattage here\n"
        )
        assert state["adapter_watts"] is None
        assert state["on_ac_power"] is True

    def test_adapter_does_not_block_the_run(self, monkeypatch) -> None:
        """Disclosure, not a gate: a small adapter must not refuse a run."""
        from vetch.calibrate_metal import assert_power_state_usable

        assert_power_state_usable({
            "on_ac_power": True, "low_power_mode": False, "adapter_watts": 45,
        })


class TestIdleBaselinesAreSymmetric:
    """Two idle windows measured differently make their difference meaningless.

    The "after" window had no settle at all, so a figure taken off a workload
    tail was compared with one taken after a pause. Both `idle_drift_pct` and
    the mean that every row's idle subtraction uses inherit that asymmetry.
    """

    def test_window_settles_before_it_samples(self) -> None:
        """Settle, then sample. A window opened before the machine settles
        measures the workload tail rather than idle."""
        from vetch.calibrate_metal import (
            IDLE_SETTLE_S,
            IDLE_WINDOW_S,
            measure_idle_window,
        )

        slept: list[float] = []
        ticks = iter([100.0, 100.0 + IDLE_WINDOW_S])
        start, end = measure_idle_window(
            "", sleep=slept.append, clock=lambda: next(ticks),
        )
        assert slept == [IDLE_SETTLE_S, IDLE_WINDOW_S]
        assert abs((end - start) - IDLE_WINDOW_S * 1000) < 1e-6

    def test_repeated_windows_are_indistinguishable(self) -> None:
        """Two baselines taken through the helper cannot diverge.

        Asymmetry is now impossible by construction -- one helper, both call
        sites -- rather than something a test has to police. This pins the
        helper's own symmetry: calling it twice must sleep and sample the same.
        """
        from vetch.calibrate_metal import IDLE_WINDOW_S, measure_idle_window

        def one_window() -> tuple[list[float], float]:
            slept: list[float] = []
            ticks = iter([0.0, IDLE_WINDOW_S])
            start, end = measure_idle_window(
                "", sleep=slept.append, clock=lambda: next(ticks),
            )
            return slept, end - start

        assert one_window() == one_window()


class TestIdleGateUsesEnergyShare:
    """The Apple Silicon gate rejected 3 of 4 runs whose decode coefficients
    agree with each other to 0.5%. Only the run that was actually 19% wrong
    should fail."""

    @staticmethod
    def _reasons(idle_share_pct, idle_drift_pct=40.0):
        from vetch.calibrate_metal import _active_calibration_rejection_reasons

        class _Fit:
            invalid_reasons: list[str] = []
            valid = True
            condition_number = 2.0

        return _active_calibration_rejection_reasons(
            fit=_Fit(),  # type: ignore[arg-type]
            power_state={"on_ac_power": True, "low_power_mode": False},
            gpu_power_captured=True,
            idle_drift_pct=idle_drift_pct,
            samples=11,
            idle_share_pct=idle_share_pct,
        )

    def test_high_drift_but_low_energy_share_is_installable(self) -> None:
        # Observed run: 21.7% drift, 2.3% energy share, coefficient within 0.04%
        # of the cleanest run. The old gate rejected it.
        assert self._reasons(2.3, idle_drift_pct=21.7) == []

    def test_high_energy_share_is_rejected(self) -> None:
        # Observed run: 74.5% drift, 40.2% energy share, coefficient 19% wrong.
        reasons = self._reasons(40.2, idle_drift_pct=74.5)
        assert any("typical row's energy" in r for r in reasons)

    def test_falls_back_to_the_percentage_without_a_run_table(self) -> None:
        reasons = self._reasons(None, idle_drift_pct=40.0)
        assert any("idle_drift_pct" in r and "no run table" in r for r in reasons)
        assert self._reasons(None, idle_drift_pct=3.0) == []


class TestFitRecordBlocks:
    """The writer must carry the fit's verdict onto the record.

    Reverting `commit_calibration(..., **fit_record_blocks(fit))` back to a bare
    call must fail here -- that regression shipped once and was invisible because
    the invariant checks skip every per-axis test when the status map is absent.
    """

    @staticmethod
    def _fit():
        from vetch.calibrate_v2 import fit_nested_models

        rows = []
        for i in range(24):
            text, out = ((1024.0 * (1 + i % 4), 8.0) if i % 2 == 0
                         else (32.0, 256.0 * (1 + i % 4)))
            energy = 0.004 * text / 1000 + 0.5 * out / 1000
            rows.append({
                "text_tokens": text, "output_tokens": out, "n_images": 0,
                "energy_wh": energy, "raw_energy_wh": energy + 0.05,
                "duration_ms": 3000.0,
            })
        return fit_nested_models(rows, idle_watts=100.0)

    def test_blocks_carry_status_reasons_rung_and_basis(self) -> None:
        from vetch.calibrate_metal import fit_record_blocks

        blocks = fit_record_blocks(self._fit())
        assert set(blocks) == {"coefficient_meta", "energy_basis"}
        for key in ("status", "null_reasons", "published_rung"):
            assert key in blocks["coefficient_meta"], key
        assert "marginal" in blocks["energy_basis"]

    def test_energy_basis_agrees_with_the_published_coefficient(self) -> None:
        """The record must not contradict itself about its own marginal basis."""
        from vetch.calibrate_metal import fit_record_blocks

        fit = self._fit()
        blocks = fit_record_blocks(fit)
        assert blocks["energy_basis"]["marginal"]["wh_per_1k_output"] == (
            fit.wh_per_1k_output
        )

    def test_writer_hands_both_blocks_to_commit(self, monkeypatch) -> None:
        """Behavioural: intercept commit_calibration and inspect what it gets."""
        import vetch.calibration_store as store
        from vetch.calibrate_metal import fit_record_blocks

        seen: dict = {}
        monkeypatch.setattr(
            store, "commit_calibration",
            lambda *a, **kw: seen.update(kw) or Path("/tmp/x.json"),
        )
        store.commit_calibration(None, None, {}, **fit_record_blocks(self._fit()))
        assert "coefficient_meta" in seen and "energy_basis" in seen
        assert seen["coefficient_meta"]["published_rung"]


# ---------------------------------------------------------------------------
# Sampler selection (IOReport counters vs powermetrics) — no hardware
# ---------------------------------------------------------------------------

class TestResolveSampler:
    def test_auto_prefers_ioreport_when_readable(self) -> None:
        from vetch.calibrate_metal import resolve_sampler
        with patch("vetch.ioreport.available", return_value=True):
            assert resolve_sampler("auto") == "ioreport"

    def test_auto_falls_back_to_powermetrics(self) -> None:
        from vetch.calibrate_metal import resolve_sampler
        with patch("vetch.ioreport.available", return_value=False):
            assert resolve_sampler("auto") == "powermetrics"

    def test_explicit_ioreport_fails_rather_than_switching_basis(self) -> None:
        import pytest

        from vetch.calibrate_metal import resolve_sampler
        with patch("vetch.ioreport.available", return_value=False):
            with pytest.raises(RuntimeError, match="--sampler powermetrics"):
                resolve_sampler("ioreport")

    def test_explicit_powermetrics_never_probes_ioreport(self) -> None:
        from vetch.calibrate_metal import resolve_sampler
        with patch("vetch.ioreport.available", side_effect=AssertionError("probed")):
            assert resolve_sampler("powermetrics") == "powermetrics"

    def test_unknown_sampler_rejected(self) -> None:
        import pytest

        from vetch.calibrate_metal import resolve_sampler
        with pytest.raises(ValueError):
            resolve_sampler("nvml")


class TestSamplerBasisConsistency:
    """The source string decides the admission floor and must agree with the basis."""

    def _rows(self) -> list[dict[str, Any]]:
        # One short prefill-style row under the 2 s sampling floor, one long one.
        return [
            {"energy_wh": 0.0004, "raw_energy_wh": 0.0005, "duration_ms": 900.0,
             "output_tokens": 8, "text_tokens": 2048, "n_images": 0, "requests": 1},
            {"energy_wh": 0.02, "raw_energy_wh": 0.021, "duration_ms": 30000.0,
             "output_tokens": 1024, "text_tokens": 32, "n_images": 0, "requests": 1},
        ]

    def test_counter_source_keeps_short_rows(self) -> None:
        from vetch.calibrate_metal import ENERGY_SOURCE_BY_SAMPLER
        from vetch.calibrate_v2 import admit_rows
        admitted, _, _, under_floor = admit_rows(
            self._rows(), energy_source=ENERGY_SOURCE_BY_SAMPLER["ioreport"]
        )
        assert len(admitted) == 2
        assert under_floor == 0

    def test_powermetrics_source_applies_floor(self) -> None:
        from vetch.calibrate_metal import ENERGY_SOURCE_BY_SAMPLER
        from vetch.calibrate_v2 import admit_rows
        admitted, _, _, under_floor = admit_rows(
            self._rows(), energy_source=ENERGY_SOURCE_BY_SAMPLER["powermetrics"]
        )
        assert len(admitted) == 1
        assert under_floor == 1

    def test_source_and_basis_pass_the_invariant_check(self) -> None:
        from vetch.calibrate_metal import (
            ENERGY_SOURCE_BY_SAMPLER,
            MEASUREMENT_BASIS_BY_SAMPLER,
            SAMPLER_CHOICES,
        )
        from vetch.calibration_invariants import _check_measurement_basis
        for sampler in SAMPLER_CHOICES:
            if sampler == "auto":
                continue
            out: list[Any] = []
            _check_measurement_basis(
                {"provenance": {
                    "energy_source": ENERGY_SOURCE_BY_SAMPLER[sampler],
                    "measurement_basis": MEASUREMENT_BASIS_BY_SAMPLER[sampler],
                }},
                out,
            )
            assert out == [], f"{sampler}: {out}"


class TestCalibrateSamplerWiring:
    def test_ioreport_path_skips_root_and_records_counter_provenance(self) -> None:
        """Drive calibrate_apple_silicon to the sampler choice with everything
        after it stubbed out, and check root is not required on the IOReport path."""
        import vetch.calibrate_metal as cm

        with (
            patch.object(cm, "assert_apple_silicon"),
            patch.object(cm, "_require_numpy_for_calibration"),
            patch("vetch.ioreport.available", return_value=True),
            patch.object(cm, "assert_root", side_effect=AssertionError("root demanded")),
            patch.object(cm, "get_hardware_info", side_effect=RuntimeError("stop here")),
        ):
            try:
                cm.calibrate_apple_silicon(model="m", precision="apple-native")
            except RuntimeError as e:
                assert str(e) == "stop here"

    def test_powermetrics_path_still_requires_root(self) -> None:
        import vetch.calibrate_metal as cm

        with (
            patch.object(cm, "assert_apple_silicon"),
            patch.object(cm, "_require_numpy_for_calibration"),
            patch.object(cm, "assert_root", side_effect=PermissionError("need sudo")),
        ):
            try:
                cm.calibrate_apple_silicon(
                    model="m", precision="apple-native", sampler="powermetrics"
                )
            except PermissionError as e:
                assert str(e) == "need sudo"
            else:
                raise AssertionError("assert_root was not called")

    def test_cli_passes_sampler_through(self) -> None:
        from vetch import cli

        captured: dict[str, Any] = {}

        def fake(**kwargs: Any) -> Any:
            captured.update(kwargs)
            raise SystemExit(0)

        with (
            patch("vetch.calibrate_metal.calibrate_apple_silicon", fake),
            patch.object(cli.sys, "argv", [
                "vetch", "calibrate-apple-silicon", "--model", "m",
                "--precision", "apple-native", "--sampler", "powermetrics",
            ]),
        ):
            try:
                cli.main()
            except SystemExit:
                pass
        assert captured.get("sampler") == "powermetrics"
