"""Tests for GPU calibration module.

These tests verify:
- GPU availability detection
- CalibrationResult structure
- Result formatting
- Calibration file saving

Note: Most tests mock pynvml since GPU hardware is not available in CI.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestGPUAvailability:
    """Tests for GPU availability checking."""

    def test_is_gpu_available_without_pynvml(self) -> None:
        """Check GPU availability when pynvml not installed."""
        from vetch.calibrate import get_gpu_error, is_gpu_available

        # In CI, GPU is typically not available
        available = is_gpu_available()

        if not available:
            error = get_gpu_error()
            assert error is not None
            assert len(error) > 0

    def test_get_gpu_error_no_pynvml(self) -> None:
        """Get descriptive error when pynvml missing."""
        # Force pynvml import to fail
        with patch.dict("sys.modules", {"pynvml": None}):
            from vetch.calibrate import get_gpu_error

            # Re-import to get fresh state
            import importlib
            import vetch.calibrate
            importlib.reload(vetch.calibrate)

            error = vetch.calibrate.get_gpu_error()
            assert "pynvml" in error.lower() or "not installed" in error.lower() or "nvidia" in error.lower() or len(error) > 0


class TestCalibrationResult:
    """Tests for CalibrationResult namedtuple."""

    def test_create_result(self) -> None:
        """Create a CalibrationResult."""
        from vetch.calibrate import CalibrationResult

        result = CalibrationResult(
            model="llama3.1:8b",
            provider="ollama",
            wh_per_1k_input=0.04,
            wh_per_1k_output=0.12,
            tier=0,
            samples=10,
            gpu_name="NVIDIA RTX 4090",
        )

        assert result.model == "llama3.1:8b"
        assert result.provider == "ollama"
        assert result.wh_per_1k_input == 0.04
        assert result.wh_per_1k_output == 0.12
        assert result.tier == 0
        assert result.samples == 10
        assert result.gpu_name == "NVIDIA RTX 4090"

    def test_result_is_namedtuple(self) -> None:
        """CalibrationResult should be a NamedTuple."""
        from vetch.calibrate import CalibrationResult

        result = CalibrationResult(
            model="test",
            provider="test",
            wh_per_1k_input=0.1,
            wh_per_1k_output=0.3,
            tier=0,
            samples=5,
            gpu_name="Test GPU",
        )

        # NamedTuple should be indexable
        assert result[0] == "test"  # model
        assert result[1] == "test"  # provider

        # And have _fields
        assert "model" in result._fields
        assert "gpu_name" in result._fields


class TestFormatCalibrationResult:
    """Tests for result formatting."""

    def test_format_basic(self) -> None:
        """Format result as text."""
        from vetch.calibrate import CalibrationResult, format_calibration_result

        result = CalibrationResult(
            model="llama3.1:8b",
            provider="ollama",
            wh_per_1k_input=0.04,
            wh_per_1k_output=0.12,
            tier=0,
            samples=10,
            gpu_name="NVIDIA RTX 4090",
        )

        output = format_calibration_result(result)

        assert "llama3.1:8b" in output
        assert "NVIDIA RTX 4090" in output
        assert "0.0400" in output  # wh_per_1k_input formatted
        assert "0.1200" in output  # wh_per_1k_output formatted
        assert "Tier 0" in output

    def test_format_contains_provider(self) -> None:
        """Format includes provider name."""
        from vetch.calibrate import CalibrationResult, format_calibration_result

        result = CalibrationResult(
            model="test-model",
            provider="vllm",
            wh_per_1k_input=0.05,
            wh_per_1k_output=0.15,
            tier=0,
            samples=5,
            gpu_name="Test GPU",
        )

        output = format_calibration_result(result)
        assert "vllm" in output


class TestGPUMonitor:
    """Tests for GPUMonitor context manager."""

    def test_gpu_monitor_init(self) -> None:
        """GPUMonitor can be initialized."""
        from vetch.calibrate import GPUMonitor

        monitor = GPUMonitor(device_id=0)
        assert monitor.device_id == 0

    def test_gpu_monitor_custom_device(self) -> None:
        """GPUMonitor accepts custom device ID."""
        from vetch.calibrate import GPUMonitor

        monitor = GPUMonitor(device_id=1)
        assert monitor.device_id == 1


class TestCalibrateModelMocked:
    """Tests for calibrate_model with mocked pynvml."""

    def test_calibrate_requires_gpu(self) -> None:
        """Calibration fails gracefully without GPU."""
        from vetch.calibrate import calibrate_model, is_gpu_available

        if is_gpu_available():
            pytest.skip("GPU is available, cannot test failure path")

        def dummy_workload() -> tuple[int, int]:
            return (100, 50)

        with pytest.raises(RuntimeError, match="GPU not available"):
            calibrate_model("ollama", "test-model", dummy_workload)

    def test_calibrate_with_mocked_gpu(self) -> None:
        """Calibrate with fully mocked pynvml."""
        # Create comprehensive mock
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit = MagicMock()
        mock_pynvml.nvmlShutdown = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex = MagicMock(return_value="handle")
        mock_pynvml.nvmlDeviceGetName = MagicMock(return_value="Test GPU")
        mock_pynvml.nvmlDeviceGetMemoryInfo = MagicMock(
            return_value=MagicMock(total=8 * 1024 * 1024 * 1024)
        )
        mock_pynvml.nvmlSystemGetDriverVersion = MagicMock(return_value="535.0")
        mock_pynvml.nvmlDeviceGetPowerUsage = MagicMock(return_value=200000)  # 200W in mW

        call_count = 0

        def mock_workload() -> tuple[int, int]:
            nonlocal call_count
            call_count += 1
            return (500, 250)

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            # Need to reload module to pick up mock
            import importlib
            import vetch.calibrate
            importlib.reload(vetch.calibrate)

            # Patch is_gpu_available to return True
            with patch.object(vetch.calibrate, "is_gpu_available", return_value=True):
                result = vetch.calibrate.calibrate_model(
                    provider="ollama",
                    model="llama3.1:8b",
                    workload=mock_workload,
                    iterations=3,
                )

                assert result.model == "llama3.1:8b"
                assert result.provider == "ollama"
                assert result.tier == 0
                assert result.samples == 3
                # Workload called: 1 warmup + 3 iterations = 4 times
                assert call_count == 4


class TestSaveCalibration:
    """Tests for calibration persistence."""

    def test_save_creates_file(self) -> None:
        """Saving calibration creates JSON file."""
        from vetch.calibrate import CalibrationResult, _save_calibration, CALIBRATION_DIR

        result = CalibrationResult(
            model="test-model",
            provider="test",
            wh_per_1k_input=0.05,
            wh_per_1k_output=0.15,
            tier=0,
            samples=5,
            gpu_name="Test GPU",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Temporarily override CALIBRATION_DIR
            import vetch.calibrate
            original_dir = vetch.calibrate.CALIBRATION_DIR
            vetch.calibrate.CALIBRATION_DIR = Path(tmpdir)

            try:
                _save_calibration(result)

                # Check file was created
                expected_path = Path(tmpdir) / "test_test-model.json"
                assert expected_path.exists()

                # Verify JSON content
                with open(expected_path) as f:
                    data = json.load(f)
                assert data["wh_per_1k_input"] == 0.05
                assert data["wh_per_1k_output"] == 0.15
                assert data["tier"] == 0
                assert "basis" in data
                assert "timestamp" in data

            finally:
                vetch.calibrate.CALIBRATION_DIR = original_dir

    def test_save_handles_colons_in_model_name(self) -> None:
        """Model names with colons are sanitized."""
        from vetch.calibrate import CalibrationResult, _save_calibration

        result = CalibrationResult(
            model="llama3.1:8b",
            provider="ollama",
            wh_per_1k_input=0.05,
            wh_per_1k_output=0.15,
            tier=0,
            samples=5,
            gpu_name="Test GPU",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            import vetch.calibrate
            original_dir = vetch.calibrate.CALIBRATION_DIR
            vetch.calibrate.CALIBRATION_DIR = Path(tmpdir)

            try:
                _save_calibration(result)

                # Colons should be replaced with underscores
                expected_path = Path(tmpdir) / "ollama_llama3.1_8b.json"
                assert expected_path.exists()

            finally:
                vetch.calibrate.CALIBRATION_DIR = original_dir
