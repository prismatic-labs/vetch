"""Tests for the IOReport energy-counter monitor (vetch.ioreport).

The private framework is never touched here. A fake reader hands the monitor
per-domain deltas, and a fake clock places them, so the cumulative series and
the window interpolation are tested exactly, on any platform.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from vetch.ioreport import (
    DOMAINS,
    SOC_DOMAINS,
    EnergySample,
    IOReportEnergyMonitor,
    available,
    classify_channel,
)

# ---------------------------------------------------------------------------
# Channel classification
# ---------------------------------------------------------------------------


class TestClassifyChannel:
    @pytest.mark.parametrize(
        ("name", "domain"),
        [
            ("CPU Energy", "cpu"),
            ("GPU Energy", "gpu"),
            ("ANE0", "ane"),
            ("ANE Energy", "ane"),
            ("DRAM0", "dram"),
        ],
    )
    def test_totals_map_to_domains(self, name: str, domain: str) -> None:
        assert classify_channel(name) == domain

    @pytest.mark.parametrize(
        "name",
        ["PACC0_CPU1", "EACC_CPM", "GPU0", "GPU SRAM0", "ISP0", "PCIe Port 0 Energy"],
    )
    def test_components_are_ignored_to_avoid_double_counting(self, name: str) -> None:
        assert classify_channel(name) is None


# ---------------------------------------------------------------------------
# Monitor: cumulative series and window interpolation, no hardware
# ---------------------------------------------------------------------------


class _FakeReader:
    """Returns a scripted sequence of per-domain deltas (joules)."""

    def __init__(self, deltas: list[dict[str, float]]) -> None:
        self._deltas = list(deltas)
        self.channels = {"CPU Energy": "mJ", "GPU Energy": "nJ"}
        self.closed = False

    def read_delta_joules(self) -> dict[str, float]:
        if self._deltas:
            return self._deltas.pop(0)
        return dict.fromkeys(DOMAINS, 0.0)

    def close(self) -> None:
        self.closed = True


def _monitor_with_samples(samples: list[EnergySample]) -> IOReportEnergyMonitor:
    m = IOReportEnergyMonitor.__new__(IOReportEnergyMonitor)
    m._samples = samples
    return m


def _sample(t_ms: float, **joules: float) -> EnergySample:
    j = dict.fromkeys(DOMAINS, 0.0)
    j.update(joules)
    return EnergySample(mono_ms=t_ms, joules=j)


class TestIntegrate:
    def test_constant_power_over_exact_sample_edges(self) -> None:
        # 10 J per 100 ms on the CPU domain = 100 W
        m = _monitor_with_samples([_sample(t, cpu=t / 10.0) for t in (0.0, 100.0, 200.0, 300.0)])
        wh = m.integrate(100.0, 300.0)
        assert wh == pytest.approx(20.0 / 3600.0)
        assert m.mean_watts(100.0, 300.0) == pytest.approx(100.0)

    def test_window_edges_between_samples_are_interpolated(self) -> None:
        m = _monitor_with_samples([_sample(t, cpu=t / 10.0) for t in (0.0, 100.0, 200.0, 300.0)])
        # [50, 250] ms at 100 W = 20 J
        assert m.integrate(50.0, 250.0) == pytest.approx(20.0 / 3600.0)

    def test_result_does_not_depend_on_sample_phase(self) -> None:
        # Same underlying 100 W, samples shifted by 37 ms: the window energy is
        # unchanged, which is what the counter basis buys over a trapezoid.
        a = _monitor_with_samples([_sample(t, cpu=t / 10.0) for t in (0.0, 100.0, 200.0, 300.0)])
        b = _monitor_with_samples(
            [_sample(t, cpu=t / 10.0) for t in (37.0, 137.0, 237.0, 337.0)]
        )
        assert a.integrate(100.0, 250.0) == pytest.approx(b.integrate(100.0, 250.0))

    def test_default_domains_are_cpu_gpu_ane_not_dram(self) -> None:
        m = _monitor_with_samples([
            _sample(0.0),
            _sample(1000.0, cpu=1.0, gpu=2.0, ane=3.0, dram=100.0),
        ])
        assert SOC_DOMAINS == ("cpu", "gpu", "ane")
        assert m.integrate(0.0, 1000.0) == pytest.approx(6.0 / 3600.0)
        assert m.integrate(0.0, 1000.0, ("gpu",)) == pytest.approx(2.0 / 3600.0)
        by_domain = m.integrate_by_domain(0.0, 1000.0)
        assert by_domain["dram"] == pytest.approx(100.0 / 3600.0)
        assert set(by_domain) == set(DOMAINS)

    def test_window_outside_sampled_range_reads_zero_not_a_guess(self) -> None:
        m = _monitor_with_samples([_sample(0.0), _sample(1000.0, cpu=50.0)])
        assert m.integrate(1500.0, 2500.0) == 0.0
        assert m.integrate(-500.0, 500.0) == 0.0

    def test_empty_or_inverted_window_is_zero(self) -> None:
        m = _monitor_with_samples([_sample(0.0), _sample(1000.0, cpu=50.0)])
        assert m.integrate(500.0, 500.0) == 0.0
        assert m.integrate(800.0, 200.0) == 0.0
        assert m.mean_watts(800.0, 200.0) == 0.0

    def test_no_samples_is_zero(self) -> None:
        assert _monitor_with_samples([]).integrate(0.0, 1000.0) == 0.0


class TestMonitorLifecycle:
    def test_thread_accumulates_deltas_and_reports_domains(self) -> None:
        reader = _FakeReader([
            {"cpu": 1.0, "gpu": 0.5},
            {"cpu": 1.0, "gpu": 0.5},
            {"cpu": 1.0, "gpu": 0.5},
        ])
        with IOReportEnergyMonitor(
            sample_interval_ms=10, reader_factory=lambda: reader
        ) as m:
            import time

            time.sleep(0.15)
        assert len(m._samples) >= 3
        last = m._samples[-1].joules
        assert last["cpu"] == pytest.approx(3.0)
        assert last["gpu"] == pytest.approx(1.5)
        assert m.info.channels == {"CPU Energy": "mJ", "GPU Energy": "nJ"}
        assert m.info.domains_present == ["cpu", "gpu"]
        assert m.info.gpu_energy_captured is True
        assert reader.closed is True

    def test_first_sample_is_taken_at_enter(self) -> None:
        clock_values = iter([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        reader = _FakeReader([])
        m = IOReportEnergyMonitor(
            sample_interval_ms=1000, reader_factory=lambda: reader,
            clock=lambda: next(clock_values),
        )
        with m:
            pass
        assert m._samples[0].mono_ms == pytest.approx(1000.0)
        assert all(v == 0.0 for v in m._samples[0].joules.values())


# ---------------------------------------------------------------------------
# Availability gate
# ---------------------------------------------------------------------------


class TestAvailable:
    def test_false_off_darwin(self) -> None:
        with patch("vetch.ioreport.platform.system", return_value="Linux"):
            assert available() is False

    def test_false_on_intel_mac(self) -> None:
        with (
            patch("vetch.ioreport.platform.system", return_value="Darwin"),
            patch("vetch.ioreport.platform.machine", return_value="x86_64"),
        ):
            assert available() is False

    def test_false_when_binding_fails(self) -> None:
        def boom() -> Any:
            raise OSError("no dylib")

        with (
            patch("vetch.ioreport.platform.system", return_value="Darwin"),
            patch("vetch.ioreport.platform.machine", return_value="arm64"),
            patch("vetch.ioreport._Binding", side_effect=boom),
        ):
            assert available() is False


class TestRunTablePassthrough:
    def test_per_domain_energy_survives_into_the_record(self) -> None:
        """The stored run table is built from a key list; the split must be on it."""
        import inspect

        from vetch.calibration_store import measurement_provenance_core

        row = {
            "n_images": 0, "text_tokens": 1000, "output_tokens": 8, "requests": 3,
            "energy_wh": 0.01, "raw_energy_wh": 0.011, "duration_ms": 5000.0,
            "replicate": False,
            "raw_energy_by_domain_wh": {"cpu": 0.002, "gpu": 0.008, "ane": 0.0, "dram": 0.001},
        }
        params = inspect.signature(measurement_provenance_core).parameters
        kwargs: dict[str, Any] = {}
        for name, p in params.items():
            if p.default is not inspect.Parameter.empty:
                continue
            kwargs[name] = {
                "samples": 1, "energy_source": "ioreport_energy_counter",
                "measurement_basis": "ioreport_soc_energy", "energy_domain": "apple_soc",
                "energy_domain_includes": ["cpu", "gpu", "ane"], "energy_domain_excludes": [],
                "idle_watts_before": 0.5, "idle_watts_after": 0.5, "idle_drift_pct": 0.0,
                "run_records": [row], "fit": _FakeFit(),
            }.get(name)
        prov = measurement_provenance_core(**kwargs)
        stored = prov["raw_run_table"][0]
        assert stored["raw_energy_by_domain_wh"] == row["raw_energy_by_domain_wh"]
        without = dict(row)
        del without["raw_energy_by_domain_wh"]
        kwargs["run_records"] = [without]
        stored = measurement_provenance_core(**kwargs)["raw_run_table"][0]
        assert "raw_energy_by_domain_wh" not in stored


class _FakeFit:
    """Enough of a fit for measurement_provenance_core to serialise."""

    r2 = 0.99
    condition_number = 3.0
    input_ci95 = output_ci95 = image_ci95 = intercept_ci95 = None
    residuals_structured = False

    def __getattr__(self, name: str) -> Any:
        return None
