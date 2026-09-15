"""Apple Silicon energy counters through IOReport.

The ``Energy Model`` channel group of Apple's private ``IOReport`` framework
exposes a monotonic energy accumulator per SoC subsystem (CPU, GPU, ANE, DRAM
and others). Reading it needs no root, and energy over a window is the
difference of two counter reads rather than a trapezoid over sampled power.
That puts Apple Silicon on the same counter basis ``calibrate_cuda`` prefers
when ``nvmlDeviceGetTotalEnergyConsumption`` is available: no sampling gaps to
bridge and no duration floor on short rows.

``IOReportEnergyMonitor`` has the same ``integrate`` / ``mean_watts`` interface
as the ``powermetrics`` monitor in ``calibrate_metal``, so the calibration loop
does not care which one it is holding. The framework is private, so
``available()`` is the gate and ``powermetrics`` stays as the fallback.

Channel names are per chip generation. On an M1 Pro the totals are
``CPU Energy`` (mJ), ``GPU Energy`` (nJ), ``ANE0`` (mJ) and ``DRAM0`` (mJ); the
unit is carried on the channel and read per channel, never assumed.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import platform
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

DOMAINS: tuple[str, ...] = ("cpu", "gpu", "ane", "dram")

#: Combined CPU + GPU + ANE, the domain ``powermetrics`` reports as
#: ``Combined Power`` and the one existing ``apple_soc`` records integrate.
SOC_DOMAINS: tuple[str, ...] = ("cpu", "gpu", "ane")

_UNIT_TO_JOULES: Mapping[str, float] = {
    "nJ": 1e-9,
    "uJ": 1e-6,
    "mJ": 1e-3,
    "J": 1.0,
}

_CF_STRING_ENCODING_UTF8 = 0x08000100
_IOREPORT_PATH = "/usr/lib/libIOReport.dylib"
_ENERGY_GROUP = "Energy Model"


def classify_channel(name: str) -> str | None:
    """Map an ``Energy Model`` channel name to a domain, or ``None`` to ignore it.

    Only the per-subsystem totals are kept. The per-cluster and per-core
    counters (``PACC0_CPU1``, ``GPU0``, ``GPU SRAM0``) are components of the
    totals and would double count.
    """
    if name.startswith("CPU Energy"):
        return "cpu"
    if name.startswith("GPU Energy"):
        return "gpu"
    if name.startswith("ANE"):
        return "ane"
    if name.startswith("DRAM"):
        return "dram"
    return None


@dataclass
class EnergySample:
    """Cumulative energy per domain, in joules, since the monitor started."""

    mono_ms: float
    joules: dict[str, float]


@dataclass
class IOReportInfo:
    sample_interval_ms: int = 100
    channels: dict[str, str] = field(default_factory=dict)  # channel name -> unit label
    domains_present: list[str] = field(default_factory=list)

    @property
    def gpu_energy_captured(self) -> bool:
        return "gpu" in self.domains_present

    #: Same flag under the name the powermetrics ``MonitorInfo`` uses, so the
    #: calibration loop and the record can treat both monitors alike.
    @property
    def gpu_power_captured(self) -> bool:
        return self.gpu_energy_captured


class _Binding:
    """The handful of IOReport and CoreFoundation calls the monitor needs."""

    def __init__(self) -> None:
        cf_path = ctypes.util.find_library("CoreFoundation")
        if cf_path is None:
            raise OSError("CoreFoundation not found")
        self.cf = ctypes.CDLL(cf_path)
        self.ior = ctypes.CDLL(_IOREPORT_PATH)
        ref = ctypes.c_void_p
        sigs: list[tuple[Any, str, Any, list[Any]]] = [
            (self.ior, "IOReportCopyChannelsInGroup", ref,
             [ref, ref, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]),
            (self.ior, "IOReportCreateSubscription", ref,
             [ctypes.c_void_p, ref, ctypes.POINTER(ref), ctypes.c_uint64, ref]),
            (self.ior, "IOReportCreateSamples", ref, [ref, ref, ref]),
            (self.ior, "IOReportCreateSamplesDelta", ref, [ref, ref, ref]),
            (self.ior, "IOReportChannelGetChannelName", ref, [ref]),
            (self.ior, "IOReportChannelGetUnitLabel", ref, [ref]),
            (self.ior, "IOReportSimpleGetIntegerValue", ctypes.c_int64, [ref, ctypes.c_int32]),
            (self.cf, "CFStringCreateWithCString", ref, [ref, ctypes.c_char_p, ctypes.c_uint32]),
            (self.cf, "CFStringGetCString", ctypes.c_bool,
             [ref, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]),
            (self.cf, "CFDictionaryGetValue", ref, [ref, ref]),
            (self.cf, "CFArrayGetCount", ctypes.c_long, [ref]),
            (self.cf, "CFArrayGetValueAtIndex", ref, [ref, ctypes.c_long]),
            (self.cf, "CFRelease", None, [ref]),
        ]
        for lib, name, restype, argtypes in sigs:
            fn = getattr(lib, name)
            fn.restype = restype
            fn.argtypes = argtypes
        self._channels_key = self.cfstr("IOReportChannels")
        group = self.cfstr(_ENERGY_GROUP)
        self.channels = self.ior.IOReportCopyChannelsInGroup(group, None, 0, 0, 0)
        if not self.channels:
            raise OSError(f"IOReport group {_ENERGY_GROUP!r} has no channels")
        self.sub_channels = ctypes.c_void_p()
        self.subscription = self.ior.IOReportCreateSubscription(
            None, self.channels, ctypes.byref(self.sub_channels), 0, None
        )
        if not self.subscription:
            raise OSError("IOReportCreateSubscription failed")

    def cfstr(self, s: str) -> Any:
        return self.cf.CFStringCreateWithCString(None, s.encode(), _CF_STRING_ENCODING_UTF8)

    def pystr(self, ref: Any) -> str:
        if not ref:
            return ""
        buf = ctypes.create_string_buffer(256)
        self.cf.CFStringGetCString(ref, buf, 256, _CF_STRING_ENCODING_UTF8)
        return buf.value.decode(errors="replace")

    def sample(self) -> Any:
        return self.ior.IOReportCreateSamples(self.subscription, self.sub_channels, None)

    def delta(self, before: Any, after: Any) -> dict[str, tuple[int, str]]:
        """Return ``{channel name: (integer value, unit label)}`` for ``after - before``."""
        d = self.ior.IOReportCreateSamplesDelta(before, after, None)
        if not d:
            return {}
        try:
            arr = self.cf.CFDictionaryGetValue(d, self._channels_key)
            out: dict[str, tuple[int, str]] = {}
            for i in range(self.cf.CFArrayGetCount(arr)):
                ch = self.cf.CFArrayGetValueAtIndex(arr, i)
                name = self.pystr(self.ior.IOReportChannelGetChannelName(ch))
                unit = self.pystr(self.ior.IOReportChannelGetUnitLabel(ch))
                out[name] = (int(self.ior.IOReportSimpleGetIntegerValue(ch, 0)), unit)
            return out
        finally:
            self.cf.CFRelease(d)

    def release(self, ref: Any) -> None:
        if ref:
            self.cf.CFRelease(ref)


class _Reader:
    """A sampler that returns per-domain energy deltas in joules.

    Wraps ``_Binding`` so the monitor sees only ``read_delta_joules()`` and the
    tests can hand it a fake with the same shape.
    """

    def __init__(self, binding: _Binding) -> None:
        self._b = binding
        self._prev = binding.sample()
        self.channels: dict[str, str] = {}

    def read_delta_joules(self) -> dict[str, float]:
        cur = self._b.sample()
        raw = self._b.delta(self._prev, cur)
        self._b.release(self._prev)
        self._prev = cur
        out = dict.fromkeys(DOMAINS, 0.0)
        for name, (value, unit) in raw.items():
            domain = classify_channel(name)
            if domain is None:
                continue
            scale = _UNIT_TO_JOULES.get(unit)
            if scale is None:
                continue
            self.channels.setdefault(name, unit)
            out[domain] += value * scale
        return out

    def close(self) -> None:
        self._b.release(self._prev)


def available() -> bool:
    """True when this process can read the ``Energy Model`` counters.

    False off Apple Silicon, off macOS, and whenever the private framework or
    the channel group is missing, so the caller falls back to ``powermetrics``.
    """
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    try:
        _Binding()
    except (OSError, AttributeError):
        return False
    return True


class IOReportEnergyMonitor:
    """Context manager that accumulates IOReport energy counters in the background.

    Every ``sample_interval_ms`` the reader thread takes a counter delta and
    appends the running cumulative energy per domain. ``integrate`` interpolates
    that cumulative series at the window edges, so a window's energy is exact
    to the counter cadence and does not depend on where the samples fell
    relative to the window.
    """

    def __init__(
        self,
        sample_interval_ms: int = 100,
        *,
        reader_factory: Callable[[], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._interval_ms = sample_interval_ms
        self._reader_factory = reader_factory or (lambda: _Reader(_Binding()))
        self._clock = clock
        self._samples: list[EnergySample] = []
        self._cumulative = dict.fromkeys(DOMAINS, 0.0)
        self._reader: Any = None
        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self.info = IOReportInfo(sample_interval_ms=sample_interval_ms)

    # -- sampling ------------------------------------------------------------

    def _tick(self) -> None:
        """Take one counter delta and append the new cumulative sample."""
        delta = self._reader.read_delta_joules()
        now_ms = self._clock() * 1000
        for domain in DOMAINS:
            self._cumulative[domain] += delta.get(domain, 0.0)
        self._samples.append(EnergySample(mono_ms=now_ms, joules=dict(self._cumulative)))

    def _collect_loop(self) -> None:
        assert self._stop_event is not None
        interval_s = self._interval_ms / 1000.0
        while not self._stop_event.wait(interval_s):
            self._tick()
        self._tick()  # close the last partial interval so the final window is covered

    def __enter__(self) -> IOReportEnergyMonitor:
        self._reader = self._reader_factory()
        self._samples.append(
            EnergySample(mono_ms=self._clock() * 1000, joules=dict(self._cumulative))
        )
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        if self._reader is not None:
            self.info.channels = dict(getattr(self._reader, "channels", {}))
            self.info.domains_present = [
                d for d in DOMAINS if any(classify_channel(n) == d for n in self.info.channels)
            ]
            close = getattr(self._reader, "close", None)
            if close is not None:
                close()

    # -- energy over a window ------------------------------------------------

    def _cumulative_at(self, t_ms: float, domains: tuple[str, ...]) -> float | None:
        """Cumulative joules over ``domains`` at ``t_ms``, interpolated between samples.

        Returns ``None`` when ``t_ms`` lies outside the sampled range, so a
        window the monitor never covered reads as zero rather than as a guess.
        """
        samples = self._samples
        if not samples or t_ms < samples[0].mono_ms or t_ms > samples[-1].mono_ms:
            return None
        lo, hi = 0, len(samples) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if samples[mid].mono_ms <= t_ms:
                lo = mid
            else:
                hi = mid
        s0, s1 = samples[lo], samples[hi]
        e0 = sum(s0.joules[d] for d in domains)
        e1 = sum(s1.joules[d] for d in domains)
        span = s1.mono_ms - s0.mono_ms
        if span <= 0:
            return e1
        frac = (t_ms - s0.mono_ms) / span
        return e0 + frac * (e1 - e0)

    def integrate(
        self,
        t_start_mono_ms: float,
        t_end_mono_ms: float,
        domains: tuple[str, ...] = SOC_DOMAINS,
    ) -> float:
        """Energy in Wh over ``[t_start, t_end]`` for the given domains.

        Defaults to CPU + GPU + ANE, the ``apple_soc`` domain existing records
        use. Pass ``("gpu",)`` for the GPU-only axis.
        """
        if t_end_mono_ms <= t_start_mono_ms:
            return 0.0
        e0 = self._cumulative_at(t_start_mono_ms, domains)
        e1 = self._cumulative_at(t_end_mono_ms, domains)
        if e0 is None or e1 is None:
            return 0.0
        return max(0.0, e1 - e0) / 3600.0

    def integrate_by_domain(
        self, t_start_mono_ms: float, t_end_mono_ms: float
    ) -> dict[str, float]:
        """Energy in Wh over the window for every domain separately."""
        return {d: self.integrate(t_start_mono_ms, t_end_mono_ms, (d,)) for d in DOMAINS}

    def mean_watts(
        self,
        t_start_mono_ms: float,
        t_end_mono_ms: float,
        domains: tuple[str, ...] = SOC_DOMAINS,
    ) -> float:
        """Mean power over the window, from the energy difference."""
        duration_s = (t_end_mono_ms - t_start_mono_ms) / 1000.0
        if duration_s <= 0:
            return 0.0
        return self.integrate(t_start_mono_ms, t_end_mono_ms, domains) * 3600.0 / duration_s
