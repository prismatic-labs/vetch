#!/usr/bin/env python3
"""Measure how stable this machine's idle SoC power actually is.

Calibration subtracts an idle baseline sampled in two short windows, one at each
end of a run, and gates the record on how far those two windows disagree. That
gate assumes idle is a stable property of the machine. On a laptop it may not
be: powermetrics reports SoC-wide power, so "idle" includes whatever else is
running, and the absolute floor is low enough (~1 W) that a few hundred
milliwatts of background activity is tens of percent.

This samples idle with no workload at all and reports the distribution, which is
what the drift gate is implicitly claiming to know. Raw samples are written to a
file so a run is not lost to terminal scrollback.

    sudo python3 scripts/measure_idle_floor.py --minutes 5

Requires root: powermetrics reads Apple PMU counters.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Parse with the calibration's own parser, and the same samplers, so this
# measures the identical quantity the idle baseline does. A hand-rolled regex
# here would be a second implementation to keep in step -- and the obvious one
# ("take the last field") picks up the unit rather than the number.
from vetch.calibrate_metal import (  # noqa: E402
    MonitorInfo,
    _parse_combined_power_watts,
)


def sample(minutes: float, interval_ms: int) -> list[float]:
    """Return idle SoC power samples in watts."""
    count = max(1, int(minutes * 60 * 1000 / interval_ms))
    samplers = ",".join(MonitorInfo().samplers)
    print(f"Sampling idle for ~{minutes:g} min "
          f"({count} samples at {interval_ms} ms, samplers: {samplers}). "
          "Do not touch the machine.")
    proc = subprocess.run(
        ["powermetrics", "--samplers", samplers, "-n", str(count),
         "-i", str(interval_ms)],
        capture_output=True, text=True, check=False,
    )
    values = [w for line in proc.stdout.splitlines()
              if (w := _parse_combined_power_watts(line)) is not None]
    if not values:
        print("No 'Combined Power' lines parsed from powermetrics output.",
              file=sys.stderr)
        if proc.returncode != 0:
            print(proc.stderr.strip()[:400], file=sys.stderr)
    return values


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--interval-ms", type=int, default=5000)
    ap.add_argument("--out", default=None,
                    help="where to write raw samples (default: ./idle-floor-<ts>.json)")
    ap.add_argument("--label", default="", help="note stored with the samples")
    args = ap.parse_args()

    if os.geteuid() != 0:
        print("ERROR: powermetrics requires root. Re-run with sudo.", file=sys.stderr)
        return 2

    values = sample(args.minutes, args.interval_ms)
    if len(values) < 3:
        print(f"ERROR: only {len(values)} sample(s) parsed; nothing to report.",
              file=sys.stderr)
        return 1

    mean = statistics.mean(values)
    out = Path(args.out or f"idle-floor-{time.strftime('%Y%m%dT%H%M%S')}.json")
    out.write_text(json.dumps({
        "label": args.label,
        "interval_ms": args.interval_ms,
        "samples_w": values,
    }, indent=2))
    # Written as the invoking user so a sudo run leaves a readable file.
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if uid:
        try:
            os.chown(out, int(uid), int(gid) if gid else -1)
        except (OSError, ValueError):
            pass

    print()
    print(f"n            {len(values)}")
    print(f"mean         {mean:.3f} W")
    print(f"median       {statistics.median(values):.3f} W")
    print(f"min / max    {min(values):.3f} / {max(values):.3f} W")
    print(f"stdev        {statistics.pstdev(values):.3f} W "
          f"({100 * statistics.pstdev(values) / mean:.1f}% of mean)")
    print(f"full range   {100 * (max(values) - min(values)) / mean:.1f}% of mean")
    print(f"first->last  {values[0]:.3f} -> {values[-1]:.3f} W "
          f"({100 * (values[-1] - values[0]) / mean:+.1f}%)")
    print()
    # The calibration gate compares two 3 s windows taken minutes apart. Emulate
    # that here: any two windows from THIS quiet sample should agree, and if they
    # do not, the gate is measuring the machine's background, not the run.
    per_window = max(1, int(3000 / args.interval_ms))
    windows = [statistics.mean(values[i:i + per_window])
               for i in range(0, len(values) - per_window + 1, per_window)]
    if len(windows) >= 2:
        worst = max(
            abs(b - a) / a * 100
            for i, a in enumerate(windows) for b in windows[i + 1:] if a
        )
        print("Emulating the calibration gate on this idle-only data:")
        print(f"  {len(windows)} windows of ~3 s; worst pairwise disagreement "
              f"{worst:.1f}%")
        print(f"  the drift gate rejects a record at 15%, so an idle machine "
              f"{'WOULD' if worst > 15 else 'would not'} trip it")
    print(f"\nraw samples -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
