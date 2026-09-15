Closes #3. Refs #1 (the powermetrics path is now the fallback; its gap fix is still worth doing separately) and #2 (each row now carries the per-domain split).

## What

`vetch.ioreport`: a `ctypes` binding for the `Energy Model` channel group of the IOReport framework, and `IOReportEnergyMonitor`, a background reader that accumulates per-domain energy every 100 ms and exposes the same `integrate(t0, t1)` / `mean_watts(t0, t1)` interface as `AppleSiliconMonitor`. Window energy is the interpolated difference of the cumulative series at the two edges, so it does not depend on where samples fell relative to the window.

`calibrate_apple_silicon` gains `sampler="auto" | "ioreport" | "powermetrics"` (CLI `--sampler`). `auto` uses IOReport when `ioreport.available()` and falls back to powermetrics otherwise, per the fail-open rule; an explicit `ioreport` on a machine that cannot read it raises rather than silently producing a record on a different basis. Root is asserted only on the powermetrics path.

Record changes on the IOReport path:

- `energy_source: ioreport_energy_counter`, `measurement_basis: ioreport_soc_energy`, `integration_method: energy_counter`, plus `sampler` and `energy_channels` (channel name to unit) in provenance. Source and basis are derived from the sampler in one table so they cannot disagree, following `calibrate_cuda._measurement_basis`.
- `admit_rows` applies no duration floor to this source (the string names no power sampling), which is the point of a counter basis.
- Each run row carries `raw_energy_by_domain_wh` (`cpu`, `gpu`, `ane`, `dram`). The fit still uses the combined CPU + GPU + ANE figure, so coefficients stay in the `apple_soc` domain existing records use.

## Evidence

Apple M1 Pro, macOS 26.6.2, uid 501, release wheels. Channels found: `CPU Energy` (mJ), `GPU Energy` (nJ), `ANE0` (mJ), `DRAM0` (mJ).

Idle, then a 2.5 s MPS matmul loop, mean watts by domain from the counters:

| | cpu | gpu | ane | dram |
|---|---|---|---|---|
| idle | 1.16 | 0.36 | 0.00 | 0.56 |
| MPS matmul | 1.15 | 19.45 | 0.00 | 3.81 |

A CPU-bound NumPy burst moves `cpu` to 12.7 W with `gpu` flat at 0.17 W. So the split attributes correctly and the GPU-only axis raised in #2 is available from the same sample.

## Tests

`tests/test_ioreport.py` (23): channel classification, including that per-core and per-cluster components are dropped to avoid double counting; interpolation exactness and independence from sample phase; default domains exclude DRAM; out-of-range windows read zero rather than a guess; thread lifecycle through a fake reader. `tests/test_calibrate_metal.py` (+10): sampler resolution, that the counter source keeps sub-floor rows while the powermetrics source drops them, that every (source, basis) pair passes `_check_measurement_basis`, that the IOReport path never calls `assert_root` and the powermetrics path still does, and that the CLI passes `--sampler` through. None touch the framework, so CI on Linux is unaffected.

Locally: 1817 passed, 28 skipped, coverage 74.5%; `ruff` clean; `mypy --strict` clean on the touched files (the tree currently fails on numpy's stubs under `python_version = 3.10` before reaching them, which is independent of this change).

## Not done here

- No end-to-end Ollama calibration in this PR: I have not pulled a model on this machine. Happy to attach a record from a full `--quick` run if you want one before merging.
- The powermetrics monitor is unchanged; #1 stays open for it.
- `docs/calibration-protocol-v2.md` gets one paragraph; the calibration issue template still lists the old drift gate and could be refreshed separately.
