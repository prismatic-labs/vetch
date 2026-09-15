# Calibration protocol v2

`methodology_version`: `"2.0"` (on the calibration record, not the SDK event
`methodology_version`).

This is the interchange contract for hardware energy calibrations. It is
provider-agnostic. The portable product is `vetch validate-calibration`: a
command that runs on someone else's record, on a laptop, with no GPU, and
reports which coefficients their design could actually resolve.

Intended reader: an engineer who already has an internal per-token energy
figure and has never been told whether its coefficients are determined or
merely quiet.

## 1. Two keys

`identity.provider` is the event label. For local inference it is always
`self-hosted`. Infrastructure owner belongs on `host.infrastructure_provider`.
Writing `vast.ai` or `ovhcloud` into `identity.provider` breaks runtime matching.

- **Resolve key** (unchanged): provider, model, gpu, serving engine, precision,
  concurrency. Multiple measurement files under one resolve key already
  tier-cap unless a hint selects one.
- **Storage / measurement key**: resolve identity, plus `host` (infrastructure
  provider, site, tenancy, form factor, instance type), plus
  `measurement_conditions.enforced_power_limit_w`, plus a hash of
  `serving_config`. Two providers, two power caps, or two serving
  configurations never overwrite each other.

`identity.instance_type` is deprecated. The field lives on `host`. Do not
carry both.

Schema 1 files remain resolvable at inference with their original
clamp-to-zero semantics, behind a warning. Newly written records are schema 2.
`active: true` on a newly written record requires schema 2.

## 2. Host, facility, serving, node

`host.region` is the operator's own region label (for example `gra11`).
`host.grid_zone` is an Electricity Maps zone identifier (for example `FR`).
They are different things.

`provider_kind` takes `marketplace`, `hyperscaler`, `sovereign_cloud`,
`on_premise`, `academic_hpc`. It predicts whether facility data is obtainable.

`tenancy` takes `dedicated_host`, `shared_host`, `unknown`. No visible
neighbour (`compute_process_count_at_idle: 1`, `gpu_shared_warning: false`) is
not evidence of dedicated tenancy. Default `unknown`.

`gpu_form_factor` takes `sxm`, `pcie`, `soc`, `unknown`. Apple Silicon records
share this schema.

`measurement_conditions.enforced_power_limit_w` is part of the measurement
key, not prose. Batch 1 mixed 525 W and 700 W across records that later appear
in the same table.

`host.facility` stays null unless sourced. Never infer PUE from a marketing
page. Water reuses the SDK vocabulary (`water_boundary`,
`water_accounting_kind`); do not invent a parallel WUE-scope enum.
`pue_basis` takes `annualised`, `seasonal`, `instantaneous`, `design`.
`grid_accounting_convention` takes `location_based_average`,
`location_based_marginal`, `market_based`. The SDK's grid path today is
location-based. Market-based is a new pipeline, not a flag a consumer can flip
on an existing event.

`provenance.pue_applied` and `provenance.grid_applied` mean "were facility
values folded into the published coefficients". Publish silicon-level
coefficients and let the consumer apply the facility layer.

`serving_config` is engine-agnostic. Common keys stay small
(`prefix_caching`, `speculative_decoding`, `tensor_parallel_size`). Everything
else goes in `engine_specific` and `flags_raw`. The same model with prefix
caching on and off is two records.

`node_power.available` defaults to false. Never fill `node_to_board_ratio`
from the literature when no meter was present. arXiv:2412.08602 is a prior for
discussion, not a substitute for measurement. NVML reports board power (GPU
core and HBM). Host CPU, DRAM, NVSwitch, NICs and PSU losses are outside the
domain.

Do not build a parallel facility registry. `wue.json` and
`docs/region-config.md` already cover this ground.

## 3. Null coefficients

A null coefficient means "do not attribute energy on this axis, and emit a
warning". It never means zero.

`validate_energy_override` accepts a partial override. A record with a null
input term and an identified decode term must attach. Discarding the whole
override because one axis is unidentified is the clamp by another route.

JavaScript: `Number(null)` is `0`. Guard on `null` / `undefined` before
conversion, not after.

v1 files in `~/.vetch/calibrations/` continue to attach at inference with
clamp-to-zero semantics, behind a warning.

## 4. Nested models and per-coefficient status

A per-coefficient status machine that still invalidates the record whenever
one coefficient is impossible reproduces v1. Fit nested models instead:

1. output only
2. output plus intercept
3. output plus intercept plus input
4. output plus intercept plus input plus image (vision models)

Publish the largest nested model in which every included term is
`identified`. Null the excluded terms with a reason string. Prefer output plus
intercept over output only whenever the intercept is identified.
Through-origin regression absorbs a real per-request fixed cost into the
slope; falling back to rung 1 is a last resort.

Per-coefficient status:

| Status | Meaning |
|--------|---------|
| `identified` | Interval excludes zero, sign is physical, **and** the design had adequate leverage on that axis. |
| `not_identified` | Interval includes zero, **or** the design lacked leverage on that axis regardless of what the interval says. Publish with the coefficient null. |
| `rejected` | Adequate leverage **and** an impossible sign. Only this case invalidates a record. |

A record is `active: true` when at least one coefficient in the published
nested model is `identified` and none is `rejected`.

**Leverage beats the interval.** Gemma 4 31B fp8 has a strictly positive
full-model `input_ci95` of about `[0.012, 0.029]` from a design that never had
a prefill arm. The same model at bf16, same card, same day, has an input
interval spanning zero. Two measurements of one model giving incompatible
prefill stories is what an unidentified axis looks like when it happens to
land positive. Status is `not_identified`.

Pinned leverage rule: design span of an axis must be at least **ten times**
the predicted idle uncertainty (`LEVERAGE_MULTIPLE = 10`,
`IDLE_INTEGRATION_UNCERTAINTY = 0.02`). Input additionally requires a prefill
isolation arm: at least two rows with input ≥ 1024 tokens and output ≤ 8, and
an input span of at least 1024 tokens.

Two preconditions sit underneath the leverage rule, because a leverage ratio
can only be computed when the ingredients exist and the design has rank.

**Missing idle is fatal, never a free pass.** With no `idle_watts_before` or
`idle_watts_after` there is no sigma, every leverage ratio is undefined, and an
undefined ratio must fail the gate rather than skip it. A record without an
idle figure identifies nothing.

**Every rung is gated on its own conditioning and fit** (`MAX_CONDITION_NUMBER
= 30`, `MIN_R2 = 0.85`, the thresholds v1 used).

**An axis is published on realised precision.** The bootstrap interval is the
measured standard error, and it already accounts for design span, sample size,
collinearity and which rows inform the axis. A coefficient is published when its
CI half-width is within `MAX_RELATIVE_CI_HALF_WIDTH` (10%) of the estimate.
Judging a measured fit against a *predicted* noise level, when the standard
error is sitting in the interval, is what this replaces. The rule also subsumes
a separate test: an interval containing zero is at least 100% wide, so any
target below that already enforces "interval excludes zero". `LEVERAGE_MULTIPLE`
and `IDLE_INTEGRATION_UNCERTAINTY` survive only for pre-run design planning,
where a prior is legitimate because no residuals exist yet.

**Rejection requires the whole interval below zero.** A slightly negative point
estimate whose interval spans zero is blind, not wrong, and condemning it would
turn an unidentified axis into a whole-record rejection.

**A record states the gate it was judged under** (`provenance.fit.gate`), so a
verdict is reproducible after the thresholds move, and an operator who disagrees
with a value can re-run against their own and point at the difference.

**A record is checked against itself before identifiability is considered.**
`vetch.calibration_invariants` refuses states that contradict: a stored `active`
its data does not support, a `measurement_basis` naming a counter while the
numbers came from power sampling, energy exceeding the gross reading it was
subtracted from, a requested concurrency against an achieved figure of zero.

**One fitting path serves every accelerator.** Identifiability is a property of
the design, not the silicon, so CUDA and Apple Silicon share it. What differs is
measurement: the energy domain, the basis, and whether a duration floor applies.

Apple Silicon now uses the same v2 unbalanced workload design as CUDA. Its
power-sampling path aggregates the short prefill arm into repeated unique
requests and records the aggregate with a `requests` divisor, so the fit sees
the same per-request design points. `grid_design_id` is `v2-unbalanced` on both
paths; only the sampler and hardware energy domain differ.

Apple Silicon has a counter basis too. The default sampler reads the
`Energy Model` group of the IOReport framework, a monotonic energy accumulator
per subsystem (CPU, GPU, ANE, DRAM), without root. A window's energy is the
difference of two counter reads, so no duration floor applies
(`energy_source: ioreport_energy_counter`, `measurement_basis:
ioreport_soc_energy`), and each row carries `raw_energy_by_domain_wh` so the
GPU-only share can be read against `gpu_board` records. The published
coefficients still integrate CPU + GPU + ANE, the `apple_soc` domain, so they
stay comparable with powermetrics records. `--sampler powermetrics` keeps the
sampling path, which requires root and the floor.

The prefill isolation arm spans 811 to 12331 input tokens per request, with
output pinned at 8. It determines the prefill coefficient to ±3.1% versus
decode's ±4.7% on the same run. Batch 1 tops out at 427 input tokens without
the arm and cannot resolve the axis.

Measured prefill cost is **0.745% of decode per token** on an H100 at a 575 W
cap, with chunked prefill at vLLM 0.28 defaults (not confirmed disabled; see
`serving_config`). A roofline argument from HBM bandwidth over compute predicts
0.34%, and is wrong by roughly 2x because it is a *timing* ratio: prefill runs
near the power cap while decode does not, so the energy ratio exceeds the time
ratio. Use roofline to seed a design, never to gate a publication.

**The sampling duration floor and the prefill arm conflict by construction.**
An isolation point pins output at 8 tokens, so a single request can finish in
under 100 ms, below the floor that applies on any power-sampling path. A row may
therefore aggregate several identical requests into one timing window: prompts
stay unique, tokens and energy are summed, and the row is normalised per request
before fitting, since the design point is the per-request one. A collinear rung cannot
separate its own terms however tight the bootstrap interval looks, because
every resample inherits the same degeneracy: the interval comes back narrow
and excludes zero while the split between the terms is arbitrary. A rung that
fails either check hands out no `identified` status and is never published.

`residuals_structured` is a misspecification flag attached to the linear
model. It is surfaced in the record and in `validate-calibration` output. It
does not by itself discard an identified decode slope.

Power-cap records are deliberately fail-closed at inference. A record with
`measurement_conditions.enforced_power_limit_w` is capped to curated/Tier 1
unless the caller supplies the same value through `ResolveHints(power_cap_w=)`
or `VETCH_CALIB_POWER_CAP_W`. Storage separation alone is not evidence that the
runtime is operating at the same cap.

## 5. Admission

Admission is a **pre-run leverage prediction**, not a post-fit residual ratio.
Build the prior from idle watts × expected duration × integration
uncertainty, and refuse an axis whose planned design span falls below the
multiple in §4.

Do **not** drop rows where subtracted idle exceeds 30% of raw energy. Of 121
batch-1 rows, 96 of 120 non-zero rows exceed that fraction. Four records lose
every row. At concurrency 1, idle near 124 W on a card whose decode never
approaches TDP means a 30–50% idle fraction is the operating regime.

Do **not** apply a 2 second duration floor on the energy-counter path.
`nvmlDeviceGetTotalEnergyConsumption` is a hardware accumulator. Yang, Adamek
and Armour (arXiv:2312.02741) concerns `nvidia-smi` power **sampling** gaps.
Apply a 2000 ms floor only on the power-sampling fallback.

Keep short rows. They are poor for intercept and input and they are how
decode was identified. Drop only corrupt rows (non-finite or zero raw
energy). Batch 1's Llama table has one such row.

## 6. Dual energy basis

Subtracting idle gives a marginal number: what one more token costs. That is
right for asking whether a generation is worth doing and wrong for anything
an operator reports, because idle is what they pay for.

```json
"energy_basis": {
  "marginal": { "wh_per_1k_output": null, "idle_subtracted": true },
  "attributed": {
    "attribution_rule": "idle_amortised_over_measured_tokens",
    "levels": [
      { "assumed_utilisation": 0.2, "wh_per_1k_output": null },
      { "assumed_utilisation": 0.5, "wh_per_1k_output": null },
      { "assumed_utilisation": 0.8, "wh_per_1k_output": null }
    ]
  }
}
```

Attributed levels are fixed at 20%, 50% and 80% so two labs emit the same
columns. Attributed-at-assumed-utilisation is still not fleet overhead:
provisioned-but-idle capacity, MIG partitions held for isolation, and
autoscaling headroom are outside both bases. The coefficients are a floor,
not an operational estimate.

## 7. Design (concurrency 1)

Deliberately unbalance the space:

- **Prefill isolation**: inputs from 1k to 32k tokens, outputs capped near 8.
- **Decode isolation**: inputs near 32 tokens, outputs from 512 to 4096.
- **Verification band**: a middle set used to check the fit rather than to
  build it.

Replicate at every corner. Vary prompts across replicates unless prefix
caching is the thing being measured.

`vetch calibrate-cuda --grid v2` generates this design. The default `--grid v1`
is the historical 11-point grid, still nested-fit on write.

## 8. Phase separation, and where it stops

At concurrency 1, prefer measuring prefill and decode as separate phases
(TTFT boundary, energy counter integrated over each phase). The linear model
is then a fallback.

**Phase separation does not compose with concurrency.** At concurrency 32 the
device mixes prefills and decodes belonging to different requests. There is
no TTFT that carves board energy into two phases.

- Phase-separated coefficients at concurrency 1 only.
- At concurrency above 1, keep the shipped amortization curve
  `Wh/1k_out ≈ a/C + b` from `vetch calibrate-cuda --batched`.

v2 does not rebuild the concurrency ladder. `--batched`, unique prompts, and
`identity.concurrency` already ship. v2's job is stricter admission, dual
energy basis, serving flags pinned, and phase separation restricted to
concurrency 1.

Default the first v2 run to concurrency 1 with the serving path pinned. Add
concurrency 8 and 32 as a separate arm, not as the same grid. Prefill
isolation × decode isolation × verification × concurrency × replicates ×
models is multiplicative GPU time.

## 9. Serving-path pinning

Recent vLLM builds enable chunked prefill and automatic prefix caching by
default. Unrecorded, they turn a measured axis into an environmental accident.

Pin the serving path before measuring: disable chunked prefill, prefix
caching, and speculative decoding for the baseline arm. Per-request
`extra_body` keys are a request, not a confirmed engine-start state; vLLM
ignores several of them after launch. Record the request in `flags_raw` /
`engine_specific.requested` and leave `prefix_caching` / `speculative_decoding`
null unless the engine confirmed them. Then run those flags as a deliberate
axis afterwards.

Batch 1 cannot support a precise prefill claim: `serving_config` was not
captured, and the records say only `server_version: "openai-compatible (...)"`.

## 10. Control

- Lock clocks and enable persistence mode. Record `sm_clock_locked` and
  `sm_clock_mhz`.
- Record and fix `enforced_power_limit_w`. Treat it as measurement identity.
- For CUDA v2's sampling path, measure idle for at least 60 seconds before and
  after and report the distribution. Other samplers may use a shorter window
  when documented in the record. Drift is a validity signal: surface it and
  apply the energy-share gate; do not reject a completed record on percentage
  drift alone.
- Fail closed on `compute_process_count_at_idle != 1` if tenancy is unknown.
- Sample count at least 22 for a v2 write (the v1 grid remains 11 points
  when `--grid v1` is used to re-fit historical tables).

## 11. Hardware-local records

Records that go `active: true` under a v2 re-fit of batch 1 stay
hardware-local. They must not be copied into the shipped community set, and
they must not be copied into someone else's `~/.vetch/calibrations/` as
default Tier 0 for other people's GPUs. `resolve()` only loads that directory.
A file sitting in this repository is an audit artifact, not an auto-loaded
coefficient.

## 11b. Idle drift, and the batched lane

**Idle drift is a flag, not a reject.** It affects the idle subtraction used by
the marginal basis and must be disclosed. The shipped batched records remain
usable, but drift should be reported wherever their ratios are quoted.

**The gate is the energy share, not the drift percentage.**
`idle_drift_pct` is scale-free, and that is precisely why it cannot decide
anything: it says how far idle moved relative to idle, not whether the movement
threatens the coefficients. It was sized against a ~90 W H100 idle, where a
0.3 W background excursion is 0.3%. Apple Silicon idles near 1 W, where the same
excursion is 30%.

Measured directly on an M5 with no workload running at all — 5 minutes, 60
samples: mean 1.019 W, min 0.983, max 1.313, and a worst-case 33.6%
disagreement between two sampling windows. **An idle machine trips a 15% gate.**
Idle is not a stable quantity at that power scale; it is a floor with sporadic
background excursions, and two short windows minutes apart sample different
points of it.

What decides the question is how much the idle uncertainty contributes to the
rows actually being fitted:

```
MAX_IDLE_UNCERTAINTY_ROW_SHARE_PCT = 10
share = |idle_after - idle_before| x row_duration / row_energy   (median over rows)
```

Accelerator-agnostic by construction, and it separates the evidence cleanly.
Across seven Apple Silicon runs of one identity and every shipped H100 record:

| | drift% | share% |
|---|---|---|
| the one run whose decode coefficient came out 19% wrong | 74.5 | **40.2** |
| a run agreeing with the clean runs to 0.04% | 21.7 | 2.3 |
| clean Apple Silicon runs | 2.0-3.2 | 0.24-0.52 |
| all 8 shipped H100 records | 0.4-4.0 | 0.32-2.13 |

The percentage gate rejected four of seven Apple Silicon runs, three of them
with coefficients agreeing to 0.5%. The share gate rejects exactly the run that
was wrong, and leaves every H100 record passing with margin.

`idle_drift_pct` stays on the record: it is the number an operator can
sanity-check by eye, and the schema does not drop fields. A record with no run
table to weigh the drift against falls back to the percentage.

Two supporting fixes make the number mean something. Both idle baselines go
through one helper (`measure_idle_window`) so they cannot be measured
differently - the "after" window once had no settle at all, so a figure taken
off a workload tail was compared against one taken after a pause, and idle_after
came in below idle_before in three runs of four. Correcting that moved the
measured decode coefficient by 4.4%, because the inflated baseline had been
over-subtracting from every row.

**The batched lane skips nested-OLS identifiability only. It never skips
provenance sanity.** The lane exists because the estimator differs, not because
the evidence bar is lower. Entry to it cannot rest on a self-declared label: a
record qualifies only when it carries the corroborating fields emitted by a
real `--batched` run. These fields are not cryptographic proof against a
deliberate edit:

- `provenance.calibration_mode == "batched"` (a machine-written enum, not a
  substring search over free text)
- `identity.concurrency` a positive integer
- `provenance.amortization_fit` with finite numeric `a` and `b`
- `provenance.samples` and `requests_per_level` at least 8
- `raw_run_table` of exactly one row, with positive `energy_wh` and
  `output_tokens` (zero rows is no evidence, not a throughput aggregate)
- idle watts present, and a named `energy_source`

A record failing any clause falls through to the nested lane and is judged
there. All twelve shipped batched records satisfy the conjunction.

`validate-calibration` re-derives the record's claims from its own run table
and fails on any disagreement - not just `active`, but every published
coefficient, every per-axis status, and the published rung. Comparing only
`active` let a record ship a coefficient that its own data does not support
(doubling the decode figure was accepted), which is the class of disagreement
the tool is designed to catch when pointed at someone else's file.

It also recomputes `profile_hash` and `content_hash` from the record's own
fields. That is a cheap integrity check on an unedited record, not a trust
anchor - whoever edited a record can recompute its hashes too. The refit above
is the check that cannot be forged without also forging the run table.

### Resolve-time checking

Inference resolution validates every schema-2 record before exposing its
coefficients, but not to the same depth. Schema, self-consistency and both
attestation hashes always run: stdlib, microseconds, and enough to reject a
hand-edited, truncated or stale-schema record. The refit does **not** run by
default, because it is the full nested fit (eight rungs by 500 bootstrap
resamples) at roughly 35 ms per record, landing on the first call of every
process on a path budgeted at 5 ms.

What the refit uniquely catches is a record whose coefficients were changed and
whose mirrors *and* hashes were then recomputed, leaving the run table intact.
That is a real attack on a shared calibration directory, so set
`VETCH_CALIB_REFIT_ON_RESOLVE=1` when the directory is not solely yours.
`vetch validate-calibration` always performs the full check - point it at any
record you did not produce.

## 12. Validator

```
vetch validate-calibration <file.json>
vetch validate-calibration <file.json> --schema-only

# Sweep a directory. Use find, not a `**` glob: bash without globstar (and
# macOS bash 3.2, which has no globstar at all) silently expands `**` to `*`
# and checks one directory level.
find h100-calibration-records -name '*.json' -exec vetch validate-calibration {} \;
```

JSON Schema (`src/vetch/data/calibration_record.schema.json`) checks types and
required fields only. Identifiability lives in the CLI and has two estimators:

- Nested OLS + bootstrap for concurrency-1 tables. Needs the `calibrate`
  extra (NumPy).
- Batched amortization (`Wh/1k_out ≈ a/C + b`) for `--batched` records.
  Those files carry a one-row throughput aggregate, so nested OLS cannot
  run. The validator reads the stored decode coefficient. No NumPy.

Never import identifiability from `vetch.calculation`. The runtime path
stays stdlib-only.

## 13. Limits to state with any published table

- Utilisation dominates. The coefficients are a floor.
- Marginal and attributed answer different questions.
- NVML sees the board. Node-level draw is excluded.
- Annualised PUE against a per-token figure is an approximation. Grid
  convention must be declared and never mixed inside a comparison.
- Water can invert a ranking; state `water_boundary` or omit water.
- Batch 1 had no serving config. Even after a v2 re-fit it cannot support a
  precise prefill claim.
- Power caps differ inside batch 1 (525 W against 700 W).
- At concurrency 1, decode is memory-bandwidth-bound. A band of decode energy
  per parameter may be a restatement of HBM bandwidth. Halving bytes per
  parameter should roughly halve decode energy; the Gemma 4 31B bf16 vs fp8
  pair is that test.
- `METHODOLOGY.md` refuses a universal per-1B constant. Any band is "on this
  card, at concurrency 1, among these models, at these power caps".
