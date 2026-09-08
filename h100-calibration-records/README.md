# H100 calibration records

Hardware-local audit artifacts from batch 1 (Vast.ai H100 SXM, marketplace
tenancy unknown). Schema 2 after `scripts/migrate_calibration_records_v1_to_v2.py`.

These files must **not** be copied into `src/vetch/data/community_calibrations.json`
or into someone else's `~/.vetch/calibrations/` as default Tier 0.
`calibration_store.resolve()` only loads that user directory.

`serving_config` was not captured at measurement time, so even after the v2
re-fit these records cannot support a precise prefill claim. Qwen3.8-27B was
measured at 700 W; the other six at 525 W.

## Batch 2: the first prefill measurement

`...__pl575.0-3703bcbb.json` is the first record in this set produced by a
design carrying a prefill isolation arm, and the first anywhere here with an
identified input coefficient.

| | value | precision |
|---|---|---|
| `wh_per_1k_input` | 0.003664 Wh/1k | ±3.1% |
| `wh_per_1k_output` | 0.491684 Wh/1k | ±4.7% |
| `intercept_wh` | withheld | ±133% |

Qwen2.5-7B-Instruct, bf16, vLLM 0.28, prefix caching disabled at engine start,
575 W cap, 32 rows over two iterations, idle drift 0.73%, hardware energy
counter. **Chunked prefill was not confirmed disabled**: vLLM 0.28 enables it by
default, the disable was requested per-request only, and the record's
`serving_config` says `confirmed: false`. The protocol's baseline arm calls for
it off, so read `wh_per_1k_input` as the cost of prefill *as vLLM 0.28 serves
it* (chunked), not of unchunked prefill. Clocks were not locked either
(`sm_clock_locked: null`), as in batch 1.
`published_rung` is `output_input`: the intercept is not resolvable on this
design, and the nested lattice publishes decode and prefill without it.

Three things this settles:

- **Prefill is measurable, and batch 1's failure was the design.** Those seven
  records top out at 427 input tokens with no isolation arm, so their input
  axis is unidentifiable in principle. This design spans 811 to 12331 tokens
  per request and determines the coefficient to ±3.1%, better than decode.
- **Decode is consistent, not replicated.** 0.4917 Wh/1k here against 0.4863 in
  the batch-1 record for the same model on the same GPU class. The two agree to
  1.1%, but they differ in host, power cap (575 W vs 525 W) and design, so this
  is one measurement landing near another, not a reproduction: nothing here
  separates model-and-GPU agreement from cap sensitivity, and no experiment on
  this branch measures cap sensitivity at all. Read it as "same value to within
  a few percent across two uncontrolled setups"; the automated check asserts
  <5%, which is the honest tolerance for what was varied.
- **Prefill costs 0.745% of decode per token**, against the 0.34% a
  bandwidth-over-compute roofline argument predicts. A roofline ratio is a
  *timing* ratio; prefill runs near the power cap while decode does not, so the
  energy ratio is roughly twice the time ratio. Measured, not derived.

Caveat: the cap here is 575 W, against 525 W for most of batch 1 and 700 W for
Qwen3.8-27B. Three caps across four machines, which is why
`enforced_power_limit_w` is part of the measurement key. The records do not
collide, and they are not a like-for-like coefficient comparison either.

See `docs/calibration-protocol-v2.md`.
