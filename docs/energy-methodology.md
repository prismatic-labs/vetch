# Energy and carbon methodology

Vetch reports energy, carbon, and water for every inference call. These numbers are useful for comparison, prioritization, and reduction decisions. They are not carbon certification, regulatory disclosure, or water accounting. This document explains where each number comes from and how much to trust it.

The short version: hosted APIs expose no power meter, so their energy is estimated from infrastructure-aware benchmarks, not measured. Only self-hosted models you calibrate yourself produce a genuinely metered figure. Every event tells you which case it is.

## Read the confidence, not just the value

Every event carries three fields that describe how the energy figure was produced:

- **`energy_tier`:** the method used (see the tier table below).
- **`energy_uncertainty_pct`:** the uncertainty band for that method.
- **`model_match`:** how the model name resolved to a registry entry (`exact`, `alias`, `prefix`, `family`, or `fallback`). See [model-resolution.md](model-resolution.md).

The tier matters as much as the value. A `0.42 Wh` reading at Tier 0 is a measurement; the same reading at Tier 3 is an order-of-magnitude guess. Always carry the tier alongside the value in any report you build.

## The four tiers

| Tier | Name | Uncertainty | Source |
|------|------|-------------|--------|
| 0 | Measured | ±10–20% | Direct GPU power telemetry from a local `vetch calibrate` run |
| 1 | Inferred | ±20–50% | Infrastructure-aware benchmarking of hosted APIs |
| 2 | Validated | ±50–100% | Crowdsourced aggregates / estimation fallback |
| 3 | Estimated | Order of magnitude | Parameter-based calculation, or a proxy match |

In practice, bundled registry rows are almost all Tier 1 (hosted-API benchmarks) or Tier 3 (parameter estimates and proxies). Tier 0 is produced at runtime by your own calibration, and Tier 2 is a fallback band used when a figure has to be estimated (for example, a script-aware character-to-token ratio when exact token counts are unavailable) rather than a shelf of pre-benchmarked models. Do not expect a random model to land in Tier 2.

### Tier 0: Measured

The only genuinely metered tier. You run the model on your own hardware, and `vetch calibrate` reads GPU power draw through `pynvml` (or `powermetrics` on Apple Silicon) during a real workload. The result is a per-token energy coefficient measured on your silicon, stored in `~/.vetch/calibrations/` and picked up automatically on later runs.

This tier only exists for self-hosted models you measure yourself. No hosted API can reach it.

### Tier 1: Inferred

Hosted-API models (OpenAI, Anthropic, Google). These figures come from infrastructure-aware benchmarking, not power telemetry. Because hosted APIs expose no power meter, energy is inferred from public API behavior, provider environmental multipliers, and statistical hardware inference. Useful for comparing models against each other; not a measurement of a specific call.

Tier 1 coverage spans the major hosted families: recent OpenAI GPT and o-series models, Anthropic Claude (including Extended Thinking), DeepSeek R1/V3, and open-weight Llama measurements, among others. That set grows as benchmarks land, so treat any list here as a snapshot: **`vetch methodology` reports the exact Tier 1 coverage in your installed registry.**

The Tier 1 figures derive from [Jegham et al. (2025)](https://arxiv.org/abs/2505.09598), an infrastructure-aware benchmark across roughly 30 commercial models.

### Tier 2: Validated

Crowdsourced aggregates. Wider uncertainty than Tier 1 but grounded in reported real-world usage rather than pure calculation.

### Tier 3: Estimated

Everything with no benchmark. Two things land here:

1. Models resolved by a `prefix` or `family` match, a proxy to a similar model. These are deliberately downgraded to Tier 3 even when the proxy target has better data, so a current-generation model the bundled registry has not caught up to is flagged low-confidence instead of dressed up as exact.
2. Fully unknown models, estimated from parameter count alone.

Tier 3 figures are directional only. Use them to say "this route is roughly 10x that one," not "this route emitted 4.2 grams."

## Cross-validation

Open-weight and self-hosted figures can be checked against independent direct-measurement work:

- [ML.ENERGY](https://ml.energy): open data plus the Zeus measurement toolkit
- [Samsi et al. (2023)](https://arxiv.org/abs/2310.03003): LLM inference energy benchmarking
- TokenPowerBench

Closed hosted-API models have no public power telemetry, so they stay inferred by necessity. There is no independent measurement to cross-validate them against.

## Water

Water estimates are the most facility-dependent number Vetch produces. Consumption depends on datacenter cooling design, local climate, and the source of the grid's own generation, none of which a client-side SDK can observe. Vetch reports a directional estimate so water shows up in the same conversation as energy and carbon. Do not use it for any external claim.

## Per-model provenance

The bundled registry covers 60 models directly, with curated aliases resolving 170+ dated and versioned names to them. For the exact tier and source behind any single model:

```bash
vetch methodology
```

This reports the per-model tier and provenance actually loaded in your install, including any rows pulled from the remote registry after release.

## Using these numbers

- **Do** use them for FinOps prioritization, model-choice trade-offs, engineering decisions, and internal sustainability tracking.
- **Do** carry the tier and uncertainty into any dashboard or report.
- **Do not** use them for regulatory reporting, carbon certification, or external marketing claims without independent verification and the notes from `vetch methodology`.

## Related

- [model-resolution.md](model-resolution.md): how a model name becomes a registry entry, and why proxies downgrade to Tier 3
- [region-config.md](region-config.md): the grid side of the carbon calculation
- [OPENTELEMETRY.md](OPENTELEMETRY.md): exporting energy and carbon with their tiers attached
