# Vetch Methodology

methodology_version: "1.3"
sdk_version: "0.11.1"

## Preamble
Vetch exists because AI systems currently operate with no feedback on their energy consumption. Every inference draws power from infrastructure with real costs—financial, environmental, and systemic. None of this is visible to the developer making the API call.

This methodology is our first attempt to create that feedback loop. It is imperfect. The energy estimates are uncertain. We publish it openly so it can be challenged, corrected, and improved.

We believe imperfect measurement, honestly reported, is better than no measurement at all.

## SDK Instrumentation Model

**As of v0.3.0, automatic instrumentation is production-ready.**

```python
import vetch
vetch.instrument()  # All LLM calls are now tracked automatically
```

`instrument()` patches the OpenAI, Anthropic, Google GenAI, and Vertex AI clients at the module level. Every call — sync, async, streaming, and non-streaming — is tracked without any code changes in the calling application.

**How it works:**

1. `instrument()` stores default `region` and `tags` globally.
2. Provider wrappers intercept each API response or completed stream.
3. If a manual `wrap()` context is active, the event is attributed to it.
4. Otherwise, a short-lived auto-context is created for that single call, calculates energy/carbon/cost, and emits the event.

This means the tracking boundary is the individual API call, not a manually delimited block. For explicit attribution (e.g., attaching tags to a specific feature), `wrap()` remains available and takes precedence.

**Streaming:** Streaming calls (`stream=True`) are fully tracked under `instrument()`. The event is emitted when the stream is exhausted (last chunk consumed). If the stream is abandoned mid-way, the event is still emitted with `complete=False` and the characters counted so far.

## Methodology Version
This document is versioned. If we change the energy heuristics (e.g., input:output ratio from 1:3 to 1:2.5), methodology_version will increment. Check this field to understand why historical data may differ from current calculations.

**Current: methodology_version 1.3**

- **1.3 (v0.10.x):** Extended the visual energy term to the registry path and added honest-incompleteness reporting. Registry VLM rows may declare `wh_per_visual_unit` (Wh per normalized visual unit) and `visual_tokens_per_unit`; when present, the visual portion is priced separately and its tokens are removed from the text total (`energy_wh += wh_per_visual_unit × visual_units`, mirroring the calibration path's `wh_per_image`). A call that carries visual input but has no visual coefficient is flagged `energy_completeness="text_only"` and warns, rather than emitting a text-only figure as if it were complete. No shipped registry row declares a visual coefficient yet — populate one only from a live-verified measurement or a Tier-0 calibration (see registry hygiene in CLAUDE.md); until then visual calls on cloud VLMs are correctly reported as text-only partials.
- **1.2 (v0.8.0):** Added VLM image energy term (`wh_per_image`), Apple Silicon powermetrics Tier-0 calibration, and per-request fixed overhead intercept (`intercept_wh`). Energy formula now optionally includes `n_images × wh_per_image + intercept_wh` for VLM providers.
- **1.1:** Reasoning token path, schema v2, session advisory engine.

## The Formulas

### Energy (primary measurement)
`energy_wh = (input_tokens × wh_per_1k_input + output_tokens × wh_per_1k_output) / 1000`

Energy is the primary metric. It is derived from model-specific estimates (see tier system below) and exact token counts (from provider response).

### Carbon (derived from energy + grid + PUE)
`carbon_g = energy_wh × PUE × grid_intensity / 1000`

Carbon is a derived metric. It compounds energy uncertainty with grid data. Grid intensity is real-time and accurate when available (signal_quality: live). Carbon inherits all uncertainty from energy, plus regional and temporal variation in grid mix.

### Water (derived from energy + WUE)
`water_l = (energy_wh / 1000) × WUE`

Water measures datacenter cooling consumption. WUE (Water Usage Effectiveness, liters per kWh) varies significantly by datacenter location and cooling technology (range: 0.2–3.5 L/kWh). Water estimates carry ±200% uncertainty vs ±50% for carbon.

## Token Counts
Exact. Extracted from provider response usage field. For interrupted streams where usage is unavailable, we report null. We do not estimate tokens from text. We do not bundle tokenizer libraries.

## Energy per Token
This is our largest uncertainty.

### Energy Tiers

| Tier | Name | Definition | Uncertainty |
|------|------|------------|-------------|
| 0 | **Measured** | Direct hardware measurement via GPU sensors (pynvml, rocm-smi). User-provided from their actual inference runs. | ±10-20% |
| 1 | **Vendor-Published** | Official provider data or peer-reviewed academic measurements on production hardware. | ±20-50% |
| 2 | **Validated** | Derived from published research with clear methodology. Aggregated from multiple sources. | ±50-100% |
| 3 | **Estimated** | Calculated from parameter count, architecture class, and theoretical compute requirements. | order of magnitude |

### Tier Confidence

**Tier 0 (Measured)**: Available when users run local GPU inference (Ollama, vLLM, llama.cpp) and use hardware sensors to capture actual power draw. Limitations: Requires compatible GPU (NVIDIA via pynvml, AMD via rocm-smi), baseline subtraction introduces noise, short inferences are less accurate.

#### Local hardware calibration records (v1)

NVIDIA (`vetch calibrate-cuda`) and Apple Silicon (`vetch calibrate-apple-silicon`) produce Tier-0 coefficients for **your** stack. CUDA writes a versioned record (`schema_version: 1`) keyed by a full **identity** — `(provider, model, canonical_gpu, backend, precision)` — so the same model on H100-SXM vs A100, or BF16 vs FP8, cannot silently overwrite each other. At inference the SDK only has `(provider, model)`, so `calibration_store.resolve` is deliberately honest:

| Situation | `energy_source` | Tier | `calibration_match` |
|-----------|-----------------|------|---------------------|
| One same-provider identity | `local_calibration` | measured (0) | `exact` |
| Several same-provider identities (e.g. bf16 **and** fp8) | `reused_calibration` | capped ≥ 1 | `proxy` |
| Same, but env hints uniquely select one **without** `VETCH_CALIB_HINTS_TRUSTED` | `reused_calibration` | capped ≥ 1 | `curated` |
| Same, with `VETCH_CALIB_HINTS_TRUSTED=1` | `local_calibration` | measured (0) | `exact` |
| Hints set but match nothing | _(none)_ | — | refuse attach |
| Cross-label among self-hosted providers only | `reused_calibration` | capped ≥ 1 | `curated` / `proxy` |
| Heuristic / unknown / missing `gpu_known` | (never exact Tier 0) | capped ≥ 1 | `proxy` |
| Event / calib provider `openai` | `reused_calibration` or refuse write | capped ≥ 1 | `curated` (never exact) |
| Cloud event with no same-provider file | _(none)_ | — | `null` |

Events also carry `calibration_match` (and `vetch.calibration_match` on OTel). Session confidence rollups / strict mode use `min(registry model_match, calibration_match)`. Ambiguity includes the stored model string and forward-compat identity dims (`instance_type`, …) so `:latest` variants cannot collapse to false exact. Calibrate under the label your events emit (`--provider self-hosted`; `openai` is refused at write). Both CUDA and Apple require `--precision`. Identity JSON uses `serving_engine`. Overwrites archive under `archive/`. Legacy flats still resolve for a provider when that provider has no v1 record.

**Tier 1 (Vendor-Published)**: The gold standard. Academic measurements on specific hardware with rigorous methodology qualify for Tier 1. As of v0.3.0, 21 models have Tier 1 data from Jegham et al. (2025).

**Tier 2 (Validated)**: Aggregated from multiple crowdsourced Tier 0 measurements or independent academic studies. Example: "Llama 3.1 8B averages 0.12 Wh/1k tokens across 47 user submissions (std dev 0.03)."

**Tier 3 (Estimated)**: Current default for models without measurements. Based on:
- Parameter count → FLOPs per token
- Architecture class (dense, MoE, hybrid, reasoning)
- Hardware efficiency assumptions (H100 baseline)
- Price-tier proxying from architecturally similar measured models

Tier 3 estimates should be treated as order-of-magnitude guidance (±1000%), not precise measurements.

### Match Confidence (how the model was resolved)

Energy tier answers *how good is the coefficient*. Match confidence answers a separate question: *is this even the right model's coefficient*. Every event carries `model_match`, recording how the model name resolved against the registry. For aggregation and reporting these five precisions collapse onto four confidence classes:

| `model_match` | Confidence class | Meaning |
|---------------|------------------|---------|
| `exact` | **exact** | Exact registry identity. |
| `alias` | **curated** | A curated, trusted one-to-one equivalence. |
| `prefix`, `family` | **proxy** | An approximate stand-in; energy tier is also floored to 3. |
| `fallback` | **none** | No match; conservative generic fallback. |

Because energy/cost/carbon are summed, confidence aggregates too. `SessionStats.summary()["confidence"]` (and `rollup_confidence_from_events`) report per-class totals plus the fraction of energy and cost derived from exact-or-curated resolutions versus proxies, and the fraction of energy that is a `text_only` partial. A consumer uses these to judge whether a total is fit to report.

For an audited figure, silent substitution is a liability: once summed, an approximated value and a measured one are indistinguishable. Strict mode makes it enforceable and is opt-in (the default stays permissive for observability):

- `vetch.set_min_match_confidence("curated")` (or `VETCH_MIN_MATCH_CONFIDENCE`) sets a floor.
- `filter_events_by_confidence(events)` quarantines below-floor records out of the aggregate.
- `require_confidence(events)` fails loudly (`ConfidenceError`) if any below-floor record is present.

The safe direction under uncertainty is disclosure — flag, quarantine, or refuse — never presenting an estimate with the authority of a measurement.

### Architecture-Aware Estimation

For Mixture-of-Experts (MoE) models, we estimate energy based on **active parameters per token**, not total parameters. This prevents significant overestimation:

| Model | Total Params | Active Params | Correction Factor |
|-------|-------------|---------------|-------------------|
| GPT-4 | ~1.8T | ~220B | 8x |
| GPT-4o | ~200B | ~50B | 4x |
| Mixtral 8x7B | 47B | 13B | 3.6x |
| Gemini 1.5 Pro | ~500B | ~100B | 5x |
| DeepSeek-R1 | 671B | 37B | 18x |

**Dense models** (Claude, Llama) use all parameters per token, so total = active.

### Quantization Factors

Lower precision reduces memory bandwidth and compute:

| Quantization | Relative Energy | Typical Use |
|--------------|----------------|-------------|
| bf16/fp16 | 1.0x (baseline) | Cloud inference |
| int8 | ~0.5x | Optimized serving |
| int4 | ~0.25x | Local inference |

Registry entries include a `quantization` field. When known, estimates are adjusted accordingly.

### Upgrading Tiers

We actively work to upgrade estimates. The path for any model:

```
Tier 3 (day 1: parameter-based estimate)
    ↓ crowdsourced measurements
Tier 2 (validated: aggregated user data)
    ↓ academic publication
Tier 1 (vendor/peer-reviewed)
    ↓ user calibration on their hardware
Tier 0 (measured: your specific setup)
```

### Tier 1 Measurements: Jegham et al. (2025)

Vetch 0.3.0 incorporates the first large-scale, infrastructure-aware benchmarking of LLM energy consumption from **Jegham et al. (2025)** published in "How Hungry is AI? Benchmarking Energy, Water, and Carbon Footprint of LLM Inference" (arXiv:2505.09598).

This research provided hardware measurements for 30 state-of-the-art models deployed in commercial datacenters. We have incorporated measurements for **21 models at Tier 1**, covering most major providers:

| Model | Energy (medium prompt) | Tier | Notes |
|-------|----------------------|------|-------|
| GPT-4.1 nano | 0.271 Wh | 1 | Most efficient in benchmark |
| GPT-4.1 mini | 0.847 Wh | 1 | |
| GPT-4o | 1.214 Wh | 1 | |
| GPT-4o-mini | 1.418 Wh | 1 | More energy than GPT-4o (smaller model, less efficient batching) |
| GPT-4.1 | 2.513 Wh | 1 | |
| Claude-3.7 Sonnet | 2.781 Wh | 1 | Best eco-efficiency among large models |
| Claude-3.7 Sonnet (thinking) | 5.684 Wh | 1 | ~2x standard variant |
| GPT-4 | 6.512 Wh | 1 | Legacy MoE, significantly less efficient |
| GPT-4 Turbo | 6.759 Wh | 1 | |
| GPT-4.5 | 20.500 Wh | 1 | Comparable to o3 |
| Llama 3.1 8B | 0.329 Wh | 1 | |
| Llama 3.3 70B | 0.857 Wh | 1 | ~4x more efficient than Llama 3.1 70B |
| Llama 3.1 70B | 3.559 Wh | 1 | |
| Llama 3.1 405B | 6.911 Wh | 1 | Largest open-weight dense model |
| o1-mini | 1.599 Wh | 1 | Lightweight reasoning |
| o3-mini | 2.448 Wh | 1 | |
| o4-mini | 1.679 Wh | 1 | Medium reasoning effort |
| o1 | 12.100 Wh | 1 | Reasoning model |
| o3 | 21.414 Wh | 1 | Advanced reasoning |
| DeepSeek-V3 | 9.129 Wh | 1 | |
| DeepSeek-R1 | 29.000 Wh | 1 | Most energy-intensive in benchmark |

### Gemini Calibration Caveat

Google reports that a median Gemini Apps text prompt consumes **0.24 Wh** full-stack energy. That is useful methodology evidence, but it is not a per-token Gemini 2.0 Flash measurement: the prompt token count, input/output split, batching conditions, and exact model mix are not disclosed. Vetch therefore treats Gemini 2.0 Flash as **Tier 3** despite the official source. The registry keeps a provisional scalar so dashboards can still compare workloads, but audit reports should read its p5/p95 bounds and basis string rather than treating the point estimate as measured.

**Key Findings:**
- **Reasoning models consume 40-107x more energy** than efficient models like GPT-4.1 nano
- **Non-linear energy scaling:** Short prompts use more energy per token than long prompts due to fixed overhead costs
- **Range of efficiency:** Most energy-intensive model (DeepSeek-R1) consumes **107x more** than the most efficient (GPT-4.1 nano) for identical prompts
- **Smaller is not always greener:** GPT-4o-mini uses more energy than GPT-4o at medium prompt lengths (1.418 vs 1.214 Wh) — batching efficiency matters more than parameter count

### GPT-5 Family (Tier 3 Estimates)

The GPT-5 family (gpt-5, 5-mini, 5-nano, 5-pro, 5.1, 5.2, 5.2-pro, 5.4, 5.4-mini, 5.4-nano, 5.4-pro) has no published energy data. All entries are **Tier 3 (±1000%)**, proxied from architecturally similar measured models:

| GPT-5 Variant | Proxy Source | Rationale |
|---------------|-------------|-----------|
| gpt-5, 5.1, 5.4 | GPT-4.1 medium | Same frontier MoE architecture class |
| gpt-5-mini, 5.4-mini | GPT-4.1-mini medium | Similar price tier, mini architecture |
| gpt-5-nano, 5.4-nano | GPT-4.1-nano medium | Similar price tier, nano architecture |
| gpt-5-pro, 5.4-pro | o3 medium | Reasoning architecture, premium compute |
| gpt-5.2 | GPT-4.1 medium (scaled 1.27x) | Priced ~40% above GPT-5 |
| gpt-5.2-pro | o3 medium | Most expensive reasoning variant |

These estimates will be upgraded to Tier 1 when measurements become available.

### Non-Linear Energy Model

**Critical Discovery:** Energy consumption is **not linear** with token count. Jegham's measurements reveal:

**GPT-4o Energy by Prompt Length (composite per 1k total tokens):**
- **Short** (<1k tokens): ~1.05 Wh per 1k tokens
- **Medium** (1k-5k tokens): ~0.61 Wh per 1k tokens
- **Long** (>5k tokens): ~0.16 Wh per 1k tokens

Longer prompts are ~6x more efficient per-token due to amortization of fixed costs (model loading, memory allocation, batching overhead).

**Implementation:** Vetch uses prompt-length-aware coefficients for models with measured data. The calculation engine automatically selects appropriate coefficients based on total token count:

```python
if total_tokens < 1000:
    # Use "short" coefficients (higher per-token cost)
elif total_tokens < 5000:
    # Use "medium" coefficients
else:
    # Use "long" coefficients (lower per-token cost)
```

**Limitation:** Not all models have measurements across all prompt lengths. Where data is unavailable, we use the medium prompt baseline (most representative of typical usage).

### References & Data Provenance

**Primary Sources and Methodology Anchors:**

1. **Jegham, N., et al. (2025).** "How Hungry is AI? Benchmarking Energy, Water, and Carbon Footprint of LLM Inference."
   - Published: May 2025 (arXiv:2505.09598)
   - Methodology: infrastructure-aware **estimation** combining public API
     behavior with modeled hardware/utilization assumptions (the authors do not
     have sensor access inside third-party production datacenters)
   - Coverage: 30 models including GPT-4o, Claude-3.7, o3, DeepSeek-R1
   - Impact: First infrastructure-aware benchmark at scale
   - URL: https://arxiv.org/abs/2505.09598

2. **Google Environmental Report (August 2025).** "Measuring the Environmental Impact of Delivering AI at Google Scale."
   - Data: Median Gemini Apps text prompt consumes 0.24 Wh
   - Methodology: Full stack accounting (TPU, CPU, datacenter overhead)
   - PUE: 1.10 (2023 fleet average)
   - Vetch tier: Tier 3 for Gemini per-token coefficients, because the token count and per-model decomposition are not published
   - URL: https://cloud.google.com/blog/products/infrastructure/measuring-the-environmental-impact-of-ai-inference

3. **Epoch AI (2025).** "How Much Energy Does ChatGPT Use?"
   - Data: GPT-4o estimated at ~0.3 Wh per query (assumption: 500 token response)
   - Methodology: Analytical model based on MoE architecture assumptions
   - URL: https://epoch.ai/gradient-updates/how-much-energy-does-chatgpt-use

**Why we use Jegham (2025) as our Tier-1 source (with caveats):**
- **Infrastructure-aware:** Models batching, memory bandwidth, and idle capacity
  rather than a flat per-token figure
- **Broad, consistent coverage:** 30 models benchmarked with one methodology
- **Published methodology:** arXiv preprint (arXiv:2505.09598), not yet
  peer-reviewed; the figures are *modeled estimates*, not in-datacenter sensor
  measurements, so we label them Tier 1 (vendor/academic-published) rather than
  Tier 0 (our own hardware measurement)

**Comparison with Earlier Estimates:**
Earlier widely-cited research (de Vries, 2023) estimated ~3 Wh per GPT-3 query. Jegham's measurements show GPT-4o at ~0.3 Wh for short queries—**10x more efficient** than earlier estimates, reflecting both architectural improvements and measurement accuracy.

### Future: Crowdsourced Measurements (Vetch-Sensor)

We plan to enable crowdsourced energy measurements from local inference users:

```bash
# NVIDIA (identity-keyed v1 record; precision required)
vetch calibrate-cuda \
  --provider self-hosted \
  --backend openai --serving-engine vllm \
  --precision bf16 \
  --model google/gemma-4-31B-it \
  --base-url http://127.0.0.1:8000/v1

# Apple Silicon (same v1 store; --precision required, e.g. apple-native)
sudo vetch calibrate-apple-silicon --model moondream:latest --precision apple-native
```

Records land in `~/.vetch/calibrations/` (prior files for the same identity move to `archive/`) and auto-load when the event's provider/model matches (see Tier 0 section above). Anonymized community aggregation into Tier 2 remains a separate path (`scripts/aggregate_calibrations.py`); no prompt content is ever transmitted.

## PUE (Power Usage Effectiveness)

PUE measures datacenter overhead efficiency (cooling, power distribution). Vetch uses **provider-specific PUE values** from official sustainability reports:

| Provider | PUE | Tier | As of | Source |
|----------|-----|------|-------|--------|
| **Google Cloud** | 1.09 | 1 | 2026-06 | [Google 2026 Environmental Report (FY2025)](https://sustainability.google/reports/google-2026-environmental-report/) |
| **Microsoft Azure** | 1.12 | 1 | 2025-05 | [Microsoft 2025 Environmental Sustainability Report](https://blogs.microsoft.com/on-the-issues/2025/05/29/environmental-sustainability-report/) |
| **AWS** | 1.14 | 1 | 2025 | [AWS 2025 Sustainability Report](https://aws.amazon.com/sustainability/data-centers/) |
| **Unknown/Default** | 1.2 | 3 | — | Industry average (Uptime Institute 2025 enterprise ~1.54) |

**Basis caveat:** Google and AWS values are fleet-wide averages. Microsoft does not publish a clean operational fleet average, so Azure/OpenAI use Microsoft's newest-generation figure (~1.12) — a slightly more favorable basis. Reconciling to a common basis would require an *estimated* Azure fleet PUE (~1.18, tier 2), a deliberate methodology change rather than a report refresh.

**PUE Tier Definitions:**
- **Tier 1**: Known value (vendor-published OR user-configured via `VETCH_DEFAULT_PUE`)
- **Tier 3**: Default fallback when provider unknown (1.2 industry average)

Note: There is no Tier 0 or Tier 2 for PUE because:
- No "measured PUE" exists (users don't have datacenter-level sensors)
- No "crowdsourced PUE" makes sense (PUE is a facility property, not per-request)

**Auto-detection:** Vetch infers provider from model name:
- `gpt-*`, `o1-*`, `o3-*`, `o4-*` → OpenAI (Azure-backed, PUE 1.12)
- `claude-*` → Anthropic (AWS-backed, PUE 1.14)
- `gemini-*`, `gemma-*` → Vertex AI (Google Cloud, PUE 1.09)

Google's 1.09 PUE is a fleet-average value. Combining a regional grid intensity with a fleet-average PUE is intentionally transparent but mixed precision: it is better than a generic datacenter default, but it is not a facility-specific request measurement. If you know the deployment PUE, set `VETCH_DEFAULT_PUE` for the audit.

### PUE Limitations

**Critical caveats:**

1. **Fleet averages, not inference-specific**: PUE reports include training workloads (high utilization) and inference (bursty, lower utilization). Inference PUE may be 10-20% worse than fleet average.

2. **Regional variation ignored**: Google's PUE ranges from 1.04 (best datacenter) to 1.20+ (worst). We use fleet average (1.09), which masks 5-15% regional differences.

3. **Provider inference is fragile**: We guess provider from model name. If OpenAI switches from Azure to AWS, or uses multi-cloud routing, our PUE will be incorrect until we update the mapping.

4. **PUE measures datacenter, not model efficiency**: A model on a V100 GPU in a PUE 1.09 datacenter may use MORE energy than the same model on an H100 in a PUE 1.15 datacenter. **GPU generation matters more than PUE.**

5. **Embodied carbon excluded**: PUE only measures operational energy. Manufacturing GPUs, building datacenters, and decommissioning equipment contribute 10-20% of lifecycle emissions but are not captured by PUE.

**Bottom line:** Provider-specific PUE improves the datacenter efficiency component of carbon calculation by ~5-10%, but **overall carbon estimates remain order-of-magnitude** due to Tier 3 energy-per-token estimates and regional grid variations (which matter 10-100x more than PUE).

## WUE (Water Usage Effectiveness)

WUE measures datacenter water consumption for cooling, in liters per kWh of IT energy. Vetch uses **provider-specific WUE values**:

| Provider | WUE (L/kWh) | Source |
|----------|-------------|--------|
| **Google Cloud** | 1.1 | Efficient water-free cooling in many datacenters |
| **Microsoft Azure / OpenAI** | 1.7 | Microsoft sustainability disclosures |
| **AWS / Anthropic** | 2.2 | AWS sustainability report |
| **Unknown/Default** | 1.8 | Industry average for air-cooled datacenters |

**Water calculation:**
`water_l = (energy_wh / 1000) × WUE`

**Cascading lookup:** WUE resolution follows: explicit override → region-specific (from `wue.json`) → provider-specific → default (1.8).

**Limitations:**
- WUE varies dramatically by datacenter location (arid vs. humid climates, water-free vs. evaporative cooling)
- Provider WUE values are fleet averages; individual datacenters may differ by 3-5x
- Water estimates carry **±200% uncertainty** — higher than carbon estimates
- Some datacenters use water-free cooling (WUE ≈ 0), others rely heavily on evaporative cooling (WUE > 3.0)

## Grid Carbon Intensity

Source: Electricity Maps API, marginal carbon intensity.

### 4-Tier Fallback Hierarchy

Vetch uses a multi-tier caching strategy to balance accuracy with reliability:

| Level | Source | TTL | signal_quality |
|-------|--------|-----|----------------|
| 1 | **Memory Cache** | 5 min | `live` |
| 2 | **File Cache** | 30 min | `delayed` |
| 3 | **Electricity Maps API** | Real-time | `live` |
| 4 | **Regional Averages** | Static | `blind` |

The grid intensity lookup proceeds through each level in order. If Level 3 (API) fails (timeout, rate limit, no API key), Vetch falls back to Level 4 regional averages and sets `signal_quality: blind` in the event.

This ensures carbon estimates are always available, with the `signal_quality` field indicating data freshness.

## Contributing Energy Estimates
We want better data. If you have inference energy measurements—from internal benchmarks, published research, or provider relationships—we'll incorporate them with attribution.

### Submission Format
```json
{
  "model": "gemini-1.5-pro",
  "wh_per_1k_input": 0.6,
  "wh_per_1k_output": 1.8,
  "tier": 2,
  "source": "Internal benchmark, 2026-01",
  "methodology": "Power measurement via pynvml during inference...",
  "hardware": "NVIDIA H100 80GB",
  "sample_size": 100,
  "contributor": "Your Name/Org"
}
```

### Tier Assignment for Contributions
- **Tier 0**: Reserved for user-calibrated values on their own hardware
- **Tier 1**: Requires peer-reviewed publication or official vendor data
- **Tier 2**: Requires clear methodology, sample size >= 10, reproducible setup
- **Tier 3**: Theoretical estimates (parameter-based) - default for new models

### How to Submit
- Pull request against `registry/energy.json`
- Email to marco@prismaticlabs.ai
- Open an issue with the data
- (Coming soon) `vetch calibrate --submit` for automated Tier 0 -> Tier 2 aggregation

## Serving-configuration sensitivity (batching, precision, scale)

Per-token energy is not a single number for a model: it depends on how the model
is served. A hardware calibration (`calibrate-cuda` / `calibrate-apple-silicon`)
is deliberately measured at **batch size 1, single stream** (the record stamps
`batch_size=1`, `concurrency=1`, `tensor_parallel_size=null`). That makes it
reproducible and an honest **upper bound** on per-token decode energy at a given
precision. It is *not* representative of batched production serving, where
per-request energy is substantially lower.

The shipped batch=1 records (H100 SXM 80 GB, vLLM, bf16) measure, for Wh/1k
output:

| model | Wh/1k out (batch=1, H100) |
|---|---|
| Qwen2.5-7B-Instruct | 0.50 |
| NousResearch Llama-3.1-8B-Instruct | 0.54 |
| Qwen2.5-VL-7B-Instruct | 0.51 |
| Qwen2.5-32B-Instruct | 2.37 |
| Qwen3.8-27B | 1.90 |

These records ship as `active:false` audit artifacts (their fits were gated
by the quality checks below). They document the measurement; they are not
auto-resolved defaults. Qwen3.8-27B was measured on the same H100 SXM / vLLM /
bf16 stack as the Qwen2.5-32B record (2026-08-14). The batch=1 grid gave
`wh_per_1k_output=1.90` (22 runs, idle drift 1.0%, fit R²=0.994, active=false
via negative-intercept gate). Thinking was disabled for that serve
(`enable_thinking: false`).

### Same-box concurrency sweep (Qwen3.8-27B vs Qwen2.5-32B)

Exploratory `calibrate-cuda --batched` on one rented H100 SXM (vLLM 0.27.1,
bf16, `max-model-len=8192`, unique prompts, 64 requests/level, 64 out tokens).
Not Tier-0. Wh/1k output:

| C | Qwen3.8-27B | Qwen2.5-32B | 3.8 / 2.5 |
|---|---:|---:|---:|
| 1 | 1.97 | 2.44 | 0.81 |
| 2 | 1.01 | 1.25 | 0.81 |
| 4 | 0.52 | 0.64 | 0.81 |
| 8 | 0.29 | 0.34 | 0.85 |
| 16 | 0.17 | 0.19 | 0.88 |
| 32 | 0.11 | 0.12 | 0.89 |

Amortization fits `Wh/1k_out ≈ a/C + b` (R²≈1.0 on both):

| model | a | b | C1→C32 |
|---|---:|---:|---:|
| Qwen3.8-27B | 1.93 | 0.045 | 18.3x |
| Qwen2.5-32B | 2.40 | 0.042 | 20.2x |

At batch-of-one, Qwen3.8-27B is about 19% lower than Qwen2.5-32B. The ratio
stays near 0.81 through C=8, then edges toward 0.89 as both approach the floor.
Absolute energy still falls roughly an order of magnitude with concurrency on
both models.

### Observed qualitatively (direction only; not shipped as records)
Exploratory runs on rented H100 and A100 GPUs (not Tier-0 defaults) show three
mechanistic patterns. We state them as direction, not as calibrated coefficients:

- **Batching amortizes decode energy toward a floor.** Wh/1k output falls with
  serving concurrency and is well fit by `Wh/1k(C) ≈ a/C + b`. Pure `1/C` holds
  only at low concurrency and saturates above it; a batch=1 value can overstate
  per-request energy at high concurrency by roughly an order of magnitude. The
  normalized curve was consistent across a ~4x model-size range in our runs
  (including the Qwen3.8 vs Qwen2.5 sweep above).
- **Precision is a roughly multiplicative lever.** fp8 was about half of bf16
  energy per token, roughly stable across concurrency.
- **Decode energy scales ~linearly with active parameter count** (decode is
  memory-bandwidth bound). The slope is GPU-dependent, so we do not publish a
  universal per-1B constant. The shipped record set is consistent with ~linear
  scaling among these dense models (MoE untested). Qwen3.8-27B sitting below
  Qwen2.5-32B at similar dense scale is consistent with its hybrid
  linear-attention backbone doing less full-attention decode work.

**Prefill/input energy** is much smaller per token than decode, but not
negligible: in our records the input coefficient sits near the regression floor
and is flagged rather than reported with false precision. (An earlier internal
"~0.0002 Wh/1k" figure was a prefix-cache measurement artifact and is withdrawn.)

### Batched calibration mode (`--batched`; experimental preview)
`vetch calibrate-cuda --batched` sweeps serving concurrency and fits
`Wh/1k_out(C) ≈ a/C + b`. This mode is an **experimental preview**: no batched
records are shipped, and its output must not be treated as Tier-0. Use the batch=1
path for reproducible calibration.

### Relationship to published API estimates
Our batch=1 bf16 figures run higher than published API-based estimates such as
Jegham et al. (2025). We do **not** claim the two reconcile. The gap is confounded
by precision (an fp8 batch=1 measurement alone can close much of it) and by
unknown serving concurrency, and those axes are not separately identifiable from
the published numbers. Treat any cross-comparison as suggestive only.

Because serving conditions dominate, the record captures them in provenance: a
coefficient is only comparable to another measured under the same
batch/precision/GPU/stack.
