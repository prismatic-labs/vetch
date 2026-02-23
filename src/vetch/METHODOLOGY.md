# Vetch Methodology

methodology_version: "1.0"

## Preamble
Vetch exists because AI systems currently operate with no feedback on their energy consumption. Every inference draws power from infrastructure with real costs—financial, environmental, and systemic. None of this is visible to the developer making the API call.

This methodology is our first attempt to create that feedback loop. It is imperfect. The energy estimates are uncertain. We publish it openly so it can be challenged, corrected, and improved.

We believe imperfect measurement, honestly reported, is better than no measurement at all.

## Methodology Version
This document is versioned. If we change the energy heuristics (e.g., input:output ratio from 1:3 to 1:2.5), methodology_version will increment. Check this field to understand why historical data may differ from current calculations.

**Current: methodology_version 1.0**

## The Formula
Energy (primary measurement):
`energy_wh = (input_tokens × wh_per_1k_input + output_tokens × wh_per_1k_output) / 1000`

Carbon (derived from energy + grid):
`carbon_g = energy_wh × PUE × grid_intensity / 1000`

Energy is the primary metric. It is derived from model-specific estimates (uncertain, tier 3) and exact token counts (from provider response).

Carbon is a derived metric. It compounds energy uncertainty with grid data. Grid intensity is real-time and accurate when available (signal_quality: live). Carbon inherits all uncertainty from energy, plus regional and temporal variation in grid mix.

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
| 3 | **Estimated** | Calculated from parameter count, architecture class, and theoretical compute requirements. | ±10x |

### Tier Confidence

**Tier 0 (Measured)**: Available when users run local GPU inference (Ollama, vLLM, llama.cpp) and use hardware sensors to capture actual power draw. Limitations: Requires compatible GPU (NVIDIA via pynvml, AMD via rocm-smi), baseline subtraction introduces noise, short inferences are less accurate.

**Tier 1 (Vendor-Published)**: The gold standard. If OpenAI published "GPT-4o uses 0.5 Wh per 1k tokens," we'd use that. Currently, no major provider publishes this data. Academic measurements on specific hardware with rigorous methodology qualify for Tier 1.

**Tier 2 (Validated)**: Aggregated from multiple crowdsourced Tier 0 measurements or independent academic studies. Example: "Llama 3.1 8B averages 0.12 Wh/1k tokens across 47 user submissions (std dev 0.03)."

**Tier 3 (Estimated)**: Current default for most models. Based on:
- Parameter count → FLOPs per token
- Architecture class (dense, MoE, hybrid)
- Hardware efficiency assumptions (H100 baseline)

Tier 3 estimates should be treated as order-of-magnitude guidance, not precise measurements.

### Architecture-Aware Estimation

For Mixture-of-Experts (MoE) models, we estimate energy based on **active parameters per token**, not total parameters. This prevents significant overestimation:

| Model | Total Params | Active Params | Correction Factor |
|-------|-------------|---------------|-------------------|
| GPT-4 | ~1.8T | ~220B | 8x |
| GPT-4o | ~200B | ~50B | 4x |
| Mixtral 8x7B | 47B | 13B | 3.6x |
| Gemini 1.5 Pro | ~500B | ~100B | 5x |

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

### Future: Crowdsourced Measurements (Vetch-Sensor)

We plan to enable crowdsourced energy measurements from local inference users:

```bash
# Calibrate your local setup (requires pynvml)
vetch calibrate --provider ollama --model llama3.1:8b

# Output:
# Baseline: 145W | Inference: 287W (Δ142W) | Duration: 3.2s
# Energy: 0.126 Wh / 1k tokens (vs registry: 0.089 Wh, Tier 3)
#
# Submit to Vetch Registry? [y/N]
```

Anonymized submissions (model + tokens + energy + hardware) will aggregate into Tier 2 estimates. No prompt content is ever transmitted.

## PUE (Power Usage Effectiveness)
Default: 1.1 (Uptime Institute 2023 hyperscaler average)

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
- **Tier 2**: Requires clear methodology, sample size ≥10, reproducible setup
- **Tier 3**: Theoretical estimates (parameter-based) - default for new models

### How to Submit
- Pull request against `registry/energy.json`
- Email to marco@prismaticlabs.ai
- Open an issue with the data
- (Coming soon) `vetch calibrate --submit` for automated Tier 0 → Tier 2 aggregation
