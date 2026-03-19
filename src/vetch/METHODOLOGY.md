# Vetch Methodology

methodology_version: "1.0"
sdk_version: "0.2.3"

## Preamble
Vetch exists because AI systems currently operate with no feedback on their energy consumption. Every inference draws power from infrastructure with real costs—financial, environmental, and systemic. None of this is visible to the developer making the API call.

This methodology is our first attempt to create that feedback loop. It is imperfect. The energy estimates are uncertain. We publish it openly so it can be challenged, corrected, and improved.

We believe imperfect measurement, honestly reported, is better than no measurement at all.

## SDK Instrumentation Model

**As of v0.2.2, automatic instrumentation is production-ready.**

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

**Streaming (v0.2.3):** Streaming calls (`stream=True`) are now fully tracked under `instrument()`. The event is emitted when the stream is exhausted (last chunk consumed). If the stream is abandoned mid-way, the event is still emitted with `complete=False` and the characters counted so far.

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
| 3 | **Estimated** | Calculated from parameter count, architecture class, and theoretical compute requirements. | order of magnitude |

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

### 2025 Breakthrough: Tier 1 Measurements

**Major Update (March 2026):** Vetch 0.1.7 incorporates the first large-scale, infrastructure-aware benchmarking of LLM energy consumption from **Jegham et al. (2025)** published in "How Hungry is AI? Benchmarking Energy, Water, and Carbon Footprint of LLM Inference" (arXiv:2505.09598).

This research provided hardware measurements for 30 state-of-the-art models deployed in commercial datacenters, enabling us to upgrade key models from Tier 3 (estimated) to **Tier 1 (measured)**:

| Model | Status | Energy (medium prompt) | Tier |
|-------|--------|----------------------|------|
| GPT-4.1 nano | ✅ Upgraded | 0.271 Wh (most efficient) | 1 |
| GPT-4o | ✅ Upgraded | 1.214 Wh | 1 |
| Claude-3.7 Sonnet | ✅ Upgraded | 2.781 Wh | 1 |
| o1 | ✅ Added | 12.1 Wh (reasoning model) | 1 |
| o3 | ✅ Added | 21.4 Wh (advanced reasoning) | 1 |
| DeepSeek-R1 | ✅ Added | 29.0 Wh (most intensive) | 1 |

**Key Findings:**
- **Reasoning models consume 40-100x more energy** than efficient models like GPT-4.1 nano
- **Non-linear energy scaling:** Short prompts use more energy per token than long prompts due to fixed overhead costs
- **Range of efficiency:** Most energy-intensive model (DeepSeek-R1) consumes **107x more** than the most efficient (GPT-4.1 nano) for identical prompts

### Non-Linear Energy Model (0.1.7+)

**Critical Discovery:** Energy consumption is **not linear** with token count. Jegham's measurements reveal:

**GPT-4o Energy by Prompt Length:**
- **Short** (<1k tokens): 1.05 Wh per 1k tokens
- **Medium** (1k-5k tokens): 0.61 Wh per 1k tokens
- **Long** (>5k tokens): 0.16 Wh per 1k tokens

Longer prompts are ~6x more efficient per-token due to amortization of fixed costs (model loading, memory allocation, batching overhead).

**Implementation:** Vetch 0.1.7 introduces prompt-length-aware coefficients for models with measured data. The calculation engine automatically selects appropriate coefficients based on total token count:

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

**Primary Sources (Tier 1):**

1. **Jegham, N., et al. (2025).** "How Hungry is AI? Benchmarking Energy, Water, and Carbon Footprint of LLM Inference."
   - Published: May 2025 (arXiv:2505.09598)
   - Methodology: Hardware measurements (pynvml) in commercial datacenters
   - Coverage: 30 models including GPT-4o, Claude-3.7, o3, DeepSeek-R1
   - Impact: First infrastructure-aware benchmark at scale
   - URL: https://arxiv.org/abs/2505.09598

2. **Google Environmental Report (August 2025).** "Measuring the Environmental Impact of Delivering AI at Google Scale."
   - Data: Median Gemini Apps text prompt consumes 0.24 Wh
   - Methodology: Full stack accounting (TPU, CPU, datacenter overhead)
   - PUE: 1.10 (2023 fleet average)
   - URL: https://cloud.google.com/blog/products/infrastructure/measuring-the-environmental-impact-of-ai-inference

3. **Epoch AI (2025).** "How Much Energy Does ChatGPT Use?"
   - Data: GPT-4o estimated at ~0.3 Wh per query (assumption: 500 token response)
   - Methodology: Analytical model based on MoE architecture assumptions
   - URL: https://epoch.ai/gradient-updates/how-much-energy-does-chatgpt-use

**Why Jegham (2025) is Authoritative:**
- **Infrastructure-aware:** Accounts for batching, memory bandwidth, idle capacity
- **Production environments:** Commercial datacenters, not lab setups
- **Reproducible:** Published methodology and measurement scripts
- **Peer-reviewed:** Academic rigor with clear error bounds

**Comparison with Earlier Estimates:**
Earlier widely-cited research (de Vries, 2023) estimated ~3 Wh per GPT-3 query. Jegham's measurements show GPT-4o at ~0.3 Wh for short queries—**10x more efficient** than earlier estimates, reflecting both architectural improvements and measurement accuracy.

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

PUE measures datacenter overhead efficiency (cooling, power distribution). Vetch uses **provider-specific PUE values** from official sustainability reports:

| Provider | PUE | Tier | Source |
|----------|-----|------|--------|
| **Google Cloud** | 1.10 | 1 | [Google Data Centers Efficiency Report 2023](https://datacenters.google/efficiency/) |
| **Microsoft Azure** | 1.12 | 1 | [Microsoft Datacenters Sustainability 2024](https://datacenters.microsoft.com/sustainability/efficiency/) |
| **AWS** | 1.15 | 1 | [AWS Sustainability Report 2024](https://aws.amazon.com/sustainability/data-centers/) |
| **Unknown/Default** | 1.2 | 3 | Industry average (Uptime Institute 2023) |

**PUE Tier Definitions:**
- **Tier 1**: Known value (vendor-published OR user-configured via `VETCH_DEFAULT_PUE`)
- **Tier 3**: Default fallback when provider unknown (1.2 industry average)

Note: There is no Tier 0 or Tier 2 for PUE because:
- No "measured PUE" exists (users don't have datacenter-level sensors)
- No "crowdsourced PUE" makes sense (PUE is a facility property, not per-request)

**Auto-detection:** Vetch infers provider from model name:
- `gpt-*`, `o1-*`, `o3-*` → OpenAI (Azure-backed, PUE 1.12)
- `claude-*` → Anthropic (AWS-backed, PUE 1.15)
- `gemini-*`, `gemma-*` → Vertex AI (Google Cloud, PUE 1.10)

### PUE Limitations

**Critical caveats:**

1. **Fleet averages, not inference-specific**: PUE reports include training workloads (high utilization) and inference (bursty, lower utilization). Inference PUE may be 10-20% worse than fleet average.

2. **Regional variation ignored**: Google's PUE ranges from 1.04 (best datacenter) to 1.20+ (worst). We use fleet average (1.10), which masks 5-15% regional differences.

3. **Provider inference is fragile**: We guess provider from model name. If OpenAI switches from Azure to AWS, or uses multi-cloud routing, our PUE will be incorrect until we update the mapping.

4. **PUE measures datacenter, not model efficiency**: A model on a V100 GPU in a PUE 1.09 datacenter may use MORE energy than the same model on an H100 in a PUE 1.15 datacenter. **GPU generation matters more than PUE.**

5. **Embodied carbon excluded**: PUE only measures operational energy. Manufacturing GPUs, building datacenters, and decommissioning equipment contribute 10-20% of lifecycle emissions but are not captured by PUE.

**Bottom line:** Provider-specific PUE improves the datacenter efficiency component of carbon calculation by ~5-10%, but **overall carbon estimates remain order-of-magnitude** due to Tier 3 energy-per-token estimates and regional grid variations (which matter 10-100x more than PUE).

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
