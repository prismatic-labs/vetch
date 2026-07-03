# Vetch Energy Registry Provenance

This document details the methodology and sources used for the Tier 3 (Estimated) values in `energy.json`.

## Methodology Overview

All Tier 3 values are derived using a **Proxy Scaling Model**. Since model providers do not currently publish per-inference energy data, we use three primary signals:

1.  **Parameter Count Proxies**: We map models to their estimated parameter counts (e.g., GPT-4 ~1.8T MoE, Llama-3-70B).
2.  **Hardware Efficiency Baselines**: We assume inference on NVIDIA H100 or A100 GPUs. Efficiency is calculated using a baseline of ~0.2 - 0.5 Joules per parameter per token, adjusted for MoE (Mixture of Experts) activation ratios.
3.  **The 1:3 Input-to-Output Heuristic**:
    *   **Input (Prefill)**: KV-cache population. This is compute-bound but highly parallelizable.
    *   **Output (Decode)**: Autoregressive generation. This is memory-bandwidth bound and requires one full forward pass per token.
    *   *Source*: Based on architectural analysis in *Pope et al. (2022) "Efficiently Scaling Transformer Inference"*.

## Model-Specific Baselines

### GPT-4 Family
*   **Source**: Estimates from *SemiAnalysis* blog and *Luccioni et al. (2023) "Power Hungry Processing"*.
*   **Logic**: Scaled from Bloom (175B) measurements, adjusted for the MoE architecture which reduces active parameters per token while maintaining high memory overhead.

### Claude 3 Family
*   **Source**: Benchmarked latency scaling relative to GPT-4.
*   **Logic**: Energy is correlated with latency under similar load. Claude 3 Opus displays latency patterns consistent with ~1.5x the compute intensity of GPT-4 Turbo.

### Llama 3 Family
*   **Source**: Meta's published training energy, reverse-scaled for inference ratios.
*   **Logic**: Uses official 70B parameter count as the anchor point for the registry's linear scaling.

## Uncertainty Margin
**Confidence Level: Low (order of magnitude)**

### The 1.2x Hardware Overhead Multiplier

Registry values represent **IT equipment energy** (GPU + server overhead), *not* total facility energy. PUE is applied separately in `calculate_carbon()`.

To account for non-GPU server components, we apply a **1.2x Hardware Overhead Multiplier** to raw GPU benchmarks:

*   **Networking, CPU host coordination, and memory-bandwidth idle draw**: 1.2x

This multiplier covers the energy consumed by server components beyond the GPU itself (CPUs, DRAM, NVLink, host networking). It does **not** include data center cooling or power distribution — those are accounted for by the PUE factor applied during carbon calculation.

**PUE (Power Usage Effectiveness)** is applied separately in `calculate_carbon()` with a configurable default of 1.2, based on:
*   Google: 1.09 (fleet average, Google 2026 Environmental Report, FY2025)
*   AWS: 1.14 (AWS 2025 Sustainability Report, global average)
*   Azure: 1.12 (Microsoft 2025 Environmental Sustainability Report, newest-gen)
*   Industry average: ~1.54 (Uptime Institute 2025)

Override via `VETCH_DEFAULT_PUE` environment variable.

### Model Assumptions and Limitations

*   Energy per token is modeled as **independent of context length**. In reality, output token energy increases with KV-cache size (longer inputs → more expensive outputs). This is a known simplification.
*   **Batch size effects** are not modeled. Higher batch sizes improve throughput efficiency but are provider-controlled.

These values are intended for **relative comparison** and **trend analysis**, not audit-grade reporting.

## Tier 1 Entries: Jegham et al. (2025)

As of v0.2.4, most major commercial models are upgraded to **Tier 1** using empirical measurements from:

> Jegham, N. et al. (2025). "How Hungry is AI? Benchmarking Energy, Water, and Carbon Footprint of LLM Inference." arXiv:2505.09598 (current: v6, revised 2025-11-24)

### Coverage (v0.2.4)

**Jegham-measured (Tier 1, prompt-length-aware):**
GPT-4 Turbo, GPT-4o, GPT-4o-mini, GPT-4.1, GPT-4.1-mini, GPT-4.1-nano, GPT-4.5, o1, o3, o3-mini, o4-mini, DeepSeek-R1, DeepSeek-V3, LLaMA 3.1 8B/70B/405B, LLaMA 3.3 70B, Claude 3.7 Sonnet, Claude 3.7 Sonnet (Extended Thinking)

**Jegham v6 — figures reported, registry upgrade pending verification:**
Claude 3.5 Haiku, Claude 3.5 Sonnet — appear in Jegham et al. v6 (arXiv:2505.09598v6, revised 2025-11-24) with measured energy figures. Both remain Tier 3 in Vetch pending confirmation of per-scenario (short/medium/long) values and correct PUE adjustment. See Known Registry Gaps.

**Note on source version:** Existing Tier 1 `basis` strings cite `arXiv:2505.09598` without a version pin. The paper is now at v6. Existing coefficients have not been reconciled against v6 figures; any differences are expected to fall within measurement uncertainty (±20–50%) but should be confirmed before audit-grade use. New entries added from v6 onward should include the version string in their `basis`.

**No Jegham coverage (Google models, Tier 3):**
Jegham et al. (2025) does **not** cover any Google/Gemini models. Gemini 2.0 Flash is anchored to Google's published 0.24 Wh median Gemini Apps text prompt, but that source does not publish token counts, input/output split, or per-model coefficients, so it remains Tier 3. Gemini 2.5 Flash and Pro also use Tier 3 proxy estimates only.

### Methodology

Jegham reports per-query energy (Wh) for three prompt scenarios:
- **Short:** 100 input + 300 output tokens
- **Medium:** 1,000 input + 1,000 output tokens
- **Long:** 10,000 input + 1,500 output tokens

Per-token coefficients are derived using the 3:1 output/input ratio:
```
wh_in  = E_wh × 1000 / (in_tokens + 3 × out_tokens)
wh_out = 3 × wh_in
```

**Limitation:** The 3:1 ratio is an architectural assumption from Pope et al. (2022), not a direct per-phase measurement from Jegham's data. Jegham reports per-query totals only; the input/output split is inferred. Phase-aligned benchmarks (e.g. TokenPowerBench, arXiv:2512.03024) provide prefill/decode attribution directly. Future registry entries may adopt per-phase fields (`prefill_wh_per_1k`, `decode_wh_per_1k`) once phase-level measurements are validated against the same infrastructure scope. Existing entries use the 3:1 heuristic for compatibility and should not be compared directly to phase-aligned figures without accounting for this difference.

### PUE Handling (Critical)

Jegham measures energy at the API layer (server IT equipment draw) **before** datacenter PUE overhead. Registry values are therefore **IT-equipment-only (pre-PUE)**. Vetch applies PUE once in `calculate_carbon()` via `VETCH_DEFAULT_PUE` (default 1.2×). Never apply PUE to registry values directly — that was the v0.1.6 double-counting bug. Every Jegham-derived `basis` string explicitly states: "IT equipment energy only (pre-PUE). Vetch applies PUE separately in calculate_carbon()."

Jegham reports datacenter PUE context for transparency (AWS 1.14, Azure 1.12) but does **not** include it in per-token figures.

### Extended Thinking Auto-Detection (v0.2.4)

When `thinking={"type": "enabled"}` is passed to `anthropic.messages.create()`, Vetch automatically appends `-thinking` to the model name before registry lookup. This resolves to `claude-3.7-sonnet-thinking` which has higher energy coefficients reflecting measured Extended Thinking overhead.

### Known Registry Gaps

**claude-3.5-sonnet** — Currently Tier 3 (±1000%). Jegham et al. v6 (arXiv:2505.09598v6) reports measured figures. Registry upgrade pending verification of per-scenario (short/medium/long) values and correct PUE divisor (AWS 1.14).

**claude-3.5-haiku** — Currently Tier 3. Previously noted as "listed but not measured" in the version of Jegham consulted during v0.2.4. Jegham v6 appears to include measured figures. Same verification requirements as claude-3.5-sonnet. Note: Jegham's data shows smaller or less-efficient models sometimes use more energy than larger ones at medium prompt length (batching overhead dominates); verify figures before adding a basis note on this.

**Indic scripts** — Devanagari, Bengali, Tamil, and similar scripts tokenize poorly in all current models (3–8 Unicode code points per token). The char-count fallback will underestimate token counts for Indic text.

## Measurement Kind Taxonomy

Each registry source can be classified by how its energy data was obtained. This is prose-only for now; it is not yet a JSON field in `energy.json`. The taxonomy is intended to guide future provenance documentation and to clarify which sources can support Tier promotion.

| Kind | Definition | Tier eligibility |
|---|---|---|
| `api_observed` | Measured at the provider API boundary in commercial infrastructure | Tier 1 eligible with publication and clear methodology |
| `provider_disclosure` | Vendor-published aggregate or product-level figure; no per-model or per-token breakdown | Sanity anchor only; not Tier 1 |
| `local_measurement` | User-calibrated via GPU sensors (pynvml, rocm-smi) on owned hardware | Tier 0 (user-calibrated) or Tier 2 (crowdsourced via `vetch calibrate --submit`) |
| `proxy` | Derived from parameter counts, architecture class, or similar-model analogy | Always Tier 3 |

**Key distinctions:**

- `api_observed` is the only kind eligible for automatic Tier 1 promotion. It requires a peer-reviewed publication or official provider disclosure with a stated measurement methodology and scope.
- `provider_disclosure` data (Google's 0.24 Wh median Gemini Apps prompt; OpenAI's 0.34 Wh average ChatGPT query) gives a useful sanity anchor against aggregate product behaviour but cannot substitute for per-model, per-token coefficients. These sources stay at Tier 3.
- `local_measurement` via TokenPowerBench (arXiv:2512.03024) or `vetch calibrate` reaches Tier 0 on the user's own hardware. Submitted aggregates reach Tier 2 once sample size and methodology quality thresholds are met.
- `proxy` is always Tier 3 regardless of how carefully the scaling model is constructed.

The existing Jegham-derived entries are `api_observed`. The GPT-5 family entries are `proxy`. All Gemini entries are currently `proxy` with a `provider_disclosure` sanity check noted in GEMINI_CALIBRATION.md.

## Research Context: Cost of Reasoning Strategies

From Aglin et al. (2026), arXiv:2603.20224 — "Beyond Test-Time Compute Strategies: Advocating Energy-per-Token in LLM Inference":

- **Chain-of-thought prompting on smaller models uses 120–150× more energy** than baseline inference for the same task, due to dramatically longer output sequences.
- **Majority voting** adds 72–177% energy with minimal accuracy improvement.
- The paper advocates intelligent query routing — selectively deploying reasoning techniques only where task complexity justifies the cost — rather than applying CoT or extended thinking universally.

**Relationship to vetch's Extended Thinking tracking:** The 2–4× energy overhead vetch measures for `claude-3.7-sonnet-thinking` reflects API-level measurement at medium prompt length. The 120–150× figure is a task-level finding for CoT on small models (LLaMA 1B/8B on MMLU), where reasoning overhead dominates. Both are valid in their context — the Aglin figure illustrates the extreme end of the reasoning cost spectrum.

**Registry implication:** Vetch currently has no mechanism to track whether a reasoning strategy (CoT prompt, extended thinking, majority voting) was used. `vetch.thinking_mode` (added v0.2.4) covers the Extended Thinking case explicitly. General CoT via prompt engineering remains invisible to the SDK.

## Registry freshness process

The registry drifts behind reality every time a provider ships a model. When a new
frontier model appears:

1. **Energy** — add an `energy.json` row. If no direct measurement or Jegham entry
   exists, proxy from the nearest same-class prior-gen sibling (Pro→Pro, Flash→Flash,
   Opus→Opus) at **tier 3**, and say so in the `basis` string ("No published energy
   figures or Jegham measurement... proxied from... Tier 3 ±1000%"). When in doubt,
   proxy from the *higher*-energy sibling — never silently undercount.
2. **Pricing** — add a `pricing.json` row with **verified** numbers from the provider's
   official pricing page, plus an `as_of` date. Never proxy or invent a price.
3. **Aliases** — add the dated / `-preview` / `-latest` forms to `aliases.json`,
   pointing at the canonical (suffix-free) key.

`scripts/check_registry_freshness.py` (run in CI) enforces energy↔pricing parity and
flags priced rows whose `as_of` is over a year old.

**v0.9.0 rows (verified 2026-06-23):** Gemini `3-flash` ($0.50/$3.00), `3.5-flash`
($1.50/$9.00), `3.1-flash-lite` ($0.25/$1.50), `3.1-pro` ($2/$12, 2x input >200k)
from ai.google.dev/gemini-api/docs/pricing; Claude `sonnet-4-5`/`sonnet-4-6`
($3/$15) and `opus-4-5`/`-4-6`/`-4-7`/`-4-8` ($5/$25) from the official Anthropic
pricing page (platform.claude.com), which lists Opus 4.5 through 4.8 at the same
$5/$25 rate. Energy for all is Tier-3 proxy (no measurement exists). Open-weight / self-hosted rows can
be cross-validated against the direct-measurement sources below (ML.ENERGY, Samsi,
TokenPowerBench); closed hosted-API models have no public power telemetry and stay
inferred/Tier 3 by necessity.

## Model capability map (`model_capabilities.json`)

**File:** `src/vetch/registry/model_capabilities.json`

**Purpose:** Kind C capability observability — maps model name prefixes/families to declared capability kinds (e.g. `chat`, `embedding`, `image`) for `capabilities_invoked` derivation when no function tools fire.

**Methodology:**
- Curated from public provider documentation and model cards (OpenAI, Anthropic, Google, Meta, Mistral, etc.)
- Prefix/family keys match the same resolution order as `energy.json` / pricing aliases
- Overrides at runtime via `vetch.set_model_capability_map()` for private or fine-tuned models

**Refresh:** `python scripts/check_registry_freshness.py` in CI; bump `_comment` version when editing.

**Confidence:** Medium for major hosted models; low for niche or rapidly renamed SKUs — treat silent Kind C as directional, not proof of missing features.

## References

*   Pope, R. et al. (2022). "Efficiently Scaling Transformer Inference." MLSys 2023. arXiv:2211.05102
*   Luccioni, A.S. et al. (2023). "Power Hungry Processing: Watts Driving the Cost of AI Deployment?" FAccT 2024. arXiv:2311.16863
*   Jegham, N. et al. (2025). "How Hungry is AI? Benchmarking Energy, Water, and Carbon Footprint of LLM Inference." arXiv:2505.09598 (current: v6, revised 2025-11-24)
*   Uptime Institute (2023). Global Data Center Survey.
*   Aglin, G. et al. (2026). "Beyond Test-Time Compute Strategies: Advocating Energy-per-Token in LLM Inference." arXiv:2603.20224
*   Samsi, S. et al. (2023). "From Words to Watts: Benchmarking the Energy Costs of Large Language Model Inference." arXiv:2310.03003 (direct GPU power measurement, open models)
*   Chung, J.-W. et al. (2025). "The ML.ENERGY Benchmark: Toward Automated Inference Energy Measurement and Optimization." arXiv:2505.06371 (open data + Zeus toolkit; direct power metering of open-weight models)
*   TokenPowerBench (2025). "Benchmarking the Power Consumption of LLM Inference." arXiv:2512.03024
