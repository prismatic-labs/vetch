# Gemini Energy Calibration Methodology

**Version:** 1.0
**Date:** March 2026
**Status:** Tier 1 (Flash models - scenario-based ranges), Tier 3 (Pro models)

---

## ⚠️ CRITICAL: This Calibration Presents Ranges, Not Single Values

**This methodology cannot produce a single "correct" energy value.** We present plausible ranges based on different median prompt length assumptions.

**Why ranges?** Google's 0.24 Wh measurement [1] does not specify token count. All per-token estimates scale linearly with assumed prompt length.

**Recommended use:**
- **Short prompts** (Q&A): Use conservative estimates (0.273 Wh/1k input)
- **Long prompts** (code/docs): Use optimistic estimates (0.068 Wh/1k input)
- **Mixed workloads**: Use moderate estimates with ±50% uncertainty

---

## Executive Summary

**Input:** Google Environmental Report (Aug 2025): **0.24 Wh per median Gemini Apps text prompt** [1]

**Output:** Scenario-based energy ranges for Gemini Flash and Pro models

**Result for Gemini 2.0 Flash (input tokens):**
| Scenario | Assumed Median | Energy (Wh/1k) | Use Case |
|----------|----------------|----------------|----------|
| Conservative | 400 tokens | 0.273 | Short Q&A |
| Moderate | 800 tokens | 0.136 | Typical usage |
| **Optimistic** | **1,600 tokens** | **0.068** | **Long context (selected)** |
| Maximum | 2,000 tokens | 0.055 | Power users |

**Selected for vetch registry:** 0.068 Wh/1k (optimistic scenario) - balances conservatism with plausibility.

---

## Methodology

### Step 1: Extract IT-Level Energy

```
Total datacenter energy = 0.24 Wh [1]
PUE = 1.10 (Google's 2023 fleet average) [1]
IT equipment energy = 0.24 / 1.10 = 0.218 Wh
```

### Step 2: Scenario Analysis

**Formula:**
```
Let N_total = median prompt tokens (UNKNOWN - we model 4 scenarios)
Let N_in = N_out = N_total / 2 (equal split assumption)
Let E_out ≈ 3 × E_in (autoregressive overhead heuristic) [6]

Solve:
  IT_energy = (N_in × E_in) + (N_out × 3 × E_in)
  0.218 = 4 × N_in × E_in
  E_in = 0.218 / (4 × N_in)
```

**Scenario 1: Conservative (400-token median)**
- Assumes median by request count
- E_in = 0.218 / (4 × 200) = **0.273 Wh/1k**
- E_out = 0.273 × 3 = **0.819 Wh/1k**
- Validation: (200×0.273 + 200×0.819) / 1000 × 1.10 = 0.240 Wh ✓

**Scenario 2: Moderate (800-token median)**
- Excludes trivial prompts
- E_in = 0.218 / (4 × 400) = **0.136 Wh/1k**
- E_out = **0.409 Wh/1k**

**Scenario 3: Optimistic (1,600-token median) [SELECTED]**
- Assumes energy-weighted median
- E_in = 0.218 / (4 × 800) = **0.068 Wh/1k**
- E_out = **0.205 Wh/1k**
- **Rationale:** Balances uncertainty; plausible for energy-weighted distribution

**Scenario 4: Maximum (2,000-token median)**
- Heavy long-context usage
- E_in = 0.218 / (4 × 1,000) = **0.055 Wh/1k**
- E_out = **0.164 Wh/1k**

### Step 3: Propagate to Model Family

Using Google's efficiency claims [7][8]:
- Gemini 1.5 → 2.0: 16% improvement → ×1.16 multiplier
- Gemini 2.0 → 2.5: 25% improvement → ×0.80 multiplier

**⚠️ Caveat:** "Fewer tokens needed" may reflect better task performance, not reduced per-token compute. We conservatively interpret as per-token efficiency gains.

---

## Results: Selected Values (Optimistic Scenario)

| Model | Input (Wh/1k) | Output (Wh/1k) | Tier | Range (Conservative-Maximum) |
|-------|---------------|----------------|------|------------------------------|
| **gemini-2.0-flash** | **0.068** | **0.205** | **1** | 0.055 - 0.273 |
| **gemini-1.5-flash** | **0.079** | **0.238** | **1** | 0.064 - 0.317 |
| **gemini-2.5-flash** | **0.054** | **0.164** | **1** | 0.044 - 0.218 |
| gemini-1.5-pro | 0.302 | 0.909 | 3 | 0.244 - 1.211 |
| gemini-2.5-pro | 0.206 | 0.627 | 3 | 0.168 - 0.833 |

**Flash models:** Tier 1 (derived from Google measurement with documented assumptions)
**Pro models:** Tier 3 (theoretical 3.82x scaling from Flash)

---

## Assumptions & Uncertainty

| Assumption | Evidence | Uncertainty | Impact |
|------------|----------|-------------|--------|
| **Median prompt = 1,600 tokens** | Energy-weighted hypothesis | **±50%** | **Linear scaling** |
| Output/input ≈ 3:1 | Autoregressive heuristic [6] | ±25% | Input/output split |
| Anchor = Flash-class model | Timeline | Medium | Could be 1.5 or 2.0 |
| PUE = 1.10 | Google fleet average [1] | ±5% | 1.05-1.15 range |

**Key limitation:** Google has not published median token count or clarified whether "median" refers to request count or energy consumption.

---

## Validation

✅ All scenarios reproduce 0.24 Wh ±0.001 Wh
✅ Efficiency ratios preserved (1.16x, 0.80x)
✅ Output/input ratio maintained (~3:1)

**Cross-validation needed:**
- Google publishes median token count with methodology
- Independent prompt length distribution analysis
- Local inference measurements (expect 2-5x higher due to batching)

---

## References

[1] **Google Cloud Blog (Aug 2025):** "Measuring the environmental impact of AI inference"
https://cloud.google.com/blog/products/infrastructure/measuring-the-environmental-impact-of-ai-inference
*0.24 Wh per median Gemini Apps prompt. PUE 1.10.*

[2] **JS Interactive (2024):** "ChatGPT Statistics & Trends"
https://js-interactive.com/chatgpt-trends-report-statistics/
*Median prompt: 10 words (~13 tokens). Average: 70 words.*

[3] **Google Support:** "Gemini Apps limits & upgrades"
https://support.google.com/gemini/answer/16275805
*Free tier: ~50 prompts/day suggests substantial complexity per prompt.*

[4] **Google AI Developers:** "Understand and count tokens"
https://ai.google.dev/gemini-api/docs/tokens
*1 token ≈ 4 characters. Context windows: 128k-200k tokens.*

[5] **OpenRouter (2025):** "State of AI 2025: 100T Token Study"
https://openrouter.ai/state-of-ai
*Heavy-tailed distribution in LLM usage.*

[6] **Industry heuristic:** Autoregressive generation
*Output ≈ 2-4x compute per token vs input. 3:1 is midpoint.*

[7] **Google Developers Blog:** "Gemini 2.0 family expands"
https://developers.googleblog.com/en/gemini-2-family-expands/
*16% efficiency improvement 1.5 → 2.0.*

[8] **Google Blog:** "Gemini 2.5: Most intelligent models"
https://blog.google/innovation-and-ai/models-and-research/google-deepmind/google-gemini-updates-io-2025/
*"20-30% fewer tokens needed" for 2.5 vs 2.0.*

[9] **arXiv:2508.15734:** "Measuring Environmental Impact at Google Scale"
https://arxiv.org/abs/2508.15734
*Full methodology.*

---

## Recommendations

**For carbon accounting:**
- Use **Conservative scenario** (0.273 Wh/1k) for safety margin
- Document assumptions in reporting
- Apply ±50% uncertainty bars

**For optimization:**
- Measure actual prompt lengths in your application
- Select scenario matching your workload
- Validate with local measurements if possible

**Call to action:** We need Google to publish:
1. Median token count for the 0.24 Wh measurement
2. Clarification: median by request count or energy consumption?
3. Distribution of prompt lengths (not just median)

---

## Changelog

**Version 1.0 (March 2026)**
- Initial calibration from Google's 0.24 Wh
- Four-scenario analysis (400, 800, 1,600, 2,000 tokens)
- Selected optimistic scenario (1,600 tokens) for vetch registry
- Flash models: Tier 1 with documented uncertainty
- Pro models: Tier 3 (theoretical)

---

**This methodology prioritizes transparency over precision.** We cannot eliminate uncertainty without Google publishing token counts. Use these ranges with documented assumptions.
