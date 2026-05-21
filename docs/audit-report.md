# 7-Day Inference Waste Audit

This document defines the Inference Waste Audit — a concrete adoption motion that produces a structured picture of where inference spend is going and what is wasting it.

**Current state of `vetch audit`:** The CLI command produces an advisory list with session context and a session summary (total requests, total tokens, average input:output ratio). Implemented advisories include STALL-001, CACHE-001, RAG-001, BABBLE-001, ZOMBIE-001, CTX-001, EMPTY-001, and TRUNC-001. The full report format described below is planned. Sections marked `[PLANNED]` are not yet produced by the current CLI.

Audit advisories are deterministic signals from metadata, not proof of waste. Confidence labels indicate signal strength and review priority, not statistical certainty. Automatic kill and reroute are currently scoped to confirmed STALL-001 patterns; other advisories should feed a review and remediation queue.

---

## How to run the audit

### Step 1 — Instrument (Day 1)

```bash
pip install vetch
```

```python
import vetch
vetch.instrument(region="us-east-1", tags={"service": "my-service"})
vetch.set_stall_action("warn")  # observe without blocking
```

One import, one line. Every LLM call across all providers is now tracked.

### Step 2 — Tag your calls (Days 1–7)

Add tags to attribute spend by feature, customer, or workflow:

```python
with vetch.wrap(tags={"feature": "rag-search", "customer": "acme", "env": "prod"}) as ctx:
    response = client.chat.completions.create(...)
```

You do not need tags on every call to run the audit — but tagged calls produce a richer report.

### Step 3 — Run the audit (Day 7)

```bash
vetch audit
```

### Step 4 — Act on the findings

Promote confirmed `STALL-001` findings from warn to kill or reroute after you have checked the workflow:

```python
vetch.set_stall_action("kill")
```

For non-stall advisories, fix the underlying workflow first: cache setup, retriever limits, response caps, context trimming, or attribution gaps.

---

## Target report format

The following is the target output format for `vetch audit`. Sections marked `[PLANNED]` are not produced by the current CLI but define the intended output.

---

### Executive summary `[PLANNED]`

```
Inference Waste Audit — 7 days ending 2026-05-05
─────────────────────────────────────────────────
Estimated monthly inference spend:    $4,200
Estimated avoidable spend:            $1,050  (25%)
Estimated avoided tokens (potential): 42M
Estimated avoided energy (potential): 18.4 Wh
Estimated avoided carbon (potential): 8.0 gCO2e

Top waste sources:
  1. STALL-001   rag-search feature         $620/mo estimated avoidable
  2. CACHE-001   document-qa feature        $310/mo estimated avoidable
  3. RAG-001     enterprise-chat feature    $120/mo estimated avoidable
```

All figures are estimates with uncertainty. Energy, carbon, and water estimates use registry data at different confidence tiers depending on model and region. See the methodology note at the end of the report.

---

### Advisory events `[Current CLI produces this]`

```
[CRITICAL] STALL-001 — Stalled agent loop
  Session: rag-search · acme · prod
  Calls in window: 20 | Low-output calls: 17 (85%) | Input similarity: 70%
  Estimated cost of stalled calls: $8.20
  Recommended action: set_stall_action("kill")

[WARNING] CACHE-001 — Prompt caching opportunity
  Session: document-qa · prod
  Calls with identical input tokens: 142 of 200 (71%)
  Potential saving: up to 90% on input tokens with Anthropic prompt caching
  Recommended action: enable cache_control on static system prompt

[INFO] RAG-001 — RAG bloat
  Session: enterprise-chat · prod
  Average input:output ratio: 82:1
  Recommended action: tighten relevance threshold on retriever
```

---

### Session summary `[Current CLI produces this]`

```
Session Summary
──────────────────────────────
Total requests:            1,842
Total tokens:          2,847,000
Avg input:output ratio:    34.2:1
```

---

### Waste by dimension `[PLANNED]`

```
Cost by feature (last 7 days)
─────────────────────────────────────────────────────────
Feature              Calls    Cost      Energy    Carbon
rag-search           8,420   $1,840    42.1 Wh   18.3 gCO2e
document-qa          4,210   $620      14.2 Wh    6.2 gCO2e
enterprise-chat      2,180   $380       8.7 Wh    3.8 gCO2e
(untagged)           1,100   $290       6.6 Wh    2.9 gCO2e
─────────────────────────────────────────────────────────
Total               15,910  $3,130     71.6 Wh   31.2 gCO2e

Cost by model (last 7 days)
─────────────────────────────────────────────────────────
Model                Calls    Cost      Energy    Carbon
gpt-4o              12,400   $2,480    56.4 Wh   24.6 gCO2e
claude-3.7-sonnet    2,800   $520      12.6 Wh    5.5 gCO2e
gpt-4o-mini            710   $130       2.6 Wh    1.1 gCO2e
─────────────────────────────────────────────────────────

Estimated avoidable spend by advisory
─────────────────────────────────────────────────────────
Advisory             Occurrences   Est. avoidable/month
STALL-001                    12            $620
CACHE-001                     6            $310
RAG-001                       3            $120
─────────────────────────────────────────────────────────
Total                        21          $1,050
```

---

### Recommended policies `[PLANNED]`

```
Based on advisory events, the following policies are recommended:

1. STALL-001 on rag-search (HIGH signal)
   Action: set_stall_action("kill")
   Rationale: 12 stall events over 7 days, avg cost per stall $8.20
   Expected monthly saving: $620

2. CACHE-001 on document-qa (MEDIUM signal)
   Action: Enable Anthropic prompt caching on system prompt
   Rationale: 71% of calls share identical input token counts
   Expected monthly saving: up to $310

3. RAG-001 on enterprise-chat (LOW signal)
   Action: Investigate retriever — consider relevance threshold of 0.7+
   Rationale: 82:1 average input:output ratio; verify this is not a
   legitimate summarization workload before acting
```

---

### Avoided estimates `[PLANNED]`

Estimates of what would have been avoided if recommended policies had been in effect for the full 7-day period.

```
If STALL-001 kill policy had been active:
  Calls stopped (est.):         ~145
  Tokens avoided (est.):        ~2.4M
  Cost avoided (est.):          $145
  Energy avoided (est.):        3.3 Wh  (Tier 1, ±50%)
  Carbon avoided (est.):        1.4 gCO2e  (±50%, global avg grid)

If CACHE-001 prompt caching had been enabled:
  Tokens cached (est.):         ~840k input tokens/day
  Cost avoided (est.):          ~$72/week
  Energy avoided (est.):        ~1.6 Wh/week  (Tier 1, ±50%)
```

All avoided estimates are directional — they assume the waste pattern continues unchanged without intervention, and do not account for changes in traffic or model behaviour.

---

### Methodology and confidence notes

```
Cost figures: calculated from provider pricing tables in Vetch model registry.
Advisory confidence: signal strength from metadata thresholds, not statistical certainty.
Energy figures: Tier 1 (±20–50%) for measured models; Tier 3 (order-of-magnitude)
  for unmeasured models. See vetch methodology for full tier definitions.
Carbon figures: derived from energy × PUE × grid intensity.
  Grid intensity: static regional annual averages (last updated Jan 2024).
  These are indicative, not certified. Provider renewable energy commitments
  may reduce effective intensity below regional average.
Water figures: directional operational cooling estimates. Global default 1.7 L/kWh;
  regional WUE in SDK only. Not standalone water accounting.

Avoidable spend estimates assume waste patterns are stable over the billing period.
Actual savings will vary.
```

---

### Sustainability appendix `[PLANNED]`

For teams preparing internal sustainability, FinOps, or engineering reduction inputs:

```
Inference emissions summary (7 days)
─────────────────────────────────────────────────────────
Total energy consumed (est.):       71.6 Wh  [Tier 1/3 mix]
Total carbon emitted (est.):        31.2 gCO2e
  — of which avoidable waste:        8.0 gCO2e  (26%)

Methodology: top-down estimation from token counts × energy intensity.
Uncertainty: ±50% for Tier 1 models; order-of-magnitude for Tier 3.
These figures are suitable for directional reduction tracking and internal
reporting. They are not suitable for regulatory disclosure, water accounting,
certification, or external carbon claims without independent verification.

Region used: us-east-1 (380 gCO2e/kWh, static annual average, IEA 2023)
```
