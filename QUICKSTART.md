# Vetch Quickstart

Stop runaway LLM inference. One import — all calls tracked, waste detected, spend attributed.

## Install

```bash
pip install vetch
```

## Instrument (One Line)

```python
import vetch

# Call at the top of your entry point. Instruments all LLM clients automatically.
vetch.instrument(region="us-east-1", tags={"service": "chat-api"})

# Now use any LLM client normally — no other changes needed.
import openai
client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}]
)
# Every call is now tracked: cost, energy, carbon, and waste advisory signals.
```

Works with: **OpenAI**, **Anthropic**, **Azure OpenAI**, **Vertex AI**, and any OpenAI-compatible endpoint (Ollama, vLLM, OpenRouter).

Non-blocking and fail-open — if Vetch fails for any reason, your LLM calls continue normally.

## Detect Waste

After letting Vetch observe real traffic, run the audit:

```bash
vetch audit              # last 7 days of stored metadata
vetch audit --window 24h # shorter window
vetch audit --format json
```

**Example output:**

```
[CRITICAL] STALL-001 — Stalled agent loop
  Session: chat-api · prod
  Calls in window: 20  |  Low-output calls: 17 (85%)  |  Input similarity: 70%
  Est. cost of stalled calls: $8.20  →  set_stall_action("kill")

[WARNING]  CACHE-001 — Prompt caching opportunity
  Session: document-qa · prod
  Calls with identical input tokens: 142 of 200 (71%)
  Potential saving: up to 90% on input tokens  →  enable cache_control

[INFO]     RAG-001 — RAG bloat
  Session: enterprise-chat · prod
  Avg input:output ratio: 82:1  →  tighten relevance threshold on retriever

Session summary: 1,842 requests · 2,847,000 tokens
```

Vetch observes metadata only — model, token counts, latency, region, and tags. It never reads prompts or completions.

## Stop Waste (Automatic Intervention)

Once you've seen the advisories and validated them, promote to automatic action:

```python
import vetch

vetch.instrument(region="us-east-1", tags={"service": "chat-api"})
vetch.set_stall_action("kill")   # or "warn" or "reroute"

# Your existing agent loop — unchanged.
# Vetch raises StallDetected before the next wasted call.
with vetch.Session() as session:
    try:
        response = client.chat.completions.create(...)
    except vetch.StallDetected:
        session.clear_stall()    # human-in-the-loop, then resume
```

**Reroute to a cheaper model automatically:**

```python
vetch.set_stall_action("reroute", fallback_model="gpt-4o-mini")
# On STALL-001, Vetch silently substitutes gpt-4o-mini for the stalled call.
```

## Attribute Spend

Tag every call to know which feature, customer, or team is driving cost:

```python
with vetch.wrap(tags={"feature": "rag-search", "customer": "acme"}) as ctx:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": query}]
    )

print(f"Cost:   ${ctx.event['estimated_cost_usd']:.5f}")
print(f"Energy: {ctx.event['estimated_energy_wh']:.4f} Wh")
print(f"Carbon: {ctx.event['estimated_carbon_g']:.4f} gCO2e")
```

Tags accumulate in stored metadata — `vetch audit --tags feature=rag-search` shows you only that feature's waste patterns.

## Verify Setup

```bash
vetch status
vetch estimate --model gpt-4o --input-tokens 1000 --output-tokens 500
```

## Region & Carbon Accuracy

For accurate carbon estimates, specify the region where inference runs:

```python
vetch.instrument(region="us-east-1")   # AWS/Azure/Vertex regions work
vetch.instrument(region="europe-west4") # Match your actual deployment
```

Set via environment to keep code region-agnostic:

```bash
export VETCH_REGION=us-east-1
```

**If you don't specify region:** Vetch falls back to `AWS_REGION` and similar environment variables, then a timezone-based heuristic. Carbon estimates will be less accurate.

## Async Support

```python
from openai import AsyncOpenAI
from vetch import awrap

client = AsyncOpenAI()

async with awrap(region="us-east-1", tags={"feature": "search"}) as ctx:
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": query}]
    )
```

## Configure Output

```bash
export VETCH_OUTPUT=/tmp/vetch.jsonl  # Log events to file
export VETCH_OUTPUT=none              # Silence event output
export VETCH_DISABLED=true           # Disable entirely
```

## Next Steps

1. **[Audit report](README.md#inference-waste-audit)** — full stored-event audit with filtering
2. **[Budget alerts](README.md#budget-alerts)** — cost/energy/carbon thresholds per session
3. **[Session aggregation](README.md#session-aggregation-agentic-ai)** — track agentic workflows end-to-end
4. **[OTLP export](README.md#otlp-export)** — send to Datadog, Grafana, Honeycomb
5. **[Energy methodology](src/vetch/METHODOLOGY.md)** — uncertainty tiers and provenance

---

**Questions or issues?** [github.com/prismatic-labs/vetch/issues](https://github.com/prismatic-labs/vetch/issues)
