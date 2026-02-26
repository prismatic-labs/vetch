# Vetch Quickstart

Get planet-aware LLM observability in 60 seconds.

## Install

```bash
pip install vetch
```

## Track All Calls (One Line)

```python
import vetch

# Best practice: Call instrument() at the very top of your entry point
# Pro tip: Set VETCH_REGION=us-east-1 via environment to keep code region-agnostic
vetch.instrument(region="us-east-1", tags={"service": "chat-api"})

# Non-blocking and fail-open: if Vetch fails, your LLM calls still succeed
# Overhead: <5ms per call. TTFT: Zero added latency for streaming.

# Now use any LLM client normally (requires: pip install openai)
import openai
client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}]
)
# ✅ Energy, cost, and carbon tracked automatically
```

**Output example** (logged to stderr by default):
```json
{
  "estimated_cost_usd": 0.0025,
  "estimated_energy_wh": 0.0234,
  "estimated_carbon_g": 1.87,
  "model": "gpt-4o",
  "tags": {"service": "chat-api"}
}
```

Works with: **OpenAI**, **Anthropic**, **Azure OpenAI**, **Vertex AI**, and OpenAI-compatible APIs (OpenRouter, Together.ai, Ollama, vLLM).

## Verify It's Working

```bash
# Check Vetch status
vetch status

# See metrics in action (no API key needed)
vetch estimate --model gpt-4o --input-tokens 1000 --output-tokens 500
```

## Region & Carbon Accuracy

For accurate carbon calculations, specify your region (should match [Electricity Maps zone IDs](https://app.electricitymaps.com/map)):

```python
# For providers where you control region (Azure, Vertex, Bedrock):
vetch.instrument(region="us-east-1")  # Match your actual deployment region

# For providers with opaque routing (OpenAI, Anthropic):
vetch.instrument(region="us-east-1")  # Your best estimate or data center preference
```

**Which providers support region control?**

| Provider | Can You Control Region? | Region Format |
|----------|-------------------------|---------------|
| **Azure OpenAI** | ✅ Yes | `eastus`, `westeurope` (from endpoint URL) |
| **Vertex AI** | ✅ Yes | `us-central1`, `europe-west4` (from `vertexai.init()`) |
| **AWS Bedrock** | ✅ Yes | `us-east-1`, `eu-west-1` (from `boto3.client()`) |
| **OpenAI** | ❌ No | Global routing - specify preference |
| **Anthropic** | ❌ No | Global routing - specify preference |

**If you don't specify region:** Vetch falls back to environment variables (`VETCH_REGION`, `AWS_REGION`, etc.) or timezone-based heuristic (coarse approximation that often causes carbon calculation errors).

## Per-Call Control (Granular Wrapper)

For per-call control or when you prefer explicit wrappers over global patching:

```python
from vetch import wrap

with wrap(region="us-east-1", tags={"team": "ml"}) as ctx:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}]
    )

# Access metrics directly (cost shown first - the "money shot")
print(f"Cost:   ${ctx.event['estimated_cost_usd']:.4f}")
print(f"Energy: {ctx.event['estimated_energy_wh']:.4f} Wh")
print(f"Carbon: {ctx.event['estimated_carbon_g']:.4f} gCO2e")
```

**Async support** (full example):

```python
from openai import AsyncOpenAI
from vetch import awrap

client = AsyncOpenAI()

async with awrap(region="us-east-1") as ctx:
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}]
    )
    print(f"Cost: ${ctx.event['estimated_cost_usd']}")

# Properly close async client
await client.close()
```

## Configure Output

By default, events are logged to `stderr`. Configure with environment variables:

```bash
export VETCH_OUTPUT=/tmp/vetch.jsonl  # Log to file
export VETCH_OUTPUT=none               # Silence output
```

## Cost Attribution with Tags

Tags appear in every event, enabling financial attribution (cost-per-feature, cost-per-team):

```bash
# Via environment (recommended for prod)
export VETCH_REGION=us-east-1

# Or in code
vetch.instrument(tags={"team": "ml", "cost_center": "research"})
```

This creates a **financial ledger** you can query, not just a technical log.

## Emergency Kill Switch

```bash
export VETCH_DISABLED=true  # Completely disable Vetch
```

## Next Steps

Now that you can **see** your spend, here's how to **control** it:

1. **[Budget alerts](README.md#budget-alerts)** - Set spending guardrails (CFO-friendly)
2. **[Session aggregation](README.md#session-aggregation-agentic-ai)** - Track agentic AI workflows (CrewAI, LangGraph)
3. **[OTLP export](README.md#otlp-export)** - Send to Datadog/Grafana/Honeycomb
4. **[CLI tools](README.md#cli-usage)** - Compare models, estimate costs before coding
5. **[Energy methodology](src/vetch/METHODOLOGY.md)** - Understand estimate tiers and uncertainty

---

**Questions or issues?** [github.com/prismatic-labs/vetch/issues](https://github.com/prismatic-labs/vetch/issues)
