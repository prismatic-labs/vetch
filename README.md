# Vetch SDK

[![PyPI version](https://img.shields.io/pypi/v/vetch.svg)](https://pypi.org/project/vetch/)
[![Python versions](https://img.shields.io/pypi/pyversions/vetch.svg)](https://pypi.org/project/vetch/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/prismatic-labs/vetch/actions/workflows/ci.yml/badge.svg)](https://github.com/prismatic-labs/vetch/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prismatic-labs/vetch/blob/main/demo.ipynb)

Planet-aware observability for LLM inference.

Vetch is a Python SDK that wraps LLM API calls to log energy consumption, cost, and carbon per inference using live grid data. It never reads prompt or completion content—only metadata from the response usage.

- **[Get started in 60 seconds (Cloud APIs)](QUICKSTART.md)**
- **[Track local models (Ollama, vLLM, llama.cpp)](QUICKSTART-LOCAL.md)**
- **[Interactive Inference Calculator](https://prismatic-labs.github.io/vetch/calculator/)** — Compare energy, cost, and carbon across 48 models

## Why Vetch?

**Attributed Spend, Not Just Total Spend**

Provider dashboards (OpenAI Usage, Anthropic Console, Google Cloud Billing) show you *total* spend. Vetch shows you *attributed* spend. Using tags, you can track cost-per-feature, cost-per-user, or cost-per-environment in real-time—without building custom infrastructure.

**Sustainability Instrumentation**

Begin tracking AI inference emissions for future CSRD (EU) and SEC (US) Scope 3 reporting. Vetch includes **Tier 1 (±50%)** hardware-measured energy data for popular models:
- **GPT-4o, GPT-4o-mini, GPT-4.1 family, GPT-4.5, o1, o3, o4-mini** - Measured in Azure datacenters
- **Claude-3.7 Sonnet** (standard + Extended Thinking) - Measured in AWS datacenters
- **DeepSeek-R1, DeepSeek-V3** - Reasoning and MoE benchmarks
- **Llama 3.1 (8B, 70B, 405B), Llama 3.3 70B** - Open-weight measurements
- **GPT-5 family** (gpt-5, gpt-5-mini, gpt-5-nano, gpt-5.4 etc.) - Tier 3 estimates
- **48 models** in the registry, with Tier 3 (order-of-magnitude) estimates for unmeasured models

Source: [Jegham et al. (2025)](https://arxiv.org/abs/2505.09598) - First large-scale LLM energy measurements in commercial datacenters.

## Design Guarantees

### Fail-Open Architecture

Vetch is architected with a non-blocking, fail-open boundary. Every Vetch operation (patching, calculation, emission) is wrapped in isolated error handlers. If Vetch fails, your LLM call proceeds normally, and a `tracking_disabled: true` event is logged. Vetch will never cause an inference outage.

### Privacy & Data Perimeter

Vetch never reads or stores prompt/completion content. It only extracts metadata (token counts, model names, timing) directly from SDK response objects. No PII or proprietary prompt data ever leaves your execution environment.

### Thread Safety (v0.1.4+)

Vetch is fully thread-safe and supports multi-client isolation. It uses `contextvars` for async safety and `WeakKeyDictionary` for client patching, ensuring that unpatching one client doesn't affect another in the same process.

## Features

- **Fail-Open**: LLM calls always proceed even if Vetch fails
- **Privacy-First**: No prompt or completion data is ever read or buffered
- **Multi-tier Caching**: Memory -> File -> API -> Regional averages for grid data
- **Observability-Transparent**: Works seamlessly with Datadog, OpenTelemetry, and Sentry
- **Low Overhead**: Under 5ms overhead for sync calls; zero TTFT latency for streaming
- **MoE-Aware**: Energy estimates account for active parameters in Mixture-of-Experts models
- **Session Aggregation**: Group multiple LLM calls into sessions for agentic AI tracking
- **Cache-Aware Pricing**: Accurate cost calculation with prompt cache discounts

## Supported Providers

| Provider | Status | Instrumentation |
|----------|--------|----------------|
| OpenAI | Supported | `vetch.instrument()` or `vetch.wrap()` |
| Azure OpenAI | Supported | `vetch.instrument()` (auto-detects `AzureOpenAI`) |
| Anthropic | Supported | `vetch.instrument()` or `vetch.wrap()` |
| Vertex AI (Gemini) | Supported | `vetch.instrument()` or `vetch.wrap()` |
| OpenRouter | Compatible | Uses OpenAI instrumentation (OpenAI-compatible API) |
| Together.ai | Compatible | Uses OpenAI instrumentation (OpenAI-compatible API) |
| Anyscale | Compatible | Uses OpenAI instrumentation (OpenAI-compatible API) |
| Ollama | Compatible | Uses OpenAI instrumentation (OpenAI-compatible API) |
| vLLM / TGI | Compatible | Uses OpenAI instrumentation (OpenAI-compatible API) |

**OpenAI-compatible endpoints** (OpenRouter, Together.ai, Ollama, vLLM, TGI) work automatically with `vetch.instrument()` since they use the `openai` Python SDK under the hood.

**For local models (Ollama, vLLM, llama.cpp)**: See [QUICKSTART-LOCAL.md](QUICKSTART-LOCAL.md) for setup, GPU calibration, and TCO analysis.

## Installation

```bash
pip install vetch
```

## Quick Start

Vetch offers two instrumentation modes — choose the one that fits your use case:

### `instrument()` — Global, Zero-Touch

One line at startup. Every LLM call across all providers is tracked automatically. Best for services, APIs, and production deployments where you want blanket coverage:

```python
import vetch
import openai

vetch.instrument(region="us-east-1", tags={"service": "chat-api"})

# All LLM calls are now automatically tracked — nothing else to change
client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello world"}]
)
# Energy, cost, and carbon events emitted automatically
```

### `wrap()` — Per-Call, Explicit

Context manager around individual calls. Best when you need per-call metrics, different tags per call, or want to avoid global patching:

```python
from vetch import wrap

with wrap(region="us-east-1", tags={"team": "ml", "env": "prod"}) as ctx:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello world"}]
    )

# Access inference metadata directly
print(f"Cost:   ${ctx.event['estimated_cost_usd']}")
print(f"Energy: {ctx.event['estimated_energy_wh']} Wh")
print(f"Carbon: {ctx.event['estimated_carbon_g']} gCO2e")
```

**When to use which:**

| | `instrument()` | `wrap()` |
|--|----------------|----------|
| Setup | One line at startup | Context manager per call |
| Scope | All calls, all providers | Individual calls |
| Tags | Same tags for everything | Different tags per call |
| Metrics access | Via event callbacks | Via `ctx.event` dict |
| Best for | Production services | Notebooks, experiments, per-feature attribution |

Both are fail-open (never break your LLM calls) and add <5ms overhead.

**See [QUICKSTART.md](QUICKSTART.md) for a complete 60-second guide.**

### Async Support

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

await client.close()
```

## Understanding Region Configuration

The `region` parameter determines which electricity grid is used for carbon intensity calculations. It should match the **Electricity Maps zone identifier** (which typically aligns with cloud provider region names: `us-east-1`, `eu-west-1`, `eastus`, etc.).

Region availability varies by provider:

### Providers with Regional Control

For these providers, **you control where inference happens** and can specify the exact region:

| Provider | How to Control Region | Example Region Format |
|----------|----------------------|----------------------|
| **Azure OpenAI** | Region embedded in endpoint URL | `eastus`, `westeurope` (no hyphens) |
| **Vertex AI (Google)** | Set via `vertexai.init()` | `us-central1`, `europe-west4` (hyphenated) |
| **AWS Bedrock** | Standard AWS region parameter | `us-east-1`, `eu-west-1` (hyphenated) |

**For these providers:** Specify the region you're actually using for accurate carbon calculations:

```python
# Azure OpenAI - use the region from your endpoint
# Vetch attempts auto-detection from endpoint URL, but explicit config is more reliable
vetch.instrument(region="eastus")  # Matches eastus.openai.azure.com

# Vertex AI - match your vertexai.init() location
vetch.instrument(region="us-central1")

# AWS Bedrock - match your boto3 region
vetch.instrument(region="us-east-1")
```

### Providers without Regional Control

For these providers, **inference location is not exposed** — requests are routed across global infrastructure (Azure, AWS, GCP) and the physical location of a specific inference call is not available to the client:

- **OpenAI** (standard API): Global routing across cloud providers
- **Anthropic**: Global routing across cloud providers

**For these providers:** Use your best estimate based on your location or expected data center:

```python
# OpenAI/Anthropic - specify your expected or preferred region
vetch.instrument(region="us-east-1")  # Reasonable default for US users
vetch.instrument(region="eu-west-1")  # Reasonable default for EU users
```

### Region Fallback Behavior

If you don't specify `region`, Vetch uses this fallback hierarchy:

1. **`VETCH_REGION` environment variable** (highest priority)
2. **Cloud provider env vars** (`AWS_REGION`, `GOOGLE_CLOUD_REGION`, `AZURE_REGION`)
3. **Timezone-based heuristic** (coarse approximation, often results in significant carbon calculation errors)

**Best practice:** Always set `region` explicitly or via `VETCH_REGION` environment variable for accurate carbon calculations.

```bash
# Set globally via environment
export VETCH_REGION=us-east-1
```

## Session Aggregation (Agentic AI)

Group multiple LLM calls into sessions for agentic frameworks like CrewAI, AutoGPT, or LangGraph:

```python
import vetch

with vetch.Session(tags={"agent": "researcher", "task": "summarize"}) as session:
    with vetch.wrap() as ctx1:
        response1 = client.chat.completions.create(...)

    # Nested sessions for sub-agents
    with vetch.Session(tags={"agent": "summarizer"}) as sub_session:
        with vetch.wrap() as ctx2:
            response2 = client.chat.completions.create(...)

# Aggregate metrics across all calls
print(f"Total energy: {session.total_energy_wh} Wh")
print(f"Total cost: ${session.total_cost_usd}")
print(f"Call count: {session.call_count}")
```

Sessions support distributed propagation across microservices:

```python
# In FastAPI service:
headers = session.inject_headers({})
celery_task.delay(task_id, headers=headers)

# In Celery worker:
with vetch.Session.from_headers(task_headers) as worker_session:
    with vetch.wrap() as ctx:
        response = client.chat.completions.create(...)
```

## Budget Alerts

Set spending thresholds with automatic alerting:

```python
import vetch

vetch.set_budget("hourly", cost_usd=10.0, energy_wh=50.0)

@vetch.on_budget_alert
def handle_alert(alert):
    print(f"Budget alert: {alert}")

# Check budget status
status = vetch.get_budget_status()
```

## OTLP Export (Grafana, Datadog)

Export metrics to any OpenTelemetry-compatible backend:

```python
import vetch

vetch.configure_otlp_export(
    endpoint="http://localhost:4317",
    service_name="my-llm-service"
)

# Export a pre-built Grafana dashboard
# vetch dashboard --export grafana --output grafana_vetch.json
```

## MCP Server (AI Agent Integration)

Vetch ships an [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server that gives AI agents real-time access to energy, cost, and carbon data. Agents can check budgets, compare models, and make sustainability-aware decisions mid-conversation.

### Setup

```bash
pip install vetch[mcp]
```

Add to your MCP client configuration (e.g., Claude Desktop `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "vetch": {
      "command": "vetch-mcp",
      "env": {
        "VETCH_REGION": "us-east-1"
      }
    }
  }
}
```

### Available Tools

| Tool | Description |
|------|-------------|
| `vetch_estimate` | Estimate energy, carbon, water, and cost for a model + token count |
| `vetch_compare` | Compare multiple models side-by-side (flags cheapest/greenest) |
| `vetch_session_stats` | Aggregated session metrics + waste advisories |
| `vetch_status` | Health check, version, and budget status |
| `vetch_check_budget` | Remaining budget (threshold, accumulated, percentage used) |
| `vetch_grid_intensity` | Live carbon intensity for a grid region |
| `vetch_cleanest_region` | Find the lowest-carbon region from a list |
| `vetch_registry_lookup` | Raw energy/pricing data for a model |

### Available Resources

| URI | Description |
|-----|-------------|
| `vetch://registry/models` | All model names in the registry |
| `vetch://config` | Current Vetch configuration |
| `vetch://version` | Vetch version string |

The MCP server uses stdio transport and dispatches synchronous Vetch calls via `asyncio.to_thread` to avoid blocking the event loop.

## CLI Usage

```bash
# Check Vetch status and configuration
vetch status

# Estimate energy/carbon for a model without running code
vetch estimate --model gpt-4o --input-tokens 1000 --output-tokens 500

# Compare multiple models
vetch compare --models gpt-4o,claude-3-opus,gemini-1.5-pro --tokens 1000

# Analyze token usage patterns
vetch audit

# Export Grafana dashboard
vetch dashboard --export grafana --output dashboard.json

# Freeze registry for CI/CD (eliminates cold-start latency)
vetch registry freeze --output vetch_registry.json

# Generate usage reports
vetch report --days 7 --tags team=ml
```

## Token Waste Audit

Vetch tracks token usage patterns across your session and provides actionable recommendations:

```python
from vetch import wrap, get_session_stats, generate_advisories

# Make multiple LLM calls
for _ in range(10):
    with wrap() as ctx:
        response = client.chat.completions.create(...)

# Analyze patterns
stats = get_session_stats()
advisories = generate_advisories(stats)

for a in advisories:
    print(f"[{a.level.value}] {a.title}")
    print(f"  {a.description}")
```

**What it detects:**
- **Static system prompts**: Repeated input token counts suggest cacheable prompts
- **High input:output ratios**: Large inputs producing small outputs
- **Expensive model usage**: Opportunities to use smaller, cheaper models

## GPU Calibration (Local Inference)

For local inference (Ollama, vLLM, llama.cpp), calibrate energy measurements using actual GPU power draw:

```python
from vetch.calibrate import calibrate_model, format_calibration_result

def my_inference():
    response = ollama.generate(model="llama3.1:8b", prompt="Hello world")
    return 100, 50  # (input_tokens, output_tokens)

result = calibrate_model("ollama", "llama3.1:8b", workload=my_inference)
print(format_calibration_result(result))
```

**Requirements:** NVIDIA GPU with `pynvml` (`pip install nvidia-ml-py3`)

## Clean Test Isolation

Remove instrumentation for clean test environments:

```python
import vetch

vetch.instrument()
# ... run your code ...
vetch.uninstrument()  # Restore original SDK methods
```

## Energy Tiers

Vetch uses a tiered system for energy estimate confidence:

| Tier | Name | Uncertainty | Source |
|------|------|-------------|--------|
| 0 | **Measured** | +-10-20% | Direct GPU measurement (pynvml) |
| 1 | **Vendor-Published** | +-20-50% | Official provider data |
| 2 | **Validated** | +-50-100% | Crowdsourced aggregates |
| 3 | **Estimated** | order of magnitude | Parameter-based calculation |

Run `vetch methodology` to see full methodology documentation.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `VETCH_DISABLED` | Set to `true` to completely disable Vetch (emergency kill switch) |
| `VETCH_REGION` | Default grid region (e.g., `us-east-1`, `eu-west-1`) |
| `VETCH_OUTPUT` | Output target: `none` (default), `stderr`, or file path |
| `VETCH_HOME` | Vetch home directory (default: `~/.vetch/`) |
| `VETCH_REGISTRY_REMOTE` | Set to `false` to disable remote registry updates |
| `VETCH_REGISTRY_PATH` | Path to offline registry directory (air-gapped environments) |
| `VETCH_REGISTRY_URL` | Custom remote registry URL |
| `ELECTRICITY_MAPS_API_KEY` | API key for live grid carbon intensity data |
| `VETCH_CACHE_MODE` | Set to `memory-only` for serverless/Lambda environments |

## Alpha Limitations

This is an alpha release. Please be aware of:

1. **Energy estimates are uncertain**: Most models use Tier 3 estimates (order of magnitude uncertainty). See `vetch methodology` for details.

2. **Region inference is a coarse heuristic**: Without explicit `VETCH_REGION`, timezone-based fallback often results in significant carbon calculation errors. Always set `region` parameter or `VETCH_REGION` environment variable for accurate carbon calculations. See [Understanding Region Configuration](#understanding-region-configuration) for details.

3. **Experimental modules**: `vetch.calibrate`, `vetch.storage`, and `vetch.ci` emit `FutureWarning` and may change in future versions.

## Troubleshooting

**Vetch is blocking my LLM calls:**
```bash
export VETCH_DISABLED=true  # Emergency kill switch
```

**Too much output:**
```bash
export VETCH_OUTPUT=none  # Silence all output
```

**Need to debug:**
```python
import logging
logging.getLogger("vetch").setLevel(logging.DEBUG)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing guidelines, and how to contribute energy data.

## License

Apache License 2.0. See `LICENSE` and `NOTICE` for details.
