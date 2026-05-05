# Vetch

[![PyPI version](https://img.shields.io/pypi/v/vetch.svg)](https://pypi.org/project/vetch/)
[![Python versions](https://img.shields.io/pypi/pyversions/vetch.svg)](https://pypi.org/project/vetch/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/prismatic-labs/vetch/actions/workflows/ci.yml/badge.svg)](https://github.com/prismatic-labs/vetch/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prismatic-labs/vetch/blob/main/demo.ipynb)

**Stop runaway inference.**

Vetch detects stalled agents, RAG bloat, zombie LLM calls, retry storms, and premium-model overuse — then warns, kills, reroutes, or throttles wasteful inference before it burns budget, latency, energy, and carbon.

- **[Live demo: kill a runaway agent](examples/circuit_breaker_demo.py)** — `vetch.set_stall_action("kill")` and watch
- **[Get started in 60 seconds (Cloud APIs)](QUICKSTART.md)**
- **[Track local models (Ollama, vLLM, llama.cpp)](QUICKSTART-LOCAL.md)**
- **[Interactive Inference Calculator](https://prismatic-labs.github.io/vetch/calculator/)** — Compare energy, cost, and carbon across 48 models

```python
import vetch

vetch.instrument()
vetch.set_stall_action("kill")  # or "warn", or "reroute"

# Your agent loop here. Vetch detects stalls — short outputs with
# high input similarity, the signature of a stuck loop — and raises
# vetch.StallDetected before more money is wasted.
```

## The problem

Old cloud waste was idle infrastructure: overprovisioned servers, forgotten instances, jobs that ran once and stayed scheduled. You could fix it by turning things off.

AI waste is different. It is active, accumulating, and invisible until the bill arrives. A stalled agent loop burns tokens on every iteration. A RAG pipeline retrieving irrelevant context bloats every prompt. A session that should have ended 40 calls ago is still running. Provider dashboards show total spend. They do not show which feature, customer, workflow, or agent session produced the waste — and they cannot stop the next occurrence automatically.

Every wasted inference call is wasted money, compute, energy, and carbon.

## What counts as inference waste

| Pattern | What it looks like |
|---------|-------------------|
| **Stalled agent loop** | Agent iterating without meaningful output progress |
| **RAG bloat** | Retrieval context overwhelming the prompt with low-signal content |
| **Zombie inference** | Sessions or background tasks making LLM calls after they should have stopped |
| **Retry storm** | Repeated identical or near-identical calls after failures |
| **Premium model overuse** | High-capability model used where a cheaper one would suffice |
| **Prompt cache misses** | Repeated prompt structures that could be cached but aren't |
| **Unattributed spend** | Inference cost that cannot be tied to a feature, customer, or workflow |

## What Vetch does

### Detect waste

Vetch analyzes every inference call for behavioral patterns that indicate waste:

| Advisory | Pattern | Signal | Status |
|----------|---------|--------|--------|
| `STALL-001` | Stalled agent loop | ≥80% of last 20 calls produce short output with repeated input | ✅ Implemented |
| `CACHE-001` | Prompt caching opportunity | >50% of calls share identical input token counts across ≥6 calls | ✅ Implemented |
| `RAG-001` | RAG bloat | Average input:output ratio exceeds 50:1 | ✅ Implemented |
| `SESSION-BUDGET-001` | Session over budget | Configured cost/energy/carbon threshold exceeded | ⚠️ Partial — alerts only, no advisory ID |
| `ATTRIBUTION-001` | Unattributed spend | Required tags missing from calls | ⚠️ Partial — infrastructure only |
| `RETRY-001` | Retry storm | Burst of repeated failed or near-identical calls | 🔜 Planned |
| `PREMIUM-001` | Premium model overuse | Expensive model used for low-complexity tasks | 🔜 Planned |
| `ZOMBIE-001` | Zombie inference | Active calls past expected session completion | 🔜 Planned |

Full taxonomy with detection signals, false positives, and recommended actions: [docs/inference-waste-taxonomy.md](docs/inference-waste-taxonomy.md)

### Attribute waste

Every inference call is tagged and attributed to a session. Sessions can carry any tags you define — `feature`, `customer`, `user`, `workflow`, `environment`, `team`. Cost, energy, and carbon accumulate per session and per tag combination.

```python
with vetch.wrap(tags={"feature": "rag-search", "customer": "acme"}) as ctx:
    response = client.chat.completions.create(...)

print(f"Cost:   ${ctx.event['estimated_cost_usd']:.5f}")
print(f"Energy: {ctx.event['estimated_energy_wh']:.4f} Wh")
print(f"Carbon: {ctx.event['estimated_carbon_g']:.4f} gCO2e")
```

### Stop waste automatically

When a waste advisory fires, Vetch can intervene without manual intervention:

| Action | What happens |
|--------|-------------|
| `"log"` (default) | Generate the advisory, take no action. Backwards compatible. |
| `"warn"` | Log a stderr warning on the next call after a stall. |
| `"kill"` | Raise `vetch.StallDetected` on the next call — the loop breaks. |
| `"reroute"` | Transparently substitute the model with `fallback_model`. |

`set_stall_action` is currently wired to `STALL-001`. Configurable policies per advisory, tag, and session are planned — see [ROADMAP.md](ROADMAP.md).

`vetch.StallDetected` inherits from `RuntimeError` so a generic `except ValueError:` handler will not swallow it. Recover with `session.clear_stall()` after a human-in-the-loop fix.

### Prove savings

Run `vetch audit` to see which advisories fired and a summary of token usage patterns in the current session. Attribution reports by feature, customer, and workflow — with estimated avoided cost, energy, and carbon — are planned. See [docs/audit-report.md](docs/audit-report.md) for the target output format.

## Why not just use your provider dashboard?

- **No attribution.** Provider dashboards show cost by model and date. They do not show cost by agent session, customer, or feature flag. If one customer's workflow is burning 30% of your inference budget, the dashboard will not tell you which customer or why.
- **Read-only.** Provider dashboards cannot fire a circuit breaker when a session exceeds a budget threshold or when an agent loop stalls.
- **No pattern detection.** A dashboard cannot identify that 80% of your agent's outputs over the last ten calls were under 20 tokens — the signature of a stalled loop.
- **No per-call energy or carbon data.** If you need to report or act on inference resource use, you need per-call instrumentation the provider does not expose.

## 7-day Inference Waste Audit

A concrete adoption motion. By day 7 you will have a clear picture of where inference spend is going and which patterns are causing it.

**Day 1 — Instrument**

One import, one line:

```python
import vetch
vetch.instrument(region="us-east-1", tags={"service": "my-service"})
```

All LLM calls across all providers are now tracked. No other code changes required.

**Days 1–7 — Tag and observe**

Add tags to attribute spend by feature or workflow. Run in warn-only mode to observe advisories without intervention:

```python
vetch.set_stall_action("warn")

with vetch.wrap(tags={"feature": "document-qa", "customer": "acme"}) as ctx:
    response = client.chat.completions.create(...)
```

**Day 7 — Run the audit**

```bash
vetch audit
```

Current output: advisory list (STALL-001, CACHE-001, RAG-001 detections with session context) and session summary (total requests, total tokens, average input:output ratio).

Full attribution reports by feature/customer/workflow with estimated avoided cost, energy, and carbon are planned — see [docs/audit-report.md](docs/audit-report.md) for the target format.

**Next — Promote to kill or reroute**

For advisories where you are confident, promote the action:

```python
vetch.set_stall_action("kill")  # or "reroute", fallback_model="gpt-4o-mini"
```

Runaway inference is now stopped automatically.

## Energy and carbon

Every wasteful call you prevent is money saved, tokens not burned, compute not consumed, and estimated emissions avoided. Vetch treats energy and carbon as first-class outputs alongside cost. The same stalled agent loop, bloated RAG context, or retry storm that burns budget also consumes unnecessary compute.

Energy and carbon figures should be interpreted with explicit uncertainty. Tier 1 vendor-published estimates carry approximately ±20–50% uncertainty; Tier 3 estimates are order-of-magnitude directional figures. These numbers are useful for comparison, prioritization, and reduction decisions — not exact carbon certification.

**Supported models with Tier 1 (±20–50%) data:**
- **GPT-4o, GPT-4o-mini, GPT-4.1 family, GPT-4.5, o1, o3, o4-mini** — measured in Azure datacenters
- **Claude 3.7 Sonnet** (standard + Extended Thinking) — measured in AWS datacenters
- **DeepSeek-R1, DeepSeek-V3** — reasoning and MoE benchmarks
- **Llama 3.1 (8B, 70B, 405B), Llama 3.3 70B** — open-weight measurements
- **48 models** in total; unmeasured models use Tier 3 order-of-magnitude estimates

Source: [Jegham et al. (2025)](https://arxiv.org/abs/2505.09598) — first large-scale LLM energy measurements in commercial datacenters.

Begin tracking AI inference emissions for CSRD (EU) and SEC (US) Scope 3 reporting with `vetch methodology` for full methodology documentation.

**Energy tiers:**

| Tier | Name | Uncertainty | Source |
|------|------|-------------|--------|
| 0 | **Measured** | ±10–20% | Direct GPU measurement (pynvml) |
| 1 | **Vendor-Published** | ±20–50% | Official provider benchmark data |
| 2 | **Validated** | ±50–100% | Crowdsourced aggregates |
| 3 | **Estimated** | Order of magnitude | Parameter-based calculation |

## Quick start

Two lines to start tracking inference waste in your existing LLM calls.

### `instrument()` — Global, zero-touch

One line at startup. Every LLM call across all providers is tracked automatically:

```python
import vetch
import openai

vetch.instrument(region="us-east-1", tags={"service": "chat-api"})

client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello world"}]
)
# Cost, energy, carbon, and advisory events emitted automatically
```

### `wrap()` — Per-call, explicit

Context manager around individual calls. Best for per-call metrics, different tags per call, or avoiding global patching:

```python
from vetch import wrap

with wrap(region="us-east-1", tags={"team": "ml", "env": "prod"}) as ctx:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello world"}]
    )

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

Both are fail-open and add <5ms overhead.

**See [QUICKSTART.md](QUICKSTART.md) for a complete 60-second guide.**

### Async support

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

## Understanding region configuration

The `region` parameter determines which electricity grid is used for carbon intensity calculations. It should match the **Electricity Maps zone identifier** (which typically aligns with cloud provider region names: `us-east-1`, `eu-west-1`, `eastus`, etc.).

### Providers with regional control

| Provider | How to control region | Example region format |
|----------|----------------------|----------------------|
| **Azure OpenAI** | Region embedded in endpoint URL | `eastus`, `westeurope` |
| **Vertex AI (Google)** | Set via `vertexai.init()` | `us-central1`, `europe-west4` |
| **AWS Bedrock** | Standard AWS region parameter | `us-east-1`, `eu-west-1` |

### Providers without regional control

For **OpenAI** and **Anthropic**, inference is routed across global infrastructure and the physical location of a specific call is not exposed. Use your best estimate based on location or expected datacenter:

```python
vetch.instrument(region="us-east-1")  # Reasonable default for US users
vetch.instrument(region="eu-west-1")  # Reasonable default for EU users
```

### Region fallback

If `region` is not specified, Vetch uses this fallback hierarchy:

1. `VETCH_REGION` environment variable
2. Cloud provider env vars (`AWS_REGION`, `GOOGLE_CLOUD_REGION`, `AZURE_REGION`)
3. Timezone-based heuristic (coarse approximation, often results in significant carbon calculation errors)

**Best practice:** Always set `region` explicitly or via `VETCH_REGION` for accurate carbon calculations.

## Session aggregation and attribution

Sessions are the unit of attribution — every call within a session accumulates cost, energy, carbon, and advisory events that can be queried or exported together.

```python
import vetch

with vetch.Session(tags={"agent": "researcher", "task": "summarize"}) as session:
    with vetch.wrap() as ctx1:
        response1 = client.chat.completions.create(...)

    with vetch.Session(tags={"agent": "summarizer"}) as sub_session:
        with vetch.wrap() as ctx2:
            response2 = client.chat.completions.create(...)

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

## Budget alerts

Set spending thresholds with automatic alerting:

```python
import vetch

vetch.set_budget("hourly", cost_usd=10.0, energy_wh=50.0)

@vetch.on_budget_alert
def handle_alert(alert):
    print(f"Budget alert: {alert}")

status = vetch.get_budget_status()
```

Budget thresholds never block LLM calls — they trigger alerts only. Blocking policies are planned.

## OTLP export (Grafana, Datadog)

Export metrics to any OpenTelemetry-compatible backend. OTLP export is how Vetch evidence — waste advisories, per-call cost, energy, and carbon — reaches your existing observability stack:

```python
import vetch

vetch.configure_otlp_export(
    endpoint="http://localhost:4317",
    service_name="my-llm-service"
)

# Export a pre-built Grafana dashboard focused on inference waste
# vetch dashboard --export grafana --output grafana_vetch.json
```

## MCP server (AI agent integration)

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

### Available tools

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

### Available resources

| URI | Description |
|-----|-------------|
| `vetch://registry/models` | All model names in the registry |
| `vetch://config` | Current Vetch configuration |
| `vetch://version` | Vetch version string |

## CLI usage

```bash
# Check Vetch status and configuration
vetch status

# Estimate energy/carbon for a model without running code
vetch estimate --model gpt-4o --input-tokens 1000 --output-tokens 500

# Compare multiple models
vetch compare --models gpt-4o,claude-3-opus,gemini-1.5-pro --tokens 1000

# Analyze token usage patterns and generate waste advisories
vetch audit

# Generate usage reports
vetch report --days 7 --tags team=ml

# Export Grafana dashboard
vetch dashboard --export grafana --output dashboard.json

# Freeze registry for CI/CD (eliminates cold-start latency)
vetch registry freeze --output vetch_registry.json
```

## Token waste audit

Vetch tracks token usage patterns across your session and generates actionable advisories:

```python
from vetch import wrap, get_session_stats, generate_advisories

for _ in range(10):
    with wrap() as ctx:
        response = client.chat.completions.create(...)

stats = get_session_stats()
advisories = generate_advisories(stats)

for a in advisories:
    print(f"[{a.level.value}] {a.title}")
    print(f"  {a.description}")
```

**What it detects:**
- **Static system prompts** — repeated input token counts suggest cacheable prompts
- **High input:output ratios** — large inputs producing small outputs
- **Stalled loops** — short outputs with high input similarity across multiple calls

## GPU calibration (local inference)

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

## Clean test isolation

Remove instrumentation for clean test environments:

```python
import vetch

vetch.instrument()
# ... run your code ...
vetch.uninstrument()  # Restore original SDK methods
```

## Environment variables

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

## Supported providers

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

## Design guarantees

### Fail-open architecture

Every Vetch operation (patching, calculation, emission) is wrapped in isolated error handlers. If Vetch fails, your LLM call proceeds normally and a `tracking_disabled: true` event is logged. Vetch will never cause an inference outage.

### Privacy and data perimeter

Vetch never reads or stores prompt or completion content. It only extracts metadata (token counts, model names, timing) directly from SDK response objects. No PII or proprietary prompt data ever leaves your execution environment.

### Thread safety (v0.1.4+)

Vetch is fully thread-safe and supports multi-client isolation. It uses `contextvars` for async safety and `WeakKeyDictionary` for client patching, ensuring that unpatching one client does not affect another in the same process.

## Current limitations

1. **Energy estimates are uncertain.** Most models use Tier 3 estimates (order-of-magnitude uncertainty). See `vetch methodology` for details.

2. **Region inference is a coarse heuristic.** Without explicit `VETCH_REGION`, timezone-based fallback often results in significant carbon calculation errors. Always set `region` or `VETCH_REGION` for accurate carbon calculations.

3. **Automatic intervention is currently wired to STALL-001 only.** Configurable policies per advisory, tag, and session are planned.

4. **Experimental modules.** `vetch.calibrate`, `vetch.storage`, and `vetch.ci` emit `FutureWarning` and may change in future versions.

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
