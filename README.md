# Vetch

[![PyPI version](https://img.shields.io/pypi/v/vetch.svg)](https://pypi.org/project/vetch/)
[![Python versions](https://img.shields.io/pypi/pyversions/vetch.svg)](https://pypi.org/project/vetch/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/prismatic-labs/vetch/actions/workflows/ci.yml/badge.svg)](https://github.com/prismatic-labs/vetch/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prismatic-labs/vetch/blob/main/demo.ipynb)

**Stop runaway inference.**

Vetch detects stalled agents, RAG bloat, excessive generation, zombie LLM calls, context snowballs, invisible output burn, prompt cache opportunities, repeated truncation, and large-model rightsizing candidates. It turns those patterns into metadata-only advisory signals, and can warn, kill, or reroute confirmed stalled loops before they burn budget, latency, energy, and carbon. Retry storms remain tracked in the taxonomy as a planned detector.

- **[Live demo: kill a runaway agent](examples/circuit_breaker_demo.py)** — `vetch.set_stall_action("kill")` and watch
- **[Get started in 60 seconds (Cloud APIs)](QUICKSTART.md)**
- **[Vercel AI SDK (Next.js / Edge)](QUICKSTART-VERCEL.md)** — [`@prismatic-labs/vetch-ai-sdk`](packages/vetch-ai-sdk/) · [scope](docs/SCOPE-v0.8.0-vercel.md) · [npm publish](docs/NPM_PUBLISH.md)
- **[Track local models (Ollama, vLLM, llama.cpp)](QUICKSTART-LOCAL.md)**
- **[Interactive Inference Calculator](https://prismatic-labs.github.io/vetch/calculator/)** — Compare energy, cost, and carbon across the bundled registry (130+ models). After editing `src/vetch/registry/energy.json` or `pricing.json`, run `python scripts/build_calculator.py` to refresh the page embed.

```python
import vetch

vetch.instrument()
vetch.set_stall_action("kill")  # or "warn", or "reroute"

# Your agent loop here. Vetch detects stalls — short outputs with
# high input similarity, the signature of a stuck loop — and raises
# vetch.StallDetected before more money is wasted.
```

## Capabilities

- **Providers (auto-instrumented):** OpenAI, Anthropic, Google GenAI (`google-genai`), Google Vertex AI, Azure OpenAI, Ollama. `pip install vetch[openai]`, `vetch[anthropic]`, `vetch[genai]`, `vetch[vertexai]`, `vetch[ollama]`.
- **Self-hosted / OpenAI-compatible:** vLLM, TGI, LM Studio, llama.cpp — via an OpenAI client with a custom `base_url` (classified automatically), or raw HTTP through `vetch.proxy` / the `vetch.wrap()` record API. See [Self-hosted and raw HTTP](#self-hosted-and-raw-http).
- **Streaming:** input/output token instrumentation for sync and async streams, without buffering response content.
- **Framework integrations:** LangChain and LlamaIndex callback handlers; first-party [Vercel AI SDK middleware](packages/vetch-ai-sdk/) (JS/TS).
- **Control:** circuit breakers (stall detection → warn/kill/reroute) and warn-only budgets on cost/energy/carbon.
- **Export & tooling:** OpenTelemetry / OTLP export (GenAI semantic conventions), an MCP server for agents, a CLI (`vetch estimate|compare|audit|calibrate`), and local GPU calibration.
- **Metadata-only:** never reads prompt or completion text — only model, token counts, timing, and finish reason.

## The problem

Old cloud waste was idle infrastructure: overprovisioned servers, forgotten instances, jobs that ran once and stayed scheduled. You could fix it by turning things off.

AI waste is different. It is active, accumulating, and invisible until the bill arrives. A stalled agent loop burns tokens on every iteration. A RAG pipeline retrieving irrelevant context bloats every prompt. A session that should have ended 40 calls ago is still running. Provider dashboards show total spend. They do not show which feature, customer, workflow, or agent session produced the waste — and they cannot stop the next occurrence automatically.

Every wasted inference call is wasted money, compute, energy, and carbon.

## What counts as inference waste

| Pattern | What it looks like |
|---------|-------------------|
| **Stalled agent loop** | Agent iterating without meaningful output progress |
| **RAG bloat** | Retrieval context overwhelming the prompt with low-signal content |
| **Excessive generation** | Model producing unusually long outputs regardless of task complexity |
| **Zombie inference** | Sessions or background tasks making LLM calls after they should have stopped |
| **Context snowballing** | Conversation history or failed tool context growing on every turn |
| **Invisible output burn** | Output tokens consumed while little or no visible answer is returned |
| **Retry storm** | Repeated identical or near-identical calls after failures |
| **Large-model rightsizing candidate** | Stable workflow using a premium model where smaller candidates deserve eval |
| **Prompt cache misses** | Repeated prompt structures that could be cached but aren't |
| **Unattributed spend** | Inference cost that cannot be tied to a feature, customer, or workflow |

## What Vetch does

### Detect waste

Vetch analyzes every inference call for behavioral patterns that indicate waste:

| Advisory | Pattern | Signal | Status |
|----------|---------|--------|--------|
| `STALL-001` | Stalled agent loop | ≥80% of last 20 calls produce short output with repeated input | ✅ Implemented |
| `CACHE-001` | Prompt caching opportunity | >50% of calls share identical input token counts across ≥6 calls | ✅ Implemented |
| `CACHE-002` | Cache not active | Same repetition signal as CACHE-001 but no cache reads observed | ✅ Implemented |
| `RAG-001` | RAG bloat | Average input:output ratio exceeds 50:1 | ✅ Implemented |
| `BABBLE-001` | Excessive generation | Recent average output exceeds 1,500 tokens without long-form task signal | ✅ Implemented |
| `ZOMBIE-001` | Post-completion drift | Repeated normal-length outputs after likely task completion | ✅ Implemented |
| `CTX-001` | Context snowball | The prompt gets larger every turn while useful output stays low | ✅ Implemented |
| `EMPTY-001` | Invisible output burn | Output tokens consumed while visible output is near-empty | ✅ Implemented |
| `TRUNC-001` | Repeated response truncation | Frequent `finish_reason=max_tokens` or `length` across recent calls | ✅ Implemented |
| `STREAM-001` | Incomplete streams | ≥30% of streaming calls cancelled before completion | ✅ Implemented |
| `REASONING-001` | Reasoning model, no reasoning | o1/o3 calls return no reasoning tokens | ✅ Implemented |
| `ERROR-001` | Error storm | ≥3 consecutive errors or ≥40% error rate in recent window | ✅ Implemented |
| `SESSION-BUDGET-001` | Session over budget | Configured cost/energy/carbon threshold exceeded | ⚠️ Partial — alerts only, no advisory ID |
| `ATTRIBUTION-001` | Unattributed spend | Required tags missing from calls | ⚠️ Partial — infrastructure only |
| `RETRY-001` | Retry storm | Burst of repeated failed or near-identical calls | 🔜 Planned (`retry_count` field available; detector not yet wired) |
| `PREMIUM-001` | Large model rightsizing candidate | Stable tagged workflow mostly uses a premium model and has cheaper eval candidates | ✅ Implemented — audit-only |

Full taxonomy with detection signals, false positives, and recommended actions: [docs/inference-waste-taxonomy.md](docs/inference-waste-taxonomy.md)

Advisories are deterministic signals, not proof of waste. Confidence labels indicate signal strength from metadata patterns, not statistical certainty. Non-stall runtime advisories are warn-only; `PREMIUM-001` is audit-only and queues eval candidates rather than recommending an automatic downgrade. Automatic kill and reroute are scoped to `STALL-001`.

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

When `STALL-001` fires, Vetch can intervene without manual intervention:

| Action | What happens |
|--------|-------------|
| `"log"` (default) | Generate the advisory, take no action. Backwards compatible. |
| `"warn"` | Log a stderr warning on the next call after a stall. |
| `"kill"` | Raise `vetch.StallDetected` on the next call — the loop breaks. |
| `"reroute"` | Transparently substitute the model with `fallback_model`. |

`set_stall_action` is currently wired to `STALL-001`. Configurable policies per advisory, tag, and session are planned — see [ROADMAP.md](ROADMAP.md).

`vetch.StallDetected` inherits from `RuntimeError` so a generic `except ValueError:` handler will not swallow it. Recover with `session.clear_stall()` after a human-in-the-loop fix.

### Prove savings

Run `vetch audit` to generate a stored-event audit over the last 7 days (configurable with `--window`). The report shows which advisories fired, per-tag attribution breakdowns, observed avoidable cost, and a projected monthly avoidable cost estimate:

```bash
vetch audit                    # last 7 days
vetch audit --window 24h       # shorter window
vetch audit --tags team=ml     # filter by tag
vetch audit --format json      # machine-readable
vetch audit --format markdown  # for sharing
```

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
vetch audit             # reads stored metadata from the last 7 days
vetch audit --window 7d --tags feature=rag-search  # filter to one feature
vetch audit --format json  # machine-readable output
```

Output includes advisory findings (STALL-001, CACHE-001, RAG-001, BABBLE-001, ZOMBIE-001, CTX-001, EMPTY-001, TRUNC-001, PREMIUM-001) with signal-strength labels and recommended actions; per-tag attribution breakdowns; observed avoidable cost; projected monthly avoidable cost; and data quality indicators (tagged fraction, methodology versions used).

**Next — Promote confirmed stalls to kill or reroute**

For confirmed `STALL-001` patterns, promote the action:

```python
vetch.set_stall_action("kill")  # or "reroute", fallback_model="gpt-4o-mini"
```

Runaway inference is now stopped automatically.

For non-stall advisories, treat the audit as a review queue. Fix the workflow, retriever, cache configuration, response limits, or attribution gaps before adding automation.

Tune thresholds per workflow when a pattern is expected. For example, a
classification route that normally returns three tokens can lower the STALL-001
low-output threshold without changing other routes:

```python
with vetch.Session(
    tags={"route": "classifier"},
    advisory_thresholds={"STALL-001": {"low_output_threshold": 1}},
):
    response = client.chat.completions.create(...)
```

## Energy and carbon

Every wasteful call you prevent is money saved, tokens not burned, compute not consumed, and estimated emissions avoided. Vetch treats energy and carbon as first-class outputs alongside cost. The same stalled agent loop, bloated RAG context, or retry storm that burns budget also consumes unnecessary compute.

Energy, carbon, and water figures should be interpreted with explicit uncertainty, and not all numbers are equal. There are three kinds:

- **Measured (Tier 0, ±10–20%)** — direct GPU power telemetry from a local calibration run (`vetch calibrate`). This is the only genuinely metered tier, and it only exists for self-hosted models you measure yourself.
- **Inferred (Tier 1, ±20–50%)** — hosted-API models. These come from infrastructure-aware benchmarking, not power telemetry: hosted APIs (OpenAI, Anthropic, Google) expose no power meter, so figures are estimated from public API behaviour, provider environmental multipliers, and statistical hardware inference. Useful for comparison; not a measurement.
- **Estimated fallback (Tier 3, order-of-magnitude)** — models with no benchmark, or a proxy/family match. Directional only.

Every event carries `energy_tier`, `energy_uncertainty_pct`, and `model_match` (`exact`/`alias`/`prefix`/`family`/`fallback`) so you can tell measured from inferred from guessed. A `prefix` or `family` match is a proxy and is automatically downgraded to Tier 3, so a current-generation model that the bundled registry hasn't caught up to is flagged low-confidence rather than dressed up as exact. These numbers are useful for comparison, prioritization, and reduction decisions. They are not exact carbon certification, regulatory disclosure, or water accounting. Water estimates are especially facility-dependent.

**Inferred (Tier 1) coverage includes:**
- **GPT-4o, GPT-4o-mini, GPT-4.1 family, GPT-4.5, o1, o3, o4-mini**
- **Claude 3.7 Sonnet** (standard + Extended Thinking)
- **DeepSeek-R1, DeepSeek-V3** — reasoning and MoE benchmarks
- **Llama 3.1 (8B, 70B, 405B), Llama 3.3 70B** — open-weight measurements

The bundled registry holds 130+ models; run `vetch methodology` for the current per-model tier and provenance. The Tier 1 figures derive from [Jegham et al. (2025)](https://arxiv.org/abs/2505.09598) (infrastructure-aware benchmarking across ~30 commercial models). Open-weight / self-hosted figures can be cross-validated against direct-measurement work: [ML.ENERGY](https://ml.energy) (open data + Zeus toolkit), [Samsi et al. (2023)](https://arxiv.org/abs/2310.03003), and TokenPowerBench. Closed hosted-API models have no public power telemetry and stay inferred by necessity.

Use these estimates as internal inputs for FinOps, engineering, and sustainability planning. For regulatory reporting or external claims, use independent verification and the methodology notes from `vetch methodology`.

**Energy tiers:**

| Tier | Name | Uncertainty | Source |
|------|------|-------------|--------|
| 0 | **Measured** | ±10–20% | Direct GPU measurement (pynvml) |
| 1 | **Empirical/provider benchmark** | ±20–50% | Commercial API or provider benchmark data |
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
# Cost, energy, carbon, water, and advisory events emitted automatically
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

Sessions are the unit of attribution — every call within a session accumulates cost, energy, carbon, water, and advisory events that can be queried or exported together.

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

# Stored-event audit — last 7 days by default
vetch audit
vetch audit --window 24h --tags team=ml --format json

# Generate usage reports
vetch report --days 7 --tags team=ml

# Export Grafana dashboard
vetch dashboard --export grafana --output dashboard.json

# Freeze registry for CI/CD (eliminates cold-start latency)
vetch registry freeze --output vetch_registry.json
```

## Inference waste audit

After instrumenting and letting Vetch observe real traffic, run the CLI audit:

```bash
vetch audit                         # stored events, last 7 days
vetch audit --window 30d            # longer window
vetch audit --tags customer=acme    # filter by tag
vetch audit --format markdown       # shareable report
```

The audit reads locally stored metadata, runs advisory detection, computes per-tag attribution, and estimates observed and projected avoidable cost.

**What it detects:**
- **STALL-001** — short outputs with high input similarity across multiple calls (stalled agent loop)
- **CACHE-001** — repeated identical input token counts (uncached prompt structure)
- **RAG-001** — high input:output ratio (retrieval context overwhelming the prompt)
- **BABBLE-001** — unusually high average output tokens (excessive generation)
- **ZOMBIE-001** — repeated normal-length outputs after likely completion
- **CTX-001** — context/input tokens snowballing across a no-progress session
- **EMPTY-001** — output tokens consumed while visible output is near-empty
- **TRUNC-001** — repeated `finish_reason=max_tokens` or `length`, often causing cut-off JSON, tool calls, or answers
- **ERROR-001** — error storm: ≥3 consecutive errors or ≥40% error rate in the recent window
- **CACHE-002** — repetition pattern with no cache reads observed (caching available but not active)

**Lower-level Python API** (for programmatic access or custom reporting):

```python
from vetch.audit_report import build_audit_report, format_audit_report
from datetime import datetime, timedelta, timezone

now = datetime.now(timezone.utc)
report = build_audit_report(start=now - timedelta(days=7), end=now)
print(format_audit_report(report, "markdown"))
```

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

### Apple Silicon (M-series)

On Apple Silicon, use `vetch calibrate-apple-silicon` for powermetrics-based hardware measurement (requires `sudo`). Results are stored in `~/.vetch/calibrations/` and picked up automatically at inference time. See [QUICKSTART-LOCAL.md](QUICKSTART-LOCAL.md) for details.

### Community calibrations

`data/calibrations.json` ships with community-contributed coefficients and is populated from accepted GitHub submissions. To share your calibration results, open a PR adding your `_apple_detail.json` output from `~/.vetch/calibrations/` to the `community/` directory.

## Self-hosted and raw HTTP

Self-hosted serving (vLLM, TGI, LM Studio, llama.cpp) is usually reached through the OpenAI SDK with a custom `base_url`, or via raw HTTP. Vetch handles both, and crucially **never bills a self-hosted or third-party endpoint at OpenAI's per-token rates.**

**OpenAI SDK with a custom `base_url` (auto-classified):**

```python
import vetch
from openai import OpenAI

vetch.instrument()
client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")  # local vLLM
# Vetch classifies the endpoint from base_url:
#   api.openai.com / *.openai.azure.com  -> "openai"        (OpenAI energy + list price)
#   localhost / 127.0.0.1 / private IPs   -> "self-hosted"   (calibration energy, cost = 0)
#   other public hosts (OpenRouter, ...)  -> "openai-compatible" (registry energy, cost = unknown)
```

A local model resolves its energy from a calibration (`vetch calibrate`) or the model registry, and reports cost `0` (you pay for the hardware, captured as energy). A public OpenAI-compatible host reports cost `unknown` rather than OpenAI's price. If a private host can't be detected, set `VETCH_SELF_HOSTED=true`.

**Raw HTTP (no SDK to patch):**

Raw HTTP requests bypass every SDK patch, so instrument them explicitly with the context-manager / record API:

```python
import vetch, httpx

with vetch.wrap(region="us-east-1") as ctx:
    resp = httpx.post("http://localhost:8000/v1/chat/completions", json={...}).json()
    usage = resp["usage"]
    ctx.capture(
        model="llama-3-70b",
        provider="self-hosted",
        usage={"text": {
            "input_tokens": usage["prompt_tokens"],
            "output_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
        }},
    )
# ctx.event has the energy/carbon/cost estimate
```

`vetch.proxy` is also available for wrapping a callable transparently. See [QUICKSTART-LOCAL.md](QUICKSTART-LOCAL.md).

## Model coverage and resolution

Vetch resolves a model name to a registry entry through a ladder, and each event records which rung matched in `model_match`:

1. **`exact`** — the model ID is a registry key. Full confidence (entry's tier).
2. **`alias`** — a curated equivalence (dated snapshots, `-latest`, `-preview`) → canonical key. Full confidence.
3. **`prefix`** — an algorithmic shorten (`claude-sonnet-4-6-experimental` → `claude-sonnet-4-6`). Low confidence: **downgraded to Tier 3**.
4. **`family`** — no name match, but the provider family is known; proxied to a conservative same-family representative (biased to the larger sibling so it never undercounts). Low confidence: **Tier 3**.
5. **`fallback`** — fully unknown; generic conservative estimate.

Matching is case-insensitive. If a current-generation model isn't in the bundled registry yet, you have three options:

- **Add a row** — add `energy.json` (+ `pricing.json` with a verified price, + `aliases.json` for dated forms). See [src/vetch/registry/PROVENANCE.md](src/vetch/registry/PROVENANCE.md). Existing installs pick it up via the remote registry without upgrading.
- **Calibrate** — `vetch calibrate --model <name> --provider <provider>` for a hardware-measured Tier 0 figure on self-hosted models.
- **Override** — pass `energy_override` to `vetch.wrap()` for a one-off.

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
| `VETCH_SELF_HOSTED` | Set to `true` to force the self-hosted provider label (no per-token list price) when the endpoint host can't be auto-classified |
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
| Ollama | Supported | Native SDK (`vetch.providers.ollama`) or OpenAI-compat API (auto-detected) |
| vLLM / TGI | Compatible | Uses OpenAI instrumentation (OpenAI-compatible API) |

**OpenAI-compatible endpoints** (OpenRouter, Together.ai, Ollama, vLLM, TGI) work automatically with `vetch.instrument()` since they use the `openai` Python SDK under the hood. Vetch classifies them by `base_url` so they are **not** billed at OpenAI's per-token rates: private/local hosts are treated as self-hosted (cost `0`), other compatible hosts report cost `unknown`. See [Self-hosted and raw HTTP](#self-hosted-and-raw-http).

**For local models (Ollama, vLLM, llama.cpp)**: See [QUICKSTART-LOCAL.md](QUICKSTART-LOCAL.md) for setup, GPU calibration, and TCO analysis.

## Design guarantees

### Fail-open architecture

Every Vetch operation (patching, calculation, emission) is wrapped in isolated error handlers. If Vetch fails, your LLM call proceeds normally and a `tracking_disabled: true` event is logged. Vetch will never cause an inference outage.

### Privacy and data perimeter

Vetch does not store prompt or completion text. It extracts metadata directly from SDK response objects: token counts, model names, timing, tags, finish reason, and visible output character count. For output diagnostics, Vetch may count visible completion characters and immediately discard the text. No PII or proprietary prompt data ever leaves your execution environment.

### Thread safety

Vetch uses `contextvars` for async session isolation, locks session statistics
updates, and uses `WeakKeyDictionary` for client patching so unpatching one
client does not affect another in the same process. In web or worker systems,
create a `Session` per request, job, or agent invocation. Set global process
configuration, such as `set_stall_action()` or `set_advisory_thresholds()`, at
startup rather than mutating it concurrently per request.

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

## v0.8.0 release train (Python + Vercel AI SDK)

v0.8.0 ships two install surfaces:

1. **Python (PyPI):** [`QUICKSTART.md`](QUICKSTART.md) — `pip install vetch`, `vetch.instrument()`, audits, stall circuit breaker.
2. **Vercel AI SDK:** [`QUICKSTART-VERCEL.md`](QUICKSTART-VERCEL.md) — `@prismatic-labs/vetch-ai-sdk` for AI SDK 6.x ([publish guide](docs/NPM_PUBLISH.md), [Next.js example](packages/vetch-ai-sdk/examples/nextjs-app-router/README.md)).

Registry parity check (when changing energy/pricing data): `python scripts/sync_ai_sdk_registries.py`

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing guidelines, and how to contribute energy data.

## License

Apache License 2.0. See `LICENSE` and `NOTICE` for details.
