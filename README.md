# Vetch

[![PyPI version](https://img.shields.io/pypi/v/vetch.svg)](https://pypi.org/project/vetch/)
[![Python versions](https://img.shields.io/pypi/pyversions/vetch.svg)](https://pypi.org/project/vetch/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/prismatic-labs/vetch/actions/workflows/ci.yml/badge.svg)](https://github.com/prismatic-labs/vetch/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prismatic-labs/vetch/blob/main/demo.ipynb)

**Stop runaway inference.**

Vetch detects stalled agents, RAG bloat, excessive generation, zombie LLM calls, context snowballs, invisible output burn, prompt cache opportunities, repeated truncation, and large-model rightsizing candidates. It turns those patterns into metadata-only advisory signals, and can warn, kill, or reroute confirmed stalled loops before they burn budget, latency, energy, and carbon.

```python
import vetch

vetch.instrument()
vetch.set_stall_action("kill")  # or "warn", or "reroute"

# Your agent loop here. Vetch detects stalls (short outputs with high
# input similarity, the signature of a stuck loop) and raises
# vetch.StallDetected before more money is wasted.
```

- **[Live demo: kill a runaway agent](examples/circuit_breaker_demo_web.py)** — browser dashboard (mock mode needs no API key); CLI twin: [`circuit_breaker_demo.py`](examples/circuit_breaker_demo.py)
- **[Get started in 60 seconds (Cloud APIs)](QUICKSTART.md)**
- **[Vercel AI SDK (Next.js / Edge)](QUICKSTART-VERCEL.md)**: [`@prismatic-labs/vetch-ai-sdk`](packages/vetch-ai-sdk/)
- **[Track local models (Ollama, vLLM, llama.cpp)](QUICKSTART-LOCAL.md)**
- **[Interactive Inference Calculator](https://prismatic-labs.github.io/vetch/calculator/)**: compare energy, cost, and carbon across the bundled model registry

## Capabilities

- **Providers (auto-instrumented):** OpenAI, Anthropic, Google GenAI (`google-genai`), Google Vertex AI, Azure OpenAI, Ollama. `pip install vetch[openai]`, `vetch[anthropic]`, `vetch[genai]`, `vetch[vertexai]`, `vetch[ollama]`.
- **Self-hosted / OpenAI-compatible:** vLLM, TGI, LM Studio, and llama.cpp, reached through an OpenAI client with a custom `base_url` (classified automatically) or raw HTTP through `vetch.proxy` / `vetch.wrap()`. See [Self-hosted and raw HTTP](#self-hosted-and-raw-http).
- **Streaming:** input/output token instrumentation for sync and async streams, without buffering response content.
- **Framework integrations:** LangChain and LlamaIndex callback handlers; first-party [Vercel AI SDK middleware](packages/vetch-ai-sdk/) (JS/TS).
- **Control:** circuit breakers (stall detection → warn/kill/reroute) and warn-only budgets on cost/energy/carbon.
- **Export & tooling:** OpenTelemetry / OTLP export (GenAI semantic conventions), an MCP server for agents, a CLI (`vetch estimate|compare|audit|calibrate`), and local GPU calibration.
- **Metadata-only:** never reads prompt or completion text. It touches only the model, token counts, timing, and finish reason.

## The problem

Old cloud waste was idle infrastructure: overprovisioned servers, forgotten instances, jobs that ran once and stayed scheduled. You could fix it by turning things off.

AI waste is different. It is active, accumulating, and invisible until the bill arrives. A stalled agent loop burns tokens on every iteration. A RAG pipeline retrieving irrelevant context bloats every prompt. A session that should have ended 40 calls ago is still running. Provider dashboards show total spend. They do not show which feature, customer, workflow, or agent session produced the waste, and they cannot stop the next occurrence automatically.

Every wasted inference call is wasted money, compute, energy, and carbon.

### Why not just use your provider dashboard?

- **No attribution.** Dashboards show cost by model and date, not by agent session, customer, or feature flag.
- **Read-only.** They cannot fire a circuit breaker when a session exceeds a budget or an agent loop stalls.
- **No pattern detection.** A dashboard cannot spot that 80% of your agent's last ten outputs were under 20 tokens, the signature of a stalled loop.
- **No per-call energy or carbon data.** Reporting on inference resource use needs per-call instrumentation the provider does not expose.

## Detected waste patterns

Vetch analyzes every inference call for behavioral patterns that indicate waste. Each pattern has a stable advisory ID.

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
| `PREMIUM-001` | Large model rightsizing candidate | Stable tagged workflow mostly uses a premium model with cheaper eval candidates | ✅ Implemented (audit-only) |
| `TOOL-DEAD-001` | Dead function tools | Tools offered on many requests but never invoked | ✅ Implemented |
| `CAP-001` | Declared capabilities silent | Expected `kind:name` routes never fired in audit window | ✅ Implemented (audit-only) |
| `SESSION-BUDGET-001` | Session over budget | Configured cost/energy/carbon threshold exceeded | ⚠️ Partial (alerts only) |
| `ATTRIBUTION-001` | Unattributed spend | Required tags missing from calls | ⚠️ Partial (infrastructure only) |
| `RETRY-001` | Retry storm | Burst of repeated failed or near-identical calls | 🔜 Planned |

Full taxonomy with detection signals, false positives, and recommended actions: [docs/inference-waste-taxonomy.md](docs/inference-waste-taxonomy.md).

Advisories are deterministic signals, not proof of waste. Confidence labels indicate signal strength from metadata patterns, not statistical certainty. Non-stall runtime advisories are warn-only; `PREMIUM-001` is audit-only and queues eval candidates rather than recommending an automatic downgrade. Automatic kill and reroute are scoped to `STALL-001`.

## Quick start

Two lines to start tracking inference waste in your existing LLM calls. See [QUICKSTART.md](QUICKSTART.md) for the complete 60-second guide.

### `instrument()`: global, zero-touch

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

### `wrap()`: per-call, explicit

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

Both are fail-open and add <5ms overhead. Async is supported via `awrap()`.

**Import order matters.** `instrument()` can only patch SDKs already imported: import your SDK *before* calling `instrument()`, or call it again afterwards. Check coverage at runtime with `vetch.instrumentation_status()`. Full coverage matrix (frameworks, versions tested, transitive support): [QUICKSTART.md](QUICKSTART.md).

## Attribute waste

Every inference call is tagged and attributed to a session. Sessions can carry any tags you define (`feature`, `customer`, `user`, `workflow`, `environment`, `team`), and cost, energy, and carbon accumulate per session and per tag combination.

```python
import vetch

with vetch.Session(tags={"agent": "researcher", "task": "summarize"}) as session:
    with vetch.wrap(tags={"feature": "rag-search", "customer": "acme"}) as ctx:
        response = client.chat.completions.create(...)

print(f"Total cost:   ${session.total_cost_usd}")
print(f"Total energy: {session.total_energy_wh} Wh")
print(f"Call count:   {session.call_count}")
```

Sessions nest, and support distributed propagation across microservices via `session.inject_headers()` / `vetch.Session.from_headers()`. See [docs/attribution.md](docs/attribution.md) for tool/capability observability (dead-tool schema waste, `configure_capabilities`) and header propagation examples.

## Stop waste automatically

When `STALL-001` fires, Vetch can intervene without manual action:

| Action | What happens |
|--------|-------------|
| `"log"` (default) | Generate the advisory, take no action. Backwards compatible. |
| `"warn"` | Log a stderr warning on the next call after a stall. |
| `"kill"` | Raise `vetch.StallDetected` on the next call, breaking the loop. |
| `"reroute"` | Transparently substitute the model with `fallback_model`. |

`set_stall_action` is currently wired to `STALL-001`. Per-advisory, per-tag, and per-session policies are planned; see [ROADMAP.md](ROADMAP.md).

`vetch.StallDetected` inherits from `RuntimeError`, so a generic `except ValueError:` handler will not swallow it. Recover with `session.clear_stall()` after a human-in-the-loop fix. The circuit breaker needs an explicit `Session`. [docs/how-detection-works.md](docs/how-detection-works.md) explains why `instrument()` alone is observability-only.

Thresholds can be tuned per workflow when a pattern is expected. For example, a classification route that returns three tokens can lower the STALL-001 low-output threshold without affecting other routes:

```python
with vetch.Session(
    tags={"route": "classifier"},
    advisory_thresholds={"STALL-001": {"low_output_threshold": 1}},
):
    response = client.chat.completions.create(...)
```

## Prove savings: the 7-day audit

`vetch audit` reads locally stored metadata, runs advisory detection, computes per-tag attribution, and estimates observed and projected avoidable cost. A typical adoption motion:

1. **Instrument.** Call `vetch.instrument(region="us-east-1", tags={"service": "my-service"})`. All providers now tracked.
2. **Tag and observe.** Add tags to attribute spend, and run in `set_stall_action("warn")` to watch advisories without intervening.
3. **Audit.** After real traffic accumulates, run the report.
4. **Promote.** For confirmed `STALL-001` patterns, move to `"kill"` or `"reroute"`. Treat non-stall advisories as a review queue: fix the workflow, retriever, cache config, or attribution gaps before automating.

```bash
vetch audit                    # last 7 days (default)
vetch audit --window 24h       # shorter window
vetch audit --tags team=ml     # filter by tag
vetch audit --format json      # machine-readable
vetch audit --format markdown  # shareable report
```

Output includes advisory findings with signal-strength labels and recommended actions, per-tag attribution breakdowns, observed and projected avoidable cost, and data-quality indicators (tagged fraction, methodology versions). Full report format and adoption walkthrough: [docs/audit-report.md](docs/audit-report.md).

Programmatic access is available via `vetch.audit_report.build_audit_report()` / `format_audit_report()`.

## Energy and carbon

Vetch treats energy and carbon as first-class outputs alongside cost. Figures should be read with explicit uncertainty, because not all numbers are equal. Every event carries `energy_tier`, `energy_uncertainty_pct`, and `model_match` so you can tell measured from inferred from guessed.

| Tier | Name | Uncertainty | Source |
|------|------|-------------|--------|
| 0 | **Measured** | ±10–20% | Direct GPU telemetry from a local `vetch calibrate` run |
| 1 | **Inferred** | ±20–50% | Infrastructure-aware benchmarking of hosted APIs (no power meter exists) |
| 2 | **Validated** | ±50–100% | Crowdsourced aggregates |
| 3 | **Estimated** | Order of magnitude | Parameter-based calculation, or a proxy/family match |

The bundled registry covers 60 models, with aliases resolving many more dated and versioned names. Run `vetch methodology` for the current per-model tier and provenance. Use these estimates as internal inputs for FinOps, engineering, and sustainability planning. They are not carbon certification, regulatory disclosure, or water accounting. For methodology, citations, and Tier 1 coverage, see [docs/energy-methodology.md](docs/energy-methodology.md).

### Model coverage and resolution

Vetch resolves a model name to a registry entry through a ladder; each event records which rung matched in `model_match`: `exact` → `alias` → `prefix` (downgraded to Tier 3) → `family` (Tier 3) → `fallback`. If a current model isn't in the bundled registry yet, add a row (see [src/vetch/registry/PROVENANCE.md](src/vetch/registry/PROVENANCE.md)), `vetch calibrate` it, or pass `energy_override` to `wrap()`. Details: [docs/model-resolution.md](docs/model-resolution.md).

## Region configuration

The `region` parameter selects the electricity grid used for carbon intensity, and should match the **Electricity Maps zone identifier** (typically aligned with cloud region names: `us-east-1`, `eu-west-1`, `eastus`). OpenAI and Anthropic route globally and do not expose per-call location, so use your best estimate.

If `region` is unset, Vetch falls back to `VETCH_REGION`, then cloud provider env vars, then a coarse timezone heuristic (often inaccurate). **Always set `region` or `VETCH_REGION` explicitly for accurate carbon numbers.** See [docs/region-config.md](docs/region-config.md).

## Budget alerts

```python
import vetch

vetch.set_budget("hourly", cost_usd=10.0, energy_wh=50.0)

@vetch.on_budget_alert
def handle_alert(alert):
    print(f"Budget alert: {alert}")
```

Budget thresholds never block LLM calls; they trigger alerts only. Blocking policies are planned.

## OTLP export (Grafana, Datadog)

Export waste advisories, per-call cost, energy, and carbon to any OpenTelemetry-compatible backend:

```python
import vetch

vetch.configure_otlp_export(endpoint="http://localhost:4317", service_name="my-llm-service")
# vetch dashboard --export grafana --output grafana_vetch.json   # pre-built dashboard
```

## MCP server (AI agent integration)

Vetch ships an [MCP](https://modelcontextprotocol.io/) server (`pip install vetch[mcp]`) that gives agents real-time access to energy, cost, and carbon data, so they can check budgets, compare models, and make sustainability-aware decisions mid-conversation.

```json
{
  "mcpServers": {
    "vetch": {
      "command": "vetch-mcp",
      "env": { "VETCH_REGION": "us-east-1" }
    }
  }
}
```

Tools include `vetch_estimate`, `vetch_compare`, `vetch_session_stats`, `vetch_status`, `vetch_check_budget`, `vetch_grid_intensity`, `vetch_cleanest_region`, and `vetch_registry_lookup`. Full tool and resource reference: [docs/mcp.md](docs/mcp.md).

## CLI usage

```bash
vetch status                                                  # status and configuration
vetch estimate --model gpt-4o --input-tokens 1000 --output-tokens 500
vetch compare --models gpt-4o,claude-3-opus,gemini-1.5-pro --tokens 1000
vetch audit --window 24h --tags team=ml --format json         # stored-event audit
vetch report --days 7 --tags team=ml                          # usage report
vetch dashboard --export grafana --output dashboard.json
vetch registry freeze --output vetch_registry.json            # freeze for CI/CD
```

## Self-hosted and raw HTTP

Self-hosted serving (vLLM, TGI, LM Studio, llama.cpp) is usually reached through the OpenAI SDK with a custom `base_url`, or via raw HTTP. Vetch handles both, and **never bills a self-hosted or third-party endpoint at OpenAI's per-token rates:**

```python
import vetch
from openai import OpenAI

vetch.instrument()
client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")  # local vLLM
#   api.openai.com / *.openai.azure.com  -> "openai"             (OpenAI energy + list price)
#   localhost / 127.0.0.1 / private IPs  -> "self-hosted"        (calibration energy, cost = 0)
#   other public hosts (OpenRouter, ...) -> "openai-compatible"  (registry energy, cost = unknown)
```

If a private host can't be auto-detected, set `VETCH_SELF_HOSTED=true`. Raw HTTP requests bypass every SDK patch, so meter them explicitly with `record_usage()` from the token counts the response already carries:

```python
import vetch, httpx

resp = httpx.post("http://localhost:8000/v1/chat/completions", json=body).json()  # body = your request payload
usage = resp["usage"]
vetch.record_usage(
    model="llama-3-70b",
    input_tokens=usage["prompt_tokens"],
    output_tokens=usage["completion_tokens"],
    provider_hint="self-hosted",  # cost 0, energy/carbon still computed
    region="us-east-1",
)
```

`record_usage` runs the same calculation and emit path as an instrumented call, so the event is schema-identical and flows into the same sessions, budgets, and exporters. Full examples and `vetch.proxy` usage: [QUICKSTART-LOCAL.md](QUICKSTART-LOCAL.md).

## GPU calibration (local inference)

For local inference, calibrate energy measurements against actual GPU power draw for a Tier 0 figure:

```python
from vetch.calibrate import calibrate_model, format_calibration_result

def my_inference():
    response = ollama.generate(model="llama3.1:8b", prompt="Hello world")
    return 100, 50  # (input_tokens, output_tokens)

result = calibrate_model("ollama", "llama3.1:8b", workload=my_inference)
print(format_calibration_result(result))
```

**Requirements:** NVIDIA GPU with `pynvml` (`pip install nvidia-ml-py3`). On Apple Silicon, use `vetch calibrate-apple-silicon` (powermetrics-based, requires `sudo`). Both paths write a versioned identity-keyed record under `~/.vetch/calibrations/` (picked up automatically). To share a calibration, open a GitHub issue with that JSON (see the CLI result footer). See [QUICKSTART-LOCAL.md](QUICKSTART-LOCAL.md).

## Supported providers

| Provider | Status | Instrumentation |
|----------|--------|----------------|
| OpenAI | Supported | `vetch.instrument()` or `vetch.wrap()` |
| Azure OpenAI | Supported | `vetch.instrument()` (auto-detects `AzureOpenAI`) |
| Anthropic | Supported | `vetch.instrument()` or `vetch.wrap()` |
| Vertex AI (Gemini) | Supported | `vetch.instrument()` or `vetch.wrap()` |
| Ollama | Supported | Native SDK or OpenAI-compat API (auto-detected) |
| OpenRouter / Together.ai / Anyscale | Compatible | OpenAI-compatible API |
| vLLM / TGI | Compatible | OpenAI-compatible API |

OpenAI-compatible endpoints work automatically with `vetch.instrument()` and are classified by `base_url` so they are **not** billed at OpenAI's rates. See [Self-hosted and raw HTTP](#self-hosted-and-raw-http).

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
| `VETCH_SELF_HOSTED` | Force the self-hosted provider label when the host can't be auto-classified |
| `ELECTRICITY_MAPS_API_KEY` | API key for live grid carbon intensity data |
| `VETCH_CACHE_MODE` | Set to `memory-only` for serverless/Lambda environments |

## Design guarantees

- **Fail-open.** Every operation (patching, calculation, emission) is wrapped in isolated error handlers. If Vetch fails, your LLM call proceeds normally and a `tracking_disabled: true` event is logged. Vetch will never cause an inference outage.
- **Privacy.** Vetch never stores prompt or completion text. It records only token counts, model names, timing, tags, finish reason, and visible output character count. No PII or prompt data leaves your environment.
- **Thread safety.** `contextvars` for async session isolation, locked session stats, and `WeakKeyDictionary` for client patching. Create a `Session` per request/job/agent; set global config (`set_stall_action`, `set_advisory_thresholds`) at startup, not per request.

## Current limitations

1. **Energy estimates are uncertain.** Most models use Tier 3 estimates. See `vetch methodology`.
2. **Region inference is a coarse heuristic.** Set `region` or `VETCH_REGION` explicitly for accurate carbon numbers.
3. **Automatic intervention is wired to STALL-001 only.** Per-advisory/tag/session policies are planned.
4. **Experimental modules.** `vetch.calibrate`, `vetch.storage`, and `vetch.ci` emit `FutureWarning` and may change.

## Troubleshooting

```bash
export VETCH_DISABLED=true   # emergency kill switch
export VETCH_OUTPUT=none     # silence all output
```

```python
import logging
logging.getLogger("vetch").setLevel(logging.DEBUG)  # debug logging
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing guidelines, and how to contribute energy data.

## License

Apache License 2.0. See `LICENSE` and `NOTICE` for details.
