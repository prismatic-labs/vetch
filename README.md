# Vetch SDK

[![PyPI version](https://img.shields.io/pypi/v/vetch.svg)](https://pypi.org/project/vetch/)
[![Python versions](https://img.shields.io/pypi/pyversions/vetch.svg)](https://pypi.org/project/vetch/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/prismatic-labs/vetch/actions/workflows/ci.yml/badge.svg)](https://github.com/prismatic-labs/vetch/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prismatic-labs/vetch/blob/main/demo.ipynb)

Planet-aware observability for LLM inference.

Vetch is a Python SDK that wraps LLM API calls to log energy consumption, cost, and carbon per inference using live grid data. It never reads prompt or completion content—only metadata from the response usage.

## Why Vetch?

**Attributed Spend, Not Just Total Spend**

Provider dashboards (OpenAI Usage, Anthropic Console, Google Cloud Billing) show you *total* spend. Vetch shows you *attributed* spend. Using tags, you can track cost-per-feature, cost-per-user, or cost-per-environment in real-time—without building custom infrastructure.

**Sustainability Instrumentation**

Begin tracking AI inference emissions for future CSRD (EU) and SEC (US) Scope 3 reporting. Note: Current estimates are Tier 3 (order-of-magnitude). Vetch provides the instrumentation infrastructure—audit-grade accuracy requires Tier 1/2 energy data from providers or calibrated measurements.

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
- **Multi-tier Caching**: Memory → File → API → Regional averages for grid data
- **Observability-Transparent**: Works seamlessly with Datadog, OpenTelemetry, and Sentry
- **Low Overhead**: Under 5ms overhead for sync calls; zero TTFT latency for streaming
- **MoE-Aware**: Energy estimates account for active parameters in Mixture-of-Experts models

## Installation

```bash
pip install vetch
```

## Quick Start

```python
from vetch import wrap
from openai import OpenAI

client = OpenAI()

with wrap(region="us-east-1", tags={"team": "ml", "env": "prod"}) as ctx:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello world"}]
    )

# Access inference metadata
print(f"Energy: {ctx.event['estimated_energy_wh']} Wh")
print(f"Carbon: {ctx.event['estimated_carbon_g']} gCO2e")
```

## CLI Usage

Estimate energy/carbon for a model without running code:
```bash
vetch estimate --model gpt-4o --input-tokens 1000 --output-tokens 500 --region us-east-1
```

Compare multiple models:
```bash
vetch compare --models gpt-4o,claude-3-opus,gemini-1.5-pro --tokens 1000
```

Analyze your token usage patterns:
```bash
vetch audit
```

Check your environment:
```bash
vetch check
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
    # Run your inference workload
    # Return (input_tokens, output_tokens)
    response = ollama.generate(model="llama3.1:8b", prompt="Hello world")
    return 100, 50  # Your actual token counts

result = calibrate_model("ollama", "llama3.1:8b", workload=my_inference)
print(format_calibration_result(result))

# Use calibrated values for accurate tracking
with wrap(energy_override=result.to_override()) as ctx:
    response = ollama.generate(...)
```

Check calibration status:
```bash
vetch calibrate --status
```

**Requirements:** NVIDIA GPU with `pynvml` (`pip install nvidia-ml-py3`)

## Historical Analysis & Reporting

Vetch can persist events to SQLite for historical FinOps analysis:

```python
from vetch import configure_storage, query_usage, wrap
from datetime import datetime, timedelta

# Enable persistent storage
configure_storage()  # Uses ~/.vetch/usage.db

# Your LLM calls are now tracked
with wrap(tags={"team": "ml", "feature": "chat"}) as ctx:
    response = client.chat.completions.create(...)

# Query historical usage
summary = query_usage(
    start=datetime.now() - timedelta(days=7),
    tags={"team": "ml"}
)

print(f"Total cost: ${summary.total_cost_usd:.2f}")
print(f"Total energy: {summary.total_energy_wh:.2f} Wh")
print(f"Requests: {summary.total_requests}")
```

Generate reports from CLI:
```bash
# Weekly report
vetch report --days 7

# Filter by team
vetch report --tags team=ml

# Show top consumers
vetch report --top --top-by team --days 30

# JSON output for dashboards
vetch report --format json
```

## Energy Tiers

Vetch uses a tiered system for energy estimate confidence:

| Tier | Name | Uncertainty | Source |
|------|------|-------------|--------|
| 0 | **Measured** | ±10-20% | Direct GPU measurement (pynvml) |
| 1 | **Vendor-Published** | ±20-50% | Official provider data |
| 2 | **Validated** | ±50-100% | Crowdsourced aggregates |
| 3 | **Estimated** | order of magnitude | Parameter-based calculation |

Run `vetch methodology` to see full methodology documentation.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `VETCH_DISABLED` | Set to `true` to completely disable Vetch (emergency kill switch) |
| `VETCH_REGION` | Default grid region (e.g., `us-east-1`, `eu-west-1`) |
| `VETCH_OUTPUT` | Output target: `none` (default), `stderr`, or file path |
| `ELECTRICITY_MAPS_API_KEY` | API key for live grid carbon intensity data |
| `VETCH_CACHE_MODE` | Set to `memory-only` for serverless/Lambda environments |

## Alpha Limitations

This is an alpha release. Please be aware of:

1. **Energy estimates are uncertain**: Most models use Tier 3 estimates (±10x uncertainty). See `vetch methodology` for details.

2. **Region inference is approximate**: Without explicit `VETCH_REGION`, timezone-based inference is ~30% accurate. Set the region explicitly for accurate carbon calculations.

3. **Experimental modules**: `vetch.calibrate`, `vetch.storage`, and `vetch.ci` emit `FutureWarning` and may change in future versions.

4. **Provider support**: Currently supports OpenAI, Anthropic, and Vertex AI. Other providers coming soon.

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

## License

Apache License 2.0. See `LICENSE` and `NOTICE` for details.
