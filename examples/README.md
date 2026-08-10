# Vetch Examples

This directory contains examples demonstrating vetch usage patterns.

## Circuit Breaker (kill a runaway agent)

**Files:** [`circuit_breaker_demo_web.py`](circuit_breaker_demo_web.py), [`circuit_breaker_demo.py`](circuit_breaker_demo.py)

Live STALL-001 demo: a deliberately stalled agent loop until Vetch warns, kills, or reroutes.

```bash
# Browser dashboard (default: mock events, no API key)
python examples/circuit_breaker_demo_web.py
# Then open http://localhost:8765

# Real OpenAI calls in the dashboard
python examples/circuit_breaker_demo_web.py --real

# CLI-only twin (requires OPENAI_API_KEY)
python examples/circuit_breaker_demo.py
```

## Auto-Instrumentation Example

**File**: [`auto_instrument_example.py`](auto_instrument_example.py)

Demonstrates the simplest way to use vetch: call `vetch.instrument()` once at startup and all LLM calls are automatically tracked.

```bash
export OPENAI_API_KEY=your_key_here   # and/or other provider keys
python examples/auto_instrument_example.py
```

## OpenTelemetry Integration Example

**File**: [`opentelemetry_example.py`](opentelemetry_example.py)

Demonstrates how to export vetch inference events to OpenTelemetry for integration with observability platforms like Datadog, Jaeger, and New Relic.

**Supported SDKs**:
- Google GenAI (`google-genai`)
- OpenAI (`openai`)
- Anthropic (`anthropic`)
- Vertex AI (`google-cloud-aiplatform`)
- Azure OpenAI (via `openai` with `azure_endpoint`)

```bash
export OPENAI_API_KEY=your_key_here
python examples/opentelemetry_example.py
```

## Environment Variables

Vetch respects these environment variables:

- `VETCH_REGION`: Default region for carbon calculations (e.g., `us-east-1`)
- `VETCH_ENABLED`: Set to `false` to disable tracking (default: `true`)
- `VETCH_DISABLED`: Legacy env var, same as `VETCH_ENABLED=false`

## More Examples

For additional examples and use cases, see:
- [QUICKSTART.md](../QUICKSTART.md) - 60-second cloud API guide
- [QUICKSTART-LOCAL.md](../QUICKSTART-LOCAL.md) - Local / self-hosted model tracking
- [QUICKSTART-VERCEL.md](../QUICKSTART-VERCEL.md) - Vercel AI SDK (JS/TS)
- [demo.ipynb](../demo.ipynb) - Interactive Jupyter notebook
