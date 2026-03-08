# Vetch Examples

This directory contains examples demonstrating vetch usage patterns.

## Auto-Instrumentation Example

**File**: [`auto_instrument_example.py`](auto_instrument_example.py)

Demonstrates the simplest way to use vetch: call `vetch.instrument()` once at startup and all LLM calls are automatically tracked.

## OpenTelemetry Integration Example

**File**: [`opentelemetry_example.py`](opentelemetry_example.py)

Demonstrates how to export vetch inference events to OpenTelemetry for integration with observability platforms like Datadog, Jaeger, and New Relic.

**Supported SDKs**:
- Google GenAI (`google-genai`)
- OpenAI (`openai`)
- Anthropic (`anthropic`)
- Vertex AI (`google-cloud-aiplatform`)
- Azure OpenAI (via `openai` with `azure_endpoint`)

**Usage**:
```bash
# Set your API keys
export GOOGLE_API_KEY=your_key_here
export OPENAI_API_KEY=your_key_here
export ANTHROPIC_API_KEY=your_key_here

# Run the example
python auto_instrument_example.py
```

**What you'll see**:
- Automatic tracking of LLM calls without context managers
- JSON events logged to stderr with cost, energy, and carbon
- Works with chat completions and embeddings

## Environment Variables

Vetch respects these environment variables:

- `VETCH_REGION`: Default region for carbon calculations (e.g., `us-east-1`)
- `VETCH_ENABLED`: Set to `false` to disable tracking (default: `true`)
- `VETCH_DISABLED`: Legacy env var, same as `VETCH_ENABLED=false`

## More Examples

For additional examples and use cases, see:
- [QUICKSTART.md](../QUICKSTART.md) - 60-second cloud API guide
- [QUICKSTART-LOCAL.md](../QUICKSTART-LOCAL.md) - Local model tracking
- [demo.ipynb](../demo.ipynb) - Interactive Jupyter notebook
