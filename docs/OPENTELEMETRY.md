# OpenTelemetry Integration Guide

Vetch can export inference events as OpenTelemetry spans, enabling integration with observability platforms like Datadog, Jaeger, New Relic, and more.

## Quick Start

### Installation

```bash
pip install vetch[opentelemetry]
pip install opentelemetry-sdk
pip install opentelemetry-exporter-otlp-proto-grpc
```

### Basic Setup

```python
import vetch
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME

# 1. Configure OpenTelemetry
resource = Resource(attributes={SERVICE_NAME: "my-llm-service"})
provider = TracerProvider(resource=resource)
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

# 2. Enable vetch auto-export to OTel
vetch.configure_otel_export(enabled=True)

# 3. Auto-instrument LLM SDKs
vetch.instrument(region="us-east-1", tags={"service": "chat-api"})

# 4. Make LLM calls - automatically exported to OTel!
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
# Span created automatically with GenAI semantic conventions + vetch metrics
```

## Configuration Options

### Auto-Export Mode (Recommended)

Automatically export all vetch events as OTel spans:

```python
import vetch

# Enable once at startup
vetch.configure_otel_export(enabled=True)

# All LLM calls now create OTel spans
with vetch.wrap(region="us-east-1") as ctx:
    response = client.chat.completions.create(...)
# Span created on context exit
```

### Manual Export Mode (Advanced)

For fine-grained control over which events to export:

```python
import vetch
from vetch.exporters.opentelemetry import export_event_as_span
from opentelemetry import trace

# Disable auto-export
vetch.configure_otel_export(enabled=False)

tracer = trace.get_tracer(__name__)

with vetch.wrap(region="us-east-1") as ctx:
    response = client.chat.completions.create(...)

# Manually export only expensive calls
if ctx.event['estimated_cost_usd'] > 0.01:
    export_event_as_span(ctx.event, tracer=tracer)
```

## Production Deployment

### OTLP Export to Jaeger

```python
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

otlp_exporter = OTLPSpanExporter(
    endpoint="http://localhost:4317",  # Jaeger OTLP endpoint
)
provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
```

**Run Jaeger locally:**
```bash
docker run -d -p 4317:4317 -p 16686:16686 jaegertracing/all-in-one
```

**View traces:** http://localhost:16686

### OTLP Export to Datadog

```python
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
import os

otlp_exporter = OTLPSpanExporter(
    endpoint="https://api.datadoghq.com:4317",
    headers={
        "DD-API-KEY": os.getenv("DD_API_KEY")
    }
)
provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
```

**Environment variables:**
```bash
export DD_API_KEY=your_datadog_api_key
export DD_SITE=datadoghq.com  # or datadoghq.eu for EU
```

### OTLP Export to New Relic

```python
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
import os

otlp_exporter = OTLPSpanExporter(
    endpoint="https://otlp.nr-data.net:4317",
    headers={
        "api-key": os.getenv("NEW_RELIC_LICENSE_KEY")
    }
)
provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
```

**Environment variables:**
```bash
export NEW_RELIC_LICENSE_KEY=your_license_key
```

### OTLP Export to Honeycomb

```python
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
import os

otlp_exporter = OTLPSpanExporter(
    endpoint="https://api.honeycomb.io:4317",
    headers={
        "x-honeycomb-team": os.getenv("HONEYCOMB_API_KEY"),
        "x-honeycomb-dataset": "llm-traces"
    }
)
provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
```

## Environment Variables

OpenTelemetry supports configuration via environment variables:

```bash
# OTLP endpoint
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# Service name
export OTEL_SERVICE_NAME=my-llm-service

# Resource attributes
export OTEL_RESOURCE_ATTRIBUTES="deployment.environment=production,service.version=1.0.0"

# Headers (for authentication)
export OTEL_EXPORTER_OTLP_HEADERS="api-key=your-api-key"

# Sampling (1.0 = 100% of traces)
export OTEL_TRACES_SAMPLER=parentbased_traceidratio
export OTEL_TRACES_SAMPLER_ARG=1.0
```

## Span Attributes

Vetch exports spans with the following attributes:

### GenAI Semantic Conventions (Standard)

Following [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/):

- `gen_ai.system`: Provider name (`openai`, `anthropic`, `vertex_ai`, `google_genai`)
- `gen_ai.request.model`: Model identifier (`gpt-4o`, `claude-3-5-sonnet-20241022`, etc.)
- `gen_ai.usage.input_tokens`: Number of input tokens
- `gen_ai.usage.output_tokens`: Number of output tokens
- `gen_ai.response.id`: Response ID from provider
- `gen_ai.operation.name`: Operation type (`chat`, `completion`, `embedding`)

### Custom Vetch Attributes

Vetch adds custom attributes for sustainability and cost tracking:

- `vetch.cost.usd`: Estimated cost in USD (e.g., `0.00045`)
- `vetch.energy.wh`: Energy consumption in watt-hours (e.g., `1.2`)
- `vetch.carbon.g`: Carbon emissions in grams CO2e (e.g., `0.48`)
- `vetch.water.l`: Water usage in liters (e.g., `0.0013`)
- `vetch.region`: Cloud region for carbon calculation (`us-east-1`, `eu-west-1`, etc.)
- `vetch.tier`: Energy estimate quality (1=measured, 2=derived, 3=estimated)
- `vetch.tracking_degraded`: Boolean indicating reduced tracking accuracy

### User-Defined Tags

All tags passed to `vetch.wrap()` or `vetch.instrument()` are added as span attributes with the `vetch.tag.` prefix:

```python
vetch.instrument(tags={"team": "ml", "env": "production"})
# Results in span attributes:
# - vetch.tag.team = "ml"
# - vetch.tag.env = "production"
```

## Example: Parent-Child Span Hierarchy

Create a span hierarchy for complex LLM workflows:

```python
from opentelemetry import trace
import vetch

tracer = trace.get_tracer(__name__)
vetch.configure_otel_export(enabled=True)
vetch.instrument(region="us-east-1")

# Parent span for the entire conversation
with tracer.start_as_current_span("rag_conversation") as conversation_span:
    conversation_span.set_attribute("conversation.id", "conv-123")
    conversation_span.set_attribute("user.id", "user-456")

    # Child span 1: Embedding query
    with tracer.start_as_current_span("embed_query") as embed_span:
        embeddings = client.embeddings.create(
            model="text-embedding-3-small",
            input="What is the capital of France?"
        )
        # Vetch creates a child span with embedding metrics

    # Child span 2: Vector search (your custom instrumentation)
    with tracer.start_as_current_span("vector_search") as search_span:
        results = vector_db.search(embeddings)
        search_span.set_attribute("results.count", len(results))

    # Child span 3: LLM completion
    with tracer.start_as_current_span("llm_completion") as completion_span:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ]
        )
        # Vetch creates a child span with cost/energy/carbon
```

**Resulting trace structure:**
```
rag_conversation (parent)
├── embed_query
│   └── vetch.embedding.openai (auto-created by vetch)
├── vector_search
└── llm_completion
    └── vetch.chat.openai (auto-created by vetch)
```

## Distributed Tracing

Vetch automatically propagates W3C trace context for distributed tracing:

### Service A (API Gateway)
```python
from opentelemetry import trace
import vetch

tracer = trace.get_tracer(__name__)
vetch.instrument(region="us-east-1", tags={"service": "api-gateway"})

with tracer.start_as_current_span("handle_request") as span:
    # Make a request to Service B
    # Trace context automatically propagated via HTTP headers
    response = requests.post("http://service-b/chat", json={"query": "..."})
```

### Service B (LLM Service)
```python
import vetch

vetch.instrument(region="us-east-1", tags={"service": "llm-backend"})

# LLM calls will be child spans of the incoming request
# Trace context automatically extracted from HTTP headers
client = openai.OpenAI()
response = client.chat.completions.create(...)
# Span automatically linked to parent trace from Service A
```

## Querying Vetch Metrics in Observability Platforms

### Datadog APM

**Find expensive LLM calls:**
```
@vetch.cost.usd:>0.01 service:my-llm-service
```

**Track carbon emissions by region:**
```
avg:vetch.carbon.g by {vetch.region}
```

**Alert on high energy usage:**
```
avg:vetch.energy.wh > 10
```

### Jaeger

**Filter by model:**
```
gen_ai.request.model="gpt-4o" AND vetch.cost.usd > 0.01
```

**Find degraded tracking:**
```
vetch.tracking_degraded=true
```

### Honeycomb

**Create a heatmap of cost by model:**
```
GROUP BY gen_ai.request.model
AVG(vetch.cost.usd)
HEATMAP(duration_ms)
```

## Sampling Strategies

For high-volume production systems, use sampling to reduce trace volume:

### Head-Based Sampling (Recommended)

```python
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

# Sample 10% of traces
sampler = TraceIdRatioBased(0.1)
provider = TracerProvider(sampler=sampler, resource=resource)
```

### Tail-Based Sampling (Advanced)

Sample traces based on vetch metrics (requires tail-based sampling collector):

```yaml
# OpenTelemetry Collector config
processors:
  tail_sampling:
    decision_wait: 10s
    policies:
      # Always sample expensive calls
      - name: expensive-calls
        type: numeric_attribute
        numeric_attribute:
          key: vetch.cost.usd
          min_value: 0.01
      # Always sample high carbon calls
      - name: high-carbon
        type: numeric_attribute
        numeric_attribute:
          key: vetch.carbon.g
          min_value: 10.0
      # Sample 1% of other traces
      - name: probabilistic
        type: probabilistic
        probabilistic:
          sampling_percentage: 1
```

## Troubleshooting

### No spans appearing in backend

1. **Check OTel SDK is installed:**
   ```bash
   pip list | grep opentelemetry
   ```

2. **Enable debug logging:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

3. **Verify endpoint connectivity:**
   ```bash
   curl -v http://localhost:4317
   ```

4. **Check vetch auto-export is enabled:**
   ```python
   from vetch.exporters.opentelemetry import is_auto_export_enabled
   print(is_auto_export_enabled())  # Should be True
   ```

### Spans created but missing vetch attributes

1. **Verify vetch is installed with opentelemetry extra:**
   ```bash
   pip install vetch[opentelemetry]
   ```

2. **Check vetch.configure_otel_export() was called:**
   ```python
   import vetch
   vetch.configure_otel_export(enabled=True)
   ```

3. **Verify LLM calls are wrapped:**
   ```python
   # Auto-instrumentation
   vetch.instrument()

   # Or manual wrapping
   with vetch.wrap() as ctx:
       response = client.chat.completions.create(...)
   ```

## Best Practices

1. **Enable auto-export at startup:**
   ```python
   # Call once in main() or __init__.py
   vetch.configure_otel_export(enabled=True)
   ```

2. **Use resource attributes for service metadata:**
   ```python
   from opentelemetry.sdk.resources import Resource

   resource = Resource.create({
       "service.name": "my-llm-service",
       "service.version": "1.2.3",
       "deployment.environment": "production",
   })
   ```

3. **Set meaningful span names:**
   ```python
   tracer = trace.get_tracer(__name__)
   with tracer.start_as_current_span("user_query_handler"):
       # LLM calls will be children of this span
       response = client.chat.completions.create(...)
   ```

4. **Use sampling in production:**
   ```python
   # Sample 10% of traces to reduce costs
   from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
   sampler = TraceIdRatioBased(0.1)
   ```

5. **Monitor vetch metrics:**
   - Set up alerts for high `vetch.cost.usd`
   - Track `vetch.carbon.g` trends over time
   - Monitor `vetch.tracking_degraded` for data quality

## Related Documentation

- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Vetch Auto-Instrumentation Guide](../examples/auto_instrument_example.py)
- [Vetch Quickstart](../QUICKSTART.md)
