"""Example: OpenTelemetry integration with vetch.

This example demonstrates how to export vetch inference events to OpenTelemetry
for integration with observability platforms like Datadog, Jaeger, or New Relic.

Prerequisites:
    pip install vetch[opentelemetry]
    pip install opentelemetry-sdk
    pip install opentelemetry-exporter-otlp-proto-grpc  # For OTLP export

Usage:
    export OPENAI_API_KEY=your_api_key_here
    python opentelemetry_example.py
"""

from __future__ import annotations

import os

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

import vetch

# Example 1: Basic OpenTelemetry Setup with Console Export
print("=" * 60)
print("Example 1: OpenTelemetry Setup (Console Export)")
print("=" * 60)

# Create a resource to identify your service
resource = Resource(attributes={SERVICE_NAME: "vetch-demo-service"})

# Set up the tracer provider
provider = TracerProvider(resource=resource)

# Add console exporter for demo purposes (prints spans to stdout)
console_exporter = ConsoleSpanExporter()
provider.add_span_processor(BatchSpanProcessor(console_exporter))

# Set as global tracer provider
trace.set_tracer_provider(provider)

print("✓ OpenTelemetry configured with console exporter")
print()

# Example 2: Enable Vetch Auto-Export to OTel
print("=" * 60)
print("Example 2: Vetch Auto-Export to OpenTelemetry")
print("=" * 60)

# Configure vetch to auto-export to OpenTelemetry
vetch.configure_otel_export(enabled=True)

# Auto-instrument all LLM SDKs
vetch.instrument(region="us-east-1", tags={"service": "demo", "env": "dev"})

print("✓ Vetch auto-export enabled")
print("✓ All LLM calls will create OTel spans with GenAI semantic conventions")
print()

# Example 3: Make an LLM Call (auto-tracked + auto-exported to OTel)
try:
    import openai

    print("=" * 60)
    print("Example 3: LLM Call with OTel Export")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠ OPENAI_API_KEY not set, skipping LLM call")
        print()
    else:
        # Create a custom span to group LLM calls
        tracer = trace.get_tracer(__name__)

        with tracer.start_as_current_span("chat_conversation") as parent_span:
            parent_span.set_attribute("conversation.id", "demo-123")
            parent_span.set_attribute("user.id", "demo-user")

            # This LLM call will:
            # 1. Be tracked by vetch (cost, energy, carbon)
            # 2. Create a child OTel span with GenAI semantic conventions
            # 3. Include custom vetch attributes (energy, carbon, cost)
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say hello in one word"}],
            )

            print(f"Response: {response.choices[0].message.content}")
            print()
            print("✓ Vetch event logged (cost, energy, carbon)")
            print("✓ OTel span created with GenAI semantic conventions")
            print("✓ Check console output above for span details")
            print()

except ImportError:
    print("⚠ openai not installed, skipping example")
    print()

# Example 4: OTLP Export to Jaeger/Datadog/etc.
print("=" * 60)
print("Example 4: OTLP Export (Production Setup)")
print("=" * 60)
print()
print("For production, replace ConsoleSpanExporter with OTLP exporter:")
print()
print("```python")
print("from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import \\")
print("    OTLPSpanExporter")
print()
print("# Configure OTLP endpoint (Jaeger, Datadog, etc.)")
print("otlp_exporter = OTLPSpanExporter(")
print('    endpoint="http://localhost:4317",  # Jaeger OTLP endpoint')
print("    # Or for Datadog:")
print('    # endpoint="https://api.datadoghq.com:4317",')
print('    # headers={"DD-API-KEY": os.getenv("DD_API_KEY")}')
print(")")
print()
print("provider.add_span_processor(BatchSpanProcessor(otlp_exporter))")
print("```")
print()

# Example 5: Manual Export (Advanced)
print("=" * 60)
print("Example 5: Manual Export (Advanced Use Case)")
print("=" * 60)
print()
print("For fine-grained control, disable auto-export and manually export:")
print()
print("```python")
print("import vetch")
print("from vetch.exporters.opentelemetry import export_event_as_span")
print()
print("# Disable auto-export")
print("vetch.configure_otel_export(enabled=False)")
print()
print("with vetch.wrap(region='us-east-1') as ctx:")
print("    response = client.chat.completions.create(...)")
print()
print("# Manually export specific events")
print("if ctx.event['estimated_cost_usd'] > 0.01:  # Only export expensive calls")
print("    export_event_as_span(ctx.event, tracer=tracer)")
print("```")
print()

# Example 6: Environment Variables for OTLP
print("=" * 60)
print("Example 6: Environment Variables")
print("=" * 60)
print()
print("OpenTelemetry supports configuration via env vars:")
print()
print("```bash")
print("# OTLP endpoint")
print("export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317")
print()
print("# Service name")
print("export OTEL_SERVICE_NAME=my-llm-service")
print()
print("# Resource attributes")
print('export OTEL_RESOURCE_ATTRIBUTES="deployment.environment=production"')
print()
print("# Headers (for authentication)")
print('export OTEL_EXPORTER_OTLP_HEADERS="api-key=your-api-key"')
print()
print("python your_app.py")
print("```")
print()

print("=" * 60)
print("OpenTelemetry Span Attributes")
print("=" * 60)
print()
print("Vetch exports spans with these attributes:")
print()
print("GenAI Semantic Conventions (Standard):")
print("  - gen_ai.system: 'openai', 'anthropic', 'vertex_ai'")
print("  - gen_ai.request.model: 'gpt-4o', 'claude-3-5-sonnet', etc.")
print("  - gen_ai.usage.input_tokens: Number of input tokens")
print("  - gen_ai.usage.output_tokens: Number of output tokens")
print()
print("Custom Vetch Attributes:")
print("  - vetch.cost.usd: Estimated cost in USD")
print("  - vetch.energy.wh: Energy consumption in watt-hours")
print("  - vetch.carbon.g: Carbon emissions in grams CO2e")
print("  - vetch.water.l: Water usage in liters")
print("  - vetch.region: Cloud region (e.g., 'us-east-1')")
print("  - vetch.tier: Energy estimate tier (1=measured, 2=derived, 3=estimated)")
print()

print("=" * 60)
print("Integration with Observability Platforms")
print("=" * 60)
print()
print("Datadog:")
print("  1. Install: pip install opentelemetry-exporter-otlp")
print("  2. Set DD_API_KEY environment variable")
print("  3. Configure endpoint: https://api.datadoghq.com:4317")
print("  4. Traces appear in Datadog APM with custom vetch metrics")
print()
print("Jaeger:")
print("  1. Run Jaeger: docker run -p 4317:4317 jaegertracing/all-in-one")
print("  2. Configure endpoint: http://localhost:4317")
print("  3. View traces at http://localhost:16686")
print()
print("New Relic:")
print("  1. Install: pip install opentelemetry-exporter-otlp")
print("  2. Set NEW_RELIC_LICENSE_KEY environment variable")
print("  3. Configure endpoint: https://otlp.nr-data.net:4317")
print("  4. Add header: api-key=<your-license-key>")
print()

print("✅ OpenTelemetry setup complete!")
print()
print("Next steps:")
print("  1. Run this example: python examples/opentelemetry_example.py")
print("  2. Check console output for span data")
print("  3. Configure OTLP exporter for your observability platform")
print("  4. Deploy to production with env vars")
