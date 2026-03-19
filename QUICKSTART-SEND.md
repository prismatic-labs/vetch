# Quickstart: Sending Vetch Data Somewhere

By default, vetch runs silently (`VETCH_OUTPUT=none`). This guide shows how to route events to wherever your team needs them.

---

## Option 1 — Local debug (stderr)

The fastest way to see what vetch is capturing:

```bash
VETCH_OUTPUT=stderr python your_app.py
```

Each LLM call prints a JSON line to stderr:

```json
{"provider":"openai","model":"gpt-4o","estimated_cost_usd":0.00245,"estimated_energy_wh":0.0031,"estimated_carbon_g":1.42,"is_stream":false}
```

---

## Option 2 — Internal HTTP endpoint (behind a firewall)

Send events to any HTTP endpoint your infrastructure already runs — no external traffic required.

**Environment variables (zero-code-change):**

```bash
export VETCH_ENDPOINT=https://analytics.internal.corp/vetch/ingest
python your_app.py
```

With authentication (e.g., a shared secret your internal service checks):

```bash
export VETCH_ENDPOINT=https://analytics.internal.corp/vetch/ingest
export VETCH_API_KEY=your-internal-secret
python your_app.py
```

**Programmatic setup:**

```python
import vetch

vetch.configure_http_endpoint(
    "https://analytics.internal.corp/vetch/ingest",
    api_key="your-internal-secret",  # omit if behind firewall with no auth
)
vetch.instrument()
```

Events are POSTed as newline-delimited JSON, one event per request, asynchronously via a background thread. Your app is never blocked.

**Docker / Kubernetes:**

```yaml
# docker-compose.yml
environment:
  - VETCH_ENDPOINT=https://analytics.internal.corp/ingest
  - VETCH_API_KEY=${VETCH_API_KEY}

# k8s deployment
env:
  - name: VETCH_ENDPOINT
    value: "https://analytics.internal.corp/ingest"
  - name: VETCH_API_KEY
    valueFrom:
      secretKeyRef:
        name: vetch-secrets
        key: api-key
```

**What your endpoint receives:**

```http
POST /vetch/ingest HTTP/1.1
Content-Type: application/json
Authorization: Bearer your-internal-secret

{"schema_version":"2","provider":"openai","model":"gpt-4o","estimated_cost_usd":0.00245,...}
```

A minimal receiver in FastAPI (< 20 lines):

```python
from fastapi import FastAPI, Request
app = FastAPI()

@app.post("/vetch/ingest")
async def ingest(request: Request):
    event = await request.json()
    # Store in your database, forward to analytics, etc.
    print(f"{event['provider']}/{event['model']} — ${event.get('estimated_cost_usd', 0):.5f}")
    return {"ok": True}
```

---

## Option 3 — Existing OTLP / observability stack

If your team already runs Grafana, Datadog, Honeycomb, or Jaeger:

```python
import vetch

vetch.configure_otlp_export(
    endpoint="https://otel.internal.corp:4317",
    headers={"x-honeycomb-team": "your-key"},
    service_name="chat-api",
)
vetch.instrument()
```

Or via environment variables:

```bash
export VETCH_OTEL_EXPORT=true
export OTEL_EXPORTER_OTLP_ENDPOINT=https://otel.internal.corp:4317
export VETCH_OTEL_SERVICE_NAME=chat-api
python your_app.py
```

Events appear as OpenTelemetry spans with `gen_ai.*` and `vetch.*` attributes following the [GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/).

---

## Option 4 — File (for batch pipelines or CI)

```bash
VETCH_OUTPUT=/var/log/vetch/events.jsonl python your_app.py
```

Each run appends JSON lines. Works with `jq`, Splunk, Elasticsearch, etc.

---

## Route to multiple destinations

```python
import vetch

# Internal endpoint for real-time analytics
vetch.configure_http_endpoint("https://analytics.internal.corp/ingest")

# OTLP for existing tracing infrastructure
vetch.configure_otlp_export(endpoint="https://otel.internal.corp:4317")

# Both fire on every event
vetch.instrument()
```

---

## Green routing: send batch jobs to the cleanest region

Before dispatching a long batch job, check which of your available regions has the lowest carbon intensity right now:

```python
import vetch

region, intensity = vetch.get_cleanest_region(
    candidates=["us-east-1", "eu-west-1", "us-west-2"],
    api_key="your-electricity-maps-key",  # optional: ELECTRICITY_MAPS_API_KEY env var
)
print(f"Routing to {region} — {intensity:.0f} gCO₂e/kWh")

# Use the result directly in your vetch context
with vetch.wrap(region=region, tags={"job": "nightly-embeddings"}):
    # ... run your batch inference ...
    pass
```

Falls back to regional averages if live data is unavailable.

---

## Summary

| Goal | Config |
|------|--------|
| Local debug | `VETCH_OUTPUT=stderr` |
| Internal endpoint | `VETCH_ENDPOINT=https://...` |
| Authenticated endpoint | `VETCH_ENDPOINT=https://... VETCH_API_KEY=...` |
| Programmatic HTTP | `vetch.configure_http_endpoint(url, api_key)` |
| OTLP (Grafana/Datadog) | `vetch.configure_otlp_export(endpoint, headers)` |
| File output | `VETCH_OUTPUT=/path/to/file.jsonl` |
| Green routing | `vetch.get_cleanest_region(candidates)` |

> **Hosted dashboard and team analytics** — for a managed solution with cross-service aggregation,
> cost attribution dashboards, and CSRD-ready reports, see [vetch.io](https://vetch.io).
