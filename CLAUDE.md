# Vetch SDK Development Guidelines

## Project Overview

Vetch is an energy-aware observability SDK for LLM inference. It wraps API calls to log energy, cost, and carbon without reading prompt/completion content.

**License:** Apache 2.0

---

## Core Principles (Non-Negotiable)

### Fail-Open
- If Vetch fails (grid API down, calculation error, patch failure), the LLM call MUST proceed
- Never crash the host application
- Wrap errors, log them, continue

```python
try:
    # Vetch logic
except Exception:
    logger.warning("Vetch failed, continuing without tracking")
    ctx.tracking_disabled = True
# LLM call always proceeds
```

### Fail-Loud
- Every log includes `signal_quality` - no silent degradation
- Stale data? Say so. Missing region? Say so. Patching disabled? Say so.
- Collect warnings during processing, emit in `vetch_warnings` field

```python
# Validation functions return (result, warnings) tuples
def validate_something(data: dict) -> tuple[Result | None, list[str]]:
    warnings: list[str] = []
    if problem:
        warnings.append("Specific warning message")
    return result, warnings

# Wrapper collects warnings throughout lifecycle
self._warnings: list[str] = []
validated, warnings = validate_energy_override(override)
self._warnings.extend(warnings)
# ... emit vetch_warnings=self._warnings in event
```

### Privacy-First
- ZERO access to prompt or completion content
- Only touch: model name, token counts, region, latency
- Never buffer response content

### Observability-Transparent
- Detect prior patches (Datadog, OpenTelemetry, Sentry)
- Forward all attribute access
- Never break the observability chain

---

## Technical Constraints

### Dependencies
- **Runtime**: stdlib only (urllib, contextvars, tempfile, json, etc.)
- **Optional peers**: openai>=1.0,<2.0, google-cloud-aiplatform>=1.0
- **Test**: pytest, hypothesis, pytest-cov

### Python Version
- Minimum: Python 3.9
- Use `Union[X, Y]` not `X | Y` (3.10+ syntax)
- Use `dict[str, int]` not `Dict[str, int]` (3.9 supports lowercase generics)

### Performance
- Under 5ms overhead for synchronous calls
- Zero latency added to TTFT for streaming
- Memory-safe: count streaming chunks, don't accumulate

---

## Key Implementation Patterns

### Wrapper Transparency
```python
# Patches must preserve all attributes
original_func.some_attr  # Must still work after patching
original_func.__name__   # Must be preserved
```

### Stream Handling (Memory-Safe)
```python
# CORRECT: count, don't accumulate
accumulated_chars = 0
for chunk in stream:
    accumulated_chars += len(extract_text(chunk))
    yield chunk  # Pass through immediately
# Emit event in finally block

# WRONG: accumulates in memory
chunks = []
for chunk in stream:
    chunks.append(chunk)  # Memory leak!
    yield chunk
```

### File Locking (Cross-Platform)
```python
# Unix: fcntl.flock()
# Windows: msvcrt.locking()
# Always with timeout (100ms)
```

### Exception Hierarchy
```python
# All Vetch exceptions inherit from VetchError (which inherits ValueError)
class VetchError(ValueError): pass
class RegistryError(VetchError):
    def __init__(self, message: str, model: str | None = None): ...
class ProviderError(VetchError):
    def __init__(self, message: str, provider: str | None = None): ...
class ConfigurationError(VetchError):
    def __init__(self, message: str, field: str | None = None): ...

# Subclasses store contextual fields for debugging
```

### Token Estimation Fallback
When streaming lacks usage data, estimate tokens from character count:
```python
# ~4 characters per token (English text heuristic)
estimated_output_tokens = max(1, accumulated_chars // 4)
# Mark clearly: usage_estimated=True, usage_estimation_method="char_ratio"
```

---

## Testing Requirements

- 90%+ coverage enforced via pytest-cov
- Chaos tests: verify LLM calls work when Vetch fails
- Property tests with Hypothesis for calculation edge cases
- Multi-process cache tests
- Streaming memory tests (verify no accumulation)

### Before Pushing
```bash
pytest tests/ -v --cov=vetch --cov-fail-under=90
ruff check src/ tests/
mypy src/ --strict
```

---

## Commit Message Format

```
<type>(vetch): <description>

Types: feat, fix, docs, refactor, test, chore
Example: feat(vetch): add OpenAI provider wrapper
```

---

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `VETCH_REGION` | Grid region for carbon calculation | (inferred or unknown) |
| `VETCH_OUTPUT` | Output target: `stderr`, `none`, or file path | `stderr` |
| `VETCH_DEFAULT_PUE` | Power Usage Effectiveness multiplier | `1.1` |
| `VETCH_CACHE_MODE` | Set to `memory-only` for serverless | (file-based) |
| `ELECTRICITY_MAPS_API_KEY` | API key for live grid data | (optional) |

---

## signal_quality Values

| Value | Meaning |
|-------|---------|
| `live` | Grid data <5 min old |
| `delayed` | Grid data 5-30 min old |
| `blind` | API failed, using fallback |
| `unknown` | Region not determined |

---

## Event Schema Guarantees (v1.x)

- `schema_version: "1"` in every event
- Fields never removed
- Field names never changed
- Field types never changed
- New fields may be added

---

## Roadmap (Post-Alpha)

### OTel Bridge
Ship a `vetch-otel` bridge module that exports InferenceEvents as OpenTelemetry spans:
- `VetchOtelExporter` class implementing OTel SpanExporter
- Maps energy/carbon/cost to span attributes
- Integrates with existing OTel pipelines (Datadog, Honeycomb, etc.)

### Price Multiplier
Add `price_multiplier` parameter to `wrap()` for discount/premium pricing:
```python
with wrap(price_multiplier=0.8) as ctx:  # 20% discount
    response = client.chat.completions.create(...)
```
- Affects `estimated_cost_usd` calculation
- Document in `billing_tier` field (e.g., "list×0.8")
