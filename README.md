# vetch

Energy and cost observability for LLM inference.

> ⚠️ **Alpha** — Shipping week of 2026-02-17. Star/watch for updates.

## What it does

Wraps OpenAI and Anthropic SDK calls to log energy consumption, cost, and carbon emissions per inference. Zero access to prompt content. Under 5ms overhead.

- **Energy**: Estimated Wh per request (model-specific)
- **Cost**: Token-based cost attribution with tags
- **Carbon**: Live grid intensity from Electricity Maps
- **Privacy**: Only touches model name, token counts, region, latency

## Install
```bash
pip install vetch
```

## Quick start
```python
from vetch import wrap
from openai import OpenAI

client = OpenAI()

with wrap(region="us-east-1") as ctx:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}]
    )

print(f"Energy: {ctx.event['estimated_energy_wh']} Wh")
print(f"Cost: ${ctx.event['estimated_cost_usd']}")
print(f"Carbon: {ctx.event['estimated_carbon_g']}g CO₂")
```

## Output

Each inference emits a structured JSON event to stderr:
```json
{
  "model": "gpt-4o",
  "estimated_energy_wh": 0.42,
  "estimated_cost_usd": 0.003,
  "estimated_carbon_g": 0.15,
  "signal_quality": "live",
  "usage": {"prompt_tokens": 12, "completion_tokens": 48}
}
```

## Documentation

- [Methodology](METHODOLOGY.md) — How we calculate energy and carbon
- [Configuration](docs/configuration.md) — Environment variables and options

## Design principles

- **Fail-open**: If Vetch fails, your LLM call still works
- **Fail-loud**: Every log includes `signal_quality` — no silent degradation
- **Privacy-first**: Zero access to prompts or completions
- **Observable**: Works alongside Datadog, OpenTelemetry, Sentry

## License

Apache 2.0 — see [LICENSE](LICENSE)

---

Built by [Prismatic Labs](https://prismaticlabs.ai)
