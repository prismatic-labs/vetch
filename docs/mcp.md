# MCP server

Vetch ships a [Model Context Protocol](https://modelcontextprotocol.io/) server that gives an AI agent live access to the same energy, cost, and carbon data the SDK produces. An agent can check its remaining budget, compare two models before choosing one, or look up the greenest region to run a workload, mid-conversation, without you wiring any of it by hand.

The server is read-and-estimate only. It reports numbers and reads session state; it does not intercept the agent's own LLM calls. To track an agent's calls, instrument the process it runs in (see the main README).

## Install and configure

```bash
pip install vetch[mcp]
```

Add the server to your MCP client. For Claude Desktop, in `claude_desktop_config.json`:

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

`VETCH_REGION` sets the default grid region for any tool that computes carbon without an explicit `region` argument. Any other Vetch environment variable works here too.

## Tools

| Tool | Arguments | Returns |
|------|-----------|---------|
| `vetch_estimate` | `model`, `input_tokens`, `output_tokens`, `region?` | Energy, carbon, water, and cost for one call, with tier, uncertainty, grid `signal_quality`, and a cost breakdown |
| `vetch_compare` | `models[]`, `input_tokens`, `output_tokens`, `region?`, `sort_by?` | The same estimate per model, sorted, with `is_cheapest` and `is_greenest` flagged |
| `vetch_session_stats` | (none) | Accumulated session energy, carbon, water, cost, models used, and any active waste advisories |
| `vetch_status` | (none) | Version, health, and budget status |
| `vetch_check_budget` | (none) | Per-budget remaining amount, or a note if none is configured |
| `vetch_grid_intensity` | `region` | Live carbon intensity for one region, with `signal_quality` |
| `vetch_cleanest_region` | `regions[]` | The lowest-carbon region from the candidate list, with its intensity |
| `vetch_registry_lookup` | `model` | Raw energy and pricing rows for a model, or an error if unknown |

Notes on behavior:

- `vetch_estimate` maps the energy tier to a `confidence` label (`high` / `medium` / `low`) so an agent can act on trust level without parsing tier numbers. It also returns water in both liters and milliliters, since per-call values are more readable in milliliters.
- `vetch_compare` calls `vetch_estimate` per model, drops any that error into a separate `errors` list, and sorts the rest by `sort_by` (default `cost_usd`). Cheapest and greenest are computed independently, so they can point at different models.
- `vetch_cleanest_region` and `vetch_grid_intensity` reflect live grid data only when `ELECTRICITY_MAPS_API_KEY` is set; otherwise they use annual-average intensity. See [region-config.md](region-config.md).
- Every tool is wrapped to fail safe: an internal error returns an `error` field rather than crashing the agent's turn.

## Resources

| URI | Returns |
|-----|---------|
| `vetch://registry/models` | Every model name in the registry |
| `vetch://registry/energy/{model}` | Energy coefficients for one model |
| `vetch://registry/pricing/{model}` | Pricing row for one model |
| `vetch://config` | Current configuration: region, output target, PUE, cache mode |
| `vetch://version` | Vetch version string |

## Example agent flow

A cost-aware agent deciding which model to run a batch job on:

1. Read `vetch://config` to learn the configured region.
2. Call `vetch_compare` with the candidate models and the expected token counts.
3. Pick the cheapest or greenest result, depending on the job's priority.
4. Call `vetch_check_budget` before committing, and skip or downshift if the run would blow the remaining budget.

Everything the agent needs to reason about cost and carbon is in the tool responses, so the decision happens in one turn.

## Related

- [region-config.md](region-config.md): what `region` and `signal_quality` mean
- [energy-methodology.md](energy-methodology.md): how to read the tier and confidence in `vetch_estimate`
- [model-resolution.md](model-resolution.md): why an unknown model may return low confidence
