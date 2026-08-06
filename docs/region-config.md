# Region configuration

The `region` you pass to Vetch selects the electricity grid used for carbon intensity. Energy is a property of the model and the tokens; carbon is energy multiplied by how dirty the grid was at that moment. Get the region wrong and the energy figure stays fine while the carbon figure drifts by a large factor, because grid intensity ranges from roughly 30 gCO2e/kWh (hydro, nuclear) to over 700 (coal-heavy).

Region only affects carbon and water. Cost and energy are region-independent.

## What a region string should be

Use an [Electricity Maps zone identifier](https://app.electricitymaps.com/map). These line up with the region names cloud providers already use, so in most deployments you can pass the value you already have:

```python
vetch.instrument(region="us-east-1")   # AWS-style
vetch.instrument(region="eastus")      # Azure-style
vetch.instrument(region="europe-west4") # GCP-style
```

Matching is best-effort. If a zone is not recognized, Vetch falls back to a default intensity and marks the event so you can see the estimate was blind rather than measured (see `signal_quality` below).

## Which providers expose a real region

Some providers pin inference to a datacenter you control. There the region you configure is the region the call actually ran in.

| Provider | How region is set | Example |
|----------|-------------------|---------|
| Azure OpenAI | Embedded in the endpoint URL | `eastus`, `westeurope` |
| Vertex AI (Google) | `vertexai.init(location=...)` | `us-central1`, `europe-west4` |
| AWS Bedrock | Standard AWS region parameter | `us-east-1`, `eu-west-1` |

For these, set `region` to match the datacenter and the carbon figure is as accurate as the grid data behind it.

**OpenAI and Anthropic do not expose per-call location.** Inference is routed across global infrastructure and the physical location of any one call is not visible to the client. Pass your best estimate based on where your account and users sit:

```python
vetch.instrument(region="us-east-1")  # reasonable default for US traffic
vetch.instrument(region="eu-west-1")  # reasonable default for EU traffic
```

The carbon number for these providers is a planning input, not a measurement of where the electrons came from. Treat it accordingly.

## Fallback order

If you never pass `region`, Vetch resolves one in this order and stops at the first hit:

1. **`region=` argument** to `instrument()` or `wrap()`.
2. **`VETCH_REGION`** environment variable.
3. **Cloud provider environment variables**, checked in order: `AWS_REGION`, `AWS_DEFAULT_REGION`, `GOOGLE_CLOUD_REGION`, `AZURE_REGION`.
4. **Timezone heuristic:** a coarse guess from the host's UTC offset.

The timezone heuristic is a last resort with roughly 30% accuracy, and it maps only a handful of offsets:

| UTC offset | Inferred region |
|------------|-----------------|
| 0 | `eu-west-2` (London) |
| +1 to +3 | `eu-central-1` (Europe) |
| −4 to −5 | `us-east-1` (US East) |
| −7 to −8 | `us-west-2` (US West) |
| +8 to +9 | `asia-northeast-1` (Tokyo/Seoul) |

Any offset outside those bands yields no region at all, and Vetch falls back to a default grid intensity. When the heuristic does fire, the event carries a warning telling you it guessed. Do not ship a production service on the heuristic.

These bands are a deliberately coarse last resort; the mapping lives in `_infer_region` in [../src/vetch/wrapper.py](../src/vetch/wrapper.py) if you need the exact current values.

**Set `region` explicitly, or set `VETCH_REGION`, in every environment you care about.** It is one line and it removes the single largest source of error in the carbon figure.

## Live grid data

By default Vetch uses annual-average intensity for each zone. If you set `ELECTRICITY_MAPS_API_KEY`, it fetches live carbon intensity so the same model run at 2am on wind power reports lower carbon than at 6pm on gas peakers.

Every event carries a `signal_quality` field so you can tell how fresh the grid data was:

| Value | Meaning |
|-------|---------|
| `live` | Grid data less than 5 minutes old |
| `delayed` | Grid data 5–30 minutes old |
| `blind` | Grid data older than 30 minutes, or the grid API failed and Vetch fell back to average intensity |
| `unknown` | Region could not be determined, or data age is unknown |

`blind` and `unknown` are not errors. Vetch is fail-open, so a grid outage never blocks your LLM call. They signal that the carbon number on those events is coarser than usual.

## Finding the cleanest region

If you have latitude to choose where a workload runs, the MCP server and CLI can rank candidate regions by current intensity. See [mcp.md](mcp.md) (`vetch_cleanest_region`) and `vetch compare`.

## Related

- [energy-methodology.md](energy-methodology.md): how the energy half of the calculation works
- [OPENTELEMETRY.md](OPENTELEMETRY.md): exporting carbon and `signal_quality` to your observability stack
