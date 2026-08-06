# Model resolution

Vetch prices and estimates energy for a model by looking it up in a registry. Model names in the wild are messy: dated snapshots (`gpt-4-0613`), `-latest` and `-preview` suffixes, experimental variants, and brand-new releases the bundled registry has never seen. Resolution is the process that turns whatever string the SDK reported into a registry entry, and records how confident that match was.

Every event carries a `model_match` field with the rung that matched. Read it before you trust the number: an `exact` match is the model's own data, while a `family` match is a proxy that borrows a similar model's coefficients.

## The resolution ladder

`resolve_model_match()` walks these rungs in order and stops at the first hit. Matching is case-insensitive; registry keys are stored lowercase.

| Rung | `model_match` | What it means | Confidence |
|------|---------------|---------------|------------|
| 1 | `exact` | The model name is a registry key | Full: the entry's own tier |
| 2 | `alias` | A curated equivalence resolved to a canonical key | Full: the canonical entry's tier |
| 3 | `prefix` | An algorithmic shorten matched a shorter key | Low: **downgraded to Tier 3** |
| 4 | `family` | No name match, proxied to a same-family representative | Low: **Tier 3** |
| 5 | `fallback` | Fully unknown; generic parameter-based estimate | Lowest |

### 1. Exact

The lowercased model name is a key in the registry. Full confidence: you get that entry's real energy tier and verified price.

### 2. Alias

A curated equivalence table maps known variants to a canonical key. Dated snapshots, `-latest`, and `-preview` forms live here, so `claude-3-5-sonnet-latest` resolves to the same entry as the pinned version. Aliases are hand-maintained, so a match here is as trustworthy as an exact one.

### 3. Prefix

If no exact or alias match exists, Vetch strips trailing `-`-delimited segments and retries, longest prefix first. `claude-sonnet-4-6-experimental` shortens to `claude-sonnet-4-6`; `gpt-4-0613` shortens to `gpt-4`. This catches new dated forms of known models automatically.

A prefix match is a proxy, not a confirmation. The energy tier is **downgraded to Tier 3** even if the target row has Tier 1 data, because the shorten is a guess that the newer variant behaves like the older one. That is often true and sometimes not, and Vetch flags it as low-confidence rather than pretending otherwise.

### 4. Family

If prefix shortening finds nothing, Vetch infers the provider family from the name and proxies to a conservative representative of that family. The sub-tier is chosen deliberately:

- Names containing `flash`, `lite`, `nano`, `mini`, `haiku`, `small`, or `tiny`, or a parameter size at or below 15B, map to the **small** representative.
- Names containing `pro`, `ultra`, `opus`, `max`, or `large`, or a parameter size at or above 30B, map to the **large** representative.
- Anything ambiguous biases to **large**.

The bias to the larger sibling is intentional: an unknown model should never silently *undercount* energy or cost. A family match is Tier 3, and it only succeeds if the representative row is actually present in the loaded registry. If the registry failed to load, Vetch declines to claim a proxy is known and falls through to `fallback`.

> The exact hint sets and size thresholds live in `_FAMILY_LARGE_HINTS`, `_FAMILY_SMALL_HINTS`, and the classification logic in [../src/vetch/calculation.py](../src/vetch/calculation.py) (`_classify_family_subtier`). The values above are current at the time of writing; that module is the source of truth if they ever diverge.

### 5. Fallback

Nothing matched. Vetch produces a generic conservative estimate from parameter count and marks the model as unknown. Directional only.

## When a current model isn't recognized

If a new model resolves to `prefix`, `family`, or `fallback` and you want a real figure, you have three options:

**Add a registry row.** Add the model to `energy.json`, add a verified price to `pricing.json`, and add any dated forms to `aliases.json`. See [../src/vetch/registry/PROVENANCE.md](../src/vetch/registry/PROVENANCE.md) for the format and sourcing rules. Existing installs pick up new rows through the remote registry without upgrading the package.

> Verify every price against the official provider pricing page with a live check before writing it. Training-data prices go stale, and model pricing changes often. If a confirmed figure is genuinely unavailable, add the row with a `basis` of `"Estimated — verify against official pricing page"` and flag it.

**Calibrate.** For self-hosted models, `vetch calibrate --model <name> --provider <provider>` produces a hardware-measured Tier 0 figure. See [energy-methodology.md](energy-methodology.md).

**Override for a single call.** Pass `energy_override` to `vetch.wrap()` when you just need one call estimated correctly without touching the registry.

## Inspecting a match

To see how a specific name resolves and what data backs it:

```bash
vetch registry lookup <model>      # energy + pricing rows for a model
vetch methodology                  # per-model tier and provenance for the whole registry
```

The MCP server exposes the same lookup as `vetch_registry_lookup` (see [mcp.md](mcp.md)).

## Related

- [energy-methodology.md](energy-methodology.md): what the tiers mean and why proxies are Tier 3
- [../src/vetch/registry/PROVENANCE.md](../src/vetch/registry/PROVENANCE.md): registry format and sourcing rules
