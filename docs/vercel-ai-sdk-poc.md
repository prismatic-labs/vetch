# Vercel AI SDK adapter for Vetch

**User-facing quickstart:** [`QUICKSTART-VERCEL.md`](../QUICKSTART-VERCEL.md)

The TypeScript package is `packages/vetch-ai-sdk` (`@prismatic-labs/vetch-ai-sdk`, version **0.8.1**). It attaches `LanguageModelV3Middleware` via `wrapLanguageModel` (`wrapGenerate` / `wrapStream`).

## What it provides

1. **Vetch schema v2 events** with local energy, carbon, water, and cost estimates (bundled registries synced from Python)
2. **Usage, cache, reasoning, and tool telemetry** from AI SDK v6 generate results and stream `finish` parts
3. **Reasoning token split** — `usage.text.output_tokens` excludes reasoning; reasoning is in `usage.reasoning`
4. **App protocol metadata** via `providerOptions.vetch` without storing prompt or response text
5. **Session-scoped advisories** keyed by `attribution.sessionId`
6. **Python-aligned kill switch and schema fields** (`VETCH_DISABLED`, `VETCH_ENABLED`, budgets, retry metadata)
7. **Edge-safe emission** — fail-open by default, optional `waitUntil`, no default filesystem access

## Verification

```bash
npm --prefix packages/vetch-ai-sdk run check
npm --prefix packages/vetch-ai-sdk test
python scripts/sync_ai_sdk_registries.py
```

## Release hygiene

- Published on npm as `@prismatic-labs/vetch-ai-sdk` (`private: false`)
- Bump version + run `npm publish` for patch releases; tag `v*` triggers CI when versions match
- Run registry sync when Python `src/vetch/registry/*.json` changes
- Optional: OTLP exporter and collector enrich mode

**API reference:** [`packages/vetch-ai-sdk/README.md`](../packages/vetch-ai-sdk/README.md)
