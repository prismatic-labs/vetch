# Scope: `@prismatic-labs/vetch-ai-sdk` for Vercel AI SDK

**Status:** published on npm (`@prismatic-labs/vetch-ai-sdk`, currently **0.10.5**).
Design-partner / production pilot ready for AI SDK middleware. Python Vetch may
be ahead on calibration/confidence (e.g. **0.11.x**); see the parity matrix.

Honest read of what the promise covers and what it does not.

## What it is

First-party Vetch middleware for Vercel AI SDK 6.x. Wraps any AI SDK language
model with `wrapLanguageModel` / `withVetch`, records privacy-preserving usage
metadata, and emits one schema v2 Vetch event per model `generate` or consumed
`stream`. Fail-open and metadata-only.

## Production-ready today

- **AI SDK 6.x**: generate + stream, v3 usage/finish parsing. Multi-step tool
  loops emit one event per internal model step.
- **Schema v2 events**: energy, cost, carbon, water, cache fields, reasoning
  split, tier + uncertainty; `retry_count` and budget metadata fields.
- **Edge / serverless safe**: no SQLite; `createFetchEmitter` with timeout and
  retries; documented `waitUntil` pattern.
- **Privacy**: metadata only, no prompt or completion text.
- **Reliability**: fail-open by default; stream emits once on completion; a
  cancelled stream emits one partial event (`complete: false`).
- **Kill switch**: `VETCH_DISABLED` / `VETCH_ENABLED` and `disabled: true`.
- **Adapter differentiator**: `providerOptions.vetch` protocol hints drive
  advisories provider-only telemetry cannot see (PROTO-001, VOID-001, TOOL-SPIN,
  TOOL-TREADMILL, POSTDONE-DECODE-001, EXPECTED-LENGTH-001, STRUCT-REPAIR-001).
- **Gateway model IDs** (e.g. `openai/gpt-4.1-mini`) parsed into provider + model.
- **Local Ollama via OpenAI-compat**: `localhost:11434` / `OLLAMA_HOST` → `provider: ollama` (matches Python).
- **Rolling session advisories** when `attribution.sessionId` is set: STALL-001,
  CACHE-001, CACHE-002, ERROR-001, STREAM-001, REASONING-001.
- **Per-call advisories**: TRUNC-001, EMPTY-001, BABBLE-001, BUDGET-001 (when `budget_exceeded`), plus the protocol set above.
- **Budgets**: per-call thresholds via options, `providerOptions.vetch.budget`, or `VETCH_BUDGET_*` env vars; optional `onBudgetExceeded` callback.

## Explicitly out of scope

- **No intervention.** The middleware emits advisories only. It does NOT kill or
  reroute. Python's `set_stall_action("kill")` / `StallDetected` is not in the JS
  package.
- **Not full Python parity.** No JS session rollup for RAG-001, ZOMBIE, CTX; no
  rolling session budget accumulation; no `vetch audit` / savings + interventions
  reporting (Python only).
- **No Python 0.11+ calibration / confidence surfaces yet.** The CUDA Tier-0
  identity store, `calibration_match`, multimodal `energy_completeness`, and
  session confidence floors live in Python only. JS keeps **registry** parity
  for cloud Wh/$/gCO2e estimates; Node can still load legacy
  `~/.vetch/calibrations/` flats via `loadLocalCalibration` / `energyOverride`.
- **No live grid data.** Regional carbon uses bundled static intensities (no live
  Electricity Maps in TS).
- **No local calibrations on Edge.** Tier-0 hardware energy needs an explicit
  `energyOverride`.
- **No bundled collector.** Bring your own HTTPS endpoint / pipeline.

## Parity matrix (TS vs Python)

| Capability | Python `vetch` | `@prismatic-labs/vetch-ai-sdk` |
|---|---|---|
| Per-call energy / cost / carbon / water | Yes | Yes (bundled registries) |
| Live Electricity Maps grid | Yes | Fallback intensities only |
| Rolling STALL / CACHE / ERROR / STREAM / REASONING | Yes | Yes (`attribution.sessionId`) |
| RAG / ZOMBIE / CTX rollups | Yes | Not yet |
| App-protocol advisories (PROTO / VOID / TOOL-* / …) | No | Yes |
| Per-call budget flags + BUDGET-001 advisory | Partial (session alerts) | Per-call thresholds + advisory |
| Automatic stall kill / reroute | Yes (`set_stall_action`) | No (advisories only) |
| `vetch audit` / savings report | Yes | No |
| CUDA / Apple Tier-0 identity store + `calibration_match` | Yes (0.11+) | Not yet (registry + optional Node flat load) |
| Multimodal `energy_completeness` / confidence floors | Yes (0.11+) | Not yet |
| Edge runtime | N/A | Yes (`waitUntil` + emitter) |
| npm install | n/a (PyPI) | `@prismatic-labs/vetch-ai-sdk` |

## Learn more

- [QUICKSTART-VERCEL.md](../QUICKSTART-VERCEL.md)
- [NPM_PUBLISH.md](./NPM_PUBLISH.md)
- [Next.js example](../packages/vetch-ai-sdk/examples/nextjs-app-router/README.md)
