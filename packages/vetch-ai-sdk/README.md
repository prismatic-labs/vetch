# @vetch/ai-sdk

First-party Vetch middleware for observing Vercel AI SDK 6.x calls with Vetch schema v2 events, local energy/carbon/water/cost estimates, usage/cache/reasoning telemetry, app-protocol advisories, and Edge-safe emission.

**Quickstart (install, Next.js, Edge):** [`QUICKSTART-VERCEL.md`](../../QUICKSTART-VERCEL.md) in the repo root.

The adapter wraps any AI SDK language model with `wrapLanguageModel`, records privacy-preserving usage metadata, and emits one Vetch event per model `generate` or consumed model `stream` call. Multi-step `generateText` tool loops produce one event per internal model step.

This package lives in the Vetch repo and follows the Vetch release train, but it is distributed as an npm package because Vercel AI SDK applications run in the JavaScript ecosystem and often in Edge/serverless runtimes.

```ts
import { gateway, generateText } from "ai";
import { createFetchEmitter, withVetch } from "@vetch/ai-sdk";

const model = withVetch(gateway("openai/gpt-4.1-mini"), {
  emitter: createFetchEmitter({ endpoint: "https://example.com/vetch/events" }),
});

await generateText({
  model,
  prompt: "Say hello in one sentence.",
});
```

## What This Provides

- **Vetch schema v2 events** compatible with Python Vetch tooling (field names and nested usage)
- **Usage, cache, reasoning, and tool telemetry** from AI SDK v6 `LanguageModelV3` responses and stream `finish` parts
- **Local estimates** for energy, carbon, water, cost, cache savings, uncertainty bounds, tiers, and tracking quality using bundled Vetch registries
- **App protocol advisories** (truncation, tool spin, expected-tool voids, and related workflow signals)
- **Edge-safe emission** — no filesystem storage; fail-open by default; optional `waitUntil` for serverless

## Install (v0.8.0 monorepo)

The package is `private: true` until npm publish. From the Vetch repo:

```bash
cd packages/vetch-ai-sdk && npm ci && npm run build
npm install /absolute/path/to/vetch/packages/vetch-ai-sdk
```

`enrichVetchEvent` auto-loads Tier-0 coefficients from `~/.vetch/calibrations/` on Node.js (not Edge). Pass `energyOverride` explicitly on Edge or when you need custom coefficients.

## Current Limits

- The bundled registries are snapshots copied from Python Vetch; run `python scripts/sync_ai_sdk_registries.py` from the repo root after registry edits (CI enforces parity).
- Live grid lookups are not performed in this package. Region-aware fallback intensities from `global_averages.json` are used, with `signal_quality: "blind"` when a region is supplied and `"unknown"` without one.
- There is no local SQLite storage path; send events to a collector with `createFetchEmitter` or a custom emitter.

## Why this belongs in the AI SDK layer

Provider wrappers can see provider responses, but they usually cannot see the application protocol. AI SDK middleware can also observe request-level metadata from `providerOptions`, which means an app can tell Vetch when a workflow expected tool progress, retrieval progress, or a terminal tool call.

That closes one of the current Kudzu gaps: calls can spend plenty of output tokens while failing to make protocol progress, yet look normal from a raw provider response alone.

## API

```ts
const model = withVetch(model, {
  tags: { app: "docs-agent" },
  attribution: { sessionId: "request-123" },
  region: "US-CA",
  protocol: { expectedToolUse: true },
  emitter: async (event) => sendToYourCollector(event),
  failOpen: true,
  onAdvisory: (advisories, event) => console.warn(advisories, event.event_id),
});
```

Per-request metadata can be passed through AI SDK `providerOptions`:

```ts
await generateText({
  model,
  prompt,
  providerOptions: {
    vetch: {
      tags: { route: "retrieval" },
      attribution: {
        sessionId: "request-123",
        traceId: "trace-abc",
      },
      protocol: {
        expectedToolUse: true,
        stepCount: 3,
      },
    },
  },
});
```

## Advisories

- `TRUNC-001`: generation ended at the token limit.
- `EMPTY-001`: provider reported output tokens but app-visible output was empty.
- `BABBLE-001`: high decode volume.
- `TOOL-SPIN-001`: many tool calls in a single response.
- `PROTO-001`: app expected tool progress, but recent calls spent output tokens without valid tool calls.
- `VOID-001`: app reported repeated invalid tool calls.
- `TOOL-TREADMILL-001`: app reported repeated tool failures.
- `STRUCT-REPAIR-001`: app reported repeated schema or JSON repair attempts.
- `POSTDONE-DECODE-001`: app reported that calls continued after terminal progress.
- `EXPECTED-LENGTH-001`: a short, JSON, or tool-oriented step produced a long decode.

Rolling advisories are scoped by `attribution.sessionId` when supplied. Shared middleware instances can therefore be reused across requests without cross-request `PROTO-001` bleed. Calls without a session ID use an ephemeral advisory session.

## Streaming behavior

- Events emit **once** when a stream is consumed to completion (`TransformStream.flush`), including full usage from the v6 `finish` part (cache, reasoning, and text totals).
- If the consumer **cancels** the stream (`reader.cancel()`), a **partial** event emits once via the transform `cancel` hook with `complete: false` and a `stream_cancelled_partial` warning.
- Visible output counts **text deltas only**; reasoning and tool payloads are not counted as visible assistant output.

## Privacy and reliability

- No SQLite or local filesystem storage, so the adapter can run in Edge runtimes.
- No prompt, response text, tool arguments, or plaintext provider error messages are stored. It counts visible characters and tool events only.
- Treat tags and attribution as metadata: pass non-sensitive values, or hash user/session identifiers before sending them.
- Middleware telemetry fails open silently by default, so Vetch should not break or clutter a successful model call if an emitter fails. Use `debug: true` or `onEmitterError` while developing.
- Emission runs in the background by default with a timeout. Use `emissionMode: "await"` when tests or local scripts need deterministic delivery, or pass `waitUntil` in Edge/serverless runtimes.
