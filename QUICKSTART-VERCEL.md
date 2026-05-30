# Vetch + Vercel AI SDK — Quickstart

Observe Vercel AI SDK 6.x calls with **Vetch schema v2** events: token usage, energy/carbon/cost estimates, cache savings, and workflow advisories — **without** storing prompts or completions.

**Requirements:** Node 22+, `ai` ^6.x, `@ai-sdk/provider` ^3.x.

---

## Install

`@vetch/ai-sdk` is **first-party** in the Vetch monorepo. It is **`private: true`** until npm publish — use a path install today:

```bash
git clone https://github.com/prismatic-labs/vetch.git
cd vetch/packages/vetch-ai-sdk
npm ci && npm run build
```

In your app (from the app root):

```bash
npm install /absolute/path/to/vetch/packages/vetch-ai-sdk
```

When the package is published:

```bash
npm install @vetch/ai-sdk
```

---

## 60-second setup

Wrap your language model once. Every `generateText` / `streamText` call through that model is tracked.

```ts
import { generateText } from "ai";
import { openai } from "@ai-sdk/openai";
import { consoleJsonEmitter, withVetch } from "@vetch/ai-sdk";

const model = withVetch(openai("gpt-4.1-mini"), {
  region: "US-CA",
  tags: { service: "my-app" },
  emitter: consoleJsonEmitter, // swap for your collector in production
});

const { text } = await generateText({
  model,
  prompt: "Say hello in one sentence.",
});

console.log(text);
```

You should see one JSON line per **internal model step** (a multi-step tool loop produces one event per model call).

**Fail-open by default** — emitter failures do not break the LLM response. Use `debug: true` or `onEmitterError` while developing.

---

## Send events to your collector

```ts
import { createFetchEmitter, withVetch } from "@vetch/ai-sdk";

const model = withVetch(openai("gpt-4.1-mini"), {
  emitter: createFetchEmitter({
    endpoint: "https://your-collector.example/vetch/events",
    bearerToken: process.env.VETCH_COLLECTOR_TOKEN,
    timeoutMs: 2000,
    retries: 1,
  }),
});
```

Or push events yourself:

```ts
const model = withVetch(openai("gpt-4.1-mini"), {
  emitter: async (event) => {
    await yourPipeline.ingest(event);
  },
});
```

There is **no** built-in SQLite path in the JS package — use an emitter or log to stdout during development.

---

## Tags and session attribution

Set stable metadata on the middleware, or per request via `providerOptions`:

```ts
await generateText({
  model,
  prompt: "Summarize this ticket.",
  providerOptions: {
    vetch: {
      tags: { route: "support", env: "prod" },
      attribution: {
        sessionId: "req-9f3a2c", // scopes rolling advisories to this request
        traceId: "trace-abc",
      },
    },
  },
});
```

Use **non-sensitive** tag values (or hashed IDs). Vetch never stores prompt or completion text.

---

## App protocol advisories (the Vercel-specific value)

Provider responses alone cannot see whether your **app** expected a tool call, JSON repair, or terminal state. Pass optional protocol hints:

```ts
await generateText({
  model,
  prompt: runAgentStep(),
  providerOptions: {
    vetch: {
      protocol: {
        expectedToolUse: true,
        stepCount: 3,
        invalidToolCallCount: 0,
        toolFailureCount: 0,
        acceptedFinalResult: false,
      },
    },
  },
});
```

After each call, check `event.advisories` (or `onAdvisory`) for codes such as:

| Code | Meaning |
|------|---------|
| `PROTO-001` | Expected tool progress, but recent calls spent tokens without valid tool calls |
| `VOID-001` | App reported repeated invalid tool calls |
| `TOOL-SPIN-001` | Many tool calls in one response |
| `TRUNC-001` | Stopped at output token limit |
| `EMPTY-001` | Output tokens but no visible text |
| `POSTDONE-DECODE-001` | Calls continued after terminal progress |

Full list: [`packages/vetch-ai-sdk/README.md`](packages/vetch-ai-sdk/README.md#advisories).

---

## Streaming

`streamText` is supported. One event is emitted when the stream is **fully consumed** (flush), with usage from the v6 `finish` part. If the consumer **cancels** early, a partial event is emitted once with `complete: false`.

```ts
import { streamText } from "ai";

const result = streamText({ model, prompt: "Hello" });

for await (const chunk of result.textStream) {
  process.stdout.write(chunk);
}
// Event emits after the stream completes (or on cancel).
```

For deterministic delivery in tests:

```ts
withVetch(model, { emitter: myEmitter, emissionMode: "await" });
```

---

## Next.js / Vercel (Edge and serverless)

On Edge, do not rely on reading `~/.vetch/calibrations/` (filesystem). Pass `energyOverride` explicitly if you use Tier-0 hardware coefficients.

Extend the platform `waitUntil` so background emission finishes after the response is sent:

```ts
import { waitUntil } from "@vercel/functions";
import { generateText } from "ai";
import { withVetch } from "@vetch/ai-sdk";

const model = withVetch(openai("gpt-4.1-mini"), {
  emitter: createFetchEmitter({ endpoint: process.env.VETCH_EVENTS_URL! }),
  waitUntil: (promise) => waitUntil(promise),
});

export async function POST(req: Request) {
  const { prompt } = await req.json();
  const { text } = await generateText({ model, prompt });
  return Response.json({ text });
}
```

---

## Node.js: Tier-0 local calibration (optional)

If you run Apple Silicon (or other) calibration with Python Vetch and save files under `~/.vetch/calibrations/`, the middleware **auto-loads** matching `provider` + `model` on **Node** only:

```bash
# Python side (separate install)
pip install 'vetch[apple-silicon]'
sudo vetch calibrate-apple-silicon --model moondream:latest --provider ollama
```

Use the same **provider label** in TS as in the calibration file (`ollama`, not `openai`, unless you calibrated that way). Model aliases like `moondream` vs `moondream:latest` are resolved automatically.

On **Edge**, pass coefficients manually:

```ts
withVetch(model, {
  energyOverride: {
    wh_per_1k_input: 0.00031,
    wh_per_1k_output: 0.00185,
    wh_per_image: 0.00054,
    visual_tokens_per_image: 729,
    tier: 0,
    source: "local_calibration",
  },
});
```

---

## Python Vetch vs this package

| Capability | Python `vetch.instrument()` | `@vetch/ai-sdk` |
|------------|---------------------------|-----------------|
| Energy / cost / carbon per call | Yes | Yes (bundled registries) |
| Live Electricity Maps grid | Yes | Fallback intensities only |
| Session advisories (STALL, CACHE, RAG, …) | Yes (`vetch` Session) | Per-call + protocol advisories; **no** rolling STALL/CACHE session engine yet |
| Automatic stall kill/reroute | Yes (`set_stall_action`) | No — advisories only |
| Edge runtime | N/A | Yes (emitter + `waitUntil`) |

Use **both** if you have a Python collector and a Next.js front-end: emit the same schema v2 JSON from each path.

---

## Verify your install

From the Vetch repo root:

```bash
npm --prefix packages/vetch-ai-sdk run check
npm --prefix packages/vetch-ai-sdk test
python scripts/sync_ai_sdk_registries.py
```

---

## Learn more

- **API and limits:** [`packages/vetch-ai-sdk/README.md`](packages/vetch-ai-sdk/README.md)
- **Architecture notes:** [`docs/vercel-ai-sdk-poc.md`](docs/vercel-ai-sdk-poc.md)
- **Waste taxonomy (Python session codes):** [`docs/inference-waste-taxonomy.md`](docs/inference-waste-taxonomy.md)
- **Python quickstart:** [`QUICKSTART.md`](QUICKSTART.md)
