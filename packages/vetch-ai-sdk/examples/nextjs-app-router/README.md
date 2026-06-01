# Next.js App Router + `@prismatic-labs/vetch-ai-sdk`

Minimal reference for Vercel: Node and Edge API routes, rolling advisories via `sessionId`, and background emission with `waitUntil`.

## Setup

```bash
cd packages/vetch-ai-sdk/examples/nextjs-app-router
npm install
cp .env.example .env.local
# Edit .env.local: OPENAI_API_KEY required; VETCH_EVENTS_URL optional
npm run dev
```

## Try it

```bash
# Node runtime
curl -s -X POST http://localhost:3000/api/chat \
  -H 'content-type: application/json' \
  -d '{"prompt":"Say hello in one sentence.","sessionId":"demo-session-1"}' | jq

# Edge runtime (same Vetch wiring, runtime = "edge")
curl -s -X POST http://localhost:3000/api/chat-edge \
  -H 'content-type: application/json' \
  -d '{"prompt":"Say hello in one sentence.","sessionId":"demo-session-1"}' | jq
```

Without `VETCH_EVENTS_URL`, Vetch events print as JSON lines in the Next.js server log (`consoleJsonEmitter`).

With `VETCH_EVENTS_URL` set, events POST to your collector; `waitUntil` from `@vercel/functions` keeps emission from blocking the HTTP response on Vercel.

## Files

| File | Role |
|------|------|
| `src/vetch-model.ts` | Shared `withVetch` model: emitter + `waitUntil` |
| `app/api/chat/route.ts` | `POST` handler on the Node runtime, with `attribution.sessionId` |
| `app/api/chat-edge/route.ts` | Same handler on the Edge runtime (`runtime = "edge"`) |

## Production notes

- Set **`attribution.sessionId`** on every request (shown in `app/api/chat/route.ts`). Without it, rolling advisories (STALL, CACHE, ERROR, …) are disabled to avoid cross-tenant false positives.
- This demo reuses one module-level `vetchModel`; that is fine when each HTTP request passes its own `sessionId`.
- On Edge, pass **`energyOverride`** if you use Tier-0 hardware coefficients (no `~/.vetch` on Edge).
- The middleware **does not** kill stalled loops; handle `STALL-001` in your app or use Python Vetch for `set_stall_action("kill")`.
