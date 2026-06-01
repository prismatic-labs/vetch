# `@prismatic-labs/vetch-ai-sdk` examples

| Example | What it shows |
|---------|----------------|
| [`basic.ts`](./basic.ts) | Minimal `withVetch` + AI SDK Gateway + `generateText` |
| [`protocol-progress.ts`](./protocol-progress.ts) | `providerOptions.vetch.protocol` and `onAdvisory` |
| [`nextjs-app-router/`](./nextjs-app-router/README.md) | Next.js App Router API route, `sessionId`, `waitUntil`, collector env |

## Run a TypeScript script (monorepo)

From `packages/vetch-ai-sdk`:

```bash
npm ci && npm run build
export AI_GATEWAY_API_KEY=...   # for basic.ts / protocol-progress.ts if using gateway
npx tsx examples/basic.ts
```

Scripts import from `../src/index.js` for development inside the repo. Published apps use `import { withVetch } from "@prismatic-labs/vetch-ai-sdk"`.

## Run the Next.js reference app

See [nextjs-app-router/README.md](./nextjs-app-router/README.md). Uses `file:../..` to link the local package before npm publish.

## Docs

- [QUICKSTART-VERCEL.md](../../../QUICKSTART-VERCEL.md)
- [SCOPE-v0.8.0-vercel.md](../../../docs/SCOPE-v0.8.0-vercel.md)
- [NPM_PUBLISH.md](../../../docs/NPM_PUBLISH.md)
