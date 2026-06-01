# Publish `@prismatic-labs/vetch-ai-sdk` - operator checklist

`@prismatic-labs/vetch-ai-sdk` is on npm (the `@vetch` scope was unavailable). Use this checklist for **patch releases** (e.g. `0.8.1`).

## 1. npm organization and package name

1. Log in at [https://www.npmjs.com](https://www.npmjs.com).
2. Confirm you own the **`prismatic-labs`** npm org (scope `@prismatic-labs`), matching the GitHub org.
3. Check the name is free:

   ```bash
   npm view @prismatic-labs/vetch-ai-sdk version 2>/dev/null || echo "Name available (or not published yet)"
   ```

4. The `vetch` org was unavailable on npm, so the package ships under the `@prismatic-labs` scope. If you ever change scope or name, update `name` in `packages/vetch-ai-sdk/package.json` and the docs.

## 2. Local dry run (recommended)

From the repo:

```bash
cd packages/vetch-ai-sdk
npm ci
npm run build
npm test
npm pack --dry-run
```

Inspect the tarball list: `dist/`, `README.md`, `LICENSE`, `NOTICE` only (no source leaks).

Optional install test in a temp app:

```bash
npm pack
npm install /path/to/vetch-ai-sdk-0.8.0.tgz
```

## 3. First manual publish (one time)

You need an npm access token with **publish** rights for the `@prismatic-labs` scope.

```bash
cd packages/vetch-ai-sdk
npm login              # account must belong to the prismatic-labs org with publish rights
npm ci && npm run build
npm publish --access public --provenance=false
```

- Do **not** add inline shell comments after `npm publish`; zsh may pass `# ...` as npm arguments.
- Do **not** pass `--provenance` for a local publish: npm only generates provenance from a supported CI (GitHub Actions). The `npm-release` job adds it automatically for tagged releases.
- `--access public` is required the first time you publish a scoped package.
- If your npm account enforces 2FA for publishing, add `--otp=123456`.
- Version in `package.json` must match the git tag you intend (`0.8.0` to `v0.8.0`).

After publish, verify:

```bash
npm view @prismatic-labs/vetch-ai-sdk
```

## 4. GitHub Actions (automated releases)

CI includes an **`npm-release`** job on tags `v*`. It publishes only when the tag version matches `packages/vetch-ai-sdk/package.json`. It uses **Trusted Publishing (OIDC)**, so **no `NPM_TOKEN` secret is required** and provenance is automatic.

One-time setup on npm (after the package exists):

1. npmjs.com -> the `@prismatic-labs/vetch-ai-sdk` package -> **Settings** -> **Trusted Publisher** -> **GitHub Actions**.
2. Repository: `prismatic-labs/vetch`. Workflow filename: `ci.yml`. Leave environment blank unless you add one.
3. Save. From then on, the `npm-release` job authenticates via OIDC, with no stored token.

Tag flow:

```bash
# Ensure package.json version is bumped, committed, and pushed
git tag v0.8.1   # use the next version for new releases
git push origin v0.8.1
```

PyPI and npm can share the same `v0.8.0` tag; npm job skips if JS version ≠ tag.

## 5. Vercel contact handoff

Send:

1. **Install:** `npm install @prismatic-labs/vetch-ai-sdk ai @ai-sdk/openai` (peers as in package README).
2. **Quickstart:** [QUICKSTART-VERCEL.md](../QUICKSTART-VERCEL.md)
3. **Scope (honest):** [SCOPE-v0.8.0-vercel.md](./SCOPE-v0.8.0-vercel.md) - observe + advise, not stall kill in JS.
4. **Reference app:** [packages/vetch-ai-sdk/examples/nextjs-app-router](../packages/vetch-ai-sdk/examples/nextjs-app-router/README.md)

Env vars for production:

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Model calls (example uses OpenAI; swap provider as needed) |
| `VETCH_EVENTS_URL` | HTTPS collector for `createFetchEmitter` |
| `VETCH_COLLECTOR_TOKEN` | Optional bearer token for collector |

## 6. What stays on you (not npm)

- **Hosted collector** - BYO endpoint; Vetch does not ship a SaaS ingest URL.
- **Website** - WordPress/HTML in `website/` repo folder (separate from this git repo).
- **Vercel partnership / marketplace** - business, not this package.
- **Stall kill in TypeScript** - Python only today; do not promise JS `StallDetected` until implemented.

## 7. Release checklist (each version)

- [ ] Bump `packages/vetch-ai-sdk/package.json` version
- [ ] `python scripts/sync_ai_sdk_registries.py` if Python registries changed
- [ ] `npm --prefix packages/vetch-ai-sdk test`
- [ ] Update `CHANGELOG.md` / tag notes
- [ ] Commit, push, tag `vX.Y.Z`, confirm CI green (including `npm-release` if versions match)
- [ ] `npm view @prismatic-labs/vetch-ai-sdk version` matches tag
