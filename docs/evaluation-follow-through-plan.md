# Vetch SDK: Evaluation Follow-Through

## Context

A code-level evaluation of the vetch SDK (energy/cost waste-control for LLM inference) surfaced four tracks of work. The architecture is sound; the gaps are hardening, drift, and coverage. All four tracks are in scope. This plan grounds each item in the current checkout, corrects claims that no longer hold, and sequences the work by risk and value.

Repo: `/Users/prismaticrawr/prismaticlabs-codebase/vetch` (normal git repo, CI in `.github/workflows/ci.yml`). Work lands on a feature branch per track; the user handles PR/merge.

This is the consolidated plan after several review passes. The most important course-corrections from earlier drafts are folded in:

- **Sequencing reverted to A → C(pricing/schema) → (B ∥ C-rest ∥ D).** An earlier draft moved methodology math (Track D) ahead of advisory work (Track B) on the theory that energy-math changes would desync advisory thresholds. That premise was traced and found false: every Track B detector triggers on token counts, token ratios, timing, or pricing — none read the `wh_per_1k_*` energy coefficients. Energy appears in advisories only as evidence/savings annotations, never as a trigger. So B and D are independent and run in parallel; B6 (PREMIUM-001) gates only on pricing data landing (C1), not on any energy-math change.
- **STALL-001 goes to a full-input salted hash, runtime-only.** Similarity is a runtime refinement, deliberately absent on the audit replay path (see B1/B0 for why this is forced by the privacy model, not a shortcut).
- **RETRY-001 is temporal-burst-primary**, with similarity as a runtime-only confidence booster, because failed calls frequently carry no token usage.
- **New advisories must be wired into both the runtime and audit code paths** — they are separate dispatchers today, and "audit-only PREMIUM-001" is the proof.

### Corrections from exploration (do NOT act on these)

- **README advisory drift is already fixed.** The current `README.md` table documents ERROR-001, STREAM-001, REASONING-001, CACHE-002 correctly. The only live drift is ATTRIBUTION-001 (in README as "partial", no detector in code). The CI-generated-table item (B7) is still worth doing as drift *prevention* — but it is not fixing the originally-described drift.
- **JS package is already insulated.** `packages/vetch-ai-sdk` is excluded from the Python sdist (the `[tool.hatch.build.targets.sdist]` include list omits `/packages`). It is a published npm package (v0.8.1), not a PoC. No action beyond an optional README note.
- **`.coverage` is already gitignored.** Only `bandit_report.txt` and `coverage.json` are tracked and need untracking.

---

## Track A — Hygiene + hermetic tests

Fast, low risk, high trust-signal. Unblocks clean local testing for everything else. **Pull B7 into this branch** — it is independent of the rest of Track B, and landing the generated-table guard before B adds new advisory codes prevents manual table churn.

### A1. Untrack CI artifacts
- `git rm --cached bandit_report.txt coverage.json` (leave files on disk).
- Add `bandit_report.txt` and `coverage.json` to `.gitignore` (the `.coverage*` / `coverage.xml` patterns already exist near lines 41–48).

### A2. Relocate internal scope/receipt docs (low urgency — cosmetic)
- `docs/V0.8.0_PRODUCTION_FIXES_RECEIPT.md`, `docs/SCOPE-v0.8.0-vercel.md`, `docs/vercel-ai-sdk-poc.md` are internal notes.
- Correction: the sdist include list (`/src`, `/tests`, `/data`, README, LICENSE, NOTICE, SECURITY, CHANGELOG) does not ship `docs/`, so these are a repo-presentation issue only, not a packaging leak. Priority drops accordingly. Action unchanged: move to a gitignored `internal/` dir, or delete if obsolete. **Confirm with user before deleting vs. relocating.**

### A3. Fix the coverage gate so single-file runs work
- Move *all* coverage flags out of `addopts` in `pyproject.toml` (lines 93–97), not just `--cov-fail-under`. Leaving `--cov=vetch` in `addopts` keeps single-file runs slow and produces a misleading coverage report — that was half the original complaint. `addopts` keeps only non-coverage defaults.
- CI already passes the full coverage invocation on the command line (`.github/workflows/ci.yml` ~line 33), so the gate stays enforced in CI. The calculator job already uses `--no-cov`; no change there.

### A4. Make tiktoken hermetic — without committing a blob
- Root cause: `_get_tiktoken_encoding()` (`src/vetch/calculation.py:361–399`) calls `tiktoken.get_encoding("cl100k_base")`, which downloads from `openaipublic.blob.core.windows.net` on first use. No mocking, no cache dir.
- **Do NOT commit the `cl100k_base` cache** (~1.7MB binary) — the sdist ships `/tests`, so a `tests/fixtures/` blob bloats every source distribution. Instead:
  - Default unit tests monkeypatch `_get_tiktoken_encoding` to a deterministic stub (fake encoder, fixed token mapping) in `tests/conftest.py`. The suite then never touches the network and never depends on the real encoding.
  - Live tests get `@pytest.mark.network`. Register the marker in `pyproject.toml` (`[tool.pytest.ini_options] markers = [...]`) and mark the genuinely live tests (`tests/test_calculation.py::TestTokenEstimationWithTiktoken::test_tiktoken_usage_if_installed` ~247 and `test_prepare_metrics_tiktoken_fallback` ~567). CI runs all; offline dev runs `-m "not network"`.
  - CI caches the tiktoken dir via `actions/cache` keyed on a fixed `TIKTOKEN_CACHE_DIR`, so live tests download once, not per matrix entry.
- **Verify hermeticity with the network actually blocked** — `pytest-socket`'s `--disable-socket` (with `--allow-unix-socket`, since sqlite and other local-socket users must still work) or a deny-all resolver. Running on a connected machine does not prove hermeticity. **Before declaring success, grep that the remote-registry and grid-sensing tests are fully mocked** — if any aren't, they will surface as new failures misattributed to the tiktoken change.
- Document `TIKTOKEN_CACHE_DIR` (set before tiktoken import) for air-gapped users, alongside the existing `VETCH_REGISTRY_PATH` offline path.

### A5. Drop Python 3.9 — OPTIONAL, verify first
- 3.9 hit EOL Oct 2025. But enterprise compliance stacks (exactly the paid-audit audience) move glacially, and the SDK has `dependencies = []`, so supporting 3.9 is nearly free. **Do not drop reflexively.** Check PyPI download stats for the 3.9 share first; only drop if negligible and we actually want a 3.10+ feature.
- If dropping: `requires-python = ">=3.10"` (`pyproject.toml:11`), remove the 3.9 classifier (line 23) and `"3.9"` from the CI matrix (~line 16), set `[tool.ruff] target-version = "py310"` (line 101). mypy already targets 3.10. Then opportunistically simplify `typing.Dict`/`Optional` to PEP 604/585.
- This is a breaking change for any 3.9 consumer: **bump the minor version and add a CHANGELOG entry.**

---

## Track B — Advisory completeness

Highest product value. All advisories follow the same additive pattern: extend `_RecentCall` (`stats.py:32–46`) if a new per-call field is needed → compute the metric in `SessionStats._compute_summary()` (`stats.py:205–330`) and add it to the returned summary dict → register an `AdvisorySpec` in `_ADVISORY_SPECS` (`advisory.py`, dict starts ~line 301) with evidence/confidence callables reading the summary, plus a generation branch. Reuse `_summary_evidence(*keys)` (`advisory.py:70`) and `_threshold()` (`advisory.py:78`) for config-overridable thresholds.

### B0. Unify runtime and audit dispatch (prerequisite for every new advisory)
- The runtime engine (`advisory.py` reading live `SessionStats`) and `vetch audit` (`audit_report.py` re-running detection over stored events) are **separate code paths**. PREMIUM-001 existing as audit-only proves it. Every new advisory below must be wired into *both*, or "run `vetch audit` and confirm new codes appear" fails because the implementation never built it there.
- Approach: have the audit path reconstruct a `SessionStats` by replaying stored events through `_update_locked`, then run the shared `_ADVISORY_SPECS` registry — one detection definition, two entry points. **Keep audit-tier thresholds separate from runtime thresholds** (don't collapse them); only the spec *dispatch* is unified, not the tuning.
- **Replay-safety boundary (forced by B1's privacy model):** input-similarity is runtime-only and structurally unavailable on replay (see B1). The audit path computes only the robust, replay-safe signals — low-output fraction, error-burst timing, token ratios, pricing — and omits the similarity confidence-booster.
  - Represent the absent signal as `None`, never `0.0`. "Signal unavailable" and "inputs measured as dissimilar" are opposite claims and must not be conflated downstream.
  - Consequence: `_stall_confidence` requires `input_similarity >= 0.8` for "high", so **audit-path STALL-001 caps at "medium" by construction.** That's the conservative direction and is correct, but the audit evidence dict must read `input_similarity: unavailable (runtime-only signal)`, and the taxonomy doc must state that audit confidence for STALL/RETRY caps below runtime confidence — otherwise the first customer comparing a runtime advisory to the audit report reads the gap as a bug in an evidence-quality product.

### B1. STALL-001 input-similarity fix — full-input salted hash, runtime-only (`stats.py:241–249`)
- Current signal = fraction of the window sharing the exact most-common input token count → false positives on fixed-template workloads, false negatives on drifting stuck agents.
- **Use a full-input salted hash, not a tolerance band and not prefix/suffix.** Rationale, settled across review:
  - Tolerance-band (±2–3%) clustering is analytically noisy — two unrelated prompts can share a token count.
  - A 200-char *prefix* hash collides on shared system prompts (saturating similarity app-wide) — a *broader* false-positive class than the one it replaces.
  - Prefix+suffix doesn't help: append-only drift mutates the suffix (new tool result at the tail), so it misses the drift case exactly as full-input does, while costing more code. CTX-001 already owns the growing-input pattern (`advisory.py:630` delegates the growing-input case to it). Full-input strictly dominates.
- Hash input definition (pin it): the **exact serialized role+content sequence, no normalization.** Any whitespace/casing "smart" normalization is a false-match generator and scope-creep invitation.
- Touch points: capture the hash where the event is built in the wrapper layer (`src/vetch/wrappers/`), add `input_hash: str | None` to `_RecentCall`, compute `recent_input_similarity` in `_compute_summary` as the largest same-hash fraction (replacing token-count bucketing at `stats.py:243–249`). Summary key name unchanged → STALL-001/ZOMBIE-001 confidence functions unchanged at runtime.
- **Privacy guard (hard requirements):** salt is per-process and never persisted; `input_hash` lives in `_RecentCall` only and is **never emitted into stored events or OTLP spans.** Short prompts under a deterministic or persisted salt are rainbow-table-recoverable, which would breach the "no prompt data leaves the environment" guarantee the README stakes the product on. This is also *why* similarity is runtime-only: a per-process salt makes cross-run audit similarity semantically void (a `vetch audit` window spans many process lifetimes; matches would only ever occur within-run, understating similarity in proportion to process churn), and persisting with a stable salt to fix that reintroduces the exposure. Degrade gracefully on the audit path instead.
- What this fixes, stated precisely: false **negatives** on drifting stuck agents (a stuck loop appending failed tool results keeps a stable full input only if it genuinely resubmits identically; where it grows, CTX-001 owns it). It does **not** newly fix identical-template false positives — an identical template hashes identically too; those remain handled by per-route threshold overrides plus the existing low-output requirement. Don't claim both.
- Known booster limitation to document in the taxonomy entry: retries that are near-identical except a mid-prompt nonce (regenerated timestamp, request-id, trace-id) hash differently each attempt, so the similarity boost is a no-op there. Acceptable — temporal burst-rate is RETRY-001's primary signal — but write "similarity boost requires byte-identical resubmission" as a documented limit, not a discovered one.
- **Risk + required mitigations:** this changes when STALL-001 fires, and STALL-001 can drive `set_stall_action("kill")` on production loops. All required: (a) fixture-based regression tests over recorded windows — a drifting stuck agent must now fire; a fixed-template classifier must not change behavior; **and an explicit must-not-fire case: long shared system prompt, varying user payload, short outputs** (the prefix-collision case this design exists to avoid); (b) a CHANGELOG entry calling out the detection-logic change; (c) version the advisory detection logic the same way audits report a methodology version, so a change in firing behavior is traceable.
- This is the one Track B item touching the wrapper layer; scope it as its own commit. **Hard dependency: B1 lands before B2.**

### B2. RETRY-001 — temporal burst-rate primary
- ERROR-001 already computes consecutive-error and error-fraction signals (`advisory.py:447–465`; `_count_trailing_errors` in `stats.py:49–57`).
- **Design gap to respect:** error responses frequently carry no usage block, and `_compute_summary` filters out calls with `in_tokens <= 0`. So "errors with high input-similarity" is structurally unmeasurable for exactly the failing calls we want. Similarity is **not** the primary signal.
- Primary signal is temporal: add a monotonic timestamp (`time.monotonic()`) to `_RecentCall` in `_update_locked` and compute a burst-rate metric — N calls within T seconds with error fraction above threshold. **Verify `_update_locked` appends error-only calls (no usage) to the window before any early-return**, since the burst metric depends on those entries existing. Input-similarity (post-B1, runtime-only) is a confidence booster only when usage is present.
- Fold in `retry_count`, which already flows through events (`audit_report.py:695–704`) — surface a `recent_retry_fraction` metric.
- Register RETRY-001 as a distinct spec; update README table from "planned" to implemented.

### B3. Reasoning-token burn — REASONING-002 (distinct code)
- `_RecentCall` carries `is_reasoning_model` and `has_reasoning_tokens` (`stats.py:45–46`) as booleans only. Add `reasoning_output_tokens: int` and populate it in `_update_locked` (`stats.py:155–157` already reads `reasoning_usage["output_tokens"]` — capture the count, not just the bool).
- New metric in `_compute_summary`: fraction of reasoning-model calls where reasoning tokens dominate (`reasoning / (reasoning + visible output)`) while visible output stays short.
- **Register as REASONING-002, a distinct code** — not an extension of REASONING-001, whose semantics ("missing reasoning tokens") are nearly the opposite signal.
- Threshold configurable and model-family-specific: wire through `_threshold(stats, "REASONING-002", "dominance", 0.7)` with per-family overrides. A flat 0.7 is risky — o-series and extended-thinking models have very different baseline reasoning verbosity.
- **Gate on `has_reasoning_tokens` being true** so it never fires merely because a provider doesn't report the field. Exposure varies by provider (OpenAI reports `reasoning_tokens`; Anthropic extended-thinking accounting differs). Note per-provider coverage in the taxonomy entry.

### B4. MAXTOK-002 — unbounded-generation guard
- `_RecentCall.requested_max_tokens` already exists (None when omitted); `output_cap_hit_fraction` already computed (`stats.py:285–294`).
- **Guard the obvious false-positive wall:** most SDKs/wrappers omit `max_tokens` by default, so "omitted" alone must NOT fire. Trigger only when (a) `max_tokens` omitted AND actual output repeatedly runs high (e.g. ≥2k tokens), OR (b) `max_tokens` set ≥10× the rolling output median. Require `window_size >= 5` before trusting the median.
- Define "expensive models" concretely: registry `usd_per_1k_output` above a `_threshold`-configurable floor — no hand-waving.
- The omission branch is effectively OpenAI-family-only (Anthropic's API requires `max_tokens`). **Severity INFO** — a tuning suggestion, not a warning. Register as the prescriptive companion to BABBLE-001/TRUNC-001 (point at the parameter).

### B5. SESSION-BUDGET-001 — give it a real advisory ID
- Currently alert-only via `src/vetch/budget.py` (`check_budgets()`, `BudgetAlert`). Route budget breaches through the advisory pipeline so `vetch audit` is the single source of truth, without changing the measure-don't-gate philosophy (still non-blocking).
- **Scope mismatch to resolve:** budget alerts are process-global, but advisories are per-`SessionStats`, and a budget can span many sessions. Don't inject the advisory into a single session's summary (which session would own it?). **Emit through the advisory hook/event layer with the budget key as the subject**, decoupled from any one session.

### B6. PREMIUM-001 — crude runtime detector (revenue MVP; gate only on C1 pricing)
- Full PREMIUM-001 exists audit-only (`audit_report.py:307–441`). Ship a v1 runtime advisory: registry pricing × short-output/low-input patterns per tag, enough to flag "this gpt-5.x route averages N output tokens." No task-complexity modeling. Generates the audit line items that convert free scans to paid reviews. Ship before the configurable per-advisory policy engine on the roadmap.
- **False-positive guard required:** short outputs on a premium model can be deliberate (accuracy-critical classification routes). **Severity INFO, per-tag overridable, never eligible for auto-action.** Mirror the existing audit-only thresholds (`min_avg_output_tokens`, CV bounds, retry/tool-call rate caps at `audit_report.py:314–341`).
- Gated on C1 pricing/aliases landing — not on any energy-math change.

### B7. README advisory table generated from source (rides with Track A)
- Add `scripts/gen_advisory_table.py` that reads `_ADVISORY_SPECS` + Advisory codes and emits the markdown table between sentinel comments in `README.md`.
- **Not a hard CI failure gate** — a contributor forgetting to run it and getting a red build is friction. Use a **pre-commit hook** (`.pre-commit-config.yaml` already exists). Do **not** use an auto-committing Action — it has no write token on fork PRs (the standard contributor flow) and dies there.
- Fixes the live ATTRIBUTION-001 drift in the process (either implement the detector or drop it from the table).

---

## Track C — Registry coverage + staleness

Registry lives in `src/vetch/registry/{energy,pricing,aliases,wue}.json`; loading/tiers in `calculation.py`; remote refresh + circuit breaker in `registry/remote.py`; provenance in `registry/PROVENANCE.md`.

### C1. Add missing high-volume models — split pricing from energy
- Models: Anthropic Claude Opus 4.x + Sonnet 4.5/4.6 (currently stops at `claude-sonnet-4`/`claude-haiku-4-5`); Google Gemini 3.x; xAI Grok; Amazon Nova; Qwen.
- **Land pricing + aliases first and independently** — verifiable from provider pages today, and pricing drift is the failure mode that corrupts the "avoidable cost" numbers the paid review depends on. Quick, high-value, and unblocks B6.
- Energy entries are the Tier 3 guesses. Add as Tier 3 with conservative values, **each carrying its own `PROVENANCE.md` note** (not just the C2 methodology tag). Explicit entries still beat the generic `get_conservative_energy()` fallback (`calculation.py:255–262`).

### C2. Per-entry `as_of` and `methodology` fields
- Extend the energy/pricing entry schema with `as_of` (ISO date) and `methodology` (`jegham-2025`, `provider-published`, `proxy-tier3`, `market-based`). Needed for C4 staleness and the Track D comparability caveat. Update the schema doc in `PROVENANCE.md` and any loader validation in `calculation.py`.
- **Forward/backward compat:** the loader must ignore unknown fields (so older SDKs pulling the registry remotely don't break on new fields) and treat the new fields as optional (so older bundled registries without them still load). Verify the remote-merge path in `registry/remote.py` tolerates the additive schema. Add a test asserting an entry missing both fields still loads.

### C3. Registry coverage check — scheduled warn-only, off the PR path
- A CI job hitting provider model-list endpoints needs API keys (OpenAI/Anthropic model endpoints require them), so on the PR path it means external-API flakiness blocks merges and forks can't run CI.
- Instead: a **scheduled (weekly cron) warn-only job** that diffs the registry against provider endpoints and opens an issue on drift. Ship the logic as `scripts/check_registry_coverage.py` for on-demand runs.
- PR-blocking checks run only against committed snapshots. Pricing-drift detection from a committed snapshot diff is deterministic and may live on the PR path if wanted — it's the live endpoint scraping that must stay off it.

### C4. Staleness warning in `vetch status`
- Using `as_of` from C2, have `vetch status` (CLI in `src/vetch/cli.py`) warn "pricing data is N days old" past a threshold. Remote refresh is opt-in (`VETCH_REGISTRY_REMOTE`, off by default per `remote.py:11`), so bundled data can silently age.
- **When remote refresh is enabled, read the effective (post-merge) entry's `as_of`**, not the bundled file's — otherwise it warns about data the user isn't using.

---

## Track D — Methodology references

Research-heavy. Independent of Track B (no energy coefficient feeds any advisory trigger), so it runs in parallel.

### D0. Verify sources before integrating
Several cited papers are recent. First step is to `web_fetch` each arXiv ID to confirm it exists, says what's claimed, and its license/citability. Do not bake numbers in unverified.

### D1. Google production methodology (arXiv:2508.15734)
- Provider-published, in-production full-stack measurement (median Gemini text prompt ~0.24 Wh / 0.03 gCO2e / 0.26 mL). Use to (a) ground the flat 1.2× hardware-overhead multiplier (`PROVENANCE.md:33–41`) with their host/idle decomposition, (b) promote Gemini entries toward a Tier-1-class "provider-published production" category.
- **Scope guard:** TPU-based production serving measured at the median-prompt level. It grounds the overhead decomposition (host/idle/PUE structure) and Gemini-specific entries only — **not** a per-token coefficient source for other models. Caveat to encode in the C2 `methodology` field: market-based emissions, not cross-provider comparable.

### D2. Caravaca et al. "From Prompts to Power" (arXiv:2511.05597)
- 32.5k measurements, 21 GPU configs, 155 architectures on vLLM, with a predictor for unseen architectures/hardware. Best available replacement for the Tier-3 joules-per-parameter proxy (`calculation.py:255–262`, `PROVENANCE.md:9–10`). Use their fitted predictor to tighten Tier 3 toward Tier-2 bands, especially the open-weight/local path.

### D3. Prefill vs decode + context-length energy
- `PROVENANCE.md:53` flags "energy per token independent of context length" as a known simplification. Recent work splits prefill/decode and shows prefill grows superlinearly with input length. Add context-length scaling — matters most for the RAG-bloat advisory (RAG-001 fires on the token *ratio*, but the energy *annotation* on those calls is exactly where the current flat model is most wrong). Bump `METHODOLOGY_VERSION` (`calculation.py:19`, currently "1.2").

### D3a. Methodology version pinning — backward-compat requirement
- A math change can move a customer's reported carbon/cost 20–30% between upgrades. For a paid-audit product that breaks baselining.
- **v1 deliverable (data-pinning via existing machinery):** `freeze_registry()` already exists (`remote.py:648`). Ship `vetch audit --methodology=<version>` / frozen-registry support so customers reproduce a prior baseline's *data*. **Stamp every audit report with both `METHODOLOGY_VERSION` and a registry snapshot identifier** (hash or `as_of` rollup) so a baseline is fully self-describing — when a re-run doesn't match, the stamps say which axis moved. That's most of the trust value at near-zero cost.
- **Boundary to state in the deliverable language now:** freeze pins data, not D3's formula (context-length scaling is code). Re-running yesterday's events through an upgraded binary with a frozen registry still applies the new formula. Scope the promise as **"reproducible to registry snapshot; formula version stamped"** before a customer scopes it for you.
- **Cheap middle option — check before fully deferring math replay:** D3 is the *only* planned formula change and is localized to `calculate_energy`. Keeping the pre-D3 path behind a single branch (`methodology < "1.3"` → flat per-token) for one release cycle delivers *exact* math replay for the immediately-prior version — the only version any existing customer baseline can be on — without N-version machinery. If that branch costs more than ~20 lines, drop it and ship as-is. The difference between "your last baseline replays exactly" and "your last baseline is labeled non-replayable" is a real sentence in a renewal conversation.

### D4. Provider sanity anchors + embodied basis (lowest leverage; last)
- OpenAI's ~0.34 Wh/query and Mistral/ADEME 2025 LCA as provider-stated anchors and an embodied-carbon basis for `calculate_embodied_carbon` (`calculation.py:923–994`, currently Patterson 2021).

### D5. Reporting alignment (SCI / ISO/IEC 21031) — pull EARLY
- Align `vetch methodology` terminology with the Software Carbon Intensity spec and GSF SCI-for-AI so customers can drop numbers into existing CSRD pipelines. Cheap (terminology/docs, largely independent of the math) and a direct conversion lever — "ISO/IEC 21031-aligned audit" is what a CTO wants to see. Do it early (alongside D0/D1), not last.

---

## Sequencing

1. **Track A** (one branch, fast) + **B7** — unblocks clean local testing; lands the generated-table guard before B adds codes. (Py3.9 drop stays optional/verify-first.)
2. **Track C schema + pricing** (C2, then C1 pricing/aliases) — small, additive, high-value; C2 unblocks D1's methodology tagging and C1 unblocks B6.
3. **Tracks B, C-remainder, and D run in parallel** (no cross-dependency):
   - **Track B**, in order: **B0** (dispatch unification — prerequisite) → **B1** (hard dependency for B2) → revenue MVPs **B6** (gated on C1 pricing, already landed) and **B2** → then **B5, B3, B4**.
   - **Track C remainder:** C1 energy entries, C4 staleness. C3 is a scheduled warn-only cron, off the PR path.
   - **Track D:** D0 → D1, D2, D3 + D3a, D5 early. D4 last.

---

## Verification

- **Track A hermeticity (exercised, not assumed):** `cd vetch && python -m pytest -m "not network" --disable-socket --allow-unix-socket` (or a deny-all resolver) must pass with the network actually blocked, proving no tiktoken/blob fetch. Full `pytest tests/ --cov=vetch --cov-fail-under=70` must pass (mirrors CI; coverage flags now live only on the CI command line, not `addopts`).
- **Track B advisories:** for each new advisory, add unit tests under `tests/` (follow existing `test_advisory*.py` patterns) feeding synthetic event windows that trip each trigger; assert code/severity. **B1 specifically** needs the three regression fixtures: drifting stuck agent (must fire), fixed-template classifier (unchanged), long-shared-system-prompt + varying payload + short outputs (must NOT fire). For **B0**, run `vetch audit` against a fixture session and confirm new codes appear **and** that audit-path STALL/RETRY confidence caps below runtime (similarity reported `unavailable`, not `0.0`).
- **Track C:** run `vetch status` and confirm the staleness warning fires on a back-dated `as_of`; run the coverage-diff script and confirm it flags a removed model; assert an entry missing `as_of`/`methodology` still loads.
- **Track D:** calculator parity job in CI must still pass after methodology changes; diff a known model's Wh before/after and confirm the change is explained by the new source; confirm audit reports carry both the methodology-version and registry-snapshot stamps.
- **CI:** push branch, confirm all jobs green (test matrix, lint, calculator parity, ai-sdk, package, security). `ruff check` and `mypy` (strict) clean before each PR.

---

Residual risk is concentrated in two test-coverage problems, not design problems: B1's regression fixtures (especially the prefix-collision must-not-fire case) and the B0 audit/runtime threshold split. Both are addressed by the verification steps above; if either fixture set is thin, that's where a regression will slip through.

---

## Work split (Claude Code ∥ Cursor)

Split by subsystem to minimize shared-file contention, not by item count. Claude Code owns the coupled, serialized, high-judgment detection/energy-math core; Cursor owns the additive, file-localized, well-specified work (infra, registry data, docs). Each side works its own branches.

### Claude Code — advisory engine + methodology correctness
Owns `stats.py`, `advisory.py`, `audit_report.py`, `budget.py`, `src/vetch/wrappers/`, and the energy-math/version path in `calculation.py`.

- **Track B (all):** B0 (runtime/audit dispatch unification — prerequisite), B1 (salted-hash similarity + privacy guard), B2 (RETRY-001 temporal), B3 (REASONING-002), B4 (MAXTOK-002), B5 (SESSION-BUDGET-001), B6 (PREMIUM-001 runtime). Done in dependency order: B0 → B1 → B6 ∥ B2 → B5, B3, B4.
- **Track D math/correctness:** D0 (verify arXiv sources before any number lands), D2 (Tier-3 predictor), D3 (prefill/decode + context-length scaling in `calculate_energy`, bump `METHODOLOGY_VERSION`), D3a (version pinning, freeze_registry replay, audit stamping). Owned here because D3a's replay/stamp semantics couple to the B0 audit path.

### Cursor — infra, registry data, docs
Owns `pyproject.toml`, `.gitignore`, `tests/conftest.py`, CI/pre-commit, `scripts/`, the registry JSON files, the registry loader path in `calculation.py`, `registry/remote.py`, `PROVENANCE.md`, `cli.py`, `README.md`.

- **Track A (all):** A1 (untrack artifacts), A2 (relocate internal docs — confirm with user before deleting), A3 (coverage gate), A4 (hermetic tiktoken — conftest stub + `network` marker + CI cache + socket-blocked verify), A5 (Py3.9 drop — verify-first, optional).
- **B7:** generated advisory table (script + pre-commit hook + README sentinels). Rides with the Track A branch per sequencing; only edits `README.md` + a script, not detector logic.
- **Track C (all):** C2 (per-entry `as_of`/`methodology` schema + loader + additive-compat test) first, then C1 (pricing/aliases first, energy entries Tier 3 second), C3 (coverage-check script + weekly cron), C4 (staleness in `vetch status`).
- **Track D docs/research:** D1 (Google methodology grounding → PROVENANCE + Gemini entry tagging), D4 (provider anchors + embodied basis), D5 (SCI / ISO/IEC 21031 terminology alignment — pull early).

### Coordination boundaries (the only cross-side handoffs)
- **`calculation.py`** is touched by both (Cursor: C2 loader/validation; Claude: D2/D3 `calculate_energy`). Different functions, but **Cursor lands C2 first** (it's the early sequencing step regardless), Claude rebases D2/D3 on top.
- **`PROVENANCE.md`** touched by both (Cursor: C1 per-entry notes + D1; Claude: D3 simplification removal + methodology-version note). Cursor's registry/doc edits land first; Claude appends the formula/version note after.
- **B6 ← C1:** Claude's B6 gates on Cursor's C1 pricing/aliases landing. No energy-math dependency.
- **D1 → Claude (conditional):** if Cursor's D1 host/idle decomposition implies a change to the 1.2× overhead inside `calculate_energy` (not just PROVENANCE prose), that change hands to Claude as a spec rather than Cursor editing the math path.
- Both sides update `CHANGELOG.md` for their own user-visible changes; resolve any conflict at merge (it's append-only).
