# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.10.0] - 2026-07-03

Capability observability: function tools offered vs invoked, cache-aware wasted
schema cost, Kind C model routes via registry map, TOOL-DEAD-001 / CAP-001.

### Changed — Datacenter PUE refresh (vendor reports)

- **Google / Vertex AI: 1.10 → 1.09** (Google 2026 Environmental Report, FY2025
  fleet-wide PUE).
- **AWS / Anthropic / Bedrock: 1.15 → 1.14** (AWS 2025 Sustainability Report,
  global average).
- **Azure / OpenAI: 1.12 unchanged**; source refreshed to the Microsoft 2025
  Environmental Sustainability Report (`as_of` 2025-05-29). Microsoft publishes
  no clean operational fleet average, so this remains a newest-generation
  figure — a slightly more favorable basis than the Google/AWS fleet averages
  (documented in METHODOLOGY.md).
- **Impact:** estimated energy and carbon shift for Google and AWS/Anthropic
  workloads (roughly −1% PUE overhead each). Re-baseline dashboards/budgets that
  track absolute carbon for these providers.

### Fixed — LangChain integration (`VetchCallbackHandler`)

- Subclass the real `BaseCallbackHandler`, so chat-model dispatch no longer
  escapes with `AttributeError('raise_error')` (the handler now inherits
  `ignore_chat_model` / `raise_error` and the default `on_chat_model_start`).
- Extract usage from the message `usage_metadata` (Gemini and the standardized
  LangChain shape) in addition to legacy `llm_output["token_usage"]`;
  Gemini-shaped results are captured instead of silently dropped.
- Map `gemini*` models to the `google_genai` provider key (was `vertexai`).

### Added — OpenAI SDK 2.x support

- Support `openai >= 2.0` (verified against 2.44.0): `TESTED_OPENAI_VERSIONS`
  is now `[1.0.0, 3.0.0)`, so `instrument()` covers 2.x without
  `VETCH_FORCE_PATCH`.

### Fixed — OpenAI module double-instrumentation recursion

- `instrument()` / `uninstrument()` cycles could leave `openai.OpenAI.__init__`
  wrapping itself, so the next real client construction raised
  `RecursionError`. Module instrumentation is now idempotent (a sentinel guard
  never re-wraps an already-patched `__init__`) and closure-safe (the original
  `__init__` is captured in a local, not a mutable global).

### Added — `instrumentation_status()` and import-order visibility

- New `vetch.instrumentation_status()` reports, per provider (openai, anthropic,
  azure_openai, vertexai, google_genai, ollama):
  `installed` / `imported` / `instrumented` / `version` / `tested`.
- `instrument()` now warns when an SDK is installed but was not imported before
  the call (the `sys.modules` gate means it is silently not instrumented), and
  the docstring documents the import-order requirement.

### Fixed — Tag allowlist no longer leaks rejected keys

- Allowlist-filtered tags previously echoed the rejected key name into the
  event's `vetch_warnings` (a fail-closed control re-leaking what it blocked).
  The event-bound warning is now key-free (`"Tag not in allowlist, filtered
  out."`); full detail (key + remediation) goes to the local log only. The
  filtered-tag counter (`get_tracking_stats()["allowlist_filtered"]`) is
  preserved.

### Added — Capability fields on `InferenceEvent` (schema v2, additive)

- `tools_offered`, `tools_invoked`, `tool_call_count`, `capabilities_invoked`,
  `tool_schema_tokens` on Python events and `CapturedCall` / `capture()`.
- `SessionStats.summary()` rollups: `function_tools_never_called`,
  `wasted_tool_schema_tokens`, `wasted_tool_schema_cost_per_request_usd`,
  `wasted_tool_schema_session_cost_usd`, `wasted_tool_schema_cost_usd` (session
  headline = per-request × `dead_tool_offer_request_count`), `dead_tool_offer_request_count`,
  `declared_capabilities_silent`, `capability_invocation_counts`,
  `tool_call_event_rate`.
- New module `vetch.capabilities` with OpenAI, Anthropic, GenAI, Vertex, and
  Ollama extraction; streaming accumulators for OpenAI + Anthropic.
- Registry `model_capabilities.json` with freshness CI; `configure_capabilities()`.
- Advisories: **TOOL-DEAD-001** (per-session dead tools + wasted cost);
  **CAP-001** (audit-only, windowed declared-route silence).
- OTel: `gen_ai.tool.*` array attributes + `vetch.tools_never_called` +
  `vetch.wasted_tool_schema_tokens` (per-request dead-schema footprint).
- CLI: `vetch audit --expected-capabilities` for CAP-001 manifest.
- `rollup_capability_summary_from_events()` for stored-event rollups.
- Per-event `vetch_warnings` when dead tools ride on fully cached requests.
- JS SDK: stream `tools_invoked` via `toolNamesInvoked`; exports
  `extractToolsOffered` / `extractToolsInvoked`.

## [0.9.0] - 2026-06-23

Registry freshness, honest match precision, and correct self-hosted cost
accounting. `METHODOLOGY_VERSION` → `1.3`. Schema stays v2 (the new field is
additive). Pre-1.0: the uncertainty change below can widen reported bounds for
models that previously prefix-matched a measured row — review dashboards/alerts.

### Added — Registry rows for current-generation models

- **Gemini 3.x**: `gemini-3-flash`, `gemini-3.5-flash`, `gemini-3.1-flash-lite`,
  `gemini-3.1-pro` (tiered >200k), with `-preview`/`-latest` aliases. Energy is
  Tier-3, proxied from the nearest 2.5-class sibling (no empirical measurement
  exists); **pricing is verified** against the official Google pricing page.
- **Current Claude**: `claude-sonnet-4-5`, `claude-sonnet-4-6`,
  `claude-opus-4-5/4-6/4-7/4-8`, with dated/`-latest` aliases. Energy proxied
  from `claude-sonnet-4` / `claude-3-opus`; pricing verified against the official
  Anthropic pricing page. Each new pricing row carries an `as_of` date.
- Ships via the remote registry, so existing installs gain coverage without an
  upgrade. New `scripts/check_registry_freshness.py` (in CI) enforces
  energy↔pricing parity and flags pricing rows older than a year.

### Added — `model_match` on `InferenceEvent` (schema v2, additive)

- New field: `"exact" | "alias" | "prefix" | "family" | "fallback"`, exported as
  the `vetch.model_match` OpenTelemetry attribute. Lets downstream tell an exact
  hit from a proxy.

### Changed — Resolver hardening (`resolve_model_match`)

- **Case-insensitive** matching (e.g. `GPT-4O`, `Gemma-4-31b-it` now resolve).
- **Deterministic, conservative family fallback**: an unknown model in a known
  provider family is proxied to a representative same-family row, biased to the
  larger (higher-energy) sibling so it never silently undercounts.
- **Proxy matches no longer masquerade as exact**: `prefix`/`family` matches are
  floored to Tier 3 uncertainty even when the matched row is Tier 1.
- `resolve_model()` is retained as a thin `(name, known)` back-compat wrapper.

### Changed — Self-hosted / OpenAI-compatible cost accounting

- The OpenAI provider classifies `base_url` into `openai` / `ollama` /
  `self-hosted` / `openai-compatible`. A non-OpenAI endpoint is **no longer
  billed OpenAI's per-token rates**: self-hosted/local reports cost `0` (you pay
  for hardware, captured as energy); other compatible hosts report cost
  `unknown`. `VETCH_SELF_HOSTED=true` forces the self-hosted label.

### Fixed — `tracking_degraded` now actually fires

The degradation-score threshold was `2.5`, but the score maxes at `2.0`, so
`tracking_degraded` was always `false` (dead). Recalibrated the threshold to
`1.0` (both SDKs) so the flag means what it says: it fires for unknown models,
prefix/family proxies, estimated usage, and missing usage, while a healthy call
(exact match, real usage, even an honest Tier-3 model) stays clean. The Python
proxy weight added above feeds this. **Behavioral**: events that were silently
`false` will now correctly report `true`; downstream filters/alerts on
`tracking_degraded` should expect to see it populated.

### Packaging

- Description/README precision corrected: distinguishes **measured** (Tier 0 GPU
  telemetry) from **inferred** (hosted-API, including Jegham Tier 1) from
  **fallback**; the "measured in Azure/AWS datacenters" wording is fixed (Jegham
  is infrastructure-aware modeling, not power metering).
- Keywords expanded (google, gemini, opentelemetry, ollama, self-hosted, vllm,
  cost, gpu, finops, sustainability). `anthropic` and `ollama` are now first-class
  install extras. New README sections: Capabilities, Self-hosted and raw HTTP,
  Model coverage and resolution.

### JS/TS SDK (`@prismatic-labs/vetch-ai-sdk` 0.9.0)

Ported to parity with the Python SDK:

- `resolveModelMatch()` with the same precision tiers, case-insensitive matching,
  conservative family fallback (same current-gen targets), and the prefix/family
  Tier-3 energy floor. `resolveModel()` kept as a back-compat wrapper.
- `model_match` added to `VetchEvent` and populated on every event.
- Cost routing: a `base_url`-classified self-hosted/Ollama endpoint reports cost 0
  (`billing_tier: "self-hosted"`); an OpenAI-compatible third-party host reports
  cost `null` (`billing_tier: "unknown"`); official OpenAI/Azure keeps list pricing.
  `provider-label` classification mirrors Python, with a guard so a Google/Anthropic
  AI-SDK provider is not reclassified from its own `baseURL` alone.
- Registry rows (Gemini 3.x, current Claude) ship via the synced TS registry copy.

## [0.7.0] - 2026-05-28

### Added — Savings & Intervention Accounting

The headline feature of v0.7.0: Vetch now tracks and surfaces what it has
actually saved, in dollars, Wh, and carbon. Two strictly separate buckets
are reported — realized cache savings (genuine, measurable) and circuit
breaker cost-at-risk (waste interrupted, not guaranteed savings) — so the
numbers are always trustworthy enough to show an engineering leader or
FinOps team.

#### Schema (`InferenceEvent`)

- **`cache_cost_saving_usd`** — Realized cost saving vs. uncached baseline:
  full input price minus discounted cache-read price. Distinct from
  `estimated_cost_cache_read_usd`, which is the cost paid. `None` when no
  cache tokens were read.
- **`cache_carbon_saving_g`** — Carbon saving vs. uncached baseline, computed
  from the same grid intensity and provider PUE path as `estimated_carbon_g`.
- `estimated_cost_cache_read_usd` comment updated to clarify it is the
  discounted cost *paid*, not the saving.

#### Session

Five new properties on `Session`:

- `total_cache_cost_saving_usd` — accumulated realized cache savings
- `total_cache_energy_saving_wh` — accumulated energy savings (Wh)
- `total_cache_carbon_saving_g` — accumulated carbon savings (gCO2e)
- `circuit_breaker_interventions` — count of STALL-001 kill/reroute fires
- `circuit_breaker_cost_at_risk_usd` — cost at risk interrupted (not savings)

New method:

- `session.record_circuit_breaker_intervention(cost_at_risk_usd)` — called
  centrally by `_stall.apply_stall_action()` on kill or reroute.

#### Storage (`~/.vetch/usage.db`)

- New columns on `events` table: `cache_cost_saving_usd`,
  `cache_energy_saving_wh`, `cache_carbon_saving_g`
- New `interventions` table: durable record of every circuit breaker fire,
  survives sessions killed mid-flight.
- `UsageSummary` carries five new fields: `total_cache_cost_saving_usd`,
  `total_cache_energy_saving_wh`, `total_cache_carbon_saving_g`,
  `total_circuit_breaker_interventions`, `total_intervention_cost_at_risk_usd`.
- New `store_intervention()` helper: writes synchronously for durability.
- Migration is automatic and non-destructive on existing databases.

#### Audit Report

`AuditReport` gains six new fields:

- `realized_cache_savings_usd`
- `realized_cache_energy_savings_wh`
- `realized_cache_carbon_savings_g`
- `projected_monthly_cache_savings_usd`
- `circuit_breaker_interventions`
- `intervention_cost_at_risk_usd`

Text and Markdown formats include a new **SAVINGS & INTERVENTIONS** section
above FINDINGS, with the two buckets kept visually distinct.

`PREMIUM-001` is now launched as an audit-only **Large Model Rightsizing
Candidate** finding. It flags stable, tagged workflows that mostly use a
large/premium model and have cheaper same-provider candidates worth testing.
It does not claim avoidable cost and does not recommend automatic downgrade;
it queues the workflow for offline or shadow eval.

#### CLI: `vetch savings`

New command: `vetch savings [--days N] [--format text|json]`

```
SAVINGS SUMMARY  (last 30 days)
════════════════════════════════════════════════
Requests tracked:              1,204

Realized cache savings
  Cost saved via caching:      $12.40
  Energy saved via caching:    3.20 Wh
  Carbon saved via caching:    1.05 gCO2e

Circuit breaker interventions
  Interventions:               3
  Cost at risk interrupted:    $8.70

Total realized cache savings:  $12.40
Monthly run-rate:              $12.40 / month
════════════════════════════════════════════════
Generated by Vetch v0.7.0
```

#### OpenTelemetry

Both export paths (`otel.py` and `exporters/opentelemetry.py`) now emit
`vetch.cache_cost_saving_usd` and `vetch.cache_carbon_saving_g` as span
attributes alongside the existing `vetch.cache_energy_saving_wh`.

## [0.8.1] - 2026-05-31

### Added — `@prismatic-labs/vetch-ai-sdk` parity patch (081 margherita)

Thin-crust release: same middleware oven, extra toppings aligned with Python.

- **npm** ships as `@prismatic-labs/vetch-ai-sdk` (the `@vetch` scope was unavailable).
- **Session advisories in TS**: ERROR-001, STREAM-001, REASONING-001 (rolling window with `sessionId`).
- **Budget parity**: `retry_count`, per-call `budget_*` fields, `VETCH_BUDGET_COST_USD` / `ENERGY_WH` / `CARBON_G` env vars, `BUDGET-001` advisory (per-request; not Python session rollup), `onBudgetExceeded` callback.
- **Session isolation**: rolling advisories require `attribution.sessionId`; calls without it only emit per-call advisories (no shared `__ephemeral__` bucket).
- **Edge graph**: `loadLocalCalibration` is dynamically imported from `enrichVetchEvent` so the default middleware path does not statically bundle filesystem code.
- **Ollama provider label**: `localhost:11434`, `OLLAMA_HOST`, or `providerOverride` — matches Python OpenAI-compat detection for Tier-0 calibrations.
- **Naples release switch**: opt in with `easterEggs: true` or `VETCH_EASTER_EGGS=true` to emit `NAPLES-081`.
- **Build**: `VETCH_VERSION` generated from `package.json` via `scripts/write-version.mjs`.

### Documentation

- Scope and quickstart tables updated for real TS capabilities (no longer list ERROR/STREAM/REASONING as Python-only).

## [0.8.0] - 2026-05-30

### Added — Apple Silicon Calibration Toolbox

New `vetch calibrate-apple-silicon` command produces Tier 0 hardware-measured
energy coefficients for Ollama models running on Apple Silicon. Uses
`powermetrics` (requires `sudo`) to read Apple PMU counters (CPU+GPU+ANE).

- **VLM energy model**: four-parameter least-squares fit — `β0` (intercept),
  `β_img` (`wh_per_image`), `β_in` (`wh_per_1k_input`), `β_out` (`wh_per_1k_output`)
- **`CalibrationResult`** extended with optional VLM fields: `wh_per_image`,
  `visual_tokens_per_image`, `intercept_wh`, `active`, `rejection_reasons`
- **`save_calibration()`** writes to `~/.vetch/calibrations/` and is picked up
  automatically by `calculate_energy()` at inference time
- **Validation gates**: rejects calibrations with R² < 0.85, negative
  coefficients, condition number > 30, idle power drift > 15%, or missing GPU power
- **Workload grid**: ~22 runs covering `n_images` ∈ {0,1,2}, text tokens ∈
  {20,128,512}, max tokens ∈ {5,32,128,256}; unique images per run to prevent
  Ollama KV-cache contamination
- **Image set**: Wikimedia public-domain images (`vetch calibrate-apple-silicon
  --fetch-images`); falls back to procedurally generated synthetic images
- **`--strict-images`**: fail if any Wikimedia download is missing
- **Detail JSON** (`*_apple_detail.json`): shareable calibration record including
  fit statistics, CI bounds, hardware metadata, and power sampler provenance

Note: measures estimated SoC power (CPU+GPU+ANE), not wall power. Numbers are
internally consistent for relative comparisons on the same hardware; do not
use them directly in $/kWh or gCO2e/kWh narratives without accounting for
the measurement basis (`"not_wall_power": true` in the detail JSON).

### Added — Vercel AI SDK Middleware (`@prismatic-labs/vetch-ai-sdk`)

First-party Vetch middleware for Vercel AI SDK 6.x. Schema v2 events,
local energy/carbon/cost estimates, advisories, and Edge-safe emission.

- **`createVetchMiddleware()` / `withVetch()`**: wraps Vercel AI SDK model
  calls and emits `VetchEvent` objects after each inference
- **`enrichVetchEvent()`**: local energy/cost/carbon calculation from the
  bundled registry; supports `EnergyOverride` for calibration-sourced values
- **`EnergyOverride`**: interface for local calibration values — includes
  `wh_per_image`, `visual_tokens_per_image`, and `intercept_wh` for VLMs
- **`loadLocalCalibration()`**: reads `~/.vetch/calibrations/` on Node.js;
  auto-loaded into `enrichVetchEvent` when present
- **`createVetchSession()` / `detectAdvisories()`**: per-session state and
  protocol advisory detection (VOID, PROTO-001, etc.)
- **Emitters**: `consoleJsonEmitter`, `createFetchEmitter` (with retries and
  bearer token), `noopEmitter`; all throw on timeout (no silent drops)
- **Session LRU**: `onSessionEvicted` callback with debug warning for evicted
  sessions with incomplete state
- **`VETCH_VERSION`**: exported constant, aligned with Python package version
- **Registry sync**: `scripts/sync_ai_sdk_registries.py` + CI step keeps
  energy/pricing/alias/WUE registries in sync between Python and TypeScript

### Added — Native Ollama SDK Instrumentation

`providers/ollama.py` instruments `ollama.Client.generate` and `.chat`
without requiring the OpenAI-compatible endpoint. Captures model, token
counts (from `prompt_eval_count` and `eval_count`), and image count from
request kwargs.

- Auto-enabled when `instrument()` is called and the `ollama` package is imported
- Reversible via `uninstrument()`
- Using Ollama via the OpenAI SDK (`base_url="http://localhost:11434/v1"`) now
  also works: `providers/openai.py` auto-detects localhost:11434 (and the
  `OLLAMA_HOST` env var) and sets `provider="ollama"` so Tier-0 calibrations
  apply. No code change required.

### Added — Session Advisories (Python)

Three new advisories added to `advisory.py` alongside the existing nine:

- **CACHE-002**: High input-token repetition (same signal as CACHE-001) with no
  `cache_read_tokens` observed — caching is available but not yet active.
  Fires when >50% of recent calls share the same input count and none return
  cache reads.
- **STREAM-001**: High incomplete-stream fraction — fires when ≥30% of streaming
  calls over a ≥5-call window complete with `complete=False`. Indicates streams
  being cancelled before finishing.
- **REASONING-001**: Reasoning model called without returning reasoning tokens —
  fires when o1/o3/deepseek-r1-style models have no `usage.reasoning` output
  across ≥5 recent calls, suggesting the reasoning path is not being activated.

`_RecentCall` (internal rolling-window record) extended with five new fields
required by the above: `cache_read_tokens`, `is_stream`, `complete`,
`is_reasoning_model`, `has_reasoning_tokens`.

### Added — Session Advisories (`@prismatic-labs/vetch-ai-sdk`)

`detectAdvisories` in `advisories.ts` now includes three session-level
checks (using the rolling 40-event window alongside existing per-call checks):

- **STALL-001**: last 5 non-error calls each produced ≤5 output tokens.
- **CACHE-001**: >50% of the recent window shares the same input token count
  (≥6-event minimum).
- **CACHE-002**: CACHE-001 conditions met with no `cache_read_tokens > 0`
  anywhere in the window.

7 new vitest tests added; total parity test suite is 14 tests.

### Added — `retry_count` on `InferenceEvent`

`InferenceEvent` (schema v2) now includes `retry_count: int | None`. The
wrapper emits `0` by default (first try, no retries). Applications performing
explicit retry logic can set this field to activate the `PREMIUM-001`
retry-rate gate in the audit engine, which filters stable workflows from
downgrade recommendations. The default of `0` means the gate is inert for
callers that do not set it — this is documented in `audit_report.py` and
`METHODOLOGY.md`.

### Added — Ollama OpenAI-compat provider auto-detection

`providers/openai.py` now calls `_infer_openai_provider()` during
`patch_openai_client()`. When the client's `base_url` contains
`localhost:11434` or `127.0.0.1:11434`, or `OLLAMA_HOST` is set, the
provider label is set to `"ollama"` instead of `"openai"`. This means
Tier-0 Ollama calibrations apply automatically when using Ollama via the
OpenAI-compat API — no code change required.

### Added — Community Calibrations

`src/vetch/community_calibrations.py` + bundled `data/community_calibrations.json`.
Provides a fallback registry of community-contributed hardware-measured
coefficients for models not covered by the vendor-published registry.
Currently empty; populated by `scripts/aggregate_calibrations.py` from
submitted detail JSON files.

### Added — ERROR-001 Advisory

New advisory: fires when ≥3 consecutive API errors are detected, or when
≥40% of recent calls return `error=True`. Covers provider outages, quota
exhaustion, safety filter blocks, and malformed request patterns.

Severity: CRITICAL when ≥5 consecutive errors; WARNING otherwise.
Security signal flagged (OWASP-LLM04, OWASP-LLM10).

### Fixed

- **TRUNC-001**: Python rolling-window detection only checked
  `finish_reason == "max_tokens"` (Anthropic). Now also catches
  `"length"` (OpenAI), so TRUNC-001 fires correctly for all providers.
- **`calculation.py`**: `_effective_text_input_tokens()` correctly subtracts
  visual tokens from text input when `visual_tokens_per_image` is set,
  preventing double-counting in `wh_per_1k_input` when a calibration is
  active for a VLM.
- **`load_calibration()`**: resolves model aliases (`moondream` ↔
  `moondream:latest`) so calibrations are found regardless of tag suffix.

## [0.4.0] - 2026-04-27

### Added — Uncertainty bounds (UACA Phase 1)

`InferenceEvent` now carries explicit absolute lower/upper confidence
bounds on energy and carbon, derived from the existing
`energy_uncertainty_pct`. No new modelling — just exposes the uncertainty
band so downstream tooling (dashboards, compliance reports, audits) can
read absolute numbers without recomputing the math.

- `energy_p5_wh` — lower bound (clamped at 0.0)
- `energy_p95_wh` — upper bound
- `carbon_p5_g` — lower bound (clamped at 0.0)
- `carbon_p95_g` — upper bound

Bands match the per-tier uncertainty already documented:

- Tier 0 (hardware-measured): ±20%
- Tier 1 (vendor-published): ±50%
- Tier 2 (validated): ±100%
- Tier 3 (estimated): ±1000%

Deferred from the UACA proposal to a future release: the latency-based
intensity heuristic (TTFT/TBT-derived energy scaling) and Time-of-Use
carbon factors. Both would need calibration data we don't yet have.

### Added — Circuit Breaker

The headline feature of v0.4.0: STALL-001 detection now stops runaway agent
loops before they burn more money. Configurable per-session, fail-open by
design.

- **`vetch.set_stall_action(action, fallback_model=None)`** — Configure how
  Vetch responds when STALL-001 fires:
  - `"log"` (default — backwards compatible): generate the advisory, take
    no action.
  - `"warn"`: log a stderr WARNING on the next call after detection.
  - `"kill"`: raise `vetch.StallDetected` on the next call, stopping the
    loop.
  - `"reroute"`: transparently substitute the model with `fallback_model`
    on the next call. If the substituted call rejects the parameters
    (param mismatch, missing capability, etc.), Vetch falls back to the
    original model — fail-open.
- **`vetch.StallDetected`** exception. Inherits from a new
  `vetch.VetchInterrupt(RuntimeError)` umbrella class — *not* from
  `VetchError`/`ValueError`. Generic `except ValueError:` handlers in user
  code will not swallow it.
- **`session.clear_stall()`** — re-arms the breaker after a human-in-the-loop
  fix (corrected prompt, fixed retriever, etc.). The next stall will trip
  the breaker again.
- **`Advisory.request_count`** — new field on the advisory namedtuple
  carrying the number of stalled calls (used by the exception payload).
- Lazy STALL-001 detection: the advisory cycle is skipped entirely until a
  session has at least 10 calls. Eliminates per-call overhead in short
  agent runs.

### Changed

- `pyproject.toml` description rewritten to reflect the resource-aware
  framing of v0.4.0.
- README hero rewritten: "The circuit breaker for runaway AI inference."
- Energy and carbon metering remain core. They now travel with a sharper
  cost-savings story.

### Fixed

- `tests/conftest.py` now resets the global `_session_stats` singleton
  between tests, fixing pre-existing test-pollution where a previous test's
  MagicMocks leaked into `vetch_session_stats` MCP tool output.
- `__version__` in `vetch/__init__.py` was out of sync with `pyproject.toml`
  (`0.3.0` vs `0.3.1`). Both now reflect `0.4.0`.

### Provider coverage

The circuit breaker is wired into every provider wrapper:

- OpenAI (sync + async + streaming)
- Anthropic (sync + async + streaming)
- Azure OpenAI (auto — uses the OpenAI patches under the hood)
- Vertex AI (sync + async). Reroute degrades to log-mode here because
  Vertex binds the model to the `GenerativeModel` instance, not kwargs.
  Kill / warn / log all work.
- Google GenAI (sync + async). Reroute is applied to `generate_content*`
  methods only; embedding methods are deliberately left untouched.

### Notes

- **Backwards compatible:** existing users see no behaviour change unless
  they call `set_stall_action(...)`. Default remains `"log"`.
- **In-flight calls are not interrupted.** Streaming calls complete
  naturally; the circuit breaker only intercepts calls that *start* after
  a stall is detected. With `asyncio.gather([...10 calls...])`, all 10
  in-flight calls complete; the 11th onwards is intercepted.
- **What STALL-001 actually detects:** short outputs *and* high input
  similarity. A succinct classifier returning 1-token answers from varied
  inputs is not a stall — input similarity is low. STALL-001 fires only
  when the agent is producing little output AND repeating roughly the
  same input pattern.

### Example

```python
import vetch
from openai import OpenAI

vetch.instrument()
vetch.set_stall_action("kill")

client = OpenAI()
try:
    with vetch.Session() as session:
        for step in agent.run():
            client.chat.completions.create(...)
except vetch.StallDetected as e:
    print(f"Stopped: {e.request_count} calls, ~${e.wasted_cost_usd:.2f} wasted")
```

See `examples/circuit_breaker_demo.py` for a runnable end-to-end demo.

## [0.2.4] - 2026-03-22

### ⚠️ BREAKING BEHAVIOR (data changes — no API changes)

- **`gpt-4o-mini` energy ~3× lower** — `gpt-4o-mini` and `gpt-4o-mini-2024-07-18` were incorrectly aliased to `gpt-4o` since v0.1.0. Both models now resolve to a dedicated `gpt-4o-mini` Tier 3 entry (~0.10/0.30 Wh/1k input/output). Historical data will appear ~3× higher than post-upgrade data. **This was a data error, not a regression.** Re-baseline any dashboards or budgets tracking this model.

- **`claude-3-5-haiku-*` energy ~20% higher** — `claude-3-5-haiku`, `claude-3-5-haiku-20241022`, `claude-3-5-haiku-latest`, and `claude-haiku-3-5` were aliased to `claude-3-haiku` (Claude 3 Haiku, a different model). All aliases now point to `claude-3.5-haiku` with a Tier 3 estimate (~0.035/0.105 Wh/1k). Historical data will appear ~20% lower than post-upgrade data. Re-baseline dashboards tracking this model.

### Added

- **Energy registry: 14 new Tier 1 models** from Jegham et al. (arXiv:2505.09598, 2025) — the same paper underpinning GPT-4o since v0.2.0. All new entries are prompt-length-aware (short/medium/long buckets) with IT-equipment-only basis (pre-PUE):
  - **GPT-4.1**, **GPT-4.5**, **o3-mini**, **o1-mini**, **o4-mini** (medium reasoning effort)
  - **Claude 3.7 Sonnet Extended Thinking** (`claude-3.7-sonnet-thinking`)
  - **DeepSeek V3**, **LLaMA 3.1 8B**, **LLaMA 3.1 70B**, **LLaMA 3.3 70B**
  - **GPT-4o-mini** (Tier 3 — no Jegham figures; own entry fixes alias bug)
  - **Claude 3.5 Haiku** (Tier 3 — Jegham listed but no figures; own entry fixes alias bug)
  - **Gemini 2.5 Flash** and **Gemini 2.5 Pro** (Tier 3 proxy — no published data; closes pricing.json reverse gap)

- **Energy registry: 8 upgrades from flat → prompt-length-aware** (Tier 1, Jegham):
  `gpt-4-turbo` (Tier 3→Tier 1), `gpt-4.1-mini`, `gpt-4.1-nano`, `o1`, `o3`, `deepseek-r1`, `claude-3.7-sonnet`, `llama-3.1-405b`

- **Pricing.json: complete coverage** — 11 new model entries + 6 backfilled energy-only gaps (o1, o3, deepseek-r1, claude-3.7-sonnet, gpt-4.1-mini, gpt-4.1-nano now have non-null cost estimates)

- **Extended Thinking auto-detection** — When `thinking={"type": "enabled"}` is passed to `anthropic.messages.create()`, Vetch automatically uses the `claude-3.7-sonnet-thinking` registry entry, which reflects the higher measured energy of Extended Thinking mode. Requires no user changes.

- **OpenAI streaming: exact token counts** — Vetch now injects `stream_options={"include_usage": True}` before all OpenAI streaming calls. The existing usage-reading code at `StreamWrapper._process_chunk` already handles it. When available, OpenAI returns exact `prompt_tokens` + `completion_tokens` in the final chunk, eliminating estimated tokens for streaming. Prompt cache credits (`prompt_tokens_details.cached_tokens`) are captured automatically.

- **Two-tier streaming token estimation** (for Anthropic, VertexAI, and OpenAI fallback):
  - **Tier 1 (tiktoken installed):** Per-chunk tiktoken encoding gives ~99% accuracy across all scripts and languages. No buffering.
  - **Tier 2 (tiktoken not installed):** Script-aware char ratio: Japanese (hiragana/katakana >10%) → 1.7 chars/token; CJK/Hangul (>15%) → 1.5 chars/token; English/other → 4.0 chars/token. Previously always used 4.0.

- **Uncertainty floor** — When token counts are estimated (any tier), `energy_uncertainty_pct` is floored at 50% regardless of model tier. The warning message now includes the content type and ratio used.

- **SDK version compatibility warning** — `vetch.instrument()` now emits a `logging.WARNING` if any installed SDK (openai, vertexai) is outside the tested version range. Previously these checks ran but the results were never surfaced.

- **Cache-hit energy discounting** — When `cache_read_tokens > 0`, Vetch applies a `CACHE_READ_ENERGY_FACTOR = 0.15` discount to cached input tokens (cache reads skip KV-cache recomputation, using ~15% of standard prefill energy). The new `cache_energy_saving_wh` field in `InferenceEvent` shows the Wh saved vs. the uncached baseline (Tier 2 estimate, ±100%). No user changes required — cache token counts are already captured from Anthropic and OpenAI usage payloads.

- **OTel Extended Thinking transparency** — `_export_event_sync()` now sets `vetch.thinking_mode = True` on the span when the resolved model ends with `-thinking`. Enables filtering on thinking vs. standard inference in APM dashboards (Datadog, Honeycomb, Grafana Tempo). Also adds `vetch.cache_energy_saving_wh` span attribute when a cache saving is present.

### Fixed

- **Alias corrections:**
  - `gpt-4o-mini` → `gpt-4o-mini` (was `gpt-4o`)
  - `gpt-4o-mini-2024-07-18` → `gpt-4o-mini` (was `gpt-4o`)
  - `claude-3-5-haiku`, `claude-3-5-haiku-20241022`, `claude-3-5-haiku-latest`, `claude-haiku-3-5` → `claude-3.5-haiku` (was `claude-3-haiku`)
  - Removed `deepseek-chat` alias (was ambiguously V2 on some providers — use `deepseek-v3-0324` for versioned access)

---

## [0.2.3] - 2026-03-19

### Fixed
- **[CRITICAL] Streaming Calls Not Tracked Under instrument()** (Streaming Auto-Context Bug)
  - Problem: `instrument()` correctly tracked non-streaming calls (fixed in v0.2.2) but streaming calls (`stream=True`) still silently dropped events. `StreamWrapper._capture_to_context()` checked `if ctx is None: return`, finding no active context and doing nothing.
  - Root Cause: The v0.2.2 fix used `auto_context_for_instrumented_call` at call-time for non-streaming responses. For streams, context creation needed to happen at stream-completion time (inside `StreamWrapper`), not at API-call time.
  - Solution: Updated `StreamWrapper._capture_to_context()` in OpenAI, Anthropic, and Vertex AI providers to use `auto_context_for_instrumented_call` when no active context is found at stream exhaustion.
  - Affected providers: OpenAI (sync + async), Anthropic (sync + async), Vertex AI (sync + async). Google GenAI was already correct.
  - No double-emission: `auto_context_for_instrumented_call` is a no-op when a manual `wrap()` context is active, preventing duplicate events.
  - Test coverage: Added `tests/test_streaming_instrument.py` with 11 cases covering all providers, error paths, and the no-double-emit guarantee.

- **Failing test: `test_instrument_genai_module_returns_false_when_not_installed`**
  - `google-genai` is installed in the dev/CI environment, so the test needed to mock the import rather than assume it was absent.
  - Fixed with `patch.dict(sys.modules, {"google.genai": None})` + reset of `_module_instrumented` state.

### Added
- **VETCH_ENDPOINT: First-Class HTTP Output**
  - `VETCH_ENDPOINT=https://your-endpoint.example.com/ingest` now wires up the HTTP emitter unconditionally — no `VETCH_ENABLE_REMOTE=true` flag required.
  - `VETCH_API_KEY=...` sends `Authorization: Bearer {key}` on every POST. Leave unset for internal/firewall-protected endpoints that don't require auth.
  - Legacy `VETCH_OUTPUT=https://...` still works but now prints a hint to use `VETCH_ENDPOINT` instead.
  - Rate-limited error logging in `HttpHandler`: connection failures log at most once per minute to stderr (previously silent).

- **`vetch.configure_http_endpoint(url, api_key=None)`** — programmatic alternative to `VETCH_ENDPOINT` env var. Useful for multi-destination routing or dynamic configuration.

- **QUICKSTART-OUTPUT.md** — new guide covering all output destinations: local stderr, internal HTTP endpoints, OTLP stacks, file output, multi-destination routing, and green routing with `get_cleanest_region()`.

- **METHODOLOGY.md: SDK Instrumentation Model section**
  - Documents that `instrument()` is production-ready as of v0.2.2.
  - Explains the auto-context lifecycle for both non-streaming and streaming calls.
  - Clarifies the `wrap()` vs `instrument()` relationship and priority.

### Changed
- `HttpHandler` now accepts `api_key: str | None = None` parameter (backward compatible).
- `_configure_logging()` checks `VETCH_ENDPOINT` before `VETCH_OUTPUT` on module import.

## [0.2.2] - 2026-03-16

### Fixed
- **[CRITICAL] instrument() Not Tracking Calls Without wrap()** (Auto-Context Creation Bug)
  - Problem: `instrument()` patched LLM SDK clients but didn't automatically track calls. Users still needed explicit `wrap()` context managers, contradicting the docstring promise of "All calls are now automatically tracked!"
  - Root Cause: All provider wrappers checked `if ctx is None: return` and skipped tracking when no context was active. Since `instrument()` didn't create automatic contexts, tracking never happened.
  - Solution: Implemented automatic context creation in all provider wrappers. When no active context exists, wrappers now auto-create an implicit `VetchContext` using defaults from `instrument(region=..., tags=...)`
  - Changes:
    - Added `_default_tags` storage in `vetch/__init__.py` to store tags from `instrument()`
    - Added `get_default_tags()` function to retrieve stored default tags
    - Updated all provider wrappers to auto-create contexts when `ctx is None`:
      - Google GenAI: `_WeakMethodWrapper`, `_WeakAsyncMethodWrapper`, `_WeakEmbedWrapper`
      - OpenAI: `_after_create()`, `_on_create_error()`, `_after_embeddings_create()`, `_on_embeddings_error()`
      - Anthropic: `_after_create()`, `_on_create_error()`
      - Vertex AI: `_after_generate()`, `_on_generate_error()`
      - Azure OpenAI: Automatically covered (uses OpenAI's patching logic)
  - Impact: `instrument()` now works as documented - truly automatic tracking without needing `wrap()`
  - Backward Compatible: Manual `wrap()` usage continues to work perfectly, with tags merging correctly
  - Test Coverage: Added comprehensive integration tests in `tests/test_auto_context.py`

## [0.2.1] - 2026-03-09

### Added
- **VETCH_ENABLED Environment Variable**: Complement to VETCH_DISABLED for more intuitive control
  - Set `VETCH_ENABLED=false` to disable tracking (equivalent to `VETCH_DISABLED=true`)
  - Default: `true` (tracking enabled)
  - Priority: `VETCH_DISABLED` takes precedence over `VETCH_ENABLED` for backward compatibility
  - Example: `export VETCH_ENABLED=false && python app.py`

- **Thread-Safe Instrumentation**: Added `threading.Lock` to `instrument()` function
  - Prevents race conditions during concurrent initialization
  - Safe for multi-threaded applications calling `instrument()` at startup
  - No performance impact on single-threaded usage

- **Examples Directory**: Added comprehensive auto-instrumentation examples
  - `examples/auto_instrument_example.py`: Demonstrates one-line setup for all providers
  - `examples/opentelemetry_example.py`: Full OTel integration guide
  - `examples/README.md`: Documentation for examples directory
  - Shows Google GenAI, OpenAI, and Anthropic usage patterns

### Changed
- **Instrumentation Documentation**: Updated docstrings to clarify thread-safety and env var behavior
- **Kill Switch Feedback**: Made disabled tracking message opt-in via `VETCH_VERBOSE=true`
  - Previously: Message printed to stderr on every import when `VETCH_DISABLED=true` or `VETCH_ENABLED=false`
  - Now: Silent by default, only prints when `VETCH_VERBOSE=true` (opt-in for debugging)
  - Rationale: Reduces noise for "quiet" CLI tools and production environments

### Performance
- **Config Module Hotpath Optimization**: Reduced overhead in `get_config()` and `_normalize_region()`
  - Lazy imports for optional dependencies to reduce import-time cost
  - Cached config values to avoid repeated environment variable lookups
  - ~15-20% reduction in overhead for high-throughput scenarios

- **Google GenAI Provider Optimization**: Reduced method call overhead using weak references
  - Eliminated circular reference patterns that prevented garbage collection
  - Used `__slots__` in wrapper classes to reduce memory footprint
  - Faster patching/unpatching with fewer dict operations

### Fixed
- **[CRITICAL] Memory Leak in Google GenAI Provider** (Issue #1): Fixed circular references in client patching
  - Problem: Closures capturing client references created GC cycles (`client → method → closure → client`)
  - Solution: Implemented `_WeakMethodWrapper` classes using `weakref.ref(client)` to break cycles
  - Impact: Long-running services with frequent client instantiation no longer leak memory

- **[CRITICAL] Reasoning Tokens Not Tracked** (Issue #2): Added support for extended thinking models
  - Problem: Gemini 2.0 Flash Thinking and similar models return `thought_token_count` in usage metadata
  - Solution: Extract reasoning tokens as separate modality in usage dict (`usage["reasoning"]`)
  - Impact: Reasoning tokens can be 10x+ the visible output and were previously uncounted

- **[CRITICAL] Azure OpenAI Region Inference Failing** (Issue #3): Fixed URL parsing for custom domains
  - Problem: `extract_region_from_azure_endpoint()` assumed standard `*.openai.azure.com` format
  - Solution: Added fallback to deployment location header and better error handling
  - Impact: Azure deployments with custom domains now correctly infer region for carbon calculation

- **[CRITICAL] Session Aggregation Race Condition** (Issue #4): Thread-safe session metric accumulation
  - Problem: Multiple threads calling `session.add_event()` simultaneously could corrupt totals
  - Solution: Added `threading.Lock` around all session metric updates
  - Impact: Multi-threaded applications with sessions no longer risk incorrect cost/energy totals

- **[BUG] OpenAI Streaming Chunks Missing Token Counts** (Issue #5): Defensive access for partial chunks
  - Problem: Some streaming chunks lack `usage` field entirely, causing AttributeError
  - Solution: Use `getattr(chunk, "usage", None)` with None fallback
  - Impact: Streaming no longer crashes on partial chunks from OpenAI's API

- **[BUG] Anthropic Cache Tokens Miscounted** (Issue #6): Correct field names for prompt caching
  - Problem: Used `cache_read_input_tokens` instead of correct `cache_read_tokens` field name
  - Solution: Updated field extraction to match Anthropic API v2025-01 specification
  - Impact: Cache hit tracking now accurate for Anthropic Claude models

- **Type Checking**: Resolved all mypy strict mode errors and ruff linting issues
  - Added missing type annotations for Python 3.9 compatibility
  - Fixed Union type syntax (use `Union[X, Y]` instead of `X | Y` for Python 3.9)
  - Removed unused imports and variables flagged by ruff

- **Provider Wrapper __slots__**: Fixed dynamic attribute assignment in wrapper classes
  - Problem: Wrapper classes used `__slots__` but attempted to set `vetch_patched` attribute dynamically
  - Solution: Added `"vetch_patched"` and `"_vetch_original"` to `__slots__` tuples
  - Affected providers: Anthropic, OpenAI, VertexAI
  - Impact: Provider patching tests now pass consistently (fixed 5 failing integration tests)

- **Version Consistency**: Synchronized `__init__.py` version with `pyproject.toml`

## [0.2.0] - 2026-03-08

### Added - Google GenAI Provider
- **Google GenAI SDK Support**: Native integration with `google-genai` Python SDK
  - Automatic tracking for `client.models.generate_content()` (sync and async)
  - Embedding support via `client.models.embed_content()`
  - Streaming response tracking
  - Model name normalization (strips "models/" prefix and version suffixes like "-001")
  - Thread-safe per-client patching using WeakKeyDictionary
  - Privacy guarantee: only reads usage metadata, not prompt/completion content
  - Install with: `pip install vetch[genai]`
  - Example:
    ```python
    import google.genai as genai
    import vetch

    vetch.instrument()  # Auto-instruments all GenAI clients
    client = genai.Client(api_key="...")
    response = client.models.generate_content(model="gemini-2.0-flash", contents="Hello!")
    # Events automatically emitted with energy/carbon/cost
    ```

### Added - OpenTelemetry Semantic Conventions Exporter
- **GenAI Semantic Conventions**: Export vetch events as OpenTelemetry spans
  - Follows OpenTelemetry GenAI semantic conventions (v1.28+)
  - Required attributes: `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.*`
  - Custom vetch attributes: `vetch.cost.*`, `vetch.energy.*`, `vetch.carbon.*`, `vetch.water.*`
  - Auto-export mode: `vetch.configure_otel_export(enabled=True)` automatically exports on context exit
  - Manual export: `vetch.export_event_as_span(event, tracer=tracer, parent_span=parent)`
  - Graceful degradation if `opentelemetry-api` not installed
  - Install with: `pip install vetch[opentelemetry]`
  - Example:
    ```python
    import vetch
    from opentelemetry import trace

    vetch.configure_otel_export(enabled=True)  # Enable auto-export

    with vetch.wrap() as ctx:
        response = client.chat.completions.create(...)
    # Span automatically created with GenAI semantic conventions + vetch attributes
    ```

### Added - Energy Calibration Tools
- **Calibration Methodology Framework**: Reverse-engineer per-token energy from aggregate measurements
  - `calibrate_from_measurement()`: Derive Wh/1k values from published datacenter measurements
  - `validate_calibration()`: Verify calibrated values reproduce original measurements
  - `explore_scenarios()`: Generate multiple calibration scenarios for uncertainty analysis
  - `propagate_to_family()`: Apply efficiency ratios to related models
  - `format_calibration_table()`: Generate markdown tables for documentation
  - Comprehensive documentation with formula derivations, assumptions, and validation
  - Example:
    ```python
    from vetch.tools.calibration import calibrate_from_measurement

    result = calibrate_from_measurement(
        measurement_wh=0.24,  # Google's published measurement
        pue=1.10,             # Google's PUE
        median_tokens=800,    # Estimated median prompt size
    )
    print(f"Input: {result.energy_per_1k_input:.3f} Wh/1k")
    print(f"Output: {result.energy_per_1k_output:.3f} Wh/1k")
    ```
- **Gemini Calibration Document**: `GEMINI_CALIBRATION.md` with full methodology
  - Anchored to Google's 0.24 Wh per median Gemini Apps prompt (August 2025)
  - 5 scenarios: Very Conservative (40 tokens), Conservative (400), Moderate (800), Optimistic (1600), Maximum (4000)
  - Transparent assumptions: PUE 1.10, output/input ratio 3:1, 50/50 input/output split
  - Validation script reproduces Google's measurement with <1% error
  - Critical assumption: Median prompt = 1,600 tokens (if actually 400 tokens, values would be 4x higher)

### Added - Tiered Pricing
- **Tiered Pricing Support**: Accurate cost calculation for long-context models
  - Gemini Pro models charge 2x per token for >128k context window
  - Schema: `tier_threshold` (128000) and `tier_multiplier` (2.0) in `pricing.json`
  - Implementation uses multiplier approach (single source of truth, no duplicate values)
  - Example: Gemini 2.5 Pro charges $1.25/M for ≤128k tokens, $2.50/M for >128k tokens
  - Calculation:
    ```python
    # For 200k input tokens:
    # First 128k: 128 × $1.25/M = $0.16
    # Remaining 72k: 72 × $2.50/M = $0.18
    # Total: $0.34
    ```

### Changed
- **Test Coverage**: Maintained at 69%
  - Added tests for calibration tool (19 tests)
  - Added tests for tiered pricing (5 tests)
  - Added tests for Google GenAI provider (9 tests)
  - Added tests for OpenTelemetry exporter (5 tests)
  - 764 tests passing total

### Fixed
- Google GenAI provider now correctly extracts model names with version suffixes (e.g., "models/gemini-1.5-pro-001" → "gemini-1.5-pro")

## [0.1.8] - 2026-03-03

### Added - Schema v2 & Multimodal Support
- **Schema Version 2**: Upgraded event schema to support multimodal inference (images, audio, video)
  - New `ImageUsage` TypedDict with `input_tokens`, `output_tokens`, `image_count`, `total_pixels`
  - New `AudioUsage` TypedDict with `input_tokens`, `output_tokens`, `input_seconds`, `output_seconds`
  - Updated `Usage` container to support `text`, `image`, and `audio` modalities
  - New `multimodal` flag on InferenceEvent (True if request includes non-text modalities)
  - Image token extraction from OpenAI GPT-4 Vision API responses (`prompt_tokens_details.image_tokens`)
- **Distributed Tracing Support**: W3C trace propagation for APM integration
  - New fields: `trace_id`, `span_id`, `parent_span_id` for correlation with Datadog, New Relic, etc.
  - Ready for OpenTelemetry context extraction (implementation in future release)
- **Water Usage Tracking**: Datacenter cooling water consumption
  - New `estimated_water_l` field (liters per inference)
  - Provider-specific WUE (Water Usage Effectiveness) values
  - Google: 1.1 L/kWh (efficient water-free cooling), Azure: 1.7 L/kWh, AWS: 2.2 L/kWh (evaporative cooling)
  - Formula: `water_l = (energy_wh / 1000) * WUE`
- **Embodied Carbon Tracking**: Hardware manufacturing emissions
  - New `embodied_carbon_g` field (gCO2e from GPU/TPU manufacturing amortized over lifetime)
  - Factor: ~0.075 gCO2e per 1k tokens (based on Patterson et al. 2021)
  - Adds 5-15% to operational carbon footprint
- **Degraded Mode Indicator**: Transparency on tracking accuracy
  - New `tracking_degraded` flag (True if tracking is active but with reduced accuracy)
  - Triggers when: unknown model, token estimation used, Tier 3 energy data, or Tier 3 PUE
  - Helps users understand confidence level of metrics
- **Batch API Support Placeholders**: Schema fields for OpenAI Batch API (50% pricing discount)
  - New `is_batch` flag (ready for implementation)
  - New `is_embedding` flag for embedding generation (different energy profile than text generation)
- **Time-of-Day Carbon Tracking**: Hourly grid intensity awareness
  - New `grid_intensity_time_of_day` flag (False for now, True when hourly data implemented)
  - Schema ready for future hourly Electricity Maps integration

### Added - Security & Compliance
- **Tag Cardinality Limits**: DoS protection for unbounded tag values
  - Default limit: 1000 unique values per tag key
  - Configurable via `vetch.set_tag_cardinality_limit(limit)`
  - Automatic warnings when limit exceeded, values filtered
- **Tag Allowlist Mode**: Strict security filtering for sensitive environments
  - `vetch.set_tag_allowlist(['team', 'env', 'service'])` for whitelisting
  - Non-allowlisted tags filtered with warnings
  - Prevents accidental leakage of PII or sensitive data via tags
- **Sensitive Tag Redaction**: PII protection via SHA256 hashing
  - `vetch.set_redacted_tags(['user_email', 'customer_id'])` to hash sensitive values
  - Redacted values shown as `redacted-{hash8}` in logs and exports
  - Prevents accidental PII leakage (GDPR/CCPA compliance)
- **Circuit Breaker for Remote Registry**: Prevents hammering GitHub on repeated failures
  - Opens after 3 consecutive fetch failures
  - 5-minute timeout before retry
  - Exponential backoff with jitter
  - New properties: `circuit_breaker_open`, `failure_count` for diagnostics
- **Registry Signature Verification**: Supply chain attack prevention
  - SHA256 checksum validation for registry files
  - Opt-in via `VETCH_REGISTRY_VERIFY_SIGNATURES=true`
  - Loads `checksums.json` from remote, verifies all downloads
  - Logs errors on mismatch, rejects updates (possible supply chain attack)
- **SSRF Protection**: Validates registry URLs to prevent internal network access
  - Blocks private/internal IPs (10.x, 192.168.x, 127.x, link-local, etc.)
  - Only allows http/https schemes (blocks file://, ftp://, etc.)
  - DNS resolution validation before fetch

### Added - Observability
- **Configurable OTLP Queue Size**: Tunable export buffer for high-throughput environments
  - Environment variable: `VETCH_EXPORT_QUEUE_SIZE` (default: 1000)
  - Prevents memory growth in high-traffic services
  - Events dropped if queue full (backpressure documented)

### Added - Framework Integrations
- **Native LangChain Callback Handler**: First-class LangChain integration
  - `from vetch.integrations.langchain import VetchCallbackHandler`
  - Automatic tracking for all LLM calls in chains, agents, and LCEL pipelines
  - Aggregates metrics across multiple calls: `handler.total_cost`, `handler.total_energy_wh`
  - Session support for distributed tracing across LangChain chains

### Fixed
- Tag validation now happens after allowlist filtering (correct order)
- SSRF validation prevents registry from resolving private hostnames

## [0.1.7] - 2026-03-03

### Added
- **Provider-Specific PUE**: Auto-detection of datacenter efficiency from cloud provider sustainability reports
  - Google Cloud (Vertex AI): 1.10 PUE (2023 average)
  - Microsoft Azure (OpenAI): 1.12 PUE (2024 newest generation)
  - AWS (Anthropic, Bedrock): 1.15 PUE (2024 global average)
  - Fallback to 1.20 for unknown providers (industry average)
  - Model-based provider inference (gpt-4 → Azure, claude → AWS, gemini → Google)
  - New event fields: `pue`, `pue_tier`, `pue_source` for transparency
  - PUE tier system: Tier 1 (known value from vendor or user config), Tier 3 (default fallback)
- **Tier 1 Energy Data from Jegham et al. (2025)**: First large-scale hardware measurements in commercial datacenters
  - **GPT-4o upgraded to Tier 1** with non-linear model (short/medium/long prompt awareness)
  - **New reasoning models**: o1 (12.1 Wh), o3 (21.4 Wh), DeepSeek-R1 (29.0 Wh) - 40-100x more energy than efficient models
  - **GPT-4.1 nano**: Most efficient model (0.271 Wh per medium prompt)
  - **Claude-3.7 Sonnet upgraded to Tier 1** (2.781 Wh, highest eco-efficiency among large models)
  - Source: arXiv:2505.09598 "How Hungry is AI? Benchmarking Energy, Water, and Carbon Footprint of LLM Inference"
- **Non-Linear Energy Model**: Prompt-length-aware coefficients capture efficiency gains for longer prompts
  - Short prompts (<1k tokens): Higher per-token cost due to fixed overhead
  - Medium prompts (1k-5k tokens): Baseline for typical usage
  - Long prompts (>5k tokens): ~6x more efficient per-token due to amortization
  - Automatic category selection based on total token count
- **Updated METHODOLOGY.md**: Comprehensive 2025 research citations, non-linearity explanation, data provenance
- **Backward Compatibility**: `calculate_carbon()` supports legacy `pue` parameter (alias for `pue_override`)

### Changed
- **Carbon calculation now returns tuple**: `(carbon_g, pue, pue_tier, pue_source)` instead of just `carbon_g`
  - Provides full transparency on PUE assumptions used in carbon calculations
  - Breaking change for direct callers of `calculate_carbon()` (use tuple unpacking)
- **Language clarity**: "±10x uncertainty" changed to "order of magnitude uncertainty" for Tier 3 estimates
- **Remote registry is now opt-in**: Set `VETCH_REGISTRY_REMOTE=true` to enable (disabled by default)
  - Prevents silent accuracy regressions when bundled registry is newer than remote
  - Will be re-enabled by default in 0.1.8 once remote registry is synced with Tier 1 data

### Fixed
- User-configured PUE (`VETCH_DEFAULT_PUE`) now correctly classified as Tier 1 (known value) instead of Tier 0

## [0.1.6] - 2026-02-25

### Added
- **Session Aggregation**: `vetch.Session()` for grouping multi-call agentic workflows
  - Hierarchical sessions with parent/child nesting
  - Distributed propagation via HTTP headers (`X-Vetch-Session-Id`)
  - Thread-safe metric accumulation (energy, cost, carbon, tokens)
  - Memory safety: `max_calls` limit (default 10,000) prevents OOM in runaway loops
  - Metadata set caps (100 unique models/providers tracked)
  - Cache metrics: `total_cache_read_tokens`, `total_cache_creation_tokens`
  - `session_id` field in inference events links calls to sessions
- **Azure OpenAI Provider**: Auto-detected via `vetch.instrument()`
  - Region inference from Azure endpoint URLs
  - Full sync/async/streaming support
- **Dynamic Registry**: Remote registry updates from GitHub without SDK upgrade
  - Local `.vetch/` overrides for custom energy/pricing values
  - Offline mode via `VETCH_REGISTRY_PATH` for air-gapped environments
  - `vetch registry freeze` CLI command for CI/CD
- **`vetch status` CLI**: Check configuration, environment, and provider detection
- **`vetch dashboard` CLI**: Export pre-built Grafana dashboard template
- **Cache-Aware Pricing**: Cost calculation now applies cache read discounts and creation premiums
- **Alert Cooldown**: `alert_cooldown_seconds` parameter on `set_budget()` prevents alert flooding
- **Price Multiplier**: `price_multiplier` parameter on `wrap()` and `awrap()` for discount pricing
- **Quiet Mode**: `emit=False` parameter on `wrap()` and `awrap()` (metrics available in `ctx.event`)
- **Python 3.13**: Added to CI test matrix
- **Bandit Security Scanning**: Added to CI pipeline

### Changed
- **PUE default changed from 1.1 to 1.2** (aligned with cloud provider averages: Google 1.09, AWS 1.14, Azure 1.12)
- **Energy registry values no longer include PUE** — PUE is applied once in `calculate_carbon()` only. Previous versions double-counted PUE (baked into registry via 1.3x multiplier AND applied in carbon calculation).
- Registry system multiplier reduced from 1.3x to 1.2x (hardware overhead only)
- `vetch.wrap()` and `vetch.awrap()` now expose `price_multiplier` and `emit` parameters
- OTLP service version now uses `__version__` instead of hardcoded string

### Fixed
- **Cache-aware pricing was not connected**: `calculate_cost()` now receives cache token counts from the wrapper (was silently ignoring them)
- **Atomic uninstrumentation**: Provider teardown now restores per-client methods under lock before restoring `__init__` (prevents race condition)
- URL parameter injection: region parameter in Electricity Maps API calls now URL-encoded
- Path traversal prevention in `VETCH_OUTPUT` file paths
- Restrictive file permissions on SQLite database (`0o600`) and directory (`0o700`)
- MagicMock guard on cache token values from captured calls

### Removed
- `pue_overrides.json` (was never loaded by any code — dead since v0.1.3)

## [0.1.5] - 2026-02-23

### Added
- **Budget Alerts** (warn-only): `set_budget()`, `on_budget_alert()`, `get_budget_status()`
  - Configure thresholds for cost, energy, or carbon
  - Alerts logged to stderr and fire callbacks
  - **Never blocks inference** - fail-open by design
  - Thread-safe: accumulation protected by `threading.Lock`
  - Bounded memory: warning deduplication uses LRU with max 1000 keys
  - Environment variables: `VETCH_BUDGET_COST_USD`, `VETCH_BUDGET_ENERGY_WH`, `VETCH_BUDGET_CARBON_G`
- **OTLP Export**: `configure_otlp_export()` for Datadog, Honeycomb, Grafana, Jaeger
  - Export spans and metrics to any OTLP-compatible backend
  - **Non-blocking**: uses background thread queue (max 1000 events)
  - Auto-configure via `VETCH_OTEL_EXPORT=true` and `OTEL_EXPORTER_OTLP_ENDPOINT`
  - Metrics: `vetch.energy_wh`, `vetch.carbon_g`, `vetch.cost_usd`, `vetch.requests_total`
- **Global Instrumentation**: `vetch.instrument()` for zero-code integration
  - Auto-patches OpenAI, Anthropic, and Vertex AI clients at import time
  - Works with LangChain, LlamaIndex, and other frameworks
- **Prompt Cache Detection**: Track cache hits for Anthropic and OpenAI
  - New event fields: `cache_read_tokens`, `cache_creation_tokens`, `cache_hit`
  - Attached to OTel spans when available
- **Grafana Dashboard Template**: `vetch/dashboards/grafana_vetch.json`
  - Pre-built panels for cost, energy, carbon, and request rate
  - Export via `vetch dashboard --export grafana` (CLI coming in v0.1.6)

### Changed
- OTLP integration now supports both span decoration (existing) and full export (new)
- Budget fields in events (`budget_exceeded`, `budget_cost_usd`, etc.) now populated
- Budget `window` parameter now only accepts `"request"` or `"session"` (removed misleading `"hour"`/`"day"` options that had no implementation)

### Environment Variables (New)
| Variable | Purpose | Default |
|----------|---------|---------|
| `VETCH_BUDGET_COST_USD` | Per-request cost alert threshold | (none) |
| `VETCH_BUDGET_ENERGY_WH` | Per-request energy alert threshold | (none) |
| `VETCH_BUDGET_CARBON_G` | Per-request carbon alert threshold | (none) |
| `VETCH_BUDGET_SESSION_COST_USD` | Session-wide cost alert threshold | (none) |
| `VETCH_OTEL_EXPORT` | Enable automatic OTLP export | `false` |
| `VETCH_OTEL_SERVICE_NAME` | Service name for OTLP export | `vetch` |

## [0.1.4] - 2026-02-23

### Added
- `emit=False` parameter for quiet mode (no JSON output to stderr)
- `vetch quickstart` CLI command with usage examples
- Package metadata with GitHub URLs (homepage, issues, repository)
- Expanded model aliases (claude-3-5-sonnet, gemini-flash-2.0, llama3.1-405b, etc.)

### Changed
- Default output changed to `none` (quiet by default). Set `VETCH_OUTPUT=stderr` for JSON.
- FutureWarning for experimental modules now only triggers on first use
- Improved CLI messaging for `vetch report` when storage is disabled

### Fixed
- Model alias mismatch: `claude-3-5-sonnet` now correctly maps to registry (was 12x overestimate)
- Reduced warning spam from experimental modules (storage, ci, calibrate)
- Multi-client patching: each client's original method now stored separately (was losing first client's original)
- Thread-safe patching: added locks to prevent race conditions in multi-threaded environments
- Context isolation: `_cleanup_patches` now only unpatches clients from its own context (was breaking other threads)
- Streaming fragility: defensive access for `chunk.choices` in OpenAI provider

## [0.1.3] - 2026-02-19

### Added
- `energy_uncertainty_pct` field in events (20/50/100/1000 for tiers 0-3)
- MoE active parameter accounting in energy registry (fixes ~300% overestimation)
- Architecture metadata (`architecture`, `total_params_b`, `active_params_b`, `quantization`)
- Provider-specific PUE table (removed in v0.1.6 — was never wired up)
- Expanded model aliases for Claude 3.5, Gemini 2.0, Llama 3.1

### Changed
- Energy estimates now based on active parameters for MoE models
- CLI output shows uncertainty as percentage or "order of magnitude" for Tier 3

## [0.1.2] - 2026-02-18

### Added
- Live API examples in demo.ipynb for OpenAI, Anthropic, and Vertex AI providers

### Changed
- Thread-safe file locking using non-blocking fcntl with polling
- Improved retry logic with capped exponential backoff

## [0.1.1] - 2026-02-18

### Added
- Quick Start Colab notebook (`demo.ipynb`) for interactive exploration

## [0.1.0] - 2026-02-18

### Added
- Initial **alpha** release of Vetch SDK
- Core context manager: `wrap()` for energy/carbon/cost tracking
- OpenAI provider wrapper (sync & streaming)
- Vertex AI provider wrapper (sync & streaming)
- Anthropic provider wrapper (sync & streaming)
- Multi-tier grid intensity cache (Memory + File with locking)
- Serverless mode (`VETCH_CACHE_MODE=memory-only`)
- Energy calculation engine with model registry
- Pricing registry for list cost estimation with `price_multiplier` support
- CLI tool: `estimate`, `compare`, `methodology`, `check`, `audit`, `report`
- Token estimation with tiktoken integration and language-aware fallback
- Session statistics: `get_session_stats()` for pattern detection
- Advisory engine: `generate_advisories()` for optimization recommendations
- OpenTelemetry span decoration (attach metrics to existing spans)
- SQLite local storage for historical analysis (experimental)
- GPU calibration module for Tier 0 measurements (experimental)
- CI summary mode for GitHub Actions

### Safety Features
- **Fail-open behavior**: LLM calls always proceed even when Vetch fails
- **Fail-loud diagnostics**: `vetch_warnings` field captures all issues
- **Kill switch**: `VETCH_DISABLED=true` completely disables Vetch
- **Privacy-first**: Zero access to prompt/completion content

### Alpha Limitations
- Timezone-based region inference has ~30% accuracy - use `VETCH_REGION` for production
- Experimental modules (`calibrate`, `storage`, `ci`) marked with `FutureWarning`
- Integration tests require live API credentials
- Energy estimates are Tier 3 (order of magnitude uncertainty) for most models

### Environment Variables
| Variable | Purpose | Default |
|----------|---------|---------|
| `VETCH_REGION` | Grid region for carbon calculation | (inferred) |
| `VETCH_OUTPUT` | Output target: `stderr`, `none`, or file path | `none` |
| `VETCH_DEFAULT_PUE` | Power Usage Effectiveness multiplier | `1.1` |
| `VETCH_CACHE_MODE` | Set to `memory-only` for serverless | (file-based) |
| `VETCH_DISABLED` | Set to `true` to disable all tracking | `false` |
| `ELECTRICITY_MAPS_API_KEY` | API key for live grid data | (optional) |

### Dependencies
- **Runtime**: Zero dependencies (stdlib only)
- **Optional**: `openai>=1.0,<2.0`, `google-cloud-aiplatform>=1.0`, `tiktoken>=0.5.0`
- **Test**: `pytest>=7.0`, `hypothesis>=6.0`
