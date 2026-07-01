# Capability observability: build plan for Cursor

> **Status (v0.10.0):** Implemented. Checkboxes below are marked complete; see
> [capability-observability-plan.md](capability-observability-plan.md) for the
> living spec.

Execution checklist for [capability-observability-plan.md](capability-observability-plan.md) (the spec). This file says *how to build it, in order*; the spec says *what and why*. When they disagree, the spec wins; fix this file.

## How to use this

- One PR per section below. Land in order: each depends on the previous.
- Slices 2+ start only after the [pre-flight read](capability-observability-plan.md#pre-flight-read-de-risk-the-approach) confirms whether the patched-SDK path or the manual `capture()` path is primary. If `tools=` is rarely visible in stored events, reorder slice 8 (manual capture) ahead of slice 2.
- Fields default to `None` until their extraction slice lands, so every slice is shippable on its own.

### Commands (must pass before each PR is done)

```bash
pytest tests/ -q                       # full suite; or tests/test_capabilities.py -q while iterating
ruff check src/ tests/                 # lint
ruff format --check src/ tests/        # formatting
mypy src/vetch                         # types — src/vetch is py.typed, keep it clean
pre-commit run --all-files             # ruff + mypy + bandit + detect-secrets gate
```

Add a CHANGELOG.md entry under a new `## [0.10.0]` heading in every PR. Bump `version` in `pyproject.toml` to `0.10.0` in PR 1.

## Global guardrails (do not regress these — each was a real bug caught in review)

1. **Cost rate is cache-aware.** Never `total_cost_usd / total_input_tokens`. Use `Σ(estimated_cost_input_usd − estimated_cost_cache_read_usd) / Σ(max(0, input_tokens − cache_read_tokens))`. Fully-cached session → `$0`, not a divide-by-zero. See [Cost formula](capability-observability-plan.md#cost-formula).
2. **Two field families, never merged.** Function tools → `tools_offered`/`tools_invoked`/`tool_call_count`. Non-function → `capabilities_invoked`. No `kind` filtering for the common "which function tools" query.
3. **Extraction runs before stall reroute.** Read `tools=` and model before `apply_stall_action` mutates `kwargs["model"]`.
4. **Fail-open, hot path.** Any extraction error → `None` + optional `vetch_warnings` note. No exceptions as control flow. Memoize offered extraction; do **not** key on `request_fingerprint` (it excludes `tools=`).
5. **Cardinality cap is transport-only.** Feed the rollup the untruncated lists; cap at 64 only on the serialized/OTel copy. Session memory is bounded separately by a per-session cap.
6. **Names are egress.** Route every tool/capability name through `redact_capability_name` before it lands on an event or span. Rollups key on the post-redaction string.
7. **Model map is registry-sourced**, not an inline dict in `capabilities.py`.
8. **OTel aligns to `gen_ai.*` semconv**; `vetch.*` only for waste-specific fields. Array attributes, never CSV.
9. **Privacy:** names + counts only. Never store tool arguments, results, or message content.

---

## PR 1 — Schema and plumbing

**Goal:** every field exists and flows end to end, all `None`. No extraction.

**Files:** `src/vetch/schema.py`, `src/vetch/context.py`, `src/vetch/wrapper.py`, `src/vetch/stats.py`, `pyproject.toml`, `CHANGELOG.md`, `packages/vetch-ai-sdk/src/types.ts`.

- [x] Add `CapabilityRef` TypedDict to `schema.py`.
- [x] Add `tools_offered`, `tools_invoked`, `tool_call_count`, `capabilities_invoked` to `InferenceEvent` (near `finish_reason` / `requested_max_tokens`), all `Union[..., None]`.
- [x] Add the same four to `CapturedCall` (`context.py`) and to `TrackingContext.capture()` params, default `None`.
- [x] Thread them through `wrapper._emit_event` from `captured` into the event dict.
- [x] `SessionStats.update()`: accept the new fields (store nothing yet) and add cost accumulators `total_effective_input_usd`, `total_billable_input_tokens` summing `estimated_cost_input_usd − estimated_cost_cache_read_usd` and `max(0, input_tokens − cache_read_tokens)`. Guard mutable state with the existing lock pattern.
- [x] OTel: register semconv-aligned attribute *stubs* (`gen_ai.tool.*` arrays + `vetch.wasted_tool_schema_tokens`) in `otel.py` / `exporters/opentelemetry.py`, emitting only when present.
- [x] TS: mirror the four fields in `packages/vetch-ai-sdk/src/types.ts`.
- [x] Bump `pyproject.toml` version to `0.10.0`.

**Data-contract check:** confirm parquet/BigQuery/other columnar sinks in `storage.py` and `exporters/` accept new nullable columns (write a round-trip test with the fields set).

**Acceptance:** `mypy` clean; an event constructed with the new fields serializes/deserializes and round-trips through storage; suite green.

---

## PR 2 — Kind A extraction, non-streaming (OpenAI + Azure + Anthropic)

**Goal:** offered/invoked function tools captured for non-streamed calls.

**Files:** new `src/vetch/capabilities.py`; `src/vetch/providers/openai.py`, `providers/anthropic.py`; tests.

- [x] `capabilities.py`: `normalize_function_tools` (de-dupe by name, stable sort, wrap as `kind="function"`); `extract_openai_tools_offered/invoked`; `extract_anthropic_tools_offered/invoked`. Fail-open → `None`.
- [x] Per-tool token size: tokenize each offered tool's JSON once, return `dict[name, tokens]`, discard payload. Feeds `wasted_tool_schema_tokens`.
- [x] Memoize offered extraction on `id(tools)` / serialized-payload hash.
- [x] Wire into `openai.py._after_create` (covers Azure via the shared patch) and `anthropic.py` sync finalize, **before** any reroute substitution. Call `redact_capability_name` on every name.
- [x] Set `tool_call_count = len(invoked)`.

**Tests** (`tests/test_capabilities.py`): normalizer; malformed/empty `tools=`; parallel invocations; **recorded real SDK objects** (pydantic types / cassettes), not hand dicts; reroute ordering (with `stall_action="reroute"`, offered tools attributed to the original model).

**Acceptance:** OpenAI + Anthropic non-stream calls produce correct `tools_offered`/`tools_invoked`/`tool_call_count`; redaction honored; no latency regression on the no-tools path (memoization verified).

---

## PR 3 — Kind A streaming (OpenAI + Anthropic)

**Goal:** same capture for streamed responses, or clean `None`.

**Files:** `providers/openai.py`, `providers/anthropic.py` stream wrappers; tests.

- [x] OpenAI: accumulate `delta.tool_calls[]` by `index` until stream completes.
- [x] Anthropic: accumulate `content_block_start` blocks with `type == "tool_use"`, finalize alongside `stop_reason`.
- [x] Stream error mid tool-call → `None`, never partial/wrong names.

**Tests:** streaming accumulation for both; interrupted stream → `None`.

**Acceptance:** streamed tool calls captured; partial streams never emit wrong names.

---

## PR 4 — Kind A remaining providers

**Files:** `providers/genai.py`, `providers/vertexai.py`, `providers/ollama.py`; tests.

- [x] genai/vertexai: `function_declarations` (offered) and `function_call` parts (invoked), non-streaming first.
- [x] ollama: OpenAI-compatible tool shape.

**Acceptance:** each provider's non-stream path captures function tools; streaming for these stays `None` (documented), not wrong.

---

## PR 5 — Kind C derivation + registry model map + config

**Goal:** populate `capabilities_invoked` for model routes; ship a zero-config default map.

**Files:** `src/vetch/registry/` (new `model_capabilities.json` + `PROVENANCE.md` note), `scripts/check_registry_freshness.py`, `src/vetch/wrapper.py`, `src/vetch/config.py`, `src/vetch/capabilities.py`; tests.

- [x] Add `registry/model_capabilities.json` (model id → capability name): `gpt-image-1`/`dall-e-3` → `image`, `tts-1*` → `audio`, `whisper-1` → `transcription`, `text-embedding-*` → `embedding`, Imagen/Veo/Gemini TTS, etc. Document provenance and add it to `check_registry_freshness.py` like `pricing.json`.
- [x] Loader in `capabilities.py` reads the registry map; `configure_capabilities(model_capability_map=...)` merges overrides on top (do not replace defaults).
- [x] `wrapper._emit_event`: derive Kind C refs from `is_embedding`, non-zero `usage.image/audio/video`, and the merged map → `capabilities_invoked` (independent of the `multimodal` boolean). Never merge into `tools_invoked`.
- [x] `config.py`: `configure_capabilities(expected=..., model_capability_map=...)`, re-export from `__init__.py`. (Spec allows separate `set_*` setters if preferred for consistency.)

**Tests:** embedding event; multimodal input; registry-map hit with zero config; override map; unmapped image-gen id → documented false negative.

**Acceptance:** model routes appear in `capabilities_invoked` with no config; freshness CI covers the new registry file.

---

## PR 6 — Session rollup + cache-aware cost

**Goal:** the customer-facing summary numbers.

**Files:** `src/vetch/stats.py`; tests.

- [x] Accumulate under the lock: `function_tools_offered/invoked` sets, `capabilities_invoked` set, `capability_invocation_counts`, `tool_schema_tokens`.
- [x] Per-session cardinality bound (reuse `set_tag_cardinality_limit` precedent); stop accumulating distinct names past it + `vetch_warnings` note.
- [x] `summary()` adds: `function_tools_never_called`, `wasted_tool_schema_tokens`, `wasted_tool_schema_cost_usd` (cache-aware rate from PR 1 accumulators), `declared_capabilities_silent` (per session), `capability_invocation_counts`, `tool_call_event_rate`.

**Tests:** cache-aware cost (fully-cached → `$0`; mixed; zero-billable guard; rate never from `total_cost_usd`); set-difference correctness; per-session bound; concurrent `update()` does not corrupt sets.

**Acceptance:** summary numbers correct and cache-aware; concurrency-safe.

---

## PR 7 — Advisories + audit

**Goal:** TOOL-DEAD-001 (runtime + audit) and CAP-001 (audit-only, windowed).

**Files:** `src/vetch/advisory.py`, `src/vetch/audit_report.py`, `docs/inference-waste-taxonomy.md`, CLI for `vetch audit`; tests.

- [x] Register `TOOL-DEAD-001` (INFO, per-session, reports `wasted_tool_schema_cost_usd`) and `CAP-001` (WARNING, windowed) in the taxonomy doc.
- [x] Runtime (`advisory.py`): TOOL-DEAD-001 from `SessionStats` when per-session thresholds met. **Do not** compute CAP-001 from `SessionStats` (no cross-session state).
- [x] Audit (`audit_report.py`): recompute TOOL-DEAD-001 and per-session Kind C silence from stored events; CAP-001 by scanning across the window.
- [x] Manifest at audit time: persist a manifest snapshot with the session record **or** add `vetch audit --expected-capabilities ...`. Pick one, implement it.
- [x] Thresholds via `set_advisory_thresholds` (`TOOL-DEAD-001`: `min_requests` default 10, `min_offered_tools` default 1; CAP-001 window size).
- [x] Verify `tests/test_audit_report.py` still passes (PREMIUM-001 now has `tool_call_count`); add non-zero cases.

**Acceptance:** TOOL-DEAD-001 fires per session; CAP-001 fires only from audit across the window; `vetch audit` computes silence offline with the manifest supplied.

---

## PR 8 — Manual capture + docs

**Goal:** first-class path for agent frameworks where `tools=` is invisible.

**Files:** `src/vetch/context.py` (already has the params from PR 1 — document and validate them), `README.md`, `QUICKSTART-LOCAL.md`; tests.

- [x] Confirm `TrackingContext.capture()` accepts and forwards `tools_offered`/`tools_invoked`/`tool_call_count`/`capabilities_invoked` from manual callers.
- [x] README / QUICKSTART-LOCAL: documented example for LangGraph/CrewAI/OpenAI-Agents-style integrations. State clearly that `mcp/` is the FinOps server, not this path.

**Acceptance:** a manual `capture()` call with tool metadata produces the same summary output as a patched call.

---

## PR 9 — JS SDK parity

**Files:** `packages/vetch-ai-sdk/src/` middleware + types, vitest tests.

- [x] Populate offered/invoked named lists in middleware (extends existing `tool_call_count`).
- [x] Reconcile codes: keep `TOOL-*` for function tools alongside the existing `TOOL-SPIN-001` / `TOOL-TREADMILL-001`.
- [x] Parity tests mirroring the Python golden cases.

**Acceptance:** TS events carry the same tool fields; parity tests green.

---

## Definition of done (whole feature)

- Spec [success criteria](capability-observability-plan.md#success-criteria) met: zero-config dead-tool list + cache-aware wasted cost; registry-map capabilities; declared-route silence after manifest.
- All nine PRs merged; `pytest`, `ruff`, `mypy`, `pre-commit` green on each.
- CHANGELOG `## [0.10.0]` complete; version bumped.
- No global guardrail regressed (grep the diff for `total_cost_usd /`, `request_fingerprint` in extraction, CSV OTel attrs, un-redacted name paths).
- Deferred items (Kind B, Kind D, Responses API, baselines) untouched and still listed as deferred.
