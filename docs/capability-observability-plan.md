# Capability observability: detecting uncalled tools and silent capabilities

| | |
|---|---|
| **Status** | Implemented (v0.10.0) |
| **Target** | v0.10.0 = **Kind A (function tools) + Kind C (silent model routes) + honest cost** |
| **Problem** | Customers cannot tell which offered tools / wired routes never execute, or what the dead schemas cost |
| **Pre-flight** | One-day [validation read](#pre-flight-read-de-risk-the-approach) over stored events to confirm the patch vs manual-capture path before extraction lands. Not a scope gate (scope is decided); an implementation de-risk. |
| **Related** | [inference-waste-taxonomy.md](inference-waste-taxonomy.md), [how-detection-works.md](how-detection-works.md), [evaluation-follow-through-plan.md](evaluation-follow-through-plan.md) (runtime vs audit dispatch) |

## Problem statement

Customers report: *"some tools are not being called."* In production this is two different failures, both in scope:

1. **Function tools (Kind A).** The model was offered `get_weather`, `refund_order`, etc. via `tools=` but never emitted a `tool_use` / `tool_calls` block. The dead schemas are still re-sent as input every turn, so this is a coverage signal *and* a cost signal.
2. **Silent capability routes (Kind C).** A pipeline step (image gen, embeddings, a dedicated model) was wired but the code path never runs. Not visible as a "declined tool"; it is a missing separate call or modality, detected against a declared roster.

Vetch today records neither the **offered** tool set nor **invoked** capability identifiers on Python events. The audit path already *reads* `tool_call_count` for PREMIUM-001 gating (`audit_report._tool_call_event_rate`), but at plan time **Python `InferenceEvent` did not define or emit that field** — only the Vercel AI SDK middleware did (`packages/vetch-ai-sdk/src/event.ts`). This plan closed that gap and unified both senses under one abstraction. *(Implemented in v0.10.0; `tool_call_count` and the capability fields now ship on Python events.)*

### What this plan does *not* claim

- **Not proof of a bug.** A model that declines to call `refund_order` is often correct. Offered-but-never-called is informational, surfaced as a cost figure, not an alarm.
- **Not an eval.** "In *this* task the model should have called `search` and didn't" is a per-scenario quality failure that an aggregate never-called rollup cannot see.
- **Not full coverage on day one.** Tool lists are only knowable when Vetch sees the provider request (`tools=` in patched SDK hooks). `wrap().capture()` and raw HTTP do not see offered tools unless the caller passes them (see [Capture surfaces](#5-capture-surfaces-manual-capture-and-agent-frameworks)).
- **Out of scope for this document:** where any aggregated, cross-account, or hosted view of this signal lives.

---

## Pre-flight read (de-risk the approach)

Scope is decided; this is not a go/no-go. Before extraction (slice 2) lands, spend a day reading **already-stored events** so the implementation targets the right path:

- **% of events with `tools=` visible today.** If near zero, SDK patches are not the bottleneck and the manual `capture()` path is the primary product, not a fallback. This changes which slice matters most.
- **Offered vs invoked set-difference where visible.** Confirms there is dead-tool waste to report and sizes it.
- **For Kind C:** which model ids / modalities actually appear, so the registry capability map covers the real fleet.

---

## Why this is a Vetch feature, not generic tracing

LangSmith, Langfuse, and Arize already show tool-call traces. Vetch's wedge is **waste and cost**, so the output has to be dollars, not a coverage matrix. The headline:

**Wasted tool-schema cost.** Tool definitions are re-transmitted as input tokens every request. If a session offers 12 tools and the model only ever calls 3, the other 9 schemas are paid-for input on every turn. Vetch reports `wasted_tool_schema_tokens` and `wasted_tool_schema_cost_usd`.

**This number must be cache-aware or it is wrong.** For any serious multi-tool agent the tool/system prefix is the *cached* prefix (Anthropic explicit caching, OpenAI automatic caching), billed at roughly 10 to 50 percent. Full list price per turn would overstate the waste on exactly the agents with the most tools, and a customer reconciling against their bill would catch it. Vetch already tracks cache-aware input pricing per event (`estimated_cost_input_usd`, `estimated_cost_cache_read_usd`, `cache_read_tokens`), and `calculation.py` already splits cached vs billable input ([calculation.py:1274-1277](../src/vetch/calculation.py#L1274)). Reuse it (see [Cost formula](#cost-formula)).

Document it honestly as a **directional estimate**, same caveat Vetch puts on energy and carbon: tiktoken on schema JSON plus cache-boundary assumptions, directionally correct, not for invoice-level reconciliation.

---

## Taxonomy

| Kind | Example | How we learn "available" | How we learn "invoked" | Scope |
|------|---------|--------------------------|------------------------|-------|
| **A — function** | `get_weather` | `tools=` on the request | `tool_calls` / `tool_use` in response | **v0.10.0** |
| **B — builtin** | `web_search`, `code_interpreter`, grounding | Provider-specific response blocks | Same | Deferred |
| **C — model** | `embedding`, `image`, dedicated gen model | Declared manifest + registry map | Derive from event (`is_embedding`, usage modalities, model-id map) | **v0.10.0** |
| **D — agent** | tagged sub-agent / pipeline step | Tags / manifest | Tag presence on events | Deferred (schema-ready) |

- **A** is self-describing wherever instrumentation sees the request; set-difference needs no config.
- **C** needs a declared roster (you cannot infer "expected to fire" from a single call), but ships with a registry-sourced default map so first run is non-empty with zero config.
- **B** excluded: provider shapes diverge and change often. Add on demand.
- **D** cheap later: `kind="agent"` refs from configured tag keys. Schema reserves the shape now.

**Explicitly out of scope (v1):** learned baselines for *"used to fire, stopped"* regressions. Noisy and cold-start prone.

---

## Core abstraction

Represent every capability as a **`CapabilityRef`**, not a bare string.

```python
# schema.py — new TypedDict
class CapabilityRef(TypedDict):
    name: str   # "get_weather", "image", "embedding"
    kind: Literal["function", "builtin", "model", "agent"]
```

**Do not mix kinds in one field.** "Which function tools" is the common query and must not need a runtime `kind` filter. Function tools live under `tools_*`; non-function kinds go in `capabilities_invoked`.

**Per-event fields** (on `InferenceEvent` and `CapturedCall`):

| Field | Meaning |
|-------|---------|
| `tools_offered` | Function tools (`kind="function"`) on *this* request. `None` if unknown. |
| `tools_invoked` | Function tools invoked on *this* response. `None` if unknown. |
| `tool_call_count` | Count of function tool invocations on *this* response (parallel calls count individually). `None` if unknown. |
| `capabilities_invoked` | Kind C (model) refs derived for *this* event; reserved for B/D later. `None` if none. |

**Session rollup fields** (on `SessionStats.summary()`):

| Field | Meaning |
|-------|---------|
| `function_tools_never_called` | Union(`tools_offered`) − union(`tools_invoked`) over the session |
| `wasted_tool_schema_tokens` | Input-token footprint of the never-called function tools |
| `wasted_tool_schema_cost_usd` | `wasted_tool_schema_tokens` × **cache-aware** session input rate (see [Cost formula](#cost-formula)) |
| `declared_capabilities_silent` | `expected_capabilities` − union(`capabilities_invoked`) over the session |
| `capability_invocation_counts` | `dict[str, int]` keyed by `"kind:name"` |

Kind A needs no manifest; Kind C requires `expected_capabilities` for "never fired" detection.

---

## Cost formula

> **Shipped semantics (v0.10.0):** the formula below gives the *per-request* dead-schema cost. The headline `wasted_tool_schema_cost_usd` in `summary()` is the **session total** = per-request cost × `dead_tool_offer_request_count` (the number of requests that offered at least one never-called tool). The per-request figure is exposed as `wasted_tool_schema_cost_per_request_usd`.

The session input rate must come from input cost net of cache reads, not blended totals. The earlier draft used `total_cost_usd / total_input_tokens`, wrong twice: it folds output cost into an "input rate" and ignores cache discounts. Correct, matching the cached/billable split already in `calculation.py`:

```text
effective_input_usd   = Σ( estimated_cost_input_usd − estimated_cost_cache_read_usd )
billable_input_tokens = Σ( max(0, input_tokens − cache_read_tokens) )
session_input_rate    = effective_input_usd / billable_input_tokens        # guard /0
wasted_tool_schema_cost_usd = wasted_tool_schema_tokens × session_input_rate
```

**Prerequisite:** `SessionStats` does not accumulate input cost or cache reads today (`total_cost_usd` is full per-call cost). Add `total_effective_input_usd` and `total_billable_input_tokens` accumulators in `update()`. The rate derives from those, never from `total_cost_usd`.

If `billable_input_tokens == 0` (everything cached), report `wasted_tool_schema_cost_usd = 0` with a `vetch_warnings` note rather than dividing by zero. That is the correct answer: a fully-cached tool prefix costs almost nothing per turn, itself worth telling the customer.

---

## Privacy

Metadata only — same stance as `CapturedCall` and existing diagnostics:

- Record **names** and **counts** only. Never record tool arguments, results, or message content.
- Normalizer must not log or retain raw `tools=` payloads. It extracts `.name` strings and a per-tool token **count** (for `wasted_tool_schema_tokens`), then discards the payload.

**Tool names are a new egress surface, not inherently safe.** Per-user tool factories generate names like `send_email_to_john_at_acme` that can embed PII, flowing into stored events and OTel spans. **Do not reuse `set_redacted_tags` / `set_tag_allowlist` as-is** — those operate on tag dict keys, not arbitrary name strings. Add capability-specific helpers in `capabilities.py`, sharing the existing HMAC key (`VETCH_REDACTION_KEY`, via `redact_tags`):

```python
def redact_capability_name(name: str) -> str:
    """Return ``name`` or ``redacted-<hmac>`` when redaction is enabled."""

def set_redacted_capability_names(names: Iterable[str]) -> None:
    """Exact names to hash before emission (e.g. customer-specific factories)."""
```

- When `VETCH_REDACTION_KEY` is set, hash **all** tool names by default; `set_redacted_capability_names` adds explicit entries.
- Rollups key on the **post-redaction** string so set-difference and counts survive hashing (`redacted-abc123` is stable across events).

---

## Kind C derivation (constraints)

Model-as-capability is **not** fully automatic from today's event fields:

| Signal | Today | Limitation |
|--------|-------|------------|
| `is_embedding` | Set on embedding hooks | Reliable for embedding routes |
| `multimodal` | `usage.image` / `usage.audio` on the event | Reflects **input** image/audio only; excludes `usage.video` and output-only generation |
| Model id | `event.model` | Needs a map for image/TTS/etc. (e.g. `gpt-image-1` → `image`) |

**v1 rule:** Kind C invoked refs are produced in `wrapper._emit_event` from:

1. `is_embedding` → `{"name": "embedding", "kind": "model"}`
2. Non-zero `usage.image` / `usage.audio` / `usage.video` → modality ref in `capabilities_invoked` (independent of the `multimodal` boolean)
3. A `model_capability_map` (model id → capability name) for routes not visible via usage flags

**The map is registry-sourced, not an inline dict.** Classifications (`gpt-image-1` → `image`, Imagen / Veo / Gemini TTS, embedding families) live in registry JSON with `PROVENANCE.md` and the freshness CI already used for `pricing.json`. One source of truth, owned and dated, not a hand-list in `capabilities.py` that rots in two release cycles. It ships populated so first run is non-empty with zero config; `set_model_capability_map` merges customer overrides on top.

**Honest limitation:** output-only generation on an unmapped model id is a false negative. Document in README; do not imply automatic detection of all GenAI routes.

---

## Configuration API

Follow existing `config.py` patterns (`set_advisory_thresholds`, `add_global_tags`); do **not** invent `vetch.configure()`. To avoid setter sprawl, prefer one grouped entry point:

```python
def configure_capabilities(
    *,
    expected: list[str] | None = None,         # ["model:image", "model:embedding"]; kind A ignores this
    model_capability_map: dict[str, str] | None = None,  # merges over the registry defaults
) -> None: ...
```

If kept as separate setters for consistency with the existing module, name them `set_expected_capabilities` and `set_model_capability_map` and re-export from `vetch.__init__.py`. Optional env mirrors only if consistent with existing `VETCH_*` patterns.

---

## Implementation

### 1. Schema and plumbing

**Files:** `schema.py`, `context.py`, `wrapper.py`, `stats.py`

- Add `CapabilityRef` TypedDict.
- Extend `CapturedCall` and `TrackingContext.capture()` with `tools_offered`, `tools_invoked`, `tool_call_count`, `capabilities_invoked` (default `None`).
- Extend `InferenceEvent` with the same four fields (near `finish_reason` / `requested_max_tokens`).
- Thread through `wrapper._emit_event`.
- Extend `SessionStats.update()` for the new fields plus the cost accumulators in [Cost formula](#cost-formula). `update()` mutates shared sets: guard with the same lock pattern the tag limiter uses.
- **Data-contract check:** confirm columnar export sinks (parquet/BigQuery in `storage.py` / exporters) tolerate new nullable columns.

**OTel (align to semconv, do not invent).** The exporter already emits `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.*`. Add GenAI semantic-convention tool attributes (`gen_ai.tool.*` per current semconv) as **arrays**, and use `vetch.*` only for waste-specific fields semconv has no hook for (`vetch.wasted_tool_schema_tokens`, `vetch.tools_never_called`). Array-valued attributes only.

**Parity:** mirror fields in `packages/vetch-ai-sdk/src/types.ts`. TS already ships `tool_call_count` and `TOOL-SPIN-001` / `TOOL-TREADMILL-001`; keep `TOOL-*` for function tools when reconciling.

### 2. Shared extraction helpers (Kind A)

**New module:** `src/vetch/capabilities.py` (keeps providers thin)

```python
def normalize_function_tools(names: Iterable[str]) -> list[CapabilityRef]: ...
def extract_openai_tools_offered(kwargs: dict) -> list[CapabilityRef] | None: ...
def extract_openai_tools_invoked(result) -> tuple[list[CapabilityRef], int] | None: ...
# anthropic, genai, vertexai, ollama analogs
```

Rules:

- Fail-open: parse error → `None`, optional `vetch_warnings` note. No exceptions for normal control flow (hot path).
- De-dupe by `name`, stable sort for deterministic events.
- Support OpenAI-style `{"type":"function","function":{"name":...}}` and Anthropic flat `{"name":...}`.
- `tool_call_count` = len(invoked function refs), not len(offered).

**Hot-path cost.** Memoize offered extraction on `id(tools)` when the list object is stable, or a short hash of the serialized payload when frameworks re-allocate. **Do not use `request_fingerprint`**: it excludes `tools=`.

**Extraction must run before stall reroute.** `apply_stall_action` mutates `kwargs["model"]` for `reroute`. Read `tools=` and the model **before** substitution, or tools get attributed to the substituted model.

**Provider matrix (v1)**

| Provider module | Request hook | Notes |
|-----------------|--------------|-------|
| `providers/openai.py` | `_after_create`, stream wrappers | Covers **Azure OpenAI** (same patch). **Chat Completions only**; OpenAI **Responses API** carries tools differently — out of v1, tracked follow-up. |
| `providers/anthropic.py` | sync + stream finalize | Accumulate `tool_use` blocks like `stop_reason` |
| `providers/genai.py`, `providers/vertexai.py` | non-streaming first | `function_declarations` / `function_call` parts |
| `providers/ollama.py` | chat hook | OpenAI-compat tool shape |

**Streaming (separate slice, highest effort)**

- OpenAI: accumulate `delta.tool_calls[]` by `index` until completion.
- Anthropic: accumulate `content_block_start` with `type=="tool_use"`.
- Error mid tool-call → `None`, never partial wrong names.

### 3. Kind C in wrapper

In `wrapper._emit_event`, after usage/multimodal flags, derive Kind C refs (from `is_embedding`, usage modalities, and the registry + override `model_capability_map`) into `capabilities_invoked`. Function refs stay in `tools_invoked`; never merged.

### 4. Session rollup + cost

**File:** `stats.py` — extend `SessionStats` sets, accumulators, and `summary()` (under the lock):

- `function_tools_offered`, `function_tools_invoked: set[str]`
- `capabilities_invoked: set[tuple[str, str]]`, `capability_invocation_counts: dict[str, int]`
- `tool_schema_tokens: dict[str, int]` (per offered tool → estimated input-token size, counted once, payload discarded)
- `total_effective_input_usd`, `total_billable_input_tokens` (cache-aware rate)

**Bound session memory.** Sets are unbounded for long agents with dynamic names. Apply a per-session cap (reuse `set_tag_cardinality_limit`), stop accumulating past it, note in `vetch_warnings`. The per-event cap in [Cardinality](#cardinality-and-event-size) does not bound the session.

Compute `function_tools_never_called`, `wasted_tool_schema_tokens`/`_cost_usd` (per [Cost formula](#cost-formula)), `declared_capabilities_silent` (per session), and `tool_call_event_rate`.

### 5. Capture surfaces: manual capture and agent frameworks

Not a fallback. Multi-tool agents live in frameworks (LangGraph, CrewAI, OpenAI Agents SDK) where tools do not always pass through a raw provider `tools=`.

**Scope note:** [`mcp/`](../src/vetch/mcp) is Vetch's **FinOps MCP server** (`vetch_estimate`, `vetch_compare`), not instrumentation of a customer's tool roster. Customer "tools not called" is observed when:

1. The framework passes `tools=` through a patched provider client (Kind A extraction — primary), or
2. The integrator supplies metadata via `TrackingContext.capture()` (manual path — required for opaque wrappers and raw HTTP).

- `TrackingContext.capture()` accepts optional `tools_offered` / `tools_invoked` / `tool_call_count` / `capabilities_invoked`.
- Document the manual path in README / QUICKSTART-LOCAL as the supported route when patches cannot see offered tools.
- **Out of v1:** a dedicated MCP *client* hook intercepting customer tool lists (on demand only).

### 6. Advisories and audit

| Code | Signal | Severity | Automation |
|------|--------|----------|------------|
| **TOOL-DEAD-001** | Per session: function tools offered on ≥N requests, never invoked; reported with `wasted_tool_schema_cost_usd` | INFO | Report-only |
| **CAP-001** | Declared capability silent across a **window** (cross-session) | WARNING | Report-only |

Single-session offered-but-never-called (Kind A) and per-session `declared_capabilities_silent` (Kind C) stay **summary stats** on `session.stats.summary()`. TOOL-DEAD-001 is the optional per-session advisory.

**"Silent" is a window for CAP-001.** A session legitimately may not exercise image gen, so firing per session is noise. CAP-001's unit is a volume/time window across sessions (e.g. `model:image` silent over N sessions / 24h).

**Runtime vs audit dispatch:**

| Signal | Runtime (`advisory.py` + `SessionStats`) | Audit (`audit_report.py`) |
|--------|------------------------------------------|---------------------------|
| `function_tools_never_called`, `wasted_tool_schema_*` | Summary; TOOL-DEAD-001 when per-session thresholds met | Recompute from stored events |
| `declared_capabilities_silent` (per session) | Summary stat only | Recompute when manifest supplied |
| **CAP-001** (windowed) | **Not on `SessionStats`** — no cross-session state | Primary path: scan stored events across the window with manifest |

`SessionStats` is per-session; do not implement CAP-001 by reading it after each event.

**Audit needs the manifest at audit time.** `expected_capabilities` is runtime config absent from stored events. Persist a manifest snapshot with the session record, or require `vetch audit --expected-capabilities ...`. Without this, `vetch audit` cannot compute CAP-001 or per-session Kind C silence offline.

PREMIUM-001: populating `tool_call_count` fixes `_tool_call_event_rate` for stored-event audits. Verify `tests/test_audit_report.py`; add non-zero cases.

---

## Cardinality and event size

Cap each per-event list (`tools_offered`, `tools_invoked`, `capabilities_invoked`) at **64** entries **at transport/OTel emission only**, never before the rollup. On truncation append `tools_offered_truncated: offered=N, recorded=64` to `vetch_warnings`. Sort before truncate.

**The cap must not corrupt audit parity.** Feed the rollup from the *untruncated* lists; apply the cap only on the serialized/exported copy. Otherwise `audit_report.py` (recomputing from stored events) sees a subset and disagrees with the runtime union for >64-tool agents. Memory is protected by the per-session bound, not this transport guard.

---

## Tests

| Area | Cases |
|------|-------|
| `capabilities.py` | Normalizer, malformed entries, empty `tools=`, parallel invocations |
| OpenAI / Anthropic | Non-stream offered + invoked; streaming accumulation. Pin to **recorded real SDK objects** (pydantic / cassettes) |
| Streaming edge | Error mid `tool_call` → `None`, never partial names |
| Reroute ordering | `stall_action="reroute"`: tools attributed to the **original** model |
| Cost | Cache-aware rate: fully-cached → `$0`; mixed cache; zero-billable guard; rate never from `total_cost_usd` |
| Kind C | Embedding; multimodal input; registry-map hit with zero config; override map; unmapped id → documented false negative |
| Rollup | `function_tools_never_called`, `declared_capabilities_silent`, `wasted_tool_schema_*` |
| Concurrency | Concurrent `SessionStats.update()` does not corrupt sets (lock) |
| Cardinality parity | >64-tool agent: runtime rollup and audit recompute agree |
| Advisory | TOOL-DEAD-001 per session; CAP-001 only from audit across the window |
| Audit | `tool_call_count` stored; PREMIUM-001 unchanged when zero; manifest at audit time |
| Privacy | No argument payloads; `redact_capability_name` honored; rollup survives hashing |
| Data contract | New nullable columns accepted by export sinks |
| OTel | semconv-aligned array attributes present when set |
| JS SDK | Parity tests mirroring Python golden cases |

---

## PR slicing

Each slice independently shippable; fields default to `None` until extraction lands.

1. **Schema + plumbing** — Python + TS types, `CapturedCall`, event emission, semconv OTel stubs, stats ingestion + cost accumulators with lock.
2. **Kind A non-stream** — `capabilities.py` (memoization + reroute-ordering) + OpenAI (+ Azure) + Anthropic.
3. **Kind A streaming** — OpenAI + Anthropic accumulators.
4. **Kind A remaining providers** — genai, vertexai, ollama.
5. **Kind C** — wrapper derivation + registry-sourced `model_capability_map` + `configure_capabilities`.
6. **Session rollup + cache-aware cost** — `stats.py`, `wasted_tool_schema_*`, `declared_capabilities_silent`, per-session bound.
7. **Advisories + audit** — TOOL-DEAD-001 (runtime + audit); CAP-001 (audit-only, windowed); manifest-at-audit-time; taxonomy doc.
8. **Manual capture + docs** — `capture()` fields, README / QUICKSTART-LOCAL.
9. **JS SDK parity** — middleware lists, vitest, reconcile with `TOOL-*` codes.

---

## Success criteria

A customer can execute their agent and, **with zero configuration** where the patch sees `tools=`, read from `session.stats.summary()` or `vetch audit`:

- Which function tools were offered but never called, and the **cache-aware** wasted input-token cost of re-sending those dead schemas (the headline), labeled a directional estimate.
- Which built-in capabilities (image/audio/embedding via the registry-sourced map) did or did not fire.

After declaring expected capabilities, they additionally get which declared routes have been silent across the window (CAP-001).

Without exposing tool arguments, with names passing through `redact_capability_name`, and without expanding schema beyond v2 additive fields.

---

## Deferred

| Item | Rationale |
|------|-----------|
| Kind B (builtin/server tools) | Provider-specific, unstable shapes |
| Kind D (agent from tags) | Near-free once the manifest pattern exists; schema reserves the shape |
| OpenAI Responses API tool extraction | Different request/response shape than Chat Completions |
| Regression / "went silent" baselines | Cold-start noise |
| `tool_choice: required` violation detection | Separate advisory; needs finish_reason + offered set |
