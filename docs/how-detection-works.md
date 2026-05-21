# How Vetch detects waste patterns

This document explains the two-layer detection model: how API calls are intercepted, and how advisory patterns are identified from the metadata they produce.

---

## Layer 1 — Call interception

### `instrument()`: module-level monkeypatching

When you call `vetch.instrument()` at startup, it patches the provider SDK class itself — not an instance. For OpenAI this means patching `openai.OpenAI.__init__` (and `AsyncOpenAI.__init__`) so that every new client object that is instantiated after the call automatically has its `chat.completions.create` method replaced.

The replacement is a `_WeakChatWrapper` (sync) or `_WeakAsyncChatWrapper` (async). The wrapper:

1. Stores the original `create` function in a `WeakKeyDictionary` keyed by the `completions` object, which avoids GC cycles.
2. On every call, runs `apply_stall_action()` **before** forwarding to the original (the circuit breaker check).
3. Calls the real `create(*args, **kwargs)` — the LLM call is never skipped or delayed.
4. On success, calls `_after_create(result, ...)` which reads `response.usage` and captures `{input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens}` from the provider's metadata fields. Prompt text is not read. For EMPTY-001 diagnostics, providers may count visible completion characters and immediately discard the text.
5. Passes those values to `ctx.capture(...)`, which triggers `VetchContext._emit_event()`, which writes to local storage, updates stats, and exports to OTel.

For streaming responses, the response is wrapped in `StreamWrapper`, which accumulates chunk-level token counts and fires `_emit_event()` on the final chunk.

If a `wrap()` context is already active (manual instrumentation), `_after_create` reuses it — calls are never double-counted. If no context exists, `auto_context_for_instrumented_call` creates a temporary `VetchContext` scoped to that single call.

### `wrap()`: explicit per-call context

`wrap()` is a context manager that creates a `VetchContext` and sets it as the active context via a `ContextVar`. The provider interceptors check `get_active_context()` and capture into it. On `__exit__`, `_emit_event()` runs. This is the lower-level primitive; `instrument()` builds on top of it.

### What is and is not read

Vetch reads provider metadata and, for output diagnostics, may count visible completion characters:

| Field read | Source (OpenAI example) |
|---|---|
| `input_tokens` | `usage.prompt_tokens` |
| `output_tokens` | `usage.completion_tokens` minus reasoning tokens |
| `cache_read_tokens` | `usage.prompt_tokens_details.cached_tokens` |
| `cache_creation_tokens` | provider-specific field |
| `model` | `response.model` |
| `finish_reason` | provider response metadata |
| `requested_max_tokens` | request kwargs, when available |
| `visible_output_chars` | counted from visible completion blocks, then discarded |

Prompt text, tool call arguments, and system prompts are never accessed. Completion text is not stored or exported; for output diagnostics, Vetch may count visible completion characters and then discard the text.

---

## Layer 2 — Advisory detection

Every captured event updates a `SessionStats` instance via `track_session_event()`. `SessionStats` maintains:

- A rolling `deque(maxlen=20)` of recent call metadata, including input tokens, output tokens, cost, visible output character count, finish reason, and requested output cap.
- A frequency histogram `input_token_counts[n] += 1` — used by both STALL-001 and CACHE-001.
- Running totals of input/output tokens, energy, cost, carbon, and water.

`generate_advisories(stats)` is called after each event (once inside a `Session.register_event()` call) and checks these patterns against the metadata:

| Advisory | Signal | Threshold |
|---|---|---|
| STALL-001 | Fraction of last 20 calls with `output_tokens < 5` + fraction with identical input token count | ≥80% low-output AND ≥50% input similarity, after ≥10 total calls |
| CACHE-001 | Fraction of all calls sharing the same input token count | >50% share the same count, after ≥6 total calls |
| RAG-001 | `total_input_tokens / total_output_tokens` | >50:1 average ratio |
| BABBLE-001 | Average `output_tokens` in last 20 calls | >1,500 tokens average, after ≥10 total calls |
| ZOMBIE-001 | High input similarity + normal-length outputs with low output-token variation | ≥80% input similarity, output CV ≤0.15, window ≥5 |
| CTX-001 | Prompt/input tokens grow while output does not grow proportionally | ≥3× input growth, ≥70% increasing turns, average ratio ≥4:1 |
| EMPTY-001 | Output tokens consumed while visible output is near-empty | ≥4 recent calls and ≥50% of visible-counted window |
| TRUNC-001 | Provider reports `finish_reason=max_tokens` repeatedly | ≥3 recent calls and ≥50% of window |

No prompt content is read by any of these checks. Detection signals are derived from metadata such as token counts, cost, visible output character count, finish reason, and requested output cap.

### Why STALL-001 needs two signals

The low-output signal alone (`output_tokens < 5`) would fire on any classifier, routing step, or intentionally terse tool-use workflow. The input similarity signal (`input_token_counts` histogram) breaks the tie: a stuck loop re-sends the same prompt at the same length, so the most common input token count dominates. A legitimate multi-tool agent sends varied inputs to different tool endpoints, so the histogram stays flat. Both signals must exceed their thresholds simultaneously.

---

## The `instrument()` vs `Session` gap

This is the most important behaviour to understand before configuring stall protection.

### What each mode provides

| | `instrument()` alone | `instrument()` + `Session()` | `wrap()` + `Session()` |
|---|---|---|---|
| Event storage | ✅ | ✅ | ✅ |
| OTel metrics | ✅ | ✅ | ✅ |
| Advisory detection (audit reports) | ✅ global stats | ✅ per-session stats | ✅ per-session stats |
| STALL-001 circuit breaker (kill/warn/reroute) | ❌ | ✅ | ✅ |

### Why the circuit breaker requires `Session()`

The stall circuit breaker works through a flag on the `Session` object:

```
session.stall_triggered = True   ← set by Session.register_event() when STALL-001 fires
```

`apply_stall_action()` checks `get_active_session().stall_triggered` before every call. If there is no active `Session`, this check always returns `(False, None)` and the loop continues regardless of what the advisory engine has seen.

`Session.register_event()` is the only place where `generate_advisories()` can
trip the live stall circuit breaker against per-session stats. Without a
`Session`, advisory detection still runs against the global `_session_stats`
singleton for audit and optional `on_advisory()` callbacks, but the result is
not used to kill or reroute calls.

### Using `instrument()` with stall protection

Wrap the agentic loop in `vetch.Session()`. The `instrument()` interceptors check for an active `Session` via `ContextVar` — no code change to the actual agent is needed:

```python
import vetch
import openai

vetch.instrument()
vetch.set_stall_action("kill")

client = openai.OpenAI()

with vetch.Session(tags={"feature": "agent-loop"}) as session:
    try:
        while True:
            response = client.chat.completions.create(  # intercepted automatically
                model="gpt-4o",
                messages=agent.get_messages(),
            )
            agent.step(response)
    except vetch.StallDetected as e:
        log.warning("Stall detected: %s", e)
        # fix the agent state, then session.clear_stall() to re-arm
```

Without the `with vetch.Session()` wrapper, `set_stall_action("kill")` has no effect — the kill check is never reached because there is no session to read `stall_triggered` from.

### `instrument()` alone is observability only

`instrument()` without `Session` is appropriate when you want audit data across many unrelated calls (a shared API client, a web service handling multiple users) and do not want a global circuit breaker that could affect unrelated requests. In this mode:

- Every call is stored and counted.
- Advisories appear in `vetch audit` output.
- No call is ever blocked or rerouted.

Per-request stall protection in a web service context requires constructing a `Session` per request (or per agent invocation) and letting it go out of scope when the request ends.

---

## Detection limits

These are intentional, not oversights:

**Content-blind by design.** Advisories are approximate by construction. A classifier returning single-word answers can look like a stall from the token-count perspective; the input similarity guard reduces false positives but cannot eliminate them for all workloads. Treat advisories as signals requiring confirmation, not ground truth.

**BABBLE-001 is warn-only.** Long output is expected for code generation, long-form writing, and analysis tasks. There is no automated action for BABBLE-001 because shortening generation without quality checks would produce wrong answers.

**RAG-001 is warn-only.** Summarisation and extraction tasks have high input:output ratios by design. RAG-001 requires human review to distinguish genuine bloat from legitimate workloads.

**ZOMBIE-001, CTX-001, EMPTY-001, and TRUNC-001 are warn-only.** They identify suspicious patterns that often explain wasted inference, but each can be legitimate in some workflows. Use them to prioritize review and fixes before adding automation.

**STALL-001 requires a minimum call count.** Detection does not fire until ≥10 calls have accumulated in the session. This prevents false positives on fresh sessions where the rolling window has too few data points to be meaningful.
