# Inference Waste Taxonomy

This document defines the advisory IDs used by Vetch to identify inference waste patterns. Each advisory has a stable ID, defined detection signal, and recommended action.

Advisories are deterministic signals from metadata, not proof of waste. Severity and confidence labels describe signal strength and review priority, not statistical certainty. Except for `STALL-001`, current implemented advisories are warn-only. Automatic kill and reroute are deliberately scoped to stalled-loop detection until workflow-specific policies are configured.

For a full explanation of how calls are intercepted and how detection signals are derived from token-count metadata, see [how-detection-works.md](how-detection-works.md).

**Status key:**
- ✅ **Implemented** — fires automatically when the signal threshold is met
- ⚠️ **Partial** — infrastructure exists but advisory does not fire
- 🔜 **Planned** — not yet implemented

---

## STALL-001 — Stalled agent loop

**Status:** ✅ Implemented

**Definition:** An agent session is producing little output while repeating similar inputs — the canonical "stuck in a loop" signature.

**Signal:**
- Rolling window of the last 20 calls
- ≥80% of calls in the window produce fewer than 5 output tokens
- ≥50% of calls in the window share the same input token count (input similarity)
- Advisory only fires after ≥10 total calls (to avoid false positives on fresh sessions)

**Severity:**
- `WARNING` if estimated cost of stalled calls ≤ $5.00
- `CRITICAL` if estimated cost of stalled calls > $5.00

**Why it matters:** A stalled agent loop burns tokens on every iteration with no useful output. Left unchecked, it drains budget, consumes compute, and emits avoidable carbon until it hits a timeout or an account limit.

**Example:** A ReAct-style agent that receives an ambiguous tool result, attempts the same tool call repeatedly, and produces near-empty reasoning outputs on each iteration.

**Possible false positives:**
- A classifier returning 1-token answers from varied inputs (high variety in input token counts will not trigger the similarity threshold)
- A streaming endpoint where token counts are estimated and the estimate is consistently low

**Recommended actions:**

| Action | When to use |
|--------|-------------|
| `"warn"` | During evaluation — observe without blocking |
| `"kill"` | Production — raise `vetch.StallDetected`, let caller handle recovery |
| `"reroute"` | Substitute a fallback model that may handle the stuck state differently |

**Recovery:** Call `session.clear_stall()` after a human-in-the-loop fix to re-arm detection.

**Wire it:**
```python
vetch.set_stall_action("kill")
# or
vetch.set_stall_action("reroute", fallback_model="gpt-4o-mini")
```

---

## CACHE-001 — Prompt caching opportunity

**Status:** ✅ Implemented

**Definition:** More than half of calls in the session share identical input token counts, suggesting a static system prompt or context block that could be cached at the provider level.

**Signal:**
- ≥6 calls in the session
- >50% of calls share the same input token count

**Severity:** `WARNING`

**Why it matters:** Providers including Anthropic and DeepSeek offer prompt caching at up to 90% cost reduction for cached input tokens. A repeated static system prompt that is not cached is direct, avoidable spend on every call.

**Example:** An API service that prepends a 2,000-token system prompt to every user message. Without caching, those 2,000 tokens are billed on every call at full price.

**Possible false positives:**
- Short, template-driven prompts where identical token counts are coincidental
- Load testing with identical requests

**Recommended action:** Enable provider prompt caching for the static portion of the prompt. For Anthropic, use the `cache_control` parameter. For OpenAI, prefix caching is automatic for prompts over 1,024 tokens.

---

## RAG-001 — RAG bloat

**Status:** ✅ Implemented

**Definition:** The average input-to-output token ratio is excessively high, suggesting the retrieval pipeline is stuffing large amounts of context that produces little useful output.

**Signal:**
- Average input:output ratio > 50:1 across the session

**Severity:** `INFO`

**Why it matters:** A RAG pipeline that retrieves 10,000 tokens to produce a 50-token answer is paying for 99.5% of its input cost on context that does not contribute to the output. This also increases latency, raises energy consumption, and can degrade answer quality through context saturation.

**Example:** A document QA system that retrieves the top-20 chunks without a relevance threshold, producing massive prompts regardless of how focused the query is.

**Possible false positives:**
- Summarization workloads (high input:output ratio is expected and appropriate)
- Extraction tasks where a large document produces a short structured output

**Recommended actions:**
- Add or tighten a relevance score threshold on retrieved chunks
- Reduce the number of retrieved chunks
- Use a smaller, cheaper model for the initial retrieval / reading step
- Consider a map-reduce pattern to avoid single large-context calls

---

## BABBLE-001 — Excessive generation

**Status:** ✅ Implemented

**Definition:** Recent calls are producing unusually long outputs, suggesting the model is generating without effective length constraints or stop conditions.

**Signal:**
- Rolling window of the last 20 calls
- Average `output_tokens` across the window exceeds 1,500 tokens
- Advisory only fires after ≥10 total calls

**Severity:** `WARNING`

**Why it matters:** Unconstrained generation is the output-side equivalent of RAG bloat. Long outputs increase latency, cost, and energy consumption on every call. If the workflow does not require long-form generation — summarization, extraction, classification, tool use — excessive output length suggests missing `max_tokens`, absent stop conditions, or a response format that allows the model to pad its answer.

**Example:** A structured data extraction pipeline that does not set `max_tokens` or `response_format`. The model returns the extracted fields plus an explanation, an apology, and a restatement of the task on every call.

**Possible false positives:**
- Code generation (expected to be long)
- Long-form writing or analysis tasks
- Document summarization where the source is long

**Recommended actions:**
- Set `max_tokens` to a value appropriate for the expected output length
- Use `response_format={"type": "json_object"}` or a JSON schema to constrain structure
- Add explicit stop conditions if the model should halt after a known delimiter
- If long outputs are legitimately required, this advisory can be ignored for that workflow

**Note:** BABBLE-001 is warn-only. There is no automatic kill or reroute action because shortening generation without quality verification would produce truncated or wrong answers. Review is always required before tightening length constraints.

---

## ZOMBIE-001 — Post-completion drift

**Status:** ✅ Implemented

**Definition:** A session keeps sending highly similar prompts and producing normal-length responses after the workflow appears to have reached a terminal state.

**Signal:**
- Rolling window of recent calls
- ≥80% input similarity across the window
- Average output is above the STALL-001 low-output threshold
- Output token counts have low variation (`recent_output_token_cv` ≤ 0.15)

**Severity:** `WARNING`

**Why it matters:** A loop can be wasteful even when the model is still producing full sentences. The model may be restating completion, apologies, or status text while the caller keeps asking for another step.

**Example:** An agent submits its final answer, then receives the same state again and repeatedly returns "The final answer has been submitted" for the remaining loop budget.

**Possible false positives:**
- Batch jobs that intentionally ask the same prompt over similar items
- Evaluation harnesses with repeated prompts
- Workflows where consistent response length is expected and useful

**Recommended actions:**
- Add an explicit completion guard or terminal sentinel
- Stop the caller loop when the model reports a completed task
- Set a max-step budget for agent sessions
- Check `CTX-001` if context is growing while the repeated output continues

**Note:** ZOMBIE-001 is warn-only. It should feed review and workflow fixes, not automatic intervention, until task-specific evals prove the repeated normal-length responses are wasteful.

---

## CTX-001 — Context snowball

**Status:** ✅ Implemented

**Definition:** Input tokens grow quickly across recent calls while the workflow still fails to make proportional progress.

**Signal:**
- Rolling window of recent calls
- Recent input tokens grow by ≥3× from the start of the window to the end
- Input tokens increase on ≥70% of transitions in the window
- Average input:output ratio is ≥4:1

**Severity:** `WARNING`

**Why it matters:** Failed tool turns, apologies, and repeated instructions often get fed back into the next request. The model may not look stalled, but each turn becomes more expensive as the prompt expands.

**Example:** A tool repeatedly fails, the agent appends every failed attempt to the conversation, and each new request pays for the full failure history again.

**Possible false positives:**
- Long research sessions where accumulated context is expected
- Summarization or document-review workflows
- Multi-step tasks that legitimately build state over time

**Recommended actions:**
- Trim prior failed turns from the prompt
- Summarize tool history instead of replaying it verbatim
- Cap context growth per session
- Stop after repeated tool-contract failures

**Note:** CTX-001 is warn-only. Automatic truncation can remove useful context, so workflow-specific quality checks are required.

---

## EMPTY-001 — Invisible output burn

**Status:** ✅ Implemented

**Definition:** Recent calls consume output tokens while returning little or no visible answer text.

**Signal:**
- Rolling window with at least 5 calls that have visible-output character counts
- At least 4 calls consume ≥20 output tokens while returning ≤2 visible characters
- At least 50% of visible-counted calls in the window match that pattern
- Tool-call finish reasons are excluded from the empty-visible count

**Severity:** `WARNING`

**Why it matters:** Some providers and model modes can spend tokens on hidden reasoning, malformed output, whitespace, or truncated attempts that produce no useful visible answer.

**Example:** A reasoning-mode call uses output tokens but returns only whitespace or an empty visible message because it hits a cap before producing a final answer.

**Possible false positives:**
- Tool-use workflows where the useful output is a tool call rather than text
- Providers with unusual content block formats
- Intentionally hidden intermediate reasoning

**Recommended actions:**
- Check whether the model is hitting an output cap
- Use a non-reasoning model if hidden reasoning is not needed
- Add a response-format guard
- Inspect provider-specific finish reasons and content blocks

**Note:** EMPTY-001 is warn-only. Empty visible output can be legitimate in some reasoning or tool-call workflows.

---

## TRUNC-001 — Repeated response truncation

**Status:** ✅ Implemented

**Definition:** Recent calls frequently end because the model hit its output token limit.

**Signal:**
- Rolling window of at least 5 calls
- At least 3 calls in the window have `finish_reason=max_tokens`
- At least 50% of the window ends with `finish_reason=max_tokens`

**Severity:** `WARNING`

**Why it matters:** Repeated truncation can produce broken JSON, incomplete tool calls, cut-off answers, and retry loops. It is often a configuration issue rather than a model-quality issue.

**Example:** A tool-calling workflow sets `max_tokens=8`, so the model starts a JSON tool argument but cannot finish it. The caller retries, causing the same truncation again.

**Possible false positives:**
- Workflows that intentionally cap output and accept truncation
- Streaming or sampling tests that deliberately exercise small token caps

**Recommended actions:**
- Increase `max_tokens`
- Add a stop sequence that encourages earlier completion
- Use a stricter response schema or shorter expected answer format
- Validate JSON/tool-call completeness before accepting a result

**Note:** TRUNC-001 is warn-only. Confirm truncation is unintended before changing output limits.

---

## SESSION-BUDGET-001 — Session over budget

**Status:** ⚠️ Partial — budget monitoring fires alerts, no advisory ID or circuit breaker

**Definition:** A session has exceeded a configured cost, energy, or carbon threshold.

**Signal (planned):** Session accumulated cost/energy/carbon exceeds `set_budget()` threshold.

**Severity (planned):** `WARNING` → `CRITICAL` based on overage percentage.

**Why it matters:** Without per-session budget enforcement, a single runaway agent session can consume a disproportionate share of total inference budget before anyone notices.

**Current behaviour:** `set_budget()` fires a callback alert when thresholds are crossed. The session is not blocked. No advisory ID is assigned.

**Planned:** Advisory ID, severity levels, and optional circuit breaker action (kill or reroute) when session budget is exceeded.

**Wire it (current):**
```python
vetch.set_budget("session", cost_usd=2.0)

@vetch.on_budget_alert
def handle(alert):
    print(f"Session over budget: {alert}")
```

---

## ATTRIBUTION-001 — Unattributed spend

**Status:** ⚠️ Partial — `require_tags()` flags events, no advisory fires

**Definition:** Inference spend that cannot be attributed to a feature, customer, workflow, or team because required tags are missing.

**Signal (planned):** Events where one or more required tags are absent.

**Why it matters:** Unattributed spend is invisible spend. You cannot optimize what you cannot measure. If 30% of your inference cost has no feature tag, you have no way to know which feature is the problem.

**Current behaviour:** `require_tags(["feature", "customer"])` marks events with `tracking_disabled: true` when tags are missing. No advisory fires and no summary is produced.

**Planned:** Advisory with a count of untagged calls, the most common missing tags, and a suggested fix.

**Wire it (current):**
```python
vetch.require_tags(["feature", "customer"])
```

---

## RETRY-001 — Retry storm

**Status:** 🔜 Planned

**Definition:** A session is making repeated failed or near-identical calls in rapid succession, indicating a retry loop without appropriate backoff or a persistent upstream error.

**Signal (proposed):**
- Error rate > 50% across the last 10 calls, or
- ≥5 consecutive calls with identical input token counts that resulted in errors

**Severity (proposed):** `WARNING` → `CRITICAL` based on error rate and cost.

**Why it matters:** Retry storms are pure waste — every retried call costs tokens, adds latency, and consumes compute, with no successful output. They also risk triggering provider rate limits, compounding the problem.

**Example:** An agent that catches a `RateLimitError` and immediately retries without exponential backoff, triggering further rate limits on every attempt.

**Possible false positives:**
- Legitimate retries after transient errors (1–2 retries with backoff is normal)
- Load tests that intentionally exercise error paths

**Recommended actions:**
- Add exponential backoff with jitter
- Set a maximum retry count
- Alert on sustained error rates before the budget is exhausted

---

## PREMIUM-001 — Premium model overuse

**Status:** 🔜 Planned

**Definition:** A high-capability, high-cost model is being used for tasks where a cheaper model would produce equivalent results.

**Signal (proposed):**
- Session average cost-per-output-token is in the top tier of the model registry
- Output complexity indicators suggest low-reasoning tasks (short outputs, structured extraction, classification)

**Severity (proposed):** `INFO` → `WARNING` based on estimated cost difference.

**Why it matters:** Using GPT-4o or Claude Sonnet for tasks that GPT-4o-mini or Haiku handle equally well is a direct margin hit. The cost difference between tiers can be 10–50×.

**Example:** A customer support triage pipeline using GPT-4.5 ($0.075/1k input tokens) to classify incoming tickets into five categories — a task GPT-4o-mini ($0.00015/1k input tokens) handles accurately.

**Possible false positives:**
- Tasks that appear simple but require nuanced reasoning
- Cases where the premium model is explicitly required for quality guarantees

**Recommended actions:**
- Identify the lowest-cost model that meets quality thresholds for the task
- Use `vetch compare` to see cost and energy differences across model tiers
- Implement model routing based on task complexity

---

## CACHE-MISS-001 — Cache miss pattern

**Status:** 🔜 Planned

**Definition:** The session is writing to the prompt cache (paying cache creation costs) but not reading from it on subsequent calls, indicating the cache is not being reused as intended.

**Signal (proposed):**
- `cache_creation_tokens > 0` across multiple calls
- `cache_read_tokens` remains low or zero relative to creation tokens
- Cache write cost exceeds cache read savings

**Severity (proposed):** `WARNING`

**Why it matters:** Prompt caching has a creation cost (typically 1.25× normal input token price on Anthropic). If the cache is written but never read — because the prompt structure changes between calls, or the cache TTL expires — you pay the premium without the discount.

**Recommended actions:**
- Ensure the cacheable prefix (system prompt, static context) is identical across calls in the same session
- Check that calls within the cache TTL window are reusing the same client session
- Use `vetch.wrap()` with session tracking to verify cache hit rates
