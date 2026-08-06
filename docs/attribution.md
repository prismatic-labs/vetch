# Attribution and sessions

A provider dashboard tells you what you spent. It does not tell you *who* spent it: which feature, customer, workflow, or agent run. Attribution is how Vetch answers that question. Every call is tagged, every call belongs to a session, and cost, energy, and carbon accumulate per session and per tag combination so you can slice spend the way your business is actually organized.

## Sessions are the unit of attribution

A session groups related calls. Everything inside it accumulates together and can be read or exported as a unit.

```python
import vetch

with vetch.Session(tags={"agent": "researcher", "task": "summarize"}) as session:
    with vetch.wrap() as ctx:
        response = client.chat.completions.create(...)

print(f"Total cost:   ${session.total_cost_usd}")
print(f"Total energy: {session.total_energy_wh} Wh")
print(f"Call count:   {session.call_count}")
```

Create a session per unit of work: one web request, one background job, one agent invocation. That boundary is what makes the numbers meaningful. A session that spans a whole process aggregates everything into one bucket and tells you nothing.

### Tags

Tags are free-form key-value pairs. Vetch does not prescribe a schema; use the dimensions you want to slice by.

```python
with vetch.Session(tags={
    "feature": "rag-search",
    "customer": "acme",
    "environment": "prod",
    "team": "ml",
}) as session:
    ...
```

Common dimensions are `feature`, `customer`, `user`, `workflow`, `environment`, and `team`. Cost, energy, and carbon roll up per session and per tag combination, so "which customer burned 30% of last week's inference budget" becomes a filter, not an investigation. The audit report ([audit-report.md](audit-report.md)) reads these tags to produce per-dimension breakdowns.

Tags on `wrap()` merge with the enclosing session's tags, so you can set broad tags on the session and per-call tags on individual `wrap()` blocks.

### Nesting

Sessions nest. A parent session records its own aggregate; child sessions track their slice and link back through `parent_session_id`. This mirrors how multi-agent systems actually run: an orchestrator with sub-agents underneath it.

```python
with vetch.Session(tags={"agent": "orchestrator"}) as root:
    with vetch.wrap():
        plan = client.chat.completions.create(...)

    with vetch.Session(tags={"agent": "summarizer"}) as child:
        with vetch.wrap():
            summary = client.chat.completions.create(...)
```

### Guarding long-running loops

`Session` accepts `max_calls` (default 10,000) to cap how many calls it accumulates. In an agent loop that could run for hours, this prevents unbounded memory growth in the session's stats. Set it higher for long legitimate runs; set it to `0` for unlimited, which is not recommended for anything long-lived.

### Per-session detector tuning

`Session` accepts `advisory_thresholds` to scope waste-detector thresholds to one route, workflow, or tenant without changing process-wide defaults:

```python
with vetch.Session(
    tags={"route": "classifier"},
    advisory_thresholds={"STALL-001": {"low_output_threshold": 1}},
):
    response = client.chat.completions.create(...)
```

This is how a classification route that legitimately returns one-token answers avoids tripping the stalled-loop detector, while every other route keeps the default. See [inference-waste-taxonomy.md](inference-waste-taxonomy.md) for the tunable fields per advisory.

## Distributed propagation

Attribution should survive a network hop. When one service kicks off work in another (an API handler dispatching a Celery task, a queue consumer picking up a job), the session travels in the request headers so the downstream calls attribute to the same session tree.

```python
# Producer: inject the current session into outbound headers
headers = session.inject_headers({})
celery_task.delay(task_id, headers=headers)

# Consumer: rebuild the session from those headers
with vetch.Session.from_headers(task_headers) as worker_session:
    with vetch.wrap() as ctx:
        response = client.chat.completions.create(...)
```

`inject_headers` adds Vetch's propagation keys to a headers dict you own (it does not overwrite the rest). `from_headers` reconstructs a session that links to the originating one, so a workflow spread across three services still rolls up to a single attribution tree.

## Tool and capability observability

Beyond cost, Vetch records what each call *offered* the model versus what the model actually *used*. This surfaces two kinds of quiet waste.

**Dead function tools (`TOOL-DEAD-001`).** If you attach a large tool schema to every request and the model never calls half of it, you pay to transmit those schemas on every call. Vetch records which function tools were offered and which were invoked, then rolls up the wasted schema tokens with a cache-aware cost estimate.

```python
import vetch

with vetch.wrap() as ctx:
    response = client.chat.completions.create(
        model="gpt-4o",
        tools=[{"type": "function", "function": {"name": "search", ...}}],
        ...
    )

summary = vetch.get_session_stats().summary()
print(summary["function_tools_never_called"])
print(summary["wasted_tool_schema_cost_per_request_usd"])  # one transmission
print(summary["wasted_tool_schema_session_cost_usd"])      # across requests with dead tools
```

Per-request footprint (`..._per_request_...`) is the cost of one transmission; session totals multiply by the number of requests that carried dead tools. A fully cached session reports `$0` with a note in `wasted_tool_schema_cost_note`, since cached schemas are not re-billed. For trustworthy math on manual `capture()`, always pass `tools_invoked` (use `[]` when nothing fired). A `None` there means "unknown," and those requests are excluded from the dead-tool count.

**Silent declared capabilities (`CAP-001`).** Declare the capabilities you expect a workflow to exercise, and Vetch flags any that never fired across an audit window: a route you built and pay to keep wired up but no longer actually use.

```python
vetch.configure_capabilities(
    expected=["model:image", "model:embedding"],
)
```

`configure_capabilities` sets this process-wide; `Session(expected_capabilities=[...])` scopes it to one session. Run the audit with `vetch audit --expected-capabilities model:image,model:embedding` to evaluate CAP-001 over stored events.

Capability names can be redacted for privacy: set `VETCH_REDACTION_KEY`, or pass `redact_names=True` / `redacted_names=[...]` to `configure_capabilities`. Session-level rollups are computed Python-side; the JS SDK derives capabilities and redacts names but leaves rollups to the Python side.

For the full design (the Kind A/B/C capability model, the registry map, and the cost semantics), see [capability-observability-build.md](capability-observability-build.md).

## Related

- [audit-report.md](audit-report.md): turning tagged sessions into a per-dimension spend report
- [inference-waste-taxonomy.md](inference-waste-taxonomy.md): the advisories, including `TOOL-DEAD-001` and `CAP-001`
- [how-detection-works.md](how-detection-works.md): why the stall circuit breaker needs an explicit `Session`
- [capability-observability-build.md](capability-observability-build.md): the full tool/capability observability design
