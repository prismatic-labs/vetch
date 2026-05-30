# Vetch Roadmap

Organised around outcomes. Each section describes what Vetch should help you accomplish, with current status and planned work.

**✅** Implemented · **⚠️ Partial** Infrastructure exists, incomplete · **🔜 Planned** Not yet implemented

---

## Detect waste

Turn every inference call into a signal. Vetch detects patterns that indicate inefficiency — stalled loops, cacheable prompts, bloated context — and makes them inspectable and actionable.

**Current:**
- ✅ Stalled agent loop detection
- ✅ Prompt caching opportunity detection
- ✅ RAG context bloat detection
- ⚠️ Session budget monitoring — warn-only, circuit breaker not yet wired

**Planned:**
- 🔜 Retry storm detection
- ✅ Large-model rightsizing candidate detection — audit-only
- 🔜 Zombie inference detection (active calls past expected session completion)
- 🔜 Cache miss pattern detection
- 🔜 Session budget advisory and circuit breaker
- 🔜 Missing attribution advisory
- 🔜 Stable severity schema with documented levels
- 🔜 Advisory schema version guarantee

Full advisory reference: [docs/inference-waste-taxonomy.md](docs/inference-waste-taxonomy.md)

---

## Attribute waste

Know exactly which feature, customer, workflow, team, or agent session produced each unit of waste. Attribution is the difference between "our LLM bill went up" and "the RAG search feature for enterprise customers is the culprit."

**Current:**
- ✅ Per-call tagging
- ✅ Global tags
- ✅ Required tag enforcement
- ✅ Tag cardinality limiting and allowlisting
- ✅ Session aggregation with parent/child hierarchy
- ✅ Distributed session propagation via HTTP headers (W3C-compatible)
- ✅ Per-session cost, energy, and carbon accumulation

**Planned:**
- 🔜 CLI waste aggregation by tag, feature, customer, and model
- 🔜 Cost per successful task or session outcome
- 🔜 Attribution drift detection (spend shifting unexpectedly between features)

---

## Intervene automatically

Move from observation to control. Stop waste before it accumulates, not just report it afterward.

**Current:**
- ✅ Configurable stall action: warn, kill, or reroute
- ✅ Transparent model substitution on reroute with fail-open fallback
- ✅ Structured exception with session recovery
- ⚠️ Budget alerts — warn-only callbacks, no automatic intervention

**Planned:**
- 🔜 Circuit breakers across all waste detection patterns (not just stall detection)
- 🔜 Policy engine — configurable actions per waste pattern, model, or tag group
- 🔜 Session budget enforcement
- 🔜 Automatic model downgrade after eval-backed rightsizing approval
- 🔜 Throttling per session or tag combination
- 🔜 Human-in-the-loop recovery for killed sessions

---

## Prove savings

Make the case for investment in inference efficiency. Produce evidence of actual savings — in cost, tokens, compute, energy, and carbon — that can be shared with engineering leadership or a FinOps team.

**Current:**
- ✅ Per-call cost, energy, and carbon in every event
- ✅ Session totals for cost, energy, carbon, and call count
- ✅ Advisory list and session token summary
- ✅ Usage report over configurable time windows

**Planned:**
- 🔜 Avoided-cost estimation against a counterfactual (no circuit breaker)
- 🔜 Baseline vs. optimised comparison for before/after policy changes
- 🔜 Full audit report: waste by feature and customer, estimated avoidable spend, recommended actions
- 🔜 Avoided tokens, calls, energy, and carbon alongside avoided cost

---

## Integrations

Meet engineers where they are. Waste detection should work across all major LLM frameworks and infrastructure patterns.

**Current:**
- ✅ OpenAI (including Azure OpenAI)
- ✅ Anthropic
- ✅ Vertex AI (Gemini)
- ✅ Google GenAI SDK
- ✅ All OpenAI-compatible endpoints (OpenRouter, Together.ai, Ollama, vLLM, TGI)
- ✅ OpenTelemetry / OTLP export (Grafana, Datadog, Honeycomb)
- ✅ MCP server for agent-native integrations
- ✅ Background jobs via distributed session propagation

**Planned:**
- 🔜 LangGraph, LangChain, LlamaIndex, and CrewAI native integrations
- 🔜 FastAPI middleware for automatic per-request session scoping
- 🔜 Grafana dashboard for inference waste
- 🔜 Slack and webhook alerts for advisory events and budget thresholds
- 🔜 CI/CD budget regression checks

---

## Enterprise readiness

Production reliability and organisational trust.

**Current:**
- ✅ Fail-open — LLM calls always proceed if Vetch fails
- ✅ Stable event schema (v1.x: no breaking changes to existing fields)
- ✅ No prompt or completion capture — metadata only
- ✅ Thread safety and async safety (contextvars)
- ✅ Emergency kill switch (`VETCH_DISABLED=true`)
- ✅ Air-gapped operation via frozen registry

**Planned:**
- 🔜 Audit log for circuit breaker interventions
- 🔜 Stable advisory schema with version guarantees
- 🔜 Privacy architecture document (verifiable, not just asserted)
- 🔜 Backward compatibility policy for event schema, advisory schema, and public API
- 🔜 Self-hosted event collector
- 🔜 Compatibility matrix across Python versions and provider SDK versions

---

## Methodology and credibility

Energy and carbon estimates are only useful if their uncertainty is clearly stated and the methodology is auditable.

**Current:**
- ✅ Four-tier confidence system (Measured / Vendor-Published / Validated / Estimated)
- ✅ Uncertainty bounds per event
- ✅ Prompt-length-aware energy model
- ✅ Provider-specific PUE
- ✅ Registry provenance documentation ([PROVENANCE.md](src/vetch/registry/PROVENANCE.md))
- ✅ Electricity Maps integration for live grid carbon intensity

**Planned:**
- 🔜 Region confidence levels (static annual average vs. live vs. unknown)
- 🔜 Safe claims guidance — what Vetch figures can and cannot be used to assert
- 🔜 Model registry update cadence and versioning policy
- 🔜 Methodology comparison with EcoLogits, CodeCarbon, and provider sustainability reports
