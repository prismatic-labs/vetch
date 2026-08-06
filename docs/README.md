# Vetch documentation

Start with the [project README](../README.md) for the overview and quick start. These pages go deeper on individual topics.

## Concepts and methodology

- [energy-methodology.md](energy-methodology.md): how energy, carbon, and water are estimated, the four confidence tiers, and how much to trust each
- [model-resolution.md](model-resolution.md): how a model name resolves to a registry entry, and why proxies are flagged low-confidence
- [region-config.md](region-config.md): choosing a grid region, the fallback order, and reading `signal_quality`
- [how-detection-works.md](how-detection-works.md): call interception, advisory detection, and why the stall circuit breaker needs an explicit `Session`
- [inference-waste-taxonomy.md](inference-waste-taxonomy.md): every advisory ID with its detection signal, false positives, and recommended action

## Guides

- [attribution.md](attribution.md): sessions, tags, nesting, distributed propagation, and tool/capability observability
- [audit-report.md](audit-report.md): the 7-day inference-waste audit: how to run it and how to read the report
- [mcp.md](mcp.md): the MCP server, its tools and resources, and an example agent flow
- [OPENTELEMETRY.md](OPENTELEMETRY.md): exporting cost, energy, carbon, and advisories to an OTLP backend

## Reference

- [../src/vetch/registry/PROVENANCE.md](../src/vetch/registry/PROVENANCE.md): registry format and sourcing rules for energy/pricing data

## Design notes and internal

These track in-progress work and release history rather than documenting stable behavior.

- [capability-observability-plan.md](capability-observability-plan.md) / [capability-observability-build.md](capability-observability-build.md): design and build notes for tool/capability observability
- [evaluation-follow-through-plan.md](evaluation-follow-through-plan.md): evaluation follow-through plan
- [vercel-ai-sdk-poc.md](vercel-ai-sdk-poc.md), [SCOPE-v0.8.0-vercel.md](SCOPE-v0.8.0-vercel.md), [NPM_PUBLISH.md](NPM_PUBLISH.md): Vercel AI SDK scope and publishing
- [V0.8.0_PRODUCTION_FIXES_RECEIPT.md](V0.8.0_PRODUCTION_FIXES_RECEIPT.md): v0.8.0 production-fix record
