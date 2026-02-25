# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- Energy estimates are Tier 3 (±10x uncertainty) for most models

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
