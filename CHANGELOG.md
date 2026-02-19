# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - 2026-02-19

### Added
- `energy_uncertainty_pct` field in events (20/50/100/1000 for tiers 0-3)
- MoE active parameter accounting in energy registry (fixes ~300% overestimation)
- Architecture metadata (`architecture`, `total_params_b`, `active_params_b`, `quantization`)
- Provider-specific PUE table (`registry/pue_overrides.json`)
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
| `VETCH_OUTPUT` | Output target: `stderr`, `none`, or file path | `stderr` |
| `VETCH_DEFAULT_PUE` | Power Usage Effectiveness multiplier | `1.1` |
| `VETCH_CACHE_MODE` | Set to `memory-only` for serverless | (file-based) |
| `VETCH_DISABLED` | Set to `true` to disable all tracking | `false` |
| `ELECTRICITY_MAPS_API_KEY` | API key for live grid data | (optional) |

### Dependencies
- **Runtime**: Zero dependencies (stdlib only)
- **Optional**: `openai>=1.0,<2.0`, `google-cloud-aiplatform>=1.0`, `tiktoken>=0.5.0`
- **Test**: `pytest>=7.0`, `hypothesis>=6.0`
