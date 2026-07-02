# Contributing to Vetch

Thank you for your interest in contributing to Vetch! This guide covers
development setup, testing, and the contribution process.

## Development Setup

### Prerequisites

- Python 3.9+
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/prismatic-labs/vetch.git
cd vetch

# Install in development mode with all extras
pip install -e ".[test,dev]"

# Verify installation
vetch --version
```

### Pre-commit Hooks (Recommended)

```bash
pip install pre-commit
pre-commit install
```

This runs ruff and mypy automatically on every commit.

## Testing

### Running Tests

```bash
# Run all tests with coverage
pytest tests/ -v --cov=vetch --cov-report=term-missing

# Run a specific test file
pytest tests/test_session.py -v

# Run a specific test
pytest tests/test_session.py::TestSession::test_basic_session -v
```

Coverage flags are not in `addopts`, so running a subset (or bare `pytest`)
does not fail the 70% gate — pass `--cov=vetch --cov-fail-under=70` explicitly
(as above) when you want the gate. CI always enforces it.

### Test Requirements

- **Coverage**: 70% minimum enforced in CI, 90% target
- **Property tests**: Use Hypothesis for calculation edge cases
- **Mock tests**: No live API calls in CI

### Before Submitting

Run the full CI check suite locally:

```bash
pytest tests/ -v --cov=vetch --cov-fail-under=70
ruff check src/ tests/
mypy src/ --strict
```

## Contributing Energy Data

Energy estimates in `src/vetch/registry/energy.json` use a tiered system:

| Tier | Source | Uncertainty |
|------|--------|-------------|
| 0 | Direct hardware measurement | ±20% |
| 1 | Vendor-published data | ±50% |
| 2 | Validated research | ±100% |
| 3 | Architecture-based estimate | Order of magnitude |

### Adding a New Model

1. Add the entry to `src/vetch/registry/energy.json`:
   ```json
   "model-name": {
     "wh_per_1k_input": 0.1,
     "wh_per_1k_output": 0.3,
     "tier": 3,
     "architecture": "dense",
     "total_params_b": 70,
     "active_params_b": 70,
     "quantization": "bf16",
     "basis": "Explain your methodology"
   }
   ```

2. Add aliases in `src/vetch/registry/aliases.json` for variant names.

3. Add pricing in `src/vetch/registry/pricing.json` if applicable.

4. Run `vetch methodology --contribute` for detailed guidelines.

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with tests
3. Run the full CI suite locally
4. Submit a PR with a clear description
5. Address review feedback

### Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(vetch): add Bedrock provider support
fix(session): handle concurrent async agents
docs(readme): update provider compatibility table
test(azure): add region mapping tests
chore(deps): update ruff to v0.5.0
```

## Code Style

- **Formatter**: ruff format
- **Linter**: ruff check
- **Type checker**: mypy --strict
- **Line length**: 100 characters
- **Python version**: 3.9+ compatible syntax

## Architecture

```
src/vetch/
├── __init__.py          # Public API (wrap, instrument, Session, etc.)
├── wrapper.py           # VetchContext - core tracking context manager
├── session.py           # Session aggregation for agentic AI
├── calculation.py       # Energy, carbon, cost calculations
├── context.py           # ContextVar management
├── providers/           # SDK-specific wrappers
│   ├── openai.py
│   ├── azure_openai.py
│   ├── anthropic.py
│   └── vertexai.py
├── registry/            # Model data (energy, pricing, aliases)
│   ├── energy.json
│   ├── pricing.json
│   ├── aliases.json
│   └── remote.py        # Dynamic registry fetching
├── sensing/             # Grid carbon and hardware sensing
├── budget.py            # Budget alerts
├── otel.py              # OpenTelemetry export
└── cli.py               # CLI commands
```

### Key Design Principles

1. **Fail-Open**: Vetch errors never crash the host application
2. **Fail-Loud**: `signal_quality` and `vetch_warnings` expose data quality
3. **Privacy-First**: Zero access to prompt/completion content
4. **Zero Dependencies**: stdlib only at runtime

## License

By contributing, you agree that your contributions will be licensed
under the Apache 2.0 License.
