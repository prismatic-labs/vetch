# Wrapper.py Refactoring Plan

**Current State:** `wrapper.py` is 955 lines doing too many things
**Goal:** Split into coherent, focused modules with clear responsibilities
**Priority:** P2 (after security fixes and tests)

---

## Current Issues

### 1. **Single Responsibility Violation**

`wrapper.py` handles:
- Context management (`VetchContext.__enter__/__exit__`)
- SDK patching (OpenAI, Anthropic, etc.)
- Event calculation (energy, carbon, cost)
- Event emission (JSON, OTLP, storage)
- Budget tracking and enforcement
- Session/CI tracking
- Region inference
- Error handling and issue URL generation

### 2. **Testing Challenges**

- Hard to unit test individual concerns
- Mock setup is complex (many dependencies)
- Integration tests are slow (need to test everything together)

### 3. **Maintenance Burden**

- Changes to event calculation affect patching logic
- Hard to add new providers without modifying core logic
- Difficult for new contributors to understand flow

---

## Proposed Module Structure

```
src/vetch/
├── wrapper.py                # Public API (wrap() function, minimal logic)
├── _context.py               # VetchContext class (lifecycle management)
├── _calculation.py           # Event calculation (energy, carbon, cost) [EXISTS]
├── _emission.py              # Event emission orchestration
├── _patching.py              # SDK patching coordinator
├── _budget.py                # Budget tracking and alerts [EXISTS as budget.py]
├── _tracking.py              # Session/CI tracking
├── _inference.py             # Region/provider inference helpers
└── _security.py              # Security utilities [NEW - CREATED]
```

---

## Detailed Refactoring

### Phase 1: Extract Helper Modules (Low Risk)

#### 1.1 Create `_inference.py`

**Extracts:** Region inference logic

**Before:**
```python
# wrapper.py lines 78-106
def _infer_region() -> tuple[str | None, str | None]:
    """Infer region from environment and cloud metadata."""
    # ... 30 lines of inference logic
```

**After:**
```python
# src/vetch/_inference.py
def infer_region_from_environment() -> tuple[str | None, str | None]:
    """Infer region from AWS/GCP/Azure environment."""
    pass

def infer_provider_from_model(model: str) -> str | None:
    """Infer provider from model name (gpt-4 -> openai)."""
    pass
```

**wrapper.py imports:**
```python
from vetch._inference import infer_region_from_environment
```

---

#### 1.2 Create `_tracking.py`

**Extracts:** Session and CI event tracking

**Before:**
```python
# wrapper.py has inline session/CI tracking calls
from vetch.session import get_active_session, track_session_event
from vetch.ci import track_ci_event
```

**After:**
```python
# src/vetch/_tracking.py
def track_event_in_session(event: InferenceEvent) -> None:
    """Track event in active session if exists."""
    try:
        from vetch.session import get_active_session, track_session_event
        session = get_active_session()
        if session:
            track_session_event(event)
    except Exception:
        pass  # Fail-open

def track_event_in_ci(event: InferenceEvent) -> None:
    """Track event for CI environment if detected."""
    try:
        from vetch.ci import track_ci_event
        track_ci_event(event)
    except Exception:
        pass  # Fail-open
```

---

#### 1.3 Create `_emission.py`

**Extracts:** Event emission orchestration

**Before:**
```python
# wrapper.py lines 800-850 (_emit_event method)
def _emit_event(self, error: bool = False, ...):
    # Budget checks
    # Storage
    # Session tracking
    # CI tracking
    # OpenTelemetry
    # OTLP export
    # JSON emission
```

**After:**
```python
# src/vetch/_emission.py
class EventEmitter:
    """Orchestrates emission of inference events to all configured outputs."""

    def __init__(self, emit_json: bool = True, emit_otel: bool = True):
        self.emit_json = emit_json
        self.emit_otel = emit_otel

    def emit(self, event: InferenceEvent) -> None:
        """Emit event to all configured outputs (fail-open)."""
        # Budget enforcement (with alerts)
        self._enforce_budgets(event)

        # Storage (if enabled)
        self._store_event(event)

        # Session tracking (if active session)
        self._track_session(event)

        # CI tracking (if CI environment)
        self._track_ci(event)

        # OpenTelemetry span decoration
        if self.emit_otel:
            self._attach_to_otel_span(event)

        # OTLP export (if configured)
        if self.emit_otel:
            self._export_otlp(event)

        # JSON emission (default output)
        if self.emit_json:
            self._emit_json(event)

    def _enforce_budgets(self, event: InferenceEvent) -> None:
        """Check budgets and add alerts to event."""
        try:
            from vetch.budget import check_budget
            # ... budget logic
        except Exception:
            pass  # Fail-open

    # ... other helper methods
```

**Usage in wrapper.py:**
```python
from vetch._emission import EventEmitter

class VetchContext:
    def __init__(self, emit: bool = True, ...):
        self._emitter = EventEmitter(emit_json=emit, emit_otel=True)

    def _emit_event(self, ...):
        event = self._event
        self._emitter.emit(event)
```

---

### Phase 2: Extract SDK Patching (Medium Risk)

#### 2.1 Create `_patching.py`

**Extracts:** SDK patching coordinator

**Before:**
```python
# wrapper.py lines 388-454 (_setup_patches, _cleanup_patches)
def _setup_patches(self) -> None:
    """Set up SDK method patches."""
    # OpenAI patching
    if self._openai_client:
        from vetch.providers.openai import patch_openai_client
        patch_openai_client(self._openai_client)
    # ... other providers
```

**After:**
```python
# src/vetch/_patching.py
class PatchManager:
    """Manages patching and unpatching of AI SDK clients."""

    def __init__(self):
        self._patched_clients: list[tuple[str, Any]] = []

    def patch_client(self, client: Any, provider: str | None = None) -> bool:
        """Auto-detect and patch a client."""
        # Auto-detect provider if not specified
        if provider is None:
            provider = self._detect_provider(client)

        if provider == "openai":
            from vetch.providers.openai import patch_openai_client
            success = patch_openai_client(client)
        elif provider == "anthropic":
            from vetch.providers.anthropic import patch_anthropic_client
            success = patch_anthropic_client(client)
        # ... other providers

        if success:
            self._patched_clients.append((provider, client))
        return success

    def unpatch_all(self) -> None:
        """Unpatch all clients."""
        for provider, client in self._patched_clients:
            self._unpatch_client(client, provider)
        self._patched_clients.clear()

    def _detect_provider(self, client: Any) -> str:
        """Auto-detect provider from client object."""
        module = type(client).__module__
        if "openai" in module:
            return "openai"
        elif "anthropic" in module:
            return "anthropic"
        # ... etc
        return "unknown"
```

---

### Phase 3: Slim Down VetchContext (High Risk)

#### 3.1 Create `_context.py`

Move `VetchContext` class to its own module with delegated responsibilities.

**Before:**
```python
# wrapper.py: VetchContext has 500+ lines
class VetchContext:
    def __init__(self, ...): ...
    def __enter__(self): ...
    def __exit__(self, ...): ...
    def _setup_patches(self): ...
    def _cleanup_patches(self): ...
    def _emit_event(self): ...
    # ... 20+ methods
```

**After:**
```python
# src/vetch/_context.py
from vetch._emission import EventEmitter
from vetch._patching import PatchManager
from vetch._tracking import TrackingContext

class VetchContext:
    """Context manager for Vetch observability (slimmed down)."""

    def __init__(
        self,
        emit: bool = True,
        region: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        tags: dict[str, str] | None = None,
        # ... other params
    ):
        self._emit = emit
        self._region = region
        self._model = model
        self._provider = provider
        self._tags = tags or {}

        # Delegates
        self._emitter = EventEmitter(emit_json=emit)
        self._patch_manager = PatchManager()
        self._tracking_ctx: TrackingContext | None = None

        # Event state
        self._event: InferenceEvent = {}

    def __enter__(self) -> VetchContext:
        """Enter context: set up tracking and patches."""
        self._setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context: emit event and clean up."""
        self._teardown(exc_type)

    def _setup(self) -> None:
        """Initialize tracking and patches."""
        # Set up tracking context
        from vetch._tracking import TrackingContext
        self._tracking_ctx = TrackingContext()
        self._tracking_ctx.start()

        # Patch SDKs
        self._patch_manager.patch_client(self._openai_client, "openai")
        # ...

    def _teardown(self, exc_type) -> None:
        """Calculate metrics, emit event, clean up."""
        try:
            # Calculate event metrics
            self._calculate_metrics()

            # Emit event
            self._emitter.emit(self._event)
        except Exception as e:
            # Handle emission failure (sanitized issue URL)
            self._handle_emission_error(e)
        finally:
            # Clean up patches
            self._patch_manager.unpatch_all()

            # Stop tracking
            if self._tracking_ctx:
                self._tracking_ctx.stop()

    def _calculate_metrics(self) -> None:
        """Calculate energy, carbon, and cost for the event."""
        from vetch.calculation import (
            calculate_carbon,
            calculate_cost,
            calculate_energy,
        )
        # ... calculation logic (stays mostly the same)
```

**wrapper.py becomes:**
```python
# src/vetch/wrapper.py (public API only)
from vetch._context import VetchContext

def wrap(
    emit: bool = True,
    region: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    tags: dict[str, str] | None = None,
) -> VetchContext:
    """Create a Vetch context manager for LLM observability.

    Usage::

        with vetch.wrap() as ctx:
            response = client.chat.completions.create(...)

        # Access metrics
        energy_wh = ctx.event['estimated_energy_wh']
        carbon_g = ctx.event['estimated_carbon_g']
    """
    return VetchContext(
        emit=emit,
        region=region,
        model=model,
        provider=provider,
        tags=tags,
    )

# Re-export for backwards compatibility
__all__ = ['wrap', 'VetchContext']
```

---

## Migration Strategy

### Step 1: Create New Modules (No Breaking Changes)

1. Create `_inference.py`, `_tracking.py`, `_emission.py`, `_patching.py`
2. Keep wrapper.py unchanged (both implementations exist)
3. Write unit tests for new modules
4. Ensure 100% test coverage on new modules

### Step 2: Update Wrapper to Use New Modules

1. Import from new modules in wrapper.py
2. Replace inline logic with delegated calls
3. Run full integration test suite
4. Check that existing tests still pass (no behavior change)

### Step 3: Clean Up Wrapper

1. Remove now-redundant code from wrapper.py
2. Update imports and docstrings
3. Run tests again
4. Benchmark performance (should be same or better)

### Step 4: Update Tests

1. Add unit tests for each new module
2. Update integration tests to test through public API
3. Add mocking helpers for testing code that uses VetchContext

---

## Testing Strategy

### Unit Tests (New Modules)

```python
# tests/unit/test_inference.py
def test_infer_region_from_aws_metadata():
    """Test AWS region inference."""
    with patch.dict(os.environ, {"AWS_REGION": "us-east-1"}):
        region, source = infer_region_from_environment()
        assert region == "us-east-1"
        assert source == "AWS_REGION"

# tests/unit/test_emission.py
def test_emitter_fail_open_on_storage_error():
    """Test that storage errors don't crash emission."""
    emitter = EventEmitter()
    with patch("vetch.storage.store_event", side_effect=Exception("DB error")):
        # Should not raise
        emitter.emit({"model": "gpt-4"})

# tests/unit/test_patching.py
def test_patch_manager_auto_detects_openai():
    """Test auto-detection of OpenAI clients."""
    from openai import OpenAI
    client = OpenAI(api_key="fake")
    manager = PatchManager()
    assert manager._detect_provider(client) == "openai"
```

### Integration Tests (Public API)

```python
# tests/integration/test_wrapper_refactored.py
def test_wrap_with_openai_still_works():
    """Test that refactored code works end-to-end."""
    from openai import OpenAI
    import vetch

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    with vetch.wrap(emit=False) as ctx:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hi"}],
        )

    # Event should be populated
    assert ctx.event is not None
    assert ctx.event["model"] == "gpt-3.5-turbo"
    assert ctx.event["estimated_energy_wh"] > 0
```

---

## Rollback Plan

If refactoring introduces bugs:

1. Revert commits (git revert)
2. Keep new modules (they're additive, not breaking)
3. Go back to monolithic wrapper.py
4. Fix issues in new modules
5. Try again when tests are green

---

## Success Metrics

### Code Quality

- [ ] Wrapper.py reduced from 955 lines to < 300 lines
- [ ] Each new module has < 300 lines
- [ ] Each module has single, clear responsibility
- [ ] Test coverage remains ≥ 90%

### Performance

- [ ] No performance regression (benchmark with hyperfine)
- [ ] Memory usage unchanged
- [ ] Import time not significantly increased

### Maintainability

- [ ] New contributors can understand flow in < 30 minutes
- [ ] Adding a new provider requires editing only _patching.py
- [ ] Emission logic changes don't affect patching

---

## Timeline Estimate

**Conservative (Safe):**
- Phase 1 (Extract Helpers): 1-2 days
- Phase 2 (Extract Patching): 2-3 days
- Phase 3 (Slim Context): 3-4 days
- Testing & Polish: 2-3 days
- **Total: 8-12 days** (full-time work)

**Aggressive (Risky):**
- All phases: 3-4 days
- Testing: 1 day
- **Total: 4-5 days** (high risk of bugs)

**Recommended:** Conservative approach with code review at each phase.

---

## Open Questions

1. **Should _context.py expose VetchContext publicly?**
   - Option A: Yes, for advanced users who want to customize
   - Option B: No, only expose through wrapper.wrap() factory

2. **Should we keep wrapper.py or rename to api.py?**
   - Option A: Keep wrapper.py for backwards compatibility
   - Option B: Rename to api.py (clearer intent)

3. **What about backwards compatibility?**
   - All public APIs stay the same (`wrap()`, `VetchContext`)
   - Internal imports change but users don't import those

---

## Alternatives Considered

### Alternative 1: Don't Refactor

**Pros:**
- No risk of introducing bugs
- Works today

**Cons:**
- Technical debt grows
- Harder to maintain over time
- New contributors struggle

**Verdict:** Not recommended for long-term health

### Alternative 2: Partial Refactor (Just Extract Helpers)

**Pros:**
- Lower risk than full refactor
- Still reduces wrapper.py size by ~200 lines
- Incremental improvement

**Cons:**
- Doesn't fully solve the problem
- VetchContext still too large

**Verdict:** Good compromise if time is limited

### Alternative 3: Complete Rewrite

**Pros:**
- Clean slate
- Can redesign from scratch

**Cons:**
- Very high risk
- Breaks backwards compatibility
- Weeks of work

**Verdict:** Only for v2.0 major release

---

## Recommendation

**Phase 1 (Low Risk) - Do Now:**
- Extract `_inference.py`, `_tracking.py`, `_emission.py`
- Reduces wrapper.py by ~300 lines
- Low risk, high reward
- Estimated: 2-3 days

**Phase 2 (Medium Risk) - Do Before Beta:**
- Extract `_patching.py`
- Further reduces wrapper.py by ~150 lines
- Medium risk, good payoff
- Estimated: 2-3 days

**Phase 3 (High Risk) - Do Before v1.0:**
- Create `_context.py` and slim down VetchContext
- Reduces wrapper.py to < 200 lines
- High risk, requires extensive testing
- Estimated: 4-5 days

**Total Time to Full Refactor:** 8-11 days (conservative)
