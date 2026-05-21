"""Session aggregation for tracking multiple LLM calls.

This module provides hierarchical session tracking for agentic AI patterns
(CrewAI, AutoGPT, LangGraph). Sessions accumulate energy, cost, and carbon
across nested wrap() calls.

Supports:
- Hierarchical sessions (parent_session_id)
- Thread-safe accumulation
- Distributed session propagation via HTTP headers

Example::

    with vetch.Session(tags={"agent": "researcher"}) as session:
        with vetch.wrap() as ctx1:
            response1 = client.chat.completions.create(...)
        with vetch.wrap() as ctx2:
            response2 = client.chat.completions.create(...)

    print(f"Total energy: {session.total_energy_wh} Wh")
    print(f"Total cost: ${session.total_cost_usd}")
"""

from __future__ import annotations

import logging
import threading
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, cast

from vetch import __version__
from vetch.emitter import emit_event
from vetch.schema import SCHEMA_VERSION
from vetch.stats import _RECENT_WINDOW, SessionStats

if TYPE_CHECKING:
    from vetch.advisory import Advisory
    from vetch.schema import InferenceEvent

# Lazy detection threshold: STALL-001 needs at least 10 calls to fire,
# so we skip the advisory cycle entirely until we have enough history.
# Half the rolling window is a safe lower bound for any future advisory too.
_STALL_DETECTION_MIN_CALLS = _RECENT_WINDOW // 2  # = 10

logger = logging.getLogger(__name__)

# ContextVar to track active session (thread-safe and async-safe)
_active_session: ContextVar[Session | None] = ContextVar("vetch_session", default=None)

# Safety limits to prevent OOM in long-running agentic loops
DEFAULT_MAX_CALLS = 10_000
DEFAULT_MAX_METADATA_ITEMS = 100

# HTTP header names for distributed propagation (W3C TraceContext style)
HEADER_SESSION_ID = "X-Vetch-Session-Id"
HEADER_PARENT_SESSION_ID = "X-Vetch-Parent-Session-Id"


def get_active_session() -> Session | None:
    """Get the currently active session, if any.

    Returns:
        Active Session or None.
    """
    return _active_session.get()


@dataclass
class SessionEvent:
    """Event emitted when a session completes."""

    schema_version: str
    vetch_version: str
    event_type: str  # "session_complete"
    session_id: str
    parent_session_id: str | None
    timestamp: str
    call_count: int
    total_energy_wh: float
    total_carbon_g: float
    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int
    total_cache_read_tokens: int
    total_cache_creation_tokens: int
    duration_ms: float | None
    tags: dict[str, str] | None
    models_used: list[str]
    providers_used: list[str]
    errors: int


class Session:
    """Session for aggregating multiple LLM inference calls.

    Sessions track cumulative energy, cost, and carbon across multiple
    wrap() calls. They support hierarchical nesting for agentic patterns.

    Attributes:
        session_id: Unique identifier for this session.
        parent_session_id: ID of parent session (for nested agents).
        tags: Key-value pairs for session attribution.
        total_energy_wh: Accumulated energy in Wh.
        total_cost_usd: Accumulated cost in USD.
        total_carbon_g: Accumulated carbon in gCO2e.
        call_count: Number of inference calls in this session.
    """

    def __init__(
        self,
        session_id: str | None = None,
        parent_session_id: str | None = None,
        tags: dict[str, str] | None = None,
        emit: bool = True,
        max_calls: int = DEFAULT_MAX_CALLS,
        advisory_thresholds: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """Initialize a session.

        Args:
            session_id: Custom session ID (auto-generated if not provided).
            parent_session_id: ID of parent session for hierarchical tracking.
            tags: Key-value pairs for session attribution.
            emit: If True, emit session_complete event on exit.
            max_calls: Maximum number of calls to track before stopping
                accumulation. Prevents OOM in long-running agentic loops.
                Set to 0 for unlimited (not recommended).
            advisory_thresholds: Optional per-session advisory threshold
                overrides. Use this to scope detectors by route, workflow, or
                tenant without changing process-wide defaults.
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.parent_session_id = parent_session_id
        self.tags = dict(tags) if tags else None
        self._emit = emit
        self._max_calls = max_calls

        # Per-session stats for advisory analysis (stall detection, etc.)
        # Isolated from the global singleton — safe for multi-user contexts.
        self.advisory_thresholds = (
            {code: dict(values) for code, values in advisory_thresholds.items()}
            if advisory_thresholds
            else None
        )
        self.stats = SessionStats(advisory_thresholds=self.advisory_thresholds)

        # v0.4.0: Stall circuit breaker state. Set by register_event when
        # STALL-001 fires; read by provider wrappers via _stall.apply_stall_action.
        # Stays True for the rest of the session unless clear_stall() is called.
        self.stall_triggered: bool = False
        self.stall_advisory: Advisory | None = None

        # Accumulation state (thread-safe)
        self._lock = threading.Lock()
        self._total_energy_wh: float = 0.0
        self._total_carbon_g: float = 0.0
        self._total_cost_usd: float = 0.0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cache_read_tokens: int = 0
        self._total_cache_creation_tokens: int = 0
        self._call_count: int = 0
        self._errors: int = 0
        self._saturated: bool = False
        self._models_used: set[str] = set()
        self._providers_used: set[str] = set()

        # Timing
        self._start_time: float | None = None
        self._end_time: float | None = None

        # Context token for cleanup
        self._token: Any = None

        # Check for parent session if not explicitly provided
        if self.parent_session_id is None:
            parent = get_active_session()
            if parent is not None:
                self.parent_session_id = parent.session_id

    @property
    def total_energy_wh(self) -> float:
        """Total accumulated energy in Wh."""
        with self._lock:
            return self._total_energy_wh

    @property
    def total_carbon_g(self) -> float:
        """Total accumulated carbon in gCO2e."""
        with self._lock:
            return self._total_carbon_g

    @property
    def total_cost_usd(self) -> float:
        """Total accumulated cost in USD."""
        with self._lock:
            return self._total_cost_usd

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens across all calls."""
        with self._lock:
            return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        """Total output tokens across all calls."""
        with self._lock:
            return self._total_output_tokens

    @property
    def call_count(self) -> int:
        """Number of inference calls in this session."""
        with self._lock:
            return self._call_count

    @property
    def models_used(self) -> list[str]:
        """List of models used in this session."""
        with self._lock:
            return sorted(self._models_used)

    @property
    def providers_used(self) -> list[str]:
        """List of providers used in this session."""
        with self._lock:
            return sorted(self._providers_used)

    @property
    def total_cache_read_tokens(self) -> int:
        """Total cache read tokens across all calls (prompt caching)."""
        with self._lock:
            return self._total_cache_read_tokens

    @property
    def total_cache_creation_tokens(self) -> int:
        """Total cache creation tokens across all calls."""
        with self._lock:
            return self._total_cache_creation_tokens

    @property
    def duration_ms(self) -> float | None:
        """Session duration in milliseconds."""
        if self._start_time is None:
            return None
        end = self._end_time or __import__("time").perf_counter()
        return (end - self._start_time) * 1000

    def register_event(self, event: InferenceEvent) -> None:
        """Register an inference event with this session.

        Called internally by VetchContext on exit. Respects max_calls
        limit to prevent OOM in long-running agentic loops.

        Also updates the per-session ``SessionStats`` instance for
        advisory analysis (stall detection, caching, RAG patterns).

        Args:
            event: The inference event to register.
        """
        # Per-session stats are always updated. SessionStats has its own lock,
        # and the session lock keeps circuit-breaker state in step with counts.
        with self._lock:
            self._call_count += 1
            self.stats.update(event)

            # v0.4.0: Lazy STALL-001 detection. STALL-001 needs at least 10
            # calls to fire (see advisory.py), so we skip the advisory cycle
            # entirely until we've accumulated enough history. Once tripped,
            # the flag stays set until clear_stall() — no thrashing.
            if (
                not self.stall_triggered
                and self.stats.total_requests >= _STALL_DETECTION_MIN_CALLS
            ):
                try:
                    from vetch.advisory import generate_advisories

                    for adv in generate_advisories(self.stats):
                        if adv.code == "STALL-001":
                            self.stall_triggered = True
                            self.stall_advisory = adv
                            break
                except Exception as exc:
                    # Fail-open: if detection itself errors, log and move on.
                    logger.debug("STALL-001 detection failed: %s", exc)

            # Safety: stop accumulating after max_calls to prevent OOM
            if self._max_calls > 0 and self._call_count > self._max_calls:
                if not self._saturated:
                    self._saturated = True
                    logger.warning(
                        f"Session '{self.session_id}' reached max_calls "
                        f"limit ({self._max_calls}). Metrics will still be "
                        f"counted but metadata sets are frozen."
                    )
                # Still accumulate scalar metrics (bounded), skip set growth
                energy = event.get("estimated_energy_wh")
                if energy is not None:
                    self._total_energy_wh += energy
                carbon = event.get("estimated_carbon_g")
                if carbon is not None:
                    self._total_carbon_g += carbon
                cost = event.get("estimated_cost_usd")
                if cost is not None:
                    self._total_cost_usd += cost
                if event.get("error"):
                    self._errors += 1
                return

            # Accumulate metrics
            energy = event.get("estimated_energy_wh")
            if energy is not None:
                self._total_energy_wh += energy

            carbon = event.get("estimated_carbon_g")
            if carbon is not None:
                self._total_carbon_g += carbon

            cost = event.get("estimated_cost_usd")
            if cost is not None:
                self._total_cost_usd += cost

            # Track tokens
            usage = event.get("usage")
            if usage:
                text = usage.get("text")
                if text:
                    self._total_input_tokens += text.get("input_tokens", 0)
                    self._total_output_tokens += text.get("output_tokens", 0)

            # Track cache metrics (critical for agentic AI efficiency)
            cache_read = event.get("cache_read_tokens")
            if cache_read is not None:
                self._total_cache_read_tokens += cache_read

            cache_creation = event.get("cache_creation_tokens")
            if cache_creation is not None:
                self._total_cache_creation_tokens += cache_creation

            # Track models and providers (capped to prevent unbounded growth)
            model = event.get("model", "unknown")
            if model != "unknown" and len(self._models_used) < DEFAULT_MAX_METADATA_ITEMS:
                self._models_used.add(model)

            provider = event.get("provider", "unknown")
            if provider != "unknown" and len(self._providers_used) < DEFAULT_MAX_METADATA_ITEMS:
                self._providers_used.add(provider)

            # Track errors
            if event.get("error"):
                self._errors += 1

    def clear_stall(self) -> None:
        """Reset the stall circuit-breaker flag so it can re-arm.

        Use after a human-in-the-loop fix (corrected prompt, fixed retriever,
        updated agent instructions) to resume normal operation in the same
        session. Subsequent calls behave as before — the next stall pattern
        will trip the breaker again.

        Thread-safe.

        Example::

            try:
                with vetch.Session() as session:
                    while True:
                        agent.step()  # raises StallDetected
            except vetch.StallDetected:
                fix_the_prompt()
                session.clear_stall()
                # Loop can now resume; next stall will re-trigger.
        """
        with self._lock:
            self.stall_triggered = False
            self.stall_advisory = None

    def inject_headers(self, headers: dict[str, str]) -> dict[str, str]:
        """Inject session IDs into HTTP headers for distributed tracing.

        Use this to propagate session context across service boundaries
        (e.g., FastAPI to Celery worker).

        Args:
            headers: Existing headers dict to update.

        Returns:
            Updated headers dict with session IDs added.

        Example::

            headers = session.inject_headers({})
            response = requests.post(worker_url, headers=headers)
        """
        headers[HEADER_SESSION_ID] = self.session_id
        if self.parent_session_id:
            headers[HEADER_PARENT_SESSION_ID] = self.parent_session_id
        return headers

    @classmethod
    def from_headers(
        cls,
        headers: dict[str, str],
        tags: dict[str, str] | None = None,
        emit: bool = True,
        resume: bool = False,
        advisory_thresholds: dict[str, dict[str, float]] | None = None,
    ) -> Session:
        """Create a session from HTTP headers for distributed tracing.

        Use this in workers/microservices to continue a session started
        elsewhere.

        Args:
            headers: HTTP headers containing session IDs.
            tags: Additional tags for this segment.
            emit: If True, emit session_complete event on exit.
            resume: If True, use the same session_id from headers (for
                aggregating into a single session). If False (default),
                create a new child session linked to the parent.
            advisory_thresholds: Optional per-session advisory threshold
                overrides for this worker/request segment.

        Returns:
            Session instance (resumed or new child).

        Example::

            # In Celery worker - create child session (default)
            @celery_app.task
            def process_task(data, headers):
                with Session.from_headers(headers) as session:
                    with vetch.wrap() as ctx:
                        response = client.chat.completions.create(...)

            # In Task Runner - resume same session
            @celery_app.task
            def run_task(data, headers):
                with Session.from_headers(headers, resume=True) as session:
                    # Same session_id as dispatcher - events aggregate together
                    with vetch.wrap() as ctx:
                        response = client.chat.completions.create(...)
        """
        session_id_from_header = headers.get(HEADER_SESSION_ID)
        parent_id = headers.get(HEADER_PARENT_SESSION_ID)

        if resume and session_id_from_header:
            # Resume: use the same session_id, keep parent chain
            return cls(
                session_id=session_id_from_header,
                parent_session_id=parent_id,
                tags=tags,
                emit=emit,
                advisory_thresholds=advisory_thresholds,
            )
        else:
            # Create new child session linked to parent
            return cls(
                session_id=None,  # Generate new ID
                parent_session_id=session_id_from_header,
                tags=tags,
                emit=emit,
                advisory_thresholds=advisory_thresholds,
            )

    def __enter__(self) -> Session:
        """Enter session context."""
        import time

        self._start_time = time.perf_counter()
        self._token = _active_session.set(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit session context and emit summary event."""
        import time

        self._end_time = time.perf_counter()

        # Reset context
        if self._token is not None:
            _active_session.reset(self._token)

        # Emit session complete event
        if self._emit:
            self._emit_session_event()

    async def __aenter__(self) -> Session:
        """Async enter session context."""
        import time

        self._start_time = time.perf_counter()
        self._token = _active_session.set(self)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async exit session context and emit summary event."""
        import time

        self._end_time = time.perf_counter()

        # Reset context
        if self._token is not None:
            _active_session.reset(self._token)

        # Emit session complete event
        if self._emit:
            self._emit_session_event()

    def _emit_session_event(self) -> None:
        """Emit session complete event."""
        try:
            event: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "vetch_version": __version__,
                "event_type": "session_complete",
                "session_id": self.session_id,
                "parent_session_id": self.parent_session_id,
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "call_count": self._call_count,
                "total_energy_wh": self._total_energy_wh,
                "total_carbon_g": self._total_carbon_g,
                "total_cost_usd": self._total_cost_usd,
                "total_input_tokens": self._total_input_tokens,
                "total_output_tokens": self._total_output_tokens,
                "total_cache_read_tokens": self._total_cache_read_tokens,
                "total_cache_creation_tokens": self._total_cache_creation_tokens,
                "duration_ms": self.duration_ms,
                "tags": self.tags,
                "models_used": self.models_used,
                "providers_used": self.providers_used,
                "errors": self._errors,
            }
            # Cast to satisfy type checker - emit_event handles both event types
            emit_event(cast("InferenceEvent", event))
        except Exception as e:
            logger.debug(f"Failed to emit session event: {e}")

    def to_dict(self) -> dict[str, Any]:
        """Convert session to dictionary for inspection.

        Returns:
            Dict with session state.
        """
        return {
            "session_id": self.session_id,
            "parent_session_id": self.parent_session_id,
            "call_count": self.call_count,
            "total_energy_wh": self.total_energy_wh,
            "total_carbon_g": self.total_carbon_g,
            "total_cost_usd": self.total_cost_usd,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cache_read_tokens": self.total_cache_read_tokens,
            "total_cache_creation_tokens": self.total_cache_creation_tokens,
            "duration_ms": self.duration_ms,
            "tags": self.tags,
            "models_used": self.models_used,
            "providers_used": self.providers_used,
            "errors": self._errors,
        }
