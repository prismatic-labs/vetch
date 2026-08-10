"""OpenTelemetry integration for Vetch metrics.

This module provides two integration modes:

1. **Span Decoration** (default): Attach Vetch metrics to existing OTel spans.
   Use when you already have OpenTelemetry tracing configured.

2. **OTLP Export**: Export Vetch metrics directly to any OTLP-compatible backend
   (Datadog, Honeycomb, Grafana, Jaeger, etc.). Use when you want Vetch to
   manage its own telemetry export.

Environment Variables:
    OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint (e.g., "http://localhost:4317")
    OTEL_EXPORTER_OTLP_HEADERS: Comma-separated key=value pairs for auth
    VETCH_OTEL_SERVICE_NAME: Service name for exported spans (default: "vetch")
    VETCH_OTEL_EXPORT: Set to "true" to enable automatic OTLP export
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import queue
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vetch.schema import InferenceEvent

logger = logging.getLogger(__name__)

# Track if OTLP exporter is configured
_otlp_configured = False
_tracer: Any = None
_meter: Any = None

# Metric instruments (created lazily)
_energy_histogram: Any = None
_carbon_histogram: Any = None
_cost_histogram: Any = None
_request_counter: Any = None
_advisory_counter: Any = None        # vetch.advisories_fired{code, severity, action}
_advisory_waste_histogram: Any = None  # vetch.advisory.waste_usd{code}

# Background export queue (non-blocking export)
# Configurable via VETCH_EXPORT_QUEUE_SIZE (default: 1000)
_DEFAULT_QUEUE_SIZE = 1000
_queue_size = int(os.environ.get("VETCH_EXPORT_QUEUE_SIZE", str(_DEFAULT_QUEUE_SIZE)))
_export_queue: queue.Queue[InferenceEvent | None] = queue.Queue(maxsize=_queue_size)
_export_thread: threading.Thread | None = None

# Dropped events tracking (for observability and backpressure detection)
_dropped_events_count: int = 0
_last_drop_warning: float = 0.0  # Monotonic timestamp of last warning
_drop_warning_interval: float = 60.0  # Warn at most once per minute
_shutdown_event = threading.Event()

# Error rate limiting (circuit breaker for logging)
_ERROR_LOG_INTERVAL = 300  # Log errors at most once per 5 minutes
_last_error_log_time: float = 0.0
_error_count_since_log: int = 0
_error_lock = threading.Lock()


def attach_to_otel_span(event: InferenceEvent) -> bool:
    """Attach Vetch metrics to the current OpenTelemetry span.

    Checks if 'opentelemetry' is installed and if there is an active
    span in the current context.

    Args:
        event: The InferenceEvent containing metrics.

    Returns:
        True if successfully attached to a span, False otherwise.
    """
    import sys

    if "opentelemetry" not in sys.modules:
        return False

    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if not span.is_recording():
            return False

        # Attach headline metrics
        span.set_attribute("vetch.energy_wh", event.get("estimated_energy_wh") or 0.0)
        span.set_attribute("vetch.carbon_g", event.get("estimated_carbon_g") or 0.0)
        span.set_attribute("vetch.cost_usd", event.get("estimated_cost_usd") or 0.0)

        # Attach metadata
        span.set_attribute("vetch.model", event.get("model", "unknown"))
        span.set_attribute("vetch.provider", event.get("provider", "unknown"))
        span.set_attribute("vetch.region", event.get("region") or "unknown")
        span.set_attribute("vetch.signal_quality", event.get("signal_quality", "unknown"))
        span.set_attribute("vetch.model_match", event.get("model_match", "fallback"))
        span.set_attribute(
            "vetch.energy_completeness", event.get("energy_completeness", "complete")
        )
        cal_match = event.get("calibration_match")
        if cal_match:
            span.set_attribute("vetch.calibration_match", cal_match)

        # Attach energy tier for confidence
        span.set_attribute("vetch.energy_tier", event.get("energy_tier", 3))

        # Attach uncertainty
        uncertainty = event.get("energy_uncertainty_pct")
        if uncertainty is not None:
            span.set_attribute("vetch.energy_uncertainty_pct", uncertainty)

        # Attach token counts
        usage = event.get("usage")
        if usage:
            text = usage.get("text")
            if text:
                span.set_attribute("vetch.input_tokens", text.get("input_tokens", 0))
                span.set_attribute("vetch.output_tokens", text.get("output_tokens", 0))

        # Attach budget status
        if event.get("budget_exceeded"):
            span.set_attribute("vetch.budget_exceeded", True)

        # Attach cache hit status (if available)
        cache_hit = event.get("cache_read_tokens")
        if cache_hit is not None:
            span.set_attribute("vetch.cache_read_tokens", cache_hit)

        from vetch.capabilities import set_otel_capability_attributes

        set_otel_capability_attributes(span, event)

        return True

    except Exception as e:
        logger.debug(f"Failed to attach to OTel span: {e}")
        return False


def configure_otlp_export(
    endpoint: str | None = None,
    headers: dict[str, str] | None = None,
    service_name: str = "vetch",
) -> bool:
    """Configure OTLP export for Vetch metrics.

    Call this once at application startup to enable automatic export
    of Vetch metrics to any OTLP-compatible backend.

    Args:
        endpoint: OTLP endpoint (default: OTEL_EXPORTER_OTLP_ENDPOINT env var).
        headers: Auth headers (default: OTEL_EXPORTER_OTLP_HEADERS env var).
        service_name: Service name for traces/metrics (default: "vetch").

    Returns:
        True if configuration succeeded, False otherwise.

    Example::

        # Honeycomb
        configure_otlp_export(
            endpoint="https://api.honeycomb.io",
            headers={"x-honeycomb-team": "your-api-key"}
        )

        # Grafana Cloud
        configure_otlp_export(
            endpoint="https://otlp-gateway-prod-us-central-0.grafana.net/otlp",
            headers={"Authorization": "Basic base64-encoded-credentials"}
        )

        # Local Jaeger
        configure_otlp_export(endpoint="http://localhost:4317")
    """
    global _otlp_configured, _tracer, _meter
    global _energy_histogram, _carbon_histogram, _cost_histogram, _request_counter
    global _advisory_counter, _advisory_waste_histogram

    try:
        # Try importing OTel SDK components
        from opentelemetry import metrics, trace
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.metrics import (
            MeterProvider,
        )
        from opentelemetry.sdk.metrics.export import (
            PeriodicExportingMetricReader,
        )
        from opentelemetry.sdk.resources import (
            Resource,
        )
        from opentelemetry.sdk.trace import (
            TracerProvider,
        )
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
        )

    except ImportError as e:
        logger.warning(
            f"OpenTelemetry SDK not installed. Install with: "
            f"pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc. "
            f"Error: {e}"
        )
        return False

    try:
        # Resolve endpoint
        if endpoint is None:
            endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        if endpoint is None:
            logger.warning("No OTLP endpoint specified. Set OTEL_EXPORTER_OTLP_ENDPOINT.")
            return False

        # Resolve headers
        if headers is None:
            headers_str = os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", "")
            headers = {}
            for pair in headers_str.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    headers[k.strip()] = v.strip()

        # Create resource
        from vetch import __version__

        resource = Resource.create(
            {
                "service.name": service_name,
                "service.version": __version__,
                "vetch.sdk": True,
            }
        )

        # Configure tracing
        tracer_provider = TracerProvider(resource=resource)
        span_exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers or None)
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        trace.set_tracer_provider(tracer_provider)
        _tracer = trace.get_tracer("vetch", __version__)

        # Configure metrics
        metric_exporter = OTLPMetricExporter(endpoint=endpoint, headers=headers or None)
        metric_reader = PeriodicExportingMetricReader(
            metric_exporter, export_interval_millis=60000
        )
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)
        _meter = metrics.get_meter("vetch", __version__)

        # Create metric instruments
        _energy_histogram = _meter.create_histogram(
            "vetch.energy_wh",
            description="Energy consumption per inference (Wh)",
            unit="Wh",
        )
        _carbon_histogram = _meter.create_histogram(
            "vetch.carbon_g",
            description="Carbon emissions per inference (gCO2e)",
            unit="g",
        )
        _cost_histogram = _meter.create_histogram(
            "vetch.cost_usd",
            description="Cost per inference (USD)",
            unit="USD",
        )
        _request_counter = _meter.create_counter(
            "vetch.requests_total",
            description="Total inference requests",
            unit="1",
        )
        _advisory_counter = _meter.create_counter(
            "vetch.advisories_fired_total",
            description="Waste advisories fired (code, severity, action)",
            unit="1",
        )
        _advisory_waste_histogram = _meter.create_histogram(
            "vetch.advisory_waste_usd",
            description="Estimated waste cost when an advisory fires (USD)",
            unit="USD",
        )

        _otlp_configured = True

        # Start background export worker
        _start_export_worker()

        logger.info(f"OTLP export configured: {endpoint}")
        return True

    except Exception as e:
        logger.warning(f"Failed to configure OTLP export: {e}")
        return False


def _export_event_sync(event: InferenceEvent) -> bool:
    """Synchronously export an inference event via OTLP (internal).

    Creates a span for the inference and records metrics.
    Only works if configure_otlp_export() was called first.

    Args:
        event: The InferenceEvent to export.

    Returns:
        True if export succeeded, False otherwise.
    """
    if not _otlp_configured or _tracer is None:
        return False

    try:
        from opentelemetry import trace

        # Create span for this inference
        with _tracer.start_as_current_span(
            "llm.inference",
            kind=trace.SpanKind.CLIENT,
        ) as span:
            # Set span attributes
            span.set_attribute("llm.model", event.get("model", "unknown"))
            span.set_attribute("llm.provider", event.get("provider", "unknown"))

            # Vetch metrics
            energy = event.get("estimated_energy_wh") or 0.0
            carbon = event.get("estimated_carbon_g") or 0.0
            cost = event.get("estimated_cost_usd") or 0.0

            span.set_attribute("vetch.energy_wh", energy)
            span.set_attribute("vetch.carbon_g", carbon)
            span.set_attribute("vetch.cost_usd", cost)
            span.set_attribute("vetch.energy_tier", event.get("energy_tier", 3))
            span.set_attribute("vetch.signal_quality", event.get("signal_quality", "unknown"))
            span.set_attribute("vetch.model_match", event.get("model_match", "fallback"))
            span.set_attribute(
                "vetch.energy_completeness", event.get("energy_completeness", "complete")
            )
            cal_match = event.get("calibration_match")
            if cal_match:
                span.set_attribute("vetch.calibration_match", cal_match)
            span.set_attribute("vetch.region", event.get("region") or "unknown")

            # Token counts
            usage = event.get("usage")
            if usage:
                text = usage.get("text")
                if text:
                    span.set_attribute("llm.input_tokens", text.get("input_tokens", 0))
                    span.set_attribute("llm.output_tokens", text.get("output_tokens", 0))

            # Extended Thinking mode transparency
            model_name = event.get("model", "")
            if isinstance(model_name, str) and model_name.endswith("-thinking"):
                span.set_attribute("vetch.thinking_mode", True)

            # Cache energy saving
            cache_saving = event.get("cache_energy_saving_wh")
            if cache_saving is not None:
                span.set_attribute("vetch.cache_energy_saving_wh", float(cache_saving))

            # Cache cost saving
            cache_cost_saving = event.get("cache_cost_saving_usd")
            if cache_cost_saving is not None:
                span.set_attribute("vetch.cache_cost_saving_usd", float(cache_cost_saving))

            # Cache carbon saving
            cache_carbon_saving = event.get("cache_carbon_saving_g")
            if cache_carbon_saving is not None:
                span.set_attribute("vetch.cache_carbon_saving_g", float(cache_carbon_saving))

            # Budget status
            if event.get("budget_exceeded"):
                span.set_attribute("vetch.budget_exceeded", True)

            # Latency
            latency = event.get("latency_ms")
            if latency:
                span.set_attribute("llm.latency_ms", latency)

            # Error
            if event.get("error"):
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                span.set_attribute("error.type", event.get("error_type", "unknown"))

            # Tags as span attributes
            tags = event.get("tags")
            if tags:
                for k, v in tags.items():
                    span.set_attribute(f"vetch.tag.{k}", v)

            # Advisory signals — surface active waste advisories on the span
            # so traces show waste context without a separate advisory lookup.
            advisories = event.get("advisories")
            if advisories and isinstance(advisories, list):
                codes = [a["code"] for a in advisories if isinstance(a, dict) and "code" in a]
                if codes:
                    span.set_attribute("vetch.advisory_codes", ",".join(codes))
                    span.add_event(
                        "vetch.advisories_active",
                        {"codes": ",".join(codes)},
                    )

        # Record metrics
        model = event.get("model", "unknown")
        provider = event.get("provider", "unknown")
        region = event.get("region") or "unknown"

        attributes = {
            "model": model,
            "provider": provider,
            "region": region,
        }

        if _energy_histogram:
            _energy_histogram.record(energy, attributes)
        if _carbon_histogram:
            _carbon_histogram.record(carbon, attributes)
        if _cost_histogram:
            _cost_histogram.record(cost, attributes)
        if _request_counter:
            _request_counter.add(1, attributes)

        return True

    except Exception as e:
        logger.debug(f"Failed to export event via OTLP: {e}")
        return False


def _log_error_rate_limited(error: Exception) -> None:
    """Log OTLP errors at most once per 5 minutes to prevent log flooding."""
    global _last_error_log_time, _error_count_since_log
    import time

    with _error_lock:
        now = time.time()
        _error_count_since_log += 1

        if now - _last_error_log_time >= _ERROR_LOG_INTERVAL:
            if _error_count_since_log > 1:
                logger.warning(
                    f"OTLP export error ({_error_count_since_log} errors in last "
                    f"{_ERROR_LOG_INTERVAL}s): {error}"
                )
            else:
                logger.warning(f"OTLP export error: {error}")
            _last_error_log_time = now
            _error_count_since_log = 0


def _export_worker() -> None:
    """Background worker thread for OTLP export."""
    while not _shutdown_event.is_set():
        try:
            # Wait for event with timeout to allow shutdown check
            event = _export_queue.get(timeout=0.5)
            if event is None:  # Shutdown sentinel
                _export_queue.task_done()
                break
            try:
                _export_event_sync(event)
            finally:
                # Always mark task as done, even if export failed
                _export_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            _log_error_rate_limited(e)


def _start_export_worker() -> None:
    """Start the background export worker thread."""
    global _export_thread
    if _export_thread is not None and _export_thread.is_alive():
        return
    _shutdown_event.clear()
    _export_thread = threading.Thread(target=_export_worker, daemon=True, name="vetch-otlp-export")
    _export_thread.start()
    atexit.register(_shutdown_export_worker)


def _shutdown_export_worker() -> None:
    """Shutdown the background export worker gracefully.

    Drains the export queue to ensure all telemetry is transmitted before
    process exit. Critical for capturing final application state on SIGTERM.
    """
    global _export_thread
    if _export_thread is None:
        return

    # First, wait for queue to drain (all pending events exported)
    # This ensures we don't lose the "tail" of telemetry on shutdown
    # Use a timeout to prevent hanging if OTLP collector is slow/hung
    import time
    try:
        deadline = time.monotonic() + 5.0  # 5 second timeout
        while not _export_queue.empty() and time.monotonic() < deadline:
            time.sleep(0.1)  # Poll every 100ms
        # Final join with minimal timeout (queue should be drained by now)
        # Note: Queue.join() doesn't support timeout directly, so we use polling above
        if _export_queue.empty():
            _export_queue.join()  # Should return immediately if queue is empty
    except Exception:
        pass  # Queue may be in invalid state, continue with shutdown

    # Signal worker to stop and send sentinel
    _shutdown_event.set()
    with contextlib.suppress(queue.Full):
        _export_queue.put_nowait(None)  # Sentinel to wake worker

    # Wait for thread to finish (should be quick since queue is drained)
    _export_thread.join(timeout=2.0)
    _export_thread = None


def export_event_otlp(event: InferenceEvent) -> bool:
    """Queue an inference event for async OTLP export.

    Non-blocking: queues the event for background export.
    Events are dropped if the queue is full (backpressure).

    Dropped events are counted and logged with rate limiting to avoid log spam.
    Check get_otlp_stats() for dropped event count.

    Args:
        event: The InferenceEvent to export.

    Returns:
        True if queued successfully, False if queue full or not configured.
    """
    global _dropped_events_count, _last_drop_warning

    if not _otlp_configured:
        return False

    try:
        _export_queue.put_nowait(event)
        return True
    except queue.Full:
        import time

        _dropped_events_count += 1

        # Rate-limited warning: log every 1000 drops AND at most once per minute
        current_time = time.monotonic()
        should_warn = (
            _dropped_events_count % 1000 == 1
            and (current_time - _last_drop_warning) > _drop_warning_interval
        )

        if should_warn:
            logger.warning(
                f"OTLP export queue full, {_dropped_events_count} total events dropped. "
                f"Consider increasing VETCH_EXPORT_QUEUE_SIZE (current: {_export_queue.maxsize}) "
                f"or reducing event volume."
            )
            _last_drop_warning = current_time
        else:
            logger.debug(
                f"OTLP export queue full, dropping event ({_dropped_events_count} total dropped)"
            )

        return False


def export_advisory_otlp(
    code: str,
    severity: str,
    action: str,
    *,
    session_id: str | None = None,
    model: str | None = None,
    estimated_waste_usd: float = 0.0,
    tags: dict[str, str] | None = None,
) -> bool:
    """Export a waste advisory event via OTLP.

    Call this whenever an advisory fires in real time (e.g. STALL-001 warn/kill/reroute).
    Creates a ``vetch.advisory`` span and increments the advisory counter metric so
    dashboards can alert on waste patterns without polling the CLI.

    Args:
        code: Advisory code, e.g. ``"STALL-001"``.
        severity: ``"WARNING"`` or ``"CRITICAL"``.
        action: The action taken — ``"warn"``, ``"kill"``, ``"reroute"``, or ``"log"``.
        session_id: Active session ID for correlation with inference spans.
        model: Model name for filtering in dashboards.
        estimated_waste_usd: Estimated cost of the wasteful calls detected.
        tags: User-defined attribution tags (feature, customer, etc.).

    Returns:
        True if the advisory was exported, False if OTLP is not configured.
    """
    if not _otlp_configured:
        return False

    try:
        attributes: dict[str, Any] = {
            "vetch.advisory.code": code,
            "vetch.advisory.severity": severity,
            "vetch.advisory.action": action,
        }
        if model:
            attributes["vetch.model"] = model
        if session_id:
            attributes["vetch.session_id"] = session_id
        if tags:
            for k, v in tags.items():
                attributes[f"vetch.tag.{k}"] = v

        # Counter metric — lets dashboards alert on advisory rate
        if _advisory_counter:
            _advisory_counter.add(1, attributes)

        # Waste histogram — lets dashboards sum avoided cost over time
        if _advisory_waste_histogram and estimated_waste_usd > 0:
            _advisory_waste_histogram.record(estimated_waste_usd, attributes)

        # Span — gives the advisory a home in distributed traces
        if _tracer is not None:
            from opentelemetry import trace
            with _tracer.start_as_current_span(
                "vetch.advisory",
                kind=trace.SpanKind.INTERNAL,
            ) as span:
                for k, v in attributes.items():
                    span.set_attribute(k, v)
                if estimated_waste_usd > 0:
                    span.set_attribute("vetch.advisory.waste_usd", estimated_waste_usd)
                if severity == "CRITICAL":
                    span.set_status(
                        trace.Status(trace.StatusCode.ERROR, f"{code} detected")
                    )

        return True

    except Exception as exc:
        logger.debug("Failed to export advisory via OTLP: %s", exc)
        return False


def is_otlp_configured() -> bool:
    """Check if OTLP export is configured.

    Returns:
        True if configure_otlp_export() was called successfully.
    """
    return _otlp_configured


def get_otlp_stats() -> dict[str, Any]:
    """Get OTLP export queue statistics for monitoring.

    Returns:
        Dictionary with queue metrics:
        - queue_size: Maximum queue capacity
        - queue_current: Current number of items in queue
        - dropped_events: Total number of dropped events due to queue full
        - configured: Whether OTLP export is enabled

    Example::

        stats = vetch.otel.get_otlp_stats()
        if stats["dropped_events"] > 0:
            logger.warning(
                f"{stats['dropped_events']} events dropped, "
                f"queue {stats['queue_current']}/{stats['queue_size']}"
            )
    """
    return {
        "queue_size": _export_queue.maxsize,
        "queue_current": _export_queue.qsize(),
        "dropped_events": _dropped_events_count,
        "configured": _otlp_configured,
    }


# Auto-configure from environment if enabled
def _auto_configure() -> None:
    """Auto-configure OTLP export from environment variables."""
    if os.environ.get("VETCH_OTEL_EXPORT", "").lower() in ("true", "1", "yes"):
        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        if endpoint:
            service_name = os.environ.get("VETCH_OTEL_SERVICE_NAME", "vetch")
            configure_otlp_export(service_name=service_name)


_auto_configure()
