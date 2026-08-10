"""OpenTelemetry exporter for vetch inference events.

This module exports vetch InferenceEvent data as OpenTelemetry spans following
the GenAI semantic conventions. It allows integration with existing observability
infrastructure (Jaeger, Datadog, New Relic, etc.).

Features:
- Maps vetch events to OTel spans with GenAI semantic conventions
- Adds custom vetch attributes (energy, carbon, cost, water)
- Optional auto-export on context exit
- Graceful degradation if opentelemetry-api not installed

Usage:
    >>> from vetch import VetchContext
    >>> from vetch.exporters.opentelemetry import configure_auto_export
    >>>
    >>> # Enable auto-export (once per application)
    >>> configure_auto_export(enabled=True)
    >>>
    >>> # Now all vetch contexts auto-export to OTel
    >>> with VetchContext() as ctx:
    ...     # Your inference code here
    ...     pass  # Span automatically created on exit

Manual export:
    >>> from vetch.exporters.opentelemetry import export_event_as_span
    >>> from opentelemetry import trace
    >>>
    >>> tracer = trace.get_tracer(__name__)
    >>> with tracer.start_as_current_span("my_operation") as parent:
    ...     with VetchContext() as ctx:
    ...         # ... inference code ...
    ...         pass
    ...     for event in ctx.get_events():
    ...         export_event_as_span(event, tracer=tracer, parent_span=parent)

Installation:
    pip install vetch[opentelemetry]

OpenTelemetry Semantic Conventions:
    https://opentelemetry.io/docs/specs/semconv/gen-ai/
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vetch.schema import InferenceEvent

logger = logging.getLogger(__name__)

# Global auto-export configuration
_AUTO_EXPORT_ENABLED = False


def configure_auto_export(enabled: bool = True) -> None:
    """Configure automatic export of inference events to OpenTelemetry spans.

    When enabled, VetchContext will automatically export all inference events
    as OpenTelemetry spans on context exit.

    Args:
        enabled: Whether to enable auto-export. Defaults to True.

    Example:
        >>> from vetch.exporters.opentelemetry import configure_auto_export
        >>> configure_auto_export(enabled=True)  # Enable once per application
        >>>
        >>> with VetchContext() as ctx:
        ...     # ... inference code ...
        ...     pass  # Spans automatically created on exit

    Note:
        Requires opentelemetry-api to be installed:
        pip install vetch[opentelemetry]
    """
    global _AUTO_EXPORT_ENABLED
    _AUTO_EXPORT_ENABLED = enabled
    logger.info(f"OpenTelemetry auto-export {'enabled' if enabled else 'disabled'}")


def is_auto_export_enabled() -> bool:
    """Check if auto-export is currently enabled.

    Returns:
        True if auto-export is enabled, False otherwise.
    """
    return _AUTO_EXPORT_ENABLED


def export_event_as_span(
    event: InferenceEvent,
    tracer: Any | None = None,
    parent_span: Any | None = None,
) -> None:
    """Export a vetch InferenceEvent as an OpenTelemetry span.

    Maps vetch inference metadata to OTel span attributes following the
    GenAI semantic conventions, plus custom vetch attributes for sustainability.

    Args:
        event: InferenceEvent to export.
        tracer: OpenTelemetry tracer. If None, uses global tracer.
        parent_span: Optional parent span for nesting.

    OpenTelemetry Attributes:
        GenAI Semantic Conventions (required):
        - gen_ai.system: Provider name (e.g., "openai", "anthropic")
        - gen_ai.request.model: Model identifier (e.g., "gpt-4")
        - gen_ai.usage.input_tokens: Number of input tokens
        - gen_ai.usage.output_tokens: Number of output tokens

        GenAI Semantic Conventions (optional):
        - gen_ai.response.finish_reasons: Why generation stopped
        - gen_ai.response.id: Response ID from provider
        - gen_ai.operation.name: Type of operation (e.g., "chat", "completion")

        Vetch Custom Attributes (sustainability):
        - vetch.cost.input_usd: Cost for input tokens (USD)
        - vetch.cost.output_usd: Cost for output tokens (USD)
        - vetch.cost.total_usd: Total cost (USD)
        - vetch.energy.input_wh: Energy for input tokens (Wh)
        - vetch.energy.output_wh: Energy for output tokens (Wh)
        - vetch.energy.total_wh: Total energy (Wh)
        - vetch.carbon.operational_g: Operational carbon emissions (gCO2eq)
        - vetch.carbon.embodied_g: Embodied carbon emissions (gCO2eq)
        - vetch.carbon.total_g: Total carbon emissions (gCO2eq)
        - vetch.water.total_l: Water usage (liters)
        - vetch.duration_s: Inference duration (seconds)
        - vetch.region: Cloud region (if known)
        - vetch.tier: Energy data tier (1=measured, 2=derived, 3=estimated)

        Cache-related (if applicable):
        - vetch.cache.read_tokens: Tokens read from cache
        - vetch.cache.creation_tokens: Tokens written to cache

    Example:
        >>> from vetch import VetchContext
        >>> from vetch.exporters.opentelemetry import export_event_as_span
        >>> from opentelemetry import trace
        >>>
        >>> tracer = trace.get_tracer(__name__)
        >>> with VetchContext() as ctx:
        ...     # ... your inference code ...
        ...     pass
        >>>
        >>> # Export all events
        >>> for event in ctx.get_events():
        ...     export_event_as_span(event, tracer=tracer)

    Note:
        Requires opentelemetry-api to be installed. If not available,
        logs a warning and returns gracefully.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode
    except ImportError:
        logger.warning(
            "opentelemetry-api not installed. "
            "Install with: pip install vetch[opentelemetry]"
        )
        return

    # Get tracer if not provided
    if tracer is None:
        tracer = trace.get_tracer(__name__)

    # Build span name from provider and model
    span_name = f"{event.get('provider', 'unknown')}.{event.get('model', 'unknown')}"

    # Determine parent context (only use recording spans)
    parent_context = None
    if parent_span is not None and parent_span.is_recording():
        parent_context = trace.set_span_in_context(parent_span)

    # Create span
    with tracer.start_as_current_span(
        span_name,
        context=parent_context,
        kind=trace.SpanKind.CLIENT,
    ) as span:
        # Set span status (assume success unless error present)
        if event.get("error"):
            span.set_status(Status(StatusCode.ERROR, description=str(event["error"])))
        else:
            span.set_status(Status(StatusCode.OK))

        # GenAI Semantic Conventions (required)
        span.set_attribute("gen_ai.system", event.get("provider", "unknown"))
        span.set_attribute("gen_ai.request.model", event.get("model", "unknown"))

        # Token usage
        usage = event.get("usage")
        if usage:
            text_usage = usage.get("text")
            if text_usage:
                input_tokens = text_usage.get("input_tokens", 0)
                output_tokens = text_usage.get("output_tokens", 0)
                span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
                span.set_attribute("gen_ai.usage.output_tokens", output_tokens)

        # Vetch Custom Attributes - Cost
        cost_total = event.get("estimated_cost_usd")
        cost_input = event.get("estimated_cost_input_usd")
        cost_output = event.get("estimated_cost_output_usd")

        if cost_total is not None:
            span.set_attribute("vetch.cost.total_usd", cost_total)
        if cost_input is not None:
            span.set_attribute("vetch.cost.input_usd", cost_input)
        if cost_output is not None:
            span.set_attribute("vetch.cost.output_usd", cost_output)

        # Vetch Custom Attributes - Energy
        energy_wh = event.get("estimated_energy_wh")
        if energy_wh is not None:
            span.set_attribute("vetch.energy.total_wh", energy_wh)

        # Vetch Custom Attributes - Carbon
        carbon_g = event.get("estimated_carbon_g")
        if carbon_g is not None:
            span.set_attribute("vetch.carbon.total_g", carbon_g)

        embodied_carbon = event.get("embodied_carbon_g")
        if embodied_carbon is not None:
            span.set_attribute("vetch.carbon.embodied_g", embodied_carbon)

        # Vetch Custom Attributes - Water
        water_l = event.get("estimated_water_l")
        if water_l is not None:
            span.set_attribute("vetch.water.total_l", water_l)

        # Vetch Custom Attributes - Latency
        latency_ms = event.get("latency_ms")
        if latency_ms is not None:
            span.set_attribute("vetch.latency_ms", latency_ms)

        # Vetch Custom Attributes - Region and Tier
        region = event.get("region")
        if region:
            span.set_attribute("vetch.region", region)

        energy_tier = event.get("energy_tier")
        if energy_tier is not None:
            span.set_attribute("vetch.tier", energy_tier)

        model_match = event.get("model_match")
        if model_match is not None:
            span.set_attribute("vetch.model_match", model_match)

        energy_completeness = event.get("energy_completeness")
        if energy_completeness is not None:
            span.set_attribute("vetch.energy_completeness", energy_completeness)

        cal_match = event.get("calibration_match")
        if cal_match:
            span.set_attribute("vetch.calibration_match", cal_match)

        # Cache-related attributes
        if "cache_read_tokens" in event and event["cache_read_tokens"]:
            span.set_attribute("vetch.cache.read_tokens", event["cache_read_tokens"])
        if "cache_creation_tokens" in event and event["cache_creation_tokens"]:
            span.set_attribute(
                "vetch.cache.creation_tokens", event["cache_creation_tokens"]
            )
        cache_cost_saving = event.get("cache_cost_saving_usd")
        if cache_cost_saving is not None:
            span.set_attribute("vetch.cache_cost_saving_usd", float(cache_cost_saving))
        cache_energy_saving = event.get("cache_energy_saving_wh")
        if cache_energy_saving is not None:
            span.set_attribute("vetch.cache_energy_saving_wh", float(cache_energy_saving))
        cache_carbon_saving = event.get("cache_carbon_saving_g")
        if cache_carbon_saving is not None:
            span.set_attribute("vetch.cache_carbon_saving_g", float(cache_carbon_saving))

        from vetch.capabilities import set_otel_capability_attributes

        set_otel_capability_attributes(span, event)

        logger.debug(f"Exported vetch event to OpenTelemetry span: {span_name}")


def export_events(
    events: list[InferenceEvent],
    tracer: Any | None = None,
    parent_span: Any | None = None,
) -> None:
    """Export multiple vetch InferenceEvents as OpenTelemetry spans.

    Convenience function to export a list of events. Each event becomes
    a separate span.

    Args:
        events: List of InferenceEvents to export.
        tracer: OpenTelemetry tracer. If None, uses global tracer.
        parent_span: Optional parent span for nesting.

    Example:
        >>> from vetch import VetchContext
        >>> from vetch.exporters.opentelemetry import export_events
        >>>
        >>> with VetchContext() as ctx:
        ...     # ... inference code ...
        ...     pass
        >>>
        >>> # Export all events at once
        >>> export_events(ctx.get_events())
    """
    for event in events:
        export_event_as_span(event, tracer=tracer, parent_span=parent_span)
