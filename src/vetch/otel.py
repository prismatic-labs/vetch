"""OpenTelemetry bridge for Vetch metrics.

This module allows Vetch to automatically attach energy, carbon, and
cost metadata to active OpenTelemetry spans.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vetch.schema import InferenceEvent

logger = logging.getLogger(__name__)


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
        from opentelemetry import trace # type: ignore[import-not-found]
        
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
        
        # Attach energy tier for confidence
        span.set_attribute("vetch.energy_tier", event.get("energy_tier", 3))
        
        return True

    except Exception as e:
        logger.debug(f"Failed to attach to OTel span: {e}")
        return False
