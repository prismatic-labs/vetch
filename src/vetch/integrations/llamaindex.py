"""LlamaIndex integration for Vetch tracking.

This module provides a native LlamaIndex callback handler that automatically
tracks energy, carbon, water, and cost for all LLM calls in LlamaIndex pipelines.

Usage::

    from vetch.integrations.llamaindex import VetchCallbackHandler
    from llama_index.core import Settings

    handler = VetchCallbackHandler(region="us-east-1", tags={"agent": "researcher"})
    Settings.callback_manager.add_handler(handler)

    # All LLM calls in LlamaIndex will now be tracked
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llama_index.core.callbacks.schema import CBEventType  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


class VetchCallbackHandler:
    """LlamaIndex callback handler for Vetch energy/carbon tracking.

    Integrates with LlamaIndex's callback system to automatically track
    all LLM calls in query engines, agents, and custom pipelines.

    Attributes:
        total_cost: Cumulative cost across all LLM calls (USD)
        total_energy_wh: Cumulative energy across all LLM calls (Wh)
        total_carbon_g: Cumulative carbon across all LLM calls (gCO2e)
        call_count: Number of LLM calls tracked
    """

    def __init__(
        self,
        region: str | None = None,
        tags: dict[str, str] | None = None,
        session_id: str | None = None,
    ) -> None:
        """Initialize VetchCallbackHandler.

        Args:
            region: Grid region for carbon calculations (e.g., "us-east-1")
            tags: Default tags to add to all events
            session_id: Optional session ID for grouping related calls
        """
        self.region = region
        self.tags = tags or {}
        self.session_id = session_id

        # Aggregated metrics
        self.total_cost: float = 0.0
        self.total_energy_wh: float = 0.0
        self.total_carbon_g: float = 0.0
        self.total_water_l: float = 0.0
        self.call_count: int = 0

        # Track individual events for detailed analysis
        self.events: deque[dict[str, Any]] = deque(maxlen=1000)  # Bounded to prevent OOM

        # Try to import vetch (fail gracefully if not installed)
        try:
            from vetch import wrap

            self._wrap = wrap
            self._vetch_available = True
        except ImportError:
            logger.warning("Vetch not installed. VetchCallbackHandler will not track metrics.")
            self._vetch_available = False

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Called when an event starts.

        Args:
            event_type: Type of event (LLM, EMBEDDING, etc.)
            payload: Event payload with metadata
            event_id: Unique event ID
            parent_id: Parent event ID if nested
            **kwargs: Additional arguments

        Returns:
            event_id for tracking
        """
        # No action needed on start
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Called when an event ends.

        This is where we capture usage metadata and calculate energy/carbon.

        Args:
            event_type: Type of event (LLM, EMBEDDING, etc.)
            payload: Event payload with metadata
            event_id: Unique event ID
            **kwargs: Additional arguments
        """
        if not self._vetch_available or not payload:
            return

        # Import here to avoid errors if llama_index not installed
        try:
            from llama_index.core.callbacks.schema import CBEventType, EventPayload
        except ImportError:
            return

        # Only track LLM events (not embedding, chunking, etc.)
        if event_type != CBEventType.LLM:
            return

        try:
            # Extract usage from payload
            response = payload.get(EventPayload.RESPONSE)
            if not response:
                return

            # Get raw response object which has usage
            raw_response = getattr(response, "raw", None) or response

            # Try to extract usage metadata
            usage_dict = {}
            model = None

            # Check for OpenAI-style usage
            if hasattr(raw_response, "usage"):
                usage = raw_response.usage
                usage_dict = {
                    "text": {
                        "input_tokens": getattr(usage, "prompt_tokens", 0),
                        "output_tokens": getattr(usage, "completion_tokens", 0),
                        "total_tokens": getattr(usage, "total_tokens", 0),
                    }
                }
                model = getattr(raw_response, "model", None)

            # Check for additional_kwargs (sometimes usage is here)
            elif hasattr(raw_response, "additional_kwargs"):
                additional = raw_response.additional_kwargs
                if "usage" in additional:
                    usage = additional["usage"]
                    usage_dict = {
                        "text": {
                            "input_tokens": usage.get("prompt_tokens", 0),
                            "output_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                        }
                    }

            if not usage_dict:
                logger.debug("No usage metadata in LlamaIndex response, skipping Vetch tracking")
                return

            # Infer provider from model name if available
            provider = "unknown"
            if model:
                model_lower = model.lower()
                if "gpt" in model_lower or "o1" in model_lower:
                    provider = "openai"
                elif "claude" in model_lower:
                    provider = "anthropic"
                elif "gemini" in model_lower:
                    provider = "vertexai"

            # Create Vetch wrapper and capture
            with self._wrap(region=self.region, tags=self.tags, emit=True) as ctx:
                from vetch.context import get_active_context

                active_ctx = get_active_context()
                if active_ctx:
                    active_ctx.capture(
                        model=model or "unknown",
                        provider=provider,
                        usage=usage_dict,  # type: ignore[arg-type]
                        is_stream=False,
                        complete=True,
                    )

            # Aggregate metrics
            if ctx.event:
                self.total_cost += ctx.event.get("estimated_cost_usd", 0) or 0
                self.total_energy_wh += ctx.event.get("estimated_energy_wh", 0) or 0
                self.total_carbon_g += ctx.event.get("estimated_carbon_g", 0) or 0
                self.total_water_l += ctx.event.get("estimated_water_l", 0) or 0
                self.call_count += 1
                self.events.append(dict(ctx.event))

        except Exception as e:
            logger.debug(f"Vetch tracking failed for LlamaIndex call: {e}")

    def start_trace(self, trace_id: str | None = None) -> None:
        """Start a new trace (no-op for compatibility)."""
        pass

    def end_trace(
        self,
        trace_id: str | None = None,
        trace_map: dict[str, list[str]] | None = None,
    ) -> None:
        """End a trace (no-op for compatibility)."""
        pass

    def reset(self) -> None:
        """Reset aggregated metrics.

        Useful for tracking metrics across multiple queries/chains.
        """
        self.total_cost = 0.0
        self.total_energy_wh = 0.0
        self.total_carbon_g = 0.0
        self.total_water_l = 0.0
        self.call_count = 0
        self.events = deque(maxlen=1000)

    def get_summary(self) -> dict[str, Any]:
        """Get summary of aggregated metrics.

        Returns:
            Dict with total_cost, total_energy_wh, total_carbon_g, call_count
        """
        return {
            "total_cost_usd": self.total_cost,
            "total_energy_wh": self.total_energy_wh,
            "total_carbon_g": self.total_carbon_g,
            "total_water_l": self.total_water_l,
            "call_count": self.call_count,
            "avg_cost_per_call": self.total_cost / self.call_count if self.call_count > 0 else 0,
            "avg_energy_per_call": (
                self.total_energy_wh / self.call_count if self.call_count > 0 else 0
            ),
        }
