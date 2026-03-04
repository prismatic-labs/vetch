"""LangChain integration for Vetch tracking.

This module provides a native LangChain callback handler that automatically
tracks energy, carbon, and cost for all LLM calls in LangChain pipelines.

Usage::

    from vetch.integrations.langchain import VetchCallbackHandler
    from langchain.llms import OpenAI

    handler = VetchCallbackHandler(region="us-east-1", tags={"agent": "researcher"})
    llm = OpenAI(callbacks=[handler])
    response = llm("What is the capital of France?")

    # Access aggregated metrics
    print(f"Total cost: ${handler.total_cost}")
    print(f"Total energy: {handler.total_energy_wh} Wh")
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain.schema import LLMResult  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


class VetchCallbackHandler:
    """LangChain callback handler for Vetch energy/carbon tracking.

    Integrates with LangChain's callback system to automatically track
    all LLM calls in chains, agents, and LCEL pipelines.

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

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        """Called when LLM starts running.

        Note: We track on_llm_end instead of on_llm_start to capture
        actual usage metadata from the response.
        """
        pass

    def on_llm_end(
        self,
        response: LLMResult,
        **kwargs: Any,
    ) -> None:
        """Called when LLM finishes running.

        This is where we capture usage metadata and calculate energy/carbon.
        """
        if not self._vetch_available:
            return

        try:
            # Extract LLM response metadata
            # LangChain's LLMResult has generations and llm_output
            if not response.generations:
                return

            # Get usage metadata from llm_output (if available)
            llm_output = response.llm_output or {}
            token_usage = llm_output.get("token_usage", {})

            if not token_usage:
                logger.debug("No token usage in LangChain response, skipping Vetch tracking")
                return

            # Extract model name (different providers use different keys)
            model = llm_output.get("model_name") or llm_output.get("model")

            # Create Vetch wrapper with inherited config
            with self._wrap(region=self.region, tags=self.tags, emit=True) as ctx:
                # Manually capture the metadata (since LangChain already made the call)
                # We use Vetch's internal context to calculate energy/carbon/cost
                from vetch.context import get_active_context

                active_ctx = get_active_context()
                if active_ctx:
                    # Detect provider from model name
                    provider = "unknown"
                    if model:
                        if "gpt" in model.lower() or "o1" in model.lower():
                            provider = "openai"
                        elif "claude" in model.lower():
                            provider = "anthropic"
                        elif "gemini" in model.lower():
                            provider = "vertexai"

                    # Build usage dict
                    usage: dict[str, Any] = {
                        "text": {
                            "input_tokens": token_usage.get("prompt_tokens", 0),
                            "output_tokens": token_usage.get("completion_tokens", 0),
                            "total_tokens": token_usage.get("total_tokens", 0),
                        }
                    }

                    # Capture metadata into Vetch context
                    active_ctx.capture(
                        model=model or "unknown",
                        provider=provider,
                        usage=usage,  # type: ignore[arg-type]
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
            logger.debug(f"Vetch tracking failed for LangChain call: {e}")

    def on_llm_error(
        self,
        error: Exception | KeyboardInterrupt,
        **kwargs: Any,
    ) -> None:
        """Called when LLM errors."""
        # Log error but don't block
        logger.debug(f"LLM error in LangChain: {error}")

    def on_llm_new_token(
        self,
        token: str,
        **kwargs: Any,
    ) -> None:
        """Called when LLM generates a new token during streaming.

        Note: For streaming calls, we track on a per-call basis and capture
        metrics when the stream completes (on_llm_end). This callback is
        here for compatibility but doesn't need to track anything since
        LangChain will call on_llm_end with full usage metadata.
        """
        # No action needed - on_llm_end will have the full usage
        pass

    def reset(self) -> None:
        """Reset aggregated metrics.

        Useful for tracking metrics across multiple chains/runs.
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
