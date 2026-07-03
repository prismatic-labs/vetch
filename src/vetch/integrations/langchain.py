"""LangChain integration for Vetch tracking.

This module provides a native LangChain callback handler that automatically
tracks energy, carbon, water, and cost for all LLM calls in LangChain pipelines.

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
from typing import TYPE_CHECKING, Any, Protocol

from vetch.integrations._langchain_base import (
    BaseCallbackHandlerFallback,
    resolve_callback_handler_base,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    _CallbackHandlerBase = BaseCallbackHandlerFallback
else:
    _CallbackHandlerBase = resolve_callback_handler_base()


class _LLMResultLike(Protocol):
    """Subset of LangChain ``LLMResult`` used by :class:`VetchCallbackHandler`."""

    llm_output: dict[str, Any] | None
    generations: list[list[Any]]


class VetchCallbackHandler(_CallbackHandlerBase):
    """LangChain callback handler for Vetch energy/carbon tracking.

    Integrates with LangChain's callback system to automatically track
    all LLM calls in chains, agents, and LCEL pipelines.

    Note:
        This handler wraps an already-completed LLM call, so ``latency_ms`` on
        events it emits reflects the wrap block (~0 ms), not the real LLM call.
        Treat latency from handler-emitted events as not meaningful.

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

    @staticmethod
    def _um_get(um: Any, key: str) -> int:
        """Read a usage_metadata field whether it is a dict or an object.

        Never raises: returns 0 for anything non-numeric (e.g. test doubles).
        """
        try:
            val = um.get(key, 0) if hasattr(um, "get") else getattr(um, key, 0)
            return int(val or 0)
        except Exception:
            return 0

    def _extract_usage_and_model(
        self, response: _LLMResultLike
    ) -> tuple[int, int, int, str | None] | None:
        """Return (input_tokens, output_tokens, total_tokens, model) or None.

        Handles both the standardized LangChain ``usage_metadata`` on the message
        (Gemini and, increasingly, all providers) and the legacy
        ``llm_output["token_usage"]`` shape (langchain-openai).
        """
        llm_output = response.llm_output or {}
        model = llm_output.get("model_name") or llm_output.get("model")

        # 1. Standardized usage_metadata on the message. Sum across generations.
        in_tok = out_tok = total = 0
        found = False
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                um = getattr(msg, "usage_metadata", None)
                if um:
                    found = True
                    in_tok += self._um_get(um, "input_tokens")
                    out_tok += self._um_get(um, "output_tokens")
                    total += self._um_get(um, "total_tokens")
                if model is None and msg is not None:
                    rm = getattr(msg, "response_metadata", None) or {}
                    model = rm.get("model_name") or rm.get("model")
        if found:
            return in_tok, out_tok, total, model

        # 2. Legacy llm_output["token_usage"] (langchain-openai)
        tu = llm_output.get("token_usage") or {}
        if tu:
            return (
                int(tu.get("prompt_tokens", 0) or 0),
                int(tu.get("completion_tokens", 0) or 0),
                int(tu.get("total_tokens", 0) or 0),
                model,
            )
        return None

    def on_llm_end(
        self,
        response: _LLMResultLike,
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

            extracted = self._extract_usage_and_model(response)
            if extracted is None:
                logger.debug("No token usage in LangChain response, skipping Vetch tracking")
                return
            in_tok, out_tok, total, model = extracted

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
                        ml = model.lower()
                        if "gpt" in ml or "o1" in ml:
                            provider = "openai"
                        elif "claude" in ml:
                            provider = "anthropic"
                        elif "gemini" in ml:
                            provider = "google_genai"

                    # Build usage dict
                    usage: dict[str, Any] = {
                        "text": {
                            "input_tokens": in_tok,
                            "output_tokens": out_tok,
                            "total_tokens": total,
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
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """Called when LLM errors."""
        # Log error but don't block
        logger.debug(f"LLM error in LangChain: {error}")

    def on_llm_new_token(
        self,
        token: str | list[str | dict[str, Any]],
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
