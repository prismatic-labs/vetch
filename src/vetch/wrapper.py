"""Context manager for wrapping LLM inference calls.

This module provides the VetchContext class that:
- Patches LLM SDK methods to capture usage metadata
- Calculates energy, carbon, and cost estimates
- Emits structured JSON events
- Maintains fail-open behavior (never blocks LLM calls)
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
import uuid
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal, cast

from vetch import __version__
from vetch.context import TrackingContext
from vetch.emitter import emit_event
from vetch.schema import SCHEMA_VERSION, InferenceEvent, validate_energy_override

if TYPE_CHECKING:
    from vetch.schema import EnergyOverride

logger = logging.getLogger(__name__)

# Flag to track if warning about missing region has been issued
_region_warning_issued = False
_timezone_warning_issued = False


def _infer_region() -> tuple[str | None, str | None]:
    """Infer region from environment or local heuristics.

    Priority:
    1. VETCH_REGION
    2. Cloud Provider Env Vars
    3. Heuristic: Local Timezone

    Returns:
        Tuple of (region_name, warning_message).
    """
    # 1. Direct Env Vars
    region = os.environ.get("VETCH_REGION")
    if region:
        return region, None

    # 2. AWS/GCP/Azure
    for var in ["AWS_REGION", "AWS_DEFAULT_REGION", "GOOGLE_CLOUD_REGION", "AZURE_REGION"]:
        val = os.environ.get(var)
        if val:
            return val, None

    # 3. Extraordinary Move: Timezone Heuristic
    try:
        import time

        # Get UTC offset in hours
        offset = -time.timezone if time.daylight == 0 else -time.altzone
        hours = offset / 3600

        # Simple mapping for common regions
        inferred = None
        if hours == 0:
            inferred = "eu-west-2"  # London
        elif 1 <= hours <= 3:
            inferred = "eu-central-1"  # Europe
        elif -5 <= hours <= -4:
            inferred = "us-east-1"  # US East
        elif -8 <= hours <= -7:
            inferred = "us-west-2"  # US West
        elif 8 <= hours <= 9:
            inferred = "asia-northeast-1"  # Tokyo/Seoul

        if inferred:
            warning = (
                f"Region '{inferred}' inferred from timezone. Accuracy ~30%. "
                "Set VETCH_REGION for precision."
            )
            return inferred, warning
    except Exception:
        pass

    return None, None


class VetchContext:
    """Context manager for tracking LLM inference energy and carbon.

    Implements fail-open behavior: if any Vetch operation fails,
    the LLM call still proceeds and an event is emitted with
    tracking_disabled=True.
    """

    def __init__(
        self,
        region: str | None = None,
        tags: dict[str, str] | None = None,
        energy_override: dict[str, object] | None = None,
        price_multiplier: float = 1.0,
        emit: bool = True,
        _disabled: bool = False,
    ) -> None:
        """Initialize tracking context.

        Args:
            region: Grid region for carbon calculation.
            tags: Key-value pairs for cost attribution.
            energy_override: User-provided energy values.
            price_multiplier: Factor to adjust list pricing (e.g. 0.8 for 20% discount).
            emit: If True, emit JSON to configured output. Set False for quiet mode.
            _disabled: Internal flag for kill switch (VETCH_DISABLED=true).
        """
        self._emit = emit
        # P0: Kill switch - store disabled state for no-op behavior
        self._globally_disabled = _disabled

        from vetch.context import get_active_context

        # Check for active parent context
        parent = get_active_context()

        # Set price multiplier, inheriting from parent if default
        self.price_multiplier = price_multiplier
        if self.price_multiplier == 1.0 and parent is not None:
            self.price_multiplier = getattr(parent, "price_multiplier", 1.0)

        # Build initial tags from global configuration
        from vetch.config import get_global_tags

        self.tags: dict[str, str] | None = dict(get_global_tags())

        # Inherit region
        self.region = region
        if self.region is None and parent is not None:
            self.region = parent.region

        # Merge tags (inner overrides parent)
        if parent is not None and parent.tags:
            self.tags.update(parent.tags)
        if tags:
            self.tags.update(tags)

        if not self.tags:
            self.tags = None

        self._energy_override: EnergyOverride | None = None
        self._tracking_disabled = False
        self._start_time: float | None = None
        self._event: InferenceEvent | None = None
        self._tracking_ctx: TrackingContext | None = None
        self._warnings: list[str] = []  # Collect diagnostic warnings
        self._patched_clients: list[tuple[str, Any]] = []  # Per-context patched clients

        # Validate energy override if provided
        if energy_override is not None:
            validated, warnings = validate_energy_override(energy_override)
            self._warnings.extend(warnings)
            if validated is None:
                logger.warning(
                    "Invalid energy_override provided, falling back to registry. "
                    "Required: wh_per_1k_input (positive float), "
                    "wh_per_1k_output (positive float)"
                )
            else:
                self._energy_override = validated

    @property
    def event(self) -> InferenceEvent | None:
        """Get the inference event after context exits.

        Returns:
            The logged InferenceEvent, or None if still in context.
        """
        return self._event

    @property
    def tracking_disabled(self) -> bool:
        """Check if tracking was disabled due to errors."""
        return self._tracking_disabled

    def _get_session_id(self) -> str | None:
        """Get the session ID if this event is part of a session."""
        try:
            from vetch.session import get_active_session

            session = get_active_session()
            return session.session_id if session else None
        except Exception:
            return None

    def __enter__(self) -> VetchContext:
        """Enter context and start tracking.

        Sets up timing and prepares for patching SDK methods.
        Fail-open: any setup errors disable tracking but don't raise.
        """
        self._setup()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> Literal[False]:
        """Exit context and emit event.

        Always emits an event (even on error). Never suppresses exceptions.

        Returns:
            False (never suppress exceptions)
        """
        self._teardown(exc_type)
        return False  # Never suppress exceptions

    async def __aenter__(self) -> VetchContext:
        """Async enter context and start tracking."""
        self._setup()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> Literal[False]:
        """Async exit context and emit event."""
        self._teardown(exc_type)
        return False

    def _setup(self) -> None:
        """Setup context state."""
        # P0: Skip setup if globally disabled
        if getattr(self, "_globally_disabled", False):
            self._tracking_disabled = True
            return

        self._start_time = time.perf_counter()

        # Infer region if not provided
        if self.region is None:
            self.region, warning = _infer_region()
            if warning:
                self._warnings.append(warning)

            if self.region is None:
                global _region_warning_issued
                if not _region_warning_issued:
                    logger.info(
                        "No region specified and could not infer from environment. "
                        "Set VETCH_REGION or pass region parameter for accurate carbon data."
                    )
                    _region_warning_issued = True

        try:
            # 1. Validate compliance (Mandatory Tags)
            from vetch.config import validate_tags

            missing = validate_tags(self.tags)
            if missing:
                msg = f"Compliance Error: Missing mandatory tags: {', '.join(missing)}"
                logger.error(msg)
                self._tracking_disabled = True
                # Still create a context to avoid AttributeErrors if user accesses it,
                # but it won't capture anything.
                return

            # 2. Create and enter tracking context
            self._tracking_ctx = TrackingContext(
                region=self.region,
                tags=self.tags,
                energy_override=self._energy_override,
            )
            self._tracking_ctx.warnings = list(self._warnings)  # Start with current warnings
            self._tracking_ctx.__enter__()

            # 3. Set up SDK patches
            self._setup_patches()
        except Exception as e:
            logger.warning(f"Vetch setup failed, tracking disabled: {e}")
            self._tracking_disabled = True

    def _teardown(self, exc_type: type[BaseException] | None) -> None:
        """Teardown context state."""
        latency_ms = None
        if self._start_time is not None:
            latency_ms = (time.perf_counter() - self._start_time) * 1000

        try:
            self._emit_event(
                error=exc_type is not None,
                error_type=exc_type.__name__ if exc_type else None,
                latency_ms=latency_ms,
            )
        except Exception as e:
            # Extraordinary Move: Pre-filled Issue Link
            import traceback
            import urllib.parse

            tb = traceback.format_exc()
            params = {
                "title": f"Alpha Error: {type(e).__name__}",
                "body": f"Vetch version: {__version__}\n\nTraceback:\n```\n{tb}\n```",
            }
            issue_url = (
                f"https://github.com/prismatic-labs/vetch/issues/new?"
                f"{urllib.parse.urlencode(params)}"
            )
            logger.warning(
                f"Vetch event emission failed: {e}\n"
                f"Please help us improve the Alpha by reporting this: {issue_url}"
            )
        finally:
            # Exit tracking context
            if self._tracking_ctx is not None:
                with contextlib.suppress(Exception):
                    # TrackingContext is sync, so we call __exit__
                    self._tracking_ctx.__exit__(exc_type, None, None)

            # Cleanup patches (fail silently)
            with contextlib.suppress(Exception):
                self._cleanup_patches()

    def _setup_patches(self) -> None:
        """Set up SDK method patches.

        Detects installed SDKs and patches their completion methods.
        Respects existing patches (Datadog, OpenTelemetry, etc.).
        Each context tracks its own patched clients for proper cleanup isolation.
        """
        # Try to patch OpenAI
        try:
            from vetch.providers.openai import detect_openai_client, patch_openai_client

            client = detect_openai_client()
            if client is not None and patch_openai_client(client):
                self._patched_clients.append(("openai", client))
        except Exception as e:
            logger.debug(f"OpenAI patching skipped: {e}")

        # Try to patch Vertex AI
        try:
            # Vertex AI requires explicit model patching, but we can try to
            # detect if the SDK is used and provide hooks.
            from vetch.providers.vertexai import detect_vertexai_model

            _ = detect_vertexai_model()
            # Note: Vertex AI patching is typically done per-model instance
            # in the current implementation. Future versions may add global patching.
        except Exception as e:
            logger.debug(f"Vertex AI patching skipped: {e}")

        # Try to patch Anthropic
        # Note: Anthropic patching is primarily manual or explicit for now
        # as there isn't a global default client.

    def _cleanup_patches(self) -> None:
        """Remove SDK method patches.

        Only unpatches clients that THIS context patched.
        Thread-safe: other contexts' clients are not affected.
        """
        for provider, client in self._patched_clients:
            try:
                if provider == "openai":
                    from vetch.providers.openai import unpatch_openai_client

                    unpatch_openai_client(client)
                elif provider == "vertexai":
                    from vetch.providers.vertexai import unpatch_vertexai_model

                    unpatch_vertexai_model(client)
                elif provider == "anthropic":
                    from vetch.providers.anthropic import unpatch_anthropic_client

                    unpatch_anthropic_client(client)
            except Exception as e:
                logger.debug(f"Failed to unpatch {provider}: {e}")

        self._patched_clients.clear()

    def _emit_event(
        self,
        error: bool = False,
        error_type: str | None = None,
        latency_ms: float | None = None,
    ) -> None:
        """Emit inference event to configured output.

        Args:
            error: Whether an exception occurred.
            error_type: Exception class name if error.
            latency_ms: Request latency in milliseconds.
        """
        from vetch.calculation import (
            calculate_carbon,
            calculate_cost,
            calculate_energy,
        )
        from vetch.sensing.grid import get_carbon_intensity

        # Look up active session once (avoids redundant ContextVar lookups)
        active_session = None
        try:
            from vetch.session import get_active_session

            active_session = get_active_session()
        except Exception:
            pass

        # Get captured data from tracking context
        captured = None
        if self._tracking_ctx is not None:
            captured = self._tracking_ctx.captured_call

        # Determine model and provider
        model = "unknown"
        provider = "unknown"
        usage = None
        is_stream = False
        accumulated_chars = 0
        model_known = False

        # Cache tokens
        cache_read_tokens: int | None = None
        cache_creation_tokens: int | None = None

        if captured is not None:
            model = captured.model
            provider = captured.provider
            usage = captured.usage
            is_stream = captured.is_stream
            accumulated_chars = captured.accumulated_chars
            raw_crt = captured.cache_read_tokens
            cache_read_tokens = raw_crt if isinstance(raw_crt, int) else None
            raw_cct = captured.cache_creation_tokens
            cache_creation_tokens = raw_cct if isinstance(raw_cct, int) else None

            # Override error info from captured call if present
            if captured.error:
                error = True
                error_type = captured.error_type

        # 1. Get grid intensity
        grid_intensity = get_carbon_intensity(self.region)
        signal_quality = grid_intensity.signal_quality
        grid_val = grid_intensity.intensity_gco2e_kwh
        grid_ts = None
        if grid_intensity.timestamp:
            ts = datetime.fromtimestamp(grid_intensity.timestamp, tz=timezone.utc)
            grid_ts = ts.isoformat().replace("+00:00", "Z")

        # 2. Perform calculations
        energy_wh = None
        energy_tier = 3
        energy_uncertainty_pct: int | None = 1000  # Tier 3 default
        energy_source = "registry"
        energy_basis = None
        carbon_g = None
        pue = None
        pue_tier = 3
        pue_source = "unknown"
        cost_usd = None
        cost_in_usd = None
        cost_out_usd = None
        billing_tier = "list"
        usage_estimated = False
        usage_estimation_method: str | None = None

        # Token estimation fallback for streaming without usage data
        if (not usage or not usage.get("text")) and accumulated_chars > 0:
            # Heuristic: ~4 chars per token (common for English text)
            # This is a rough estimate - mark it clearly
            estimated_output_tokens = max(1, accumulated_chars // 4)
            # We don't know input tokens without the prompt, use a conservative estimate
            # based on typical chat patterns (input often ~2x output)
            estimated_input_tokens = estimated_output_tokens * 2

            usage = {
                "text": {
                    "input_tokens": estimated_input_tokens,
                    "output_tokens": estimated_output_tokens,
                    "total_tokens": estimated_input_tokens + estimated_output_tokens,
                }
            }
            usage_estimated = True
            usage_estimation_method = "char_ratio"
            self._warnings.append(
                f"Token usage estimated from {accumulated_chars} chars "
                f"(~4 chars/token). Actual usage may differ."
            )

        if usage and usage.get("text"):
            text = usage["text"]
            if text:  # Type guard for mypy
                in_tokens = text.get("input_tokens", 0)
                out_tokens = text.get("output_tokens", 0)

                # Energy
                (
                    energy_wh,
                    energy_tier,
                    energy_uncertainty_pct,
                    energy_source,
                    energy_basis,
                    model_known,
                ) = calculate_energy(
                    in_tokens,
                    out_tokens,
                    model,
                    cast("dict[str, Any]", self._energy_override),
                )

                # Add warning if model not in registry
                if not model_known and model != "unknown":
                    self._warnings.append(
                        f"Model '{model}' not in registry, using conservative fallback estimates"
                    )

                # Carbon with provider-specific PUE
                if energy_wh is not None:
                    carbon_g, pue, pue_tier, pue_source = calculate_carbon(
                        energy_wh, grid_val, model=model, provider_hint=provider
                    )

                # Cost (pass cache tokens for cache-aware pricing)
                cost_usd, cost_in_usd, cost_out_usd, billing_tier = calculate_cost(
                    in_tokens, out_tokens, model,
                    cache_read_tokens=cache_read_tokens,
                    cache_creation_tokens=cache_creation_tokens,
                )

                # Apply price multiplier (e.g., enterprise discount)
                if self.price_multiplier != 1.0:
                    cost_usd *= self.price_multiplier
                    cost_in_usd *= self.price_multiplier
                    cost_out_usd *= self.price_multiplier
                    billing_tier = f"list×{self.price_multiplier}"
        # Combine all warnings (from context and captured call)
        all_warnings = list(self._warnings)
        if captured and captured.warnings:
            all_warnings.extend(captured.warnings)

        # Build event
        self._event = InferenceEvent(
            schema_version=SCHEMA_VERSION,
            vetch_version=__version__,
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            model=model,
            provider=provider,
            model_known=model_known,
            usage=usage,
            accumulated_chars=accumulated_chars if is_stream else None,
            estimated_energy_wh=energy_wh,
            estimated_carbon_g=carbon_g,
            estimated_cost_usd=cost_usd,
            estimated_cost_input_usd=cost_in_usd,
            estimated_cost_output_usd=cost_out_usd,
            billing_tier=billing_tier,
            signal_quality=signal_quality,
            energy_tier=energy_tier,
            energy_uncertainty_pct=energy_uncertainty_pct,
            energy_source=energy_source,
            energy_override_source=(
                self._energy_override.get("source") if self._energy_override else None
            ),
            energy_basis=energy_basis,
            grid_intensity_gco2e_kwh=grid_val,
            grid_intensity_timestamp=grid_ts,
            region=self.region,
            pue=pue,
            pue_tier=pue_tier,
            pue_source=pue_source,
            is_stream=is_stream,
            complete=not error and (captured.complete if captured else True),
            latency_ms=latency_ms,
            tags=self.tags,
            error=error,
            error_type=error_type,
            tracking_disabled=self._tracking_disabled,
            vetch_warnings=all_warnings if all_warnings else None,
            budget_energy_wh=None,
            budget_carbon_g=None,
            budget_cost_usd=None,
            budget_exceeded=None,
            usage_estimated=usage_estimated,
            usage_estimation_method=usage_estimation_method,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
            cache_hit=bool(isinstance(cache_read_tokens, int) and cache_read_tokens > 0),
            session_id=active_session.session_id if active_session else None,
        )

        # Budget checking (warn-only, never blocks)
        try:
            from vetch.budget import check_budgets

            exceeded, alerts = check_budgets(
                cost_usd=cost_usd,
                energy_wh=energy_wh,
                carbon_g=carbon_g,
                tags=self.tags,
            )
            if exceeded:
                self._event["budget_exceeded"] = True
            if alerts:
                # Add budget info to event
                for alert in alerts:
                    if alert.metric == "cost_usd":
                        self._event["budget_cost_usd"] = alert.threshold
                    elif alert.metric == "energy_wh":
                        self._event["budget_energy_wh"] = alert.threshold
                    elif alert.metric == "carbon_g":
                        self._event["budget_carbon_g"] = alert.threshold
        except Exception:
            pass  # Fail-open: budget checks never block

        # Emit to configured output (unless quiet mode)
        if self._emit:
            emit_event(self._event)

        # Local Storage (The Black Box Recorder)
        try:
            from vetch.storage import store_event

            store_event(self._event)
        except Exception:
            pass

        # Session Stats (The Advisory Brain)
        try:
            from vetch.stats import track_session_event

            track_session_event(cast("dict[str, Any]", self._event))
        except Exception:
            pass

        # Extraordinary Move: CI Tracking
        try:
            from vetch.ci import track_ci_event

            track_ci_event(cast("dict[str, Any]", self._event))
        except Exception:
            pass

        # Try to attach to active OTel span
        try:
            from vetch.otel import attach_to_otel_span

            attach_to_otel_span(self._event)
        except Exception:
            pass

        # Export via OTLP if configured
        try:
            from vetch.otel import export_event_otlp, is_otlp_configured

            if is_otlp_configured():
                export_event_otlp(self._event)
        except Exception:
            pass

        # Register with active session (if any)
        if active_session is not None:
            with contextlib.suppress(Exception):
                active_session.register_event(self._event)


@contextmanager
def wrap(
    region: str | None = None,
    tags: dict[str, str] | None = None,
    energy_override: dict[str, object] | None = None,
    price_multiplier: float = 1.0,
    emit: bool = True,
) -> Generator[VetchContext, None, None]:
    """Context manager for tracking LLM inference.

    This is a convenience function that creates a VetchContext.
    See VetchContext for full documentation.

    Args:
        region: Grid region for carbon calculation.
        tags: Key-value pairs for cost attribution.
        energy_override: User-provided energy values.
        price_multiplier: Factor to adjust list pricing (e.g., 0.8 for 20% discount).
        emit: If True (default), emit JSON to configured output.
              Set False for quiet mode (metrics still available in ctx.event).

    Example:
        # Quiet mode - no JSON output, access metrics programmatically
        with wrap(emit=False) as ctx:
            response = client.chat.completions.create(...)
        print(f"Energy: {ctx.event['estimated_energy_wh']} Wh")
    """
    ctx = VetchContext(
        region=region,
        tags=tags,
        energy_override=energy_override,
        price_multiplier=price_multiplier,
        emit=emit,
    )
    with ctx:
        yield ctx


@asynccontextmanager
async def awrap(
    region: str | None = None,
    tags: dict[str, str] | None = None,
    energy_override: dict[str, object] | None = None,
    price_multiplier: float = 1.0,
    emit: bool = True,
    _disabled: bool = False,
) -> AsyncGenerator[VetchContext, None]:
    """Async context manager for tracking LLM inference.

    First-class async support for async/await patterns. Equivalent to
    wrap() but designed specifically for async code.

    Args:
        region: Grid region for carbon calculation.
        tags: Key-value pairs for cost attribution.
        energy_override: User-provided energy values.
        price_multiplier: Factor to adjust list pricing (e.g., 0.8 for 20% discount).
        emit: If True (default), emit JSON to configured output.

    Example::

        async with awrap(region="us-east-1") as ctx:
            response = await client.chat.completions.create(...)
        print(f"Energy: {ctx.event['estimated_energy_wh']} Wh")

    Note:
        For sync code, use wrap() instead.
    """
    ctx = VetchContext(
        region=region,
        tags=tags,
        energy_override=energy_override,
        price_multiplier=price_multiplier,
        emit=emit,
        _disabled=_disabled,
    )
    async with ctx:
        yield ctx
