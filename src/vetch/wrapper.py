"""Context manager for wrapping LLM inference calls.

This module provides the VetchContext class that:
- Patches LLM SDK methods to capture usage metadata
- Calculates energy, carbon, water, and cost estimates
- Emits structured JSON events
- Maintains fail-open behavior (never blocks LLM calls)
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
import uuid
from collections import deque
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal, cast

from vetch import __version__
from vetch.context import TrackingContext, get_active_context
from vetch.emitter import emit_event
from vetch.schema import (
    SCHEMA_VERSION,
    CapabilityRef,
    InferenceEvent,
    Usage,
    validate_energy_override,
)

if TYPE_CHECKING:
    from vetch.schema import EnergyOverride

logger = logging.getLogger(__name__)

# Tracking metrics for observability (structured logging)
# Exposed via get_tracking_stats() for monitoring dashboards
_tracking_errors: dict[str, int] = {
    "tag_validation_failed": 0,
    "missing_required_tags": 0,
    "cardinality_exceeded": 0,
    "allowlist_filtered": 0,
    "energy_calculation_failed": 0,
    "grid_lookup_failed": 0,
    "model_unknown": 0,
    "usage_estimated": 0,
}

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
    # 1. Check module-level default (set via vetch.instrument()) then env var
    try:
        from vetch import get_default_region
        region = get_default_region()
    except ImportError:
        # Fallback if import fails (shouldn't happen but be defensive)
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


def _infer_n_images_from_usage(usage: Usage | None) -> int:
    """Infer image count from schema v2 usage for VLM energy (wh_per_image)."""
    if not usage:
        return 0
    image_usage = usage.get("image")
    if not isinstance(image_usage, dict):
        return 0
    count = image_usage.get("image_count")
    if isinstance(count, int) and count > 0:
        return count
    input_tokens = image_usage.get("input_tokens")
    if isinstance(input_tokens, int) and input_tokens > 0:
        return 1
    return 0


@contextmanager
def auto_context_for_instrumented_call(
    provider: str,
    model: str = "unknown",
    region: str | None = None,
) -> Generator[VetchContext | None, None, None]:
    """Create an automatic context for instrumented calls without explicit wrap().

    This helper eliminates code duplication across provider hooks and ensures
    proper lifecycle management using context managers.

    Args:
        provider: The provider name (e.g., "openai", "anthropic", "vertexai").
        model: The model name (default: "unknown").

    Yields:
        VetchContext if created, None if manual wrap() context is active.

    Usage:
        with auto_context_for_instrumented_call("openai", model="gpt-4"):
            # Call provider API
            response = original_method(...)
            # Capture metadata
            active_ctx = get_active_context()
            if active_ctx:
                active_ctx.capture(...)
    """
    # Import here to avoid circular dependency (vetch/__init__.py imports from this module)
    # These imports are cached by Python so the overhead is minimal
    from vetch import (
        get_default_energy_override,
        get_default_provider_hint,
        get_default_region,
        get_default_tags,
    )

    # Check if manual wrap() context is already active
    ctx = get_active_context()
    if ctx is not None:
        # Manual wrap() context exists - yield None to signal "use existing context"
        yield None
        return

    # No context exists - create automatic context using defaults from instrument()
    auto_ctx = VetchContext(
        region=region or get_default_region(),
        tags=get_default_tags(),
        energy_override=get_default_energy_override(),
        emit=True,
        provider_hint=get_default_provider_hint(),
    )

    # Use proper with statement for lifecycle management
    with auto_ctx:
        yield auto_ctx


@asynccontextmanager
async def async_auto_context_for_instrumented_call(
    provider: str,
    model: str = "unknown",
    region: str | None = None,
) -> AsyncGenerator[VetchContext | None, None]:
    """Async version of auto_context_for_instrumented_call.

    This helper eliminates code duplication across async provider hooks and ensures
    proper lifecycle management using async context managers.

    Args:
        provider: The provider name (e.g., "openai", "anthropic", "vertexai").
        model: The model name (default: "unknown").

    Yields:
        VetchContext if created, None if manual wrap() context is active.

    Usage:
        async with async_auto_context_for_instrumented_call("openai", model="gpt-4"):
            # Call provider API
            response = await original_method(...)
            # Capture metadata
            active_ctx = get_active_context()
            if active_ctx:
                active_ctx.capture(...)
    """
    # Import here to avoid circular dependency (vetch/__init__.py imports from this module)
    # These imports are cached by Python so the overhead is minimal
    from vetch import (
        get_default_energy_override,
        get_default_provider_hint,
        get_default_region,
        get_default_tags,
    )

    # Check if manual wrap() context is already active
    ctx = get_active_context()
    if ctx is not None:
        # Manual wrap() context exists - yield None to signal "use existing context"
        yield None
        return

    # No context exists - create automatic context using defaults from instrument()
    auto_ctx = VetchContext(
        region=region or get_default_region(),
        tags=get_default_tags(),
        energy_override=get_default_energy_override(),
        emit=True,
        provider_hint=get_default_provider_hint(),
    )

    # Use proper async with statement for lifecycle management
    async with auto_ctx:
        yield auto_ctx


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
        provider_hint: str | None = None,
        _disabled: bool = False,
        _manual: bool = False,
        _manual_latency_ms: float | None = None,
    ) -> None:
        """Initialize tracking context.

        Args:
            region: Grid region for carbon calculation.
            tags: Key-value pairs for cost attribution.
            energy_override: User-provided energy values.
            price_multiplier: Factor to adjust list pricing (e.g. 0.8 for 20% discount).
            emit: If True, emit JSON to configured output. Set False for quiet mode.
            provider_hint: Explicit provider override (e.g. "self-hosted",
                "openai-compatible"). Overrides the provider inferred from the
                model name / SDK client, driving the cost branch (self-hosted →
                cost 0) and PUE/water lookups. An unrecognised value is used but
                flagged in ``vetch_warnings`` (fail-loud).
            _disabled: Internal flag for kill switch (VETCH_DISABLED=true).
            _manual: Internal flag for record_usage() — skip SDK client patching
                (there is no live call to intercept, only supplied usage).
            _manual_latency_ms: Internal — caller-supplied latency for a manual
                record. Manual contexts do not time a live call, so the internal
                stopwatch (which would measure only emit overhead) is not used.
        """
        self._emit = emit
        # P0: Kill switch - store disabled state for no-op behavior
        self._globally_disabled = _disabled
        self._manual = _manual
        self._manual_latency_ms = _manual_latency_ms

        from vetch.context import get_active_context

        # Check for active parent context
        parent = get_active_context()

        # Explicit provider override for this context's emitted event. Normalised
        # to lower case (provider labels are lower case) and validated so a typo
        # like "selfhosted" is surfaced rather than silently billed at cloud
        # rates. Applies to the context it is set on (not inherited via the
        # TrackingContext parent chain).
        self._provider_hint = provider_hint.strip().lower() if provider_hint else None
        self._provider_hint_warning: str | None = None
        if self._provider_hint:
            from vetch.calculation import KNOWN_PROVIDERS

            if self._provider_hint not in KNOWN_PROVIDERS:
                self._provider_hint_warning = (
                    f"Unrecognised provider_hint '{provider_hint}'; not one of "
                    f"{sorted(KNOWN_PROVIDERS)}. Using it as-is — cost may be wrong "
                    f"(e.g. 'self-hosted' zeroes cost, a typo does not)."
                )

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
        self._warnings: deque[str] = deque(maxlen=50)
        self._patched_clients: list[tuple[str, Any]] = []  # Per-context patched clients

        # Fail-loud: surface an unrecognised provider_hint (validated above).
        if self._provider_hint_warning:
            self._warnings.append(self._provider_hint_warning)
            logger.warning(self._provider_hint_warning)

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
            from vetch.config import process_tags_single_pass

            # Single-pass tag processing: redact → filter → validate → cardinality
            # Optimized to reduce dict allocations from 4 to 1 (~40% fewer allocations)
            #
            # Operations performed in order:
            # 1. Redact sensitive tags FIRST (HMAC-SHA256 hashing for PII protection)
            # 2. Apply tag allowlist (security filtering)
            # 3. Track cardinality with rate limiting and LRU eviction
            # 4. Validate required tags are present
            #
            # Returns: (processed_tags, warnings, missing_required_tags)
            self.tags, tag_warnings, missing = process_tags_single_pass(self.tags)

            # Handle warnings from tag processing (structured logging for observability)
            if tag_warnings:
                self._warnings.extend(tag_warnings)
                # Categorize warnings for metrics tracking
                for warning in tag_warnings:
                    if "not in allowlist" in warning:
                        _tracking_errors["allowlist_filtered"] += 1
                    elif ("exceeds" in warning and "cardinality" in warning) or (
                        "exceeds" in warning and "rate limit" in warning
                    ):
                        _tracking_errors["cardinality_exceeded"] += 1

            # Handle missing required tags (compliance error)
            if missing:
                _tracking_errors["missing_required_tags"] += 1
                logger.error(
                    f"Compliance Error: Missing mandatory tags: {', '.join(missing)}. "
                    f"Add missing tags via tags={{ {', '.join(repr(t) for t in missing)}: '...' }} "
                    f"or remove requirement via vetch.require_tags([...])"
                )
                self._tracking_disabled = True
                # Still create a context to avoid AttributeErrors if user accesses it,
                # but it won't capture anything.
                return

            # 5. Create and enter tracking context
            self._tracking_ctx = TrackingContext(
                region=self.region,
                tags=self.tags,
                energy_override=self._energy_override,
            )
            self._tracking_ctx.warnings = list(self._warnings)  # Start with current warnings
            self._tracking_ctx.__enter__()

            # 6. Set up SDK patches (skipped for manual record_usage() contexts,
            # which have supplied usage and no live call to intercept).
            if not self._manual:
                self._setup_patches()
        except Exception as e:
            logger.warning(f"Vetch setup failed, tracking disabled: {e}")
            self._tracking_disabled = True

    def _teardown(self, exc_type: type[BaseException] | None) -> None:
        """Teardown context state."""
        latency_ms = None
        if self._manual:
            # Manual record: there is no live call to time. Use the caller's
            # supplied duration (or None) rather than the stopwatch, which would
            # otherwise report a fraction of a millisecond of emit overhead as if
            # it were inference latency.
            latency_ms = self._manual_latency_ms
        elif self._start_time is not None:
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

            from vetch._security import sanitize_for_url, sanitize_traceback

            # Sanitize both traceback and exception message for security
            # This prevents leaking API keys, passwords, or file paths in URLs
            tb_string = traceback.format_exc()
            sanitized_tb = sanitize_traceback(tb_string, remove_local_vars=True)
            sanitized_msg = sanitize_for_url(str(e), max_length=200)

            params = {
                "title": f"Alpha Error: {type(e).__name__}",
                "body": (
                    f"Vetch version: {__version__}\n"
                    f"Exception: {type(e).__name__}: {sanitized_msg}\n\n"
                    f"Sanitized Traceback (secrets/local vars removed):\n```\n{sanitized_tb}\n```"
                ),
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
        from vetch.calculation import InferenceMetrics, prepare_inference_metrics

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
        usage: Usage | None = None
        is_stream = False
        is_embedding = False
        accumulated_chars = 0
        accumulated_tik_tokens = 0
        content_type_hint = "en"

        # Cache tokens
        cache_read_tokens: int | None = None
        cache_creation_tokens: int | None = None
        cache_creation_1h_tokens: int | None = None
        visible_output_chars: int | None = None
        finish_reason: str | None = None
        requested_max_tokens: int | None = None
        tools_offered: list[CapabilityRef] | None = None
        tools_invoked: list[CapabilityRef] | None = None
        tool_call_count: int | None = None
        capabilities_invoked: list[CapabilityRef] | None = None
        tool_schema_tokens: dict[str, int] | None = None

        if captured is not None:
            model = captured.model
            provider = captured.provider
            usage = captured.usage
            is_stream = captured.is_stream
            is_embedding = captured.is_embedding
            accumulated_chars = captured.accumulated_chars
            accumulated_tik_tokens = captured.accumulated_tik_tokens
            content_type_hint = captured.content_type_hint
            raw_crt = captured.cache_read_tokens
            cache_read_tokens = raw_crt if isinstance(raw_crt, int) else None
            raw_cct = captured.cache_creation_tokens
            cache_creation_tokens = raw_cct if isinstance(raw_cct, int) else None
            raw_cct_1h = captured.cache_creation_1h_tokens
            cache_creation_1h_tokens = raw_cct_1h if isinstance(raw_cct_1h, int) else None
            visible_output_chars = captured.visible_output_chars
            finish_reason = captured.finish_reason
            requested_max_tokens = captured.requested_max_tokens
            tools_offered = captured.tools_offered
            tools_invoked = captured.tools_invoked
            tool_call_count = captured.tool_call_count
            capabilities_invoked = captured.capabilities_invoked
            tool_schema_tokens = captured.tool_schema_tokens

            if self._tracking_ctx is not None and self._tracking_ctx.attribution_model:
                model = self._tracking_ctx.attribution_model

            # Override error info from captured call if present
            if captured.error:
                error = True
                error_type = captured.error_type

        # Explicit provider override (wrap(provider_hint=...) / instrument default).
        # Wins over the provider inferred from the model name or SDK client, so a
        # self-hosted model is not billed at cloud rates and PUE/water use the
        # right coefficients. Applies even when nothing was captured.
        if self._provider_hint:
            provider = self._provider_hint

        # Extract image count from usage for VLM energy calculation
        n_images = _infer_n_images_from_usage(usage)

        # Delegate all energy/carbon/cost calculations to calculation.py
        metrics: InferenceMetrics = prepare_inference_metrics(
            model=model,
            provider=provider,
            usage=usage,
            accumulated_chars=accumulated_chars,
            region=self.region,
            price_multiplier=self.price_multiplier,
            energy_override=cast("dict[str, Any]", self._energy_override),
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
            existing_warnings=list(self._warnings),
            accumulated_tik_tokens=accumulated_tik_tokens,
            content_type_hint=content_type_hint,
            n_images=n_images,
            cache_creation_1h_tokens=cache_creation_1h_tokens,
        )

        # Propagate usage_estimated counter for monitoring dashboards
        if metrics.usage_estimated:
            _tracking_errors["usage_estimated"] += 1
        if not metrics.model_known and model != "unknown":
            _tracking_errors["model_unknown"] += 1

        # Unpack metrics for event construction
        energy_wh = metrics.energy_wh
        energy_tier = metrics.energy_tier
        energy_uncertainty_pct: int | None = metrics.energy_uncertainty_pct
        energy_source = metrics.energy_source
        energy_basis = metrics.energy_basis
        model_known = metrics.model_known
        model_match = metrics.model_match
        carbon_g = metrics.carbon_g
        pue = metrics.pue
        pue_tier = metrics.pue_tier
        pue_source = metrics.pue_source
        water_l = metrics.water_l
        embodied_carbon_g = metrics.embodied_carbon_g
        cost_usd = metrics.cost_usd
        cost_in_usd = metrics.cost_in_usd
        cost_out_usd = metrics.cost_out_usd
        cost_cache_write_usd = metrics.cost_cache_write_usd
        cost_cache_read_usd = metrics.cost_cache_read_usd
        billing_tier = metrics.billing_tier
        signal_quality = metrics.signal_quality
        grid_val = metrics.grid_val
        grid_ts = metrics.grid_ts
        usage = metrics.usage
        usage_estimated = metrics.usage_estimated
        usage_estimation_method = metrics.usage_estimation_method
        cache_energy_saving_wh = metrics.cache_energy_saving_wh
        cache_cost_saving_usd = metrics.cache_cost_saving_usd
        cache_carbon_saving_g = metrics.cache_carbon_saving_g
        tracking_degraded = metrics.tracking_degraded
        request_fingerprint = metrics.request_fingerprint

        # Combine all warnings (from prepare_inference_metrics and captured call)
        all_warnings = list(metrics.warnings)
        if captured and captured.warnings:
            all_warnings.extend(captured.warnings)
        from vetch.capabilities import capability_fully_cached_warning

        cap_warning = capability_fully_cached_warning(
            {
                "tools_offered": tools_offered,
                "tools_invoked": tools_invoked,
                "usage": usage,
                "cache_read_tokens": cache_read_tokens,
            }
        )
        if cap_warning:
            all_warnings.append(cap_warning)

        # Detect multimodal requests (image/audio/video)
        multimodal = False
        if usage and isinstance(usage, dict):
            multimodal = bool(usage.get("image") or usage.get("audio"))

        # Detect batch API usage (OpenAI Batch API gets 50% cost discount)
        is_batch = False
        if model and provider:
            model_lower = model.lower()
            if "batch" in model_lower or (
                provider == "openai" and billing_tier and "batch" in billing_tier.lower()
            ):
                is_batch = True

        # Apply batch API discount (OpenAI Batch API is 50% off list price)
        if is_batch and cost_usd is not None:
            cost_usd *= 0.5
            cost_in_usd = cost_in_usd * 0.5 if cost_in_usd is not None else None
            cost_out_usd = cost_out_usd * 0.5 if cost_out_usd is not None else None
            cache_cost_saving_usd = (
                cache_cost_saving_usd * 0.5
                if cache_cost_saving_usd is not None
                else None
            )
            if billing_tier and "batch" not in billing_tier.lower():
                billing_tier = f"{billing_tier} (batch 50% discount)"

        # v0.4.0: Derive explicit p5/p95 bounds from the existing uncertainty %.
        # No new modelling — just expose the uncertainty band as absolute
        # numbers so downstream tooling (dashboards, compliance reports)
        # doesn't have to repeat the math. Caps the lower bound at 0 to
        # avoid negative values when uncertainty is high.
        energy_p5_wh: float | None = None
        energy_p95_wh: float | None = None
        carbon_p5_g: float | None = None
        carbon_p95_g: float | None = None
        if energy_wh is not None and energy_uncertainty_pct is not None:
            band = energy_wh * (energy_uncertainty_pct / 100.0)
            energy_p5_wh = max(energy_wh - band, 0.0)
            energy_p95_wh = energy_wh + band
        if carbon_g is not None and energy_uncertainty_pct is not None:
            band = carbon_g * (energy_uncertainty_pct / 100.0)
            carbon_p5_g = max(carbon_g - band, 0.0)
            carbon_p95_g = carbon_g + band

        if capabilities_invoked is None:
            from vetch.capabilities import derive_capabilities_invoked

            capabilities_invoked = derive_capabilities_invoked(
                is_embedding=is_embedding,
                usage=cast("dict[str, Any]", usage) if usage else None,
                model=model,
            )

        from vetch.capabilities import sanitize_capability_capture_fields

        tools_offered, tools_invoked, tool_schema_tokens = sanitize_capability_capture_fields(
            tools_offered=tools_offered,
            tools_invoked=tools_invoked,
            tool_schema_tokens=tool_schema_tokens,
        )

        # Build event
        self._event = InferenceEvent(
            schema_version=SCHEMA_VERSION,
            vetch_version=__version__,
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            model=model,
            provider=provider,
            model_known=model_known,
            model_match=model_match,
            multimodal=multimodal,
            usage=usage,
            accumulated_chars=accumulated_chars if is_stream else None,
            estimated_energy_wh=energy_wh,
            estimated_carbon_g=carbon_g,
            estimated_water_l=water_l,
            estimated_cost_usd=cost_usd,
            estimated_cost_input_usd=cost_in_usd,
            estimated_cost_output_usd=cost_out_usd,
            estimated_cost_cache_write_usd=(
                cost_cache_write_usd if cost_cache_write_usd > 0 else None
            ),
            estimated_cost_cache_read_usd=(
                cost_cache_read_usd if cost_cache_read_usd > 0 else None
            ),
            billing_tier=billing_tier,
            signal_quality=signal_quality,
            energy_tier=energy_tier,
            energy_uncertainty_pct=energy_uncertainty_pct,
            energy_p5_wh=energy_p5_wh,
            energy_p95_wh=energy_p95_wh,
            carbon_p5_g=carbon_p5_g,
            carbon_p95_g=carbon_p95_g,
            energy_source=energy_source,
            energy_override_source=(
                self._energy_override.get("source") if self._energy_override else None
            ),
            energy_basis=energy_basis,
            grid_intensity_gco2e_kwh=grid_val,
            grid_intensity_timestamp=grid_ts,
            grid_intensity_time_of_day=False,  # TODO: Implement hourly grid data
            region=self.region,
            embodied_carbon_g=embodied_carbon_g,
            pue=pue,
            pue_tier=pue_tier,
            pue_source=pue_source,
            is_stream=is_stream,
            is_batch=is_batch,
            is_embedding=is_embedding if captured else False,
            complete=not error and (captured.complete if captured else True),
            latency_ms=latency_ms,
            visible_output_chars=visible_output_chars,
            finish_reason=finish_reason,
            requested_max_tokens=requested_max_tokens,
            tools_offered=tools_offered,
            tools_invoked=tools_invoked,
            tool_call_count=tool_call_count,
            capabilities_invoked=capabilities_invoked,
            tool_schema_tokens=tool_schema_tokens,
            tags=self.tags,
            error=error,
            error_type=error_type,
            retry_count=0,
            tracking_disabled=self._tracking_disabled,
            tracking_degraded=tracking_degraded,
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
            cache_energy_saving_wh=cache_energy_saving_wh,
            cache_cost_saving_usd=cache_cost_saving_usd,
            cache_carbon_saving_g=cache_carbon_saving_g,
            session_id=active_session.session_id if active_session else None,
            trace_id=None,  # TODO: Extract from OpenTelemetry context
            span_id=None,  # TODO: Extract from OpenTelemetry context
            parent_span_id=None,  # TODO: Extract from OpenTelemetry context
            request_fingerprint=request_fingerprint,
        )

        # Empty wrap(): a context that intercepted no call, hit no error, is not a
        # manual record, and was not tracking-disabled has nothing to report. Keep
        # self._event populated so callers inspecting ctx.event still get a
        # placeholder, but do not push this unknown/unknown event into the
        # emitted / stored / exported / session stream, where it would otherwise
        # pollute aggregation. Error and compliance (tracking_disabled) events,
        # and manual record_usage() events, still flow through below.
        if (
            captured is None
            and not error
            and not self._manual
            and not self._tracking_disabled
        ):
            # Breadcrumb (debug-only, no spam): if a real inference actually ran
            # inside this wrap() but nothing was captured, the client likely
            # wasn't instrumented. Surface it for debugging without emitting a
            # misleading unknown/unknown event into the stream.
            logger.debug(
                "wrap() closed with no intercepted call; no event emitted. If an "
                "inference ran here, instrument the client (vetch.instrument()) "
                "or meter it explicitly with vetch.record_usage()."
            )
            return

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

        # Try to attach to active OTel span (transport cap applied in otel layer)
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

        # Auto-export to OpenTelemetry if enabled
        try:
            from vetch.exporters.opentelemetry import (
                export_event_as_span,
                is_auto_export_enabled,
            )

            if is_auto_export_enabled():
                export_event_as_span(self._event)
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
    provider_hint: str | None = None,
) -> Generator[VetchContext, None, None]:
    """Context manager for tracking LLM inference.

    This is a convenience function that creates a VetchContext.
    See VetchContext for full documentation.

    Args:
        region: Grid region for carbon calculation.
        tags: Key-value pairs for cost attribution.
        energy_override: User-provided energy values.
        price_multiplier: Factor to adjust list pricing (e.g., 0.8 for 20% discount).
        provider_hint: Explicit provider override (e.g. "self-hosted").
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
        provider_hint=provider_hint,
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
    provider_hint: str | None = None,
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
        provider_hint=provider_hint,
        _disabled=_disabled,
    )
    async with ctx:
        yield ctx


def record_usage(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    region: str | None = None,
    provider_hint: str | None = None,
    tags: dict[str, str] | None = None,
    energy_override: dict[str, object] | None = None,
    price_multiplier: float = 1.0,
    reasoning_tokens: int | None = None,
    cache_read_tokens: int | None = None,
    cache_creation_tokens: int | None = None,
    duration_ms: float | None = None,
    emit: bool = True,
) -> InferenceEvent | None:
    """Meter a call Vetch did not intercept, from usage you already have.

    The escape hatch for any model reached outside a supported SDK — e.g. a
    self-hosted model called over raw HTTP — where the response body already
    carries the token counts. Runs the same calculation and emit path as an
    instrumented call, so the resulting event is schema-identical and flows into
    the same aggregation, budgets, sessions, and exporters.

    Each call emits exactly one event and returns it, independent of any active
    ``wrap()``. Attach attribution with this function's own ``tags`` / ``region``
    rather than wrapping it — ``wrap()`` is for intercepting live SDK calls, and
    a ``wrap()`` placed around ``record_usage`` will additionally emit its own
    (empty) event on exit.

    Args:
        model: Model identifier (used for registry / energy resolution).
        input_tokens: Prompt (input) tokens.
        output_tokens: Visible completion tokens, excluding reasoning.
        region: Grid region for carbon. Falls back to VETCH_REGION / inference.
        provider_hint: Explicit provider (e.g. ``"self-hosted"`` for cost 0 with
            energy/carbon still computed, ``"openai-compatible"`` for unknown
            cost). Defaults to the provider inferred from the model name. An
            unrecognised value is used but flagged in ``vetch_warnings``.
        tags: Attribution tags for this event.
        energy_override: User-provided energy coefficients (see ``wrap``).
        price_multiplier: Factor applied to list pricing (e.g. 0.8 for a 20%
            discount). Ignored when cost is 0 (self-hosted) or unknown.
        reasoning_tokens: Reasoning/thinking tokens, surfaced separately and
            counted toward output energy (as for instrumented reasoning models).
        cache_read_tokens: Prompt-cache read tokens (e.g. vLLM automatic prefix
            caching). Charged at the cache-read energy factor (and, for priced
            providers, the discounted cache-read price). Pass counts in the
            resolved provider's native convention: for OpenAI-style / self-hosted
            (the default), ``input_tokens`` INCLUDES the cached tokens (as vLLM
            and OpenAI report ``prompt_tokens``); for Anthropic they are disjoint
            (``input_tokens`` is the fresh, uncached count).
        cache_creation_tokens: Prompt-cache write tokens, if the endpoint reports
            them.
        duration_ms: Measured latency of the call, if known. Manual events have
            no live call to time, so ``latency_ms`` is otherwise ``None`` rather
            than a fabricated value.
        emit: If False, compute and return the event without emitting.

    Returns:
        The ``InferenceEvent``, or ``None`` when Vetch is disabled.
    """
    from vetch import _DISABLED  # kill switch (VETCH_DISABLED / VETCH_ENABLED)

    if _DISABLED:
        return None

    from vetch.calculation import _infer_provider_from_model

    in_tok = max(0, input_tokens)
    out_tok = max(0, output_tokens)
    reason_tok = max(0, reasoning_tokens) if reasoning_tokens is not None else 0

    provider = provider_hint or _infer_provider_from_model(model) or "unknown"

    usage: Usage = {
        "text": {
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": in_tok + out_tok + reason_tok,
        }
    }
    if reason_tok > 0:
        usage["reasoning"] = {
            "input_tokens": 0,
            "output_tokens": reason_tok,
            "total_tokens": reason_tok,
        }

    # Always our own non-patching emit cycle: one event per call, returned to the
    # caller, regardless of any active wrap(). A standalone VetchContext still
    # inherits region/tags from a parent context via __init__ if one is open.
    ctx = VetchContext(
        region=region,
        tags=tags,
        energy_override=energy_override,
        price_multiplier=price_multiplier,
        emit=emit,
        provider_hint=provider_hint,
        _manual=True,
        _manual_latency_ms=duration_ms,
    )
    with ctx:
        active = get_active_context()
        if active is not None:
            active.capture(
                model=model,
                provider=provider,
                usage=usage,
                complete=True,
                cache_read_tokens=cache_read_tokens,
                cache_creation_tokens=cache_creation_tokens,
            )
    return ctx.event


def get_tracking_stats() -> dict[str, int]:
    """Get tracking error statistics for observability.

    Returns counters for different types of tracking issues/warnings.
    Useful for monitoring dashboards and alerting.

    Returns:
        Dictionary with error type counts:
        - tag_validation_failed: Tag processing errors
        - missing_required_tags: Compliance violations
        - cardinality_exceeded: Tag cardinality limit warnings
        - allowlist_filtered: Tags filtered by allowlist
        - energy_calculation_failed: Energy estimation errors
        - grid_lookup_failed: Grid intensity lookup failures
        - model_unknown: Unknown model fallbacks
        - usage_estimated: Token usage heuristic estimations

    Example::

        stats = vetch.wrapper.get_tracking_stats()
        if stats["model_unknown"] > 100:
            logger.warning(f"Many unknown models: {stats['model_unknown']}")
    """
    return dict(_tracking_errors)  # Return copy to prevent external modification
