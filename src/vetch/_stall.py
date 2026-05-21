"""Stall circuit breaker helpers (v0.4.0).

Internal module — not part of the public API. Provides the small shared logic
that every provider wrapper calls before/around the actual LLM API call:

- :func:`apply_stall_action` — Check whether the active session has a
  stall flag set, and take the configured action (log/warn/kill/reroute).
- :func:`looks_like_param_mismatch` — Heuristic for fail-open reroute:
  was a 400/parameter-mismatch error caused by the substituted model?

The helpers are deliberately fail-open. If anything inside them goes wrong
(missing session, broken advisory, etc.) they log and return a neutral result
so the LLM call still proceeds. The only exception that propagates is
:class:`vetch.StallDetected` itself when ``action="kill"``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vetch.context import TrackingContext

logger = logging.getLogger(__name__)


def apply_stall_action(
    kwargs: dict[str, Any],
    ctx: TrackingContext | None,
) -> tuple[bool, str | None]:
    """Check the active session's stall flag and take the configured action.

    Should be called by every provider wrapper before invoking the original
    LLM API method. Mutates ``kwargs`` in-place when ``action="reroute"``
    by replacing ``kwargs["model"]``.

    Args:
        kwargs: The keyword arguments that will be passed to the original
            API call. Modified in-place for reroute.
        ctx: Active TrackingContext (from ``get_active_context()``), or None
            if not inside a ``wrap()`` block. Used to record warnings.

    Returns:
        ``(rerouted, original_model)``:

        - ``rerouted`` (bool): True if ``kwargs["model"]`` was replaced.
        - ``original_model`` (str | None): The model the user originally
          requested, captured before substitution. Used by the caller to
          retry with the original model if the rerouted call fails.

    Raises:
        StallDetected: If the action is ``"kill"``.

    Fail-open guarantee: any exception other than ``StallDetected`` is
    caught and logged. The function returns ``(False, None)`` so the
    LLM call proceeds normally.
    """
    try:
        from vetch.config import get_stall_action
        from vetch.exceptions import StallDetected
        from vetch.session import get_active_session

        session = get_active_session()
        if session is None or not session.stall_triggered:
            return (False, None)

        action, fallback_model = get_stall_action()
        advisory = session.stall_advisory
        wasted = advisory.potential_savings_usd if advisory else 0.0
        count = int(advisory.request_count or 0) if advisory else 0
        if count <= 0:
            try:
                count = int(session.stats.summary().get("recent_window_size") or 0)
            except Exception:
                count = 0

        # Export to OTLP if configured — fail-open, never blocks the call
        try:
            from vetch.otel import export_advisory_otlp, is_otlp_configured
            if is_otlp_configured():
                model_name = kwargs.get("model") or (advisory.code if advisory else None)
                tags = dict(ctx.tags) if ctx and ctx.tags else None
                export_advisory_otlp(
                    code="STALL-001",
                    severity=advisory.severity if advisory else "WARNING",
                    action=action or "log",
                    session_id=getattr(session, "session_id", None),
                    model=str(model_name) if model_name else None,
                    estimated_waste_usd=float(wasted or 0.0),
                    tags=tags,
                )
        except Exception:
            pass  # OTLP export never blocks inference

        if action == "kill":
            raise StallDetected(
                f"Agentic stall detected. {count} of recent calls produced "
                f"low output, ~${wasted:.2f} wasted. Stopping the loop. "
                "Call session.clear_stall() to re-arm.",
                wasted_cost_usd=wasted or 0.0,
                request_count=count,
                fallback_model=fallback_model,
            )

        if action == "warn":
            logger.warning(
                "STALL-001: Agentic stall detected (%d calls, ~$%.2f wasted). "
                "Set stall_action='kill' or 'reroute' to stop the loop.",
                count,
                wasted or 0.0,
            )
            return (False, None)

        if action == "reroute" and fallback_model:
            original_model = kwargs.get("model")
            if original_model is None:
                # Provider doesn't pass the model name in kwargs (e.g. Vertex
                # AI binds it to the model object). Reroute can't substitute
                # transparently here. Log once and continue with the original
                # model — kill/warn/log all still work for this provider.
                logger.warning(
                    "STALL-001 reroute requested but model name is not in kwargs. "
                    "This provider does not support transparent model substitution; "
                    "use stall_action='kill' instead. Continuing with original model."
                )
                return (False, None)
            kwargs["model"] = fallback_model
            if ctx is not None:
                ctx.warnings.append(
                    f"STALL-001 reroute: {original_model} -> {fallback_model}"
                )
            return (True, original_model)

        if action == "reroute" and not fallback_model:
            logger.warning(
                "STALL-001: reroute configured but no fallback_model set - "
                "falling back to warn."
            )
            return (False, None)

        # action == "log" or unrecognised — current behaviour, no action.
        return (False, None)

    except Exception as exc:
        # Re-raise StallDetected; swallow everything else (fail-open).
        from vetch.exceptions import StallDetected

        if isinstance(exc, StallDetected):
            raise
        logger.warning("Vetch stall handling failed (fail-open): %s", exc)
        return (False, None)


# Exception class names that indicate a parameter mismatch / 4xx-style
# provider rejection. If a rerouted call fails with one of these, we retry
# with the original model. This intentionally does NOT include things like
# ``AuthenticationError`` or ``RateLimitError`` — those would have failed
# with the original model too.
_PARAM_MISMATCH_EXC_NAMES: frozenset[str] = frozenset({
    "BadRequestError",        # OpenAI, Anthropic
    "UnprocessableEntityError",  # OpenAI
    "InvalidRequestError",    # legacy OpenAI
    "InvalidArgument",        # google-genai / vertexai
    "InvalidArgumentError",
})


def looks_like_param_mismatch(exc: Exception) -> bool:
    """Heuristic: is this exception caused by an incompatible model parameter?

    Used by provider wrappers in fail-open reroute: if the substituted
    fallback model rejects the call (e.g. it doesn't support
    ``max_completion_tokens`` or ``response_format``), we want to retry
    the call with the original model so the user's app keeps working.

    Args:
        exc: The exception raised by the provider SDK.

    Returns:
        True if this looks like a parameter-mismatch / 400 error, False
        for auth errors, rate limits, server errors, network errors, etc.
    """
    name = type(exc).__name__
    if name in _PARAM_MISMATCH_EXC_NAMES:
        return True
    # Status-code attribute check (covers SDK-internal subclasses we don't know)
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    return (
        isinstance(status, int)
        and 400 <= status < 500
        and status not in (401, 403, 429)
    )
