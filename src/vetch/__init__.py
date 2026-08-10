"""Vetch: Planet-aware observability for LLM inference.

NOTE: We use `from __future__ import annotations` to enable Python 3.10+
style type hints (str | None) on Python 3.9. This defers evaluation.

Vetch wraps LLM API calls to log energy consumption, cost, and carbon
per inference using live grid data. It never reads prompt or completion
content - only usage metadata from the response.

Basic usage::

    from vetch import wrap

    with wrap(region="us-east-1") as ctx:
        response = client.chat.completions.create(...)

    print(f"Energy: {ctx.event['estimated_energy_wh']} Wh")

Emergency kill switch::

    VETCH_DISABLED=true python my_app.py

"""

from __future__ import annotations

import os
import sys
import threading
from typing import TYPE_CHECKING

# P0: Emergency kill switch - check immediately on import
# Support both VETCH_DISABLED (legacy) and VETCH_ENABLED (new)
# Priority: VETCH_DISABLED > VETCH_ENABLED > default (enabled)
_disabled_env = os.environ.get("VETCH_DISABLED", "").lower() in ("true", "1", "yes")
_enabled_env = os.environ.get("VETCH_ENABLED", "true").lower() not in ("false", "0", "no")
_DISABLED = _disabled_env or not _enabled_env
_default_region: str | None = None  # Default region set via instrument()
_default_tags: dict[str, str] | None = None  # Default tags set via instrument()
_default_energy_override: dict[str, object] | None = None  # Default calibration override
_default_provider_hint: str | None = None  # Default provider override set via instrument()
# Only print kill-switch message if VETCH_VERBOSE=true (opt-in for debugging)
if _DISABLED and os.environ.get("VETCH_VERBOSE", "").lower() in ("true", "1", "yes"):
    reason = "VETCH_DISABLED=true" if _disabled_env else "VETCH_ENABLED=false"
    print(f"vetch: disabled via {reason}", file=sys.stderr)

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager as AsyncContextManager

    from vetch.wrapper import VetchContext

try:
    from importlib.metadata import PackageNotFoundError as _PNFError
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("vetch")
except _PNFError:
    __version__ = "0.11.0"  # fallback when running from source; keep in sync with pyproject.toml
__all__ = [
    "wrap",
    "awrap",
    "record_usage",
    "Session",
    "instrument",
    "uninstrument",
    "instrumentation_status",
    "is_client_instrumented",
    "require_tags",
    "add_global_tags",
    "set_tag_cardinality_limit",
    "set_tag_allowlist",
    "set_redacted_tags",
    "configure_storage",
    "query_usage",
    "__version__",
    "get_session_stats",
    "generate_advisories",
    # v0.1.5: Budget alerts
    "set_budget",
    "on_budget_alert",
    "get_budget_status",
    # v0.1.5: OTLP export
    "configure_otlp_export",
    "export_advisory_otlp",
    # v0.2.3: HTTP endpoint output
    "configure_http_endpoint",
    # v0.1.5: Green signal API
    "get_cleanest_region",
    # v0.1.5: Logging control
    "set_log_level",
    # v0.2.0: OpenTelemetry exporter
    "configure_otel_export",
    "export_event_as_span",
    # v0.4.0: Stall circuit breaker
    "set_stall_action",
    "get_stall_action",
    "StallDetected",
    "VetchInterrupt",
    # v0.6.0: Configurable advisory thresholds + push callbacks
    "set_advisory_thresholds",
    # v0.10.x: Confidence-aware resolution + strict reporting mode
    "set_min_match_confidence",
    "get_min_match_confidence",
    "rollup_confidence_from_events",
    "filter_events_by_confidence",
    "require_confidence",
    "ConfidenceError",
    "configure_capabilities",
    "set_expected_capabilities",
    "set_model_capability_map",
    "rollup_capability_summary_from_events",
    "on_advisory",
]

# Track instrumented state
_instrumented = False
_instrument_lock = threading.Lock()


def add_global_tags(tags: dict[str, str]) -> None:
    """Set tags that will be automatically added to every inference event.

    Example::

        vetch.add_global_tags({"env": "production", "service": "chat-api"})
    """
    from vetch.config import add_global_tags as _add_global_tags

    _add_global_tags(tags)


def require_tags(tags: list[str]) -> None:
    """Set global mandatory tags for compliance.

    Example::

        vetch.require_tags(["feature_id", "cost_center"])
    """
    from vetch.config import require_tags as _require_tags

    _require_tags(tags)


def set_tag_cardinality_limit(limit: int) -> None:
    """Set maximum unique values per tag key to prevent DoS attacks.

    Default: 1000 unique values per tag key.

    Args:
        limit: Maximum unique values allowed per tag key.

    Example::

        vetch.set_tag_cardinality_limit(500)  # More restrictive limit
    """
    from vetch.config import set_tag_cardinality_limit as _set_limit

    _set_limit(limit)


def set_tag_allowlist(allowed_tags: list[str]) -> None:
    """Set allowed tag keys for strict security environments.

    Only tags in the allowlist are permitted. Use this to prevent
    accidental leakage of sensitive data via tags.

    Args:
        allowed_tags: List of allowed tag keys.

    Example::

        vetch.set_tag_allowlist(["team", "env", "service"])
    """
    from vetch.config import set_tag_allowlist as _set_allowlist

    _set_allowlist(allowed_tags)


def set_redacted_tags(sensitive_keys: list[str]) -> None:
    """Set tag keys that should be hashed for PII protection.

    Values for these keys will be SHA256-hashed before logging/export.
    Use this to prevent accidental PII leakage.

    Args:
        sensitive_keys: List of tag keys to redact.

    Example::

        vetch.set_redacted_tags(["user_email", "customer_id"])
    """
    from vetch.config import set_redacted_tags as _set_redacted

    _set_redacted(sensitive_keys)


def get_default_region() -> str | None:
    """Get the default region for auto-instrumentation.

    Checks module-level config first, then falls back to VETCH_REGION env var.

    Returns:
        Default region or None if not set.
    """
    global _default_region
    if _default_region is not None:
        return _default_region
    return os.environ.get("VETCH_REGION")


def get_default_tags() -> dict[str, str] | None:
    """Get the default tags for auto-instrumentation.

    Returns the tags set via instrument().

    Returns:
        Default tags or None if not set.
    """
    global _default_tags
    return _default_tags


def get_default_energy_override() -> dict[str, object] | None:
    """Get the default energy override set by ``instrument()``.

    Returns:
        Energy override values or None if not set.
    """
    global _default_energy_override
    return _default_energy_override


def get_default_provider_hint() -> str | None:
    """Get the default provider override set by ``instrument()``.

    Applied to auto-instrumented calls so a whole deployment can be pinned to,
    e.g., ``"self-hosted"`` without wrapping each call site.

    Returns:
        Default provider hint or None if not set.
    """
    global _default_provider_hint
    return _default_provider_hint


def instrument(
    region: str | None = None,
    tags: dict[str, str] | None = None,
    energy_override: dict[str, object] | None = None,
    provider_hint: str | None = None,
) -> bool:
    """Auto-instrument all detected LLM SDK clients.

    Call once at application startup to automatically track all LLM calls
    without needing to use the `wrap()` context manager.

    This is useful for frameworks like LangChain or LlamaIndex, but only when
    the framework constructs a supported raw SDK client (at a tested version)
    *after* ``instrument()`` runs. ``instrument()`` patches SDK clients at
    construction time and can only patch modules already imported (the
    ``sys.modules`` gate); an SDK imported afterwards is silently not
    instrumented. Import order matters — see :func:`instrumentation_status`.

    Args:
        region: Default grid region for carbon calculation.
        tags: Default tags to add to all events.
        energy_override: Default energy values for auto-instrumented calls.
        provider_hint: Default provider override applied to auto-instrumented
            calls (e.g. "self-hosted" to pin a whole deployment).

    Returns:
        True if any clients were instrumented, False otherwise.

    Example::

        import vetch
        import openai

        # Call once at startup
        vetch.instrument(region="us-east-1", tags={"service": "chat-api"})

        # All OpenAI calls are now automatically tracked
        client = openai.OpenAI()
        response = client.chat.completions.create(...)
        # Events emitted automatically!

    Note:
        - Works with OpenAI, Anthropic, Vertex AI, and Google GenAI SDKs
        - Safe to call multiple times (idempotent)
        - Thread-safe for concurrent initialization
        - Import order matters: an SDK imported *after* ``instrument()`` runs is
          not patched. Import your SDKs first, or call ``instrument()`` again
          afterwards, and check :func:`instrumentation_status` for coverage.
        - Set VETCH_DISABLED=true or VETCH_ENABLED=false to disable
    """
    global _instrumented, _default_region, _default_tags, _default_energy_override
    global _default_provider_hint

    if _DISABLED:
        return False

    # Thread-safe instrumentation
    with _instrument_lock:
        # Store default config even if instrumentation is already active. This lets
        # callers update attribution/calibration between isolated sandbox runs.
        if region:
            _default_region = region
        if tags:
            _default_tags = tags
            add_global_tags(tags)
        if energy_override is not None:
            _default_energy_override = energy_override
        if provider_hint is not None:
            _default_provider_hint = provider_hint

        if _instrumented:
            return True

        instrumented_any = False

        # Snapshot which SDKs the user imported before this call. instrument()
        # only patches already-imported modules, and the version-range check
        # below imports SDKs to read their versions, so capture this first.
        import sys as _sys

        _imported_at_entry = {
            _n: (_m in _sys.modules) for _n, (_m, _pm) in _INSTRUMENTATION_PROVIDERS.items()
        }

        # Try to instrument OpenAI
        try:
            from vetch.providers.openai import instrument_openai_module

            if instrument_openai_module():
                instrumented_any = True
        except (ImportError, ModuleNotFoundError):
            pass  # SDK not installed
        except Exception as e:
            import logging

            logging.getLogger("vetch").debug(f"Failed to instrument OpenAI: {e}")

        # Try to instrument Anthropic
        try:
            from vetch.providers.anthropic import instrument_anthropic_module

            if instrument_anthropic_module():
                instrumented_any = True
        except (ImportError, ModuleNotFoundError):
            pass  # SDK not installed
        except Exception as e:
            import logging

            logging.getLogger("vetch").debug(f"Failed to instrument Anthropic: {e}")

        # Try to instrument Azure OpenAI
        try:
            from vetch.providers.azure_openai import instrument_azure_openai_module

            if instrument_azure_openai_module():
                instrumented_any = True
        except (ImportError, ModuleNotFoundError):
            pass  # SDK not installed
        except Exception as e:
            import logging

            logging.getLogger("vetch").debug(f"Failed to instrument Azure OpenAI: {e}")

        # Try to instrument Vertex AI
        try:
            from vetch.providers.vertexai import instrument_vertexai_module

            if instrument_vertexai_module():
                instrumented_any = True
        except (ImportError, ModuleNotFoundError):
            pass  # SDK not installed
        except Exception as e:
            import logging

            logging.getLogger("vetch").debug(f"Failed to instrument Vertex AI: {e}")

        # Try to instrument Google GenAI
        try:
            from vetch.providers.genai import instrument_genai_module

            if instrument_genai_module():
                instrumented_any = True
        except (ImportError, ModuleNotFoundError):
            pass  # SDK not installed
        except Exception as e:
            import logging

            logging.getLogger("vetch").debug(f"Failed to instrument Google GenAI: {e}")

        # Try to instrument Ollama (native Python SDK)
        try:
            from vetch.providers.ollama import instrument_ollama_module

            if instrument_ollama_module():
                instrumented_any = True
        except (ImportError, ModuleNotFoundError):
            pass
        except Exception as e:
            import logging

            logging.getLogger("vetch").debug(f"Failed to instrument Ollama: {e}")

        # Warn if any installed SDKs are outside tested version ranges
        try:
            import logging as _logging

            from vetch.compat import get_all_sdk_versions

            _vetch_log = _logging.getLogger("vetch")
            for _sdk_name, _sdk_info in get_all_sdk_versions().items():
                if _sdk_info.installed and not _sdk_info.tested:
                    _vetch_log.warning(
                        f"vetch: {_sdk_name} {_sdk_info.version} is outside the tested "
                        f"version range. Instrumentation may behave unexpectedly."
                    )
        except Exception:
            pass

        # Warn about installed-but-not-imported SDKs (snapshot taken at entry).
        # instrument() can only patch modules already imported (the sys.modules
        # gate), so an SDK present but imported later is silently NOT
        # instrumented.
        try:
            import importlib.util as _ilu
            import logging as _logging

            _vetch_log = _logging.getLogger("vetch")
            for _name, (_imp_mod, _pm) in _INSTRUMENTATION_PROVIDERS.items():
                if _imported_at_entry.get(_name):
                    continue
                try:
                    _installed = _ilu.find_spec(_imp_mod) is not None
                except (ImportError, ModuleNotFoundError, ValueError):
                    _installed = False
                if _installed:
                    _vetch_log.warning(
                        "vetch: %s is installed but was not imported before "
                        "vetch.instrument(); it is NOT instrumented. Import it "
                        "first, or call vetch.instrument() again after importing.",
                        _name,
                    )
        except Exception:
            pass

        _instrumented = instrumented_any
        return instrumented_any


def uninstrument() -> bool:
    """Remove Vetch instrumentation from all SDK clients.

    Restores original SDK methods for clean test isolation. Call this in
    test teardown to ensure tests don't affect each other.

    Returns:
        True if uninstrumentation succeeded, False otherwise.

    Example::

        import pytest
        import vetch

        @pytest.fixture(autouse=True)
        def clean_vetch():
            yield
            vetch.uninstrument()  # Restore original SDK methods

    Note:
        - Safe to call multiple times (idempotent)
        - Clears all per-client/model patches
        - Resets module instrumentation state
    """
    global _instrumented

    if not _instrumented:
        return True

    uninstrumented_all = True

    # Try to uninstrument OpenAI
    try:
        from vetch.providers.openai import uninstrument_openai_module

        if not uninstrument_openai_module():
            uninstrumented_all = False
    except (ImportError, ModuleNotFoundError):
        pass  # SDK not installed
    except Exception as e:
        import logging

        logging.getLogger("vetch").debug(f"Failed to uninstrument OpenAI: {e}")
        uninstrumented_all = False

    # Try to uninstrument Anthropic
    try:
        from vetch.providers.anthropic import uninstrument_anthropic_module

        if not uninstrument_anthropic_module():
            uninstrumented_all = False
    except (ImportError, ModuleNotFoundError):
        pass  # SDK not installed
    except Exception as e:
        import logging

        logging.getLogger("vetch").debug(f"Failed to uninstrument Anthropic: {e}")
        uninstrumented_all = False

    # Try to uninstrument Azure OpenAI
    try:
        from vetch.providers.azure_openai import uninstrument_azure_openai_module

        if not uninstrument_azure_openai_module():
            uninstrumented_all = False
    except (ImportError, ModuleNotFoundError):
        pass  # SDK not installed
    except Exception as e:
        import logging

        logging.getLogger("vetch").debug(f"Failed to uninstrument Azure OpenAI: {e}")
        uninstrumented_all = False

    # Try to uninstrument Vertex AI
    try:
        from vetch.providers.vertexai import uninstrument_vertexai_module

        if not uninstrument_vertexai_module():
            uninstrumented_all = False
    except (ImportError, ModuleNotFoundError):
        pass  # SDK not installed
    except Exception as e:
        import logging

        logging.getLogger("vetch").debug(f"Failed to uninstrument Vertex AI: {e}")
        uninstrumented_all = False

    # Try to uninstrument Google GenAI
    try:
        from vetch.providers.genai import uninstrument_genai_module

        if not uninstrument_genai_module():
            uninstrumented_all = False
    except (ImportError, ModuleNotFoundError):
        pass  # SDK not installed
    except Exception as e:
        import logging

        logging.getLogger("vetch").debug(f"Failed to uninstrument Google GenAI: {e}")
        uninstrumented_all = False

    try:
        from vetch.providers.ollama import uninstrument_ollama_module

        uninstrument_ollama_module()
    except (ImportError, ModuleNotFoundError):
        pass
    except Exception as e:
        import logging

        logging.getLogger("vetch").debug(f"Failed to uninstrument Ollama: {e}")
        uninstrumented_all = False

    _instrumented = False
    return uninstrumented_all


# Provider label -> (SDK module for install/import detection, vetch provider
# module holding the ``_module_instrumented`` flag).
_INSTRUMENTATION_PROVIDERS: dict[str, tuple[str, str]] = {
    "openai": ("openai", "vetch.providers.openai"),
    "anthropic": ("anthropic", "vetch.providers.anthropic"),
    "azure_openai": ("openai", "vetch.providers.azure_openai"),  # Azure uses the openai SDK
    "vertexai": ("google.cloud.aiplatform", "vetch.providers.vertexai"),
    "google_genai": ("google.genai", "vetch.providers.genai"),
    "ollama": ("ollama", "vetch.providers.ollama"),
}


def instrumentation_status() -> dict[str, dict[str, object]]:
    """Report per-provider instrumentation coverage.

    For each supported provider (``openai``, ``anthropic``, ``azure_openai``,
    ``vertexai``, ``google_genai``, ``ollama``) returns a dict with:

    - ``installed``: the SDK is importable (``importlib.util.find_spec``)
    - ``imported``: the SDK module is already in ``sys.modules``
    - ``instrumented``: Vetch has patched the SDK **module** (constructor /
      factory hooks). This does not guarantee every client instance is wrapped —
      use :func:`is_client_instrumented` on a specific client when verifying
      instance-level coverage.
    - ``version``: detected SDK distribution version, or ``None``
    - ``tested``: the version is within Vetch's tested range (when defined)

    ``instrument()`` only patches an SDK imported before it ran, so
    ``installed and not imported`` marks a provider that is present but was not
    instrumented (import it before calling ``instrument()``).
    """
    import importlib
    import importlib.util
    import sys

    # Snapshot "imported" BEFORE any SDK version lookup: get_all_sdk_versions()
    # imports openai/vertexai to read their versions, which would otherwise make
    # them appear imported here.
    imported_at_call = {
        name: (import_mod in sys.modules)
        for name, (import_mod, _pm) in _INSTRUMENTATION_PROVIDERS.items()
    }

    from vetch.compat import get_all_sdk_versions

    versions = get_all_sdk_versions()
    status: dict[str, dict[str, object]] = {}
    for name, (import_mod, provider_mod) in _INSTRUMENTATION_PROVIDERS.items():
        try:
            installed = importlib.util.find_spec(import_mod) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            installed = False

        imported = imported_at_call[name]

        instrumented = False
        try:
            mod = importlib.import_module(provider_mod)
            instrumented = bool(getattr(mod, "_module_instrumented", False))
        except Exception:
            instrumented = False

        # Azure reuses the openai SDK version. Others fall back to provider key.
        vinfo = versions.get(name)
        if vinfo is None and name == "azure_openai":
            vinfo = versions.get("openai")

        status[name] = {
            "installed": installed,
            "imported": imported,
            "instrumented": instrumented,
            "version": vinfo.version if vinfo else None,
            "tested": vinfo.tested if vinfo else False,
        }
    return status


def is_client_instrumented(client: object) -> bool:
    """Return whether a concrete SDK client instance is Vetch-wrapped.

    Unlike :func:`instrumentation_status` (module-level patch flags), this
    probes the client object for Vetch wrappers on common call paths.
    """
    from vetch.proxy import is_vetch_patched

    if getattr(client, "vetch_patched", False) is True:
        return True
    messages = getattr(client, "messages", None)
    if messages is not None and is_vetch_patched(getattr(messages, "create", None)):
        return True
    chat = getattr(client, "chat", None)
    if chat is not None:
        completions = getattr(chat, "completions", None)
        if completions is not None and is_vetch_patched(getattr(completions, "create", None)):
            return True
    responses = getattr(client, "responses", None)
    if responses is not None:
        if is_vetch_patched(getattr(responses, "create", None)) or is_vetch_patched(
            getattr(responses, "parse", None)
        ):
            return True
    models = getattr(client, "models", None)
    if models is not None:
        generate = getattr(models, "generate_content", None)
        if generate is not None and is_vetch_patched(generate):
            return True
    return False


def set_log_level(level: str | int) -> None:
    """Set Vetch's internal logging verbosity.

    Args:
        level: Logging level (e.g., "DEBUG", "INFO", "WARNING", "ERROR", or int).

    Example::

        import vetch
        vetch.set_log_level("ERROR")  # Silence info/warning messages
    """
    import logging

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.WARNING)
    logging.getLogger("vetch").setLevel(level)


def wrap(
    region: str | None = None,
    tags: dict[str, str] | None = None,
    energy_override: dict[str, object] | None = None,
    price_multiplier: float = 1.0,
    emit: bool = True,
    provider_hint: str | None = None,
) -> VetchContext:
    """Context manager for tracking LLM inference energy and carbon.

    Args:
        region: Grid region for carbon calculation. Falls back to VETCH_REGION
            env var, base_url parsing, or 'unknown'.
        tags: Key-value pairs for cost attribution. Avoid high-cardinality
            values (user IDs, request IDs).
        energy_override: User-provided energy values with keys:
            - wh_per_1k_input (float, required)
            - wh_per_1k_output (float, required)
            - tier (int 1-3, optional)
            - source (str, optional)
        price_multiplier: Factor to adjust list pricing (e.g. 0.8 for 20% discount).
        emit: If True (default), emit JSON to configured output. Set False for
            quiet mode (metrics still available in ctx.event).
        provider_hint: Explicit provider override (e.g. "self-hosted",
            "openai-compatible"). Overrides the provider inferred from the model
            name / SDK client, so a self-hosted model yields cost 0 with energy
            and carbon still computed.

    Returns:
        VetchContext: Context manager that logs inference events.

    Example::

        with wrap(region="eu-west-1", tags={"team": "ml"}) as ctx:
            response = client.chat.completions.create(...)

        # Access the logged event
        print(ctx.event["estimated_energy_wh"])

    Note:
        Set VETCH_DISABLED=true to completely disable Vetch (emergency kill switch).

    """
    from vetch.wrapper import VetchContext

    # P0: Kill switch - return no-op context when disabled
    if _DISABLED:
        return VetchContext(
            region=region, tags=tags, energy_override=energy_override,
            price_multiplier=price_multiplier, emit=emit,
            provider_hint=provider_hint, _disabled=True,
        )

    return VetchContext(
        region=region, tags=tags, energy_override=energy_override,
        price_multiplier=price_multiplier, emit=emit, provider_hint=provider_hint,
    )


def awrap(
    region: str | None = None,
    tags: dict[str, str] | None = None,
    energy_override: dict[str, object] | None = None,
    price_multiplier: float = 1.0,
    emit: bool = True,
    provider_hint: str | None = None,
) -> AsyncContextManager[VetchContext]:
    """Async context manager for tracking LLM inference energy and carbon.

    First-class async support for async/await patterns. Use this instead
    of wrap() when working with async code.

    Args:
        region: Grid region for carbon calculation.
        tags: Key-value pairs for cost attribution.
        energy_override: User-provided energy values.
        price_multiplier: Factor to adjust list pricing (e.g. 0.8 for 20% discount).
        emit: If True (default), emit JSON to configured output.
        provider_hint: Explicit provider override (e.g. "self-hosted"). Overrides
            the inferred provider that drives cost/PUE/water.

    Returns:
        Async context manager yielding VetchContext.

    Example::

        async with awrap(region="us-east-1", tags={"team": "ml"}) as ctx:
            response = await client.chat.completions.create(...)
        print(ctx.event["estimated_energy_wh"])

    Note:
        Set VETCH_DISABLED=true to completely disable Vetch (emergency kill switch).
    """
    from vetch.wrapper import awrap as _awrap

    # P0: Kill switch - pass _disabled flag directly (consistent with wrap())
    return _awrap(
        region=region,
        tags=tags,
        energy_override=energy_override,
        price_multiplier=price_multiplier,
        emit=emit,
        provider_hint=provider_hint,
        _disabled=_DISABLED,
    )


# Lazy imports to avoid circular dependencies
def __getattr__(name: str) -> object:
    if name == "VetchContext":
        from vetch.wrapper import VetchContext

        return VetchContext
    if name == "record_usage":
        from vetch.wrapper import record_usage

        return record_usage
    if name == "Session":
        from vetch.session import Session

        return Session
    if name == "get_session_stats":
        from vetch.stats import get_session_stats

        return get_session_stats
    if name == "generate_advisories":
        from vetch.advisory import generate_advisories

        return generate_advisories
    if name == "configure_storage":
        from vetch.storage import configure_storage

        return configure_storage
    if name == "query_usage":
        from vetch.storage import query_usage

        return query_usage
    # v0.1.5: Budget alerts
    if name == "set_budget":
        from vetch.budget import set_budget

        return set_budget
    if name == "on_budget_alert":
        from vetch.budget import on_budget_alert

        return on_budget_alert
    if name == "get_budget_status":
        from vetch.budget import get_budget_status

        return get_budget_status
    # v0.1.5: OTLP export
    if name == "configure_otlp_export":
        from vetch.otel import configure_otlp_export

        return configure_otlp_export
    if name == "export_advisory_otlp":
        from vetch.otel import export_advisory_otlp

        return export_advisory_otlp
    # v0.2.3: HTTP endpoint output
    if name == "configure_http_endpoint":
        from vetch.emitter import configure_http_endpoint

        return configure_http_endpoint
    # v0.1.5: Green signal API
    if name == "get_cleanest_region":
        from vetch.sensing.grid import get_cleanest_region

        return get_cleanest_region
    # v0.2.0: OpenTelemetry exporter
    if name == "configure_otel_export":
        from vetch.exporters.opentelemetry import configure_auto_export

        return configure_auto_export
    if name == "export_event_as_span":
        from vetch.exporters.opentelemetry import export_event_as_span

        return export_event_as_span
    # v0.4.0: Stall circuit breaker
    if name == "set_stall_action":
        from vetch.config import set_stall_action

        return set_stall_action
    if name == "get_stall_action":
        from vetch.config import get_stall_action

        return get_stall_action
    if name == "StallDetected":
        from vetch.exceptions import StallDetected

        return StallDetected
    if name == "VetchInterrupt":
        from vetch.exceptions import VetchInterrupt

        return VetchInterrupt
    if name == "set_advisory_thresholds":
        from vetch.config import set_advisory_thresholds

        return set_advisory_thresholds
    if name == "configure_capabilities":
        from vetch.capabilities import configure_capabilities

        return configure_capabilities
    if name == "set_expected_capabilities":
        from vetch.capabilities import set_expected_capabilities

        return set_expected_capabilities
    if name == "set_model_capability_map":
        from vetch.capabilities import set_model_capability_map

        return set_model_capability_map
    if name == "on_advisory":
        from vetch.stats import on_advisory

        return on_advisory
    if name == "rollup_capability_summary_from_events":
        from vetch.capabilities import rollup_capability_summary_from_events

        return rollup_capability_summary_from_events
    if name == "set_min_match_confidence":
        from vetch.config import set_min_match_confidence

        return set_min_match_confidence
    if name == "get_min_match_confidence":
        from vetch.config import get_min_match_confidence

        return get_min_match_confidence
    if name == "rollup_confidence_from_events":
        from vetch.stats import rollup_confidence_from_events

        return rollup_confidence_from_events
    if name == "filter_events_by_confidence":
        from vetch.stats import filter_events_by_confidence

        return filter_events_by_confidence
    if name == "require_confidence":
        from vetch.stats import require_confidence

        return require_confidence
    if name == "ConfidenceError":
        from vetch.exceptions import ConfidenceError

        return ConfidenceError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
