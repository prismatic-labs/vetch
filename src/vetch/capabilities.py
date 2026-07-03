"""Capability observability: tool extraction, redaction, and Kind C derivation."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from vetch.schema import CapabilityRef

logger = logging.getLogger(__name__)

_CAPABILITY_LIST_CAP = 64
_redacted_capability_names: set[str] = set()
_redaction_lock = threading.Lock()
_generated_redaction_key: bytes | None = None
_redaction_key_warning_shown = False

_offered_memo_lock = threading.Lock()
_offered_memo: OrderedDict[str, tuple[list[CapabilityRef] | None, dict[str, int] | None]] = (
    OrderedDict()
)
_OFFERED_MEMO_MAX = 256

REGISTRY_DIR = Path(__file__).resolve().parent / "registry"
_MODEL_CAPABILITIES_FILE = "model_capabilities.json"

_expected_capabilities: list[str] = []
_model_capability_overrides: dict[str, str] = {}


def set_redacted_capability_names(names: Iterable[str]) -> None:
    """Exact tool/capability names to hash before emission."""
    global _redacted_capability_names
    with _redaction_lock:
        _redacted_capability_names = set(names)


def get_redacted_capability_names() -> set[str]:
    with _redaction_lock:
        return set(_redacted_capability_names)


def _redaction_key() -> bytes | None:
    """Return HMAC key when redaction is active, else None."""
    global _generated_redaction_key, _redaction_key_warning_shown
    key_str = os.environ.get("VETCH_REDACTION_KEY", "")
    if key_str:
        return key_str.encode("utf-8")
    with _redaction_lock:
        if _redacted_capability_names:
            if _generated_redaction_key is None:
                _generated_redaction_key = secrets.token_bytes(32)
                if not _redaction_key_warning_shown:
                    _redaction_key_warning_shown = True
                    logger.warning(
                        "VETCH_REDACTION_KEY not set. Using ephemeral key for "
                        "capability name redaction."
                    )
            return _generated_redaction_key
    return None


def redact_capability_name(name: str) -> str:
    """Return ``name`` or ``redacted-<hmac>`` when redaction applies."""
    key_str = os.environ.get("VETCH_REDACTION_KEY", "")
    with _redaction_lock:
        should_redact = bool(key_str) or name in _redacted_capability_names
    if not should_redact:
        return name

    key_bytes: bytes | None
    if key_str:
        key_bytes = key_str.encode("utf-8")
    else:
        key_bytes = _redaction_key()
    if key_bytes is None:
        return name

    digest = hmac.new(key_bytes, name.encode("utf-8"), hashlib.sha256).hexdigest()[:32]
    return f"redacted-{digest}"


def redact_capability_refs(refs: list[CapabilityRef] | None) -> list[CapabilityRef] | None:
    if not refs:
        return refs
    return [
        {"name": redact_capability_name(ref["name"]), "kind": ref["kind"]}
        for ref in refs
    ]


def redact_tool_schema_tokens(tokens: dict[str, int] | None) -> dict[str, int] | None:
    if not tokens:
        return tokens
    return {redact_capability_name(name): value for name, value in tokens.items()}


def sanitize_capability_capture_fields(
    *,
    tools_offered: list[CapabilityRef] | None = None,
    tools_invoked: list[CapabilityRef] | None = None,
    tool_schema_tokens: dict[str, int] | None = None,
) -> tuple[
    list[CapabilityRef] | None,
    list[CapabilityRef] | None,
    dict[str, int] | None,
]:
    """Apply redaction to capability fields (manual capture + emit safety net)."""
    return (
        redact_capability_refs(tools_offered),
        redact_capability_refs(tools_invoked),
        redact_tool_schema_tokens(tool_schema_tokens),
    )


def _offered_memo_get(key: str) -> tuple[list[CapabilityRef] | None, dict[str, int] | None] | None:
    with _offered_memo_lock:
        if key not in _offered_memo:
            return None
        _offered_memo.move_to_end(key)
        return _offered_memo[key]


def _offered_memo_put(
    key: str,
    value: tuple[list[CapabilityRef] | None, dict[str, int] | None],
) -> None:
    with _offered_memo_lock:
        if key in _offered_memo:
            _offered_memo.move_to_end(key)
        _offered_memo[key] = value
        while len(_offered_memo) > _OFFERED_MEMO_MAX:
            _offered_memo.popitem(last=False)


def normalize_function_tools(names: Iterable[str]) -> list[CapabilityRef]:
    """De-dupe, stable-sort, and wrap names as function capability refs."""
    unique = sorted({redact_capability_name(n) for n in names if n})
    return [{"name": n, "kind": "function"} for n in unique]


def _memo_key_for_tools(tools: Any) -> str:
    try:
        return f"id:{id(tools)}"
    except Exception:
        return "none"


def _serialize_tools_for_hash(tools: Any) -> str:
    try:
        return json.dumps(tools, sort_keys=True, default=str)
    except Exception:
        return repr(tools)


def _estimate_tool_json_tokens(tool_obj: Any, *, provider: str = "openai") -> int:
    try:
        payload = json.dumps(tool_obj, default=str)
        provider_key = provider.lower()
        if provider_key in ("anthropic", "google_genai", "vertexai", "ollama"):
            return max(1, len(payload) // 4)
        from vetch.calculation import _get_tiktoken_encoding

        enc = _get_tiktoken_encoding("gpt-4o")
        if enc is not None:
            return len(enc.encode(payload))
        return max(1, len(payload) // 4)
    except Exception:
        pass
    try:
        payload = json.dumps(tool_obj, default=str)
        return max(1, len(payload) // 4)
    except Exception:
        return 1


def extract_openai_tools_offered(
    kwargs: Mapping[str, Any],
    *,
    provider: str = "openai",
) -> tuple[list[CapabilityRef] | None, dict[str, int] | None]:
    """Extract offered function tools and per-tool schema token sizes."""
    tools = kwargs.get("tools")
    if tools is None:
        return None, None
    if not isinstance(tools, list):
        return None, None

    memo_key = _memo_key_for_tools(tools)
    cached = _offered_memo_get(memo_key)
    if cached is not None:
        return cached

    try:
        names: list[str] = []
        token_sizes: dict[str, int] = {}
        for entry in tools:
            name: str | None = None
            if isinstance(entry, dict):
                fn = entry.get("function")
                if isinstance(fn, dict):
                    raw = fn.get("name")
                    if isinstance(raw, str):
                        name = raw
                elif entry.get("type") == "function":
                    inner = entry.get("function")
                    if isinstance(inner, dict):
                        raw = inner.get("name")
                        if isinstance(raw, str):
                            name = raw
            else:
                fn = getattr(entry, "function", None)
                if fn is not None:
                    raw = getattr(fn, "name", None)
                    if isinstance(raw, str):
                        name = raw
            if name:
                redacted = redact_capability_name(name)
                names.append(redacted)
                token_sizes[redacted] = _estimate_tool_json_tokens(entry, provider=provider)

        refs = normalize_function_tools(names) if names else []
        result: tuple[list[CapabilityRef] | None, dict[str, int] | None] = (
            refs if refs else [],
            token_sizes if token_sizes else {},
        )
        _offered_memo_put(memo_key, result)
        payload_key = hashlib.sha256(_serialize_tools_for_hash(tools).encode()).hexdigest()[:16]
        _offered_memo_put(payload_key, result)
        return result
    except Exception:
        logger.debug("extract_openai_tools_offered failed", exc_info=True)
        return None, None


def extract_openai_tools_invoked(
    result: Any,
) -> tuple[list[CapabilityRef] | None, int | None]:
    try:
        names: list[str] = []
        choices = getattr(result, "choices", None)
        if not choices:
            return None, None
        message = getattr(choices[0], "message", None)
        if message is None:
            return None, None
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            return [], 0
        for tc in tool_calls:
            fn = getattr(tc, "function", None)
            if fn is not None:
                raw = getattr(fn, "name", None)
                if isinstance(raw, str) and raw:
                    names.append(raw)
        refs = normalize_function_tools(names)
        # tool_call_count is raw invocations (parallel calls count individually);
        # refs are de-duped names for set-difference math.
        return refs, len(names)
    except Exception:
        logger.debug("extract_openai_tools_invoked failed", exc_info=True)
        return None, None


def extract_anthropic_tools_offered(
    kwargs: Mapping[str, Any],
) -> tuple[list[CapabilityRef] | None, dict[str, int] | None]:
    tools = kwargs.get("tools")
    if tools is None:
        return None, None
    if not isinstance(tools, list):
        return None, None

    memo_key = _memo_key_for_tools(tools)
    cached = _offered_memo_get(memo_key)
    if cached is not None:
        return cached

    try:
        names: list[str] = []
        token_sizes: dict[str, int] = {}
        for entry in tools:
            name: str | None = None
            if isinstance(entry, dict):
                raw = entry.get("name")
                if isinstance(raw, str):
                    name = raw
            else:
                raw = getattr(entry, "name", None)
                if isinstance(raw, str):
                    name = raw
            if name:
                redacted = redact_capability_name(name)
                names.append(redacted)
                token_sizes[redacted] = _estimate_tool_json_tokens(entry, provider="anthropic")

        refs = normalize_function_tools(names) if names else []
        result = (refs if refs else [], token_sizes if token_sizes else {})
        _offered_memo_put(memo_key, result)
        return result
    except Exception:
        logger.debug("extract_anthropic_tools_offered failed", exc_info=True)
        return None, None


def extract_anthropic_tools_invoked(
    result: Any,
) -> tuple[list[CapabilityRef] | None, int | None]:
    try:
        names: list[str] = []
        content = getattr(result, "content", None)
        if not content:
            return [], 0
        for block in content:
            block_type = getattr(block, "type", None)
            if block_type == "tool_use":
                raw = getattr(block, "name", None)
                if isinstance(raw, str) and raw:
                    names.append(raw)
        refs = normalize_function_tools(names)
        # tool_call_count is raw invocations (parallel calls count individually);
        # refs are de-duped names for set-difference math.
        return refs, len(names)
    except Exception:
        logger.debug("extract_anthropic_tools_invoked failed", exc_info=True)
        return None, None


def extract_genai_tools_offered(
    kwargs: Mapping[str, Any],
) -> tuple[list[CapabilityRef] | None, dict[str, int] | None]:
    config = kwargs.get("config")
    if config is None:
        return None, None
    try:
        tools = getattr(config, "tools", None)
        if tools is None and isinstance(config, dict):
            tools = config.get("tools")
        if not tools:
            return None, None
        names: list[str] = []
        token_sizes: dict[str, int] = {}
        for tool_group in tools if isinstance(tools, list) else [tools]:
            decls = getattr(tool_group, "function_declarations", None)
            if decls is None and isinstance(tool_group, dict):
                decls = tool_group.get("function_declarations")
            if not decls:
                continue
            for decl in decls:
                raw = getattr(decl, "name", None)
                if raw is None and isinstance(decl, dict):
                    raw = decl.get("name")
                if isinstance(raw, str) and raw:
                    redacted = redact_capability_name(raw)
                    names.append(redacted)
                    token_sizes[redacted] = _estimate_tool_json_tokens(
                        decl, provider="google_genai"
                    )
        refs = normalize_function_tools(names) if names else []
        return (refs if refs else [], token_sizes if token_sizes else {})
    except Exception:
        logger.debug("extract_genai_tools_offered failed", exc_info=True)
        return None, None


def extract_genai_tools_invoked(
    result: Any,
) -> tuple[list[CapabilityRef] | None, int | None]:
    try:
        names: list[str] = []
        candidates = getattr(result, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            parts = getattr(content, "parts", None) or []
            for part in parts:
                fc = getattr(part, "function_call", None)
                if fc is None and isinstance(part, dict):
                    fc = part.get("function_call")
                if fc is None:
                    continue
                raw = getattr(fc, "name", None)
                if raw is None and isinstance(fc, dict):
                    raw = fc.get("name")
                if isinstance(raw, str) and raw:
                    names.append(raw)
        refs = normalize_function_tools(names)
        # tool_call_count is raw invocations (parallel calls count individually);
        # refs are de-duped names for set-difference math.
        return refs, len(names)
    except Exception:
        logger.debug("extract_genai_tools_invoked failed", exc_info=True)
        return None, None


def extract_openai_compat_tools_offered(
    kwargs: Mapping[str, Any],
    *,
    provider: str = "openai",
) -> tuple[list[CapabilityRef] | None, dict[str, int] | None]:
    """Ollama and other OpenAI-compatible chat APIs."""
    return extract_openai_tools_offered(kwargs, provider=provider)


def extract_openai_compat_tools_invoked(
    result: Any,
) -> tuple[list[CapabilityRef] | None, int | None]:
    try:
        message = getattr(result, "message", None)
        if message is not None:
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                names: list[str] = []
                for tc in tool_calls:
                    fn = getattr(tc, "function", None)
                    if isinstance(fn, dict):
                        raw = fn.get("name")
                    else:
                        raw = getattr(fn, "name", None) if fn else None
                    if isinstance(raw, str) and raw:
                        names.append(raw)
                refs = normalize_function_tools(names)
                return refs, len(names)
    except Exception:
        pass
    return extract_openai_tools_invoked(result)


def stage_request_tools(provider: str, kwargs: Mapping[str, Any]) -> None:
    """Extract offered tools before stall reroute mutates kwargs."""
    from vetch.context import get_active_context

    ctx = get_active_context()
    if ctx is None:
        return

    offered: tuple[list[CapabilityRef] | None, dict[str, int] | None]
    if provider in ("openai", "azure_openai", "openai-compatible", "ollama"):
        offered = extract_openai_compat_tools_offered(kwargs, provider=provider)
    elif provider == "anthropic":
        offered = extract_anthropic_tools_offered(kwargs)
    elif provider in ("google_genai", "vertexai"):
        offered = extract_genai_tools_offered(kwargs)
    else:
        return

    refs, sizes = offered
    ctx.pending_tools_offered = refs
    ctx.pending_tool_schema_tokens = sizes


def merge_capability_capture(
    *,
    tools_offered: list[CapabilityRef] | None = None,
    tools_invoked: list[CapabilityRef] | None = None,
    tool_call_count: int | None = None,
    tool_schema_tokens: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build capture kwargs from pending context + response extraction."""
    from vetch.context import get_active_context

    ctx = get_active_context()
    out: dict[str, Any] = {}
    if ctx is not None:
        if tools_offered is None and ctx.pending_tools_offered is not None:
            tools_offered = ctx.pending_tools_offered
        if tool_schema_tokens is None and ctx.pending_tool_schema_tokens is not None:
            tool_schema_tokens = ctx.pending_tool_schema_tokens
        ctx.pending_tools_offered = None
        ctx.pending_tool_schema_tokens = None

    if tools_offered is not None:
        out["tools_offered"] = tools_offered
    if tools_invoked is not None:
        out["tools_invoked"] = tools_invoked
    if tool_call_count is not None:
        out["tool_call_count"] = tool_call_count
    if tool_schema_tokens:
        out["tool_schema_tokens"] = tool_schema_tokens
    return out


def accumulate_openai_stream_tool_call(
    accumulator: dict[int, dict[str, str]],
    chunk: Any,
) -> None:
    choices = getattr(chunk, "choices", None)
    if not choices:
        return
    delta = getattr(choices[0], "delta", None)
    if delta is None:
        return
    tool_calls = getattr(delta, "tool_calls", None)
    if not isinstance(tool_calls, list):
        return
    for tc in tool_calls:
        idx = int(getattr(tc, "index", 0) or 0)
        entry = accumulator.setdefault(idx, {"name": "", "id": ""})
        fn = getattr(tc, "function", None)
        if fn is not None:
            name = getattr(fn, "name", None)
            if isinstance(name, str) and name:
                entry["name"] = name


def finalize_openai_stream_tools(
    accumulator: dict[int, dict[str, str]],
    *,
    complete: bool,
    error: bool,
) -> tuple[list[CapabilityRef] | None, int | None]:
    if error or not complete:
        return None, None
    names = [entry["name"] for entry in accumulator.values() if entry.get("name")]
    if not names:
        return [], 0
    refs = normalize_function_tools(names)
    return refs, len(names)


def accumulate_anthropic_stream_tool_use(
    accumulator: list[str],
    chunk: Any,
) -> None:
    event_type = getattr(chunk, "type", "")
    if event_type != "content_block_start":
        return
    block = getattr(chunk, "content_block", None)
    if block is None:
        return
    if getattr(block, "type", None) == "tool_use":
        name = getattr(block, "name", None)
        if isinstance(name, str) and name:
            accumulator.append(name)


def finalize_anthropic_stream_tools(
    accumulator: list[str],
    *,
    complete: bool,
    error: bool,
) -> tuple[list[CapabilityRef] | None, int | None]:
    if error or not complete:
        return None, None
    if not accumulator:
        return [], 0
    refs = normalize_function_tools(accumulator)
    return refs, len(accumulator)


def load_model_capability_map() -> dict[str, str]:
    path = REGISTRY_DIR / _MODEL_CAPABILITIES_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {k: str(v) for k, v in data.items() if not str(k).startswith("_")}
    except Exception:
        logger.warning("Failed to load %s", path, exc_info=True)
        return {}


def get_merged_model_capability_map() -> dict[str, str]:
    merged = load_model_capability_map()
    merged.update(_model_capability_overrides)
    return merged


def resolve_model_capability(model: str) -> str | None:
    """Longest-prefix match against the merged model capability map."""
    if not model:
        return None
    cap_map = get_merged_model_capability_map()
    if model in cap_map:
        return cap_map[model]
    model_lower = model.lower()
    best_prefix = ""
    best_cap: str | None = None
    for prefix, capability in cap_map.items():
        prefix_lower = prefix.lower()
        if model_lower.startswith(prefix_lower) and len(prefix_lower) > len(best_prefix):
            best_prefix = prefix_lower
            best_cap = capability
    return best_cap


def set_expected_capabilities(capabilities: list[str]) -> None:
    global _expected_capabilities
    _expected_capabilities = list(capabilities)


def get_expected_capabilities() -> list[str]:
    from vetch.session import get_active_session

    session = get_active_session()
    if session is not None and session.expected_capabilities:
        return list(session.expected_capabilities)
    return list(_expected_capabilities)


def set_model_capability_map(mapping: dict[str, str]) -> None:
    global _model_capability_overrides
    _model_capability_overrides = dict(mapping)


def configure_capabilities(
    *,
    expected: list[str] | None = None,
    model_capability_map: dict[str, str] | None = None,
) -> None:
    if expected is not None:
        set_expected_capabilities(expected)
    if model_capability_map is not None:
        set_model_capability_map(model_capability_map)


def derive_capabilities_invoked(
    *,
    is_embedding: bool,
    usage: Mapping[str, Any] | None,
    model: str,
) -> list[CapabilityRef] | None:
    refs: list[CapabilityRef] = []
    if is_embedding:
        refs.append({"name": "embedding", "kind": "model"})

    if usage and isinstance(usage, dict):
        for modality, name in (
            ("image", "image"),
            ("audio", "audio"),
            ("video", "video"),
        ):
            block = usage.get(modality)
            if not isinstance(block, dict):
                continue
            tokens = int(block.get("input_tokens") or block.get("output_tokens") or 0)
            if tokens > 0:
                refs.append({"name": name, "kind": "model"})

    cap = resolve_model_capability(model)
    if cap:
        refs.append({"name": cap, "kind": "model"})

    if not refs:
        return None
    seen: set[tuple[str, str]] = set()
    unique: list[CapabilityRef] = []
    for ref in refs:
        key = (ref["kind"], ref["name"])
        if key not in seen:
            seen.add(key)
            unique.append(ref)
    return sorted(unique, key=lambda r: (r["kind"], r["name"]))


def truncate_capability_lists_for_transport(
    event: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Apply per-event 64-cap on serialized copy only (transport/OTel)."""
    out = dict(event)
    for field in ("tools_offered", "tools_invoked", "capabilities_invoked"):
        value = out.get(field)
        if not isinstance(value, list) or len(value) <= _CAPABILITY_LIST_CAP:
            continue
        sorted_list = sorted(
            value,
            key=lambda item: (
                item.get("kind", ""),
                item.get("name", ""),
            )
            if isinstance(item, dict)
            else str(item),
        )
        out[field] = sorted_list[:_CAPABILITY_LIST_CAP]
        warnings.append(f"{field}_truncated: offered={len(value)}, recorded={_CAPABILITY_LIST_CAP}")
    if warnings:
        existing = out.get("vetch_warnings") or []
        if isinstance(existing, list):
            out["vetch_warnings"] = [*existing, *warnings]
        else:
            out["vetch_warnings"] = warnings
    return out


def set_otel_capability_attributes(span: Any, event: Mapping[str, Any]) -> None:
    """Attach semconv-aligned tool/capability attributes (transport cap applied)."""
    warnings: list[str] = []
    transport = truncate_capability_lists_for_transport(dict(event), warnings)

    offered = transport.get("tools_offered")
    if isinstance(offered, list):
        names = [
            ref["name"] for ref in offered if isinstance(ref, dict) and ref.get("name")
        ]
        if names:
            span.set_attribute("gen_ai.tool.definitions", names)

    invoked = transport.get("tools_invoked")
    if isinstance(invoked, list):
        names = [
            ref["name"] for ref in invoked if isinstance(ref, dict) and ref.get("name")
        ]
        if names:
            span.set_attribute("gen_ai.tool.calls", names)

    caps = transport.get("capabilities_invoked")
    if isinstance(caps, list):
        cap_names = [
            f"{ref.get('kind')}:{ref.get('name')}"
            for ref in caps
            if isinstance(ref, dict)
        ]
        if cap_names:
            span.set_attribute("vetch.capabilities_invoked", cap_names)

    count = transport.get("tool_call_count")
    if isinstance(count, int):
        span.set_attribute("gen_ai.tool.call.count", count)

    never_called = [
        ref["name"]
        for ref in (offered or [])
        if isinstance(ref, dict)
        and ref.get("name")
        and ref["name"]
        not in {
            r.get("name")
            for r in (invoked or [])
            if isinstance(r, dict)
        }
    ]
    if never_called:
        span.set_attribute("vetch.tools_never_called", never_called)

    schema = transport.get("tool_schema_tokens")
    if isinstance(schema, dict) and never_called:
        wasted_tokens = sum(
            int(schema.get(name, 0))
            for name in never_called
            if isinstance(name, str)
        )
        if wasted_tokens > 0:
            span.set_attribute("vetch.wasted_tool_schema_tokens", wasted_tokens)


def capability_fully_cached_warning(event: Mapping[str, Any]) -> str | None:
    """Return a vetch_warnings note when dead tools ride on a fully-cached request."""
    offered = event.get("tools_offered")
    if not isinstance(offered, list) or not offered:
        return None
    invoked = event.get("tools_invoked") or []
    invoked_names = {
        ref.get("name")
        for ref in invoked
        if isinstance(ref, dict) and isinstance(ref.get("name"), str) and ref["name"]
    }
    offered_names = {
        ref.get("name")
        for ref in offered
        if isinstance(ref, dict) and isinstance(ref.get("name"), str) and ref["name"]
    }
    if not offered_names - invoked_names:
        return None
    usage = event.get("usage") or {}
    text = usage.get("text") if isinstance(usage, dict) else {}
    in_tok = text.get("input_tokens") if isinstance(text, dict) else None
    cache_read = event.get("cache_read_tokens")
    if not isinstance(in_tok, int) or in_tok <= 0:
        return None
    if not isinstance(cache_read, int):
        return None
    if max(0, in_tok - cache_read) > 0:
        return None
    return (
        "fully_cached_session: dead tool schemas on this request; "
        "billable_input_tokens=0 so per-request schema cost is ~$0"
    )


def rollup_capability_summary_from_events(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute capability rollup fields from stored inference events."""
    from vetch.stats import SessionStats

    stats = SessionStats()
    for event in events:
        stats.update(dict(event))
    summary = stats.summary()
    keys = (
        "function_tools_never_called",
        "wasted_tool_schema_tokens_per_request",
        "wasted_tool_schema_tokens",
        "wasted_tool_schema_session_tokens",
        "wasted_tool_schema_cost_per_request_usd",
        "wasted_tool_schema_session_cost_usd",
        "wasted_tool_schema_cost_usd",
        "dead_tool_offer_request_count",
        "declared_capabilities_silent",
        "capability_invocation_counts",
        "tool_call_event_rate",
        "wasted_tool_schema_cost_note",
        "capability_cardinality_bounded",
    )
    return {key: summary[key] for key in keys if key in summary}


def reset_capability_state() -> None:
    """Test-only reset."""
    global _expected_capabilities, _model_capability_overrides
    with _offered_memo_lock:
        _offered_memo.clear()
    _expected_capabilities = []
    _model_capability_overrides = {}
    set_redacted_capability_names([])
