"""Inference event schema definitions.

This module defines the TypedDict for inference events logged by Vetch.
Schema version 2 adds multimodal support (images, audio, video).

Schema version 1 guarantees:
- Fields not removed
- Field names not changed
- Field types not changed
- New fields may be added

Schema version 2 changes:
- Added ImageUsage, AudioUsage, and VideoUsage TypedDicts
- Added multimodal flag to InferenceEvent
- Updated Usage container to support image, audio, and video modalities
"""

from typing import Literal, TypedDict, Union

SCHEMA_VERSION = "2"


class TextUsage(TypedDict):
    """Token usage for text modality."""

    input_tokens: int
    output_tokens: int
    total_tokens: int


class ImageUsage(TypedDict, total=False):
    """Token usage for image modality.

    Images are tokenized by LLM providers for processing.
    Token counts vary by resolution and model.
    """

    input_tokens: int  # Tokens from input images
    output_tokens: int  # Tokens from generated images (e.g., DALL-E)
    total_tokens: int
    image_count: int  # Number of images processed
    total_pixels: int  # Total pixels across all images (for energy estimation)


class AudioUsage(TypedDict, total=False):
    """Token usage for audio modality.

    Audio is tokenized or measured in seconds depending on provider.
    """

    input_tokens: int  # Tokens from input audio (e.g., Whisper)
    output_tokens: int  # Tokens from generated audio (e.g., TTS)
    total_tokens: int
    input_seconds: float  # Duration of input audio in seconds
    output_seconds: float  # Duration of output audio in seconds


class VideoUsage(TypedDict, total=False):
    """Token usage for video modality.

    Providers usually report video as token-equivalent counts, sometimes with
    a duration estimate. Energy modelling for video remains high-uncertainty.
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_seconds: float
    output_seconds: float


class Usage(TypedDict, total=False):
    """Multi-modal usage container.

    The nested structure supports future extension without breaking schema.
    Schema v2 adds image, audio, and video support.
    Schema v2.1 adds reasoning (extended thinking) support.
    """

    text: Union[TextUsage, None]
    image: Union[ImageUsage, None]
    audio: Union[AudioUsage, None]
    video: Union[VideoUsage, None]
    reasoning: Union[TextUsage, None]  # Extended thinking tokens (Gemini Flash Thinking, etc.)


class InferenceEvent(TypedDict, total=False):
    """Complete inference event schema.

    All fields are optional (total=False) to support partial events
    from interrupted streams. Required fields in practice:
    - schema_version
    - vetch_version
    - event_id
    - timestamp
    - model
    - provider
    - signal_quality
    """

    # Metadata
    schema_version: str
    vetch_version: str
    event_id: str
    timestamp: str  # ISO8601 UTC with Z suffix

    # Model info
    model: str
    provider: str
    model_known: bool
    multimodal: bool  # True if request includes non-text modalities (image/audio/video)

    # Usage
    usage: Union[Usage, None]
    accumulated_chars: Union[int, None]  # For partial streams

    # Estimates
    estimated_energy_wh: Union[float, None]
    estimated_carbon_g: Union[float, None]
    estimated_water_l: Union[float, None]  # Water usage in liters
    estimated_cost_usd: Union[float, None]
    estimated_cost_input_usd: Union[float, None]
    estimated_cost_output_usd: Union[float, None]
    estimated_cost_cache_write_usd: Union[float, None]  # Cost to write tokens to cache
    # Actual discounted cost paid for cache-read tokens, not the saving.
    # See cache_cost_saving_usd for the difference vs. uncached input price.
    estimated_cost_cache_read_usd: Union[float, None]
    billing_tier: str  # Always "list" for v1

    # Signal quality
    signal_quality: Literal["live", "delayed", "blind", "unknown"]
    energy_tier: int  # 0=measured, 1=vendor, 2=validated, 3=estimated
    energy_uncertainty_pct: Union[int, None]  # Uncertainty %: 20/50/100/1000 for tier 0/1/2/3
    # v0.4.0: Explicit confidence bounds. Both are derived from
    # ``estimated_energy_wh`` and ``energy_uncertainty_pct`` — they don't
    # introduce new modelling, just expose the existing uncertainty as
    # absolute Wh / gCO2 numbers so downstream tooling doesn't have to
    # repeat the math. ``p5_*`` is the lower bound, ``p95_*`` the upper.
    energy_p5_wh: Union[float, None]
    energy_p95_wh: Union[float, None]
    carbon_p5_g: Union[float, None]
    carbon_p95_g: Union[float, None]
    energy_source: str  # "registry", "override", "fallback", "calibrated"
    energy_override_source: Union[str, None]
    energy_basis: Union[str, None]

    # Grid data
    grid_intensity_gco2e_kwh: Union[float, None]
    grid_intensity_timestamp: Union[str, None]  # ISO8601 UTC
    grid_intensity_time_of_day: bool  # True if using hourly grid data
    region: Union[str, None]
    embodied_carbon_g: Union[float, None]  # Hardware manufacturing emissions

    # PUE (Power Usage Effectiveness) metadata
    pue: Union[float, None]  # Datacenter efficiency (e.g., 1.10 for Google, 1.15 for AWS)
    pue_tier: Union[int, None]  # 1=known value (user config or vendor), 3=default fallback
    pue_source: Union[str, None]  # Source description (e.g., "Google Environmental Report 2024")

    # Request metadata
    is_stream: bool
    is_batch: bool  # True if using batch API (50% discount)
    is_embedding: bool  # True if embedding generation (different energy profile)
    complete: bool
    latency_ms: Union[float, None]
    visible_output_chars: Union[int, None]  # Output character count; no text stored
    finish_reason: Union[str, None]  # Provider finish status, e.g. stop/length
    requested_max_tokens: Union[int, None]  # Caller-requested output cap if available

    # Attribution
    tags: Union[dict[str, str], None]

    # Error handling
    error: bool
    error_type: Union[str, None]

    # Tracking status
    tracking_disabled: bool
    tracking_degraded: bool  # True if tracking is active but with reduced accuracy

    # Diagnostic warnings (fail-loud transparency)
    vetch_warnings: Union[list[str], None]

    # Budget tracking (for threshold hooks)
    budget_energy_wh: Union[float, None]
    budget_carbon_g: Union[float, None]
    budget_cost_usd: Union[float, None]
    budget_exceeded: Union[bool, None]

    # Token estimation metadata
    usage_estimated: bool  # True if tokens were heuristically estimated
    usage_estimation_method: Union[str, None]  # "char_ratio", "model_default", etc.

    # Prompt caching metadata (Anthropic, OpenAI)
    cache_read_tokens: Union[int, None]  # Tokens read from cache (cost savings)
    cache_creation_tokens: Union[int, None]  # Tokens written to cache (extra cost)
    cache_hit: Union[bool, None]  # True if any cache was used
    cache_energy_saving_wh: Union[float, None]  # Energy saved vs. uncached baseline (Wh)
    # Realized cost saving vs. uncached baseline: full input price minus discounted
    # cache-read price. Distinct from estimated_cost_cache_read_usd, the cost paid.
    cache_cost_saving_usd: Union[float, None]
    # Carbon saved vs. uncached baseline, using the same grid/PUE path as carbon_g.
    cache_carbon_saving_g: Union[float, None]

    # Session tracking (v0.1.6)
    session_id: Union[str, None]  # ID of parent session if event is part of a session

    # Distributed tracing (v0.1.8)
    trace_id: Union[str, None]  # W3C trace ID for correlation with APM tools
    span_id: Union[str, None]  # W3C span ID for correlation with APM tools
    parent_span_id: Union[str, None]  # Parent span ID for nested operations

    # Deduplication (v0.1.8)
    request_fingerprint: Union[str, None]  # SHA256 hash for duplicate detection (16-char)


class EnergyOverride(TypedDict, total=False):
    """Schema for user-provided energy values.

    Tier definitions:
        0: Measured - Direct hardware measurement (pynvml, rocm-smi)
        1: Vendor-published - Official provider data or peer-reviewed research
        2: Validated - Derived from published research with clear methodology
        3: Estimated - Calculated from parameter count and architecture class
    """

    wh_per_1k_input: float
    wh_per_1k_output: float
    tier: int  # 0, 1, 2, or 3
    source: str  # Free-text provenance
    basis: str  # Methodology/provenance detail


class ValidationResult:
    """Result of validation with warnings."""

    def __init__(
        self,
        value: Union[EnergyOverride, None],
        warnings: list[str],
    ) -> None:
        self.value = value
        self.warnings = warnings


def validate_energy_override(
    override: dict[str, object],
) -> tuple[Union[EnergyOverride, None], list[str]]:
    """Validate energy_override dict.

    Returns:
        Tuple of (validated EnergyOverride or None, list of warning messages).
        Invalid overrides return None with explanatory warnings.
    """
    warnings: list[str] = []

    if not isinstance(override, dict):
        warnings.append("energy_override must be a dict, ignoring")
        return None, warnings

    wh_input = override.get("wh_per_1k_input")
    wh_output = override.get("wh_per_1k_output")

    # Required fields must be present and positive
    if not isinstance(wh_input, (int, float)) or wh_input <= 0:
        warnings.append(
            f"energy_override.wh_per_1k_input must be a positive number, "
            f"got {wh_input!r}, falling back to registry"
        )
        return None, warnings
    if not isinstance(wh_output, (int, float)) or wh_output <= 0:
        warnings.append(
            f"energy_override.wh_per_1k_output must be a positive number, "
            f"got {wh_output!r}, falling back to registry"
        )
        return None, warnings

    result: EnergyOverride = {
        "wh_per_1k_input": float(wh_input),
        "wh_per_1k_output": float(wh_output),
    }

    # Optional tier (default to 1 = vendor-published)
    tier = override.get("tier")
    if tier is not None:
        if isinstance(tier, int) and 0 <= tier <= 3:
            result["tier"] = tier
        else:
            warnings.append(
                f"energy_override.tier must be 0-3, got {tier!r}, using default tier 1"
            )
            result["tier"] = 1

    # Optional source
    source = override.get("source")
    if source is not None and isinstance(source, str):
        result["source"] = source

    # Optional basis/methodology description
    basis = override.get("basis")
    if basis is not None and isinstance(basis, str):
        result["basis"] = basis

    return result, warnings


def normalize_usage_v1_to_v2(event: InferenceEvent) -> InferenceEvent:
    """Convert Schema v1 flat usage to Schema v2 nested structure.

    Schema v1 had `usage: TextUsage` (flat structure with input_tokens, output_tokens).
    Schema v2 has `usage: {text: TextUsage | None, image: ..., audio: ...}` (nested).

    This helper provides backward compatibility for consumers parsing v1 events.

    Args:
        event: InferenceEvent (may be v1 or v2 format).

    Returns:
        InferenceEvent in v2 format (nested usage structure).

    Example::

        # v1 event (flat usage):
        event_v1 = {"usage": {"input_tokens": 100, "output_tokens": 50, ...}}

        # Convert to v2 (nested usage):
        event_v2 = normalize_usage_v1_to_v2(event_v1)
        # Result: {"usage": {"text": {"input_tokens": 100, "output_tokens": 50, ...}, ...}}
    """
    if not event:
        return event

    usage = event.get("usage")
    if not usage:
        return event

    # Check if already in v2 format (has "text", "image", or "audio" keys)
    if isinstance(usage, dict) and any(k in usage for k in ("text", "image", "audio")):
        return event  # Already v2 format

    # v1 format detected (flat TextUsage) - convert to v2
    if isinstance(usage, dict):
        event["usage"] = {
            "text": usage,  # type: ignore[typeddict-item]
            "image": None,
            "audio": None,
        }

    return event
