"""Inference event schema definitions.

This module defines the TypedDict for inference events logged by Vetch.
Schema version 1 guarantees:
- Fields not removed
- Field names not changed
- Field types not changed
- New fields may be added

Breaking changes require schema_version: "2".
"""

from typing import Literal, TypedDict, Union

SCHEMA_VERSION = "1"


class TextUsage(TypedDict):
    """Token usage for text modality."""

    input_tokens: int
    output_tokens: int
    total_tokens: int


class Usage(TypedDict, total=False):
    """Multi-modal usage container.

    The nested structure supports future extension without breaking schema.
    """

    text: Union[TextUsage, None]
    image: None  # Reserved for v2
    audio: None  # Reserved for v2


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

    # Usage
    usage: Union[Usage, None]
    accumulated_chars: Union[int, None]  # For partial streams

    # Estimates
    estimated_energy_wh: Union[float, None]
    estimated_carbon_g: Union[float, None]
    estimated_cost_usd: Union[float, None]
    estimated_cost_input_usd: Union[float, None]
    estimated_cost_output_usd: Union[float, None]
    billing_tier: str  # Always "list" for v1

    # Signal quality
    signal_quality: Literal["live", "delayed", "blind", "unknown"]
    energy_tier: int  # 0=measured, 1=vendor, 2=validated, 3=estimated
    energy_uncertainty_pct: Union[int, None]  # Uncertainty %: 20/50/100/1000 for tier 0/1/2/3
    energy_source: str  # "registry", "override", "fallback", "calibrated"
    energy_override_source: Union[str, None]
    energy_basis: Union[str, None]

    # Grid data
    grid_intensity_gco2e_kwh: Union[float, None]
    grid_intensity_timestamp: Union[str, None]  # ISO8601 UTC
    region: Union[str, None]

    # Request metadata
    is_stream: bool
    complete: bool
    latency_ms: Union[float, None]

    # Attribution
    tags: Union[dict[str, str], None]

    # Error handling
    error: bool
    error_type: Union[str, None]

    # Tracking status
    tracking_disabled: bool

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

    return result, warnings
