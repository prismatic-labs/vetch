"""Energy, carbon, and cost calculation engine.

This module implements the core Vetch formulas to estimate:
- Energy (Wh) based on model-specific token intensity
- Carbon (gCO2e) based on energy and grid intensity
- Cost (USD) based on public list pricing
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Literal, cast

logger = logging.getLogger(__name__)

# Registry paths
_REGISTRY_DIR = Path(__file__).parent / "registry"
_ENERGY_PATH = _REGISTRY_DIR / "energy.json"
_PRICING_PATH = _REGISTRY_DIR / "pricing.json"
_ALIASES_PATH = _REGISTRY_DIR / "aliases.json"

# Lazy-loaded registries
_ENERGY: dict[str, dict[str, Any]] | None = None
_PRICING: dict[str, dict[str, float]] | None = None
_ALIASES: dict[str, str] | None = None


def _load_json_with_override(default_path: Path, override_name: str) -> dict[str, Any]:
    """Load JSON file with optional local override.

    Checks for:
    1. .vetch/{override_name} in current directory
    2. default_path

    Args:
        default_path: Path to the bundled registry file.
        override_name: Name of the override file (e.g., "energy.json").

    Returns:
        Loaded dictionary.
    """
    # Check for local override
    cwd = Path(os.getcwd())
    local_override = cwd / ".vetch" / override_name

    if local_override.exists():
        try:
            logger.info(f"Loading local registry override: {local_override}")
            return cast("dict[str, Any]", json.loads(local_override.read_text()))
        except Exception as e:
            logger.warning(f"Failed to load override {local_override}: {e}")

    # Fallback to default
    try:
        return cast("dict[str, Any]", json.loads(default_path.read_text()))
    except Exception as e:
        logger.error(f"Failed to load registry {default_path}: {e}")
        return {}


def _load_registry() -> None:
    """Load registry files into memory.

    Checks in order:
    1. VETCH_REGISTRY_PATH (offline/air-gapped mode)
    2. Local .vetch/ overrides
    3. Bundled defaults
    4. Remote registry merge (if enabled)
    """
    global _ENERGY, _PRICING, _ALIASES

    # Check for offline mode
    offline_path = os.environ.get("VETCH_REGISTRY_PATH")

    if _ENERGY is None:
        if offline_path:
            from vetch.registry.remote import load_offline_registry

            offline_energy = load_offline_registry(offline_path, "energy.json")
            if offline_energy is not None:
                _ENERGY = cast("dict[str, dict[str, Any]]", offline_energy)

        if _ENERGY is None:
            _ENERGY = cast(
                "dict[str, dict[str, Any]]",
                _load_json_with_override(_ENERGY_PATH, "energy.json"),
            )

        # Merge with remote registry if enabled
        _ENERGY = _merge_remote_energy(_ENERGY)

    if _PRICING is None:
        if offline_path:
            from vetch.registry.remote import load_offline_registry

            offline_pricing = load_offline_registry(offline_path, "pricing.json")
            if offline_pricing is not None:
                _PRICING = cast("dict[str, dict[str, float]]", offline_pricing)

        if _PRICING is None:
            _PRICING = cast(
                "dict[str, dict[str, float]]",
                _load_json_with_override(_PRICING_PATH, "pricing.json"),
            )

        # Merge with remote registry if enabled
        _PRICING = _merge_remote_pricing(_PRICING)

    if _ALIASES is None:
        if offline_path:
            from vetch.registry.remote import load_offline_registry

            offline_aliases = load_offline_registry(offline_path, "aliases.json")
            if offline_aliases is not None:
                _ALIASES = cast("dict[str, str]", offline_aliases)

        if _ALIASES is None:
            _ALIASES = cast(
                "dict[str, str]", _load_json_with_override(_ALIASES_PATH, "aliases.json")
            )


def _merge_remote_energy(bundled: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Merge bundled energy data with remote if available.

    Skips merge if bundled data is empty (indicates load failure).
    """
    if not bundled:
        return bundled
    try:
        from vetch.registry.remote import get_remote_fetcher

        fetcher = get_remote_fetcher()
        if fetcher is not None:
            return cast("dict[str, dict[str, Any]]", fetcher.get_energy(bundled))
    except Exception as e:
        logger.debug(f"Failed to merge remote energy registry: {e}")
    return bundled


def _merge_remote_pricing(bundled: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Merge bundled pricing data with remote if available.

    Skips merge if bundled data is empty (indicates load failure).
    """
    if not bundled:
        return bundled
    try:
        from vetch.registry.remote import get_remote_fetcher

        fetcher = get_remote_fetcher()
        if fetcher is not None:
            return cast("dict[str, dict[str, float]]", fetcher.get_pricing(bundled))
    except Exception as e:
        logger.debug(f"Failed to merge remote pricing registry: {e}")
    return bundled


def _reset_registries() -> None:
    """Reset registries to None. Primarily for testing."""
    global _ENERGY, _PRICING, _ALIASES
    _ENERGY = None
    _PRICING = None
    _ALIASES = None


def resolve_model(model: str) -> tuple[str, bool]:
    """Resolve a model name to a registry entry, handling aliases.

    Args:
        model: Original model name from SDK.

    Returns:
        Tuple of (resolved_model_name, is_known).
    """
    _load_registry()
    assert _ENERGY is not None
    assert _ALIASES is not None

    # 1. Direct match
    if model in _ENERGY:
        return model, True

    # 2. Alias match
    if model in _ALIASES:
        resolved = _ALIASES[model]
        if resolved in _ENERGY:
            return resolved, True

    # 3. Prefix matching (gpt-4-0613 -> gpt-4)
    # Try progressively shorter prefixes split by hyphens
    parts = model.split("-")
    for i in range(len(parts) - 1, 0, -1):
        prefix = "-".join(parts[:i])
        if prefix in _ENERGY:
            return prefix, True
        if prefix in _ALIASES:
            resolved = _ALIASES[prefix]
            if resolved in _ENERGY:
                return resolved, True

    return model, False


def get_conservative_energy() -> dict[str, Any]:
    """Get conservative fallback values for unknown models."""
    return {
        "wh_per_1k_input": 1.4,  # Slightly higher than Claude 3 Opus
        "wh_per_1k_output": 4.2,
        "tier": 3,
        "basis": "Conservative fallback for unknown model",
    }


# Uncertainty percentage by tier (upper bound of range)
# Tier 0: ±10-20% (measured hardware)
# Tier 1: ±20-50% (vendor-published)
# Tier 2: ±50-100% (validated research)
# Tier 3: order of magnitude (~1000%, i.e., could be 0.1x to 10x)
TIER_UNCERTAINTY_PCT: dict[int, int] = {
    0: 20,
    1: 50,
    2: 100,
    3: 1000,
}


def get_uncertainty_pct(tier: int) -> int:
    """Get uncertainty percentage for a given energy tier.

    Args:
        tier: Energy tier (0-3).

    Returns:
        Uncertainty as percentage (e.g., 20 means ±20%).
    """
    return TIER_UNCERTAINTY_PCT.get(tier, 1000)


# Tiktoken availability flag (lazy-loaded)
_TIKTOKEN_AVAILABLE: bool | None = None
_TIKTOKEN_WARNING_ISSUED = False


def _get_tiktoken_encoding(model: str | None) -> Any:
    """Get tiktoken encoding for a model, with lazy loading.

    Args:
        model: Model name (e.g., "gpt-4o", "claude-3-opus").

    Returns:
        tiktoken Encoding object, or None if unavailable.
    """
    global _TIKTOKEN_AVAILABLE, _TIKTOKEN_WARNING_ISSUED

    if _TIKTOKEN_AVAILABLE is False:
        return None

    try:
        import tiktoken  # type: ignore[import-not-found]

        _TIKTOKEN_AVAILABLE = True

        # Try model-specific encoding first
        if model:
            try:
                return tiktoken.encoding_for_model(model)
            except KeyError:
                pass

        # Fall back to cl100k_base (GPT-4, GPT-3.5-turbo, embeddings)
        # This is a reasonable proxy for most modern LLMs
        return tiktoken.get_encoding("cl100k_base")

    except ImportError:
        _TIKTOKEN_AVAILABLE = False
        if not _TIKTOKEN_WARNING_ISSUED:
            logger.info(
                "Performance hint: Install 'tiktoken' for more accurate token estimates. "
                "pip install vetch[tiktoken]"
            )
            _TIKTOKEN_WARNING_ISSUED = True
        return None


def estimate_tokens(text: str | None, model: str | None = None) -> int:
    """Estimate token count from text.

    Uses tiktoken if available (accurate for OpenAI models, good proxy for others).
    Falls back to character-ratio heuristic if tiktoken not installed.

    Args:
        text: The text to estimate.
        model: Optional model name for model-specific encoding.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0

    # Try tiktoken first
    encoding = _get_tiktoken_encoding(model)
    if encoding is not None:
        try:
            return len(encoding.encode(text))
        except Exception as e:
            logger.debug(f"tiktoken encoding failed: {e}, falling back to heuristic")

    # Fallback: character-ratio heuristic
    # ~4 chars/token for English, but adjust for other content types
    char_count = len(text)

    # Detect likely content type and adjust ratio
    # CJK characters are roughly 1.5-2 tokens each
    # Code tends to have more tokens per character
    cjk_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    if cjk_count > char_count * 0.3:
        # Predominantly CJK text: ~1.5 chars per token
        return max(1, int(char_count / 1.5))

    # Check for code-like content (high punctuation/operator density)
    code_chars = sum(1 for c in text if c in "{}[]()<>;:=+-*/&|!@#$%^")
    if code_chars > char_count * 0.1:
        # Code-like: ~3 chars per token
        return max(1, char_count // 3)

    # Default: English prose ~4 chars per token
    return max(1, char_count // 4)


def calculate_energy(
    input_tokens: int,
    output_tokens: int,
    model: str,
    energy_override: dict[str, Any] | None = None,
) -> tuple[float, int, int, str, str, bool]:
    """Calculate energy consumption in Watt-hours.

    Supports prompt-length-aware coefficients (non-linear model):
    - Short prompts (< 1000 input tokens)
    - Medium prompts (1000-5000 input tokens)
    - Long prompts (> 5000 input tokens)

    Note: Category is determined by input_tokens only, not total_tokens.
    This reflects the physics: prefill (input) has fixed costs that amortize,
    while decode (output) is autoregressive and linear per-token.

    Falls back to linear model for entries without prompt-length data.

    Args:
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        model: Model identifier.
        energy_override: User-provided energy values.

    Returns:
        Tuple of (energy_wh, tier, uncertainty_pct, source, basis, model_known).
    """
    # Clamp negative tokens to 0
    in_tokens = max(0, input_tokens)
    out_tokens = max(0, output_tokens)

    if energy_override:
        wh_in = energy_override["wh_per_1k_input"]
        wh_out = energy_override["wh_per_1k_output"]
        tier = energy_override.get("tier", 1)
        source = energy_override.get("source", "override")
        basis = energy_override.get("basis", "User-provided override")

        energy_wh = (in_tokens * wh_in + out_tokens * wh_out) / 1000
        uncertainty_pct = get_uncertainty_pct(tier)
        # Check if model is known in registry anyway for informational purposes
        _, known = resolve_model(model)
        return energy_wh, tier, uncertainty_pct, "override", basis, known

    resolved_model, known = resolve_model(model)
    _load_registry()
    assert _ENERGY is not None

    if known:
        entry = _ENERGY[resolved_model]

        # Check if entry has prompt-length-aware coefficients (non-linear model)
        if "prompt_length" in entry:
            # Determine prompt length category based on INPUT tokens only.
            # Rationale: Fixed-cost amortization happens during prefill (input).
            # Autoregressive generation (output) is memory-bandwidth bound and linear.
            # Using total_tokens incorrectly subsidizes "chatty" responses.
            if input_tokens < 1000:
                category = "short"
            elif input_tokens < 5000:
                category = "medium"
            else:
                category = "long"

            # Get coefficients for this prompt length
            pl_entry = entry["prompt_length"][category]
            wh_in = pl_entry["wh_per_1k_input"]
            wh_out = pl_entry["wh_per_1k_output"]
            basis = entry.get("basis", f"Jegham et al. (2025) measured data ({category} prompt)")
        else:
            # Legacy format: flat coefficients (linear model)
            wh_in = entry["wh_per_1k_input"]
            wh_out = entry["wh_per_1k_output"]
            basis = entry["basis"]

        tier = entry["tier"]
        source = "registry"
    else:
        entry = get_conservative_energy()
        wh_in = entry["wh_per_1k_input"]
        wh_out = entry["wh_per_1k_output"]
        tier = entry["tier"]
        basis = entry["basis"]
        source = "fallback"

    energy_wh = (in_tokens * wh_in + out_tokens * wh_out) / 1000
    uncertainty_pct = get_uncertainty_pct(tier)
    return energy_wh, tier, uncertainty_pct, source, basis, known


# Default PUE (Power Usage Effectiveness) for data centers
# Industry average: ~1.58 (Uptime Institute 2023)
# We use 1.2 as a reasonable default for unknown providers
DEFAULT_PUE = 1.2

# Provider-specific PUE values from official sustainability reports (Tier 1 data)
# Sources:
#   Google: https://datacenters.google/efficiency/ (2023: 1.10)
#   Azure: https://datacenters.microsoft.com/sustainability/efficiency/ (2024: 1.12)
#   AWS: https://aws.amazon.com/sustainability/data-centers/ (2024: 1.15)
PROVIDER_PUE: dict[str, float] = {
    "google": 1.10,      # Google Cloud (2023 average)
    "vertexai": 1.10,    # Vertex AI runs on Google Cloud
    "azure": 1.12,       # Microsoft Azure (2024 newest gen)
    "openai": 1.12,      # OpenAI primarily uses Azure
    "aws": 1.15,         # AWS (2024 global average)
    "anthropic": 1.15,   # Anthropic uses AWS
    "bedrock": 1.15,     # AWS Bedrock
}

# Documentation sources for transparency
PROVIDER_PUE_SOURCES: dict[str, str] = {
    "google": "Google Data Centers Efficiency Report 2023",
    "vertexai": "Google Data Centers Efficiency Report 2023",
    "azure": "Microsoft Datacenters Sustainability 2024",
    "openai": "Microsoft Datacenters Sustainability 2024 (Azure-backed)",
    "aws": "AWS Sustainability Report 2024",
    "anthropic": "AWS Sustainability Report 2024 (AWS-backed)",
    "bedrock": "AWS Sustainability Report 2024",
}


def _infer_provider_from_model(model: str) -> str | None:
    """Infer cloud provider from model name patterns.

    Args:
        model: Model identifier (e.g., "gpt-4o", "claude-3-opus")

    Returns:
        Provider key for PUE lookup, or None if unknown.
    """
    model_lower = model.lower()

    # OpenAI models (Azure-backed)
    openai_prefixes = ["gpt-", "o1-", "o3-", "text-davinci", "text-embedding"]
    if any(prefix in model_lower for prefix in openai_prefixes):
        return "openai"

    # Anthropic models (AWS-backed)
    if model_lower.startswith("claude-"):
        return "anthropic"

    # Google models
    if any(prefix in model_lower for prefix in ["gemini-", "gemma-", "palm-"]):
        return "google"

    # Unknown
    return None


def get_provider_pue(
    model: str | None = None, provider_hint: str | None = None
) -> tuple[float, int, str]:
    """Get PUE for a model's cloud provider.

    Args:
        model: Model identifier for provider inference.
        provider_hint: Explicit provider override
            ("openai", "anthropic", "vertexai", "aws", "azure", "google").

    Returns:
        Tuple of (pue, tier, source)
        - tier: 1=known value (user config or vendor-published), 3=default fallback
    """
    # Check environment variable first (user config, Tier 1: known value)
    env_pue = os.environ.get("VETCH_DEFAULT_PUE")
    if env_pue is not None:
        try:
            pue = float(env_pue)
            if pue >= 1.0:
                return pue, 1, "user config (VETCH_DEFAULT_PUE)"
        except ValueError:
            pass

    # Use explicit provider hint if provided (Tier 1: vendor-published)
    if provider_hint:
        provider_lower = provider_hint.lower()
        if provider_lower in PROVIDER_PUE:
            pue_val = PROVIDER_PUE[provider_lower]
            pue_src = PROVIDER_PUE_SOURCES.get(provider_lower, "vendor report")
            return pue_val, 1, pue_src

    # Infer provider from model name (Tier 1: vendor-published)
    if model:
        inferred = _infer_provider_from_model(model)
        if inferred and inferred in PROVIDER_PUE:
            return PROVIDER_PUE[inferred], 1, PROVIDER_PUE_SOURCES[inferred]

    # Fallback to default (Tier 3: unknown)
    return DEFAULT_PUE, 3, "industry average"


def get_default_pue() -> float:
    """Get the default PUE value (deprecated, use get_provider_pue).

    Returns:
        PUE value from VETCH_DEFAULT_PUE env var, or 1.2 if not set/invalid.
    """
    env_pue = os.environ.get("VETCH_DEFAULT_PUE")
    if env_pue is None:
        return DEFAULT_PUE

    try:
        pue = float(env_pue)
        # Sanity check: PUE must be >= 1.0 (can't be more efficient than 100%)
        if pue < 1.0:
            logger.warning(
                f"VETCH_DEFAULT_PUE={pue} is invalid (must be >= 1.0), using {DEFAULT_PUE}"
            )
            return DEFAULT_PUE
        return pue
    except ValueError:
        logger.warning(f"VETCH_DEFAULT_PUE={env_pue!r} is not a valid number, using {DEFAULT_PUE}")
        return DEFAULT_PUE


def calculate_carbon(
    energy_wh: float,
    grid_intensity_gco2e_kwh: float,
    model: str | None = None,
    provider_hint: str | None = None,
    pue_override: float | None = None,
    pue: float | None = None,  # Backward compatibility alias for pue_override
) -> tuple[float, float, int, str]:
    """Calculate carbon emissions in grams of CO2e with provider-specific PUE.

    Formula:
    carbon_g = energy_wh * PUE * grid_intensity / 1000

    Args:
        energy_wh: Energy in Watt-hours.
        grid_intensity_gco2e_kwh: Carbon intensity in gCO2e/kWh.
        model: Model identifier for provider-specific PUE inference.
        provider_hint: Explicit provider ("openai", "anthropic", "vertexai").
        pue_override: Explicit PUE value (takes precedence over all).
        pue: (Deprecated) Alias for pue_override, kept for backward compatibility.

    Returns:
        Tuple of (carbon_g, pue, pue_tier, pue_source)
        - pue_tier: 1=known value (user config or vendor-published), 3=default fallback
    """
    import math

    # Determine PUE with provider intelligence
    # Support legacy 'pue' parameter for backward compatibility
    effective_override = pue_override if pue_override is not None else pue

    if effective_override is not None:
        pue_val = effective_override
        pue_tier = 1
        pue_source = "explicit override"
    else:
        pue_val, pue_tier, pue_source = get_provider_pue(model, provider_hint)

    # Defensive handling of NaN and Inf
    intensity = grid_intensity_gco2e_kwh
    if math.isnan(intensity):
        intensity = 0.0
    # Cap extremely high intensity at a reasonable max (e.g. 2000 g/kWh)
    if math.isinf(intensity) or intensity > 2000:
        intensity = 2000.0

    carbon_g = (energy_wh * pue_val * intensity) / 1000

    return carbon_g, pue_val, pue_tier, pue_source


# Water Usage Effectiveness (WUE) - liters per kWh
# Based on datacenter cooling requirements
# Sources: Google Environmental Report 2023, Microsoft Sustainability Report 2024
DEFAULT_WUE = 1.8  # L/kWh industry average for air-cooled datacenters

PROVIDER_WUE: dict[str, float] = {
    "google": 1.1,  # Google's efficient water-free cooling in many DCs
    "vertexai": 1.1,
    "azure": 1.7,  # Microsoft's water usage per kWh
    "openai": 1.7,  # OpenAI primarily uses Azure
    "aws": 2.2,  # AWS average (higher due to evaporative cooling)
    "anthropic": 2.2,  # Anthropic uses AWS
    "bedrock": 2.2,
}

# Lazy-loaded WUE registry
_WUE_REGISTRY: dict[str, float] | None = None


def _load_wue_registry() -> dict[str, float]:
    """Load WUE registry from wue.json file.

    Returns:
        Dictionary mapping region/provider keys to WUE values (L/kWh).
    """
    global _WUE_REGISTRY
    if _WUE_REGISTRY is not None:
        return _WUE_REGISTRY

    wue_path = _REGISTRY_DIR / "wue.json"
    try:
        data = cast("dict[str, Any]", json.loads(wue_path.read_text()))
        # Filter out metadata keys (starting with _)
        _WUE_REGISTRY = {k: float(v) for k, v in data.items() if not k.startswith("_")}
        return _WUE_REGISTRY
    except Exception as e:
        logger.debug(f"Failed to load WUE registry: {e}, using defaults")
        _WUE_REGISTRY = {}
        return _WUE_REGISTRY


def calculate_water(
    energy_wh: float,
    model: str | None = None,
    provider_hint: str | None = None,
    region: str | None = None,
    wue_override: float | None = None,
) -> float:
    """Calculate water usage in liters for datacenter cooling.

    WUE varies significantly by datacenter location (0.2-3.5 L/kWh):
    - River cooling (Virginia): ~0.8 L/kWh
    - Air cooling (Oregon): ~2.8 L/kWh

    Formula:
    water_l = (energy_wh / 1000) * WUE

    Args:
        energy_wh: Energy in Watt-hours.
        model: Model identifier for provider-specific WUE inference.
        provider_hint: Explicit provider ("openai", "anthropic", "vertexai").
        region: Cloud region for location-specific WUE (e.g., "us-east-1").
        wue_override: Explicit WUE value (liters per kWh).

    Returns:
        Water usage in liters.

    Note:
        Regional WUE estimates have ±200% uncertainty vs ±50% for carbon.
    """
    # Load WUE registry
    wue_registry = _load_wue_registry()

    # Determine WUE with cascading fallback:
    # 1. Explicit override
    # 2. Region-specific (provider-region)
    # 3. Provider-level default
    # 4. Global default
    if wue_override is not None:
        wue = wue_override
    elif region and provider_hint:
        # Try provider-region specific (e.g., "aws-us-east-1")
        region_key = f"{provider_hint.lower()}-{region.lower()}"
        if region_key in wue_registry:
            wue = wue_registry[region_key]
        # Fall back to provider-level
        elif provider_hint.lower() in wue_registry:
            wue = wue_registry[provider_hint.lower()]
        elif provider_hint.lower() in PROVIDER_WUE:
            wue = PROVIDER_WUE[provider_hint.lower()]
        else:
            wue = DEFAULT_WUE
    elif provider_hint:
        # Provider-level lookup
        if provider_hint.lower() in wue_registry:
            wue = wue_registry[provider_hint.lower()]
        elif provider_hint.lower() in PROVIDER_WUE:
            wue = PROVIDER_WUE[provider_hint.lower()]
        else:
            wue = DEFAULT_WUE
    elif model:
        # Infer provider from model name
        inferred = _infer_provider_from_model(model)
        if inferred and region:
            # Try provider-region
            region_key = f"{inferred}-{region.lower()}"
            if region_key in wue_registry:
                wue = wue_registry[region_key]
            elif inferred in wue_registry:
                wue = wue_registry[inferred]
            else:
                wue = PROVIDER_WUE.get(inferred, DEFAULT_WUE)
        elif inferred:
            wue = wue_registry.get(inferred, PROVIDER_WUE.get(inferred, DEFAULT_WUE))
        else:
            wue = DEFAULT_WUE
    else:
        wue = DEFAULT_WUE

    # Convert Wh to kWh and multiply by WUE
    water_l = (energy_wh / 1000) * wue
    return water_l


def calculate_embodied_carbon(
    input_tokens: int,
    output_tokens: int,
    model: str | None = None,
) -> float:
    """Calculate embodied carbon from hardware manufacturing.

    Embodied carbon is the emissions from manufacturing and transporting
    the hardware used for inference. This is amortized over the hardware
    lifetime and scaled by usage.

    Model size affects embodied carbon significantly:
    - H100 cluster for GPT-4: 220x more emissions than L40S for Llama-8B
    - Embodied carbon scales with active parameters

    Formula:
    embodied_g = (total_tokens / 1000) * embodied_factor

    Args:
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        model: Model identifier for size-based scaling.

    Returns:
        Embodied carbon in grams CO2e.
    """
    # Default embodied carbon factor: gCO2e per 1k tokens
    # Based on GPU manufacturing emissions amortized over lifetime
    # Source: Patterson et al. (2021)
    DEFAULT_EMBODIED_FACTOR = 0.075  # Medium models (10-100B params)

    # Get model-specific embodied factor from registry
    embodied_factor = DEFAULT_EMBODIED_FACTOR
    if model:
        resolved_model, known = resolve_model(model)
        _load_registry()
        assert _ENERGY is not None

        if known and resolved_model in _ENERGY:
            entry = _ENERGY[resolved_model]
            embodied_factor = entry.get("embodied_factor", DEFAULT_EMBODIED_FACTOR)
        else:
            # Estimate by parameter count if available
            # Small (<10B): 0.02, Medium (10-100B): 0.075, Large (>100B): 0.25, MoE (>1T): 0.8
            embodied_factor = _estimate_embodied_factor_by_model_name(model)

    total_tokens = input_tokens + output_tokens
    return (total_tokens / 1000) * embodied_factor


def _estimate_embodied_factor_by_model_name(model: str) -> float:
    """Estimate embodied carbon factor from model name patterns.

    Args:
        model: Model identifier (e.g., "gpt-4o", "llama-3.1-8b").

    Returns:
        Estimated embodied factor (gCO2e per 1k tokens).
    """
    model_lower = model.lower()

    # Large MoE models (>1T active params)
    if any(x in model_lower for x in ["o1", "o3", "gpt-4", "gpt4", "claude-3-opus"]):
        return 0.25

    # Small models (<10B params)
    if any(x in model_lower for x in ["-7b", "-8b", "small", "mini", "nano"]):
        return 0.02

    # Medium models (10-100B params) - default
    return 0.075


def _calculate_tiered_cost(
    tokens: int,
    base_rate_per_1k: float,
    tier_threshold: int | None,
    tier_multiplier: float | None,
) -> float:
    """Calculate cost with optional threshold-based tiered pricing.

    IMPORTANT: Uses THRESHOLD pricing (Google Cloud model), not bracket pricing.
    When token count exceeds threshold, ALL tokens are charged at the higher rate.

    Args:
        tokens: Number of tokens
        base_rate_per_1k: Base rate per 1000 tokens
        tier_threshold: Token count where higher tier kicks in (None = no tiers)
        tier_multiplier: Multiplier for ALL tokens when over threshold (None = no tiers)

    Returns:
        Total cost in USD

    Example:
        >>> # No tiering - standard calculation
        >>> _calculate_tiered_cost(1000, 0.001, None, None)
        0.001

        >>> # Under threshold: 100k tokens, $1.25/M base, threshold at 128k
        >>> # ALL tokens @ $1.25/M = $0.125
        >>> _calculate_tiered_cost(100000, 0.00125, 128000, 2.0)
        0.125

        >>> # Over threshold: 200k tokens, $1.25/M base, 2x over 128k
        >>> # ALL 200k tokens @ $2.50/M (base * multiplier) = $0.50
        >>> _calculate_tiered_cost(200000, 0.00125, 128000, 2.0)
        0.5

    Note:
        Previous implementation used bracket pricing (tax-style), which
        under-reported Gemini costs by ~32% for long-context workloads.
    """
    if tier_threshold is None or tier_multiplier is None:
        # No tiering - standard calculation
        return (tokens * base_rate_per_1k) / 1000

    if tokens <= tier_threshold:
        # Under threshold - base rate for all tokens
        return (tokens * base_rate_per_1k) / 1000

    # Over threshold - THRESHOLD PRICING: ALL tokens at higher rate
    # This matches Google Cloud's actual billing model
    higher_rate = base_rate_per_1k * tier_multiplier
    return (tokens * higher_rate) / 1000


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str,
    cache_read_tokens: int | None = None,
    cache_creation_tokens: int | None = None,
) -> tuple[float, float, float, float, float, str]:
    """Calculate estimated cost in USD with cache tier breakdown.

    Supports cache-aware pricing: cache_read tokens are discounted
    (typically 90% cheaper), cache_creation tokens may have extra cost.

    Supports tiered pricing: models with tier_threshold and tier_multiplier
    charge different rates for tokens above the threshold (e.g., Gemini Pro
    models charge 2x for >128k tokens).

    Args:
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        model: Model identifier.
        cache_read_tokens: Tokens read from prompt cache (cost savings).
        cache_creation_tokens: Tokens written to prompt cache (extra cost).

    Returns:
        Tuple of (total_cost, input_cost, output_cost, cache_write_cost, cache_read_cost,
        billing_tier). cache_write_cost: Cost to write tokens to cache (included in total)
        cache_read_cost: Cost for cached token reads (included in total, typically discounted)

    Example:
        >>> # Standard model (no tiers)
        >>> calculate_cost(1000, 500, "gpt-4o")
        (0.0125, 0.005, 0.0075, 0.0, 0.0, 'list')

        >>> # Tiered model (Gemini 2.5 Pro): 200k input, 1k output
        >>> # Input: 128k @ $1.25/M + 72k @ $2.50/M = $0.16 + $0.18 = $0.34
        >>> # Output: 1k @ $10/M = $0.01
        >>> # Total: $0.35
        >>> calculate_cost(200000, 1000, "gemini-2.5-pro")
        (0.35, 0.34, 0.01, 0.0, 0.0, 'list')
    """
    resolved_model, known = resolve_model(model)
    _load_registry()
    assert _PRICING is not None

    if known and resolved_model in _PRICING:
        entry = _PRICING[resolved_model]
        rate_in = entry["usd_per_1k_input"]
        rate_out = entry["usd_per_1k_output"]
        tier_threshold_raw = entry.get("tier_threshold")
        tier_threshold = int(tier_threshold_raw) if tier_threshold_raw is not None else None

        # Support both old (single tier_multiplier) and new (separate input/output) formats
        # New format: tier_multiplier_input/tier_multiplier_output (e.g., Gemini 2.5 Pro)
        # Old format: tier_multiplier (applies to both, e.g., Gemini 1.5 Pro)
        tier_multiplier_input = entry.get("tier_multiplier_input") or entry.get("tier_multiplier")
        tier_multiplier_output = entry.get("tier_multiplier_output") or entry.get("tier_multiplier")
    else:
        # No pricing for unknown models
        return 0.0, 0.0, 0.0, 0.0, 0.0, "none"

    # Cache-aware cost calculation
    # Cache read tokens are charged at a discount (default: 10% of input price)
    # Cache creation tokens are charged at a premium (default: 125% of input price)
    cache_read_discount = entry.get("cache_read_discount", 0.1)
    cache_creation_premium = entry.get("cache_creation_premium", 1.25)

    # Base input tokens (excluding cached tokens)
    effective_input = input_tokens
    cache_write_cost = 0.0
    cache_read_cost = 0.0

    if cache_read_tokens and cache_read_tokens > 0:
        # Subtract cache read tokens from base input, add discounted cost
        effective_input = max(0, input_tokens - cache_read_tokens)
        cache_read_cost = (cache_read_tokens * rate_in * cache_read_discount) / 1000

    if cache_creation_tokens and cache_creation_tokens > 0:
        # Cache creation tokens cost extra on top of normal input cost
        cache_write_cost = (cache_creation_tokens * rate_in * cache_creation_premium) / 1000

    # Calculate input cost with tiered pricing
    cost_in = (
        _calculate_tiered_cost(effective_input, rate_in, tier_threshold, tier_multiplier_input)
        + cache_write_cost
        + cache_read_cost
    )

    # Calculate output cost with tiered pricing
    cost_out = _calculate_tiered_cost(
        output_tokens, rate_out, tier_threshold, tier_multiplier_output
    )

    total_cost = cost_in + cost_out

    return total_cost, cost_in, cost_out, cache_write_cost, cache_read_cost, "list"


# ---------------------------------------------------------------------------
# Metrics preparation — extracted from VetchContext._emit_event
# ---------------------------------------------------------------------------


class InferenceMetrics:
    """All computed metrics for a single inference event.

    Returned by :func:`prepare_inference_metrics` and consumed by
    ``VetchContext._emit_event`` to build the final ``InferenceEvent``.
    """

    __slots__ = (
        "energy_wh",
        "energy_tier",
        "energy_uncertainty_pct",
        "energy_source",
        "energy_basis",
        "model_known",
        "carbon_g",
        "pue",
        "pue_tier",
        "pue_source",
        "water_l",
        "embodied_carbon_g",
        "cost_usd",
        "cost_in_usd",
        "cost_out_usd",
        "cost_cache_write_usd",
        "cost_cache_read_usd",
        "billing_tier",
        "signal_quality",
        "grid_val",
        "grid_ts",
        "usage",
        "usage_estimated",
        "usage_estimation_method",
        "tracking_degraded",
        "request_fingerprint",
        "warnings",
    )

    def __init__(self) -> None:
        self.energy_wh: float | None = None
        self.energy_tier: int = 3
        self.energy_uncertainty_pct: int | None = 1000
        self.energy_source: str = "registry"
        self.energy_basis: str | None = None
        self.model_known: bool = False
        self.carbon_g: float | None = None
        self.pue: float | None = None
        self.pue_tier: int = 3
        self.pue_source: str = "unknown"
        self.water_l: float | None = None
        self.embodied_carbon_g: float | None = None
        self.cost_usd: float | None = None
        self.cost_in_usd: float | None = None
        self.cost_out_usd: float | None = None
        self.cost_cache_write_usd: float = 0.0
        self.cost_cache_read_usd: float = 0.0
        self.billing_tier: str = "list"
        self.signal_quality: Literal["live", "delayed", "blind", "unknown"] = "unknown"
        self.grid_val: float = 0.0
        self.grid_ts: str | None = None
        self.usage: Any = None
        self.usage_estimated: bool = False
        self.usage_estimation_method: str | None = None
        self.tracking_degraded: bool = False
        self.request_fingerprint: str | None = None
        self.warnings: list[str] = []


def prepare_inference_metrics(
    model: str,
    provider: str,
    usage: Any,
    accumulated_chars: int,
    region: str | None,
    price_multiplier: float,
    energy_override: dict[str, Any] | None,
    cache_read_tokens: int | None,
    cache_creation_tokens: int | None,
    existing_warnings: list[str],
) -> InferenceMetrics:
    """Compute all energy/carbon/cost metrics for a single inference call.

    Extracted from ``VetchContext._emit_event`` to keep orchestration logic in
    ``wrapper.py`` thin and all calculation logic here in ``calculation.py``.

    Args:
        model: Resolved model name (e.g. "gpt-4o").
        provider: Provider string (e.g. "openai").
        usage: Nested usage dict from the provider response.
        accumulated_chars: Character count for streams without usage data.
        region: Electricity Maps zone ID for grid lookup.
        price_multiplier: Multiplier applied to list cost (e.g. 0.8 for discount).
        energy_override: Optional user-supplied energy values.
        cache_read_tokens: Cache-read token count for cost discount.
        cache_creation_tokens: Cache-creation token count for cost premium.
        existing_warnings: Warnings accumulated earlier in the context lifecycle.

    Returns:
        :class:`InferenceMetrics` with all computed values populated.
    """
    import hashlib
    from datetime import datetime, timezone

    from vetch.sensing.grid import get_carbon_intensity

    metrics = InferenceMetrics()
    metrics.warnings = list(existing_warnings)

    # 1. Grid intensity
    grid_intensity = get_carbon_intensity(region)
    metrics.signal_quality = grid_intensity.signal_quality
    metrics.grid_val = grid_intensity.intensity_gco2e_kwh
    if grid_intensity.timestamp:
        ts = datetime.fromtimestamp(grid_intensity.timestamp, tz=timezone.utc)
        metrics.grid_ts = ts.isoformat().replace("+00:00", "Z")

    # 2. Token estimation fallback for streams without usage data
    if (not usage or not usage.get("text")) and accumulated_chars > 0:
        estimated_output_tokens = max(1, accumulated_chars // 4)
        estimated_input_tokens = estimated_output_tokens * 2
        usage = {
            "text": {
                "input_tokens": estimated_input_tokens,
                "output_tokens": estimated_output_tokens,
                "total_tokens": estimated_input_tokens + estimated_output_tokens,
            }
        }
        metrics.usage_estimated = True
        metrics.usage_estimation_method = "char_ratio"
        metrics.warnings.append(
            f"Token usage estimated from {accumulated_chars} chars "
            f"(~4 chars/token). Actual usage may differ by ±50%."
        )

    metrics.usage = usage

    # 3. Energy / carbon / cost calculations
    if usage and usage.get("text"):
        text = usage["text"]
        if text:
            in_tokens = text.get("input_tokens", 0)
            out_tokens = text.get("output_tokens", 0)

            # Include reasoning tokens (o1/o3 thinking models) in energy calc
            if usage.get("reasoning"):
                reasoning = usage["reasoning"]
                if reasoning:
                    in_tokens += reasoning.get("input_tokens", 0)

            (
                metrics.energy_wh,
                metrics.energy_tier,
                metrics.energy_uncertainty_pct,
                metrics.energy_source,
                metrics.energy_basis,
                metrics.model_known,
            ) = calculate_energy(
                in_tokens,
                out_tokens,
                model,
                cast("dict[str, Any]", energy_override),
            )

            if not metrics.model_known and model != "unknown":
                metrics.warnings.append(
                    f"Model '{model}' not in registry, using conservative fallback estimates. "
                    f"Energy/cost estimates may be inaccurate (±100% uncertainty)"
                )

            if metrics.energy_wh is not None:
                metrics.carbon_g, metrics.pue, metrics.pue_tier, metrics.pue_source = (
                    calculate_carbon(
                        metrics.energy_wh,
                        metrics.grid_val,
                        model=model,
                        provider_hint=provider,
                    )
                )
                metrics.water_l = calculate_water(
                    metrics.energy_wh,
                    model=model,
                    provider_hint=provider,
                    region=region,
                )
                metrics.embodied_carbon_g = calculate_embodied_carbon(
                    in_tokens, out_tokens, model
                )

            (
                metrics.cost_usd,
                metrics.cost_in_usd,
                metrics.cost_out_usd,
                metrics.cost_cache_write_usd,
                metrics.cost_cache_read_usd,
                metrics.billing_tier,
            ) = calculate_cost(
                in_tokens,
                out_tokens,
                model,
                cache_read_tokens=cache_read_tokens,
                cache_creation_tokens=cache_creation_tokens,
            )

            # Apply price multiplier
            if price_multiplier != 1.0 and metrics.cost_usd is not None:
                metrics.cost_usd *= price_multiplier
                if metrics.cost_in_usd is not None:
                    metrics.cost_in_usd *= price_multiplier
                if metrics.cost_out_usd is not None:
                    metrics.cost_out_usd *= price_multiplier
                metrics.cost_cache_write_usd *= price_multiplier
                metrics.cost_cache_read_usd *= price_multiplier
                metrics.billing_tier = f"list×{price_multiplier}"

    # 4. Tracking degradation score
    grid_quality_score = {"live": 0.0, "delayed": 1.0, "blind": 2.0, "unknown": 3.0}
    score = (
        (0.0 if metrics.model_known else 0.6)
        + (metrics.energy_tier / 3.0) * 0.6
        + (metrics.pue_tier / 3.0) * 0.2
        + (grid_quality_score.get(metrics.signal_quality, 3.0) / 3.0) * 0.2
        + (0.4 if metrics.usage_estimated else 0.0)
    )
    metrics.tracking_degraded = score > 2.5

    # 5. Request fingerprint for deduplication
    if usage and usage.get("text"):
        text = usage["text"]
        if text:
            ts_minute = datetime.now(timezone.utc).replace(second=0, microsecond=0)
            fp_input = (
                f"{model}:{text.get('input_tokens', 0)}"
                f":{text.get('output_tokens', 0)}:{ts_minute.isoformat()}"
            )
            metrics.request_fingerprint = hashlib.sha256(fp_input.encode()).hexdigest()[:16]

    return metrics
