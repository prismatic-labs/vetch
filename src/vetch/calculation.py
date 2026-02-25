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
from typing import Any, cast

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

    Formula:
    energy_wh = (input_tokens * wh_per_1k_input + output_tokens * wh_per_1k_output) / 1000

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
        wh_in = entry["wh_per_1k_input"]
        wh_out = entry["wh_per_1k_output"]
        tier = entry["tier"]
        basis = entry["basis"]
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
# Hyperscalers: Google 1.09, AWS 1.14, Azure 1.12
# Industry average: ~1.58 (Uptime Institute 2023)
# We use 1.2 as a reasonable default for cloud inference
# (Jiang et al. 2025, arXiv:2505.09598). Override with VETCH_DEFAULT_PUE.
DEFAULT_PUE = 1.2


def get_default_pue() -> float:
    """Get the default PUE value, respecting environment override.

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
    pue: float | None = None,
) -> float:
    """Calculate carbon emissions in grams of CO2e.

    Formula:
    carbon_g = energy_wh * PUE * grid_intensity / 1000

    Args:
        energy_wh: Energy in Watt-hours.
        grid_intensity_gco2e_kwh: Carbon intensity in gCO2e/kWh.
        pue: Power Usage Effectiveness. Defaults to VETCH_DEFAULT_PUE env var or 1.2.

    Returns:
        Carbon emissions in grams CO2e.
    """
    import math

    if pue is None:
        pue = get_default_pue()

    # Defensive handling of NaN and Inf
    intensity = grid_intensity_gco2e_kwh
    if math.isnan(intensity):
        intensity = 0.0
    # Cap extremely high intensity at a reasonable max (e.g. 2000 g/kWh)
    if math.isinf(intensity) or intensity > 2000:
        intensity = 2000.0

    return (energy_wh * pue * intensity) / 1000


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str,
    cache_read_tokens: int | None = None,
    cache_creation_tokens: int | None = None,
) -> tuple[float, float, float, str]:
    """Calculate estimated cost in USD.

    Supports cache-aware pricing: cache_read tokens are discounted
    (typically 90% cheaper), cache_creation tokens may have extra cost.

    Args:
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        model: Model identifier.
        cache_read_tokens: Tokens read from prompt cache (cost savings).
        cache_creation_tokens: Tokens written to prompt cache (extra cost).

    Returns:
        Tuple of (total_cost, input_cost, output_cost, billing_tier).
    """
    resolved_model, known = resolve_model(model)
    _load_registry()
    assert _PRICING is not None

    if known and resolved_model in _PRICING:
        entry = _PRICING[resolved_model]
        rate_in = entry["usd_per_1k_input"]
        rate_out = entry["usd_per_1k_output"]
    else:
        # No pricing for unknown models
        return 0.0, 0.0, 0.0, "none"

    # Cache-aware cost calculation
    # Cache read tokens are charged at a discount (default: 10% of input price)
    # Cache creation tokens are charged at a premium (default: 125% of input price)
    cache_read_discount = entry.get("cache_read_discount", 0.1)
    cache_creation_premium = entry.get("cache_creation_premium", 1.25)

    # Base input tokens (excluding cached tokens)
    effective_input = input_tokens
    cache_cost_adjustment = 0.0

    if cache_read_tokens and cache_read_tokens > 0:
        # Subtract cache read tokens from base input, add discounted cost
        effective_input = max(0, input_tokens - cache_read_tokens)
        cache_cost_adjustment += (cache_read_tokens * rate_in * cache_read_discount) / 1000

    if cache_creation_tokens and cache_creation_tokens > 0:
        # Cache creation tokens cost extra on top of normal input cost
        cache_cost_adjustment += (
            cache_creation_tokens * rate_in * cache_creation_premium
        ) / 1000

    cost_in = (effective_input * rate_in) / 1000 + cache_cost_adjustment
    cost_out = (output_tokens * rate_out) / 1000
    return cost_in + cost_out, cost_in, cost_out, "list"
