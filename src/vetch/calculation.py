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
import re
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

logger = logging.getLogger(__name__)

METHODOLOGY_VERSION = "1.3"

# Provider labels for endpoints that do not bill OpenAI/vendor list prices.
# Self-hosted: no per-token API charge (cost is a definite 0; you pay hardware,
# captured via energy). Unknown-price: a third-party OpenAI-compatible endpoint
# whose pricing we don't track (leave cost unknown rather than apply OpenAI's).
_SELF_HOSTED_PROVIDERS = frozenset({"self-hosted", "ollama"})
_UNKNOWN_PRICE_PROVIDERS = frozenset({"openai-compatible"})

# Provider labels Vetch recognises for cost / PUE / water / self-hosted handling.
# Used to validate an explicit provider_hint so a typo (e.g. "selfhosted") is
# surfaced loudly rather than silently billed at cloud rates.
KNOWN_PROVIDERS = frozenset(
    {
        "openai",
        "anthropic",
        "google",
        "aws",
        "azure",
        "bedrock",
        "deepseek",
        "vertexai",
        "self-hosted",
        "ollama",
        "openai-compatible",
    }
)

# Providers that report cache-read tokens DISJOINT from input_tokens.
# The cost/energy math (calculate_cost, calculate_energy) expects the OpenAI
# convention where input_tokens already INCLUDES cache-read tokens (cached is a
# subset of prompt_tokens). Anthropic instead reports usage.input_tokens as the
# fresh, uncached count and cache_read_input_tokens separately, with no overlap.
# For these providers we add cache-read back into the billable input before the
# math runs, so the subtraction in calculate_cost/calculate_energy recovers the
# correct fresh-token count. The emitted usage.input_tokens is left untouched so
# it still reconciles with the provider's own dashboard.
#
# MAINTAINER NOTE: membership is keyed on the provider STRING, and Claude reports
# disjoint counts regardless of the gateway serving it. If Claude is ever wired in
# under a different provider string (Bedrock, or Vertex/`vertexai` gaining Claude
# cache extraction), that string MUST be added here or the fresh-input cost will be
# silently zeroed again. openai/google_genai are correctly absent: openai uses the
# inclusive convention and google_genai does not surface cache-read tokens today.
_CACHE_READ_DISJOINT_PROVIDERS = frozenset({"anthropic"})

# Threshold for the tracking-degradation score (see prepare_inference_metrics).
# Calibrated so the flag means "Vetch had to compensate for degraded tracking
# inputs" — it fires for unknown models, prefix/family proxies, estimated usage,
# and missing usage, while a healthy call (exact match, real usage, even an
# honest Tier-3 model) stays clean. The score maxes at 2.0; do not set >= 2.0 or
# the flag becomes unreachable (the bug this replaced: old threshold was 2.5).
TRACKING_DEGRADED_THRESHOLD = 1.0

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




_calibration_cache: dict[tuple[str, str], Any] = {}


def _clear_calibration_cache() -> None:
    """Flush the local-calibration file cache. Call after writing a new calibration."""
    _calibration_cache.clear()


def _effective_text_input_tokens(
    input_tokens: int,
    n_images: int,
    image_input_tokens: int,
    visual_tokens_per_image: int | None,
) -> int:
    """Text-only input tokens for Tier-0 coeffs fit on decoupled prompt counts."""
    in_tokens = max(0, input_tokens)
    if not visual_tokens_per_image or visual_tokens_per_image <= 0:
        return in_tokens
    visual_total = max(0, n_images) * visual_tokens_per_image
    if image_input_tokens > 0:
        visual_total = max(visual_total, int(image_input_tokens))
    return max(0, in_tokens - visual_total)


def _get_local_calibration(provider: str, model: str) -> Any:
    """Return a cached CalibrationResult for (provider, model), or None."""
    key = (provider, model)
    if key not in _calibration_cache:
        try:
            from vetch.calibrate import load_calibration
            from vetch.community_calibrations import lookup_community_calibration

            cal = load_calibration(provider, model)
            if cal is None:
                cal = lookup_community_calibration(provider, model)
            _calibration_cache[key] = cal
        except (ImportError, OSError, ValueError) as e:
            logger.debug("Failed to load local calibration for %s/%s: %s", provider, model, e)
            _calibration_cache[key] = None
    return _calibration_cache[key]


MatchPrecision = Literal["exact", "alias", "prefix", "family", "fallback"]


class ModelMatch(NamedTuple):
    """Result of resolving a model name against the registry.

    Attributes:
        name: The resolved registry key (or the original model on no match).
        known: True if the name resolved to a registry entry (any precision
            except ``fallback``).
        precision: How the match was made. ``exact`` and ``alias`` are
            high-confidence (the entry's tier is trusted as-is); ``prefix`` and
            ``family`` are low-confidence proxies (the reported tier is floored
            to 3 by ``calculate_energy``); ``fallback`` means no match at all.
    """

    name: str
    known: bool
    precision: MatchPrecision


# Deterministic, conservative per-family fallback. When an unknown model can be
# attributed to a provider family but matches no entry, proxy to a representative
# row of that family rather than the provider-agnostic generic fallback. Each
# family declares a large and small representative; the unknown ID is classified
# by tier keyword (see _FAMILY_LARGE_HINTS / _FAMILY_SMALL_HINTS). When the class
# is ambiguous we bias to the larger (higher-energy) row so the fallback never
# silently undercounts energy. All targets must exist in energy.json.
_FAMILY_FALLBACK: dict[str, dict[str, str]] = {
    "openai": {"large": "gpt-4o", "small": "gpt-4o-mini", "default": "gpt-4o"},
    "anthropic": {
        "large": "claude-opus-4-6",
        "small": "claude-haiku-4-5",
        "default": "claude-sonnet-4-6",
    },
    "google": {
        "large": "gemini-3.1-pro",
        "small": "gemini-3-flash",
        "default": "gemini-3.1-pro",
    },
    "aws": {"large": "llama-3-70b", "small": "llama-3.1-8b", "default": "llama-3-70b"},
    "deepseek": {"large": "deepseek-r1", "small": "deepseek-v3", "default": "deepseek-r1"},
}

# Alpha tier hints, matched against hyphen/dot-split tokens (not raw substrings,
# so "mini" doesn't match inside an unrelated word). Parameter sizes ("31b") are
# parsed numerically instead, because substring matching mis-fires ("31b"
# contains "1b"). Large >= 30B, small <= 15B; the ambiguous middle biases large.
_FAMILY_LARGE_HINTS = frozenset({"pro", "ultra", "opus", "max", "large"})
_FAMILY_SMALL_HINTS = frozenset(
    {"flash", "lite", "nano", "mini", "haiku", "small", "tiny"}
)
_PARAM_SIZE_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*b(?![a-z0-9])")
_FAMILY_LARGE_PARAM_B = 30.0
_FAMILY_SMALL_PARAM_B = 15.0


def _classify_family_subtier(model_lower: str) -> str:
    """Classify an unknown model as 'large' / 'small' / 'default' (conservative).

    Bias to 'large' (higher energy) whenever the class is ambiguous, so a family
    proxy never silently undercounts.
    """
    tokens = set(re.split(r"[-_.]+", model_lower))
    sizes = [float(m) for m in _PARAM_SIZE_RE.findall(model_lower)]
    max_size = max(sizes) if sizes else None

    if tokens & _FAMILY_LARGE_HINTS or (max_size is not None and max_size >= _FAMILY_LARGE_PARAM_B):
        return "large"
    if tokens & _FAMILY_SMALL_HINTS or (max_size is not None and max_size <= _FAMILY_SMALL_PARAM_B):
        return "small"
    return "default"


def _family_fallback(model_lower: str) -> ModelMatch | None:
    """Proxy an unknown model to a conservative same-family representative."""
    provider = _infer_provider_from_model(model_lower)
    family = _FAMILY_FALLBACK.get(provider or "")
    if family is None:
        return None
    target = family[_classify_family_subtier(model_lower)]
    # Guard against a missing/failed registry load: never claim a proxy is known
    # if the target row isn't actually present (falls through to generic fallback).
    if _ENERGY is None or target not in _ENERGY:
        return None
    return ModelMatch(target, True, "family")


def resolve_model_match(model: str) -> ModelMatch:
    """Resolve a model name to a registry entry, with match precision.

    Resolution order: exact key, curated alias, algorithmic prefix shorten,
    deterministic family fallback, then no match. Matching is case-insensitive;
    the returned ``name`` preserves the canonical registry key's casing.

    Args:
        model: Original model name from SDK.

    Returns:
        A :class:`ModelMatch`.
    """
    _load_registry()
    assert _ENERGY is not None
    assert _ALIASES is not None

    # Tolerate non-string model identifiers (degraded/mocked paths) rather than
    # crashing the resolver; treat them as unknown.
    if not isinstance(model, str):
        return ModelMatch(model, False, "fallback")

    model_lower = model.lower()

    # 1. Direct match (case-insensitive; registry keys are lowercase)
    if model_lower in _ENERGY:
        return ModelMatch(model_lower, True, "exact")

    # 2. Alias match (curated, high-confidence equivalence)
    if model_lower in _ALIASES:
        resolved = _ALIASES[model_lower]
        if resolved in _ENERGY:
            return ModelMatch(resolved, True, "alias")

    # 3. Prefix matching (gpt-4-0613 -> gpt-4): low-confidence proxy
    parts = model_lower.split("-")
    for i in range(len(parts) - 1, 0, -1):
        prefix = "-".join(parts[:i])
        if prefix in _ENERGY:
            return ModelMatch(prefix, True, "prefix")
        if prefix in _ALIASES:
            resolved = _ALIASES[prefix]
            if resolved in _ENERGY:
                return ModelMatch(resolved, True, "prefix")

    # 4. Deterministic family fallback before the generic one
    family_match = _family_fallback(model_lower)
    if family_match is not None:
        return family_match

    return ModelMatch(model, False, "fallback")


def resolve_model(model: str) -> tuple[str, bool]:
    """Resolve a model name to a registry entry, handling aliases.

    Thin back-compat wrapper over :func:`resolve_model_match`. Note that a
    family-fallback proxy reports ``known=True`` here (it resolved to a real
    entry); callers needing the precision distinction should use
    :func:`resolve_model_match`.

    Args:
        model: Original model name from SDK.

    Returns:
        Tuple of (resolved_model_name, is_known).
    """
    m = resolve_model_match(model)
    return m.name, m.known


def get_conservative_energy() -> dict[str, Any]:
    """Get conservative fallback values for unknown models."""
    return {
        "wh_per_1k_input": 1.4,  # Slightly higher than Claude 3 Opus
        "wh_per_1k_output": 4.2,
        "tier": 3,
        "basis": "Conservative fallback for unknown model",
    }


def _is_reasoning_compute_model(model: str) -> bool:
    """Return True when visible tokens may omit test-time compute."""
    resolved_model, known = resolve_model(model)
    if known:
        _load_registry()
        assert _ENERGY is not None
        architecture = str(_ENERGY.get(resolved_model, {}).get("architecture", "")).lower()
        if architecture == "reasoning":
            return True

    model_l = model.lower()
    if any(hint in model_l for hint in ("thinking", "reasoning", "deepseek-r1")):
        return True
    return any(
        model_l == prefix or model_l.startswith(f"{prefix}-")
        for prefix in ("o1", "o3", "o4")
    )


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


def _int_or_zero(value: Any) -> int:
    """Coerce provider token counts to an int without leaking TypeError paths."""
    if value is None or value == "":
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# Prompt-length bucket thresholds (input tokens).
# Derived from Jegham et al. (2025) experimental design: short=100, medium=1000, long=10000.
# The boundaries sit midway between each scenario's input size.
PROMPT_LENGTH_SHORT_THRESHOLD = 1000   # < 1000 tokens → "short" bucket
PROMPT_LENGTH_MEDIUM_THRESHOLD = 5000  # 1000–4999 tokens → "medium" bucket
                                       # ≥ 5000 tokens → "long" bucket

# Tiktoken availability flag (lazy-loaded)
_TIKTOKEN_AVAILABLE: bool | None = None
_TIKTOKEN_WARNING_ISSUED = False

# Number of accumulated chars after which script-detection sampling stops.
# Language doesn't change mid-response; sampling the first 500 chars is sufficient.
_SCRIPT_SAMPLE_LIMIT = 500


def _detect_content_type_hint(
    hiragana_katakana_chars: int,
    cjk_ideograph_chars: int,
    hangul_chars: int,
    total_chars: int,
) -> str:
    """Determine content type hint from script character counts.

    Used by all streaming providers (Tier 2 fallback when tiktoken is unavailable).
    Only the first ``_SCRIPT_SAMPLE_LIMIT`` characters of a stream are sampled, so
    ``total_chars`` passed here should be the sampled count, not the full stream length.

    Returns:
        ``"ja"`` (Japanese), ``"cjk"`` (Chinese/Korean), or ``"en"`` (all others).
    """
    if total_chars == 0:
        return "en"
    ja_ratio = hiragana_katakana_chars / total_chars
    cjk_ratio = (cjk_ideograph_chars + hangul_chars) / total_chars
    if ja_ratio > 0.10:
        return "ja"
    if cjk_ratio > 0.15:
        return "cjk"
    return "en"


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


CACHE_READ_ENERGY_FACTOR = 0.15
"""Energy fraction for cached input tokens vs. fresh prefill.

Cache reads skip the prefill (KV-cache population), which is the most
compute-intensive part of inference. Only memory-bandwidth is needed to
load the existing KV-cache entries.

Estimated at 10–20% of standard prefill energy (midpoint: 15%).
Source: architectural analysis — no direct empirical measurement available;
treat as Tier 2 (±100%) estimate for the discount itself.
"""


def calculate_energy(
    input_tokens: int,
    output_tokens: int,
    model: str,
    energy_override: dict[str, Any] | None = None,
    cache_read_tokens: int = 0,
    n_images: int = 0,
    image_input_tokens: int = 0,
    _match: ModelMatch | None = None,
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
        cache_read_tokens: Tokens read from prompt cache. These skip prefill
            computation and use only ~15% of standard input energy.

    Returns:
        Tuple of (energy_wh, tier, uncertainty_pct, source, basis, model_known).
    """
    # Clamp negative tokens to 0
    in_tokens = max(0, input_tokens)
    out_tokens = max(0, output_tokens)

    # Resolve once; callers (prepare_inference_metrics) may pass a precomputed
    # match to avoid resolving the same model name twice.
    match = _match if _match is not None else resolve_model_match(model)

    if energy_override:
        wh_in = energy_override["wh_per_1k_input"]
        wh_out = energy_override["wh_per_1k_output"]
        tier = energy_override.get("tier", 1)
        source = energy_override.get("source", "override")
        basis = energy_override.get("basis", "User-provided override")

        visual_tokens_per_image = energy_override.get("visual_tokens_per_image")
        vtok = (
            int(visual_tokens_per_image)
            if isinstance(visual_tokens_per_image, int) and visual_tokens_per_image > 0
            else None
        )
        text_in_tokens = _effective_text_input_tokens(
            in_tokens, n_images, image_input_tokens, vtok
        )

        cache_tokens = min(max(0, cache_read_tokens), text_in_tokens)
        fresh_tokens = text_in_tokens - cache_tokens
        energy_wh = (
            fresh_tokens * wh_in
            + cache_tokens * wh_in * CACHE_READ_ENERGY_FACTOR
            + out_tokens * wh_out
        ) / 1000

        # Add vision-encoder energy for VLM calibrations.
        wh_per_image = energy_override.get("wh_per_image")
        if wh_per_image is not None:
            image_units = float(max(0, n_images))
            if vtok is not None and image_input_tokens > 0:
                token_units = image_input_tokens / vtok
                image_units = max(image_units, token_units)
            if image_units > 0:
                energy_wh += wh_per_image * image_units

        # Add fixed per-request intercept from 4-parameter LS fit (Apple Silicon calibration)
        intercept_wh = energy_override.get("intercept_wh")
        if intercept_wh is not None and intercept_wh > 0:
            energy_wh += intercept_wh

        uncertainty_pct = get_uncertainty_pct(tier)
        # Check if model is known in registry anyway for informational purposes
        known = match.known
        return energy_wh, tier, uncertainty_pct, source, basis, known

    resolved_model, known = match.name, match.known
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
            if input_tokens < PROMPT_LENGTH_SHORT_THRESHOLD:
                category = "short"
            elif input_tokens < PROMPT_LENGTH_MEDIUM_THRESHOLD:
                category = "medium"
            else:
                category = "long"

            # Get coefficients for this prompt length, falling back to
            # medium if the measured category is unavailable (e.g., gpt-4
            # has no "long" measurement in Jegham et al.)
            pl_data = entry["prompt_length"]
            if category not in pl_data:
                category = "medium"
            pl_entry = pl_data[category]
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

        # A prefix/family proxy is not the same model as the entry it matched,
        # so it must not inherit a measured (tier 0/1) confidence it didn't earn.
        # Floor the reported tier to 3 (order-of-magnitude) for proxy matches;
        # exact/alias matches keep the entry's real tier.
        if match.precision in ("prefix", "family") and tier < 3:
            tier = 3
            basis = (
                f"{basis} [Proxy match ({match.precision}): '{model}' is not an "
                f"exact registry entry; confidence downgraded to Tier 3.]"
            )
    else:
        entry = get_conservative_energy()
        wh_in = entry["wh_per_1k_input"]
        wh_out = entry["wh_per_1k_output"]
        tier = entry["tier"]
        basis = entry["basis"]
        source = "fallback"

    # Apply cache read discount: cached tokens skip prefill, use ~15% of normal input energy
    cache_tokens = min(max(0, cache_read_tokens), in_tokens)
    fresh_tokens = in_tokens - cache_tokens
    energy_wh = (
        fresh_tokens * wh_in
        + cache_tokens * wh_in * CACHE_READ_ENERGY_FACTOR
        + out_tokens * wh_out
    ) / 1000
    uncertainty_pct = get_uncertainty_pct(tier)
    return energy_wh, tier, uncertainty_pct, source, basis, known


# Default PUE (Power Usage Effectiveness) for data centers
# Hyperscaler average: ~1.09-1.15 (Google, Azure, AWS sustainability reports)
# Global industry average is ~1.54 (Uptime Institute 2025), but LLM inference
# runs in hyperscaler DCs, not enterprise on-prem. 1.2 is a conservative
# default for "probably a hyperscaler we don't recognize."
DEFAULT_PUE = 1.2

# Provider-specific PUE values from official sustainability reports (Tier 1 data)
# Sources:
#   Google: Google 2026 Environmental Report, FY2025 (fleet-wide 1.09)
#   Azure: Microsoft 2025 Environmental Sustainability Report (newest-gen <1.12)
#   AWS: https://aws.amazon.com/sustainability/data-centers/ (2025 report: 1.14)
#
# Basis caveat: google/aws are fleet-wide averages; azure/openai use Microsoft's
# newest-generation figure (Microsoft does not publish a clean operational fleet
# average). Azure/OpenAI are therefore on a slightly more favorable basis than
# google/aws. Reconciling to a common basis would require an estimated Azure
# fleet PUE (~1.18, tier 2) and is a deliberate methodology change, not a report
# refresh — do not silently swap the value.
PROVIDER_PUE: dict[str, float] = {
    "google": 1.09,      # Google fleet-wide (2026 report, FY2025)
    "vertexai": 1.09,    # Vertex AI runs on Google Cloud
    "azure": 1.12,       # Microsoft newest-gen (2025 report reaffirms <1.12)
    "openai": 1.12,      # OpenAI primarily uses Azure
    "aws": 1.14,         # AWS global average (2025 report)
    "anthropic": 1.14,   # Anthropic uses AWS
    "bedrock": 1.14,     # AWS Bedrock
    "deepseek": 1.27,    # DeepSeek own servers (Jegham et al.)
}

# Documentation sources for transparency
PROVIDER_PUE_SOURCES: dict[str, str] = {
    "google": "Google 2026 Environmental Report (FY2025 fleet-wide PUE)",
    "vertexai": "Google 2026 Environmental Report (FY2025 fleet-wide PUE)",
    "azure": "Microsoft 2025 Environmental Sustainability Report (newest-gen PUE)",
    "openai": "Microsoft 2025 Environmental Sustainability Report (Azure-backed)",
    "aws": "AWS 2025 Sustainability Report (global average PUE)",
    "anthropic": "AWS 2025 Sustainability Report (AWS-backed)",
    "bedrock": "AWS 2025 Sustainability Report",
    "deepseek": "Jegham et al. (2025) DeepSeek datacenter PUE",
}

# Publication/measurement date (ISO) for each provider's PUE figure. Mirrors
# PROVIDER_PUE_SOURCES; consumed by the staleness advisory. None = unknown.
PROVIDER_PUE_AS_OF: dict[str, str | None] = {
    "google": "2026-06-30",
    "vertexai": "2026-06-30",
    "azure": "2025-05-29",
    "openai": "2025-05-29",
    "aws": "2025-01-01",
    "anthropic": "2025-01-01",
    "bedrock": "2025-01-01",
    "deepseek": "2025-01-01",
}


def infer_provider_for_model(model: str) -> str | None:
    """Infer cloud provider from a model name (public API for tooling).

    Returns:
        Provider key for PUE/WUE lookup, or None if unknown.
    """
    return _infer_provider_from_model(model)


def _infer_provider_from_model(model: str) -> str | None:
    """Infer cloud provider from model name patterns.

    Args:
        model: Model identifier (e.g., "gpt-4o", "claude-3-opus")

    Returns:
        Provider key for PUE lookup, or None if unknown.
    """
    model_lower = model.lower()

    # OpenAI models (Azure-backed)
    openai_prefixes = ["gpt-", "o1", "o3", "o4", "text-davinci", "text-embedding"]
    if any(model_lower.startswith(prefix) for prefix in openai_prefixes):
        return "openai"

    # Anthropic models (AWS-backed)
    if model_lower.startswith("claude-"):
        return "anthropic"

    # Google models
    if any(prefix in model_lower for prefix in ["gemini-", "gemma-", "palm-"]):
        return "google"

    # Open-weight models commonly served on AWS (Meta Llama, Mixtral)
    if model_lower.startswith(("llama-", "mixtral-")):
        return "aws"

    # DeepSeek models (DeepSeek own infrastructure)
    if model_lower.startswith("deepseek-"):
        return "deepseek"

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
    cache_creation_1h_tokens: int | None = None,
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
        cache_creation_tokens: Total tokens written to prompt cache (extra cost).
        cache_creation_1h_tokens: Subset of cache_creation_tokens written with a
            1-hour TTL, priced at the model's ``cache_creation_premium_1h``. The
            remaining (5-minute) writes use ``cache_creation_premium``. Defaults to
            None, which prices every write at the 5-minute premium (legacy behavior).

    Returns:
        Tuple of (total_cost, input_cost, output_cost, cache_write_cost, cache_read_cost,
        billing_tier). cache_write_cost: Cost to write tokens to cache (included in total)
        cache_read_cost: Cost for cached token reads (included in total, typically discounted)

    Example:
        >>> # Standard model (no tiers)
        >>> calculate_cost(1000, 500, "gpt-4o")
        (0.0125, 0.005, 0.0075, 0.0, 0.0, 'list')

        >>> # Tiered model (Gemini 2.5 Pro): 300k input, 1k output (threshold pricing)
        >>> # Input: 300k @ $2.50/M (base × 2.0 over 200k threshold) = $0.75
        >>> # Output: 1k @ $10/M (under threshold) = $0.01
        >>> # Total: $0.76
        >>> calculate_cost(300000, 1000, "gemini-2.5-pro")
        (0.76, 0.75, 0.01, 0.0, 0.0, 'list')
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
    # Cache creation tokens: 1.0 = same as input price (OpenAI default),
    # 1.25 = Anthropic premium. Only override if the model explicitly sets it.
    cache_read_discount = entry.get("cache_read_discount", 0.1)
    cache_creation_premium = entry.get("cache_creation_premium", 1.0)
    # 1-hour TTL writes cost more than the default 5-minute writes (Anthropic:
    # 2.0x vs 1.25x input). Fall back to the 5-minute premium when a model does
    # not distinguish the two tiers.
    cache_creation_premium_1h = entry.get("cache_creation_premium_1h", cache_creation_premium)

    # Base input tokens (excluding cached tokens)
    effective_input = input_tokens
    cache_write_cost = 0.0
    cache_read_cost = 0.0

    if cache_read_tokens and cache_read_tokens > 0:
        # Subtract cache read tokens from base input, add discounted cost
        effective_input = max(0, input_tokens - cache_read_tokens)
        cache_read_cost = (cache_read_tokens * rate_in * cache_read_discount) / 1000

    if cache_creation_tokens and cache_creation_tokens > 0:
        # Cache creation tokens cost extra on top of normal input cost. Split the
        # write into 1-hour and 5-minute TTL buckets so each is priced correctly;
        # the 1h count is a subset of the total, clamped defensively.
        create_1h = min(max(0, cache_creation_1h_tokens or 0), cache_creation_tokens)
        create_5m = cache_creation_tokens - create_1h
        cache_write_cost = (
            create_5m * rate_in * cache_creation_premium
            + create_1h * rate_in * cache_creation_premium_1h
        ) / 1000

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
        "model_match",
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
        "cache_energy_saving_wh",
        "cache_cost_saving_usd",
        "cache_carbon_saving_g",
    )

    def __init__(self) -> None:
        self.energy_wh: float | None = None
        self.energy_tier: int = 3
        self.energy_uncertainty_pct: int | None = 1000
        self.energy_source: str = "registry"
        self.energy_basis: str | None = None
        self.model_known: bool = False
        self.model_match: MatchPrecision = "fallback"
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
        self.cache_energy_saving_wh: float | None = None
        self.cache_cost_saving_usd: float | None = None
        self.cache_carbon_saving_g: float | None = None


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
    accumulated_tik_tokens: int = 0,
    content_type_hint: str = "en",
    n_images: int = 0,
    cache_creation_1h_tokens: int | None = None,
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
        cache_creation_1h_tokens: Subset of cache_creation_tokens written with a
            1-hour TTL (Anthropic), priced at the higher 1h write premium. The
            remainder is priced at the 5-minute premium. None/0 keeps the legacy
            behavior of pricing all cache writes at the 5-minute premium.
        existing_warnings: Warnings accumulated earlier in the context lifecycle.

    Returns:
        :class:`InferenceMetrics` with all computed values populated.
    """
    import hashlib
    from datetime import datetime, timezone

    from vetch.sensing.grid import get_carbon_intensity

    metrics = InferenceMetrics()
    metrics.warnings = list(existing_warnings)

    # Auto-load local hardware calibration when no override is provided.
    # Calibrations saved by `vetch calibrate-apple-silicon` (or `vetch calibrate`)
    # are Tier 0 (hardware-measured) and take precedence over the registry.
    # Results are cached in-process (see _calibration_cache) to avoid file I/O
    # on every inference call.
    if energy_override is None:
        cal = _get_local_calibration(provider, model)
        if cal is not None and cal.active:
            if cal.origin == "community":
                source = "community_calibration"
                basis = (
                    f"Community calibration prior ({cal.gpu_name or 'Apple Silicon'})"
                )
            else:
                source = "local_calibration"
                basis = f"Hardware-measured on {cal.gpu_name or 'local GPU'}"
            energy_override = {
                "wh_per_1k_input": cal.wh_per_1k_input,
                "wh_per_1k_output": cal.wh_per_1k_output,
                "tier": cal.tier,
                "source": source,
                "basis": basis,
            }
            if cal.wh_per_image is not None:
                energy_override["wh_per_image"] = cal.wh_per_image
            if cal.visual_tokens_per_image is not None:
                energy_override["visual_tokens_per_image"] = cal.visual_tokens_per_image
            if cal.intercept_wh is not None:
                energy_override["intercept_wh"] = cal.intercept_wh

    # 1. Grid intensity
    grid_intensity = get_carbon_intensity(region)
    metrics.signal_quality = grid_intensity.signal_quality
    metrics.grid_val = grid_intensity.intensity_gco2e_kwh
    if grid_intensity.timestamp:
        ts = datetime.fromtimestamp(grid_intensity.timestamp, tz=timezone.utc)
        metrics.grid_ts = ts.isoformat().replace("+00:00", "Z")

    # 2. Token estimation fallback for streams without usage data
    if (not usage or not usage.get("text")) and (
        accumulated_tik_tokens > 0 or accumulated_chars > 0
    ):
        if accumulated_tik_tokens > 0:
            # Tier 1: tiktoken per-chunk counts (~99% accurate, all scripts)
            estimated_output_tokens = max(1, accumulated_tik_tokens)
            estimated_input_tokens = estimated_output_tokens * 2
            metrics.usage_estimated = True
            metrics.usage_estimation_method = "tiktoken"
            metrics.warnings.append(
                f"Token usage estimated from tiktoken ({accumulated_tik_tokens} output tokens). "
                f"Energy uncertainty floored at ±50%."
            )
        else:
            # Tier 2: script-aware char ratio
            if content_type_hint == "ja":
                _ratio = 1.7
            elif content_type_hint == "cjk":
                _ratio = 1.5
            else:
                _ratio = 4.0
            estimated_output_tokens = max(1, int(accumulated_chars / _ratio))
            estimated_input_tokens = estimated_output_tokens * 2
            metrics.usage_estimated = True
            metrics.usage_estimation_method = "char_ratio"
            metrics.warnings.append(
                f"Token usage estimated from {accumulated_chars} chars "
                f"(~{_ratio} chars/token, {content_type_hint} content). "
                f"Energy uncertainty floored at ±50%."
            )
        usage = {
            "text": {
                "input_tokens": estimated_input_tokens,
                "output_tokens": estimated_output_tokens,
                "total_tokens": estimated_input_tokens + estimated_output_tokens,
            }
        }

    metrics.usage = usage

    # 3. Energy / carbon / cost calculations
    if usage and usage.get("text"):
        text = usage["text"]
        if text:
            in_tokens = _int_or_zero(text.get("input_tokens"))
            out_tokens = _int_or_zero(text.get("output_tokens"))
            image_input_tokens = 0
            image = usage.get("image")
            if isinstance(image, dict):
                image_input_tokens = _int_or_zero(image.get("input_tokens"))

            # Include reasoning tokens (o1/o3 thinking, Gemini thinking).
            # These are generated (decode), so they count as output for energy.
            # Note: OpenAI's provider subtracts reasoning from text.output_tokens
            # to avoid double-counting (completion_tokens includes reasoning).
            # GenAI's candidates_token_count does NOT include thought tokens,
            # so they are additive here.
            reasoning_output_tokens = 0
            if usage.get("reasoning"):
                reasoning = usage["reasoning"]
                if reasoning:
                    reasoning_output_tokens = _int_or_zero(reasoning.get("output_tokens"))
                    out_tokens += reasoning_output_tokens

            if _is_reasoning_compute_model(model) and reasoning_output_tokens == 0:
                metrics.warnings.append(
                    "Reasoning-capable model did not expose reasoning/thinking "
                    "tokens. Vetch records request latency, but energy remains "
                    "visible-token based until a calibrated time-power profile is "
                    "available."
                )

            _cache_tokens = int(cache_read_tokens) if cache_read_tokens else 0
            # Normalize the billable input to the OpenAI convention the cost/energy
            # math assumes (input_tokens includes cache reads). Providers that report
            # cache reads disjoint from input (e.g. Anthropic) would otherwise have
            # their fresh-input cost zeroed out whenever cache_read >= input_tokens,
            # which is the norm for agentic traffic. The emitted usage is untouched.
            billable_in_tokens = in_tokens
            if provider in _CACHE_READ_DISJOINT_PROVIDERS and _cache_tokens > 0:
                billable_in_tokens = in_tokens + _cache_tokens
            # Resolve once and reuse for both the energy calc and the event field.
            match = resolve_model_match(model)
            # Record how the model name was resolved so downstream can flag
            # prefix/family proxies as low-confidence (see ModelMatch).
            metrics.model_match = match.precision
            (
                metrics.energy_wh,
                metrics.energy_tier,
                metrics.energy_uncertainty_pct,
                metrics.energy_source,
                metrics.energy_basis,
                metrics.model_known,
            ) = calculate_energy(
                billable_in_tokens,
                out_tokens,
                model,
                cast("dict[str, Any]", energy_override),
                cache_read_tokens=_cache_tokens,
                n_images=n_images,
                image_input_tokens=image_input_tokens,
                _match=match,
            )

            baseline_energy_wh: float | None = None

            # Compute cache energy saving vs. uncached baseline.
            # n_images is passed so image energy cancels symmetrically on both sides.
            if _cache_tokens > 0 and metrics.energy_wh is not None:
                (baseline_energy_wh, *_) = calculate_energy(
                    billable_in_tokens,
                    out_tokens,
                    model,
                    cast("dict[str, Any]", energy_override),
                    cache_read_tokens=0,
                    n_images=n_images,
                    image_input_tokens=image_input_tokens,
                )
                if baseline_energy_wh is not None:
                    metrics.cache_energy_saving_wh = baseline_energy_wh - metrics.energy_wh

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
                if baseline_energy_wh is not None and metrics.carbon_g is not None:
                    baseline_carbon_g, *_ = calculate_carbon(
                        baseline_energy_wh,
                        metrics.grid_val,
                        model=model,
                        provider_hint=provider,
                    )
                    metrics.cache_carbon_saving_g = max(
                        0.0,
                        baseline_carbon_g - metrics.carbon_g,
                    )

            if provider in _SELF_HOSTED_PROVIDERS:
                # A self-hosted model has no per-token API price (you pay for the
                # hardware, captured via energy), so cost is a definite 0.
                metrics.cost_usd = 0.0
                metrics.cost_in_usd = 0.0
                metrics.cost_out_usd = 0.0
                metrics.billing_tier = "self-hosted"
            elif provider in _UNKNOWN_PRICE_PROVIDERS:
                # An OpenAI-compatible third-party endpoint (vLLM/TGI on a public
                # host, OpenRouter, Together, ...) does NOT bill OpenAI's rates.
                # We don't know its price, so leave cost unknown rather than wrong.
                metrics.cost_usd = None
                metrics.cost_in_usd = None
                metrics.cost_out_usd = None
                metrics.billing_tier = "unknown"
            else:
                (
                    metrics.cost_usd,
                    metrics.cost_in_usd,
                    metrics.cost_out_usd,
                    metrics.cost_cache_write_usd,
                    metrics.cost_cache_read_usd,
                    metrics.billing_tier,
                ) = calculate_cost(
                    billable_in_tokens,
                    out_tokens,
                    model,
                    cache_read_tokens=cache_read_tokens,
                    cache_creation_tokens=cache_creation_tokens,
                    cache_creation_1h_tokens=cache_creation_1h_tokens,
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

            # Compute cache cost saving vs. uncached baseline.
            # Both sides scaled by price_multiplier so the comparison is apples-to-apples:
            # metrics.cost_usd already includes the multiplier; apply it to uncached_cost too.
            if _cache_tokens > 0 and metrics.cost_usd is not None:
                (uncached_cost, *_) = calculate_cost(
                    billable_in_tokens,
                    out_tokens,
                    model,
                    cache_read_tokens=0,
                    cache_creation_tokens=cache_creation_tokens,
                    cache_creation_1h_tokens=cache_creation_1h_tokens,
                )
                if uncached_cost is not None:
                    adjusted_uncached = uncached_cost * price_multiplier
                    metrics.cache_cost_saving_usd = max(0.0, adjusted_uncached - metrics.cost_usd)

    # 3b. Floor energy uncertainty when token counts are estimated.
    # Only applied when energy_wh is non-zero — a zero-energy result has no meaningful
    # uncertainty to floor (50% of 0 Wh = 0 Wh, but it pollutes dashboard filters).
    if (
        metrics.usage_estimated
        and metrics.energy_uncertainty_pct is not None
        and metrics.energy_wh
    ):
        metrics.energy_uncertainty_pct = max(metrics.energy_uncertainty_pct, 50)

    # 4. Tracking degradation score
    grid_quality_score = {"live": 0.0, "delayed": 1.0, "blind": 2.0, "unknown": 3.0}
    score = (
        (0.0 if metrics.model_known else 0.6)
        + (metrics.energy_tier / 3.0) * 0.6
        + (metrics.pue_tier / 3.0) * 0.2
        + (grid_quality_score.get(metrics.signal_quality, 3.0) / 3.0) * 0.2
        + (0.4 if metrics.usage_estimated else 0.0)
        # A prefix/family proxy is a lower-confidence model match than an exact
        # or curated-alias hit, so it adds to the degradation score.
        + (0.4 if metrics.model_match in ("prefix", "family") else 0.0)
    )
    metrics.tracking_degraded = score > TRACKING_DEGRADED_THRESHOLD

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
