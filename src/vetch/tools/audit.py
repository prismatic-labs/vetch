"""Registry audit tool for Vetch.

This tool performs automated checks on the Vetch registry to ensure:
1. All aliases point to valid models in energy.json and pricing.json.
2. Energy consumption is consistent with model naming (e.g., mini < large).
3. Pricing and Energy are correlated (detects mis-aliased models).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Registry paths
_REGISTRY_DIR = Path(__file__).parent.parent / "registry"
_ENERGY_PATH = _REGISTRY_DIR / "energy.json"
_PRICING_PATH = _REGISTRY_DIR / "pricing.json"
_ALIASES_PATH = _REGISTRY_DIR / "aliases.json"


def audit_registry() -> bool:
    """Perform a comprehensive audit of the registry files.

    Returns:
        True if the audit passed, False if issues were found.
    """
    print("=== Vetch Registry Audit ===")

    issues = []
    warnings = []

    try:
        with open(_ENERGY_PATH) as f:
            energy = json.load(f)
        with open(_PRICING_PATH) as f:
            pricing = json.load(f)
        with open(_ALIASES_PATH) as f:
            aliases = json.load(f)
    except Exception as e:
        print(f"FAILED: Could not load registry files: {e}")
        return False

    # 1. Alias Integrity
    print("\nChecking aliases...")
    for alias, target in aliases.items():
        if alias == "_comment":
            continue
        if target not in energy:
            issues.append(f"Alias '{alias}' points to missing energy entry '{target}'")
        if target not in pricing:
            issues.append(f"Alias '{alias}' points to missing pricing entry '{target}'")

    # 2. Missing Pricing/Energy Coverage
    print("Checking coverage...")
    for model in energy:
        if model not in pricing:
            issues.append(f"Model '{model}' exists in energy.json but is missing from pricing.json")
    for model in pricing:
        if model not in energy:
            issues.append(f"Model '{model}' exists in pricing.json but is missing from energy.json")

    # 3. Price-to-Energy Correlation (The "Mini Alias" Detector)
    print("Checking price-energy correlation...")
    for model, p_data in pricing.items():
        if model not in energy:
            continue

        e_data = energy[model]

        # Get baseline energy (medium prompt or flat)
        if "prompt_length" in e_data:
            wh_in = e_data["prompt_length"]["medium"]["wh_per_1k_input"]
        else:
            wh_in = e_data["wh_per_1k_input"]

        usd_in = p_data.get("usd_per_1k_input", 0)

        if wh_in > 0 and usd_in > 0:
            ratio = usd_in / wh_in
            # This is a heuristic smoke alarm, not a hard correctness check.
            # Modern pricing no longer tracks energy closely: measured reasoning
            # models can be cheap per token while still energy-heavy. Keep this
            # as a warning so structural registry problems fail the audit, while
            # economics-vs-energy surprises stay visible for human review.
            if wh_in > 0.5 and ratio < 0.005:
                warnings.append(
                    f"Anomaly in '{model}': High energy ({wh_in} Wh/1k) but ultra-low price "
                    f"(${usd_in}/1k). Ratio {ratio:.6f} suggests this might be "
                    f"a mis-aliased model or a genuine price/energy divergence."
                )

    # 4. Naming Consistency
    print("Checking naming consistency...")
    for model, e_data in energy.items():
        # Heuristic: 'mini', 'flash', 'haiku', 'nano' should be Tier 3 or have low energy
        low_cost_keywords = ["mini", "flash", "haiku", "nano"]
        is_low_cost = any(k in model.lower() for k in low_cost_keywords)

        if is_low_cost:
            if "prompt_length" in e_data:
                wh_in = e_data["prompt_length"]["medium"]["wh_per_1k_input"]
            else:
                wh_in = e_data["wh_per_1k_input"]

            # If a "mini" model has > 1.0 Wh/1k, it's probably wrong (unless it's an old frontier)
            if wh_in > 1.0:
                warnings.append(
                    f"Warning: '{model}' contains low-cost keyword but has "
                    f"high energy ({wh_in} Wh/1k)"
                )

    # Report results
    if warnings:
        print(f"\nAudit WARNED with {len(warnings)} heuristic notes:")
        for warning in warnings:
            print(f"  - {warning}")

    if issues:
        print(f"\nAudit FAILED with {len(issues)} issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\nAudit PASSED. Registry is structurally consistent.")
        return True


if __name__ == "__main__":
    import sys
    success = audit_registry()
    sys.exit(0 if success else 1)
