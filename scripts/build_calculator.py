#!/usr/bin/env python3
"""Regenerate docs/calculator/index.html from the live registry files.

Run after any change to energy.json or pricing.json:
    python scripts/build_calculator.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENERGY_PATH = REPO_ROOT / "src/vetch/registry/energy.json"
PRICING_PATH = REPO_ROOT / "src/vetch/registry/pricing.json"
CALCULATOR_PATH = REPO_ROOT / "docs/calculator/index.html"

# Provider PUE from official sustainability reports — mirrors calculation.py
PROVIDER_PUE: dict[str, float] = {
    "openai":    1.12,   # Azure (Microsoft 2024)
    "anthropic": 1.14,   # AWS (Jegham et al.)
    "google":    1.10,   # Google Data Centers 2023
    "aws":       1.14,   # Meta/Llama, Mixtral via AWS
    "deepseek":  1.27,   # DeepSeek own servers (Jegham et al.)
    "default":   1.20,   # Conservative hyperscaler average
}


def infer_provider(model: str) -> str:
    m = model.lower()
    if m.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if m.startswith("claude-"):
        return "anthropic"
    if m.startswith(("gemini-", "gemma-")):
        return "google"
    if m.startswith(("llama-", "mixtral-")):
        return "aws"
    if m.startswith("deepseek-"):
        return "deepseek"
    return "default"


def build_models(energy: dict, pricing: dict) -> dict:
    models = {}
    for model, entry in energy.items():
        price = pricing.get(model, {})
        provider = infer_provider(model)
        pue = PROVIDER_PUE[provider]

        obj: dict = {
            "tier":   entry["tier"],
            "arch":   entry.get("architecture", "dense"),
            "params": entry.get("total_params_b", 0),
            "active": entry.get("active_params_b", 0),
            "pue":    pue,
            "usd_in":  price.get("usd_per_1k_input", 0.0),
            "usd_out": price.get("usd_per_1k_output", 0.0),
        }

        if "prompt_length" in entry:
            pl = entry["prompt_length"]
            obj["pl"] = True
            for cat, key in [("short", "s"), ("medium", "m"), ("long", "l")]:
                if cat in pl:
                    obj[f"wh_in_{key}"]  = pl[cat]["wh_per_1k_input"]
                    obj[f"wh_out_{key}"] = pl[cat]["wh_per_1k_output"]
        else:
            obj["wh_in"]  = entry["wh_per_1k_input"]
            obj["wh_out"] = entry["wh_per_1k_output"]

        models[model] = obj

    return models


def inject(html: str, models: dict) -> str:
    models_js = "const MODELS = " + json.dumps(models, separators=(",", ":")) + ";"

    # Replace MODELS constant
    html = re.sub(
        r"const MODELS = \{.*?\};",
        models_js,
        html,
        flags=re.DOTALL,
    )

    # Replace flat DEFAULT_PUE usages in calculations with per-model d.pue
    # Only touch the two formula lines, not the const declaration itself
    html = re.sub(
        r"(const carbon = \(energy \* )DEFAULT_PUE( \* DEFAULT_GRID\) / 1000;)",
        r"\1d.pue\2",
        html,
    )
    html = re.sub(
        r"(const water = \(energy \* )DEFAULT_PUE( \* DEFAULT_WUE\) / 1000 \* 1000;)",
        r"\1d.pue\2",
        html,
    )

    return html


def main() -> None:
    energy  = json.loads(ENERGY_PATH.read_text())
    pricing = json.loads(PRICING_PATH.read_text())
    html    = CALCULATOR_PATH.read_text()

    models  = build_models(energy, pricing)
    updated = inject(html, models)

    CALCULATOR_PATH.write_text(updated)

    tier1 = sum(1 for v in models.values() if v["tier"] == 1)
    tier3 = sum(1 for v in models.values() if v["tier"] == 3)
    print(f"Updated {CALCULATOR_PATH.relative_to(REPO_ROOT)}")
    print(f"  {len(models)} models ({tier1} Tier 1, {tier3} Tier 3)")
    providers = {}
    for m in models:
        p = infer_provider(m)
        providers[p] = providers.get(p, 0) + 1
    for p, count in sorted(providers.items()):
        print(f"  {p}: {count} models @ PUE {PROVIDER_PUE[p]}")


if __name__ == "__main__":
    main()
