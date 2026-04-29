#!/usr/bin/env python3
"""Regenerate docs/calculator/index.html from the live registry files.

Run after any change to energy.json, pricing.json, or global_averages.json:
    python scripts/build_calculator.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENERGY_PATH    = REPO_ROOT / "src/vetch/registry/energy.json"
PRICING_PATH   = REPO_ROOT / "src/vetch/registry/pricing.json"
AVERAGES_PATH  = REPO_ROOT / "src/vetch/sensing/global_averages.json"
CALCULATOR_PATH = REPO_ROOT / "docs/calculator/index.html"

# Human-readable labels for cloud regions shown in the dropdown
REGION_LABELS: dict[str, str] = {
    "us-east-1":               "US East — Virginia",
    "us-east-2":               "US East — Ohio",
    "us-west-1":               "US West — N. California",
    "us-west-2":               "US West — Oregon",
    "ca-central-1":            "Canada — Central",
    "northamerica-northeast1": "Canada — Montréal",
    "sa-east-1":               "South America — São Paulo",
    "southamerica-east1":      "South America — São Paulo (GCP)",
    "eu-west-1":               "Europe — Ireland",
    "eu-west-2":               "Europe — London",
    "eu-west-3":               "Europe — Paris",
    "eu-central-1":            "Europe — Frankfurt",
    "europe-west1":            "Europe — Belgium",
    "europe-west2":            "Europe — London (GCP)",
    "europe-west3":            "Europe — Frankfurt (GCP)",
    "europe-west4":            "Europe — Netherlands",
    "europe-north1":           "Europe — Finland",
    "ap-northeast-1":          "Asia Pacific — Tokyo",
    "ap-northeast-2":          "Asia Pacific — Seoul",
    "ap-southeast-1":          "Asia Pacific — Singapore",
    "ap-southeast-2":          "Asia Pacific — Sydney",
    "ap-south-1":              "Asia Pacific — Mumbai",
    "asia-east1":              "Asia Pacific — Taiwan (GCP)",
    "asia-southeast1":         "Asia Pacific — Singapore (GCP)",
    "asia-northeast1":         "Asia Pacific — Tokyo (GCP)",
    "australia-southeast1":    "Australia — Sydney",
    "us-central1":             "US Central — Iowa (GCP)",
}

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


def build_grid_regions(averages: dict) -> dict:
    regions = {"global": {"label": "Global average", "gco2": averages.get("global", 436)}}
    for key, gco2 in averages.get("regions", {}).items():
        label = REGION_LABELS.get(key, key)
        regions[key] = {"label": label, "gco2": gco2}
    return regions


def inject(html: str, models: dict, grid_regions: dict) -> str:
    models_js = "const MODELS = " + json.dumps(models, separators=(",", ":")) + ";"
    grid_js   = "const GRID_REGIONS = " + json.dumps(grid_regions, separators=(",", ":")) + ";"

    # Use lambda replacement to avoid re interpreting \u escapes in JSON strings
    html = re.sub(r"const MODELS = \{.*?\};",       lambda _: models_js, html, flags=re.DOTALL)
    html = re.sub(r"const GRID_REGIONS = \{.*?\};", lambda _: grid_js,   html, flags=re.DOTALL)
    return html


def main() -> None:
    energy   = json.loads(ENERGY_PATH.read_text())
    pricing  = json.loads(PRICING_PATH.read_text())
    averages = json.loads(AVERAGES_PATH.read_text())
    html     = CALCULATOR_PATH.read_text()

    models       = build_models(energy, pricing)
    grid_regions = build_grid_regions(averages)
    updated      = inject(html, models, grid_regions)

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
