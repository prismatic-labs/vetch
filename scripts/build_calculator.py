#!/usr/bin/env python3
"""Regenerate docs/calculator/index.html from the live registry files.

Run after any change to energy.json, pricing.json, or global_averages.json:
    python scripts/build_calculator.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from vetch.calculation import (
    DEFAULT_PUE,
    DEFAULT_WUE,
    PROMPT_LENGTH_MEDIUM_THRESHOLD,
    PROMPT_LENGTH_SHORT_THRESHOLD,
    PROVIDER_PUE,
    PROVIDER_WUE,
    infer_provider_for_model,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ENERGY_PATH = REPO_ROOT / "src/vetch/registry/energy.json"
PRICING_PATH = REPO_ROOT / "src/vetch/registry/pricing.json"
AVERAGES_PATH = REPO_ROOT / "src/vetch/sensing/global_averages.json"
WUE_PATH = REPO_ROOT / "src/vetch/registry/wue.json"
CALCULATOR_PATH = REPO_ROOT / "docs/calculator/index.html"

# Human-readable labels for cloud regions shown in the dropdown
REGION_LABELS: dict[str, str] = {
    "us-east-1": "US East — Virginia",
    "us-east-2": "US East — Ohio",
    "us-west-1": "US West — N. California",
    "us-west-2": "US West — Oregon",
    "ca-central-1": "Canada — Central",
    "northamerica-northeast1": "Canada — Montréal",
    "sa-east-1": "South America — São Paulo",
    "southamerica-east1": "South America — São Paulo (GCP)",
    "eu-west-1": "Europe — Ireland",
    "eu-west-2": "Europe — London",
    "eu-west-3": "Europe — Paris",
    "eu-central-1": "Europe — Frankfurt",
    "europe-west1": "Europe — Belgium",
    "europe-west2": "Europe — London (GCP)",
    "europe-west3": "Europe — Frankfurt (GCP)",
    "europe-west4": "Europe — Netherlands",
    "europe-north1": "Europe — Finland",
    "ap-northeast-1": "Asia Pacific — Tokyo",
    "ap-northeast-2": "Asia Pacific — Seoul",
    "ap-southeast-1": "Asia Pacific — Singapore",
    "ap-southeast-2": "Asia Pacific — Sydney",
    "ap-south-1": "Asia Pacific — Mumbai",
    "asia-east1": "Asia Pacific — Taiwan (GCP)",
    "asia-southeast1": "Asia Pacific — Singapore (GCP)",
    "asia-northeast1": "Asia Pacific — Tokyo (GCP)",
    "australia-southeast1": "Australia — Sydney",
    "us-central1": "US Central — Iowa (GCP)",
}


def _load_wue_registry() -> dict[str, float]:
    data = json.loads(WUE_PATH.read_text())
    return {k: float(v) for k, v in data.items() if not k.startswith("_")}


def _resolve_provider_wue(provider: str | None, wue_registry: dict[str, float]) -> float:
    """Match calculate_water() provider-level lookup (no region)."""
    if not provider:
        return DEFAULT_WUE
    if provider in wue_registry:
        return wue_registry[provider]
    return PROVIDER_WUE.get(provider, DEFAULT_WUE)


def build_calc_constants() -> dict:
    wue_registry = _load_wue_registry()
    # sorted() makes embed output deterministic: a bare set() iterates in
    # PYTHONHASHSEED-dependent order, which shuffles the JSON key order each
    # run and breaks the "rebuild -> git diff --exit-code" CI gate.
    providers = sorted(set(PROVIDER_PUE.keys()) | set(PROVIDER_WUE.keys()) | {"default"})
    provider_wue = {
        prov: _resolve_provider_wue(prov, wue_registry)
        for prov in providers
        if prov != "default"
    }
    return {
        "prompt_short_threshold": PROMPT_LENGTH_SHORT_THRESHOLD,
        "prompt_medium_threshold": PROMPT_LENGTH_MEDIUM_THRESHOLD,
        "default_pue": DEFAULT_PUE,
        "default_wue": DEFAULT_WUE,
        "provider_pue": PROVIDER_PUE,
        "provider_wue": provider_wue,
    }


def build_models(energy: dict, pricing: dict) -> dict:
    models = {}
    for model, entry in energy.items():
        price = pricing.get(model, {})
        provider = infer_provider_for_model(model) or "default"

        obj: dict = {
            "tier": entry["tier"],
            "arch": entry.get("architecture", "dense"),
            "params": entry.get("total_params_b", 0),
            "active": entry.get("active_params_b", 0),
            "prov": provider,
            "usd_in": price.get("usd_per_1k_input", 0.0),
            "usd_out": price.get("usd_per_1k_output", 0.0),
        }

        tier_threshold = price.get("tier_threshold")
        if tier_threshold is not None:
            obj["tier_th"] = int(tier_threshold)
            tier_mi = price.get("tier_multiplier_input") or price.get("tier_multiplier")
            tier_mo = price.get("tier_multiplier_output") or price.get("tier_multiplier")
            if tier_mi is not None:
                obj["tier_mi"] = float(tier_mi)
            if tier_mo is not None:
                obj["tier_mo"] = float(tier_mo)

        if "prompt_length" in entry:
            pl = entry["prompt_length"]
            obj["pl"] = True
            for cat, key in [("short", "s"), ("medium", "m"), ("long", "l")]:
                if cat in pl:
                    obj[f"wh_in_{key}"] = pl[cat]["wh_per_1k_input"]
                    obj[f"wh_out_{key}"] = pl[cat]["wh_per_1k_output"]
        else:
            obj["wh_in"] = entry["wh_per_1k_input"]
            obj["wh_out"] = entry["wh_per_1k_output"]

        models[model] = obj

    return models


def build_grid_regions(averages: dict) -> dict:
    regions = {"global": {"label": "Global average", "gco2": averages.get("global", 436)}}
    for key, gco2 in averages.get("regions", {}).items():
        label = REGION_LABELS.get(key, key)
        regions[key] = {"label": label, "gco2": gco2}
    return regions


def inject(html: str, models: dict, grid_regions: dict, calc_constants: dict) -> str:
    models_js = "const MODELS = " + json.dumps(models, separators=(",", ":")) + ";"
    grid_js = "const GRID_REGIONS = " + json.dumps(grid_regions, separators=(",", ":")) + ";"
    constants_js = (
        "const CALC_CONSTANTS = "
        + json.dumps(calc_constants, separators=(",", ":"))
        + ";"
    )

    html = re.sub(r"const MODELS = \{.*?\};", lambda _: models_js, html, flags=re.DOTALL)
    html = re.sub(r"const GRID_REGIONS = \{.*?\};", lambda _: grid_js, html, flags=re.DOTALL)
    html = re.sub(
        r"const CALC_CONSTANTS = \{.*?\};",
        lambda _: constants_js,
        html,
        flags=re.DOTALL,
    )
    return html


def main() -> None:
    energy = json.loads(ENERGY_PATH.read_text())
    pricing = json.loads(PRICING_PATH.read_text())
    averages = json.loads(AVERAGES_PATH.read_text())
    html = CALCULATOR_PATH.read_text()

    models = build_models(energy, pricing)
    grid_regions = build_grid_regions(averages)
    calc_constants = build_calc_constants()
    updated = inject(html, models, grid_regions, calc_constants)

    CALCULATOR_PATH.write_text(updated)

    tier1 = sum(1 for v in models.values() if v["tier"] == 1)
    tier3 = sum(1 for v in models.values() if v["tier"] == 3)
    print(f"Updated {CALCULATOR_PATH.relative_to(REPO_ROOT)}")
    print(f"  {len(models)} models ({tier1} Tier 1, {tier3} Tier 3)")
    providers: dict[str, int] = {}
    for m in models:
        p = models[m]["prov"]
        providers[p] = providers.get(p, 0) + 1
    for p, count in sorted(providers.items()):
        pue = PROVIDER_PUE.get(p, DEFAULT_PUE)
        print(f"  {p}: {count} models @ PUE {pue}")


if __name__ == "__main__":
    main()
