#!/usr/bin/env python3
"""Registry freshness and parity checks.

Two guards against silent registry drift:

1. **Parity** — every model in energy.json must have a pricing.json row and
   vice versa. Catches the "added energy, forgot pricing" mistake that leaves a
   current model with an estimate but no cost.
2. **Staleness** — any pricing row carrying an ``as_of`` date older than
   ``MAX_PRICING_AGE_DAYS`` is reported. Pricing decays; a FinOps tool shipping
   silently-stale prices is a credibility liability. Staleness is a warning by
   default (exit 0) and a hard failure with ``--strict``.

Run in CI; mirror of scripts/sync_ai_sdk_registries.py style.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

REGISTRY = Path(__file__).resolve().parent.parent / "src" / "vetch" / "registry"
MAX_PRICING_AGE_DAYS = 365


def _load(name: str) -> dict[str, object]:
    data = json.loads((REGISTRY / name).read_text())
    return {k: v for k, v in data.items() if not k.startswith("_")}


def check_parity() -> list[str]:
    energy = _load("energy.json")
    pricing = _load("pricing.json")
    errors = []
    for k in sorted(set(energy) - set(pricing)):
        errors.append(f"energy.json has '{k}' but pricing.json does not")
    for k in sorted(set(pricing) - set(energy)):
        errors.append(f"pricing.json has '{k}' but energy.json does not")
    return errors


def check_alias_targets() -> list[str]:
    energy = _load("energy.json")
    aliases = _load("aliases.json")
    missing = sorted({v for v in aliases.values() if v not in energy})
    return [f"aliases.json target '{t}' is not an energy.json key" for t in missing]


def check_staleness() -> list[str]:
    pricing = _load("pricing.json")
    today = dt.date.today()
    warnings = []
    for k, v in sorted(pricing.items()):
        if not isinstance(v, dict):
            continue
        as_of = v.get("as_of")
        if not as_of:
            continue
        try:
            age = (today - dt.date.fromisoformat(str(as_of))).days
        except ValueError:
            warnings.append(f"pricing.json '{k}' has unparseable as_of '{as_of}'")
            continue
        if age > MAX_PRICING_AGE_DAYS:
            warnings.append(
                f"pricing.json '{k}' as_of {as_of} is {age} days old "
                f"(> {MAX_PRICING_AGE_DAYS}); re-verify against the provider page"
            )
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat staleness warnings as failures",
    )
    args = parser.parse_args()

    errors = check_parity() + check_alias_targets()
    warnings = check_staleness()

    for w in warnings:
        print(f"WARNING: {w}")
    for e in errors:
        print(f"ERROR: {e}")

    if errors or (args.strict and warnings):
        print(f"\nFAILED: {len(errors)} parity error(s), {len(warnings)} staleness warning(s)")
        return 1
    print(f"OK: registry parity clean ({len(warnings)} staleness warning(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
