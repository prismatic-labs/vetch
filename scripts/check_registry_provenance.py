#!/usr/bin/env python3
"""Provenance guard for every coefficient registry.

The rule this enforces, learned from the WUE registry: a coefficient that
reaches a user must either cite a source or be absent. A plausible-looking
number with nothing behind it is worse than a null, because a null is visibly
missing and a fabricated coefficient is not.

Three checks:

1. **WUE entries are fully attributed** — every entry in wue.json carries
   ``value``, ``tier``, ``boundary``, ``accounting`` and a ``basis`` naming a
   retrievable source with a URL.
2. **PUE attributions are classified** — every PROVIDER_PUE key has a source
   string, an as-of date, and a PROVIDER_PUE_TIER entry. Tier 2 (the figure is
   published but the host attribution is assumed) must say so in its source
   string, so the assumption travels with the number into events and dashboards.
3. **No undocumented DEFAULT_\\* coefficient** — a module-level default is
   allowed only where the published spread is narrow enough that a wrong pick
   stays inside the metric's stated uncertainty. Any new DEFAULT_* constant in
   calculation.py must be listed in ALLOWED_DEFAULTS with that justification.

Run in CI; mirror of scripts/check_registry_freshness.py style.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "src" / "vetch" / "registry"
CALCULATION = ROOT / "src" / "vetch" / "calculation.py"

WUE_REQUIRED = ("value", "tier", "boundary", "accounting", "basis")

# A default coefficient is permitted only with a written spread justification.
ALLOWED_DEFAULTS = {
    "DEFAULT_PUE": (
        "Published PUE spans 1.09-1.27 (~15% worst case), inside the +/-50% "
        "band already claimed for carbon."
    ),
}


def check_wue(errors: list[str]) -> None:
    data = json.loads((REGISTRY / "wue.json").read_text())
    for key, entry in data.items():
        if key.startswith("_"):
            continue
        if not isinstance(entry, dict):
            errors.append(f"wue.json[{key}]: expected an object, got {type(entry).__name__}")
            continue
        for field in WUE_REQUIRED:
            if entry.get(field) in (None, ""):
                errors.append(f"wue.json[{key}]: missing required field '{field}'")
        if "uncertainty_pct" in entry:
            errors.append(
                f"wue.json[{key}]: carries 'uncertainty_pct'. The registry states "
                "no uncertainty band — publishers report a WUE, not a band around "
                "it. Only an explicit wue_override may carry one."
            )
        basis = str(entry.get("basis") or "")
        if basis and "http" not in basis:
            errors.append(f"wue.json[{key}]: basis cites no retrievable URL")
        tier = entry.get("tier")
        if tier not in (1, 2, 3):
            errors.append(f"wue.json[{key}]: tier {tier!r} is not 1, 2 or 3")
        if tier == 3:
            # Tier 3 is a broader figure standing in for a narrower one, so the
            # basis has to name the substitution. A region-scoped key must say
            # what was applied to it; a provider-level key must declare that it
            # is an average rather than a facility measurement.
            region_scoped = "-" in key
            needle = "applied to" if region_scoped else "average"
            if needle not in basis.lower():
                errors.append(
                    f"wue.json[{key}]: tier 3 basis must name the substitution "
                    f"(expected {needle!r})"
                )


def _literal_dicts(path: Path, names: set[str]) -> dict[str, object]:
    """Read module-level dict literals without importing the module.

    Deliberately not an import: executing calculation.py to lint it makes the
    result depend on __pycache__ freshness, which bit this script during
    development (a restored file with an unchanged byte length validated
    against stale bytecode). Parsing is deterministic.
    """
    tree = ast.parse(path.read_text())
    found: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id in names and node.value:
                try:
                    found[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    pass
    return found


def check_pue(errors: list[str]) -> None:
    names = {
        "PROVIDER_PUE",
        "PROVIDER_PUE_SOURCES",
        "PROVIDER_PUE_AS_OF",
        "PROVIDER_PUE_TIER",
    }
    parsed = _literal_dicts(CALCULATION, names)
    missing = names - set(parsed)
    if missing:
        errors.append(f"calculation.py: could not parse {', '.join(sorted(missing))}")
        return

    pue = parsed["PROVIDER_PUE"]
    sources = parsed["PROVIDER_PUE_SOURCES"]
    as_of = parsed["PROVIDER_PUE_AS_OF"]
    tiers = parsed["PROVIDER_PUE_TIER"]

    for key in pue:
        if not sources.get(key):
            errors.append(f"PROVIDER_PUE[{key}]: no entry in PROVIDER_PUE_SOURCES")
        if key not in as_of:
            errors.append(f"PROVIDER_PUE[{key}]: no entry in PROVIDER_PUE_AS_OF")
        if key not in tiers:
            errors.append(
                f"PROVIDER_PUE[{key}]: no entry in PROVIDER_PUE_TIER "
                "(an unclassified attribution is never treated as disclosed)"
            )
            continue
        if tiers[key] not in (1, 2):
            errors.append(f"PROVIDER_PUE_TIER[{key}]: {tiers[key]!r} is not 1 or 2")
        # Tier and source text must agree in both directions. Checking only
        # one direction lets an entry be silently promoted to "disclosed" while
        # its own source string still describes an assumption.
        source_text = str(sources.get(key, "")).lower()
        names_assumption = "assum" in source_text
        if tiers[key] == 2 and not names_assumption:
            errors.append(
                f"PROVIDER_PUE_SOURCES[{key}]: tier 2 must name the assumption "
                "in its source string so it travels with the number"
            )
        if tiers[key] == 1 and names_assumption:
            errors.append(
                f"PROVIDER_PUE_TIER[{key}]: tier 1 claims an operator-disclosed "
                "attribution, but its source string still describes an "
                "assumption. Demote to 2 or remove the assumption."
            )

    for key in set(tiers) - set(pue):
        errors.append(f"PROVIDER_PUE_TIER[{key}]: no matching PROVIDER_PUE value")


def check_defaults(errors: list[str]) -> None:
    text = CALCULATION.read_text()
    found = set(re.findall(r"^(DEFAULT_[A-Z_]+)\s*[:=]", text, flags=re.MULTILINE))
    for name in sorted(found - set(ALLOWED_DEFAULTS)):
        errors.append(
            f"{name}: undocumented default coefficient. Either justify the "
            "published spread in ALLOWED_DEFAULTS or return None when nothing "
            "is sourced."
        )


def main() -> int:
    errors: list[str] = []
    check_wue(errors)
    check_pue(errors)
    check_defaults(errors)

    if errors:
        print("Registry provenance check FAILED:\n")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("Registry provenance check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
