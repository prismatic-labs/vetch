#!/usr/bin/env python3
"""Verify Python registry JSON matches packages/vetch-ai-sdk snapshots."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY_REGISTRY = REPO / "src" / "vetch" / "registry"
TS_REGISTRY = REPO / "packages" / "vetch-ai-sdk" / "src" / "registry"
FILES = ("energy.json", "pricing.json", "aliases.json", "wue.json")
SENSING = ("global_averages.json",)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize_json(path: Path) -> bytes:
    return json.dumps(json.loads(path.read_text()), sort_keys=True).encode()


def main() -> int:
    errors: list[str] = []
    for name in FILES:
        py_path = PY_REGISTRY / name
        ts_path = TS_REGISTRY / name
        if not py_path.exists() or not ts_path.exists():
            errors.append(f"missing file for {name}")
            continue
        if _normalize_json(py_path) != _normalize_json(ts_path):
            errors.append(
                f"{name}: content drift (run: cp src/vetch/registry/{name} "
                f"packages/vetch-ai-sdk/src/registry/{name})"
            )

    py_sensing = REPO / "src" / "vetch" / "sensing" / "global_averages.json"
    ts_sensing = REPO / "packages" / "vetch-ai-sdk" / "src" / "sensing" / "global_averages.json"
    if _normalize_json(py_sensing) != _normalize_json(ts_sensing):
        errors.append("global_averages.json: content drift")

    if errors:
        print("AI SDK registry sync check FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("AI SDK registry snapshots match Python registries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
