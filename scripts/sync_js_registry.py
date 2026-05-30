#!/usr/bin/env python3
"""Sync Python registry JSON files to the JS package.

Run from the repo root:
    python scripts/sync_js_registry.py

Exits non-zero and prints a diff if any file is out of sync (for CI).
Pass --fix to overwrite the JS files.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

SYNC_PAIRS = [
    (
        REPO_ROOT / "src/vetch/registry/energy.json",
        REPO_ROOT / "packages/vetch-ai-sdk/src/registry/energy.json",
    ),
    (
        REPO_ROOT / "src/vetch/registry/pricing.json",
        REPO_ROOT / "packages/vetch-ai-sdk/src/registry/pricing.json",
    ),
    (
        REPO_ROOT / "src/vetch/registry/aliases.json",
        REPO_ROOT / "packages/vetch-ai-sdk/src/registry/aliases.json",
    ),
    (
        REPO_ROOT / "src/vetch/registry/wue.json",
        REPO_ROOT / "packages/vetch-ai-sdk/src/registry/wue.json",
    ),
    (
        REPO_ROOT / "src/vetch/sensing/global_averages.json",
        REPO_ROOT / "packages/vetch-ai-sdk/src/sensing/global_averages.json",
    ),
]


def normalise(path: Path) -> str:
    """Return canonical JSON (sorted keys, 2-space indent) for comparison."""
    return json.dumps(json.loads(path.read_text()), sort_keys=True, indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true", help="Overwrite JS files from Python source")
    args = parser.parse_args()

    out_of_sync: list[tuple[Path, Path]] = []

    for src, dst in SYNC_PAIRS:
        if not src.exists():
            print(f"ERROR: source file missing: {src}", file=sys.stderr)
            return 1

        src_canon = normalise(src)

        if not dst.exists() or normalise(dst) != src_canon:
            out_of_sync.append((src, dst))
            if args.fix:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(src_canon)
                print(f"  updated: {dst.relative_to(REPO_ROOT)}")
            else:
                print(f"  out of sync: {dst.relative_to(REPO_ROOT)}")

    if not out_of_sync:
        print("JS registry is in sync with Python registry.")
        return 0

    if args.fix:
        print(f"\nSynced {len(out_of_sync)} file(s).")
        return 0

    print(
        f"\n{len(out_of_sync)} file(s) out of sync. "
        "Run `python scripts/sync_js_registry.py --fix` to update.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
