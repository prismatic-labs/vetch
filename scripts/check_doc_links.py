#!/usr/bin/env python3
"""Verify that relative Markdown links in the README and docs resolve.

Guards against link rot when files are renamed or moved. Only relative links
are checked; http(s) and mailto links, and pure in-page anchors, are skipped.
An anchor on a relative link (``foo.md#section``) is checked as far as the file
existing — anchor targets within the file are not validated.

Runs in CI (the ``lint`` job) and locally:

    python scripts/check_doc_links.py

Exits non-zero if any link points at a missing file, printing each break as
``source -> target``. Stdlib only, no third-party dependencies.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Repo root is the parent of this script's directory.
ROOT = Path(__file__).resolve().parent.parent

# Files to scan: the top-level README plus every Markdown file under docs/.
SCAN_GLOBS = ["README.md", "docs/**/*.md"]

# [text](target) — capture the target. Skips images? No: image links matter too,
# but they use the same syntax with a leading '!', which we allow through since a
# broken image path is also worth catching.
LINK_RE = re.compile(r"\]\(([^)]+)\)")


def _is_external(target: str) -> bool:
    return target.startswith(("http://", "https://", "mailto:"))


def iter_markdown_files() -> list[Path]:
    files: list[Path] = []
    for pattern in SCAN_GLOBS:
        files.extend(sorted(ROOT.glob(pattern)))
    return files


def check_file(md_file: Path) -> list[str]:
    """Return a list of broken-link messages for one file (empty if all resolve)."""
    broken: list[str] = []
    text = md_file.read_text(encoding="utf-8")
    for raw_target in LINK_RE.findall(text):
        target = raw_target.strip()
        # Strip an optional Markdown link title: [x](path "title")
        if " " in target:
            target = target.split(" ", 1)[0]
        if not target or target.startswith("#") or _is_external(target):
            continue
        # Drop any in-page anchor; we only verify the file exists.
        path_part = target.split("#", 1)[0]
        if not path_part:
            continue
        resolved = (md_file.parent / path_part).resolve()
        if not resolved.exists():
            rel_source = md_file.relative_to(ROOT)
            broken.append(f"{rel_source} -> {target}")
    return broken


def main() -> int:
    broken: list[str] = []
    files = iter_markdown_files()
    for md_file in files:
        broken.extend(check_file(md_file))

    if broken:
        print("Broken relative Markdown links:\n")
        for entry in broken:
            print(f"  {entry}")
        print(f"\n{len(broken)} broken link(s) across {len(files)} file(s).")
        return 1

    print(f"All relative Markdown links resolve ({len(files)} file(s) scanned).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
