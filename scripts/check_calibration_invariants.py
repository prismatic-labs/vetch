#!/usr/bin/env python3
"""Fail CI if any shipped calibration record contradicts itself.

Identifiability is checked by ``vetch validate-calibration``. This is the prior
question: no record in the repository may be internally inconsistent. It runs
without NumPy or a GPU so it can sit in the lint job.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from vetch.calibration_invariants import (  # noqa: E402
    record_invariants,
)

SEARCH = ("h100-calibration-records/**/*.json", "src/vetch/data/*.json")


def main() -> int:
    checked = failures = warned = 0
    for pattern in SEARCH:
        for path in sorted(REPO.glob(pattern)):
            try:
                record = json.loads(path.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(record, dict) or not record.get("identity"):
                continue
            checked += 1
            violations = record_invariants(record)
            for violation in violations:
                rel = path.relative_to(REPO)
                if violation.severity == "error":
                    failures += 1
                    print(f"FAIL {rel}: {violation}", file=sys.stderr)
                else:
                    warned += 1
                    print(f"warn {rel}: {violation}")
    if failures:
        print(f"\n{failures} invariant error(s) across {checked} record(s)",
              file=sys.stderr)
        return 1
    print(f"Calibration invariants passed: {checked} record(s), {warned} warning(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
