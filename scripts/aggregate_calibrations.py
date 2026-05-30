#!/usr/bin/env python3
"""Aggregate community Apple Silicon calibration submissions from GitHub Issues.

Pulls all issues labelled 'calibration', extracts and validates the JSON payload
from each, and writes accepted records to data/calibrations.json.

Usage:
    python scripts/aggregate_calibrations.py
    python scripts/aggregate_calibrations.py --repo prismatic-labs/vetch
    python scripts/aggregate_calibrations.py --dry-run   # print without writing

Requires: gh CLI authenticated, or GITHUB_TOKEN env var set.

Output: data/calibrations.json — one record per accepted submission, sorted by
chip family then model. Suspect/invalid runs are written to data/calibrations_suspect.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib import request as urllib_request
from urllib.error import URLError

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUT_PATH = DATA_DIR / "calibrations.json"
SUSPECT_PATH = DATA_DIR / "calibrations_suspect.json"

REQUIRED_IMAGE_SET = "vetch_standard_v1"
REQUIRED_IMAGE_RESOLUTION_PX = 378

# Fields extracted from the issue form (must match label: values in calibration.yml)
_FORM_FIELDS = {
    "chip": "Chip (verbatim from system_profiler)",
    "memory_gb": "Memory (GB)",
    "macos_version": "macOS version",
    "ollama_version": "Ollama version",
    "model": "Model + tag",
    "quantization": "Quantization",
    "backend": "Ollama backend hint",
    "ac_power": "On AC power?",
    "power_mode": "Power mode",
}


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------

def _gh_token() -> str:
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        return token
    try:
        result = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    print(
        "ERROR: No GitHub token found.\n"
        "Set GITHUB_TOKEN or authenticate with: gh auth login",
        file=sys.stderr,
    )
    sys.exit(1)


def fetch_calibration_issues(repo: str) -> list[dict[str, Any]]:
    """Fetch all issues with the 'calibration' label via GitHub REST API."""
    token = _gh_token()
    issues: list[dict[str, Any]] = []
    page = 1
    while True:
        url = (
            f"https://api.github.com/repos/{repo}/issues"
            f"?labels=calibration&state=all&per_page=100&page={page}"
        )
        req = urllib_request.Request(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with urllib_request.urlopen(req, timeout=30) as resp:
                batch = json.loads(resp.read())
        except URLError as e:
            print(f"ERROR fetching issues (page {page}): {e}", file=sys.stderr)
            sys.exit(1)
        if not batch:
            break
        issues.extend(batch)
        page += 1
    return issues


# ---------------------------------------------------------------------------
# Issue body parsing
# ---------------------------------------------------------------------------

def _extract_form_field(body: str, label: str) -> str | None:
    """Extract the value for a GitHub issue form field by its label heading."""
    # Issue forms render as: ### <label>\n\n<value>\n
    pattern = rf"###\s+{re.escape(label)}\s*\n+(.+?)(?=\n###|\Z)"
    m = re.search(pattern, body, re.DOTALL)
    if not m:
        return None
    return m.group(1).strip()


def _extract_json_block(body: str) -> dict[str, Any] | None:
    """Extract the first JSON object from the issue body (from the output_json textarea)."""
    # Try fenced code block first
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", body, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try bare JSON object (no fences)
    m = re.search(r"(\{[^`]*?\})\s*(?=\n###|\Z)", body, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try Gist URL — fetch and parse
    gist_m = re.search(r"https://gist\.github\.com/[^\s)]+", body)
    if gist_m:
        raw_url = gist_m.group(0)
        # Convert gist page URL to raw URL if needed
        if "/raw/" not in raw_url:
            raw_url = raw_url.rstrip("/") + "/raw"
        try:
            with urllib_request.urlopen(raw_url, timeout=15) as resp:
                return json.loads(resp.read())
        except (URLError, json.JSONDecodeError):
            pass

    return None


def parse_issue(issue: dict[str, Any]) -> dict[str, Any] | None:
    """Parse a GitHub issue into a calibration record. Returns None on parse failure."""
    body = issue.get("body") or ""
    number = issue["number"]
    url = issue["html_url"]

    detail_json = _extract_json_block(body)
    if detail_json is None:
        print(f"  #{number}: SKIP — could not parse JSON from body")
        return None

    # Extract form metadata (for cross-validation)
    form: dict[str, str | None] = {}
    for key, label in _FORM_FIELDS.items():
        form[key] = _extract_form_field(body, label)

    # Normalize chip name from form (more reliable than JSON for cross-issue comparison)
    chip_raw = form.get("chip") or detail_json.get("hardware", {}).get("chip_raw", "Unknown")
    chip_family = _normalize_chip_family(chip_raw)

    record: dict[str, Any] = {
        "issue_number": number,
        "issue_url": url,
        "model": detail_json.get("model") or form.get("model"),
        "provider": detail_json.get("provider", "ollama"),
        "wh_per_1k_input": detail_json.get("wh_per_1k_input"),
        "wh_per_1k_output": detail_json.get("wh_per_1k_output"),
        "wh_per_image": detail_json.get("wh_per_image"),
        "tier": detail_json.get("tier", 0),
        "samples": detail_json.get("samples"),
        "image_set": detail_json.get("image_set"),
        "image_resolution_px": detail_json.get("image_resolution_px"),
        "hardware": {
            "chip_raw": chip_raw,
            "chip_family": chip_family,
            "chip_tier": detail_json.get("hardware", {}).get("chip_tier", ""),
            "memory_gb": _parse_int(
                form.get("memory_gb") or str(detail_json.get("hardware", {}).get("memory_gb", 0))
            ),
            "macos_version": (
                form.get("macos_version") or detail_json.get("hardware", {}).get("macos_version")
            ),
        },
        "environment": {
            "ollama_version": (
                form.get("ollama_version")
                or detail_json.get("environment", {}).get("ollama_version")
            ),
            "quantization": form.get("quantization"),
            "backend": (
                form.get("backend") or detail_json.get("environment", {}).get("backend_hint")
            ),
            "on_ac_power": (form.get("ac_power") or "").lower().startswith("yes"),
            "low_power_mode": detail_json.get("environment", {}).get("low_power_mode", False),
            "power_mode": form.get("power_mode"),
        },
        "residuals_structured": detail_json.get("residuals_structured", False),
        "fit": detail_json.get("fit", {}),
        "vlm": detail_json.get("vlm", {}),
        "power_sampler": detail_json.get("power_sampler", {}),
        "valid": detail_json.get("valid", False),
        "invalid_reasons": detail_json.get("invalid_reasons", []),
        "suspect": False,
    }
    return record


def _normalize_chip_family(chip_raw: str) -> str:
    m = re.search(r"M(\d+)", chip_raw, re.IGNORECASE)
    return f"M{m.group(1)}" if m else "unknown"


def _parse_int(s: str | None) -> int | None:
    if not s:
        return None
    m = re.search(r"\d+", s)
    return int(m.group()) if m else None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_record(record: dict[str, Any]) -> list[str]:
    """Return a list of rejection reasons. Empty list = accepted."""
    reasons = []

    if not record.get("valid"):
        reasons.extend(record.get("invalid_reasons") or ["valid=false in detail JSON"])

    if record.get("image_set") != REQUIRED_IMAGE_SET:
        reasons.append(
            f"image_set={record.get('image_set')!r} — only {REQUIRED_IMAGE_SET!r} accepted"
        )

    resolution = record.get("image_resolution_px")
    if resolution is not None and resolution != REQUIRED_IMAGE_RESOLUTION_PX:
        reasons.append(
            f"image_resolution_px={resolution} — only {REQUIRED_IMAGE_RESOLUTION_PX}px accepted"
        )

    if record.get("residuals_structured") is True:
        reasons.append("residuals_structured=true — structured residual pattern detected")

    fit = record.get("fit") or {}
    r2 = fit.get("r2")
    if r2 is not None and r2 < 0.85:
        reasons.append(f"r2={r2:.3f} < 0.85")

    cond = fit.get("condition_number")
    if cond is not None and cond > 30:
        reasons.append(f"condition_number={cond:.1f} > 30")

    for coeff in ("wh_per_1k_input", "wh_per_1k_output"):
        v = record.get(coeff)
        if v is not None and v < 0:
            reasons.append(f"{coeff}={v} is negative")

    if record.get("wh_per_image") is not None and record["wh_per_image"] < 0:
        reasons.append(f"wh_per_image={record['wh_per_image']} is negative")

    env = record.get("environment") or {}
    if not env.get("on_ac_power"):
        reasons.append("not on AC power")

    if env.get("low_power_mode"):
        reasons.append("low_power_mode=true during calibration")

    power_mode = (env.get("power_mode") or "").strip().lower()
    if power_mode == "low power":
        reasons.append("power_mode=Low Power in issue form")

    power_sampler = record.get("power_sampler") or {}
    if not power_sampler.get("gpu_power_captured", True):
        reasons.append("gpu_power_captured=false — GPU energy not measured")

    idle_drift = power_sampler.get("idle_drift_pct")
    if idle_drift is not None and idle_drift > 15:
        reasons.append(f"idle_drift_pct={idle_drift:.1f} > 15%")

    if record.get("samples") is not None and record["samples"] < 8:
        reasons.append(f"samples={record['samples']} < 8 minimum")

    return reasons


def is_suspect_issue(issue: dict[str, Any]) -> bool:
    """Return True if a submission issue was explicitly marked suspect."""
    labels = issue.get("labels") or []
    for label in labels:
        if isinstance(label, dict) and "suspect" in str(label.get("name", "")).lower():
            return True
        if isinstance(label, str) and "suspect" in label.lower():
            return True

    body = issue.get("body") or ""
    return bool(
        re.search(r"\[x\].*suspect", body, re.IGNORECASE) or
        re.search(r"###\s+Suspect run\b.*\[\s*x\s*\]", body, re.IGNORECASE | re.DOTALL)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate Vetch calibration submissions from GitHub Issues"
    )
    parser.add_argument("--repo", default="prismatic-labs/vetch", help="GitHub repo (owner/name)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print results without writing files"
    )
    parser.add_argument(
        "--include-suspect", action="store_true", help="Include suspect runs in main output"
    )
    args = parser.parse_args()

    print(f"Fetching calibration issues from {args.repo}...")
    issues = fetch_calibration_issues(args.repo)
    print(f"Found {len(issues)} issue(s) with 'calibration' label\n")

    accepted: list[dict[str, Any]] = []
    suspect: list[dict[str, Any]] = []

    for issue in issues:
        number = issue["number"]
        title = issue.get("title", "")
        is_suspect_submission = is_suspect_issue(issue)

        print(f"  #{number}: {title}")
        record = parse_issue(issue)
        if record is None:
            continue

        reasons = validate_record(record)
        if reasons or is_suspect_submission:
            record["suspect"] = True
            record["rejection_reasons"] = reasons
            suspect.append(record)
            label = "SUSPECT" if not reasons else "INVALID"
            detail = "; ".join(reasons) if reasons else "submitter flagged as suspect"
            print(f"    → {label}: {detail}")
        else:
            accepted.append(record)
            r2 = record.get("fit", {}).get("r2")
            print(f"    → ACCEPTED  wh_per_image={record.get('wh_per_image')}  r2={r2}")

    # Sort accepted: chip family, then memory, then model
    accepted.sort(key=lambda r: (
        r.get("hardware", {}).get("chip_family", ""),
        r.get("hardware", {}).get("memory_gb") or 0,
        r.get("model") or "",
    ))

    print(f"\nAccepted: {len(accepted)}  Suspect/invalid: {len(suspect)}")

    if args.dry_run:
        print("\n[dry-run] Would write:")
        print(f"  {OUT_PATH} ({len(accepted)} records)")
        print(f"  {SUSPECT_PATH} ({len(suspect)} records)")
        return

    DATA_DIR.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(accepted, indent=2))
    SUSPECT_PATH.write_text(json.dumps(suspect, indent=2))
    print(f"\nWritten: {OUT_PATH}")
    if suspect:
        print(f"Written: {SUSPECT_PATH}")


if __name__ == "__main__":
    main()
