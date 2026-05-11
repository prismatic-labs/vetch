"""Tests for deterministic audit reports."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from vetch.audit_report import build_audit_report, format_audit_report
from vetch.calculation import METHODOLOGY_VERSION
from vetch.storage import compact_storage, configure_storage, store_event


def _event(
    event_id: str,
    timestamp: datetime,
    input_tokens: int = 500,
    output_tokens: int = 0,
    cost_usd: float = 0.1,
    session_id: str = "sess-1",
) -> dict:
    return {
        "event_id": event_id,
        "methodology_version": METHODOLOGY_VERSION,
        "timestamp": timestamp.isoformat(),
        "model": "gpt-4o",
        "provider": "openai",
        "session_id": session_id,
        "usage": {"text": {"input_tokens": input_tokens, "output_tokens": output_tokens}},
        "estimated_energy_wh": 0.01,
        "estimated_carbon_g": 0.004,
        "estimated_cost_usd": cost_usd,
        "tags": {"feature": "rag-search", "customer": "acme", "env": "prod"},
    }


def test_build_audit_report_from_storage(tmp_path) -> None:
    db_path = tmp_path / "usage.db"
    configure_storage(enabled=True, path=db_path)
    now = datetime.now(timezone.utc)

    for index in range(15):
        store_event(_event(f"event-{index}", now - timedelta(minutes=15 - index)))

    report = build_audit_report(start=now - timedelta(hours=1), end=now + timedelta(hours=1))

    assert report.total_requests == 15
    assert report.total_tokens == 7500
    assert report.data_quality.tagged_fraction == 1.0
    assert report.data_quality.methodology_versions == [METHODOLOGY_VERSION]
    assert any(finding.code == "STALL-001" for finding in report.findings)
    assert report.observed_avoidable_cost_usd > 0
    assert report.projected_monthly_avoidable_cost_usd > 0
    assert any(row.dimension == "feature" and row.value == "rag-search"
               for row in report.breakdowns)


def test_format_audit_report_json_and_markdown(tmp_path) -> None:
    db_path = tmp_path / "usage.db"
    configure_storage(enabled=True, path=db_path)
    now = datetime.now(timezone.utc)
    store_event(_event("event-1", now, output_tokens=100))

    report = build_audit_report(start=now - timedelta(hours=1), end=now + timedelta(hours=1))

    json_output = format_audit_report(report, "json")
    parsed = json.loads(json_output)
    assert parsed["total_requests"] == 1

    markdown_output = format_audit_report(report, "markdown")
    assert "# Vetch Inference Waste Audit" in markdown_output
    assert "## Data Quality" in markdown_output


def test_audit_report_uses_daily_aggregates_after_compaction(tmp_path) -> None:
    db_path = tmp_path / "usage.db"
    configure_storage(enabled=True, path=db_path)
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=2)
    store_event(_event("old-event-1", old, input_tokens=600, output_tokens=100))

    compact_storage(raw_retention_days=1)

    report = build_audit_report(start=old - timedelta(hours=1), end=now + timedelta(hours=1))

    assert report.total_requests == 1
    assert report.total_tokens == 700
    assert report.breakdowns
    assert any("daily aggregates" in warning for warning in report.data_quality.warnings)
    assert not report.findings
