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
    model: str = "gpt-4o",
    tags: dict[str, str] | None = None,
    retry_count: int = 0,
    tool_call_count: int = 0,
    finish_reason: str | None = None,
) -> dict:
    return {
        "event_id": event_id,
        "methodology_version": METHODOLOGY_VERSION,
        "timestamp": timestamp.isoformat(),
        "model": model,
        "provider": "openai",
        "session_id": session_id,
        "usage": {"text": {"input_tokens": input_tokens, "output_tokens": output_tokens}},
        "estimated_energy_wh": 0.01,
        "estimated_carbon_g": 0.004,
        "estimated_cost_usd": cost_usd,
        "tags": (
            {"feature": "rag-search", "customer": "acme", "env": "prod"}
            if tags is None
            else tags
        ),
        "retry_count": retry_count,
        "tool_call_count": tool_call_count,
        "finish_reason": finish_reason,
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
    stall = next(finding for finding in report.findings if finding.code == "STALL-001")
    assert stall.security_signal is True
    assert stall.security_refs == ("OWASP-LLM01", "OWASP-LLM10")
    assert report.observed_avoidable_cost_usd > 0
    assert report.projected_monthly_avoidable_cost_usd > 0
    assert any(row.dimension == "feature" and row.value == "rag-search"
               for row in report.breakdowns)
    # v0.7.0: three-bucket savings fields must be present (may be zero with no cache data)
    assert hasattr(report, "realized_cache_savings_usd")
    assert hasattr(report, "realized_cache_energy_savings_wh")
    assert hasattr(report, "realized_cache_carbon_savings_g")
    assert hasattr(report, "projected_monthly_cache_savings_usd")
    assert hasattr(report, "circuit_breaker_interventions")
    assert hasattr(report, "intervention_cost_at_risk_usd")
    assert report.realized_cache_savings_usd >= 0.0
    assert report.circuit_breaker_interventions >= 0


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
    assert "## Savings & Interventions" in markdown_output

    text_output = format_audit_report(report, "text")
    assert "Savings & Interventions" in text_output
    assert "Realized cache savings" in text_output
    assert "Circuit breaker interventions" in text_output


def test_audit_report_formats_security_refs(tmp_path) -> None:
    db_path = tmp_path / "usage.db"
    configure_storage(enabled=True, path=db_path)
    now = datetime.now(timezone.utc)

    for index in range(15):
        store_event(_event(f"security-event-{index}", now - timedelta(minutes=index)))

    report = build_audit_report(start=now - timedelta(hours=1), end=now + timedelta(hours=1))

    text_output = format_audit_report(report, "text")
    assert "[security signal]" in text_output
    assert "Security refs: OWASP-LLM01, OWASP-LLM10" in text_output

    markdown_output = format_audit_report(report, "markdown")
    assert "### STALL-001 🔒: Agentic Stall Detected" in markdown_output
    assert "- Security refs: OWASP-LLM01, OWASP-LLM10" in markdown_output


def test_savings_survive_compaction(tmp_path) -> None:
    """Regression: cache savings written to daily_usage must not disappear after compaction."""
    db_path = tmp_path / "usage.db"
    configure_storage(enabled=True, path=db_path)
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=2)

    savings_event = _event("savings-old-1", old)
    savings_event["cache_cost_saving_usd"] = 0.25
    savings_event["cache_energy_saving_wh"] = 2.0
    store_event(savings_event)

    compact_storage(raw_retention_days=1)

    report = build_audit_report(start=old - timedelta(hours=1), end=now + timedelta(hours=1))

    assert report.total_requests == 1
    assert abs(report.realized_cache_savings_usd - 0.25) < 1e-9
    assert abs(report.realized_cache_energy_savings_wh - 2.0) < 1e-9


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


def test_premium_model_rightsizing_candidate_fires_for_stable_tagged_workflow(
    tmp_path,
) -> None:
    db_path = tmp_path / "usage.db"
    configure_storage(enabled=True, path=db_path)
    now = datetime.now(timezone.utc)

    for index in range(60):
        store_event(
            _event(
                f"premium-event-{index}",
                now - timedelta(seconds=index),
                input_tokens=820 + (index % 5),
                output_tokens=95 + (index % 3),
                cost_usd=0.001,
                session_id=f"premium-{index}",
                model="gpt-4o",
                tags={"feature": "support-summary", "env": "prod"},
                finish_reason="stop",
            )
        )

    report = build_audit_report(start=now - timedelta(hours=1), end=now + timedelta(hours=1))

    premium = next(f for f in report.findings if f.code == "PREMIUM-001")
    assert premium.severity == "INFO"
    assert premium.title == "Large Model Rightsizing Candidate"
    assert premium.scope == "workflow:feature:support-summary"
    assert premium.observed_avoidable_cost_usd is None
    assert premium.evidence["model_cost_class"] == "premium"
    assert premium.evidence["premium_model"] == "gpt-4o"
    assert premium.evidence["premium_share"] == 1.0
    assert premium.evidence["candidate_models"]
    assert "not a downgrade decision" in premium.evidence["interpretation"]
    assert "eval" in premium.recommended_action.lower()
    assert "Do not auto-reroute" in premium.automation_guidance

    markdown = format_audit_report(report, "markdown")
    assert "Vetch cannot decide whether a smaller model is good enough" in markdown


def test_premium_model_rightsizing_requires_workflow_tags(tmp_path) -> None:
    db_path = tmp_path / "usage.db"
    configure_storage(enabled=True, path=db_path)
    now = datetime.now(timezone.utc)

    for index in range(60):
        store_event(
            _event(
                f"untagged-premium-event-{index}",
                now - timedelta(seconds=index),
                input_tokens=800,
                output_tokens=100,
                session_id=f"untagged-{index}",
                model="gpt-4o",
                tags={},
            )
        )

    report = build_audit_report(start=now - timedelta(hours=1), end=now + timedelta(hours=1))

    assert "PREMIUM-001" not in {finding.code for finding in report.findings}


def test_premium_model_rightsizing_suppresses_noisy_workflow(tmp_path) -> None:
    db_path = tmp_path / "usage.db"
    configure_storage(enabled=True, path=db_path)
    now = datetime.now(timezone.utc)

    for index in range(60):
        store_event(
            _event(
                f"noisy-premium-event-{index}",
                now - timedelta(seconds=index),
                input_tokens=800,
                output_tokens=100,
                session_id=f"noisy-{index}",
                model="gpt-4o",
                tags={"feature": "tool-agent"},
                tool_call_count=1 if index % 2 == 0 else 0,
            )
        )

    report = build_audit_report(start=now - timedelta(hours=1), end=now + timedelta(hours=1))

    assert "PREMIUM-001" not in {finding.code for finding in report.findings}


def test_premium_model_rightsizing_suppresses_unstable_token_shape(tmp_path) -> None:
    db_path = tmp_path / "usage.db"
    configure_storage(enabled=True, path=db_path)
    now = datetime.now(timezone.utc)

    for index in range(60):
        store_event(
            _event(
                f"unstable-premium-event-{index}",
                now - timedelta(seconds=index),
                input_tokens=200 if index % 2 == 0 else 2200,
                output_tokens=100,
                session_id=f"unstable-{index}",
                model="gpt-4o",
                tags={"feature": "mixed-workload"},
            )
        )

    report = build_audit_report(start=now - timedelta(hours=1), end=now + timedelta(hours=1))

    assert "PREMIUM-001" not in {finding.code for finding in report.findings}
