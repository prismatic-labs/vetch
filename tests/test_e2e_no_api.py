"""No-API end-to-end release smoke tests."""

from __future__ import annotations

import json
import urllib.request
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import vetch
from vetch.audit_report import build_audit_report, format_audit_report
from vetch.cli import audit
from vetch.context import get_active_context
from vetch.stats import _reset_session_stats
from vetch.storage import configure_storage, flush_storage, query_events, query_usage


def test_no_api_waste_scan_from_wrap_to_stored_audit(
    monkeypatch,
    capsys,
) -> None:
    """A synthetic stalled agent run produces stored findings and CLI JSON."""
    monkeypatch.delenv("ELECTRICITY_MAPS_API_KEY", raising=False)

    def fail_if_networked(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("No external API calls are allowed in this E2E test")

    monkeypatch.setattr(urllib.request, "urlopen", fail_if_networked)

    with TemporaryDirectory(prefix="vetch-e2e-") as tmpdir:
        configure_storage(enabled=True, path=Path(tmpdir) / "usage.db")
        _reset_session_stats()

        start = datetime.now(timezone.utc) - timedelta(minutes=5)
        try:
            with vetch.Session(
                session_id="e2e-agent-loop",
                tags={"team": "ml-platform"},
                emit=False,
            ) as session:
                for _ in range(15):
                    with vetch.wrap(
                        region="us-east-1",
                        tags={
                            "feature": "agent-research",
                            "customer": "acme",
                            "env": "prod",
                        },
                        emit=False,
                    ) as ctx:
                        active_context = get_active_context()
                        assert active_context is not None
                        active_context.capture(
                            model="gpt-4o",
                            provider="openai",
                            usage={"text": {"input_tokens": 500, "output_tokens": 0}},
                        )

                    assert ctx.event is not None
                    assert ctx.event["session_id"] == "e2e-agent-loop"
                    assert ctx.event["vetch_version"] == "0.5.0"
                    assert ctx.event["tracking_disabled"] is False

                assert session.call_count == 15
                assert session.stall_triggered is True
                assert session.total_input_tokens == 7500
                assert session.total_output_tokens == 0
                assert session.total_cost_usd > 0

            flush_storage()
            end = datetime.now(timezone.utc) + timedelta(minutes=5)

            stored_events = query_events(start=start, end=end)
            assert len(stored_events) == 15
            assert {event["session_id"] for event in stored_events} == {"e2e-agent-loop"}
            assert all(
                event["tags"]["feature"] == "agent-research"
                for event in stored_events
            )

            usage = query_usage(start=start, end=end, tags={"feature": "agent-research"})
            assert usage.total_requests == 15
            assert usage.total_input_tokens == 7500
            assert usage.total_output_tokens == 0
            assert usage.total_cost_usd > 0

            report = build_audit_report(start=start, end=end)
            finding_codes = {finding.code for finding in report.findings}
            assert {"CACHE-001", "RAG-001", "STALL-001"}.issubset(finding_codes)
            assert report.total_requests == 15
            assert report.observed_avoidable_cost_usd > 0
            assert report.projected_monthly_avoidable_cost_usd > 0
            assert any(
                row.dimension == "feature" and row.value == "agent-research"
                for row in report.breakdowns
            )

            markdown = format_audit_report(report, "markdown")
            assert "# Vetch Inference Waste Audit" in markdown
            assert "STALL-001" in markdown

            audit(Namespace(
                format="json",
                window=timedelta(hours=1),
                model=None,
                tags=None,
                stored=True,
                session=False,
            ))
            cli_report = json.loads(capsys.readouterr().out)
            cli_finding_codes = {finding["code"] for finding in cli_report["findings"]}
            assert "STALL-001" in cli_finding_codes
            assert cli_report["total_requests"] == 15
        finally:
            configure_storage(enabled=False)
            _reset_session_stats()
