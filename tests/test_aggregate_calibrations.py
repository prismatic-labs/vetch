"""Tests for scripts/aggregate_calibrations.py validation gates."""

from __future__ import annotations

import sys
from pathlib import Path

# Scripts live outside src/vetch; add repo root for import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.aggregate_calibrations import (  # noqa: E402
    REQUIRED_IMAGE_RESOLUTION_PX,
    REQUIRED_IMAGE_SET,
    fetch_calibration_issues,
    is_suspect_issue,
    validate_record,
)


def _valid_record() -> dict:
    return {
        "valid": True,
        "invalid_reasons": [],
        "image_set": REQUIRED_IMAGE_SET,
        "image_resolution_px": REQUIRED_IMAGE_RESOLUTION_PX,
        "residuals_structured": False,
        "wh_per_1k_input": 0.001,
        "wh_per_1k_output": 0.002,
        "wh_per_image": 0.0005,
        "samples": 22,
        "environment": {
            "on_ac_power": True,
            "low_power_mode": False,
            "power_mode": "Automatic",
        },
        "fit": {"r2": 0.9, "condition_number": 10},
        "power_sampler": {
            "gpu_power_captured": True,
            "idle_drift_pct": 5.0,
        },
    }


class TestValidateRecord:
    def test_accepts_valid_record(self) -> None:
        assert validate_record(_valid_record()) == []

    def test_rejects_low_power_mode(self) -> None:
        record = _valid_record()
        record["environment"]["low_power_mode"] = True
        reasons = validate_record(record)
        assert any("low_power_mode" in r for r in reasons)

    def test_rejects_low_power_form_field(self) -> None:
        record = _valid_record()
        record["environment"]["power_mode"] = "Low Power"
        reasons = validate_record(record)
        assert any("power_mode" in r for r in reasons)

    def test_rejects_wrong_image_resolution(self) -> None:
        record = _valid_record()
        record["image_resolution_px"] = 512
        reasons = validate_record(record)
        assert any("image_resolution_px" in r for r in reasons)

    def test_rejects_residuals_structured(self) -> None:
        record = _valid_record()
        record["residuals_structured"] = True
        reasons = validate_record(record)
        assert any("residuals_structured" in r for r in reasons)

    def test_rejects_missing_gpu_power(self) -> None:
        record = _valid_record()
        record["power_sampler"]["gpu_power_captured"] = False
        reasons = validate_record(record)
        assert any("gpu_power_captured" in r for r in reasons)

    def test_rejects_high_idle_drift(self) -> None:
        record = _valid_record()
        record["power_sampler"]["idle_drift_pct"] = 20.0
        reasons = validate_record(record)
        assert any("idle_drift" in r for r in reasons)


class TestIssueHandling:
    def test_suspect_detection_uses_labels_not_reactions(self) -> None:
        issue = {
            "labels": [{"name": "calibration"}, {"name": "suspect calibration"}],
            "reactions": {"total_count": 0, "+1": 0},
            "body": "",
        }

        assert is_suspect_issue(issue) is True

    def test_suspect_detection_handles_normal_reactions_object(self) -> None:
        issue = {
            "labels": [{"name": "calibration"}],
            "reactions": {"total_count": 0, "+1": 0},
            "body": "### Suspect run\n\n- [x] My run failed one or more quality gates",
        }

        assert is_suspect_issue(issue) is True

    def test_fetches_all_issue_states_to_preserve_closed_acceptances(self) -> None:
        captured_urls: list[str] = []

        class FakeResponse:
            def __enter__(self) -> FakeResponse:
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def read(self) -> bytes:
                return b"[]"

        def fake_urlopen(req: object, timeout: int) -> FakeResponse:
            captured_urls.append(req.full_url)  # type: ignore[attr-defined]
            assert timeout == 30
            return FakeResponse()

        import scripts.aggregate_calibrations as agg

        original_token = agg._gh_token
        original_urlopen = agg.urllib_request.urlopen
        agg._gh_token = lambda: "token"
        agg.urllib_request.urlopen = fake_urlopen  # type: ignore[assignment]
        try:
            assert fetch_calibration_issues("prismatic-labs/vetch") == []
        finally:
            agg._gh_token = original_token
            agg.urllib_request.urlopen = original_urlopen  # type: ignore[assignment]

        assert captured_urls
        assert "state=all" in captured_urls[0]
