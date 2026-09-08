"""GPU-free calibration record validator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vetch.calibrate import CalibrationResult
from vetch.calibrate_validate import (
    check_json_schema,
    format_report,
    is_batched_measurement,
    validate_calibration_file,
    validate_calibration_record,
)
from vetch.calibration_store import CalibrationIdentity, build_record

REPO = Path(__file__).resolve().parent.parent
RECORDS_DIR = REPO / "h100-calibration-records"


def _idn(**kw: object) -> CalibrationIdentity:
    base: dict[str, object] = dict(
        provider="self-hosted",
        model="google/gemma-4-31B-it",
        gpu="h100-sxm-80gb",
        serving_engine="vllm",
        precision="bf16",
    )
    base.update(kw)
    return CalibrationIdentity(**base)  # type: ignore[arg-type]


def _result(**kw: object) -> CalibrationResult:
    base: dict[str, object] = dict(
        model="google/gemma-4-31B-it",
        provider="self-hosted",
        wh_per_1k_input=None,
        wh_per_1k_output=0.5,
        tier=0,
        samples=22,
        gpu_name="NVIDIA H100 80GB HBM3",
        active=True,
    )
    base.update(kw)
    return CalibrationResult(**base)  # type: ignore[arg-type]


def test_schema_only_accepts_v2_build_record() -> None:
    rec = build_record(_result(), _idn(), {"gpu_known": True}, timestamp=1.0)
    errors = check_json_schema(rec)
    assert errors == []
    report = validate_calibration_record(rec, identifiability=False)
    assert report["ok"] is True
    assert report["schema_version"] == 2
    assert report["identifiability"] is None


def test_v1_is_not_checked_against_v2_schema() -> None:
    rec = {
        "schema_version": 1,
        "identity": {
            "provider": "self-hosted",
            "model": "m",
            "gpu": "h100-sxm-80gb",
            "serving_engine": "vllm",
            "precision": "bf16",
        },
        "coefficients": {"wh_per_1k_input": 0.0, "wh_per_1k_output": 0.5},
        "provenance": {"gpu_known": True, "samples": 11, "raw_run_table": []},
        "active": False,
    }
    report = validate_calibration_record(rec, identifiability=False)
    assert report["schema_errors"] == []
    assert any("schema_version 1" in w for w in report["warnings"])


def test_schema_only_cli_path(tmp_path: Path) -> None:
    rec = build_record(_result(), _idn(), {"gpu_known": True}, timestamp=1.0)
    path = tmp_path / "rec.json"
    path.write_text(json.dumps(rec))
    report = validate_calibration_file(path, identifiability=False)
    assert report["ok"] is True
    text = format_report(report)
    assert "schema_version: 2" in text


def test_identifiability_on_batch1_file_no_gpu() -> None:
    pytest.importorskip("numpy")
    from vetch.calibrate_v2 import STATUS_IDENTIFIED, STATUS_NOT_IDENTIFIED

    files = list(RECORDS_DIR.glob("*.json"))
    assert files, "expected top-level batch-1 records"
    # A batch-1 table: its v1 input CI is entirely negative and v2 keeps decode
    # while withholding prefill. Match on the migration note rather than the
    # model, because batch 2 measured the same model with a prefill arm and DOES
    # identify that axis, which is the opposite of what this asserts.
    rec = None
    path = None
    for candidate in sorted(files):
        data = json.loads(candidate.read_text())
        if data.get("identity", {}).get("model") != "Qwen/Qwen2.5-7B-Instruct":
            continue
        if not (data.get("provenance") or {}).get("schema_migration"):
            continue
        rec = data
        path = candidate
        break
    assert rec is not None and path is not None
    report = validate_calibration_file(path, identifiability=True)
    ident = report["identifiability"]
    assert ident is not None
    assert ident["status"]["wh_per_1k_output"] == STATUS_IDENTIFIED
    assert ident["status"]["wh_per_1k_input"] == STATUS_NOT_IDENTIFIED
    assert ident["coefficients"]["wh_per_1k_input"] is None
    assert ident["active"] is True
    # residuals_structured is a flag, not a reject.
    assert report["ok"] is True
    assert ident["residuals_structured"] in (True, False)
    assert ident["estimator"] == "nested_ols"


def test_batched_record_is_not_rejected_by_nested_ols() -> None:
    """--batched files are a one-row aggregate; nested OLS must not fail them."""
    batched_dir = RECORDS_DIR / "qwen38-vs-qwen25" / "remote"
    files = [
        p for p in batched_dir.glob("*.json")
        if "c32" in p.name and "qwen3.8" in p.name
    ]
    assert files, "expected a batched concurrency-32 record"
    path = files[0]
    rec = json.loads(path.read_text())
    assert is_batched_measurement(rec)
    report = validate_calibration_file(path, identifiability=True)
    ident = report["identifiability"]
    assert ident is not None
    assert ident["estimator"] == "amortization_1_over_c"
    assert ident["published_rung"] == "batched"
    assert ident["status"]["wh_per_1k_output"] == "identified"
    assert ident["status"]["wh_per_1k_input"] == "not_identified"
    assert ident["coefficients"]["wh_per_1k_output"] is not None
    assert ident["coefficients"]["wh_per_1k_output"] > 0
    assert ident["active"] is True
    assert report["ok"] is True
    assert "admitted rows" not in " ".join(ident.get("rejection_reasons") or [])
    text = format_report(report)
    assert "estimator: amortization_1_over_c" in text
    assert rec["coefficients"].get("wh_per_image")
    assert any("leftover" in w for w in report["warnings"])


def test_nested_batch1_record_is_not_batched() -> None:
    files = list(RECORDS_DIR.glob("*.json"))
    assert files
    rec = json.loads(files[0].read_text())
    assert is_batched_measurement(rec) is False


def test_batched_lane_needs_corroboration_not_a_label() -> None:
    """Entry to the batched lane cannot rest on a self-declared string.

    Keying off the free-text measurement_basis let a hand-edited record with
    an empty run table claim an identified decode coefficient.
    """
    from vetch.calibrate_validate import batched_corroboration

    forged = {
        "identity": {"concurrency": None},
        "provenance": {"calibration_mode": "batched", "raw_run_table": []},
        "coefficients": {"wh_per_1k_output": 0.0001},
    }
    qualifies, failures = batched_corroboration(forged)
    assert qualifies is False
    assert any("raw_run_table has 0 rows" in f for f in failures)
    assert any("amortization_fit" in f for f in failures)


def test_real_batched_records_still_qualify() -> None:
    """All twelve shipped batched records satisfy the conjunction."""
    from vetch.calibrate_validate import batched_corroboration

    remote = RECORDS_DIR / "qwen38-vs-qwen25" / "remote"
    files = sorted(remote.glob("self-hosted*.json"))
    assert len(files) == 12
    for path in files:
        qualifies, failures = batched_corroboration(json.loads(path.read_text()))
        assert qualifies is True, (path.name, failures)


def test_stored_active_must_match_computed_active() -> None:
    """A record whose claim outruns its data is not ok, however well formed."""
    from vetch.calibrate_validate import validate_calibration_record

    record = {
        "schema_version": 2,
        "active": True,
        "identity": {"concurrency": None},
        "coefficients": {"wh_per_1k_output": 0.5},
        "provenance": {"raw_run_table": [], "idle_watts_before": 124.0},
    }
    report = validate_calibration_record(record)
    assert report["active_mismatch"] == {"stored": True, "computed": False}
    assert report["ok"] is False


def test_idle_drift_is_flagged_not_rejected() -> None:
    """Drift is surfaced like residuals_structured: visible, never fatal."""
    from vetch.calibrate_validate import idle_drift_flags

    assert idle_drift_flags({"provenance": {"idle_drift_pct": 3.0}}) == (3.0, [])
    drift, warns = idle_drift_flags({"provenance": {"idle_drift_pct": 16.41}})
    assert drift == 16.41
    assert warns and "Flag, not a reject" in warns[0]
    _drift, warns_mid = idle_drift_flags({"provenance": {"idle_drift_pct": 12.0}})
    assert warns_mid and "weak" in warns_mid[0]


def test_shipped_batched_records_that_drift_stay_active() -> None:
    """The c=1 record behind the 0.81 concurrency ratio must survive its flag."""
    from vetch.calibrate_validate import validate_calibration_file

    remote = RECORDS_DIR / "qwen38-vs-qwen25" / "remote"
    drifting = [
        p for p in remote.glob("self-hosted*.json")
        if (json.loads(p.read_text())["provenance"].get("idle_drift_pct") or 0) > 15
    ]
    assert drifting, "expected records above the 15% drift flag"
    for path in drifting:
        report = validate_calibration_file(path)
        assert report["active"] is True
        assert report["ok"] is True
        assert any("idle_drift_pct" in w for w in report["warnings"])


def test_invariants_catch_a_self_contradictory_record() -> None:
    """A record may not contradict itself, however good its statistics are."""
    from vetch.calibration_invariants import errors_only, record_invariants

    record = {
        "schema_version": 2,
        "identity": {"concurrency": 32},
        "coefficients": {"wh_per_1k_output": 0.5, "status": {"wh_per_1k_output": "identified"}},
        "provenance": {
            "calibration_mode": "batched", "samples": 64,
            "raw_run_table": [{"energy_wh": 0.4, "raw_energy_wh": 0.5,
                               "output_tokens": 100, "duration_ms": 900.0}],
            "achieved_concurrency": 0.0,
        },
        "active": True,
    }
    codes = {v.code for v in errors_only(record_invariants(record))}
    assert "achieved_concurrency_zero" in codes

    record["provenance"]["achieved_concurrency"] = None  # unknown is honest
    assert not errors_only(record_invariants(record))


def test_invariants_reject_energy_exceeding_gross() -> None:
    """Idle subtraction cannot increase measured energy."""
    from vetch.calibration_invariants import errors_only, record_invariants

    record = {
        "schema_version": 2, "identity": {}, "coefficients": {},
        "provenance": {"samples": 1, "raw_run_table": [
            {"energy_wh": 0.9, "raw_energy_wh": 0.5,
             "output_tokens": 10, "duration_ms": 100.0}]},
    }
    assert "net_exceeds_gross" in {v.code for v in errors_only(record_invariants(record))}


def test_writer_refuses_a_contradictory_record(tmp_path, monkeypatch) -> None:
    """The guard belongs at write time, not only at validation time."""
    import pytest

    from vetch.calibration_store import write_record

    monkeypatch.setattr("vetch.calibrate.CALIBRATION_DIR", tmp_path)
    bad = json.loads((RECORDS_DIR / "qwen38-vs-qwen25" / "remote").glob(
        "self-hosted*c32*.json").__next__().read_text())
    bad["provenance"]["achieved_concurrency"] = 0.0
    with pytest.raises(ValueError, match="self-contradictory"):
        write_record(bad)
    assert not list(tmp_path.glob("*.json"))


def test_contended_idle_baseline_is_refused_before_the_grid() -> None:
    """Idle contention must abort before GPU time is spent, not after.

    The idle-drift check compares the windows either side of a finished run, so
    on a contended box it only reports the problem once the whole grid has run.
    Observed on a rented H100: a co-tenant held the card at 100% and 698 W of a
    700 W cap while our own server was stopped, and a container cannot see the
    neighbour's processes.
    """
    from vetch.calibrate_cuda import check_idle_baseline

    contended = check_idle_baseline(
        idle_watts=698.25, power_limit_w=700.0, utilization_pct=100,
    )
    assert len(contended) == 2
    assert any("busy while idle" in r for r in contended)
    assert any("cap" in r for r in contended)

    # Utilization catches contention even when power looks unremarkable.
    assert check_idle_baseline(
        idle_watts=200.0, power_limit_w=700.0, utilization_pct=95,
    )
    # A healthy box passes: model weights resident, nothing running.
    assert not check_idle_baseline(
        idle_watts=124.9, power_limit_w=525.0, utilization_pct=0,
    )
    # Unknown utilization must not fabricate a verdict.
    assert not check_idle_baseline(
        idle_watts=124.9, power_limit_w=525.0, utilization_pct=None,
    )


def test_basis_must_match_the_source_it_came_from() -> None:
    """A record may not claim a counter basis for power-sampled numbers."""
    from vetch.calibration_invariants import errors_only, record_invariants

    def codes(source: str, basis: str) -> set[str]:
        record = {"provenance": {"energy_source": source, "measurement_basis": basis,
                                 "samples": 1, "raw_run_table": [
                                     {"energy_wh": 0.1, "raw_energy_wh": 0.2,
                                      "output_tokens": 10, "duration_ms": 100.0}]}}
        return {v.code for v in errors_only(record_invariants(record))}

    assert "basis_source_mismatch" in codes("power_integration", "nvml_gpu_energy")
    assert "basis_source_mismatch" not in codes(
        "power_integration", "nvml_gpu_power_integration")
    assert "basis_source_mismatch" not in codes(
        "nvmlDeviceGetTotalEnergyConsumption", "nvml_gpu_energy")


def test_a_third_party_record_validates() -> None:
    """The validator's whole pitch is running against records we did not write.

    Every record it had ever judged was ours, following our conventions. A
    stranger's file — different key order, unknown extra fields, optional blocks
    absent, a different serving engine — must not fail on house style. The first
    one tried failed on `host.facility`, a block whose every field is optional
    and nullable.
    """
    from vetch.calibrate_validate import validate_calibration_record

    rows = []
    for text_tokens, out_tokens, duration in [
        (4096, 8, 3000.0), (4096, 8, 3010.0), (16384, 8, 3200.0),
        (16384, 8, 3190.0), (1024, 8, 2500.0), (1024, 8, 2520.0),
        (64, 512, 2600.0), (64, 1024, 4600.0), (64, 2048, 9000.0),
        (64, 2048, 9100.0), (64, 4096, 17000.0), (2048, 256, 2800.0),
    ]:
        energy = 0.004 + 0.0035 * text_tokens / 1000 + 0.52 * out_tokens / 1000
        rows.append({
            "duration_ms": duration, "output_tokens": out_tokens,
            "energy_wh": energy, "text_tokens": text_tokens,
            "raw_energy_wh": energy * 2.4, "n_images": 0, "requests": 1,
            "vendor_note": "an unknown field must not be fatal",
        })

    record = {
        "vendor": "acme-energy", "emitted_at": "2026-09-06T09:14:00Z",
        "schema_version": 2, "active": True,
        "coefficients": {
            "wh_per_1k_output": 0.52, "wh_per_1k_input": 0.0035,
            "wh_per_image": None, "intercept_wh": None,
            "visual_tokens_per_image": None,
        },
        "identity": {
            "gpu": "h100-sxm-80gb", "precision": "bf16",
            "provider": "self-hosted", "model": "meta/llama-3.1-8b",
            "serving_engine": "sglang", "concurrency": None,
            "visual_token_budget": None, "instance_type": None,
        },
        "host": {  # no facility block at all
            "infrastructure_provider": "lambda-labs",
            "provider_kind": "hyperscaler", "tenancy": "dedicated_host",
            "gpu_form_factor": "sxm",
        },
        "provenance": {
            "samples": 12, "raw_run_table": rows, "tier": 0,
            "idle_watts_before": 118.0, "idle_watts_after": 119.2,
            "idle_drift_pct": 1.02, "idle_subtracted": True,
            "energy_source": "nvmlDeviceGetTotalEnergyConsumption",
            "measurement_basis": "nvml_gpu_energy", "energy_domain": "gpu_board",
        },
    }

    report = validate_calibration_record(record)
    assert report["ok"] is True, report["schema_errors"]
    ident = report["identifiability"]
    assert ident["status"]["wh_per_1k_input"] == "identified"
    assert ident["status"]["wh_per_1k_output"] == "identified"
    assert ident["coefficients"]["wh_per_1k_output"] == pytest.approx(0.52, rel=1e-3)
    assert ident["coefficients"]["wh_per_1k_input"] == pytest.approx(0.0035, rel=1e-2)
    # It states no gate, so its stored statuses are not reproducible; say so.
    assert any("gate parameters" in w for w in report["warnings"])


# ---------------------------------------------------------------------------
# The validator must catch a record whose stored claims outrun its own data.
# ---------------------------------------------------------------------------


class TestRestatementDetection:
    @staticmethod
    def _batch2() -> dict:
        import json
        from pathlib import Path

        repo = Path(__file__).resolve().parent.parent
        path = next(
            p for p in (repo / "h100-calibration-records").glob("*.json")
            if (json.loads(p.read_text()).get("coefficients") or {}).get(
                "published_rung"
            ) == "output_input"
        )
        return json.loads(path.read_text())

    def test_pristine_record_validates(self) -> None:
        from vetch.calibrate_validate import validate_calibration_record

        report = validate_calibration_record(self._batch2())
        assert report["ok"] is True
        assert report["restatements"] == []
        assert report["attestation"]["match"] is True

    def test_inflated_coefficient_is_caught(self) -> None:
        import copy

        from vetch.calibrate_validate import validate_calibration_record

        record = copy.deepcopy(self._batch2())
        stored = record["coefficients"]["wh_per_1k_output"]
        record["coefficients"]["wh_per_1k_output"] = stored * 2
        record["energy_basis"]["marginal"]["wh_per_1k_output"] = stored * 2

        report = validate_calibration_record(record)
        # `active` still agrees, which is all the validator used to compare.
        assert report["active_mismatch"] is None
        assert report["ok"] is False
        assert any("wh_per_1k_output" in r for r in report["restatements"])

    def test_status_restatement_is_caught(self) -> None:
        import copy

        from vetch.calibrate_validate import validate_calibration_record

        record = copy.deepcopy(self._batch2())
        record["coefficients"]["status"]["wh_per_image"] = "identified"
        report = validate_calibration_record(record)
        assert report["ok"] is False
        assert any("wh_per_image" in r for r in report["restatements"])

    def test_published_rung_restatement_is_caught(self) -> None:
        import copy

        from vetch.calibrate_validate import validate_calibration_record

        record = copy.deepcopy(self._batch2())
        record["coefficients"]["published_rung"] = "output_intercept_input_image"
        report = validate_calibration_record(record)
        assert report["ok"] is False
        assert any("published_rung" in r for r in report["restatements"])

    def test_incomparable_published_rung_is_caught(self) -> None:
        import copy

        record = copy.deepcopy(self._batch2())
        # output_intercept and output_input have the same number of terms but
        # are different regression models.
        record["coefficients"]["published_rung"] = "output_intercept"
        report = validate_calibration_record(record)
        assert report["ok"] is False
        assert any("published_rung" in r for r in report["restatements"])

    def test_attestation_mismatch_is_caught(self) -> None:
        import copy

        from vetch.calibrate_validate import validate_calibration_record

        record = copy.deepcopy(self._batch2())
        record["content_hash"] = "sha256:" + "0" * 64
        report = validate_calibration_record(record)
        assert report["ok"] is False
        assert report["attestation"]["checked"] is True
        assert "content_hash" in report["attestation"]["mismatched"]


class TestMalformedThirdPartyRecord:
    """`validate-calibration` is advertised for someone else's file."""

    def test_null_energy_wh_reports_instead_of_raising(self) -> None:
        import copy

        from vetch.calibrate_validate import validate_calibration_record

        record = copy.deepcopy(TestRestatementDetection._batch2())
        record["provenance"]["raw_run_table"][0]["energy_wh"] = None
        # Must return a report, not raise TypeError from inside the fit.
        report = validate_calibration_record(record)
        assert isinstance(report, dict)
        assert any("malformed" in w for w in report["warnings"])

    def test_bad_duration_reports_instead_of_raising(self) -> None:
        import copy

        record = copy.deepcopy(TestRestatementDetection._batch2())
        record["provenance"]["raw_run_table"][0]["duration_ms"] = "bad"
        report = validate_calibration_record(record)
        assert isinstance(report, dict)
        assert any("malformed" in w for w in report["warnings"])


class TestRungComparisonDirection:
    """Under-claiming a rung is conservative; over-claiming is fatal.

    `output_only` is the published rung of seven of the eight shipped records,
    so it must decode to a real term set rather than to the empty set that a
    malformed name produces.
    """

    @staticmethod
    def _record(rung: str, coefficients: dict) -> dict:
        rows = []
        for text_tokens, out_tokens, duration in [
            (4096, 8, 3000.0), (4096, 8, 3010.0), (16384, 8, 3200.0),
            (16384, 8, 3190.0), (1024, 8, 2500.0), (1024, 8, 2520.0),
            (64, 512, 2600.0), (64, 1024, 4600.0), (64, 2048, 9000.0),
            (64, 2048, 9100.0), (64, 4096, 17000.0), (2048, 256, 2800.0),
        ]:
            energy = 0.004 + 0.0035 * text_tokens / 1000 + 0.52 * out_tokens / 1000
            rows.append({
                "duration_ms": duration, "output_tokens": out_tokens,
                "energy_wh": energy, "text_tokens": text_tokens,
                "raw_energy_wh": energy * 2.4, "n_images": 0, "requests": 1,
            })
        return {
            "schema_version": 2, "active": True,
            "coefficients": {**coefficients, "published_rung": rung},
            "identity": {
                "gpu": "h100-sxm-80gb", "precision": "bf16",
                "provider": "self-hosted", "model": "meta/llama-3.1-8b",
                "serving_engine": "sglang", "concurrency": None,
                "visual_token_budget": None, "instance_type": None,
            },
            "host": {
                "infrastructure_provider": "lambda-labs",
                "provider_kind": "hyperscaler", "tenancy": "dedicated_host",
                "gpu_form_factor": "sxm",
            },
            "provenance": {
                "samples": 12, "raw_run_table": rows, "tier": 0,
                "idle_watts_before": 118.0, "idle_watts_after": 119.2,
                "idle_drift_pct": 1.02, "idle_subtracted": True,
                "energy_source": "nvmlDeviceGetTotalEnergyConsumption",
                "measurement_basis": "nvml_gpu_energy",
                "energy_domain": "gpu_board",
            },
        }

    def test_output_only_under_claim_is_a_warning(self) -> None:
        from vetch.calibrate_validate import validate_calibration_record

        record = self._record("output_only", {
            "wh_per_1k_output": 0.52, "wh_per_1k_input": None,
            "wh_per_image": None, "intercept_wh": None,
        })
        report = validate_calibration_record(record)
        assert report["restatements"] == [], report["restatements"]
        assert any("published_rung" in w for w in report["restatement_warnings"])
        assert report["ok"] is True

    def test_over_claimed_rung_is_an_error(self) -> None:
        from vetch.calibrate_validate import validate_calibration_record

        record = self._record("output_intercept_input_image", {
            "wh_per_1k_output": 0.52, "wh_per_1k_input": 0.0035,
            "wh_per_image": None, "intercept_wh": 0.004,
        })
        report = validate_calibration_record(record)
        assert any("published_rung" in e for e in report["restatements"])
        assert report["ok"] is False

    def test_incomparable_rung_is_an_error(self) -> None:
        from vetch.calibrate_validate import validate_calibration_record

        # output_intercept is neither a subset nor a superset of the supported
        # output_intercept_input... but output_input is a different model of
        # the same size, which must never pass as a conservative restatement.
        record = self._record("output_image", {
            "wh_per_1k_output": 0.52, "wh_per_1k_input": None,
            "wh_per_image": None, "intercept_wh": None,
        })
        report = validate_calibration_record(record)
        assert any("published_rung" in e for e in report["restatements"])
        assert report["ok"] is False
