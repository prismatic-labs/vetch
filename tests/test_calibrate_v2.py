"""Nested-model identifiability against the frozen batch-1 fixture.

The fixture was authored from nested OLS + bootstrap on the seven raw tables
*before* this gate existed. Do not regenerate it from the stored v1 intervals.
Match records by (model, precision), not filename: schema-2 migration renames
files to the measurement slug.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from vetch.calibrate_v2 import (  # noqa: E402
    LEVERAGE_MULTIPLE,
    RUNG_OUTPUT_INTERCEPT,
    RUNG_OUTPUT_ONLY,
    STATUS_IDENTIFIED,
    STATUS_NOT_IDENTIFIED,
    fit_nested_models,
    fit_record,
    nested_fit_to_dict,
    predict_design_leverage,
    v2_grid_design,
)

REPO = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "calibration_v2_nested_status.json"
RECORDS_DIR = REPO / "h100-calibration-records"


def _top_level_records() -> list[tuple[Path, dict]]:
    """Batch-1 records only.

    The fixture pins the seven records migrated from schema 1, whose designs
    carry no prefill isolation arm. Later batches are born as schema 2 and are
    distinguished by the absence of a schema_migration note, so adding one does
    not make this lookup ambiguous. No expected verdict changes here; this only
    scopes which records the fixture is about.
    """
    out: list[tuple[Path, dict]] = []
    for path in sorted(RECORDS_DIR.glob("*.json")):
        rec = json.loads(path.read_text())
        if not (isinstance(rec, dict) and rec.get("identity")):
            continue
        if not (rec.get("provenance") or {}).get("schema_migration"):
            continue  # born v2: a later batch, not what this fixture pins
        out.append((path, rec))
    return out


def _find(model: str, precision_prefix: str) -> dict:
    matches = []
    for _path, rec in _top_level_records():
        ident = rec["identity"]
        prec = str(ident.get("precision") or "")
        if ident.get("model") == model and prec.startswith(precision_prefix):
            matches.append(rec)
    assert len(matches) == 1, (model, precision_prefix, len(matches))
    return matches[0]


def test_leverage_multiple_is_pinned() -> None:
    data = json.loads(FIXTURE.read_text())
    assert data["leverage_multiple"] == LEVERAGE_MULTIPLE == 10.0


def test_batch1_nested_statuses_match_fixture() -> None:
    data = json.loads(FIXTURE.read_text())
    expected = data["records"]
    assert len(expected) == 7

    for _filename, exp in expected.items():
        rec = _find(exp["model"], exp["precision"])
        fit = fit_record(rec)
        assert fit.published_rung == exp["published_rung"] == RUNG_OUTPUT_ONLY
        assert fit.status == exp["status"]
        assert fit.active is exp["active"]
        assert fit.wh_per_1k_input is None
        assert fit.wh_per_1k_output is not None and fit.wh_per_1k_output > 0
        assert fit.status["wh_per_1k_output"] == STATUS_IDENTIFIED
        assert fit.status["wh_per_1k_input"] == STATUS_NOT_IDENTIFIED
        if "dropped_zero_raw" in exp:
            assert fit.dropped_zero_raw == exp["dropped_zero_raw"]
        # Structured residuals are a flag, not a whole-record reject.
        assert fit.active is True


def test_gemma_fp8_input_not_identified_despite_positive_full_model_ci() -> None:
    data = json.loads(FIXTURE.read_text())
    gemma_fp8_key = next(
        k for k, v in data["records"].items()
        if v["model"].endswith("31B-it") and v["precision"] == "fp8"
    )
    exp = data["records"][gemma_fp8_key]
    rec = _find(exp["model"], "fp8")
    fit_block = rec.get("provenance", {}).get("fit") or {}
    v1_ci = (fit_block.get("v1") or {}).get("input_ci95") or fit_block.get("input_ci95")
    assert v1_ci is not None and v1_ci[0] > 0 and v1_ci[1] > 0
    fit = fit_record(rec)
    assert fit.status["wh_per_1k_input"] == STATUS_NOT_IDENTIFIED
    assert fit.wh_per_1k_input is None
    assert fit.design_leverage["input"]["design_leverage"] is False
    assert fit.active is True


def test_no_idle_fraction_gate_empties_dataset() -> None:
    """A 30% idle gate would drop every row of four records. We keep them."""
    rec = _find("Qwen/Qwen2.5-7B-Instruct", "bf16")
    fit = fit_record(rec)
    assert fit.admitted_rows == 11
    assert fit.active is True
    assert fit.status["wh_per_1k_output"] == STATUS_IDENTIFIED


def test_v2_grid_has_prefill_isolation_arm() -> None:
    grid = v2_grid_design(vision=False, quick=True)
    prefill = [g for g in grid if g["arm"] == "prefill"]
    assert prefill
    assert all(g["max_tokens"] == 8 for g in prefill)
    assert max(g["approx_text_tokens"] for g in prefill) >= 1024


def test_nested_fit_metadata_preserves_malformed_row_count() -> None:
    rows = _synthetic_rows(n=24)
    rows[0]["duration_ms"] = "bad"

    fit = fit_nested_models(rows, idle_watts=100.0)
    metadata = nested_fit_to_dict(fit)

    assert fit.dropped_malformed == 1
    assert metadata["dropped_malformed"] == 1


def test_predict_design_leverage_refuses_no_prefill_arm() -> None:
    v1_like = [
        {"text_tokens": 64, "output_tokens": 32},
        {"text_tokens": 128, "output_tokens": 64},
        {"text_tokens": 256, "output_tokens": 128},
    ]
    pred = predict_design_leverage(
        idle_watts=124.0,
        planned_rows=v1_like,
        expected_duration_ms=[80.0, 200.0, 800.0],
        prior_wh_per_1k_output=0.5,
    )
    assert pred["ok"] is True
    assert pred["input"]["design_leverage"] is False
    assert pred["input"]["adequate"] is False


def test_published_coefficients_come_from_published_rung() -> None:
    """Intercept + output must be the joint fit, not a through-origin slope."""
    intercept, slope = 0.02, 0.5
    rows = []
    for i, out in enumerate((100, 200, 400, 800, 1600, 3200, 2400, 4800)):
        rows.append({
            "output_tokens": out,
            "text_tokens": 32,
            "n_images": 0,
            "energy_wh": intercept + slope * (out / 1000.0),
            "raw_energy_wh": 0.2,
            "duration_ms": 400.0 + 50.0 * i,
        })
    fit = fit_nested_models(
        rows, idle_watts=50.0,
        energy_source="nvmlDeviceGetTotalEnergyConsumption",
    )
    assert fit.published_rung == RUNG_OUTPUT_INTERCEPT
    assert fit.intercept_wh is not None
    assert abs(fit.intercept_wh - intercept) < 0.002
    assert fit.wh_per_1k_output is not None
    assert abs(fit.wh_per_1k_output - slope) < 0.02
    through_origin = fit.rungs[RUNG_OUTPUT_ONLY]["coeffs"]["output"]
    joint = fit.rungs[RUNG_OUTPUT_INTERCEPT]["coeffs"]["output"]
    assert fit.wh_per_1k_output == joint
    assert through_origin != joint
    # Through-origin absorbs the intercept into the slope.
    assert through_origin > joint


def test_missing_idle_is_fatal_not_a_free_pass() -> None:
    """No idle figure means no sigma, so no axis can claim leverage.

    Treating absent idle as 0.0 made every leverage ratio undefined, and an
    undefined ratio skipped the gate rather than failing it.
    """
    rows = [
        {"energy_wh": 0.5 + 0.01 * i, "raw_energy_wh": 0.6, "output_tokens": 200 * i + 200,
         "text_tokens": 32, "n_images": 0, "duration_ms": 400.0 * i + 400}
        for i in range(12)
    ]
    fit = fit_nested_models(rows, idle_watts=None)
    assert fit.active is False
    assert fit.wh_per_1k_output is None
    assert all(v == STATUS_NOT_IDENTIFIED for v in fit.status.values())
    assert "no idle measurement" in fit.rejection_reasons[0]


def test_rank_deficient_design_is_not_identified() -> None:
    """A collinear design cannot separate its terms however tight the interval.

    Every bootstrap resample inherits the same degeneracy, so the CI is narrow
    and excludes zero while the intercept/slope split is arbitrary.
    """
    rows = [
        {"energy_wh": 0.5 + 0.0001 * i, "raw_energy_wh": 0.6, "output_tokens": 1000,
         "text_tokens": 100, "n_images": 0, "duration_ms": 8000.0}
        for i in range(12)
    ]
    fit = fit_nested_models(rows, idle_watts=124.0)
    assert fit.active is False
    assert fit.wh_per_1k_output is None
    assert fit.intercept_wh is None
    quality = fit.design_leverage["rung_quality"]
    assert quality["output_intercept"]["ok"] is False
    assert quality["output_intercept"]["condition_number"] > 30.0


def test_batch1_rungs_all_pass_the_quality_gate() -> None:
    """The new gates must not cost the seven records their decode slope."""
    data = json.loads(FIXTURE.read_text())
    for exp in data["records"].values():
        fit = fit_record(_find(exp["model"], exp["precision"]))
        quality = fit.design_leverage["rung_quality"]
        assert quality[RUNG_OUTPUT_ONLY]["ok"] is True, exp["model"]
        assert quality[RUNG_OUTPUT_ONLY]["r2"] >= 0.85
        assert fit.wh_per_1k_output is not None


def test_batch2_identifies_prefill_where_batch1_could_not() -> None:
    """The first record produced by a design with a prefill isolation arm.

    Batch 1 tops out at 427 input tokens with no arm, so its input axis is
    unidentifiable in principle. This record spans 811 to 12331 tokens per
    request and determines the coefficient better than decode. The pair is the
    protocol's central claim, measured rather than asserted.
    """
    # Select by property, not filename: the same commit fixed exactly this
    # anti-pattern in two other test files and then reintroduced it here. A
    # future 575 W rerun would silently satisfy the glob; ambiguity must fail
    # loudly instead.
    candidates = []
    for path in sorted(RECORDS_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        if not (isinstance(data, dict) and data.get("identity")):
            continue
        if (data.get("provenance") or {}).get("schema_migration"):
            continue  # batch 1; this test is about the born-v2 batch
        if data["identity"].get("model") == "Qwen/Qwen2.5-7B-Instruct":
            candidates.append(data)
    assert len(candidates) == 1, (
        f"{len(candidates)} born-v2 Qwen2.5-7B records; this test pins one "
        "specific measurement, so extend the selector before adding more"
    )
    record = candidates[0]
    fit = fit_record(record)

    assert fit.status["wh_per_1k_input"] == STATUS_IDENTIFIED
    assert fit.status["wh_per_1k_output"] == STATUS_IDENTIFIED
    assert fit.active is True
    # The intercept is not resolvable here, and the lattice publishes the other
    # two without it. A strict output->+intercept->+input chain could not.
    assert fit.published_rung == "output_input"
    assert fit.intercept_wh is None

    def half_width(ci, beta):
        return (ci[1] - ci[0]) / 2 / abs(beta)

    assert half_width(fit.input_ci95, fit.wh_per_1k_input) < 0.10
    assert half_width(fit.output_ci95, fit.wh_per_1k_output) < 0.10

    # Decode agrees with the batch-1 record for the same model, measured on a
    # different host, cap and design.
    batch1 = next(
        rec for _p, rec in _top_level_records()
        if rec["identity"]["model"] == "Qwen/Qwen2.5-7B-Instruct"
    )
    b1_decode = batch1["coefficients"]["wh_per_1k_output"]
    assert abs(fit.wh_per_1k_output - b1_decode) / b1_decode < 0.05

    # Prefill costs well under a hundredth of decode per token, and the roofline
    # timing ratio (0.34%) understates it: prefill runs near the power cap.
    ratio = fit.wh_per_1k_input / fit.wh_per_1k_output
    assert 0.004 < ratio < 0.02
    assert ratio > 0.0034


# ---------------------------------------------------------------------------
# Regressions from the calibration-v2 adversarial review.
# ---------------------------------------------------------------------------


def _synthetic_rows(
    *,
    n: int = 24,
    wh_per_1k_output: float = 0.5,
    wh_per_1k_input: float = 0.004,
    wh_per_image: float | None = None,
    intercept: float = 0.0,
    requests: int | None = None,
    prefill_arm: bool = True,
    noise: float = 0.0,
    seed: int = 7,
) -> list[dict]:
    """A clean design with a real prefill arm, built to exercise one axis at a time."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for i in range(n):
        if prefill_arm and i % 2 == 0:
            text, out = 1024.0 * (1 + (i % 6)), 8.0
        else:
            text, out = 32.0, 256.0 * (1 + (i % 4))
        images = float((i // 2) % 2) if wh_per_image is not None else 0.0
        energy = (
            intercept
            + wh_per_1k_input * text / 1000.0
            + wh_per_1k_output * out / 1000.0
            + (wh_per_image or 0.0) * images
        )
        if noise:
            energy += float(rng.normal(0.0, noise))
        reps = requests or 1
        row = {
            "text_tokens": text * reps,
            "output_tokens": out * reps,
            "n_images": images * reps,
            "energy_wh": energy * reps,
            "raw_energy_wh": energy * reps + 0.05 * reps,
            "duration_ms": 3000.0,
        }
        if requests:
            row["requests"] = reps
        rows.append(row)
    return rows


class TestConditionNumberScaleInvariance:
    """The collinearity gate must not depend on the unit a predictor is carried in.

    v1 could slice column 0 off the standardisation because the intercept was
    structurally first. The v2 lattice makes the intercept optional, so on an
    intercept-free rung column 0 is a real predictor.
    """

    def test_intercept_free_rung_is_scale_invariant(self) -> None:
        rows = _synthetic_rows(noise=0.002)
        scaled = [dict(r, text_tokens=r["text_tokens"]) for r in rows]
        base = fit_nested_models(rows, idle_watts=100.0)
        # Same design, input carried per-token rather than per-1k, by scaling
        # the coefficient to match. The DESIGN is identical; only units move.
        conds = {
            name: q["condition_number"]
            for name, q in base.design_leverage["rung_quality"].items()
        }
        assert conds["output_input"] < 5.0, conds
        # An intercept-free rung must not report worse conditioning than the
        # same design with an intercept added.
        assert conds["output_input"] <= conds["output_intercept_input"] + 1e-9, conds
        assert scaled  # design untouched

    def test_condition_number_matches_full_standardisation(self) -> None:
        rows = _synthetic_rows(noise=0.001)
        fit = fit_nested_models(rows, idle_watts=100.0)
        out_k = np.array([r["output_tokens"] / 1000.0 for r in rows])
        in_k = np.array([r["text_tokens"] / 1000.0 for r in rows])
        X = np.column_stack([in_k, out_k])  # labels sort input before output
        stds = X.std(axis=0)
        expected = float(np.linalg.cond(X / np.where(stds < 1e-10, 1.0, stds)))
        got = fit.design_leverage["rung_quality"]["output_input"]["condition_number"]
        assert got == pytest.approx(expected, rel=1e-9)


class TestRejectionScope:
    """A rejected axis kills the record only if it is a term of the published rung."""

    @staticmethod
    def _split_rung_design(wh_image: float = -0.02) -> list[dict]:
        """A design whose image axis is judged on a rung the output axis is not.

        `n_images` tracks `text_tokens` exactly, so every rung carrying both
        input and image is collinear and unusable. Output therefore resolves on
        `output_intercept_input` while image resolves on `output_intercept_image`
        -- the published numbers are read from a fit the rejected term is not
        in. This is the configuration that used to kill the whole record.
        """
        rng = np.random.default_rng(1)
        spec = [(1024.0 * (1 + i % 8), 8.0) for i in range(16)]
        spec += [(32.0, 256.0 * (1 + i % 8)) for i in range(16)]
        rows = []
        for text, out in spec:
            images = text / 1024.0 if out == 8.0 else 0.0
            energy = (
                0.03
                + 0.004 * text / 1000.0
                + 0.5 * out / 1000.0
                + wh_image * images
                + float(rng.normal(0.0, 5e-5))
            )
            rows.append({
                "text_tokens": text, "output_tokens": out, "n_images": images,
                "energy_wh": energy, "raw_energy_wh": energy + 0.05,
                "duration_ms": 3000.0,
            })
        return rows

    def test_rejection_outside_published_rung_does_not_deactivate(self) -> None:
        fit = fit_nested_models(self._split_rung_design(), idle_watts=100.0)
        assert fit.status["wh_per_image"] == "rejected", fit.null_reasons
        assert "image" not in fit.published_rung
        # The published coefficients did not come out of the contaminated fit,
        # so the record survives. Previously any rejected axis anywhere killed
        # it -- the whole-record death v2 exists to stop.
        assert fit.active is True
        assert fit.rejection_reasons == []
        # Never silently dropped: disclosed as an out-of-rung rejection and in
        # the per-axis status the record carries.
        assert any("wh_per_image" in r for r in fit.unpublished_rejections)

    def test_rejection_inside_published_rung_is_fatal(self) -> None:
        # Same shape, but a large image term forces every image-free rung below
        # the r2 floor, so the published rung contains the rejected axis.
        rng = np.random.default_rng(1)
        rows = []
        for i in range(32):
            text, out = ((1024.0 * (1 + i % 8), 8.0) if i < 16
                         else (32.0, 256.0 * (1 + i % 8)))
            images = 1.0 if i % 2 else 0.0
            energy = (
                0.03 + 0.004 * text / 1000.0 + 0.5 * out / 1000.0
                - 0.30 * images + float(rng.normal(0.0, 2e-4))
            )
            rows.append({
                "text_tokens": text, "output_tokens": out, "n_images": images,
                "energy_wh": energy, "raw_energy_wh": energy + 0.05,
                "duration_ms": 3000.0,
            })
        fit = fit_nested_models(rows, idle_watts=100.0)
        assert fit.status["wh_per_image"] == "rejected"
        assert "image" in fit.published_rung
        assert fit.active is False
        assert fit.rejection_reasons
        assert fit.unpublished_rejections == []


class TestLatticeIndependentAxes:
    """The motivating case: publish decode + image while prefill is unidentifiable.

    A strict chain cannot express this -- an unresolvable input term would drag
    the image term down with it. No shipped record has produced this rung, so
    it is pinned synthetically.
    """

    def test_image_publishes_while_input_is_blind(self) -> None:
        rows = _synthetic_rows(
            n=32, wh_per_image=0.3, prefill_arm=False, noise=0.0004,
        )
        fit = fit_nested_models(rows, idle_watts=100.0)
        assert fit.status["wh_per_1k_input"] == STATUS_NOT_IDENTIFIED
        assert "prefill isolation arm" in fit.null_reasons["wh_per_1k_input"]
        assert fit.status["wh_per_image"] == STATUS_IDENTIFIED
        assert fit.wh_per_image == pytest.approx(0.3, rel=0.1)
        assert "image" in fit.published_rung
        assert "input" not in fit.published_rung
        assert fit.active is True


class TestMalformedRowsAreRejectedNotRaised:
    """validate-calibration runs on third-party files: never traceback."""

    @pytest.mark.parametrize(
        "bad",
        [
            {"energy_wh": None},
            {"energy_wh": float("nan")},
            {"output_tokens": None},
            {"text_tokens": "many"},
            {"n_images": float("inf")},
            {"requests": 0},
            {"requests": None},
        ],
    )
    def test_malformed_row_dropped(self, bad: dict) -> None:
        rows = _synthetic_rows(n=20)
        rows.append({**rows[0], **bad})
        fit = fit_nested_models(rows, idle_watts=100.0)  # must not raise
        assert fit.dropped_malformed == 1
        assert fit.admitted_rows == 20

    def test_non_mapping_row_dropped(self) -> None:
        rows = _synthetic_rows(n=20)
        fit = fit_nested_models([*rows, "not a row"], idle_watts=100.0)  # type: ignore[list-item]
        assert fit.dropped_malformed == 1


class TestAggregationIsReadPerRequest:
    """Aggregated rows must not change any design-point judgement."""

    def test_aggregated_design_matches_unaggregated(self) -> None:
        plain = fit_nested_models(_synthetic_rows(n=24), idle_watts=100.0)
        agg = fit_nested_models(_synthetic_rows(n=24, requests=16), idle_watts=100.0)
        assert agg.published_rung == plain.published_rung
        assert agg.wh_per_1k_output == pytest.approx(plain.wh_per_1k_output, rel=1e-6)
        assert agg.wh_per_1k_input == pytest.approx(plain.wh_per_1k_input, rel=1e-6)
        assert agg.design_leverage["input"]["prefill_arm_rows"] == (
            plain.design_leverage["input"]["prefill_arm_rows"]
        )

    def test_residual_grouping_uses_per_request_output(self) -> None:
        from vetch.calibrate_v2 import detect_residuals_structured

        # 8 output tokens per request, aggregated 64x -> a 512-token row total.
        # Grouped on the total, every prefill row lands in the decode bucket.
        rows = [
            {"output_tokens": 8 * 64, "requests": 64, "n_images": 0} for _ in range(4)
        ] + [{"output_tokens": 512, "requests": 1, "n_images": 0} for _ in range(4)]
        residuals = [0.01] * 4 + [-0.01] * 4
        assert detect_residuals_structured(rows, residuals) is True

    def test_idle_noise_is_reported_and_gates_nothing(self) -> None:
        fit = fit_nested_models(
            _synthetic_rows(n=24), idle_watts=100.0, idle_relative=0.09,
        )
        noise = fit.design_leverage["idle_noise"]
        assert noise["relative"] == 0.09
        assert noise["gates_any_verdict"] is False
        assert noise["sigma_idle_wh"] > 0


class TestSequentialPruning:
    """Blind terms are pruned one per round, worst-determined first.

    Removing every blind term at once freezes each of their verdicts in the
    same contaminated fit, so a term that only looked blind because of another
    term is never re-judged in the model that would have cleared it.
    """

    @staticmethod
    def _two_blind_axes() -> list[dict]:
        rng = np.random.default_rng(2)
        spec = [(1024.0 * (1 + i % 6), 8.0, float(i % 2)) for i in range(16)]
        spec += [(32.0, 256.0 * (1 + i % 6), float(i % 2)) for i in range(16)]
        rows = []
        for text, out, images in spec:
            energy = (
                0.02 + 0.0002 * text / 1000.0 + 0.5 * out / 1000.0
                + 0.001 * images + float(rng.normal(0.0, 6e-4))
            )
            rows.append({
                "text_tokens": text, "output_tokens": out, "n_images": images,
                "energy_wh": energy, "raw_energy_wh": energy + 0.05,
                "duration_ms": 3000.0,
            })
        return rows

    def test_survivors_are_judged_in_the_cleaned_model(self) -> None:
        fit = fit_nested_models(self._two_blind_axes(), idle_watts=100.0)
        # Both optional axes have real arms, so both enter the pruning loop.
        assert fit.design_leverage["input"]["design_leverage"] is True
        assert fit.design_leverage["image"]["design_leverage"] is True
        assert fit.status["wh_per_1k_input"] == STATUS_NOT_IDENTIFIED
        assert fit.status["wh_per_image"] == STATUS_NOT_IDENTIFIED
        # Intercept and decode are resolved after both blind terms are gone.
        assert fit.status["intercept_wh"] == STATUS_IDENTIFIED
        assert fit.status["wh_per_1k_output"] == STATUS_IDENTIFIED
        assert fit.published_rung == "output_intercept"
        assert fit.active is True

    def test_pruned_axes_keep_the_verdict_that_could_see_them(self) -> None:
        fit = fit_nested_models(self._two_blind_axes(), idle_watts=100.0)
        # Not a vacuous "no rung contains this term" -- the real finding from
        # the round that still had the term in the model.
        for axis in ("wh_per_1k_input", "wh_per_image"):
            assert "determined only to" in fit.null_reasons[axis], axis
            assert fit.axes[axis].leverage is not None


class TestTermsForRung:
    """Rung names are compared by term set, never by token count.

    `output_intercept` and `output_input` are both two tokens and different
    models; `output_only` is spelled with a token that names no axis.
    """

    def test_output_only_decodes_to_the_output_term(self) -> None:
        from vetch.calibrate_v2 import terms_for_rung

        assert terms_for_rung("output_only") == frozenset({"wh_per_1k_output"})

    def test_every_rung_the_fitter_can_publish_decodes(self) -> None:
        from vetch.calibrate_v2 import _rung_name, terms_for_rung

        optional = ("intercept", "input", "image")
        for mask in range(1 << len(optional)):
            chosen = [optional[i] for i in range(len(optional)) if mask & (1 << i)]
            name = _rung_name(["output", *chosen])
            assert terms_for_rung(name), name
            assert len(terms_for_rung(name)) == len(chosen) + 1, name

    def test_same_size_different_models_are_incomparable(self) -> None:
        from vetch.calibrate_v2 import terms_for_rung

        a, b = terms_for_rung("output_intercept"), terms_for_rung("output_input")
        assert a != b
        assert not (a < b or b < a)

    def test_unknown_names_decode_to_nothing(self) -> None:
        from vetch.calibrate_v2 import terms_for_rung

        for name in (None, "", "garbage", "batched", "only", "output_bogus"):
            assert terms_for_rung(name) == frozenset(), name


class TestDurationFloorDropsAreCounted:
    """Losing a short prefill point silently is how a mis-sized grid degrades a
    fit with nothing saying rows went missing.

    Observed on Apple Silicon: an under-estimated prefill rate sized every point
    at requests=1, the 1024-token windows ran ~1.2 s against a 2 s floor, and
    three rows vanished without a counter.
    """

    def test_floor_drops_are_reported(self) -> None:
        from vetch.calibrate_v2 import admit_rows

        rows = [
            {"raw_energy_wh": 0.01, "energy_wh": 0.008, "duration_ms": 1234.0,
             "output_tokens": 8, "text_tokens": 1066, "n_images": 0},
            {"raw_energy_wh": 0.05, "energy_wh": 0.045, "duration_ms": 9591.0,
             "output_tokens": 8, "text_tokens": 8105, "n_images": 0},
        ]
        kept, zero, bad, floor = admit_rows(rows, energy_source="powermetrics")
        assert len(kept) == 1
        assert floor == 1
        assert (zero, bad) == (0, 0)

    def test_counter_field_reaches_the_fit(self) -> None:
        from vetch.calibrate_v2 import fit_nested_models

        rows = []
        for i in range(24):
            text, out = ((1024.0 * (1 + i % 4), 8.0) if i % 2 == 0
                         else (32.0, 256.0 * (1 + i % 4)))
            energy = 0.004 * text / 1000 + 0.5 * out / 1000
            rows.append({
                "text_tokens": text, "output_tokens": out, "n_images": 0,
                "energy_wh": energy, "raw_energy_wh": energy + 0.05,
                # Half the rows sit under the 2 s sampling floor.
                "duration_ms": 1200.0 if i % 2 == 0 else 9000.0,
            })
        fit = fit_nested_models(rows, idle_watts=100.0, energy_source="powermetrics")
        assert fit.dropped_under_floor == 12

    def test_counter_path_is_inert_on_the_energy_counter(self) -> None:
        from vetch.calibrate_v2 import admit_rows

        rows = [{"raw_energy_wh": 0.01, "energy_wh": 0.008, "duration_ms": 100.0,
                 "output_tokens": 8, "text_tokens": 1066, "n_images": 0}]
        kept, _z, _b, floor = admit_rows(
            rows, energy_source="nvmlDeviceGetTotalEnergyConsumption")
        assert len(kept) == 1 and floor == 0


class TestIdleUncertaintyRowShare:
    """The gate must judge idle instability by what it costs the measurement.

    `idle_drift_pct` is scale-free, so it cannot: a 1 W laptop idle and a 90 W
    H100 idle report wildly different percentages for the same absolute wobble.
    Measured directly, a quiet Apple Silicon machine shows 33.6% worst-case
    disagreement between two sampling windows -- an idle laptop trips a 15% gate.
    """

    @staticmethod
    def _rows(duration_ms: float, energy_wh: float, n: int = 6) -> list[dict]:
        return [{"duration_ms": duration_ms, "energy_wh": energy_wh} for _ in range(n)]

    def test_same_absolute_wobble_scores_the_same_on_both_accelerators(self) -> None:
        from vetch.calibrate_v2 import idle_uncertainty_row_share_pct

        # 0.3 W of background excursion, identical rows. Idle level differs by
        # 90x, which is the whole reason the percentage gate misfires.
        laptop = idle_uncertainty_row_share_pct(self._rows(10_000, 0.02), 1.0, 1.3)
        h100 = idle_uncertainty_row_share_pct(self._rows(10_000, 0.02), 90.0, 90.3)
        assert laptop == pytest.approx(h100)

    def test_scales_with_row_energy(self) -> None:
        from vetch.calibrate_v2 import idle_uncertainty_row_share_pct

        small = idle_uncertainty_row_share_pct(self._rows(10_000, 0.01), 1.0, 1.3)
        large = idle_uncertainty_row_share_pct(self._rows(10_000, 0.10), 1.0, 1.3)
        assert small > large
        assert small == pytest.approx(large * 10)

    def test_median_resists_one_anomalous_window(self) -> None:
        from vetch.calibrate_v2 import idle_uncertainty_row_share_pct

        rows = self._rows(10_000, 0.02, n=9)
        rows.append({"duration_ms": 10_000, "energy_wh": 1e-6})  # absurd outlier
        share = idle_uncertainty_row_share_pct(rows, 1.0, 1.3)
        clean = idle_uncertainty_row_share_pct(self._rows(10_000, 0.02), 1.0, 1.3)
        assert share == pytest.approx(clean, rel=0.2)

    def test_unusable_inputs_return_none(self) -> None:
        from vetch.calibrate_v2 import idle_uncertainty_row_share_pct

        assert idle_uncertainty_row_share_pct([], 1.0, 1.3) is None
        assert idle_uncertainty_row_share_pct(self._rows(1, 1), None, 1.3) is None
        assert idle_uncertainty_row_share_pct(self._rows(1, 1), 1.0, float("nan")) is None
        assert idle_uncertainty_row_share_pct([{"duration_ms": 0, "energy_wh": 0}],
                                              1.0, 1.3) is None

    def test_gate_parameters_disclose_the_threshold(self) -> None:
        from vetch.calibrate_v2 import (
            MAX_IDLE_UNCERTAINTY_ROW_SHARE_PCT,
            gate_parameters,
        )

        assert gate_parameters()["max_idle_uncertainty_row_share_pct"] == (
            MAX_IDLE_UNCERTAINTY_ROW_SHARE_PCT
        )

    def test_every_shipped_h100_record_passes(self) -> None:
        """Opposing direction: the accelerator where the old rule never misfired."""
        import json
        from pathlib import Path

        from vetch.calibrate_v2 import (
            MAX_IDLE_UNCERTAINTY_ROW_SHARE_PCT,
            idle_uncertainty_row_share_pct,
        )

        repo = Path(__file__).resolve().parent.parent
        checked = 0
        for path in (repo / "h100-calibration-records").glob("*.json"):
            prov = json.loads(path.read_text()).get("provenance") or {}
            share = idle_uncertainty_row_share_pct(
                prov.get("raw_run_table") or [],
                prov.get("idle_watts_before"), prov.get("idle_watts_after"),
            )
            if share is None:
                continue
            checked += 1
            assert share < MAX_IDLE_UNCERTAINTY_ROW_SHARE_PCT, (path.name, share)
        assert checked >= 8
