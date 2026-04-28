"""Tests for the v0.4.0 stall circuit breaker.

Covers:
- Configuration: set_stall_action / get_stall_action validation
- Detection: per-session stall flag set on STALL-001, lazy gating
- Actions: log (default), warn, kill, reroute
- Exception hierarchy: StallDetected propagates past `except ValueError:`
- Fail-open: stall handler errors do not break LLM calls
- Fail-open reroute: substituted-model errors retry with original
- Recovery: clear_stall() re-arms the breaker
- Session isolation: two sessions don't interfere
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import patch

import pytest

import vetch
from vetch._stall import apply_stall_action, looks_like_param_mismatch
from vetch.advisory import Advisory
from vetch.config import VALID_STALL_ACTIONS, get_stall_action, set_stall_action
from vetch.context import TrackingContext
from vetch.exceptions import ConfigurationError, StallDetected, VetchInterrupt
from vetch.session import Session


# Helper: feed a Session enough fake stalled events to trigger STALL-001.
def _stall_a_session(session: Session, n_calls: int = 16) -> None:
    """Simulate n_calls of low-output, repetitive-input events on a session.

    STALL-001 fires after 10+ calls with 80% having output_tokens < 5 and
    >=50% input similarity. We use fixed input tokens so similarity is 100%.
    """
    for _ in range(n_calls):
        event: dict[str, Any] = {
            "model": "gpt-4o",
            "usage": {"text": {"input_tokens": 500, "output_tokens": 1}},
            "estimated_cost_usd": 0.05,
        }
        session.register_event(event)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


class TestStallActionConfig:
    def test_default_is_log(self) -> None:
        """Default stall_action is 'log' (backwards compatible)."""
        action, fallback = get_stall_action()
        assert action == "log"
        assert fallback is None

    def test_set_valid_actions(self) -> None:
        """All four valid actions can be set."""
        for action in ["log", "warn", "kill"]:
            set_stall_action(action)
            assert get_stall_action() == (action, None)
        # reroute requires a fallback
        set_stall_action("reroute", fallback_model="gpt-4o-mini")
        assert get_stall_action() == ("reroute", "gpt-4o-mini")

    def test_invalid_action_raises(self) -> None:
        """Unknown actions raise ConfigurationError."""
        with pytest.raises(ConfigurationError):
            set_stall_action("nuke")

    def test_reroute_without_fallback_raises(self) -> None:
        """reroute requires fallback_model."""
        with pytest.raises(ConfigurationError):
            set_stall_action("reroute")
        with pytest.raises(ConfigurationError):
            set_stall_action("reroute", fallback_model=None)

    def test_valid_actions_set_matches_docs(self) -> None:
        """The exported VALID_STALL_ACTIONS set matches the documented modes."""
        assert VALID_STALL_ACTIONS == frozenset({"log", "warn", "kill", "reroute"})


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestStallDetectedHierarchy:
    def test_inherits_from_runtime_error(self) -> None:
        """StallDetected → VetchInterrupt → RuntimeError (NOT ValueError)."""
        assert issubclass(StallDetected, VetchInterrupt)
        assert issubclass(StallDetected, RuntimeError)
        assert not issubclass(StallDetected, ValueError)

    def test_not_caught_by_value_error_handler(self) -> None:
        """A `except ValueError:` handler must NOT swallow StallDetected.

        This is the whole point of the separate hierarchy — user code that
        catches ValueError for data validation should not eat our circuit
        breaker.
        """
        caught = False
        with pytest.raises(StallDetected):
            try:
                raise StallDetected("test", wasted_cost_usd=1.0)
            except ValueError:  # pragma: no cover — must NOT catch
                caught = True
        assert caught is False

    def test_caught_by_vetch_interrupt(self) -> None:
        """except VetchInterrupt: catches all Vetch interventions."""
        try:
            raise StallDetected("test", wasted_cost_usd=2.5, request_count=8)
        except VetchInterrupt as e:
            assert isinstance(e, StallDetected)
            assert e.wasted_cost_usd == 2.5
            assert e.request_count == 8

    def test_attributes_preserved(self) -> None:
        """Exception attributes are accessible on the instance."""
        e = StallDetected(
            "msg", wasted_cost_usd=3.14, request_count=12,
            fallback_model="gpt-4o-mini",
        )
        assert e.wasted_cost_usd == 3.14
        assert e.request_count == 12
        assert e.fallback_model == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Session detection
# ---------------------------------------------------------------------------


class TestSessionStallDetection:
    def test_flag_unset_initially(self) -> None:
        """A fresh session has no stall."""
        session = Session(emit=False)
        assert session.stall_triggered is False
        assert session.stall_advisory is None

    def test_flag_set_after_stall_pattern(self) -> None:
        """STALL-001 pattern flips the flag and stores the advisory."""
        session = Session(emit=False)
        _stall_a_session(session)
        assert session.stall_triggered is True
        assert session.stall_advisory is not None
        assert session.stall_advisory.code == "STALL-001"
        assert session.stall_advisory.request_count > 0

    def test_flag_persists_after_trigger(self) -> None:
        """Once triggered, additional events don't re-trigger or reset."""
        session = Session(emit=False)
        _stall_a_session(session)
        first_advisory = session.stall_advisory
        # Push more events — flag should stay set, advisory unchanged.
        for _ in range(5):
            session.register_event({  # type: ignore[arg-type]
                "model": "gpt-4o",
                "usage": {"text": {"input_tokens": 500, "output_tokens": 1}},
                "estimated_cost_usd": 0.05,
            })
        assert session.stall_triggered is True
        assert session.stall_advisory is first_advisory

    def test_lazy_gating_skips_early_calls(self) -> None:
        """Detection cycle is skipped for the first ~10 calls (perf)."""
        session = Session(emit=False)
        # Patch generate_advisories to detect whether it's called.
        with patch("vetch.advisory.generate_advisories") as mock_gen:
            for _ in range(5):
                session.register_event({  # type: ignore[arg-type]
                    "model": "gpt-4o",
                    "usage": {"text": {"input_tokens": 500, "output_tokens": 1}},
                    "estimated_cost_usd": 0.05,
                })
            assert mock_gen.call_count == 0  # never called below threshold

    def test_session_isolation(self) -> None:
        """Stalling one session doesn't affect another."""
        s1 = Session(emit=False)
        s2 = Session(emit=False)
        _stall_a_session(s1)
        # s2 saw nothing
        assert s1.stall_triggered is True
        assert s2.stall_triggered is False

    def test_clear_stall_resets_flag(self) -> None:
        """clear_stall() re-arms the breaker."""
        session = Session(emit=False)
        _stall_a_session(session)
        assert session.stall_triggered is True
        session.clear_stall()
        assert session.stall_triggered is False
        assert session.stall_advisory is None
        # Subsequent events can re-trigger.
        _stall_a_session(session)
        assert session.stall_triggered is True


# ---------------------------------------------------------------------------
# apply_stall_action behaviour (the helper used by all providers)
# ---------------------------------------------------------------------------


def _setup_stalled_session() -> Session:
    """Create + enter a Session and trip the stall flag inside it."""
    session = Session(emit=False)
    session.__enter__()
    _stall_a_session(session)
    return session


class TestApplyStallAction:
    def test_log_action_does_nothing(self) -> None:
        """Default 'log' returns (False, None) without side effects."""
        session = _setup_stalled_session()
        try:
            kwargs: dict[str, Any] = {"model": "gpt-4o"}
            rerouted, original = apply_stall_action(kwargs, None)
            assert rerouted is False
            assert original is None
            assert kwargs["model"] == "gpt-4o"  # unchanged
        finally:
            session.__exit__(None, None, None)

    def test_warn_action_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        """'warn' logs a warning, returns (False, None)."""
        set_stall_action("warn")
        session = _setup_stalled_session()
        try:
            kwargs: dict[str, Any] = {"model": "gpt-4o"}
            with caplog.at_level(logging.WARNING, logger="vetch._stall"):
                rerouted, original = apply_stall_action(kwargs, None)
            assert rerouted is False
            assert any("STALL-001" in r.message for r in caplog.records)
        finally:
            session.__exit__(None, None, None)

    def test_kill_action_raises_stall_detected(self) -> None:
        """'kill' raises StallDetected with populated attributes."""
        set_stall_action("kill")
        session = _setup_stalled_session()
        try:
            with pytest.raises(StallDetected) as exc_info:
                apply_stall_action({"model": "gpt-4o"}, None)
            assert exc_info.value.request_count > 0
            assert exc_info.value.wasted_cost_usd > 0
        finally:
            session.__exit__(None, None, None)

    def test_reroute_action_substitutes_model(self) -> None:
        """'reroute' replaces kwargs['model'] and returns original."""
        set_stall_action("reroute", fallback_model="gpt-4o-mini")
        session = _setup_stalled_session()
        try:
            ctx = TrackingContext()
            kwargs: dict[str, Any] = {"model": "gpt-4o"}
            rerouted, original = apply_stall_action(kwargs, ctx)
            assert rerouted is True
            assert original == "gpt-4o"
            assert kwargs["model"] == "gpt-4o-mini"
            assert any("STALL-001 reroute" in w for w in ctx.warnings)
        finally:
            session.__exit__(None, None, None)

    def test_reroute_no_model_in_kwargs_warns_and_skips(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When model is on the bound instance (Vertex AI style), reroute
        degrades gracefully with a warning.
        """
        set_stall_action("reroute", fallback_model="gpt-4o-mini")
        session = _setup_stalled_session()
        try:
            kwargs: dict[str, Any] = {}  # no "model" key
            with caplog.at_level(logging.WARNING, logger="vetch._stall"):
                rerouted, original = apply_stall_action(kwargs, None)
            assert rerouted is False
            assert original is None
            assert "model name is not in kwargs" in caplog.text
        finally:
            session.__exit__(None, None, None)

    def test_no_active_session_is_noop(self) -> None:
        """Without an active session, the helper is a no-op."""
        set_stall_action("kill")
        # No session entered.
        rerouted, original = apply_stall_action({"model": "gpt-4o"}, None)
        assert rerouted is False
        assert original is None

    def test_session_without_stall_is_noop(self) -> None:
        """Active session that hasn't tripped is a no-op even with action='kill'."""
        set_stall_action("kill")
        with Session(emit=False) as session:
            assert session.stall_triggered is False
            rerouted, original = apply_stall_action({"model": "gpt-4o"}, None)
            assert rerouted is False

    def test_fail_open_on_helper_internal_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """If something inside apply_stall_action breaks, return (False, None).

        StallDetected itself must still propagate, but other exceptions
        (e.g. broken get_stall_action) must not crash the LLM call.
        """
        session = _setup_stalled_session()
        try:
            with patch(
                "vetch.config.get_stall_action",
                side_effect=RuntimeError("boom"),
            ):
                with caplog.at_level(logging.WARNING, logger="vetch._stall"):
                    rerouted, original = apply_stall_action(
                        {"model": "gpt-4o"}, None
                    )
                assert rerouted is False
                assert original is None
                assert any("fail-open" in r.message for r in caplog.records)
        finally:
            session.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Param-mismatch heuristic (fail-open reroute)
# ---------------------------------------------------------------------------


class TestParamMismatchHeuristic:
    def test_named_exceptions_match(self) -> None:
        """Known exception class names are detected as param mismatch."""
        for name in ("BadRequestError", "InvalidRequestError", "InvalidArgument"):
            exc = type(name, (Exception,), {})("bad")
            assert looks_like_param_mismatch(exc) is True

    def test_status_400_matches(self) -> None:
        """A non-auth 4xx status_code triggers retry."""
        e = Exception("bad")
        e.status_code = 400  # type: ignore[attr-defined]
        assert looks_like_param_mismatch(e) is True

    def test_auth_and_ratelimit_do_not_match(self) -> None:
        """401, 403, 429 do NOT trigger retry — these would fail with original
        model too, retrying just doubles the failure.
        """
        for code in (401, 403, 429):
            e = Exception("err")
            e.status_code = code  # type: ignore[attr-defined]
            assert looks_like_param_mismatch(e) is False

    def test_5xx_does_not_match(self) -> None:
        """Server errors are not param mismatches."""
        e = Exception("err")
        e.status_code = 500  # type: ignore[attr-defined]
        assert looks_like_param_mismatch(e) is False

    def test_unrelated_exception_does_not_match(self) -> None:
        """A bare Exception with no status doesn't trigger retry."""
        assert looks_like_param_mismatch(Exception("?")) is False
        assert looks_like_param_mismatch(TimeoutError("nope")) is False


# ---------------------------------------------------------------------------
# Public API export
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_top_level_exports(self) -> None:
        """All v0.4.0 symbols are reachable via the top-level vetch module."""
        assert vetch.set_stall_action is set_stall_action
        assert vetch.get_stall_action is get_stall_action
        assert vetch.StallDetected is StallDetected
        assert vetch.VetchInterrupt is VetchInterrupt


# ---------------------------------------------------------------------------
# v0.4.0 UACA subset: explicit p5/p95 uncertainty bounds on inference events
# ---------------------------------------------------------------------------


class TestUncertaintyBounds:
    """The wrapper now emits explicit absolute lower/upper energy and carbon
    bounds derived from the existing ``energy_uncertainty_pct`` and the
    point estimate. No new modelling — just exposes the uncertainty as
    absolute numbers downstream tools can use without recomputing.
    """

    def _build_event_with_metrics(
        self,
        energy_wh: float,
        carbon_g: float,
        uncertainty_pct: int,
    ) -> dict:
        """Run the bound calculation in isolation, mirroring the wrapper logic."""
        # We replicate the wrapper math directly — the wrapper itself requires
        # a full SDK patching cycle to exercise end-to-end. The arithmetic is
        # tiny and worth a focused unit test.
        band_e = energy_wh * (uncertainty_pct / 100.0)
        band_c = carbon_g * (uncertainty_pct / 100.0)
        return {
            "energy_p5_wh": max(energy_wh - band_e, 0.0),
            "energy_p95_wh": energy_wh + band_e,
            "carbon_p5_g": max(carbon_g - band_c, 0.0),
            "carbon_p95_g": carbon_g + band_c,
        }

    def test_tier1_50pct_band(self) -> None:
        """Tier 1 measured = ±50% band."""
        e = self._build_event_with_metrics(10.0, 4.0, 50)
        assert e["energy_p5_wh"] == 5.0
        assert e["energy_p95_wh"] == 15.0
        assert e["carbon_p5_g"] == 2.0
        assert e["carbon_p95_g"] == 6.0

    def test_tier3_1000pct_band_clamped_at_zero(self) -> None:
        """Tier 3 estimates have ±1000% bands. p5 must clamp at 0 (no negatives)."""
        e = self._build_event_with_metrics(0.5, 0.2, 1000)
        assert e["energy_p5_wh"] == 0.0  # clamped, not -4.5
        assert e["energy_p95_wh"] == 5.5
        assert e["carbon_p5_g"] == 0.0
        assert e["carbon_p95_g"] == 2.2

    def test_schema_includes_p5_p95_fields(self) -> None:
        """The InferenceEvent TypedDict declares the new fields (no surprise
        when downstream consumers introspect the schema).
        """
        from vetch.schema import InferenceEvent

        annotations = InferenceEvent.__annotations__
        for field in ("energy_p5_wh", "energy_p95_wh", "carbon_p5_g", "carbon_p95_g"):
            assert field in annotations, f"InferenceEvent missing {field}"
