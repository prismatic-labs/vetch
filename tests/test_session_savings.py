"""Tests for session savings accumulation and circuit breaker intervention recording."""

from __future__ import annotations

import uuid

import pytest


class TestSessionSavingsAccumulation:
    """Tests for cache saving fields accumulated in Session.register_event."""

    def _make_event(
        self,
        cache_cost_saving_usd: float | None = None,
        cache_energy_saving_wh: float | None = None,
        cache_carbon_saving_g: float | None = None,
    ) -> dict:
        return {
            "event_id": str(uuid.uuid4()),
            "timestamp": "2026-01-01T00:00:00Z",
            "model": "claude-3-5-sonnet-20241022",
            "provider": "anthropic",
            "usage": {"text": {"input_tokens": 100, "output_tokens": 10}},
            "estimated_energy_wh": 0.001,
            "estimated_carbon_g": 0.0004,
            "estimated_cost_usd": 0.01,
            "cache_read_tokens": 80 if cache_cost_saving_usd else 0,
            "cache_creation_tokens": 0,
            "cache_hit": cache_cost_saving_usd is not None,
            "cache_cost_saving_usd": cache_cost_saving_usd,
            "cache_energy_saving_wh": cache_energy_saving_wh,
            "cache_carbon_saving_g": cache_carbon_saving_g,
            "is_stream": False,
            "is_batch": False,
            "is_embedding": False,
            "complete": True,
            "error": False,
            "tracking_disabled": False,
            "tracking_degraded": False,
            "usage_estimated": False,
            "usage_estimation_method": None,
        }

    def test_initial_savings_are_zero(self) -> None:
        from vetch.session import Session

        session = Session(emit=False)
        assert session.total_cache_cost_saving_usd == 0.0
        assert session.total_cache_energy_saving_wh == 0.0
        assert session.total_cache_carbon_saving_g == 0.0
        assert session.circuit_breaker_interventions == 0
        assert session.circuit_breaker_cost_at_risk_usd == 0.0

    def test_accumulates_cache_cost_saving(self) -> None:
        from vetch.session import Session

        session = Session(emit=False)
        session.register_event(self._make_event(cache_cost_saving_usd=0.05))
        session.register_event(self._make_event(cache_cost_saving_usd=0.03))
        assert abs(session.total_cache_cost_saving_usd - 0.08) < 1e-9

    def test_accumulates_cache_energy_saving(self) -> None:
        from vetch.session import Session

        session = Session(emit=False)
        session.register_event(self._make_event(cache_energy_saving_wh=0.002))
        session.register_event(self._make_event(cache_energy_saving_wh=0.001))
        assert abs(session.total_cache_energy_saving_wh - 0.003) < 1e-9

    def test_accumulates_cache_carbon_saving(self) -> None:
        from vetch.session import Session

        session = Session(emit=False)
        session.register_event(self._make_event(cache_carbon_saving_g=0.2))
        session.register_event(self._make_event(cache_carbon_saving_g=0.3))
        assert abs(session.total_cache_carbon_saving_g - 0.5) < 1e-9

    def test_none_savings_not_accumulated(self) -> None:
        """Events without cache savings (None) don't increment the total."""
        from vetch.session import Session

        session = Session(emit=False)
        session.register_event(self._make_event(cache_cost_saving_usd=None))
        session.register_event(self._make_event(cache_cost_saving_usd=None))
        assert session.total_cache_cost_saving_usd == 0.0

    def test_record_circuit_breaker_intervention(self) -> None:
        """Each stall episode counts once; clear_stall() re-arms for the next episode."""
        from vetch.session import Session

        session = Session(emit=False)
        # First intervention in episode 1
        recorded1 = session.record_circuit_breaker_intervention(cost_at_risk_usd=2.50)
        assert recorded1 is True
        # Second call in same episode is deduped
        recorded2 = session.record_circuit_breaker_intervention(cost_at_risk_usd=1.25)
        assert recorded2 is False

        assert session.circuit_breaker_interventions == 1
        assert abs(session.circuit_breaker_cost_at_risk_usd - 2.50) < 1e-9

        # After clear_stall() the next episode counts again
        session.clear_stall()
        recorded3 = session.record_circuit_breaker_intervention(cost_at_risk_usd=1.00)
        assert recorded3 is True
        assert session.circuit_breaker_interventions == 2
        assert abs(session.circuit_breaker_cost_at_risk_usd - 3.50) < 1e-9

    def test_record_intervention_zero_cost(self) -> None:
        from vetch.session import Session

        session = Session(emit=False)
        session.record_circuit_breaker_intervention()
        assert session.circuit_breaker_interventions == 1
        assert session.circuit_breaker_cost_at_risk_usd == 0.0

    def test_savings_fields_in_session_event_payload(self) -> None:
        """_emit_session_event builds a dict with the savings and intervention fields."""
        from unittest.mock import patch

        from vetch.session import Session

        session = Session(emit=False)
        session.register_event(self._make_event(cache_cost_saving_usd=0.10))
        session.register_event(self._make_event(cache_carbon_saving_g=0.20))
        session.record_circuit_breaker_intervention(5.00)

        captured = {}

        def fake_emit(event: dict) -> None:
            captured.update(event)

        with patch("vetch.session.emit_event", side_effect=fake_emit):
            session._emit_session_event()

        assert captured.get("total_cache_cost_saving_usd") == pytest.approx(0.10)
        assert captured.get("circuit_breaker_interventions") == 1
        assert captured.get("circuit_breaker_cost_at_risk_usd") == pytest.approx(5.00)
        assert captured.get("total_cache_energy_saving_wh") == pytest.approx(0.0)
        assert captured.get("total_cache_carbon_saving_g") == pytest.approx(0.20)

    def test_intervention_records_once_per_stall_episode(self) -> None:
        from vetch.session import Session

        session = Session(emit=False)
        assert session.record_circuit_breaker_intervention(2.0) is True
        assert session.record_circuit_breaker_intervention(2.0) is False
        assert session.circuit_breaker_interventions == 1
        assert session.circuit_breaker_cost_at_risk_usd == pytest.approx(2.0)
