"""Tests for vetch.Session aggregation."""

from __future__ import annotations

import pytest


class TestSession:
    """Tests for Session class."""

    def test_session_exported(self) -> None:
        """Session is in __all__ exports."""
        import vetch

        assert "Session" in vetch.__all__

    def test_session_auto_generates_id(self) -> None:
        """Session auto-generates session_id if not provided."""
        import vetch

        session = vetch.Session()
        assert session.session_id is not None
        assert len(session.session_id) > 0

    def test_session_uses_provided_id(self) -> None:
        """Session uses provided session_id."""
        import vetch

        session = vetch.Session(session_id="custom-id-123")
        assert session.session_id == "custom-id-123"

    def test_session_stores_tags(self) -> None:
        """Session stores provided tags."""
        import vetch

        session = vetch.Session(tags={"agent": "researcher", "env": "test"})
        assert session.tags == {"agent": "researcher", "env": "test"}

    def test_session_initial_values(self) -> None:
        """Session starts with zero accumulated values."""
        import vetch

        session = vetch.Session()
        assert session.total_energy_wh == 0.0
        assert session.total_carbon_g == 0.0
        assert session.total_cost_usd == 0.0
        assert session.call_count == 0
        assert session.total_input_tokens == 0
        assert session.total_output_tokens == 0


class TestSessionContextManager:
    """Tests for Session context manager behavior."""

    def test_session_context_manager(self) -> None:
        """Session works as context manager."""
        import vetch

        with vetch.Session(emit=False) as session:
            assert session is not None
            assert session.session_id is not None

    def test_session_tracks_duration(self) -> None:
        """Session tracks duration."""
        import time

        import vetch

        with vetch.Session(emit=False) as session:
            time.sleep(0.01)  # 10ms

        assert session.duration_ms is not None
        assert session.duration_ms >= 10

    def test_nested_session_gets_parent_id(self) -> None:
        """Nested session automatically gets parent_session_id."""
        import vetch

        with vetch.Session(emit=False) as parent:
            with vetch.Session(emit=False) as child:
                assert child.parent_session_id == parent.session_id

    def test_explicit_parent_session_id(self) -> None:
        """Explicit parent_session_id is used."""
        import vetch

        child = vetch.Session(parent_session_id="explicit-parent-123", emit=False)
        assert child.parent_session_id == "explicit-parent-123"


class TestSessionAccumulation:
    """Tests for Session event accumulation."""

    def test_register_event_increments_count(self) -> None:
        """register_event increments call_count."""
        import vetch

        session = vetch.Session(emit=False)
        event = {
            "model": "gpt-4o",
            "provider": "openai",
            "estimated_energy_wh": 0.1,
            "estimated_carbon_g": 5.0,
            "estimated_cost_usd": 0.01,
            "usage": {"text": {"input_tokens": 100, "output_tokens": 50}},
        }
        session.register_event(event)  # type: ignore[arg-type]

        assert session.call_count == 1

    def test_register_event_accumulates_metrics(self) -> None:
        """register_event accumulates energy, carbon, cost."""
        import vetch

        session = vetch.Session(emit=False)
        event1 = {
            "estimated_energy_wh": 0.1,
            "estimated_carbon_g": 5.0,
            "estimated_cost_usd": 0.01,
        }
        event2 = {
            "estimated_energy_wh": 0.2,
            "estimated_carbon_g": 10.0,
            "estimated_cost_usd": 0.02,
        }
        session.register_event(event1)  # type: ignore[arg-type]
        session.register_event(event2)  # type: ignore[arg-type]

        assert session.total_energy_wh == pytest.approx(0.3)
        assert session.total_carbon_g == pytest.approx(15.0)
        assert session.total_cost_usd == pytest.approx(0.03)

    def test_register_event_accumulates_tokens(self) -> None:
        """register_event accumulates input/output tokens."""
        import vetch

        session = vetch.Session(emit=False)
        event1 = {"usage": {"text": {"input_tokens": 100, "output_tokens": 50}}}
        event2 = {"usage": {"text": {"input_tokens": 200, "output_tokens": 100}}}
        session.register_event(event1)  # type: ignore[arg-type]
        session.register_event(event2)  # type: ignore[arg-type]

        assert session.total_input_tokens == 300
        assert session.total_output_tokens == 150

    def test_register_event_tracks_models(self) -> None:
        """register_event tracks unique models used."""
        import vetch

        session = vetch.Session(emit=False)
        session.register_event({"model": "gpt-4o", "provider": "openai"})  # type: ignore[arg-type]
        session.register_event({"model": "gpt-4o", "provider": "openai"})  # type: ignore[arg-type]
        session.register_event({"model": "claude-3-5-sonnet", "provider": "anthropic"})  # type: ignore[arg-type]

        assert set(session.models_used) == {"gpt-4o", "claude-3-5-sonnet"}

    def test_register_event_tracks_providers(self) -> None:
        """register_event tracks unique providers used."""
        import vetch

        session = vetch.Session(emit=False)
        session.register_event({"model": "gpt-4o", "provider": "openai"})  # type: ignore[arg-type]
        session.register_event({"model": "claude-3-5-sonnet", "provider": "anthropic"})  # type: ignore[arg-type]

        assert set(session.providers_used) == {"openai", "anthropic"}

    def test_register_event_tracks_errors(self) -> None:
        """register_event tracks error count."""
        import vetch

        session = vetch.Session(emit=False)
        session.register_event({"model": "gpt-4o", "error": False})  # type: ignore[arg-type]
        session.register_event({"model": "gpt-4o", "error": True})  # type: ignore[arg-type]
        session.register_event({"model": "gpt-4o", "error": True})  # type: ignore[arg-type]

        assert session._errors == 2


class TestSessionHeaders:
    """Tests for distributed session propagation via headers."""

    def test_inject_headers_adds_session_id(self) -> None:
        """inject_headers adds session ID to headers."""
        import vetch

        session = vetch.Session(session_id="test-session-123", emit=False)
        headers: dict[str, str] = {}
        result = session.inject_headers(headers)

        assert result["X-Vetch-Session-Id"] == "test-session-123"

    def test_inject_headers_adds_parent_id(self) -> None:
        """inject_headers adds parent session ID if present."""
        import vetch

        session = vetch.Session(
            session_id="child-123",
            parent_session_id="parent-456",
            emit=False,
        )
        headers: dict[str, str] = {}
        result = session.inject_headers(headers)

        assert result["X-Vetch-Session-Id"] == "child-123"
        assert result["X-Vetch-Parent-Session-Id"] == "parent-456"

    def test_from_headers_creates_child_session(self) -> None:
        """from_headers creates child session with parent link."""
        import vetch

        headers = {"X-Vetch-Session-Id": "parent-session-789"}
        child = vetch.Session.from_headers(headers, emit=False)

        assert child.parent_session_id == "parent-session-789"
        assert child.session_id != "parent-session-789"  # New ID generated

    def test_from_headers_accepts_tags(self) -> None:
        """from_headers accepts additional tags."""
        import vetch

        headers = {"X-Vetch-Session-Id": "parent-123"}
        child = vetch.Session.from_headers(headers, tags={"worker": "celery"}, emit=False)

        assert child.tags == {"worker": "celery"}


class TestSessionToDict:
    """Tests for Session.to_dict() method."""

    def test_to_dict_returns_state(self) -> None:
        """to_dict returns session state as dict."""
        import vetch

        session = vetch.Session(
            session_id="test-123",
            parent_session_id="parent-456",
            tags={"env": "test"},
            emit=False,
        )
        session.register_event({  # type: ignore[arg-type]
            "model": "gpt-4o",
            "provider": "openai",
            "estimated_energy_wh": 0.5,
            "estimated_carbon_g": 25.0,
            "estimated_cost_usd": 0.05,
        })

        result = session.to_dict()

        assert result["session_id"] == "test-123"
        assert result["parent_session_id"] == "parent-456"
        assert result["tags"] == {"env": "test"}
        assert result["call_count"] == 1
        assert result["total_energy_wh"] == 0.5
        assert result["total_carbon_g"] == 25.0
        assert result["total_cost_usd"] == 0.05


class TestSessionAsync:
    """Tests for async Session support."""

    @pytest.mark.asyncio
    async def test_session_async_context_manager(self) -> None:
        """Session works as async context manager."""
        import vetch

        async with vetch.Session(emit=False) as session:
            assert session is not None
            assert session.session_id is not None

    @pytest.mark.asyncio
    async def test_session_async_tracks_duration(self) -> None:
        """Async session tracks duration."""
        import asyncio

        import vetch

        async with vetch.Session(emit=False) as session:
            await asyncio.sleep(0.01)  # 10ms

        assert session.duration_ms is not None
        assert session.duration_ms >= 10


class TestGetActiveSession:
    """Tests for get_active_session() function."""

    def test_get_active_session_returns_none_outside_context(self) -> None:
        """get_active_session returns None when not in session."""
        from vetch.session import get_active_session

        assert get_active_session() is None

    def test_get_active_session_returns_session_in_context(self) -> None:
        """get_active_session returns active session."""
        import vetch
        from vetch.session import get_active_session

        with vetch.Session(emit=False) as session:
            active = get_active_session()
            assert active is session

    def test_get_active_session_returns_none_after_context(self) -> None:
        """get_active_session returns None after context exits."""
        import vetch
        from vetch.session import get_active_session

        with vetch.Session(emit=False):
            pass

        assert get_active_session() is None


class TestSessionResumption:
    """Tests for session resumption via from_headers(resume=True)."""

    def test_from_headers_resume_uses_same_session_id(self) -> None:
        """from_headers with resume=True uses the same session_id."""
        import vetch

        headers = {"X-Vetch-Session-Id": "original-session-123"}
        resumed = vetch.Session.from_headers(headers, resume=True, emit=False)

        assert resumed.session_id == "original-session-123"

    def test_from_headers_resume_preserves_parent_chain(self) -> None:
        """from_headers with resume=True preserves parent_session_id."""
        import vetch

        headers = {
            "X-Vetch-Session-Id": "child-session",
            "X-Vetch-Parent-Session-Id": "parent-session",
        }
        resumed = vetch.Session.from_headers(headers, resume=True, emit=False)

        assert resumed.session_id == "child-session"
        assert resumed.parent_session_id == "parent-session"

    def test_from_headers_resume_false_creates_child(self) -> None:
        """from_headers with resume=False (default) creates child session."""
        import vetch

        headers = {"X-Vetch-Session-Id": "parent-session-789"}
        child = vetch.Session.from_headers(headers, resume=False, emit=False)

        assert child.session_id != "parent-session-789"  # New ID
        assert child.parent_session_id == "parent-session-789"  # Linked to parent


class TestSessionCacheMetrics:
    """Tests for session cache metrics tracking."""

    def test_session_initial_cache_values(self) -> None:
        """Session starts with zero cache token values."""
        import vetch

        session = vetch.Session(emit=False)
        assert session.total_cache_read_tokens == 0
        assert session.total_cache_creation_tokens == 0

    def test_register_event_accumulates_cache_read_tokens(self) -> None:
        """register_event accumulates cache_read_tokens."""
        import vetch

        session = vetch.Session(emit=False)
        session.register_event({"cache_read_tokens": 100})  # type: ignore[arg-type]
        session.register_event({"cache_read_tokens": 200})  # type: ignore[arg-type]

        assert session.total_cache_read_tokens == 300

    def test_register_event_accumulates_cache_creation_tokens(self) -> None:
        """register_event accumulates cache_creation_tokens."""
        import vetch

        session = vetch.Session(emit=False)
        session.register_event({"cache_creation_tokens": 50})  # type: ignore[arg-type]
        session.register_event({"cache_creation_tokens": 75})  # type: ignore[arg-type]

        assert session.total_cache_creation_tokens == 125

    def test_to_dict_includes_cache_metrics(self) -> None:
        """to_dict includes cache metrics."""
        import vetch

        session = vetch.Session(emit=False)
        session.register_event({  # type: ignore[arg-type]
            "cache_read_tokens": 100,
            "cache_creation_tokens": 50,
        })

        result = session.to_dict()

        assert result["total_cache_read_tokens"] == 100
        assert result["total_cache_creation_tokens"] == 50


class TestSessionMemorySafeguards:
    """Tests for OOM prevention in long-running agentic loops."""

    def test_max_calls_default(self) -> None:
        """Default max_calls is 10,000."""
        import vetch
        from vetch.session import DEFAULT_MAX_CALLS

        session = vetch.Session(emit=False)
        assert session._max_calls == DEFAULT_MAX_CALLS

    def test_max_calls_custom(self) -> None:
        """Custom max_calls is respected."""
        import vetch

        session = vetch.Session(emit=False, max_calls=5)
        assert session._max_calls == 5

    def test_max_calls_stops_metadata_growth(self) -> None:
        """After max_calls, metadata sets stop growing."""
        import vetch

        session = vetch.Session(emit=False, max_calls=3)

        # Register 3 events (at limit)
        for i in range(3):
            session.register_event({  # type: ignore[arg-type]
                "model": f"model-{i}",
                "provider": f"provider-{i}",
                "estimated_cost_usd": 0.01,
            })

        assert session.call_count == 3
        assert len(session.models_used) == 3

        # Register 2 more beyond limit
        for i in range(3, 5):
            session.register_event({  # type: ignore[arg-type]
                "model": f"model-{i}",
                "provider": f"provider-{i}",
                "estimated_cost_usd": 0.01,
            })

        # Call count still increments, but models don't grow
        assert session.call_count == 5
        assert len(session.models_used) == 3  # Frozen at limit

    def test_max_calls_still_accumulates_scalars(self) -> None:
        """After max_calls, scalar metrics still accumulate."""
        import vetch

        session = vetch.Session(emit=False, max_calls=2)

        # Fill up
        session.register_event({  # type: ignore[arg-type]
            "estimated_cost_usd": 1.0,
            "estimated_energy_wh": 0.1,
        })
        session.register_event({  # type: ignore[arg-type]
            "estimated_cost_usd": 2.0,
            "estimated_energy_wh": 0.2,
        })

        # Beyond limit
        session.register_event({  # type: ignore[arg-type]
            "estimated_cost_usd": 3.0,
            "estimated_energy_wh": 0.3,
        })

        assert session.total_cost_usd == pytest.approx(6.0)
        assert session.total_energy_wh == pytest.approx(0.6)

    def test_metadata_sets_capped(self) -> None:
        """Models/providers sets are capped at DEFAULT_MAX_METADATA_ITEMS."""
        import vetch
        from vetch.session import DEFAULT_MAX_METADATA_ITEMS

        session = vetch.Session(emit=False, max_calls=0)  # Unlimited calls

        # Add more than the cap
        for i in range(DEFAULT_MAX_METADATA_ITEMS + 10):
            session.register_event({  # type: ignore[arg-type]
                "model": f"model-{i}",
                "provider": f"provider-{i}",
            })

        assert len(session.models_used) == DEFAULT_MAX_METADATA_ITEMS
        assert len(session.providers_used) == DEFAULT_MAX_METADATA_ITEMS

    def test_saturated_flag_set(self) -> None:
        """Session sets _saturated flag when max_calls exceeded."""
        import vetch

        session = vetch.Session(emit=False, max_calls=1)

        session.register_event({"estimated_cost_usd": 1.0})  # type: ignore[arg-type]
        assert session._saturated is False

        session.register_event({"estimated_cost_usd": 2.0})  # type: ignore[arg-type]
        assert session._saturated is True


class TestSessionStatsIsolation:
    """Tests for per-session SessionStats (stall detection isolation)."""

    def test_session_has_own_stats(self) -> None:
        """Each Session carries its own SessionStats instance."""
        import vetch

        s1 = vetch.Session(emit=False)
        s2 = vetch.Session(emit=False)
        assert s1.stats is not s2.stats

    def test_register_event_updates_session_stats(self) -> None:
        """register_event feeds the per-session SessionStats."""
        import vetch

        session = vetch.Session(emit=False)
        session.register_event({  # type: ignore[arg-type]
            "model": "gpt-4o",
            "usage": {"text": {"input_tokens": 500, "output_tokens": 0}},
            "estimated_cost_usd": 0.10,
        })

        assert session.stats.total_requests == 1
        assert session.stats.total_cost_usd == pytest.approx(0.10)
        assert len(session.stats.recent_calls) == 1

    def test_sessions_do_not_share_stall_data(self) -> None:
        """Two sessions have fully isolated stall detection state."""
        import vetch

        s1 = vetch.Session(emit=False)
        s2 = vetch.Session(emit=False)

        # s1: 15 stalled calls
        for _ in range(15):
            s1.register_event({  # type: ignore[arg-type]
                "model": "gpt-4o",
                "usage": {"text": {"input_tokens": 500, "output_tokens": 0}},
                "estimated_cost_usd": 0.10,
            })

        # s2: 15 healthy calls
        for _ in range(15):
            s2.register_event({  # type: ignore[arg-type]
                "model": "gpt-4o",
                "usage": {"text": {"input_tokens": 500, "output_tokens": 200}},
                "estimated_cost_usd": 0.10,
            })

        summary1 = s1.stats.summary()
        summary2 = s2.stats.summary()

        assert summary1["recent_low_output_count"] == 15
        assert summary2["recent_low_output_count"] == 0
