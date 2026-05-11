"""Tests for local storage module.

These tests verify:
- Storage configuration
- Event storage
- Usage queries
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from vetch.storage import (
    UsageSummary,
    compact_storage,
    configure_storage,
    flush_storage,
    get_db_path,
    get_top_consumers,
    is_storage_enabled,
    query_daily_usage,
    query_events,
    query_usage,
    store_event,
)


class TestStorageConfiguration:
    """Tests for storage configuration."""

    def test_default_disabled(self) -> None:
        """Storage is disabled by default."""
        # Reset to check default state
        import vetch.storage
        vetch.storage._STORAGE_ENABLED = False

        assert not is_storage_enabled()

    def test_configure_enabled(self) -> None:
        """Enable storage with custom path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            configure_storage(enabled=True, path=db_path)

            assert is_storage_enabled()
            assert get_db_path() == db_path

    def test_configure_disabled(self) -> None:
        """Disable storage."""
        configure_storage(enabled=False)
        assert not is_storage_enabled()


class TestEventStorage:
    """Tests for storing events."""

    def test_store_event_when_disabled(self) -> None:
        """Store returns early when disabled."""
        configure_storage(enabled=False)

        event = {
            "event_id": "test-123",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": "gpt-4o",
        }

        # Should not raise, just return silently
        store_event(event)

    def test_store_event_creates_db(self) -> None:
        """Storing event creates database if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            configure_storage(enabled=True, path=db_path)

            event = {
                "event_id": "test-456",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": "gpt-4o",
                "provider": "openai",
                "usage": {"text": {"input_tokens": 100, "output_tokens": 50}},
                "estimated_energy_wh": 0.001,
                "estimated_carbon_g": 0.05,
                "estimated_cost_usd": 0.01,
                "tags": {"team": "ml"},
            }

            store_event(event)
            flush_storage()

            # Database should now exist
            assert db_path.exists()


class TestUsageSummary:
    """Tests for UsageSummary class."""

    def test_create_summary(self) -> None:
        """Create a UsageSummary."""
        now = datetime.now(timezone.utc)
        summary = UsageSummary(start=now, end=now)

        assert summary.total_requests == 0
        assert summary.total_energy_wh == 0.0
        assert summary.total_carbon_g == 0.0
        assert summary.total_cost_usd == 0.0

    def test_summary_to_dict(self) -> None:
        """Convert summary to dict."""
        now = datetime.now(timezone.utc)
        summary = UsageSummary(start=now, end=now)
        summary.total_requests = 5
        summary.total_cost_usd = 0.50

        result = summary.to_dict()

        assert "period" in result
        assert result["totals"]["requests"] == 5
        assert result["totals"]["cost_usd"] == 0.50


class TestQueryUsage:
    """Tests for querying usage."""

    def test_query_empty_database(self) -> None:
        """Query returns empty summary for new database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "empty.db"
            configure_storage(enabled=True, path=db_path)

            now = datetime.now(timezone.utc)
            summary = query_usage(
                start=now - timedelta(days=1),
                end=now,
            )

            assert summary.total_requests == 0

    def test_query_nonexistent_database(self) -> None:
        """Query handles nonexistent database gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "nonexistent.db"
            configure_storage(enabled=True, path=db_path)

            # Don't create the database
            now = datetime.now(timezone.utc)
            summary = query_usage(
                start=now - timedelta(days=1),
                end=now,
            )

            assert summary.total_requests == 0

    def test_query_with_stored_events(self) -> None:
        """Query aggregates stored events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            configure_storage(enabled=True, path=db_path)

            now = datetime.now(timezone.utc)

            # Store some events
            for i in range(3):
                event = {
                    "event_id": f"test-{i}",
                    "timestamp": now.isoformat(),
                    "model": "gpt-4o",
                    "provider": "openai",
                    "usage": {"text": {"input_tokens": 100, "output_tokens": 50}},
                    "estimated_energy_wh": 0.001,
                    "estimated_carbon_g": 0.05,
                    "estimated_cost_usd": 0.01,
                    "tags": {},
                }
                store_event(event)

            summary = query_usage(
                start=now - timedelta(hours=1),
                end=now + timedelta(hours=1),
            )

            assert summary.total_requests == 3
            assert summary.total_cost_usd == pytest.approx(0.03, rel=0.1)

    def test_query_filters_by_model(self) -> None:
        """Query can filter by model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            configure_storage(enabled=True, path=db_path)

            now = datetime.now(timezone.utc)

            # Store events with different models
            store_event({
                "event_id": "gpt-1",
                "timestamp": now.isoformat(),
                "model": "gpt-4o",
                "usage": {"text": {"input_tokens": 100, "output_tokens": 50}},
                "estimated_cost_usd": 0.01,
                "tags": {},
            })

            store_event({
                "event_id": "claude-1",
                "timestamp": now.isoformat(),
                "model": "claude-3-opus",
                "usage": {"text": {"input_tokens": 100, "output_tokens": 50}},
                "estimated_cost_usd": 0.02,
                "tags": {},
            })

            summary = query_usage(
                start=now - timedelta(hours=1),
                end=now + timedelta(hours=1),
                model="gpt-4o",
            )

            assert summary.total_requests == 1

    def test_query_filters_by_tags(self) -> None:
        """Query can filter by tags."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            configure_storage(enabled=True, path=db_path)

            now = datetime.now(timezone.utc)

            # Store events with different tags
            store_event({
                "event_id": "prod-1",
                "timestamp": now.isoformat(),
                "model": "gpt-4o",
                "usage": {"text": {"input_tokens": 100, "output_tokens": 50}},
                "estimated_cost_usd": 0.01,
                "tags": {"environment": "production", "team": "api"},
            })

            store_event({
                "event_id": "dev-1",
                "timestamp": now.isoformat(),
                "model": "gpt-4o",
                "usage": {"text": {"input_tokens": 100, "output_tokens": 50}},
                "estimated_cost_usd": 0.02,
                "tags": {"environment": "development", "team": "api"},
            })

            # Filter by environment tag
            summary = query_usage(
                start=now - timedelta(hours=1),
                end=now + timedelta(hours=1),
                tags={"environment": "production"},
            )

            assert summary.total_requests == 1
            assert summary.total_cost_usd == 0.01

    def test_query_daily_usage_survives_raw_compaction(self) -> None:
        """Daily aggregates remain available after raw rows are compacted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            configure_storage(enabled=True, path=db_path)

            now = datetime.now(timezone.utc)
            old = now - timedelta(days=2)
            store_event({
                "event_id": "old-1",
                "timestamp": old.isoformat(),
                "model": "gpt-4o",
                "provider": "openai",
                "usage": {"text": {"input_tokens": 120, "output_tokens": 30}},
                "estimated_energy_wh": 0.25,
                "estimated_carbon_g": 0.10,
                "estimated_cost_usd": 0.04,
                "tags": {"team": "ml", "feature": "rag"},
            })

            deleted = compact_storage(raw_retention_days=1)
            assert deleted == 1

            raw_events = query_events(start=old - timedelta(hours=1), end=now)
            assert raw_events == []

            summary = query_daily_usage(start=old - timedelta(hours=1), end=now)
            assert summary.total_requests == 1
            assert summary.total_input_tokens == 120
            assert summary.total_output_tokens == 30
            assert summary.total_energy_wh == pytest.approx(0.25)
            assert summary.by_tag["team"]["ml"]["energy_wh"] == pytest.approx(0.25)

    def test_top_consumers_includes_energy(self) -> None:
        """Top consumer output includes real tag-level energy aggregation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            configure_storage(enabled=True, path=db_path)

            now = datetime.now(timezone.utc)
            store_event({
                "event_id": "energy-1",
                "timestamp": now.isoformat(),
                "model": "gpt-4o",
                "provider": "openai",
                "usage": {"text": {"input_tokens": 100, "output_tokens": 50}},
                "estimated_energy_wh": 1.5,
                "estimated_carbon_g": 0.5,
                "estimated_cost_usd": 0.01,
                "tags": {"team": "platform"},
            })

            consumers = get_top_consumers(metric="energy", tag_key="team", days=1)
            assert consumers[0]["tag_value"] == "platform"
            assert consumers[0]["energy"] == pytest.approx(1.5)
