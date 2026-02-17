"""Tests for local persistent storage."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path

from vetch.storage import _init_db, configure_storage, query_usage, store_event


class TestStorage:
    """Tests for SQLite storage."""

    def test_init_and_store(self) -> None:
        """Verify database initialization and event storage."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)

        try:
            configure_storage(enabled=True, path=db_path)
            _init_db()

            event = {
                "event_id": "test-storage",
                "timestamp": datetime.now().isoformat(),
                "model": "gpt-4o",
                "estimated_energy_wh": 0.5,
                "usage": {"text": {"input_tokens": 100, "output_tokens": 50}}
            }

            store_event(event)

            # Query back
            start = datetime(2020, 1, 1)
            end = datetime(2030, 1, 1)
            summary = query_usage(start, end)

            assert summary.total_requests == 1
            assert "gpt-4o" in summary.by_model
            assert summary.total_energy_wh == 0.5

        finally:
            if db_path.exists():
                os.unlink(db_path)
