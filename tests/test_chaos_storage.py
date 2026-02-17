"""Chaos tests for storage and advisory subsystems.

Verifies fail-open behavior when:
- SQLite database is corrupt
- Filesystem permissions deny access
- Statistics are malformed
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

from vetch.advisory import generate_advisories
from vetch.stats import SessionStats
from vetch.storage import configure_storage, query_usage, store_event


class TestChaosStorage:
    """Chaos tests for local SQLite storage."""

    def test_corrupt_database_file(self) -> None:
        """Verify fail-open when DB file is corrupt."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"NOT A SQLITE DB")
            db_path = Path(tmp.name)

        try:
            configure_storage(enabled=True, path=db_path)

            # Should not raise exception
            store_event({
                "schema_version": "1",
                "model": "gpt-4o",
                "estimated_energy_wh": 1.0
            })

            # Query should fail gracefully (return empty summary or handle error)
            # In our implementation, query_usage connects to DB.
            # SQLite might raise DatabaseError.
            try:
                from datetime import datetime
                query_usage(datetime.now(), datetime.now())
            except sqlite3.DatabaseError:
                # Acceptable behavior: raise DB error on explicit query
                # But store_event (background) should definitely NOT crash app
                pass

        finally:
            if db_path.exists():
                os.unlink(db_path)

    def test_readonly_filesystem_storage(self) -> None:
        """Verify fail-open when DB directory is not writable."""
        # Simulate readonly by patching mkdir/connect
        with patch("sqlite3.connect", side_effect=sqlite3.OperationalError("Read-only")):
            configure_storage(enabled=True)
            # Should silently fail to store
            store_event({"model": "test"})


class TestChaosAdvisory:
    """Chaos tests for advisory engine."""

    def test_malformed_stats(self) -> None:
        """Verify advisory generation with unusual stats."""
        stats = SessionStats()
        stats.total_requests = 10
        # No input tokens recorded

        advisories = generate_advisories(stats)
        # Should not crash division by zero
        assert isinstance(advisories, list)

    def test_extreme_values(self) -> None:
        """Verify handling of extreme stats."""
        stats = SessionStats()
        stats.total_requests = 1000
        stats.total_input_tokens = 10**9
        stats.total_output_tokens = 0

        advisories = generate_advisories(stats)
        # Should trigger RAG warning
        rag_warnings = [a for a in advisories if a.code == "RAG-001"]
        assert len(rag_warnings) == 1
