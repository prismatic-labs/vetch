"""Local persistent storage for Vetch events.

Uses SQLite to store inference history locally, enabling:
- Historical reporting ('vetch report')
- Trend analysis
- Pattern detection

Privacy: Database is stored in user's home directory (~/.vetch/usage.db)
and is never uploaded unless explicitly configured via sync (future).
"""

from __future__ import annotations

import atexit
import json
import logging
import sqlite3
import threading
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from vetch.schema import InferenceEvent

logger = logging.getLogger(__name__)

# Track if warning has been issued (to avoid spam on repeated imports)
_EXPERIMENTAL_WARNING_ISSUED = False

DB_PATH = Path.home() / ".vetch" / "usage.db"
_STORAGE_ENABLED = False

# Connection pool for efficient SQLite access
_connection_pool: sqlite3.Connection | None = None
_connection_lock = threading.Lock()


def _get_connection() -> sqlite3.Connection:
    """Get or create a reusable SQLite connection (thread-safe).

    Maintains a single connection per process to avoid repeated open/close overhead.
    Connection is automatically closed on process exit.
    """
    global _connection_pool
    with _connection_lock:
        if _connection_pool is None:
            # Lazy init database if needed
            if not DB_PATH.exists():
                _init_db()
            _connection_pool = sqlite3.connect(DB_PATH, check_same_thread=False)
            _connection_pool.row_factory = sqlite3.Row
            # Register cleanup on exit
            atexit.register(_close_connection)
        return _connection_pool


def _close_connection() -> None:
    """Close the pooled connection on shutdown."""
    global _connection_pool
    if _connection_pool is not None:
        try:
            _connection_pool.close()
        except Exception:
            pass
        _connection_pool = None


def configure_storage(enabled: bool = True, path: Path | None = None) -> None:
    """Enable or disable local storage."""
    global _STORAGE_ENABLED, DB_PATH, _EXPERIMENTAL_WARNING_ISSUED, _connection_pool

    # Issue warning only once when storage is first enabled
    if enabled and not _EXPERIMENTAL_WARNING_ISSUED:
        _EXPERIMENTAL_WARNING_ISSUED = True
        warnings.warn(
            "vetch.storage is experimental. API may change in future versions.",
            FutureWarning,
            stacklevel=2,
        )

    _STORAGE_ENABLED = enabled
    if path and path != DB_PATH:
        # Close existing connection if path changed
        if _connection_pool is not None:
            try:
                _connection_pool.close()
            except Exception:
                pass
            _connection_pool = None
        DB_PATH = path


def is_storage_enabled() -> bool:
    return _STORAGE_ENABLED


def _init_db() -> None:
    """Initialize the SQLite database schema."""
    if not DB_PATH.parent.exists():
        try:
            DB_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        except OSError:
            return  # Fail safe if readonly fs

    conn = sqlite3.connect(DB_PATH)

    # Ensure restrictive permissions on the database file
    try:
        if DB_PATH.exists():
            DB_PATH.chmod(0o600)
    except OSError:
        pass  # Best effort
    try:
        cursor = conn.cursor()

        # Events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                model TEXT,
                provider TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                energy_wh REAL,
                carbon_g REAL,
                cost_usd REAL,
                tags_json TEXT,
                raw_json TEXT
            )
        """)

        # Index for reporting
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_model ON events(model)")

        conn.commit()
    finally:
        conn.close()


def store_event(event: InferenceEvent) -> None:
    """Store an event in the local database using pooled connection."""
    if not _STORAGE_ENABLED:
        return

    try:
        # Use pooled connection (thread-safe)
        conn = _get_connection()
        with _connection_lock:
            cursor = conn.cursor()

            # Extract fields for columns
            usage: dict[str, Any] = cast(dict[str, Any], event.get("usage", {}) or {})
            text_usage = usage.get("text", {}) or {}

            cursor.execute(
                """
                INSERT INTO events (
                    event_id, timestamp, model, provider,
                    input_tokens, output_tokens,
                    energy_wh, carbon_g, cost_usd,
                    tags_json, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.get("event_id"),
                    event.get("timestamp"),
                    event.get("model"),
                    event.get("provider"),
                    text_usage.get("input_tokens", 0),
                    text_usage.get("output_tokens", 0),
                    event.get("estimated_energy_wh"),
                    event.get("estimated_carbon_g"),
                    event.get("estimated_cost_usd"),
                    json.dumps(event.get("tags") or {}),
                    json.dumps(event)
                )
            )
            conn.commit()
    except Exception as e:
        logger.debug(f"Failed to store event locally: {e}")


# Reporting Queries

class UsageSummary:
    def __init__(self, start: datetime, end: datetime):
        self.start_time = start
        self.end_time = end
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_energy_wh = 0.0
        self.total_carbon_g = 0.0
        self.total_cost_usd = 0.0
        self.by_model: dict[str, Any] = {}
        self.by_tag: dict[str, Any] = {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "period": {
                "start": self.start_time.isoformat(),
                "end": self.end_time.isoformat()
            },
            "totals": {
                "requests": self.total_requests,
                "tokens": self.total_input_tokens + self.total_output_tokens,
                "energy_wh": self.total_energy_wh,
                "carbon_g": self.total_carbon_g,
                "cost_usd": self.total_cost_usd
            },
            "breakdown": {
                "model": self.by_model,
                "tags": self.by_tag
            }
        }


def query_usage(
    start: datetime,
    end: datetime,
    model: str | None = None,
    tags: dict[str, str] | None = None
) -> UsageSummary:
    """Query usage stats for a time period."""
    if not DB_PATH.exists():
        return UsageSummary(start, end)

    # Use connection pool for efficiency (avoids repeated open/close overhead)
    # Note: We don't close the pooled connection - it's managed by _close_connection() at exit
    conn = _get_connection()

    # Thread-safety: Use lock for query execution since connection has check_same_thread=False
    # While SQLite allows concurrent reads, we use the lock for simplicity and safety
    with _connection_lock:
        cursor = conn.cursor()

        # Build SQL query with filters pushed down to database
        query = "SELECT * FROM events WHERE timestamp BETWEEN ? AND ?"
        params = [start.isoformat(), end.isoformat()]

        # Add model filter at SQL level
        if model:
            query += " AND model = ?"
            params.append(model)

        # Add tag filters at SQL level using JSON1 functions
        if tags:
            for key, value in tags.items():
                # Use json_extract to check tag value at database level
                # This avoids loading all rows into Python memory
                query += " AND json_extract(tags_json, ?) = ?"
                params.append(f"$.{key}")
                params.append(value)

        summary = UsageSummary(start, end)

        # Query 1: Compute totals using SQL aggregation (not Python loops)
        totals_query = """
            SELECT
                COUNT(*) as total_requests,
                COALESCE(SUM(input_tokens), 0) as total_input_tokens,
                COALESCE(SUM(output_tokens), 0) as total_output_tokens,
                COALESCE(SUM(energy_wh), 0.0) as total_energy_wh,
                COALESCE(SUM(carbon_g), 0.0) as total_carbon_g,
                COALESCE(SUM(cost_usd), 0.0) as total_cost_usd
            FROM events
            WHERE timestamp BETWEEN ? AND ?
        """
        totals_params = [start.isoformat(), end.isoformat()]

        if model:
            totals_query += " AND model = ?"
            totals_params.append(model)

        if tags:
            for key, value in tags.items():
                totals_query += " AND json_extract(tags_json, ?) = ?"
                totals_params.append(f"$.{key}")
                totals_params.append(value)

        cursor.execute(totals_query, totals_params)
        totals_row = cursor.fetchone()
        if totals_row:
            summary.total_requests = totals_row['total_requests']
            summary.total_input_tokens = totals_row['total_input_tokens']
            summary.total_output_tokens = totals_row['total_output_tokens']
            summary.total_energy_wh = totals_row['total_energy_wh']
            summary.total_carbon_g = totals_row['total_carbon_g']
            summary.total_cost_usd = totals_row['total_cost_usd']

        # Query 2: Group by model using SQL aggregation
        by_model_query = """
            SELECT
                COALESCE(model, 'unknown') as model,
                COUNT(*) as requests,
                COALESCE(SUM(cost_usd), 0.0) as cost_usd,
                COALESCE(SUM(energy_wh), 0.0) as energy_wh
            FROM events
            WHERE timestamp BETWEEN ? AND ?
        """
        by_model_params = [start.isoformat(), end.isoformat()]

        if model:
            by_model_query += " AND model = ?"
            by_model_params.append(model)

        if tags:
            for key, value in tags.items():
                by_model_query += " AND json_extract(tags_json, ?) = ?"
                by_model_params.append(f"$.{key}")
                by_model_params.append(value)

        by_model_query += " GROUP BY model"

        cursor.execute(by_model_query, by_model_params)
        for row in cursor:
            summary.by_model[row['model']] = {
                'requests': row['requests'],
                'cost_usd': row['cost_usd'],
                'energy_wh': row['energy_wh']
            }

        # Query 3: Group by tags (fetch tags_json + cost_usd and aggregate
        # in Python for tag breakdown)
        # Note: SQL can't efficiently GROUP BY arbitrary JSON keys,
        # so we fetch tags and aggregate
        tags_query = (
            "SELECT tags_json, COALESCE(cost_usd, 0.0) as cost_usd "
            "FROM events WHERE timestamp BETWEEN ? AND ?"
        )
        tags_params = [start.isoformat(), end.isoformat()]

        if model:
            tags_query += " AND model = ?"
            tags_params.append(model)

        if tags:
            for key, value in tags.items():
                tags_query += " AND json_extract(tags_json, ?) = ?"
                tags_params.append(f"$.{key}")
                tags_params.append(value)

        cursor.execute(tags_query, tags_params)

        # Build a map of tag combinations to count/cost for aggregation
        tag_stats: dict[str, dict[str, dict[str, int | float]]] = {}
        for row in cursor:
            row_tags = json.loads(row['tags_json'])
            row_cost = row['cost_usd']
            for k, v in row_tags.items():
                if k not in tag_stats:
                    tag_stats[k] = {}
                if v not in tag_stats[k]:
                    tag_stats[k][v] = {'requests': 0, 'cost_usd': 0.0}
                tag_stats[k][v]['requests'] += 1
                tag_stats[k][v]['cost_usd'] += row_cost

        summary.by_tag = tag_stats

        return summary


def get_db_path() -> Path:
    return DB_PATH


def get_top_consumers(
    metric: str = "cost",
    tag_key: str = "team",
    days: int = 7,
    limit: int = 5
) -> list[dict[str, Any]]:
    """Get top consumers by a specific tag."""
    from datetime import timedelta
    end = datetime.now()
    start = end - timedelta(days=days)
    summary = query_usage(start, end)

    if tag_key not in summary.by_tag:
        return []

    # Flatten
    items = []
    for tag_val, data in summary.by_tag[tag_key].items():
        items.append({
            "tag_value": tag_val,
            "requests": data['requests'],
            "cost": data['cost_usd'],
            "energy": 0.0  # Missing in Alpha for tag grouping
        })

    items.sort(key=lambda x: x.get(metric, 0), reverse=True)
    return items[:limit]
