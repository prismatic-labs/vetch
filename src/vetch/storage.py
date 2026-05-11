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
import queue
import sqlite3
import threading
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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

_WRITE_BATCH_SIZE = 100
_WRITE_BATCH_TIMEOUT_S = 0.25
_WRITE_QUEUE_MAXSIZE = 10_000
_AUTO_COMPACT_EVERY_WRITES = 100
_DEFAULT_RAW_RETENTION_DAYS = 90


@dataclass(frozen=True)
class _StoredEvent:
    event_row: tuple[Any, ...]
    aggregate_rows: tuple[tuple[str, str, str, int, int, int, float, float, float], ...]


_STOP_WRITER = object()
_write_queue: queue.Queue[_StoredEvent | object] | None = None
_writer_thread: threading.Thread | None = None
_writer_lock = threading.Lock()
_atexit_registered = False
_writes_since_maintenance = 0


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
            _configure_connection(_connection_pool)
            _connection_pool.row_factory = sqlite3.Row
            _register_atexit()
        return _connection_pool


def _configure_connection(conn: sqlite3.Connection) -> None:
    """Apply SQLite pragmas used by both writer and reader connections."""
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")


def _register_atexit() -> None:
    global _atexit_registered
    if not _atexit_registered:
        atexit.register(shutdown_storage)
        _atexit_registered = True


def _close_connection() -> None:
    """Close the pooled connection on shutdown."""
    global _connection_pool
    if _connection_pool is not None:
        try:
            _connection_pool.close()
        except sqlite3.Error as exc:
            logger.debug("Failed to close Vetch storage connection: %s", exc)
        _connection_pool = None


def _close_connection_locked() -> None:
    """Close the pooled connection while the caller holds ``_connection_lock``."""
    global _connection_pool
    if _connection_pool is not None:
        try:
            _connection_pool.close()
        except sqlite3.Error as exc:
            logger.debug("Failed to close Vetch storage connection: %s", exc)
        _connection_pool = None


def configure_storage(enabled: bool = True, path: Path | None = None) -> None:
    """Enable or disable local storage."""
    global _STORAGE_ENABLED, DB_PATH, _EXPERIMENTAL_WARNING_ISSUED

    # Issue warning only once when storage is first enabled
    if enabled and not _EXPERIMENTAL_WARNING_ISSUED:
        _EXPERIMENTAL_WARNING_ISSUED = True
        warnings.warn(
            "vetch.storage is experimental. API may change in future versions.",
            FutureWarning,
            stacklevel=2,
        )

    if path and path != DB_PATH:
        shutdown_storage()
        DB_PATH = path
    elif not enabled:
        shutdown_storage()

    _STORAGE_ENABLED = enabled


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
    _configure_connection(conn)

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

        # Durable daily aggregates. These keep consulting/audit totals available
        # after raw_json rows have been compacted.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_usage (
                day TEXT NOT NULL,
                dimension TEXT NOT NULL,
                value TEXT NOT NULL,
                requests INTEGER NOT NULL DEFAULT 0,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                energy_wh REAL NOT NULL DEFAULT 0.0,
                carbon_g REAL NOT NULL DEFAULT 0.0,
                cost_usd REAL NOT NULL DEFAULT 0.0,
                PRIMARY KEY (day, dimension, value)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_usage_day ON daily_usage(day)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_daily_usage_dimension "
            "ON daily_usage(dimension, value, day)"
        )

        conn.commit()
    finally:
        conn.close()


def store_event(event: InferenceEvent) -> None:
    """Store an event in the local database without blocking the caller on disk I/O."""
    if not _STORAGE_ENABLED:
        return

    try:
        stored = _prepare_stored_event(event)
        writer_queue = _ensure_writer_running()
        writer_queue.put_nowait(stored)
    except queue.Full:
        logger.warning("Vetch storage queue is full; dropping local usage event")
    except Exception as exc:
        logger.debug("Failed to enqueue event for local storage: %s", exc)


def flush_storage(timeout: float | None = 5.0) -> None:
    """Wait until pending storage writes are durable.

    Reads call this automatically so reports remain deterministic even though
    writes are asynchronous.
    """
    writer_queue = _write_queue
    writer = _writer_thread
    if writer_queue is None:
        return
    if writer is not None and not writer.is_alive() and not writer_queue.empty():
        logger.debug("Vetch storage writer is not running; pending events may be delayed")
        return

    if timeout is None:
        writer_queue.join()
        return

    deadline = time.monotonic() + timeout
    while writer_queue.unfinished_tasks and time.monotonic() < deadline:
        time.sleep(0.01)
    if writer_queue.unfinished_tasks:
        logger.warning("Timed out waiting for Vetch storage queue to flush")


def shutdown_storage() -> None:
    """Flush queued events and stop the background writer."""
    global _write_queue, _writer_thread

    writer_queue = _write_queue
    writer = _writer_thread
    if writer_queue is not None and writer is not None and writer.is_alive():
        flush_storage(timeout=None)
        writer_queue.put(_STOP_WRITER)
        writer.join(timeout=2.0)

    _write_queue = None
    _writer_thread = None
    with _connection_lock:
        _close_connection_locked()


def _ensure_writer_running() -> queue.Queue[_StoredEvent | object]:
    """Start the background storage writer if needed."""
    global _write_queue, _writer_thread

    with _writer_lock:
        if _write_queue is None:
            _write_queue = queue.Queue(maxsize=_WRITE_QUEUE_MAXSIZE)
        if _writer_thread is None or not _writer_thread.is_alive():
            if not DB_PATH.exists():
                _init_db()
            _register_atexit()
            _writer_thread = threading.Thread(
                target=_writer_loop,
                name="vetch-storage-writer",
                daemon=True,
            )
            _writer_thread.start()
        return _write_queue


def _writer_loop() -> None:
    """Batch queued event writes into SQLite commits."""
    writer_queue = _write_queue
    if writer_queue is None:
        return

    while True:
        item = writer_queue.get()
        if item is _STOP_WRITER:
            writer_queue.task_done()
            return

        batch: list[_StoredEvent] = [cast(_StoredEvent, item)]
        deadline = time.monotonic() + _WRITE_BATCH_TIMEOUT_S
        while len(batch) < _WRITE_BATCH_SIZE:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                next_item = writer_queue.get(timeout=remaining)
            except queue.Empty:
                break
            if next_item is _STOP_WRITER:
                writer_queue.task_done()
                # Put the sentinel back so the outer loop can shut down cleanly
                # after the current batch is durable.
                writer_queue.put(_STOP_WRITER)
                break
            batch.append(cast(_StoredEvent, next_item))

        try:
            conn = _get_connection()
            with _connection_lock:
                cursor = conn.cursor()
                for stored in batch:
                    _insert_stored_event(cursor, stored)
                _maybe_auto_compact_locked(cursor, len(batch))
                conn.commit()
        except Exception as exc:
            logger.debug("Failed to write Vetch storage batch: %s", exc)
        finally:
            for _ in batch:
                writer_queue.task_done()


def _prepare_stored_event(event: InferenceEvent) -> _StoredEvent:
    usage: dict[str, Any] = cast(dict[str, Any], event.get("usage", {}) or {})
    text_usage = usage.get("text", {}) or {}
    input_tokens = int(text_usage.get("input_tokens") or 0)
    output_tokens = int(text_usage.get("output_tokens") or 0)
    energy_wh = _float_or_zero(event.get("estimated_energy_wh"))
    carbon_g = _float_or_zero(event.get("estimated_carbon_g"))
    cost_usd = _float_or_zero(event.get("estimated_cost_usd"))
    tags = event.get("tags") if isinstance(event.get("tags"), dict) else {}
    tags_dict = cast(dict[str, Any], tags or {})
    timestamp = str(event.get("timestamp") or datetime.now(timezone.utc).isoformat())
    day = _day_from_timestamp(timestamp)
    model = str(event.get("model") or "unknown")
    provider = str(event.get("provider") or "unknown")

    aggregate_rows: list[tuple[str, str, str, int, int, int, float, float, float]] = [
        (day, "all", "all", 1, input_tokens, output_tokens, energy_wh, carbon_g, cost_usd),
        (day, "model", model, 1, input_tokens, output_tokens, energy_wh, carbon_g, cost_usd),
        (day, "provider", provider, 1, input_tokens, output_tokens, energy_wh, carbon_g, cost_usd),
    ]
    for key, value in tags_dict.items():
        if value is not None and value != "":
            aggregate_rows.append(
                (
                    day,
                    str(key),
                    str(value),
                    1,
                    input_tokens,
                    output_tokens,
                    energy_wh,
                    carbon_g,
                    cost_usd,
                )
            )

    return _StoredEvent(
        event_row=(
            event.get("event_id"),
            timestamp,
            event.get("model"),
            event.get("provider"),
            input_tokens,
            output_tokens,
            event.get("estimated_energy_wh"),
            event.get("estimated_carbon_g"),
            event.get("estimated_cost_usd"),
            json.dumps(tags_dict),
            json.dumps(event),
        ),
        aggregate_rows=tuple(aggregate_rows),
    )


def _insert_stored_event(cursor: sqlite3.Cursor, stored: _StoredEvent) -> None:
    cursor.execute(
        """
        INSERT OR IGNORE INTO events (
            event_id, timestamp, model, provider,
            input_tokens, output_tokens,
            energy_wh, carbon_g, cost_usd,
            tags_json, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        stored.event_row,
    )
    if cursor.rowcount == 0:
        return
    cursor.executemany(
        """
        INSERT INTO daily_usage (
            day, dimension, value, requests, input_tokens, output_tokens,
            energy_wh, carbon_g, cost_usd
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(day, dimension, value) DO UPDATE SET
            requests = requests + excluded.requests,
            input_tokens = input_tokens + excluded.input_tokens,
            output_tokens = output_tokens + excluded.output_tokens,
            energy_wh = energy_wh + excluded.energy_wh,
            carbon_g = carbon_g + excluded.carbon_g,
            cost_usd = cost_usd + excluded.cost_usd
        """,
        stored.aggregate_rows,
    )


def _maybe_auto_compact_locked(cursor: sqlite3.Cursor, writes: int) -> None:
    global _writes_since_maintenance
    _writes_since_maintenance += writes
    if _writes_since_maintenance < _AUTO_COMPACT_EVERY_WRITES:
        return
    _writes_since_maintenance = 0
    cutoff = (datetime.now(timezone.utc) - timedelta(days=_DEFAULT_RAW_RETENTION_DAYS)).date()
    cursor.execute("DELETE FROM events WHERE substr(timestamp, 1, 10) < ?", (cutoff.isoformat(),))


def _float_or_zero(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _day_from_timestamp(timestamp: str) -> str:
    if len(timestamp) >= 10:
        return timestamp[:10]
    fallback = datetime.now(timezone.utc).date().isoformat()
    logger.warning("Malformed event timestamp %r — bucketing into %s", timestamp, fallback)
    return fallback


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
    flush_storage()
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

        # Query 3: Group by tags (fetch tags_json + metrics and aggregate
        # in Python for tag breakdown)
        # Note: SQL can't efficiently GROUP BY arbitrary JSON keys,
        # so we fetch tags and aggregate
        tags_query = (
            "SELECT tags_json, "
            "COALESCE(cost_usd, 0.0) as cost_usd, "
            "COALESCE(energy_wh, 0.0) as energy_wh, "
            "COALESCE(carbon_g, 0.0) as carbon_g "
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
            row_energy = row['energy_wh']
            row_carbon = row['carbon_g']
            for k, v in row_tags.items():
                if k not in tag_stats:
                    tag_stats[k] = {}
                if v not in tag_stats[k]:
                    tag_stats[k][v] = {
                        'requests': 0,
                        'cost_usd': 0.0,
                        'energy_wh': 0.0,
                        'carbon_g': 0.0,
                    }
                tag_stats[k][v]['requests'] += 1
                tag_stats[k][v]['cost_usd'] += row_cost
                tag_stats[k][v]['energy_wh'] += row_energy
                tag_stats[k][v]['carbon_g'] += row_carbon

        summary.by_tag = tag_stats

        return summary


def query_events(
    start: datetime,
    end: datetime,
    model: str | None = None,
    tags: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Return stored raw events for a time window.

    This is used by deterministic audit reports. It returns metadata-only
    events from ``raw_json`` and applies model/tag filters in SQLite before
    deserializing rows.
    """
    flush_storage()
    if not DB_PATH.exists():
        return []

    conn = _get_connection()
    with _connection_lock:
        cursor = conn.cursor()
        query = "SELECT raw_json FROM events WHERE timestamp BETWEEN ? AND ?"
        params: list[Any] = [start.isoformat(), end.isoformat()]

        if model:
            query += " AND model = ?"
            params.append(model)

        if tags:
            for key, value in tags.items():
                query += " AND json_extract(tags_json, ?) = ?"
                params.append(f"$.{key}")
                params.append(value)

        query += " ORDER BY timestamp ASC"
        cursor.execute(query, params)

        events: list[dict[str, Any]] = []
        for row in cursor:
            raw_json = row["raw_json"]
            if not raw_json:
                continue
            try:
                raw = json.loads(raw_json)
            except json.JSONDecodeError:
                logger.debug("Skipping stored event with invalid raw_json")
                continue
            if isinstance(raw, dict):
                events.append(raw)
        return events


def query_daily_usage(
    start: datetime,
    end: datetime,
    dimensions: tuple[str, ...] | None = None,
) -> UsageSummary:
    """Query durable daily aggregates for a time period.

    Daily aggregates are less precise than raw events for sub-day windows, but
    they remain available after raw event compaction and are suitable for
    executive totals over longer audit windows.
    """
    flush_storage()
    summary = UsageSummary(start, end)
    if not DB_PATH.exists():
        return summary

    day_start = start.date().isoformat()
    day_end = end.date().isoformat()
    dimension_filter = dimensions or ()

    conn = _get_connection()
    with _connection_lock:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                COALESCE(SUM(requests), 0) as total_requests,
                COALESCE(SUM(input_tokens), 0) as total_input_tokens,
                COALESCE(SUM(output_tokens), 0) as total_output_tokens,
                COALESCE(SUM(energy_wh), 0.0) as total_energy_wh,
                COALESCE(SUM(carbon_g), 0.0) as total_carbon_g,
                COALESCE(SUM(cost_usd), 0.0) as total_cost_usd
            FROM daily_usage
            WHERE day BETWEEN ? AND ?
              AND dimension = 'all'
              AND value = 'all'
            """,
            (day_start, day_end),
        )
        totals_row = cursor.fetchone()
        if totals_row:
            summary.total_requests = int(totals_row["total_requests"])
            summary.total_input_tokens = int(totals_row["total_input_tokens"])
            summary.total_output_tokens = int(totals_row["total_output_tokens"])
            summary.total_energy_wh = float(totals_row["total_energy_wh"])
            summary.total_carbon_g = float(totals_row["total_carbon_g"])
            summary.total_cost_usd = float(totals_row["total_cost_usd"])

        cursor.execute(
            """
            SELECT
                value as model,
                COALESCE(SUM(requests), 0) as requests,
                COALESCE(SUM(input_tokens), 0) as input_tokens,
                COALESCE(SUM(output_tokens), 0) as output_tokens,
                COALESCE(SUM(cost_usd), 0.0) as cost_usd,
                COALESCE(SUM(energy_wh), 0.0) as energy_wh,
                COALESCE(SUM(carbon_g), 0.0) as carbon_g
            FROM daily_usage
            WHERE day BETWEEN ? AND ?
              AND dimension = 'model'
            GROUP BY value
            """,
            (day_start, day_end),
        )
        for row in cursor:
            summary.by_model[row["model"]] = {
                "requests": int(row["requests"]),
                "input_tokens": int(row["input_tokens"]),
                "output_tokens": int(row["output_tokens"]),
                "cost_usd": float(row["cost_usd"]),
                "energy_wh": float(row["energy_wh"]),
                "carbon_g": float(row["carbon_g"]),
            }

        tag_query = """
            SELECT
                dimension,
                value,
                COALESCE(SUM(requests), 0) as requests,
                COALESCE(SUM(input_tokens), 0) as input_tokens,
                COALESCE(SUM(output_tokens), 0) as output_tokens,
                COALESCE(SUM(cost_usd), 0.0) as cost_usd,
                COALESCE(SUM(energy_wh), 0.0) as energy_wh,
                COALESCE(SUM(carbon_g), 0.0) as carbon_g
            FROM daily_usage
            WHERE day BETWEEN ? AND ?
              AND dimension NOT IN ('all', 'model', 'provider')
        """
        params: list[Any] = [day_start, day_end]
        if dimension_filter:
            placeholders = ", ".join("?" for _ in dimension_filter)
            tag_query += f" AND dimension IN ({placeholders})"
            params.extend(dimension_filter)
        tag_query += " GROUP BY dimension, value"

        cursor.execute(tag_query, params)
        for row in cursor:
            dimension = row["dimension"]
            value = row["value"]
            summary.by_tag.setdefault(dimension, {})[value] = {
                "requests": int(row["requests"]),
                "input_tokens": int(row["input_tokens"]),
                "output_tokens": int(row["output_tokens"]),
                "cost_usd": float(row["cost_usd"]),
                "energy_wh": float(row["energy_wh"]),
                "carbon_g": float(row["carbon_g"]),
            }

    return summary


def compact_storage(raw_retention_days: int = _DEFAULT_RAW_RETENTION_DAYS) -> int:
    """Delete old raw events while preserving daily aggregates.

    Returns the number of raw rows removed. ``raw_retention_days=0`` removes
    rows older than the current UTC day.
    """
    flush_storage()
    if not DB_PATH.exists():
        return 0

    cutoff = (datetime.now(timezone.utc) - timedelta(days=raw_retention_days)).date()
    conn = _get_connection()
    with _connection_lock:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM events WHERE substr(timestamp, 1, 10) < ?",
            (cutoff.isoformat(),),
        )
        deleted = cursor.rowcount
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return int(deleted)


def get_db_path() -> Path:
    return DB_PATH


def get_top_consumers(
    metric: str = "cost",
    tag_key: str = "team",
    days: int = 7,
    limit: int = 5
) -> list[dict[str, Any]]:
    """Get top consumers by a specific tag."""
    end = datetime.now(timezone.utc)
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
            "energy": data.get('energy_wh', 0.0),
            "carbon": data.get('carbon_g', 0.0),
        })

    items.sort(key=lambda x: x.get(metric, 0), reverse=True)
    return items[:limit]
