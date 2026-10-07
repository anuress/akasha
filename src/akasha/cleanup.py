"""Retention. Bounds the events table, which nothing else bounds."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

from akasha.config import Config
from akasha.db import now
from akasha.events import emit
from akasha.knowledge import EVENT_RETENTION_DAYS


def purge_events(conn: sqlite3.Connection, days: int = EVENT_RETENTION_DAYS,
                 reference: datetime | None = None) -> int:
    """Events older than the window. `reference` lets tests fix the clock."""
    cutoff = ((reference or datetime.now(timezone.utc)) - timedelta(days=days)).isoformat()
    cursor = conn.execute("DELETE FROM events WHERE ts < ?", (cutoff,))
    conn.commit()
    return cursor.rowcount


def run(conn: sqlite3.Connection, reference: datetime | None = None) -> dict[str, int]:
    """Purge and report what was removed. Records when it ran."""
    counts = {"events": purge_events(conn, reference=reference)}
    emit(conn, "cleanup.ran", **counts)
    _record_run(conn)
    return counts


def last_run(conn: sqlite3.Connection) -> str | None:
    """When retention last ran; None means never."""
    row = conn.execute("SELECT value FROM meta WHERE key='cleanup.last_run'").fetchone()
    if row is None:
        return None
    try:
        return json.loads(row["value"])
    except (json.JSONDecodeError, TypeError):
        return None


def _due(conn: sqlite3.Connection, cfg: Config) -> bool:
    if cfg.housekeeping_interval_min <= 0:
        return True
    last = last_run(conn)
    if last is None:
        return True
    try:
        last_dt = datetime.fromisoformat(last)
    except (TypeError, ValueError):
        return True
    return (datetime.now(timezone.utc) - last_dt
            >= timedelta(minutes=cfg.housekeeping_interval_min))


def _record_run(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('cleanup.last_run', ?)"
        " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (json.dumps(now()),))
    conn.commit()


def run_if_due(conn: sqlite3.Connection, cfg: Config) -> dict[str, int] | None:
    """Purge when the interval permits; None when not due.

    Never raises: it runs from hooks. A failed purge is recorded, so a database that is
    bounded stays distinguishable from one that never was, and the failure is named in
    an event. Recording can fail too, and then there is nothing left to do but return.
    """
    try:
        if not _due(conn, cfg):
            return None
        return run(conn)
    except Exception as exc:                                        # noqa: BLE001
        try:
            conn.rollback()
            emit(conn, "cleanup.failed", error=str(exc))
            _record_run(conn)
        except Exception:                                           # noqa: BLE001
            pass
        return {"events": 0}
