"""Append-only audit trail of knowledge operations."""
from __future__ import annotations

import json
import sqlite3

from akasha.db import new_id, now

# Events carry identifiers and outcomes, never full bodies, so a reader can hold a page
# of them. Revisit the bound if a field needs more than an identifier-sized value.
MAX_FIELD = 240


def _trim(value):
    if isinstance(value, str) and len(value) > MAX_FIELD:
        return value[:MAX_FIELD] + "…"
    return value


def emit(conn: sqlite3.Connection, kind: str, **payload) -> str:
    event_id = new_id("e")
    trimmed = {k: _trim(v) for k, v in payload.items()}
    conn.execute(
        "INSERT INTO events (id, ts, kind, payload) VALUES (?,?,?,?)",
        (event_id, now(), kind, json.dumps(trimmed, default=str)),
    )
    conn.commit()
    return event_id


def recent(conn: sqlite3.Connection, limit: int = 50, kind: str | None = None) -> list[dict]:
    """Recent events, newest first."""
    sql = "SELECT * FROM events"
    params: list = []
    if kind:
        sql += " WHERE kind = ?"
        params.append(kind)
    rows = conn.execute(sql + " ORDER BY ts DESC, rowid DESC LIMIT ?", params + [limit])
    return [dict(r) for r in rows]
