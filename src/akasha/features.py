"""Features are a plain grouping tag for documents. A table, not a string, so names
cannot drift."""
from __future__ import annotations

import json
import sqlite3

from akasha.db import new_id, now


def _norm(slug: str) -> str:
    return slug.strip().lower()


def resolve_feature(
    conn: sqlite3.Connection, slug: str, repo: str | None = None, create: bool = True
) -> str | None:
    """Return the feature id for a slug, following aliases. Optionally create it."""
    key = _norm(slug)
    row = conn.execute("SELECT id FROM features WHERE slug = ?", (key,)).fetchone()
    if row:
        return row["id"]
    for candidate in conn.execute("SELECT id, aliases FROM features"):
        if key in [_norm(a) for a in json.loads(candidate["aliases"])]:
            return candidate["id"]
    if not create:
        return None
    fid = new_id("f")
    conn.execute(
        "INSERT INTO features (id, slug, title, repo, status, aliases, created_at, updated_at)"
        " VALUES (?, ?, ?, ?, 'active', '[]', ?, ?)",
        (fid, key, slug.strip(), repo, now(), now()),
    )
    return fid

def add_alias(conn: sqlite3.Connection, slug: str, alias: str) -> None:
    """Point `alias` at the feature identified by `slug`."""
    fid = resolve_feature(conn, slug, create=False)
    if fid is None:
        raise ValueError(f"unknown feature: {slug}")
    row = conn.execute("SELECT aliases FROM features WHERE id = ?", (fid,)).fetchone()
    aliases = json.loads(row["aliases"])
    if _norm(alias) not in [_norm(a) for a in aliases]:
        aliases.append(_norm(alias))
    conn.execute(
        "UPDATE features SET aliases = ?, updated_at = ? WHERE id = ?",
        (json.dumps(aliases), now(), fid),
    )

def list_features(
    conn: sqlite3.Connection, repo: str | None = None, status: str | None = None
) -> list[dict]:
    sql = "SELECT * FROM features WHERE deleted_at IS NULL"
    params: list = []
    if repo:
        sql += " AND repo = ?"
        params.append(repo)
    if status:
        sql += " AND status = ?"
        params.append(status)
    sql += " ORDER BY slug"
    return [dict(r) for r in conn.execute(sql, params)]

def show(conn: sqlite3.Connection, slug: str) -> dict:
    """A feature and how many live documents carry it. Names the canonical slug, so an
    alias answers with the feature it points at."""
    fid = resolve_feature(conn, slug, create=False)
    if fid is None:
        raise ValueError(f"unknown feature: {slug}")
    row = conn.execute("SELECT slug FROM features WHERE id = ?", (fid,)).fetchone()
    count = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE feature_id = ? AND deleted_at IS NULL",
        (fid,)).fetchone()[0]
    return {"slug": row["slug"], "documents": count}
