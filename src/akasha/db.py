"""SQLite access. The database is a derived index — never the only home of data."""
from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1


class SchemaMismatch(RuntimeError):
    """The database was built by a different schema version."""

# Dense vectors live in a vec0 virtual table that vectors.py creates on demand, because it
# needs the optional sqlite-vec extension.
SCHEMA = """
CREATE TABLE IF NOT EXISTS features (
  id TEXT PRIMARY KEY, slug TEXT NOT NULL UNIQUE, title TEXT, repo TEXT,
  status TEXT NOT NULL DEFAULT 'active',
  aliases TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL, deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY, source TEXT NOT NULL, path TEXT NOT NULL UNIQUE,
  title TEXT, repo TEXT, feature_id TEXT, kind TEXT,
  status TEXT NOT NULL DEFAULT 'active', supersedes TEXT,
  -- mtime and size together say a file is unchanged, so the indexer can skip it without
  -- reading it.
  mtime REAL, size INTEGER, indexed_at TEXT,
  -- End of the validity interval: the date an archive or supersession took the document
  -- out of search. NULL means still open. Mirrored in frontmatter, since the database
  -- is rebuilt from files.
  invalid_at TEXT,
  last_accessed TEXT, access_count INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL, deleted_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_documents_feature ON documents(feature_id);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);

CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY, document_id TEXT NOT NULL, title TEXT, heading TEXT,
  body TEXT NOT NULL, ord INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  title, heading, body, id UNINDEXED, content='chunks', content_rowid='rowid',
  tokenize='porter unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, title, heading, body, id)
  VALUES (new.rowid, new.title, new.heading, new.body, new.id);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, title, heading, body, id)
  VALUES ('delete', old.rowid, old.title, old.heading, old.body, old.id);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, title, heading, body, id)
  VALUES ('delete', old.rowid, old.title, old.heading, old.body, old.id);
  INSERT INTO chunks_fts(rowid, title, heading, body, id)
  VALUES (new.rowid, new.title, new.heading, new.body, new.id);
END;

CREATE TABLE IF NOT EXISTS links (
  id TEXT PRIMARY KEY, from_document_id TEXT NOT NULL,
  to_document_id TEXT, to_ref TEXT, kind TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_links_from ON links(from_document_id);
CREATE INDEX IF NOT EXISTS idx_links_to ON links(to_document_id);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY, ts TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def connect(db_path: Path, busy_timeout_ms: int = 5000) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    # Several agent processes write concurrently; without this, a contended write
    # raises "database is locked" instead of waiting its turn.
    conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    if row and row["value"] != str(SCHEMA_VERSION):
        conn.close()
        raise SchemaMismatch(
            f"{db_path} has schema version {row['value']}, expected {SCHEMA_VERSION}. "
            "The db is derived from markdown: delete it and run `akasha index`.")
    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('schema_version', ?)"
        " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(SCHEMA_VERSION),),
    )
    return conn
