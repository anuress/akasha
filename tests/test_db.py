import sqlite3

import pytest

from akasha.db import SCHEMA_VERSION, connect, new_id, now


def test_connect_creates_tables(tmp_path):
    conn = connect(tmp_path / "s.db")
    names = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
    )}
    for expected in ("features", "documents", "chunks", "chunks_fts", "links", "events",
                     "meta"):
        assert expected in names


def test_connect_is_idempotent(tmp_path):
    p = tmp_path / "s.db"
    connect(p).close()
    conn = connect(p)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_connect_records_the_schema_version(tmp_path):
    conn = connect(tmp_path / "s.db")
    row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    assert row["value"] == str(SCHEMA_VERSION)


def test_new_id_is_prefixed_and_unique():
    a, b = new_id("k"), new_id("k")
    assert a.startswith("k_") and a != b


def test_now_is_iso_utc():
    assert now().endswith("+00:00")


def test_fts_is_searchable(tmp_path):
    conn = connect(tmp_path / "s.db")
    conn.execute(
        "INSERT INTO documents (id, source, path, title, status, created_at, updated_at)"
        " VALUES ('d1','native','/x.md','X','active',?,?)", (now(), now()))
    conn.execute(
        "INSERT INTO chunks (id, document_id, heading, body, ord)"
        " VALUES ('c1','d1','Caching','the cache layer is warm',0)")
    conn.commit()
    rows = list(conn.execute("SELECT id FROM chunks_fts WHERE chunks_fts MATCH 'caching'"))
    assert len(rows) == 1


def test_fts_matches_on_title(tmp_path):
    conn = connect(tmp_path / "s.db")
    conn.execute(
        "INSERT INTO documents (id, source, path, title, status, created_at, updated_at)"
        " VALUES ('d1','native','/x.md','Limit sweep','active',?,?)", (now(), now()))
    conn.execute(
        "INSERT INTO chunks (id, document_id, title, heading, body, ord)"
        " VALUES ('c1','d1','Limit sweep','## Result','limit 30 wins',0)")
    conn.commit()
    rows = list(conn.execute(
        'SELECT id FROM chunks_fts WHERE chunks_fts MATCH \'"limit" "sweep"\''))
    assert len(rows) == 1


def test_fts_follows_chunk_updates_and_deletes(tmp_path):
    """The triggers keep the external-content FTS table in step; a stale row would
    return hits for text that is gone."""
    conn = connect(tmp_path / "s.db")
    conn.execute("INSERT INTO chunks (id, document_id, body, ord)"
                 " VALUES ('c1','d1','alpha text',0)")
    conn.execute("UPDATE chunks SET body='beta text' WHERE id='c1'")
    assert not list(conn.execute("SELECT id FROM chunks_fts WHERE chunks_fts MATCH 'alpha'"))
    assert len(list(conn.execute("SELECT id FROM chunks_fts WHERE chunks_fts MATCH 'beta'"))) == 1
    conn.execute("DELETE FROM chunks WHERE id='c1'")
    assert not list(conn.execute("SELECT id FROM chunks_fts WHERE chunks_fts MATCH 'beta'"))


def test_document_path_is_unique(tmp_path):
    conn = connect(tmp_path / "s.db")
    insert = ("INSERT INTO documents (id, source, path, created_at, updated_at)"
              " VALUES (?, 'native', '/x.md', 'x', 'x')")
    conn.execute(insert, ("d1",))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(insert, ("d2",))
