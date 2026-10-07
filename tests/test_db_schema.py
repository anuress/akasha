"""The schema is knowledge storage only: documents, chunks, links, features, events and
meta. Nothing else is created, and nothing a kept feature does not read is stored."""
import pytest

from akasha.db import connect


def _tables(conn):
    return {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'chunks_fts%'")}


def _columns(conn, table):
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}


def test_the_only_tables_are_the_knowledge_ones(tmp_path):
    conn = connect(tmp_path / "s.db")
    assert _tables(conn) == {"features", "documents", "chunks", "links", "events", "meta"}


def test_documents_carry_only_the_columns_a_kept_feature_reads(tmp_path):
    """invalid_at backs as-of search, last_accessed/access_count back the stale readout,
    supersedes backs search's superseded-demotion. Nothing else belongs."""
    conn = connect(tmp_path / "s.db")
    assert _columns(conn, "documents") == {
        "id", "source", "path", "title", "repo", "feature_id", "kind", "status",
        "supersedes", "mtime", "size", "indexed_at", "invalid_at", "last_accessed",
        "access_count", "created_at", "updated_at", "deleted_at"}


def test_features_are_a_plain_grouping_tag(tmp_path):
    conn = connect(tmp_path / "s.db")
    assert _columns(conn, "features") == {
        "id", "slug", "title", "repo", "status", "aliases",
        "created_at", "updated_at", "deleted_at"}


def test_events_carry_a_kind_and_a_payload_only(tmp_path):
    conn = connect(tmp_path / "s.db")
    assert _columns(conn, "events") == {"id", "ts", "kind", "payload"}


def test_a_new_document_defaults_to_active_with_no_accesses(tmp_path):
    conn = connect(tmp_path / "s.db")
    conn.execute("INSERT INTO documents (id, source, path, created_at, updated_at)"
                 " VALUES ('d', 'native', 'p.md', 'x', 'x')")
    row = conn.execute("SELECT status, access_count, invalid_at FROM documents").fetchone()
    assert (row["status"], row["access_count"], row["invalid_at"]) == ("active", 0, None)

def test_a_db_from_another_schema_version_is_refused_with_the_remedy(tmp_path):
    """The db is derived from markdown, so the remedy is delete and re-index."""
    path = tmp_path / "s.db"
    connect(path).execute("UPDATE meta SET value='0' WHERE key='schema_version'")
    with pytest.raises(RuntimeError, match="akasha index"):
        connect(path)
