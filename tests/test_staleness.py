from datetime import datetime, timedelta, timezone

import pytest

from akasha.config import load_config
from akasha.db import connect
from akasha.knowledge import stale, write
from akasha.search import search


@pytest.fixture
def env(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")
    cfg.knowledge_dir = tmp_path / "knowledge"
    cfg.knowledge_dir.mkdir()
    cfg.index_roots = []
    cfg.db_path = tmp_path / "s.db"
    return connect(tmp_path / "s.db"), cfg


def test_search_marks_hits_as_accessed(env):
    conn, cfg = env
    doc_id = write(conn, cfg, "Catalog cache miss", "## Cause\ncaching was off\n", repo="r")
    assert conn.execute(
        "SELECT access_count FROM documents WHERE id=?", (doc_id,)).fetchone()[0] == 0

    search(conn, "caching was off", all_repos=True)
    row = conn.execute(
        "SELECT access_count, last_accessed FROM documents WHERE id=?", (doc_id,)).fetchone()
    assert row["access_count"] == 1
    assert row["last_accessed"] is not None


def test_repeated_hits_increment_the_counter(env):
    conn, cfg = env
    doc_id = write(conn, cfg, "T", "## A\nrepeated token\n", repo="r")
    for _ in range(3):
        search(conn, "repeated token", all_repos=True)
    assert conn.execute(
        "SELECT access_count FROM documents WHERE id=?", (doc_id,)).fetchone()[0] == 3


def test_stale_lists_old_never_accessed_documents(env):
    conn, cfg = env
    old_id = write(conn, cfg, "Ancient", "## A\nold content\n", repo="r")
    write(conn, cfg, "Recent", "## A\nnew content\n", repo="r")
    past = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
    conn.execute("UPDATE documents SET created_at=?, updated_at=? WHERE id=?",
                 (past, past, old_id))
    conn.commit()

    listed = stale(conn, older_than_days=180)
    assert [d["id"] for d in listed] == [old_id]


def test_an_accessed_document_is_not_stale(env):
    conn, cfg = env
    doc_id = write(conn, cfg, "Ancient", "## A\nfindable token\n", repo="r")
    past = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
    conn.execute("UPDATE documents SET created_at=?, updated_at=? WHERE id=?",
                 (past, past, doc_id))
    conn.commit()
    search(conn, "findable token", all_repos=True)

    assert stale(conn, older_than_days=180, never_accessed_only=True) == []


def test_stale_does_not_delete_anything(env):
    conn, cfg = env
    doc_id = write(conn, cfg, "Ancient", "## A\nx\n", repo="r")
    write(conn, cfg, "Recent", "## A\ny\n", repo="r")
    past = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
    conn.execute("UPDATE documents SET created_at=? WHERE id=?", (past, doc_id))
    conn.commit()
    stale(conn, older_than_days=180)
    assert conn.execute(
        "SELECT COUNT(*) c FROM documents WHERE deleted_at IS NULL").fetchone()["c"] == 2
