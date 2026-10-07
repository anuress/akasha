from pathlib import Path

import pytest

from akasha.config import load_config
from akasha.db import connect
from akasha.knowledge import archive, purge, remove, restore, write
from akasha.search import search


@pytest.fixture
def env(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")
    cfg.knowledge_dir = tmp_path / "knowledge"
    cfg.knowledge_dir.mkdir()
    cfg.index_roots = []
    return connect(tmp_path / "s.db"), cfg


def test_archive_hides_from_default_search_but_file_remains(env):
    conn, cfg = env
    doc_id = write(conn, cfg, "T", "## A\narchivable-token\n", repo="r")
    path = Path(conn.execute("SELECT path FROM documents WHERE id=?", (doc_id,)).fetchone()["path"])
    archive(conn, cfg, doc_id)
    assert not search(conn, "archivable-token", all_repos=True)
    assert search(conn, "archivable-token", all_repos=True, include_archived=True)
    assert path.exists()


def test_rm_moves_to_trash_and_leaves_content_intact(env):
    conn, cfg = env
    doc_id = write(conn, cfg, "T", "## A\ntrashable\n", repo="r")
    original = Path(conn.execute("SELECT path FROM documents WHERE id=?", (doc_id,)).fetchone()["path"])
    trashed = remove(conn, cfg, doc_id)
    assert not original.exists()
    assert trashed.exists()
    assert "trashable" in trashed.read_text()
    assert not search(conn, "trashable", all_repos=True, include_archived=True)


def test_restore_brings_it_back(env):
    conn, cfg = env
    doc_id = write(conn, cfg, "T", "## A\nrestorable\n", repo="r")
    remove(conn, cfg, doc_id)
    path = restore(conn, cfg, doc_id)
    assert path.exists()
    assert search(conn, "restorable", all_repos=True)


def test_purge_empties_trash_permanently(env):
    conn, cfg = env
    doc_id = write(conn, cfg, "T", "## A\npurgeable\n", repo="r")
    trashed = remove(conn, cfg, doc_id)
    assert purge(conn, cfg) == 1
    assert not trashed.exists()
    assert conn.execute("SELECT COUNT(*) c FROM documents WHERE id=?", (doc_id,)).fetchone()["c"] == 0


def test_rm_clears_a_row_whose_file_is_already_gone(env):
    """fsck reports `missing_file` as an error, and the only command that could clear one
    crashed on it: remove() moved the file to trash and there was no file to move. So the
    fault stayed reported with no way to act on it."""
    conn, cfg = env
    doc_id = write(conn, cfg, "T", "## A\nvanished\n", repo="r")
    path = Path(conn.execute("SELECT path FROM documents WHERE id=?", (doc_id,)
                             ).fetchone()["path"])
    path.unlink()

    remove(conn, cfg, doc_id)
    assert not search(conn, "vanished", all_repos=True, include_archived=True)
    row = conn.execute("SELECT deleted_at FROM documents WHERE id=?", (doc_id,)).fetchone()
    assert row["deleted_at"], "the row is a tombstone now, not a document"
