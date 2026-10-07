"""Appending to a document that lives in an indexed root.

Every indexed root is read-only by default: appending uninvited to another tool's files
is a different risk from appending to the directory akasha owns. A root opts in with
`writable = true`.

Only append opts in. update() re-renders the whole file through render(meta, body), which
would stamp frontmatter onto a plain document that never had any; append is a plain
open("a") that cannot clobber and cannot rewrite what is already there.
"""
from __future__ import annotations

import pytest

from akasha.config import IndexRoot, load_config
from akasha.db import connect
from akasha.index import index_all
from akasha.knowledge import ReadOnlySource, append, update


def _env(tmp_path, writable):
    cfg = load_config(tmp_path / "absent.toml")
    cfg.knowledge_dir = tmp_path / "knowledge"
    cfg.knowledge_dir.mkdir()
    root = tmp_path / "ws"
    (root / "sample-repo" / "fines").mkdir(parents=True)
    cfg.index_roots = [IndexRoot(path=str(root), source="shared-notes", writable=writable)]
    conn = connect(tmp_path / "s.db")
    doc = root / "sample-repo" / "fines" / "status.md"
    doc.write_text("## Now\nfirst checkpoint\n")
    index_all(conn, cfg)
    doc_id = conn.execute("SELECT id FROM documents").fetchone()["id"]
    return conn, cfg, doc, doc_id


def test_append_is_refused_on_a_read_only_root(tmp_path):
    conn, cfg, doc, doc_id = _env(tmp_path, writable=False)
    with pytest.raises(ReadOnlySource):
        append(conn, cfg, doc_id, "second checkpoint")
    assert "second checkpoint" not in doc.read_text()


def test_append_is_allowed_on_a_writable_root(tmp_path):
    conn, cfg, doc, doc_id = _env(tmp_path, writable=True)
    append(conn, cfg, doc_id, "second checkpoint")
    text = doc.read_text()
    assert "first checkpoint" in text
    assert "second checkpoint" in text


def test_appending_leaves_the_document_in_its_own_source(tmp_path):
    """Reindexing an external document as native would relabel the row and drop the repo
    and feature the walk derives from the path."""
    conn, cfg, doc, doc_id = _env(tmp_path, writable=True)
    before = conn.execute(
        "SELECT source, repo, feature_id FROM documents WHERE id=?", (doc_id,)).fetchone()
    append(conn, cfg, doc_id, "second checkpoint")
    after = conn.execute(
        "SELECT source, repo, feature_id FROM documents WHERE id=?", (doc_id,)).fetchone()
    assert after["source"] == before["source"] == "shared-notes"
    assert after["repo"] == before["repo"] == "sample-repo"
    assert after["feature_id"] == before["feature_id"]


def test_appended_text_is_searchable_without_a_reindex(tmp_path):
    conn, cfg, doc, doc_id = _env(tmp_path, writable=True)
    append(conn, cfg, doc_id, "the fine was waived")
    bodies = [r["body"] for r in conn.execute("SELECT body FROM chunks")]
    assert any("the fine was waived" in b for b in bodies)


def test_update_stays_native_only_even_on_a_writable_root(tmp_path):
    """A body rewrite would stamp frontmatter onto a file that never had any."""
    conn, cfg, doc, doc_id = _env(tmp_path, writable=True)
    with pytest.raises(ReadOnlySource):
        update(conn, cfg, doc_id, body="## Now\nrewritten\n")
    assert "first checkpoint" in doc.read_text()


def test_a_root_is_read_only_unless_it_says_otherwise(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")
    assert all(not r.writable for r in cfg.index_roots)
