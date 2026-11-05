"""Guards against silent data loss and corrupted frontmatter in the document lifecycle."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from akasha import feed, knowledge
from akasha.config import IndexRoot, load_config
from akasha.db import connect
from akasha.index import index_all
from akasha.knowledge import append, archive, purge, remove, restore, update, write
from akasha.links import neighbour_counts, related
from akasha.search import search
from akasha.splitters import FeedSettings


@pytest.fixture
def env(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")
    cfg.knowledge_dir = tmp_path / "knowledge"
    cfg.knowledge_dir.mkdir()
    cfg.index_roots = []
    return connect(tmp_path / "s.db"), cfg


def _path(conn, doc_id) -> Path:
    return Path(conn.execute("SELECT path FROM documents WHERE id=?", (doc_id,)).fetchone()["path"])


def test_restore_never_overwrites_a_document_that_took_the_filename(env):
    """Trashing frees the filename, so a same-title write can claim it; restore must not clobber it."""
    conn, cfg = env
    a = write(conn, cfg, "Same", "## A\nbody of a\n", repo="r")
    original = _path(conn, a)
    remove(conn, cfg, a)
    b = write(conn, cfg, "Same", "## A\nbody of b\n", repo="r")
    assert _path(conn, b) == original
    restored = restore(conn, cfg, a)
    assert restored != original
    assert "body of b" in original.read_text()
    assert "body of a" in restored.read_text()


def test_restore_refuses_a_document_that_is_not_deleted(env):
    conn, cfg = env
    a = write(conn, cfg, "T", "## A\nx\n", repo="r")
    with pytest.raises(ValueError, match="not deleted"):
        restore(conn, cfg, a)


def test_restore_of_a_tombstone_with_no_file_is_a_clear_error(env):
    conn, cfg = env
    a = write(conn, cfg, "T", "## A\nx\n", repo="r")
    _path(conn, a).unlink()
    remove(conn, cfg, a)
    with pytest.raises(ValueError, match="no file"):
        restore(conn, cfg, a)


def test_restore_tolerates_a_trash_name_without_separator(env):
    conn, cfg = env
    a = write(conn, cfg, "T", "## A\nx\n", repo="r")
    trashed = remove(conn, cfg, a)
    renamed = trashed.with_name("plain.md")
    trashed.rename(renamed)
    conn.execute("UPDATE documents SET path=? WHERE id=?", (str(renamed), a))
    assert restore(conn, cfg, a).name == "plain.md"


def test_purge_ages_by_deletion_time_not_file_mtime(env):
    """rename preserves mtime, so a long-lived document trashed just now would otherwise be purged."""
    conn, cfg = env
    a = write(conn, cfg, "T", "## A\nx\n", repo="r")
    old = time.time() - 365 * 86400
    import os
    os.utime(_path(conn, a), (old, old))
    trashed = remove(conn, cfg, a)
    assert purge(conn, cfg, older_than_days=30) == 0
    assert trashed.exists()


def test_purge_skips_and_reports_a_path_outside_the_trash(env, tmp_path):
    conn, cfg = env
    a = write(conn, cfg, "T", "## A\nx\n", repo="r")
    remove(conn, cfg, a)
    victim = tmp_path / "precious.md"
    victim.write_text("keep me")
    conn.execute("UPDATE documents SET path=? WHERE id=?", (str(victim), a))
    assert purge(conn, cfg) == 0
    assert victim.exists()
    assert conn.execute("SELECT COUNT(*) c FROM documents WHERE id=?", (a,)).fetchone()["c"] == 1
    assert conn.execute(
        "SELECT COUNT(*) c FROM events WHERE kind='knowledge.purge_skipped'").fetchone()["c"] == 1


def test_removing_twice_is_an_error(env):
    conn, cfg = env
    a = write(conn, cfg, "T", "## A\nx\n", repo="r")
    remove(conn, cfg, a)
    with pytest.raises(ValueError, match="already deleted"):
        remove(conn, cfg, a)


def test_write_never_overwrites_a_file_created_after_the_name_was_picked(env, monkeypatch):
    """exists() then write is a race; exclusive create must bump the counter instead."""
    conn, cfg = env
    first = write(conn, cfg, "Same", "## A\nfirst body\n", repo="r")
    monkeypatch.setattr(Path, "exists", lambda self: False)
    second = write(conn, cfg, "Same", "## A\nsecond body\n", repo="r")
    monkeypatch.undo()
    assert "first body" in _path(conn, first).read_text()
    assert "second body" in _path(conn, second).read_text()


def test_write_with_unknown_supersedes_leaves_nothing_behind(env):
    conn, cfg = env
    with pytest.raises(KeyError):
        write(conn, cfg, "T", "## A\nx\n", repo="r", supersedes=["k_nope"])
    assert not list(cfg.knowledge_dir.rglob("*.md"))
    assert conn.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"] == 0


def test_update_rejects_frontmatter_keys_a_caller_may_not_set(env):
    conn, cfg = env
    a = write(conn, cfg, "T", "## A\nx\n", repo="r")
    with pytest.raises(ValueError, match="id"):
        update(conn, cfg, a, id="k_forged")
    assert "k_forged" not in _path(conn, a).read_text()


def test_self_citing_document_has_no_self_neighbour(env):
    conn, cfg = env
    a = write(conn, cfg, "T", "## A\nx\n", repo="r")
    update(conn, cfg, a, body=f"## A\nsee [[{a}]]\n")
    assert related(conn, a) == []
    assert neighbour_counts(conn, [a]).get(a, 0) == 0


def test_a_symlink_out_of_a_writable_root_is_not_writable(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")
    cfg.knowledge_dir = tmp_path / "knowledge"
    cfg.knowledge_dir.mkdir()
    root = tmp_path / "ws"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "link").symlink_to(outside)
    cfg.index_roots = [IndexRoot(path=str(root), source="shared-notes", writable=True)]
    target = outside / "x.md"
    target.write_text("## A\nx\n")
    assert knowledge._owning_root(cfg, root / "link" / "x.md") is None


def _feed_dataset(conn, cfg, tmp_path, name="prices", repo="r"):
    """A dataset document as `feed` writes it, with the id `remove` would take."""
    source = tmp_path / f"{name}.csv"
    source.write_text("item,price\nbolt,4\n", encoding="utf-8")
    feed.feed(conn, cfg, source, name, FeedSettings(key="item"), repo=repo)
    doc = cfg.knowledge_dir / repo / "datasets" / f"{name}.md"
    doc_id = conn.execute("SELECT id FROM documents WHERE path = ?",
                          (str(doc),)).fetchone()["id"]
    return doc, doc_id


def test_update_append_archive_remove_refuse_dataset_docs(env, tmp_path):
    """A dataset document is generated by `feed`, so an edit to it is thrown away on the
    next import: every knowledge write path refuses it and names the commands that work."""
    conn, cfg = env
    doc, doc_id = _feed_dataset(conn, cfg, tmp_path)
    text_before = doc.read_text()

    for writing in (
        lambda: update(conn, cfg, doc_id, body="replaced"),
        lambda: update(conn, cfg, doc_id, match="Fed from", replacement="Fed at"),
        lambda: append(conn, cfg, doc_id, "more"),
        lambda: archive(conn, cfg, doc_id),
        lambda: remove(conn, cfg, doc_id),
    ):
        with pytest.raises(feed.DatasetReadOnly, match="'prices' is a dataset"):
            writing()

    assert doc.read_text() == text_before
    assert search(conn, "bolt", all_repos=True)


def test_supersedes_naming_a_dataset_is_refused(env, tmp_path):
    """Superseding archives the old document, which is the same throwaway edit as any
    other write to a dataset: it is refused before the new document exists."""
    conn, cfg = env
    doc, doc_id = _feed_dataset(conn, cfg, tmp_path)
    text_before = doc.read_text()

    with pytest.raises(feed.DatasetReadOnly, match="'prices' is a dataset"):
        write(conn, cfg, "Replacement", "body\n", repo="r", supersedes=[doc_id])

    assert doc.read_text() == text_before
    assert conn.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"] == 1
