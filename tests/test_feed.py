"""Feeding a table export into a dataset document, as one crash-safe operation.

Every test writes synthetic rows to a temporary directory and inspects the result cell by
cell, so the suite needs no real export and no real knowledge base.
"""
from __future__ import annotations

import errno
import hashlib
import os
import re
import sqlite3
import sys
import threading
import types
from datetime import datetime, timezone
from importlib import reload
from pathlib import Path

import pytest

from akasha import feed, feed_store, knowledge
from akasha.config import IndexRoot, load_config
from akasha.db import connect
from akasha.index import index_path
from akasha.markdown import parse_frontmatter
from akasha.search import search
from akasha.splitters import FeedSettings

ROWS = "item,price\nbolt,4\nnut,2\n"


@pytest.fixture
def env(tmp_path):
    cfg = load_config(tmp_path / "config.toml")
    cfg.knowledge_dir = tmp_path / "knowledge"
    cfg.knowledge_dir.mkdir()
    cfg.index_roots = []
    return connect(tmp_path / "s.db"), cfg


@pytest.fixture
def source(tmp_path):
    def write(name="prices.csv", rows=ROWS):
        path = tmp_path / name
        path.write_text(rows, encoding="utf-8")
        return path

    return write


def _copy(cfg, name="prices", repo="sample-repo", suffix=".csv"):
    return cfg.knowledge_dir.parent / "files" / repo / f"{name}{suffix}"


def _doc(cfg, name="prices", repo="sample-repo"):
    return cfg.knowledge_dir / repo / "datasets" / f"{name}.md"


def _unit_chunks(conn, doc_id):
    return conn.execute(
        "SELECT c.id, c.body FROM chunks c JOIN feed_chunks f ON f.chunk_id = c.id"
        " WHERE c.document_id = ? ORDER BY c.ord", (doc_id,)).fetchall()


def _cell_ids(conn, doc_id):
    return {(r["locator"], r["field"]): r["chunk_id"] for r in conn.execute(
        "SELECT f.locator, f.field, f.chunk_id FROM feed_chunks f JOIN chunks c"
        " ON c.id = f.chunk_id WHERE c.document_id = ?", (doc_id,))}


def _boom(*_args, **_kwargs):
    raise RuntimeError("reindex exploded")


def _xlsx(tmp_path, name="prices.xlsx", rows=(("item", "price"), ("bolt", 4), ("nut", 2))):
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    for row in rows:
        sheet.append(list(row))
    path = tmp_path / name
    workbook.save(path)
    return path


def test_first_feed_copies_file_writes_doc_with_settings_and_unit_chunks(env, source):
    """A first feed is the whole contract in one call: the copy is byte-identical, the
    frontmatter round-trips its settings, and every cell becomes a search chunk."""
    conn, cfg = env
    src = source()
    settings = FeedSettings(key="item")

    result = feed.feed(conn, cfg, src, "prices", settings, repo="sample-repo")

    copy = _copy(cfg)
    assert result.unchanged is False
    assert result.diff is None
    assert (result.name, result.repo, result.units) == ("prices", "sample-repo", 2)
    assert result.copy == copy and copy.read_bytes() == src.read_bytes()

    meta, body = parse_frontmatter(_doc(cfg).read_text())
    assert "id" not in meta
    assert meta["title"] == "Dataset: prices"
    assert (meta["kind"], meta["dataset"], meta["feature"], meta["repo"]) == (
        "dataset", "prices", "prices", "sample-repo")
    assert meta["file"] == str(copy)
    assert meta["sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    assert meta["settings_sha256"] == settings.sha256()
    assert meta["settings"] == {"key": "item", "sheet": None, "header_row": 1,
                                "require": [], "exclude": [], "label": None}
    assert "Fed from prices.csv on " in body
    assert ": 2 units, 2 fields." in body
    assert "## Fields" in body and "\n- item" in body and "\n- price" in body

    assert [unit.key for unit in feed_store.load_units(conn, "prices")] == ["bolt", "nut"]
    assert feed_store.stored_state(conn, "prices") == feed.unit_state(
        meta["sha256"], settings.sha256(), cfg)
    row = conn.execute("SELECT id, kind FROM documents WHERE path = ?",
                       (str(_doc(cfg)),)).fetchone()
    assert row["kind"] == "dataset"
    assert result.doc_id == row["id"]
    # The registration path also leaves the kind usable for the next write.
    assert "dataset" in cfg.custom_kinds
    assert [chunk["body"] for chunk in _unit_chunks(conn, row["id"])] == [
        "prices bolt · item: bolt", "prices bolt · price: 4",
        "prices nut · item: nut", "prices nut · price: 2"]


def test_first_feed_without_settings_raises_and_writes_nothing(env, source):
    """Settings decide which rows and columns a table has, so a first feed cannot guess
    them, and the attempt must not leave a copy or a document behind."""
    conn, cfg = env

    with pytest.raises(ValueError, match="needs settings"):
        feed.feed(conn, cfg, source(), "prices", repo="sample-repo")

    assert not _copy(cfg).exists()
    assert not _doc(cfg).exists()
    assert feed_store.load_units(conn, "prices") == []


def test_refeed_reports_added_removed_changed_and_prepends_change_log(env, source):
    """The change report is the point of a re-feed, and the log keeps the earlier feeds
    underneath the newest one, with identities and field names but no cell values."""
    conn, cfg = env
    settings = FeedSettings(key="item", require=["price"])
    feed.feed(conn, cfg, source(), "prices", settings, repo="sample-repo")

    # nut's price is empty, so `require` drops it; screw is a new row below the old ones.
    changed = source(rows="item,price\nbolt,5\nnut,\nscrew,1\n")
    result = feed.feed(conn, cfg, changed, "prices", settings, repo="sample-repo")

    assert [(unit.key, unit.fields["price"]) for unit in result.diff.added] == [("screw", "1")]
    assert [unit.key for unit in result.diff.removed] == ["nut"]
    assert [change.identity for change in result.diff.changed] == ["key:bolt"]
    assert result.copy.read_bytes() == changed.read_bytes()

    log = _doc(cfg).read_text().split("## Change log", 1)[1]
    assert log.count("### ") == 2
    assert log.index("+1 -1 ~1") < log.index("first feed: 2 units")
    assert "- key:bolt prices bolt: price" in log
    assert "- key:screw prices screw: item, price" in log
    # 5 is the changed cell's value: an entry may name the field, never the value.
    entries = [line for line in log.splitlines() if line.startswith("- ")]
    assert entries and all("5" not in line for line in entries)


def test_refeed_identical_file_and_settings_is_unchanged_and_keeps_chunk_ids(env, source):
    """An unchanged copy must not rewrite the document or re-chunk the cells, or every
    stored vector for a dataset nobody touched would be thrown away."""
    conn, cfg = env
    src = source()
    first = feed.feed(conn, cfg, src, "prices", FeedSettings(key="item"), repo="sample-repo")
    doc_before = _doc(cfg).read_text()
    ids_before = _cell_ids(conn, first.doc_id)

    again = feed.feed(conn, cfg, src, "prices", None, repo="sample-repo")

    assert again.unchanged is True and again.diff is None
    assert (again.units, again.copy, again.doc_id) == (2, first.copy, first.doc_id)
    assert _doc(cfg).read_text() == doc_before
    assert _cell_ids(conn, first.doc_id) == ids_before


def test_refeed_same_file_new_exclude_resplits_and_stores_settings(env, source):
    """Settings given on a re-feed replace the stored ones, so the units are rebuilt from
    the same bytes rather than read back under the old schema."""
    conn, cfg = env
    src = source()
    feed.feed(conn, cfg, src, "prices", FeedSettings(key="item"), repo="sample-repo")

    narrower = FeedSettings(key="item", exclude=["price"])
    result = feed.feed(conn, cfg, src, "prices", narrower, repo="sample-repo")

    meta, _body = parse_frontmatter(_doc(cfg).read_text())
    assert meta["settings"]["exclude"] == ["price"]
    assert meta["settings_sha256"] == narrower.sha256()
    assert [unit.fields for unit in feed_store.load_units(conn, "prices")] == [
        {"item": "bolt"}, {"item": "nut"}]
    assert [change.fields for change in result.diff.changed] == [["price"], ["price"]]


def test_refeed_cosmetic_change_keeps_ids_of_unchanged_cells(env, source):
    """A chunk id carries its embedded text, so changing one cell must leave every other
    cell's id alone and spare its stored vector."""
    conn, cfg = env
    settings = FeedSettings(key="item")
    first = feed.feed(conn, cfg, source(), "prices", settings, repo="sample-repo")
    before = _cell_ids(conn, first.doc_id)

    second = feed.feed(conn, cfg, source(rows="item,price\nbolt,7\nnut,2\n"), "prices",
                       settings, repo="sample-repo")
    after = _cell_ids(conn, second.doc_id)

    assert after[("row:2", "item")] == before[("row:2", "item")]
    assert after[("row:3", "price")] == before[("row:3", "price")]
    assert after[("row:2", "price")] != before[("row:2", "price")]


def test_refeed_with_missing_key_column_changes_nothing(env, source):
    """A split that fails is not a table with fewer rows: the last good copy, units,
    settings and document must all survive it untouched."""
    conn, cfg = env
    feed.feed(conn, cfg, source(), "prices", FeedSettings(key="item"), repo="sample-repo")
    doc_before = _doc(cfg).read_text()
    copy_before = _copy(cfg).read_bytes()
    units_before = feed_store.load_units(conn, "prices")
    state_before = feed_store.stored_state(conn, "prices")
    doc_mtime = os.stat(_doc(cfg)).st_mtime

    with pytest.raises(ValueError, match="unknown column 'item'"):
        feed.feed(conn, cfg, source(rows="sku,price\nbolt,4\n"), "prices", None,
                  repo="sample-repo")

    assert _copy(cfg).read_bytes() == copy_before
    assert _doc(cfg).read_text() == doc_before
    assert feed_store.load_units(conn, "prices") == units_before
    assert feed_store.stored_state(conn, "prices") == state_before
    assert os.stat(_doc(cfg)).st_mtime == doc_mtime


def test_failure_inside_transaction_rolls_back_and_restores_copy_and_doc(
        env, source, monkeypatch):
    """The copy and the document are written before the transaction, so a failure inside it
    must put both back: a rolled-back database beside a new file is a torn feed."""
    conn, cfg = env
    settings = FeedSettings(key="item")
    feed.feed(conn, cfg, source(), "prices", settings, repo="sample-repo")
    doc_before = _doc(cfg).read_text()
    copy_before = _copy(cfg).read_bytes()
    units_before = feed_store.load_units(conn, "prices")
    state_before = feed_store.stored_state(conn, "prices")

    monkeypatch.setattr(knowledge, "_reindex", _boom)
    with pytest.raises(RuntimeError, match="reindex exploded"):
        feed.feed(conn, cfg, source(rows="item,price\nbolt,9\n"), "prices", settings,
                  repo="sample-repo")

    assert _copy(cfg).read_bytes() == copy_before
    assert _doc(cfg).read_text() == doc_before
    assert feed_store.load_units(conn, "prices") == units_before
    assert feed_store.stored_state(conn, "prices") == state_before
    assert conn.in_transaction is False


def test_failed_first_feed_leaves_no_copy_and_no_document(env, source, monkeypatch):
    """A first feed creates both files, so its rollback has nothing to restore: both must
    be gone, or the next feed would treat a torn dataset as an existing one."""
    conn, cfg = env

    monkeypatch.setattr(knowledge, "_reindex", _boom)
    with pytest.raises(RuntimeError, match="reindex exploded"):
        feed.feed(conn, cfg, source(), "prices", FeedSettings(key="item"),
                  repo="sample-repo")

    assert not _copy(cfg).exists()
    assert not _doc(cfg).exists()
    assert list((cfg.knowledge_dir.parent / "files").rglob("*.staging*")) == []
    assert feed_store.load_units(conn, "prices") == []


def test_feed_switching_extension_trashes_old_copy(env, source, tmp_path):
    """The copy's extension follows the incoming file, so the previous copy has no name to
    live at any more and goes to the trash instead of being overwritten."""
    conn, cfg = env
    settings = FeedSettings(key="item")
    feed.feed(conn, cfg, source(), "prices", settings, repo="sample-repo")
    old_bytes = _copy(cfg).read_bytes()

    book = _xlsx(tmp_path)

    result = feed.feed(conn, cfg, book, "prices", settings, repo="sample-repo")

    assert result.copy == _copy(cfg, suffix=".xlsx")
    assert not _copy(cfg, suffix=".csv").exists()
    trashed = list((cfg.knowledge_dir.parent / "files" / ".trash" / "sample-repo")
                   .glob("prices-*.csv"))
    assert len(trashed) == 1
    assert trashed[0].read_bytes() == old_bytes


def test_failed_extension_switch_restores_the_old_copy_and_drops_the_new_one(
        env, source, tmp_path, monkeypatch):
    """A failed feed that changed the extension leaves two names in play, so the rollback
    must delete the new copy and restore the old one: a stray copy is a torn feed too."""
    conn, cfg = env
    settings = FeedSettings(key="item")
    feed.feed(conn, cfg, source(), "prices", settings, repo="sample-repo")
    copy_before = _copy(cfg).read_bytes()
    doc_before = _doc(cfg).read_text()

    book = _xlsx(tmp_path, rows=(("item", "price"), ("bolt", 4)))

    monkeypatch.setattr(knowledge, "_reindex", _boom)
    with pytest.raises(RuntimeError, match="reindex exploded"):
        feed.feed(conn, cfg, book, "prices", settings, repo="sample-repo")

    assert _copy(cfg).read_bytes() == copy_before
    assert not _copy(cfg, suffix=".xlsx").exists()
    assert _doc(cfg).read_text() == doc_before
    assert not list((cfg.knowledge_dir.parent / "files").rglob("*.bak"))


def test_name_fed_under_other_repo_is_refused(env, source):
    """Names are global: a second repo feeding one name would fight over one document and
    one set of units, so the refusal names the repo that already owns it."""
    conn, cfg = env
    src = source()
    feed.feed(conn, cfg, src, "prices", FeedSettings(key="item"), repo="sample-repo")
    doc_before = _doc(cfg).read_text()

    with pytest.raises(ValueError, match="is fed under repo sample-repo"):
        feed.feed(conn, cfg, src, "prices", FeedSettings(key="item"), repo="other-repo")

    assert _doc(cfg).read_text() == doc_before
    assert not (cfg.knowledge_dir / "other-repo").exists()


def test_repo_defaults_to_global_outside_checkout(env, source, monkeypatch, tmp_path):
    """A feed from a directory that is not a checkout has no repo to derive, and `_global`
    is where every other repo-less document already lives."""
    conn, cfg = env
    monkeypatch.chdir(tmp_path)

    result = feed.feed(conn, cfg, source(), "prices", FeedSettings(key="item"))

    assert result.repo == "_global"
    assert _doc(cfg, repo="_global").exists()


def test_invalid_name_and_unsupported_extension_and_denied_file_are_refused(
        env, source, tmp_path):
    """Each refusal happens before anything is written and names what was wrong: a bad
    name, a file type no splitter reads, a missing file, and a denied file."""
    conn, cfg = env
    src = source()

    for bad in ("Prices", "-prices", "p" * 25, ""):
        with pytest.raises(ValueError, match="invalid dataset name"):
            feed.feed(conn, cfg, src, bad, FeedSettings(key="item"), repo="sample-repo")
    with pytest.raises(ValueError, match="unsupported file type"):
        feed.feed(conn, cfg, source(name="prices.txt"), "prices", FeedSettings(key="item"),
                  repo="sample-repo")
    with pytest.raises(FileNotFoundError):
        feed.feed(conn, cfg, tmp_path / "absent.csv", "prices", FeedSettings(key="item"),
                  repo="sample-repo")

    cfg.deny_files = ["prices.csv"]
    with pytest.raises(PermissionError):
        feed.feed(conn, cfg, src, "prices", FeedSettings(key="item"), repo="sample-repo")

    assert not (cfg.knowledge_dir / "sample-repo").exists()
    assert feed_store.load_units(conn, "prices") == []


def test_secret_cell_is_redacted_and_counted(env, source):
    """A cell holds what a paragraph holds, so it goes through the same scanner once on
    the way in, and the count says how much the feed scrubbed."""
    conn, cfg = env

    # The key column is ordinary text: a redacted key falls back to the locator, which
    # would make this a test of that rule instead of the cell's.
    result = feed.feed(
        conn, cfg,
        source(rows="item,note\nbolt,Authorization: Bearer abcdef1234567890abcdef\n"),
        "prices", FeedSettings(key="item"), repo="sample-repo")

    assert result.redacted == 1
    assert feed_store.load_units(conn, "prices")[0].fields["note"] == "[REDACTED:bearer]"
    assert "abcdef1234567890abcdef" not in _doc(cfg).read_text()


def test_feed_embeds_unit_chunks_when_vectors_available(env, source, monkeypatch):
    """A feed writes the chunks and then embeds them, so a just-fed dataset is searchable
    by meaning too; without the extra or the extension the feed must still succeed."""
    import numpy as np

    from akasha import vectors

    conn, cfg = env
    if vectors.missing_modules() or not vectors.available(conn):
        pytest.skip("the vectors extra is not usable here")

    def encode(texts):
        return np.array([[1.0] + [0.0] * 255 for _ in texts], dtype=np.float32)

    monkeypatch.setattr(vectors, "_encoder", lambda: encode)
    cfg.embeddings_provider = "model2vec"

    feed.feed(conn, cfg, source(), "prices", FeedSettings(key="item"), repo="sample-repo")

    chunks = conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]
    stored = conn.execute("SELECT COUNT(*) c FROM vec_chunks").fetchone()["c"]
    assert stored == chunks > 0


def test_large_file_warns(env, source, monkeypatch):
    """A big table is still fed: the warning names the cost the caller is about to pay
    rather than refusing the file."""
    conn, cfg = env
    monkeypatch.setattr(feed, "LARGE_FILE_BYTES", 10)

    result = feed.feed(conn, cfg, source(), "prices", FeedSettings(key="item"),
                       repo="sample-repo")

    assert len(result.warnings) == 1
    assert result.warnings[0].startswith("prices.csv is ")
    assert result.warnings[0].endswith(" MB; feeding anyway")


def test_splitter_warnings_are_reported(env, source):
    """A duplicate key falls back to the locator, so the rows are addressed differently
    than the settings promise; the splitter's warning has to reach the caller."""
    conn, cfg = env

    result = feed.feed(conn, cfg, source(rows="item,price\nbolt,4\nbolt,2\n"), "prices",
                       FeedSettings(key="item"), repo="sample-repo")

    assert any("duplicate" in warning for warning in result.warnings)
    assert [unit.key for unit in feed_store.load_units(conn, "prices")] == [None, None]


def test_refeed_after_copy_deleted_restores_copy(env, source):
    """A deleted copy is never unchanged: re-feeding the source must write it again."""
    conn, cfg = env
    src = source()
    settings = FeedSettings(key="item")
    feed.feed(conn, cfg, src, "prices", settings, repo="sample-repo")
    _copy(cfg).unlink()

    result = feed.feed(conn, cfg, src, "prices", settings, repo="sample-repo")

    assert result.unchanged is False
    assert _copy(cfg).read_bytes() == src.read_bytes()


def test_attach_failure_during_feed_rolls_back(env, source, monkeypatch):
    """attach records a data problem instead of raising; for a feed the units it stored
    were not the ones attached, so the feed must roll back and raise."""
    conn, cfg = env
    settings = FeedSettings(key="item")
    feed.feed(conn, cfg, source(), "prices", settings, repo="sample-repo")
    doc_before = _doc(cfg).read_text()
    copy_before = _copy(cfg).read_bytes()
    units_before = feed_store.load_units(conn, "prices")

    def broken_attach(conn, cfg, doc_id, meta, **kwargs):
        feed.ATTACH_ERRORS.append((meta["dataset"], "cannot read the copy"))
        return 0

    monkeypatch.setattr(feed, "ATTACH_ERRORS", [])
    monkeypatch.setattr(feed, "attach", broken_attach)
    with pytest.raises(ValueError, match="cannot read the copy"):
        feed.feed(conn, cfg, source(rows="item,price\nbolt,9\n"), "prices", settings,
                  repo="sample-repo")

    assert _copy(cfg).read_bytes() == copy_before
    assert _doc(cfg).read_text() == doc_before
    assert feed_store.load_units(conn, "prices") == units_before
    assert conn.in_transaction is False


def test_header_named_change_log_does_not_corrupt_carried_log(env, source):
    """A column named `## Change log` sits in the fields list, so carrying the old log by
    searching the raw text would splice the whole document into the carried part."""
    conn, cfg = env
    settings = FeedSettings(key="price")
    rows = "## Change log,price\nbolt,4\nnut,2\n"
    feed.feed(conn, cfg, source(rows=rows), "prices", settings, repo="sample-repo")
    feed.feed(conn, cfg, source(rows=rows + "screw,1\n"), "prices", settings,
              repo="sample-repo")

    body = _doc(cfg).read_text()
    assert body.count("## How to read rows") == 1
    log = body.split("## Change log", 1)[1]
    assert log.count("### ") == 2
    assert "first feed: 2 units" in log
    assert "+1 -0 ~0" in log


def test_post_commit_cleanup_failure_is_a_warning(env, source, tmp_path, monkeypatch):
    """By then the feed has committed, so a failing trash move is reported, not raised:
    raising would tell the caller nothing landed when everything did."""
    conn, cfg = env
    settings = FeedSettings(key="item")
    feed.feed(conn, cfg, source(), "prices", settings, repo="sample-repo")
    book = _xlsx(tmp_path)
    real_replace = os.replace

    def flaky_replace(src, dst):
        if ".trash" in str(dst):
            raise OSError("trash is full")
        return real_replace(src, dst)

    monkeypatch.setattr(feed.os, "replace", flaky_replace)

    result = feed.feed(conn, cfg, book, "prices", settings, repo="sample-repo")

    assert result.copy == _copy(cfg, suffix=".xlsx")
    assert any("trash is full" in warning for warning in result.warnings)
    assert [unit.key for unit in feed_store.load_units(conn, "prices")] == ["bolt", "nut"]


def test_failed_second_backup_rename_restores_first(env, source, tmp_path, monkeypatch):
    """An extension switch renames two copies aside; a failure on the second must still
    put the first back, or the rollback loses a copy it never recorded."""
    conn, cfg = env
    settings = FeedSettings(key="item")
    feed.feed(conn, cfg, source(), "prices", settings, repo="sample-repo")
    csv_before = _copy(cfg).read_bytes()
    stale = _copy(cfg, suffix=".xlsx")
    stale.write_text("not even a workbook", encoding="utf-8")
    doc_before = _doc(cfg).read_text()

    real_replace = os.replace

    def flaky_replace(src, dst):
        if str(src).endswith("prices.xlsx"):
            raise OSError("disk went away")
        return real_replace(src, dst)

    monkeypatch.setattr(feed.os, "replace", flaky_replace)
    with pytest.raises(OSError, match="disk went away"):
        feed.feed(conn, cfg, _xlsx(tmp_path), "prices", settings, repo="sample-repo")

    assert _copy(cfg).read_bytes() == csv_before
    assert _doc(cfg).read_text() == doc_before
    assert not list((cfg.knowledge_dir.parent / "files").rglob("*.bak"))


def test_failed_final_replace_restores_old_copy_and_removes_tmp(env, source, tmp_path,
                                                                 monkeypatch):
    """The swap into place is the last file step, so its failure must leave the old copy
    and no staging or temporary file behind: a partial copy is a torn feed."""
    conn, cfg = env
    settings = FeedSettings(key="item")
    feed.feed(conn, cfg, source(), "prices", settings, repo="sample-repo")
    csv_before = _copy(cfg).read_bytes()
    doc_before = _doc(cfg).read_text()
    units_before = feed_store.load_units(conn, "prices")

    real_replace = os.replace

    def flaky_replace(src, dst):
        if str(src).endswith(".staging.xlsx"):
            raise OSError("swap failed")
        return real_replace(src, dst)

    monkeypatch.setattr(feed.os, "replace", flaky_replace)
    with pytest.raises(OSError, match="swap failed"):
        feed.feed(conn, cfg, _xlsx(tmp_path), "prices", settings, repo="sample-repo")

    files = cfg.knowledge_dir.parent / "files"
    assert _copy(cfg).read_bytes() == csv_before
    assert _doc(cfg).read_text() == doc_before
    assert feed_store.load_units(conn, "prices") == units_before
    assert not list(files.rglob("*.staging*"))
    assert not list(files.rglob("*.tmp"))
    assert not list(files.rglob("*.bak"))


def test_second_feed_of_same_name_waits_for_lock(env, source, tmp_path):
    """Two feeds of one name share the staging, backup and copy paths before the database
    lock, so the second has to wait for the first rather than trade files with it."""
    conn, cfg = env
    src = source()
    settings = FeedSettings(key="item")
    hold = feed._lock(cfg, "prices")
    started = threading.Event()
    done = threading.Event()
    outcome: list = []

    def run():
        worker = connect(tmp_path / "s.db")
        started.set()
        try:
            outcome.append(feed.feed(worker, cfg, src, "prices", settings,
                                     repo="sample-repo"))
        except BaseException as exc:
            outcome.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert started.wait(5) is True        # the worker reached the feed call
        assert done.wait(0.3) is False        # blocked on the lock this test holds
    finally:
        feed._unlock(hold)
    assert done.wait(10) is True
    thread.join()
    assert isinstance(outcome[0], feed.FeedResult)
    assert outcome[0].units == 2


def test_feed_module_imports_without_fcntl(monkeypatch):
    """akasha runs on Windows, where there is no `fcntl` and `index_all` imports this
    module, so the lock is imported inside `_lock` rather than at load time."""
    monkeypatch.setitem(sys.modules, "fcntl", None)
    try:
        reload(feed)
    finally:
        monkeypatch.undo()
        reload(feed)

    assert callable(feed.feed) and callable(feed._unlock)


@pytest.mark.parametrize("repo", [".locks", ".trash", ".Locks", ".TRASH"])
def test_reserved_repo_names_are_refused(env, source, repo):
    """`files/.locks` and `files/.trash` are akasha own, so a repo of that name would put
    a dataset copy where the lock files and the trash live."""
    conn, cfg = env

    with pytest.raises(ValueError, match="invalid repo"):
        feed.feed(conn, cfg, source(), "prices", FeedSettings(key="item"), repo=repo)

    assert not (cfg.knowledge_dir / repo).exists()
    assert feed_store.load_units(conn, "prices") == []


def test_failed_restore_continues_and_reraises_original(env, source, monkeypatch):
    """A cleanup step that fails must not replace the error the caller can act on: the
    failure is noted on the original exception and the other restores still run."""
    conn, cfg = env
    settings = FeedSettings(key="item")
    feed.feed(conn, cfg, source(), "prices", settings, repo="sample-repo")
    doc_before = _doc(cfg).read_text()
    real_replace = os.replace

    def flaky_replace(src, dst):
        if str(src).endswith(".bak"):
            raise OSError("backup is gone")
        return real_replace(src, dst)

    monkeypatch.setattr(feed.os, "replace", flaky_replace)
    monkeypatch.setattr(knowledge, "_reindex", _boom)
    with pytest.raises(RuntimeError, match="reindex exploded") as raised:
        feed.feed(conn, cfg, source(rows="item,price\nbolt,9\n"), "prices", settings,
                  repo="sample-repo")

    assert any("backup is gone" in note for note in raised.value.__notes__)
    assert not list((cfg.knowledge_dir.parent / "files").rglob("*.staging*"))
    assert _doc(cfg).read_text() == doc_before


def _doc_id(conn, cfg, name="prices", repo="sample-repo"):
    return conn.execute("SELECT id FROM documents WHERE path = ?",
                        (str(_doc(cfg, name, repo)),)).fetchone()["id"]


def test_remove_dataset_trashes_doc_and_copy_and_drops_units(env, source):
    """Removing a dataset has to take all three things that make it: its document, its
    copy, and the unit rows search would otherwise keep answering from."""
    conn, cfg = env
    feed.feed(conn, cfg, source(), "prices", FeedSettings(key="item"), repo="sample-repo")
    copy_bytes = _copy(cfg).read_bytes()
    doc_text = _doc(cfg).read_text()

    doc_trash, copy_trash = feed.remove_dataset(conn, cfg, "prices")

    assert not _doc(cfg).exists()
    assert not _copy(cfg).exists()
    assert doc_trash.parent == cfg.knowledge_dir.parent / "trash"
    assert doc_trash.read_text() == doc_text
    assert copy_trash.parent == cfg.knowledge_dir.parent / "files" / ".trash" / "sample-repo"
    assert re.fullmatch(r"prices-\d{8}T\d{6}\.csv", copy_trash.name)
    assert copy_trash.read_bytes() == copy_bytes
    assert feed_store.load_units(conn, "prices") == []
    assert feed_store.stored_state(conn, "prices") is None
    assert conn.execute("SELECT COUNT(*) c FROM feed_chunks WHERE dataset = 'prices'"
                        ).fetchone()["c"] == 0
    assert not search(conn, "bolt", all_repos=True, include_archived=True)


def test_remove_unknown_dataset_is_keyerror(env):
    """The name is global, so an unknown one has to say so rather than trash whatever
    happens to sit at the path a guess would build."""
    conn, cfg = env

    with pytest.raises(KeyError, match="no dataset 'ghost'"):
        feed.remove_dataset(conn, cfg, "ghost")


def test_remove_dataset_restores_the_copy_when_the_db_part_fails(env, source, monkeypatch):
    """The copy leaves before the transaction, so a failure there must put it back: a live
    document whose copy sits in the trash can neither be fed again nor restored."""
    conn, cfg = env
    feed.feed(conn, cfg, source(), "prices", FeedSettings(key="item"), repo="sample-repo")
    copy_bytes = _copy(cfg).read_bytes()

    def boom(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(feed_store, "drop_dataset", boom)
    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        feed.remove_dataset(conn, cfg, "prices")

    assert _copy(cfg).read_bytes() == copy_bytes
    assert _doc(cfg).is_file()
    assert conn.in_transaction is False


def test_restore_dataset_brings_copy_back_and_reattaches_units(env, source):
    """A dataset's cells are split from its copy, so a restore that left the copy in the
    trash would bring back a document whose cells nothing can produce."""
    conn, cfg = env
    feed.feed(conn, cfg, source(), "prices", FeedSettings(key="item"), repo="sample-repo")
    copy_bytes = _copy(cfg).read_bytes()
    doc_id = _doc_id(conn, cfg)
    feed.remove_dataset(conn, cfg, "prices")
    assert not search(conn, "bolt", all_repos=True, include_archived=True)

    path = knowledge.restore(conn, cfg, doc_id)

    assert path == _doc(cfg)
    assert _copy(cfg).read_bytes() == copy_bytes
    assert len(feed_store.load_units(conn, "prices")) == 2
    assert len(_unit_chunks(conn, doc_id)) == 4          # two rows of two fields
    assert search(conn, "bolt", all_repos=True)


def test_restore_dataset_with_purged_copy_refuses_and_changes_nothing(env, source):
    """Without its copy the restored document would claim units it cannot produce, so the
    refusal comes before the document leaves the trash."""
    conn, cfg = env
    feed.feed(conn, cfg, source(), "prices", FeedSettings(key="item"), repo="sample-repo")
    doc_id = _doc_id(conn, cfg)
    doc_trash, copy_trash = feed.remove_dataset(conn, cfg, "prices")
    copy_trash.unlink()
    trashed_text = doc_trash.read_text()

    with pytest.raises(ValueError, match="copy purged"):
        knowledge.restore(conn, cfg, doc_id)

    assert doc_trash.read_text() == trashed_text
    assert not _doc(cfg).exists()
    row = conn.execute("SELECT path, deleted_at FROM documents WHERE id = ?",
                       (doc_id,)).fetchone()
    assert row["deleted_at"] and row["path"] == str(doc_trash)


def test_purge_deletes_trashed_copies_inside_trash_only(env, source):
    """A copy is not a document row, so nothing else would ever clear one out of
    `files/.trash`; `purge` owns it, by the same rules as the documents: the cutoff, and
    never outside the trash."""
    conn, cfg = env
    feed.feed(conn, cfg, source(), "prices", FeedSettings(key="item"), repo="sample-repo")
    _doc_trash, copy_trash = feed.remove_dataset(conn, cfg, "prices")

    outside = cfg.knowledge_dir.parent / "files" / "sample-repo" / "keep.csv"
    outside.write_text("item,price\nbolt,4\n", encoding="utf-8")
    through = copy_trash.parent / "through.csv"
    os.symlink(outside, through)

    assert knowledge.purge(conn, cfg, older_than_days=1) == 0     # trashed a moment ago

    assert knowledge.purge(conn, cfg) == 2                        # one document, one copy
    assert not copy_trash.exists()
    assert outside.exists()
    assert outside.read_text(encoding="utf-8").startswith("item,price")
    assert through.is_symlink()


def test_refeed_after_remove_starts_fresh(env, source):
    """A removed dataset has no units and no document left, so the next feed is a first
    feed: it must not diff against what the removed dataset used to hold."""
    conn, cfg = env
    settings = FeedSettings(key="item")
    feed.feed(conn, cfg, source(), "prices", settings, repo="sample-repo")
    feed.remove_dataset(conn, cfg, "prices")

    result = feed.feed(conn, cfg, source(rows="item,price\nbolt,4\n"), "prices", settings,
                       repo="sample-repo")

    assert result.diff is None
    assert result.unchanged is False
    assert result.units == 1
    _meta, body = parse_frontmatter(_doc(cfg).read_text())
    assert "first feed: 1 units" in body


def test_windows_lock_retries_until_acquired(env, monkeypatch):
    """`msvcrt.locking(LK_LOCK)` gives up after about ten seconds where `flock` blocks, so
    on Windows a second feed of one name would fail instead of waiting for the first."""
    _conn, cfg = env
    modes: list[int] = []
    fake = types.SimpleNamespace(LK_LOCK=2, LK_NBLCK=3, LK_UNLCK=0)

    def locking(_fd, mode, _size):
        modes.append(mode)
        if mode == fake.LK_LOCK and modes.count(fake.LK_LOCK) < 3:
            raise OSError(errno.EACCES, "locking failed: another process holds")

    fake.locking = locking
    monkeypatch.setitem(sys.modules, "fcntl", None)
    monkeypatch.setitem(sys.modules, "msvcrt", fake)

    handle = feed._lock(cfg, "prices")
    assert modes.count(fake.LK_LOCK) == 3      # refused twice, then taken
    feed._unlock(handle)
    assert modes[-1] == fake.LK_UNLCK


class _NoRollback:
    """A connection whose ROLLBACK fails, so the rollback is the step under test."""

    def __init__(self, conn):
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def execute(self, sql, *parameters):
        if sql.lstrip().upper().startswith("ROLLBACK"):
            raise sqlite3.OperationalError("cannot rollback - no transaction is active")
        return self._conn.execute(sql, *parameters)


def test_rollback_failure_still_restores_files_and_raises_original(env, source, monkeypatch):
    """The rollback is a cleanup step like the file restores: one that fails must not stop
    them, and must not replace the reason the feed failed."""
    conn, cfg = env
    settings = FeedSettings(key="item")
    feed.feed(conn, cfg, source(), "prices", settings, repo="sample-repo")
    doc_before = _doc(cfg).read_text()
    copy_before = _copy(cfg).read_bytes()

    monkeypatch.setattr(knowledge, "_reindex", _boom)
    with pytest.raises(RuntimeError, match="reindex exploded") as raised:
        feed.feed(_NoRollback(conn), cfg, source(rows="item,price\nbolt,9\n"), "prices",
                  settings, repo="sample-repo")

    assert any("roll back" in note for note in raised.value.__notes__)
    assert _doc(cfg).read_text() == doc_before
    assert _copy(cfg).read_bytes() == copy_before
    assert not list((cfg.knowledge_dir.parent / "files").rglob("*.staging*"))


def test_windows_lock_reraises_an_error_that_is_not_contention(env, monkeypatch):
    """Only contention is worth waiting for: retrying a bad handle forever would turn one
    clear failure into a hang."""
    _conn, cfg = env
    attempts: list[int] = []
    fake = types.SimpleNamespace(LK_LOCK=2, LK_NBLCK=3, LK_UNLCK=0)

    def locking(_fd, mode, _size):
        attempts.append(mode)
        # Bounded on purpose: a loop that retries this fails the test instead of hanging.
        if len(attempts) > 3:
            raise AssertionError("retried an error that is not contention")
        raise OSError(errno.EBADF, "bad file descriptor")

    fake.locking = locking
    monkeypatch.setitem(sys.modules, "fcntl", None)
    monkeypatch.setitem(sys.modules, "msvcrt", fake)

    with pytest.raises(OSError, match="bad file descriptor"):
        feed._lock(cfg, "prices")

    assert len(attempts) == 1


@pytest.mark.parametrize("name", ["*", "../x", "Prices", "null", "true", "false"])
def test_names_that_cannot_round_trip_are_refused_by_both_commands(env, source, name):
    """The name becomes a glob, a path and a frontmatter value, so `feed` and
    `remove_dataset` check it the same way before either looks anything up: `*` and `../x`
    would escape the datasets folder, and `null`/`true`/`false` read back as None and a
    bool."""
    conn, cfg = env

    with pytest.raises(ValueError, match="invalid dataset name"):
        feed.remove_dataset(conn, cfg, name)
    with pytest.raises(ValueError, match="invalid dataset name"):
        feed.feed(conn, cfg, source(), name, FeedSettings(key="item"), repo="sample-repo")

    assert not (cfg.knowledge_dir.parent / "files" / ".locks").exists()
    assert list(cfg.knowledge_dir.glob("*")) == []


def test_restore_of_one_dataset_ignores_a_longer_name_in_the_trash(env, source):
    """`sales` and `sales-2024` are two datasets, and both copies sit in the same trash
    folder: a restore that matched `sales-` by prefix would hand `sales` the copy of
    `sales-2024`."""
    conn, cfg = env
    settings = FeedSettings(key="item")
    feed.feed(conn, cfg, source(), "sales", settings, repo="sample-repo")
    feed.feed(conn, cfg, source(), "sales-2024", settings, repo="sample-repo")
    sales_id = _doc_id(conn, cfg, "sales")
    sales_doc_trash, sales_copy_trash = feed.remove_dataset(conn, cfg, "sales")
    feed.remove_dataset(conn, cfg, "sales-2024")
    sales_copy_trash.unlink()       # what a purge leaves behind for a tombstoned document

    with pytest.raises(ValueError, match="copy purged"):
        knowledge.restore(conn, cfg, sales_id)

    assert sales_doc_trash.exists(), "the other dataset's copy was not taken for this one"


def test_restore_ignores_a_symlinked_trash_entry(env, source, tmp_path):
    """A symlink in the trash can point anywhere; a restore that followed one would copy
    a file from outside the trash into the dataset's copy path."""
    conn, cfg = env
    feed.feed(conn, cfg, source(), "prices", FeedSettings(key="item"), repo="sample-repo")
    doc_id = _doc_id(conn, cfg)
    _doc_trash, copy_trash = feed.remove_dataset(conn, cfg, "prices")
    outside = tmp_path / "outside.csv"
    outside.write_text("item,price\nother,0\n", encoding="utf-8")
    copy_trash.unlink()
    os.symlink(outside, copy_trash)

    with pytest.raises(ValueError, match="copy purged"):
        knowledge.restore(conn, cfg, doc_id)

    assert outside.read_text(encoding="utf-8").startswith("item,price\nother")


def test_failed_copy_move_leaves_the_dataset_in_the_trash(env, source, monkeypatch):
    """A restore that moved the copy after unlinking the trashed document would lose the
    document to a failed move; both files have to end up where they were."""
    conn, cfg = env
    feed.feed(conn, cfg, source(), "prices", FeedSettings(key="item"), repo="sample-repo")
    doc_id = _doc_id(conn, cfg)
    doc_trash, copy_trash = feed.remove_dataset(conn, cfg, "prices")
    real_replace = os.replace

    def flaky_replace(src, dst):
        if Path(str(src)).name.startswith("prices-"):
            raise OSError("copy move failed")
        return real_replace(src, dst)

    monkeypatch.setattr(knowledge.os, "replace", flaky_replace)
    with pytest.raises(OSError, match="copy move failed"):
        knowledge.restore(conn, cfg, doc_id)

    assert doc_trash.exists()
    assert copy_trash.exists()
    assert not _doc(cfg).exists()
    assert not _copy(cfg).exists()
    row = conn.execute("SELECT deleted_at FROM documents WHERE id = ?",
                       (doc_id,)).fetchone()
    assert row["deleted_at"]


class _FailingUpdate:
    """A connection whose tombstone UPDATE fails, so the document has already moved."""

    def __init__(self, conn):
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def execute(self, sql, *parameters):
        if sql.lstrip().upper().startswith("UPDATE DOCUMENTS SET DELETED_AT"):
            raise sqlite3.OperationalError("cannot write to the database")
        return self._conn.execute(sql, *parameters)


def test_remove_dataset_puts_the_doc_back_when_the_db_fails(env, source):
    """`remove` moves the document to the trash before it touches the database, so a
    failure there must move it back: a live row whose document sits in the trash is a
    dataset nothing can feed, restore or purge."""
    conn, cfg = env
    feed.feed(conn, cfg, source(), "prices", FeedSettings(key="item"), repo="sample-repo")
    doc_text = _doc(cfg).read_text()
    copy_bytes = _copy(cfg).read_bytes()

    with pytest.raises(sqlite3.OperationalError, match="cannot write to the database"):
        feed.remove_dataset(_FailingUpdate(conn), cfg, "prices")

    assert _doc(cfg).read_text() == doc_text
    assert _copy(cfg).read_bytes() == copy_bytes
    assert not list((cfg.knowledge_dir.parent / "trash").rglob("*__prices.md"))
    assert conn.in_transaction is False


def test_refeed_after_remove_leaves_nothing_untracked_in_the_trash(env, source):
    """A re-feed drops the tombstone of the removed dataset, so that document file must go
    with it; left behind it is a file no row points at and no purge will ever reach."""
    conn, cfg = env
    settings = FeedSettings(key="item")
    trash = cfg.knowledge_dir.parent / "trash"
    feed.feed(conn, cfg, source(), "prices", settings, repo="sample-repo")
    feed.remove_dataset(conn, cfg, "prices")

    feed.feed(conn, cfg, source(), "prices", settings, repo="sample-repo")

    assert not list(trash.rglob("*__prices.md"))
    assert knowledge.purge(conn, cfg) == 1        # the removed copy is still purgeable
    assert list(trash.rglob("*")) == []
    # An empty repo folder may stay behind; a copy file may not.
    copies = cfg.knowledge_dir.parent / "files" / ".trash"
    assert [p for p in copies.rglob("*") if p.is_file()] == []


def test_restore_dataset_waits_for_the_name_lock(env, source, tmp_path):
    """A restore moves the copy back and reindexes, so it has to wait for a feed of the
    same name rather than trade the copy path with it."""
    conn, cfg = env
    feed.feed(conn, cfg, source(), "prices", FeedSettings(key="item"), repo="sample-repo")
    doc_id = _doc_id(conn, cfg)
    feed.remove_dataset(conn, cfg, "prices")
    hold = feed._lock(cfg, "prices")
    started = threading.Event()
    done = threading.Event()
    outcome: list = []

    def run():
        worker = connect(tmp_path / "s.db")
        started.set()
        try:
            outcome.append(knowledge.restore(worker, cfg, doc_id))
        except BaseException as exc:
            outcome.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert started.wait(5) is True
        assert done.wait(0.3) is False       # blocked on the lock this test holds
    finally:
        feed._unlock(hold)
    assert done.wait(10) is True
    thread.join()
    assert isinstance(outcome[0], Path)
    assert _copy(cfg).exists()


def test_purge_removes_a_copy_trashed_before_the_cutoff(env, source):
    """A copy's age is the stamp in its name, the way a document's is `deleted_at`, so a
    cutoff has to reach one trashed long ago and leave a fresh one alone."""
    conn, cfg = env
    feed.feed(conn, cfg, source(), "prices", FeedSettings(key="item"), repo="sample-repo")
    _doc_trash, copy_trash = feed.remove_dataset(conn, cfg, "prices")
    old = copy_trash.with_name("prices-20200101T000000.csv")
    copy_trash.rename(old)
    fresh = copy_trash
    fresh.write_bytes(b"item,price\nbolt,4\n")

    assert knowledge.purge(conn, cfg, older_than_days=1) == 1

    assert not old.exists()
    assert fresh.exists()


def _hand_written(cfg, name, frontmatter, body="## A\nmarker\n", folder="sample-repo"):
    """A document written straight to disk, as another writer or an older akasha would."""
    path = cfg.knowledge_dir / folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\n{frontmatter}---\n\n{body}", encoding="utf-8")
    return path


def test_refeed_never_drops_a_removed_note_that_shares_the_dataset_name(env, source):
    """A removed note named `prices.md` leaves the same trash file name a removed dataset
    would; dropping that tombstone to make room for a feed would delete a document the
    user can still restore."""
    conn, cfg = env
    note = _hand_written(cfg, "prices.md", "id: k_note\ntitle: Prices\nkind: reference\n"
                                          "repo: sample-repo\nstatus: active\n"
                                          "created: 2026-01-01\n"
                                          "updated: 2026-01-01\n",
                        body="## A\nordinary prose marker\n")
    index_path(conn, cfg, note, "native", root=None)
    note_id = conn.execute("SELECT id FROM documents WHERE path = ?",
                           (str(note),)).fetchone()["id"]
    note_trash = knowledge.remove(conn, cfg, note_id)

    feed.feed(conn, cfg, source(), "prices", FeedSettings(key="item"), repo="sample-repo")

    row = conn.execute("SELECT path, deleted_at FROM documents WHERE id = ?",
                       (note_id,)).fetchone()
    assert row is not None and row["deleted_at"], "the note's tombstone is not a dataset's"
    assert note_trash.exists() and "ordinary prose marker" in note_trash.read_text()
    restored = knowledge.restore(conn, cfg, note_id)
    assert "ordinary prose marker" in restored.read_text()


def test_restore_treats_a_dataset_key_without_the_kind_as_a_note(env):
    """`feed` writes `kind: dataset` and the name together; a note that only carries the
    key would otherwise be restored as a dataset and demand a copy it never had."""
    conn, cfg = env
    note = _hand_written(cfg, "plain.md", "id: k_plain\ntitle: Plain\nkind: reference\n"
                                          "repo: sample-repo\ndataset: prices\n"
                                          "status: active\n"
                                          "created: 2026-01-01\nupdated: 2026-01-01\n",
                        body="## A\nplain marker\n")
    index_path(conn, cfg, note, "native", root=None)
    note_id = conn.execute("SELECT id FROM documents WHERE path = ?",
                           (str(note),)).fetchone()["id"]
    knowledge.remove(conn, cfg, note_id)

    restored = knowledge.restore(conn, cfg, note_id)

    assert "plain marker" in restored.read_text()
    assert restored.parent == cfg.knowledge_dir / "sample-repo"


def test_restore_dataset_refuses_to_overwrite_a_copy_already_in_place(env, source):
    """The copy path is the live dataset's own; a restore that replaced it would swap one
    dataset's bytes under another's document."""
    conn, cfg = env
    feed.feed(conn, cfg, source(), "prices", FeedSettings(key="item"), repo="sample-repo")
    doc_id = _doc_id(conn, cfg)
    doc_trash, copy_trash = feed.remove_dataset(conn, cfg, "prices")
    _copy(cfg).write_text("item,price\nlive,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="already"):
        knowledge.restore(conn, cfg, doc_id)

    assert _copy(cfg).read_text(encoding="utf-8") == "item,price\nlive,1\n"
    assert doc_trash.exists() and copy_trash.exists()
    row = conn.execute("SELECT deleted_at FROM documents WHERE id = ?",
                       (doc_id,)).fetchone()
    assert row["deleted_at"]


def test_restore_dataset_refreshes_the_vectors_of_its_cells(env, source, monkeypatch):
    """A removal deletes the chunks but leaves their vectors behind, and the restore's
    reindex makes fresh chunk ids, so without a vector pass every cell has none."""
    import numpy as np

    from akasha import vectors

    conn, cfg = env
    if vectors.missing_modules() or not vectors.available(conn):
        pytest.skip("the vectors extra is not usable here")

    def encode(texts):
        return np.array([[1.0] + [0.0] * 255 for _ in texts], dtype=np.float32)

    monkeypatch.setattr(vectors, "_encoder", lambda: encode)
    cfg.embeddings_provider = "model2vec"
    feed.feed(conn, cfg, source(), "prices", FeedSettings(key="item"), repo="sample-repo")
    doc_id = _doc_id(conn, cfg)
    feed.remove_dataset(conn, cfg, "prices")
    assert conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"] == 0

    knowledge.restore(conn, cfg, doc_id)

    chunk_ids = {r["id"] for r in conn.execute("SELECT id FROM chunks")}
    vector_ids = {r["chunk_id"] for r in conn.execute("SELECT chunk_id FROM vec_chunks")}
    assert chunk_ids and chunk_ids <= vector_ids


def _freeze_trash_stamp(monkeypatch, moment=None):
    """The trash stamp has one-second resolution, so a test that needs two copies in one
    second has to hold the clock still."""
    moment = moment or datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return moment

    monkeypatch.setattr(feed, "datetime", Frozen)


def test_refeed_keeps_a_hand_written_note_that_claims_the_dataset(env, source):
    """`feed` writes `kind: dataset`, the name and the repo together, and a note may carry
    the same three; dropping a tombstone on that evidence alone would delete the note's row
    and the file the user can still restore."""
    conn, cfg = env
    note = _hand_written(cfg, "prices.md",
                         "id: k_note\ntitle: Prices\nkind: dataset\nrepo: sample-repo\n"
                         "dataset: prices\nstatus: active\n"
                         "created: 2026-01-01\nupdated: 2026-01-01\n",
                         body="## A\nforged marker\n")
    index_path(conn, cfg, note, "native", root=None)
    note_id = conn.execute("SELECT id FROM documents WHERE path = ?",
                           (str(note),)).fetchone()["id"]
    note_trash = knowledge.remove(conn, cfg, note_id)

    feed.feed(conn, cfg, source(), "prices", FeedSettings(key="item"), repo="sample-repo")

    row = conn.execute("SELECT path, deleted_at FROM documents WHERE id = ?",
                       (note_id,)).fetchone()
    assert row is not None and row["deleted_at"], "the note's tombstone is not the dataset's"
    assert note_trash.exists()
    assert "forged marker" in note_trash.read_text()


def test_restore_of_a_forged_dataset_note_restores_it_as_a_plain_note(env):
    """The path that says what a document was is gone once it is in the trash, so
    frontmatter alone would restore a note that claims the dataset into `datasets/` on a
    copy it never had; the row's id says it is not a fed dataset."""
    conn, cfg = env
    note = _hand_written(cfg, "notes/prices.md",
                         "id: k_note\ntitle: Prices\nkind: dataset\nrepo: sample-repo\n"
                         "dataset: prices\nstatus: active\n"
                         "created: 2026-01-01\nupdated: 2026-01-01\n",
                         body="## A\nforged marker\n")
    index_path(conn, cfg, note, "native", root=None)
    note_id = conn.execute("SELECT id FROM documents WHERE path = ?",
                           (str(note),)).fetchone()["id"]
    knowledge.remove(conn, cfg, note_id)

    restored = knowledge.restore(conn, cfg, note_id)

    assert restored.parent == cfg.knowledge_dir / "sample-repo"
    assert "forged marker" in restored.read_text()
    assert not (cfg.knowledge_dir / "sample-repo" / "datasets").exists()


def test_restore_refuses_while_the_name_is_live_in_another_repo(env, source):
    """Units are keyed by name alone, so a restored document beside a live dataset of the
    same name would share one set of rows and the next feed of either would overwrite the
    other; the refusal comes before anything moves."""
    conn, cfg = env
    settings = FeedSettings(key="item")
    feed.feed(conn, cfg, source(), "prices", settings, repo="sample-repo")
    doc_id = _doc_id(conn, cfg)
    doc_trash, copy_trash = feed.remove_dataset(conn, cfg, "prices")
    feed.feed(conn, cfg, source(rows="item,price\nother,7\n"), "prices", settings,
              repo="other-repo")

    with pytest.raises(ValueError, match="is live at"):
        knowledge.restore(conn, cfg, doc_id)

    assert doc_trash.exists() and copy_trash.exists()
    assert not _doc(cfg).exists()
    assert not _copy(cfg).exists()
    assert [unit.fields["item"] for unit in feed_store.load_units(conn, "prices")] == ["other"]


def test_restore_refuses_while_the_name_is_live_in_the_same_repo(env, source):
    """A live document at the name's own dataset path is the same refusal as one in another
    repo: two live documents of one name would answer from one set of unit rows."""
    conn, cfg = env
    settings = FeedSettings(key="item")
    feed.feed(conn, cfg, source(), "prices", settings, repo="sample-repo")
    doc_id = _doc_id(conn, cfg)
    doc_trash, copy_trash = feed.remove_dataset(conn, cfg, "prices")
    later = _hand_written(cfg, "datasets/prices.md",
                          "id: k_later\ntitle: Dataset: prices\nkind: dataset\n"
                          "repo: sample-repo\ndataset: prices\nstatus: active\n"
                          "created: 2026-01-01\nupdated: 2026-01-01\n")

    with pytest.raises(ValueError, match="is live at"):
        knowledge.restore(conn, cfg, doc_id)

    assert later.is_file()
    assert doc_trash.exists() and copy_trash.exists()


def test_restore_with_an_invalid_frontmatter_name_opens_no_lock(env):
    """`_lock` builds `files/.locks/<name>.lock` out of the frontmatter value, so a name
    holding `../` must be refused as a dataset before that path exists; the document
    restores as the plain note it is."""
    conn, cfg = env
    note = _hand_written(cfg, "notes/prices.md",
                         "id: k_note\ntitle: Prices\nkind: dataset\nrepo: sample-repo\n"
                         "dataset: ../x\nstatus: active\n"
                         "created: 2026-01-01\nupdated: 2026-01-01\n",
                         body="## A\nforged marker\n")
    index_path(conn, cfg, note, "native", root=None)
    note_id = conn.execute("SELECT id FROM documents WHERE path = ?",
                           (str(note),)).fetchone()["id"]
    knowledge.remove(conn, cfg, note_id)

    restored = knowledge.restore(conn, cfg, note_id)

    assert "forged marker" in restored.read_text()
    assert restored.parent == cfg.knowledge_dir / "sample-repo"
    assert not (cfg.knowledge_dir.parent / "files" / ".locks").exists()


def test_same_second_extension_switch_restores_the_recorded_copy(env, source, tmp_path,
                                                                monkeypatch):
    """An xlsx-to-csv switch and the removal right after it can share one stamp, and the
    stamp alone makes the older xlsx look newest; restore has to take the copy the
    document's `file` names."""
    conn, cfg = env
    settings = FeedSettings(key="item")
    _freeze_trash_stamp(monkeypatch)
    feed.feed(conn, cfg, _xlsx(tmp_path, name="first.xlsx"), "prices", settings,
              repo="sample-repo")
    feed.feed(conn, cfg, source(), "prices", settings, repo="sample-repo")
    doc_id = _doc_id(conn, cfg)
    _doc_trash, copy_trash = feed.remove_dataset(conn, cfg, "prices")
    csv_bytes = copy_trash.read_bytes()

    restored = knowledge.restore(conn, cfg, doc_id)

    assert restored == _doc(cfg)
    assert _copy(cfg, suffix=".csv").read_bytes() == csv_bytes
    assert not _copy(cfg, suffix=".xlsx").exists()
    meta, _body = parse_frontmatter(restored.read_text())
    assert Path(meta["file"]).suffix == ".csv"


def test_same_second_trash_moves_never_overwrite(env, source, tmp_path, monkeypatch):
    """Two copies of one name trashed inside one second get the same stamp, and a move onto
    the file already there would silently drop the copy it replaced."""
    conn, cfg = env
    settings = FeedSettings(key="item")
    _freeze_trash_stamp(monkeypatch)
    feed.feed(conn, cfg, _xlsx(tmp_path, name="first.xlsx",
                               rows=(("item", "price"), ("old", 1))),
              "prices", settings, repo="sample-repo")
    feed.feed(conn, cfg, source(rows="item,price\nolder,1\n"), "prices", settings,
              repo="sample-repo")
    feed.feed(conn, cfg, _xlsx(tmp_path, name="second.xlsx",
                               rows=(("item", "price"), ("new", 2))),
              "prices", settings, repo="sample-repo")
    first = [path for path in feed.trashed_copies(cfg, "sample-repo", "prices")
             if path.suffix == ".xlsx"][0]
    first_bytes = first.read_bytes()

    feed.remove_dataset(conn, cfg, "prices")

    copies = feed.trashed_copies(cfg, "sample-repo", "prices")
    assert [path.suffix for path in copies] == [".xlsx", ".csv", ".xlsx"]
    assert first.read_bytes() == first_bytes


def test_lifecycle_with_an_index_root_covering_the_knowledge_dir(env, source):
    """A configured root that contains the knowledge dir is part of a document's derived
    id, so a tombstone check that always derives with `root=None` no longer recognizes its
    own dataset: the re-feed collides, and a restore puts the document back outside
    `datasets/` with the copy left in the trash."""
    conn, cfg = env
    cfg.index_roots = [IndexRoot(path=str(cfg.knowledge_dir.parent), source="notes")]
    settings = FeedSettings(key="item")
    feed.feed(conn, cfg, source(), "prices", settings, repo="sample-repo")
    feed.remove_dataset(conn, cfg, "prices")

    feed.feed(conn, cfg, source(rows="item,price\nbolt,9\n"), "prices", settings,
              repo="sample-repo")
    assert [unit.fields["price"] for unit in feed_store.load_units(conn, "prices")] == ["9"]

    doc_id = _doc_id(conn, cfg)
    copy_bytes = _copy(cfg).read_bytes()
    feed.remove_dataset(conn, cfg, "prices")

    restored = knowledge.restore(conn, cfg, doc_id)

    assert restored == _doc(cfg)
    assert _copy(cfg).read_bytes() == copy_bytes
    assert [unit.fields["price"] for unit in feed_store.load_units(conn, "prices")] == ["9"]


def test_restore_takes_the_recorded_extension_over_a_newer_other_copy(env, source):
    """`files/.trash` is a directory on disk, so a copy of the other extension can sit
    there with a later stamp than the document's own; restoring that one would put bytes
    under a document whose `file` names the other copy."""
    conn, cfg = env
    feed.feed(conn, cfg, source(), "prices", FeedSettings(key="item"), repo="sample-repo")
    doc_id = _doc_id(conn, cfg)
    _doc_trash, copy_trash = feed.remove_dataset(conn, cfg, "prices")
    copy_bytes = copy_trash.read_bytes()
    newer = copy_trash.with_name("prices-20990101T000000.xlsx")
    newer.write_bytes(b"not the recorded copy")

    restored = knowledge.restore(conn, cfg, doc_id)

    assert restored == _doc(cfg)
    assert _copy(cfg, suffix=".csv").read_bytes() == copy_bytes
    assert not _copy(cfg, suffix=".xlsx").exists()
    assert newer.exists()


def _tombstone_rows(conn):
    return conn.execute("SELECT key FROM meta WHERE key LIKE 'feed_tombstone:%'").fetchall()


def test_lifecycle_survives_a_root_added_between_remove_and_restore(env, source):
    """The tombstone record is what says what a trashed row was; a document's id depends on
    today's `[[index]]` roots, so a root added after the removal makes a re-derived id
    miss, and the restore then puts the dataset back as a plain note outside `datasets/`
    with its copy left in the trash."""
    conn, cfg = env
    feed.feed(conn, cfg, source(), "prices", FeedSettings(key="item"), repo="sample-repo")
    doc_id = _doc_id(conn, cfg)
    copy_bytes = _copy(cfg).read_bytes()
    feed.remove_dataset(conn, cfg, "prices")
    cfg.index_roots = [IndexRoot(path=str(cfg.knowledge_dir.parent), source="notes")]

    restored = knowledge.restore(conn, cfg, doc_id)

    assert restored == _doc(cfg)
    assert _copy(cfg).read_bytes() == copy_bytes
    assert _tombstone_rows(conn) == []


def test_lifecycle_survives_a_root_removed_between_remove_and_restore(env, source):
    """Same in the other direction: a root that covered the knowledge dir while the feed
    ran is gone again by the time the restore happens."""
    conn, cfg = env
    cfg.index_roots = [IndexRoot(path=str(cfg.knowledge_dir.parent), source="notes")]
    feed.feed(conn, cfg, source(), "prices", FeedSettings(key="item"), repo="sample-repo")
    doc_id = _doc_id(conn, cfg)
    copy_bytes = _copy(cfg).read_bytes()
    feed.remove_dataset(conn, cfg, "prices")
    cfg.index_roots = []

    restored = knowledge.restore(conn, cfg, doc_id)

    assert restored == _doc(cfg)
    assert _copy(cfg).read_bytes() == copy_bytes


def test_refeed_after_a_root_change_leaves_no_tombstone(env, source):
    """A re-feed consumes the tombstone it replaces; looking for the id its own config
    derives would leave the row and its trashed file behind after a root changed, files no
    row points at and no purge reaches."""
    conn, cfg = env
    settings = FeedSettings(key="item")
    feed.feed(conn, cfg, source(), "prices", settings, repo="sample-repo")
    feed.remove_dataset(conn, cfg, "prices")
    cfg.index_roots = [IndexRoot(path=str(cfg.knowledge_dir.parent), source="notes")]

    feed.feed(conn, cfg, source(rows="item,price\nbolt,9\n"), "prices", settings,
              repo="sample-repo")

    assert [unit.fields["price"] for unit in feed_store.load_units(conn, "prices")] == ["9"]
    assert not list((cfg.knowledge_dir.parent / "trash").rglob("*__prices.md"))
    assert _tombstone_rows(conn) == []


def test_purge_forgets_the_tombstone_record(env, source):
    """`purge` deletes the row, so the record saying what it was has to go with it: a later
    write must not read a dataset name off an id the database no longer holds."""
    conn, cfg = env
    feed.feed(conn, cfg, source(), "prices", FeedSettings(key="item"), repo="sample-repo")
    feed.remove_dataset(conn, cfg, "prices")
    assert _tombstone_rows(conn)

    knowledge.purge(conn, cfg)

    assert _tombstone_rows(conn) == []


def test_first_feed_without_key_is_refused_by_core(env, source):
    """A key is what pairs a row across imports, so the rule belongs under the per-name lock
    in core: the CLI and the MCP tool both reach it there, and neither can race it."""
    conn, cfg = env
    with pytest.raises(ValueError, match="needs settings.key"):
        feed.feed(conn, cfg, source(), "prices", FeedSettings(exclude=["price"]),
                  repo="sample-repo")
    assert not _doc(cfg).exists()
    assert not _copy(cfg).exists()
    assert feed_store.load_units(conn, "prices") == []


def test_core_feed_refuses_header_row_below_one(env, source):
    """`FeedSettings` is constructible directly, so the 1-based rule has to hold in core: a
    header_row below 1 reads row 1 as the headers and persists settings that lie about it."""
    conn, cfg = env
    with pytest.raises(ValueError, match="header_row must be >= 1"):
        feed.feed(conn, cfg, source(), "prices", FeedSettings(key="item", header_row=0),
                  repo="sample-repo")
    assert not _doc(cfg).exists()
    assert not _copy(cfg).exists()
    assert feed_store.load_units(conn, "prices") == []


class _FailingTombstoneDelete:
    """A connection whose tombstone-record delete fails; every other statement goes through."""

    def __init__(self, conn):
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def execute(self, sql, *parameters):
        if (sql.lstrip().upper().startswith("DELETE FROM META")
                and "feed_tombstone:" in str(parameters)):
            raise sqlite3.OperationalError("cannot write to the database")
        return self._conn.execute(sql, *parameters)


def test_a_restore_that_fails_after_the_update_leaves_the_row_in_the_trash(env, source):
    """The row UPDATE and the tombstone-record delete autocommit one at a time, so a record
    delete failing after the UPDATE landed leaves a live row pointing at the document the
    failure moved back to the trash: no restore, feed or purge can reach it, and the retry
    refuses because the row no longer says it is deleted."""
    conn, cfg = env
    feed.feed(conn, cfg, source(), "prices", FeedSettings(key="item"), repo="sample-repo")
    doc_id = _doc_id(conn, cfg)
    feed.remove_dataset(conn, cfg, "prices")

    with pytest.raises(sqlite3.OperationalError, match="cannot write to the database"):
        knowledge.restore(_FailingTombstoneDelete(conn), cfg, doc_id)

    row = conn.execute("SELECT deleted_at, path FROM documents WHERE id = ?",
                       (doc_id,)).fetchone()
    assert row["deleted_at"], "a failed restore left the row live"
    assert Path(row["path"]).exists(), "a failed restore lost the trashed document"
    assert _tombstone_rows(conn), "a failed restore consumed the record"
    assert conn.in_transaction is False

    assert knowledge.restore(conn, cfg, doc_id) == _doc(cfg)
    assert _tombstone_rows(conn) == []


def test_a_purge_that_fails_keeps_the_row_its_record_and_its_file(env, source):
    """`purge` deletes the row, its chunks and links and the tombstone record one statement
    at a time and unlinks the file before any of it, so a failure partway leaves a record of
    a row that no longer exists and a file nothing points at: the file goes only once the
    row's delete has committed."""
    conn, cfg = env
    feed.feed(conn, cfg, source(), "prices", FeedSettings(key="item"), repo="sample-repo")
    doc_id = _doc_id(conn, cfg)
    feed.remove_dataset(conn, cfg, "prices")
    trashed = Path(conn.execute("SELECT path FROM documents WHERE id = ?",
                                (doc_id,)).fetchone()["path"])

    with pytest.raises(sqlite3.OperationalError, match="cannot write to the database"):
        knowledge.purge(_FailingTombstoneDelete(conn), cfg)

    assert conn.execute("SELECT COUNT(*) c FROM documents WHERE id = ?",
                        (doc_id,)).fetchone()["c"] == 1
    assert conn.execute("SELECT deleted_at FROM documents WHERE id = ?",
                        (doc_id,)).fetchone()["deleted_at"]
    assert trashed.exists()
    assert _tombstone_rows(conn)
    assert conn.in_transaction is False

    assert knowledge.purge(conn, cfg) >= 1
    assert not trashed.exists()
    assert _tombstone_rows(conn) == []
