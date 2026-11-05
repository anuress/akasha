"""Feeding a table export into a dataset document, as one crash-safe operation.

Every test writes synthetic rows to a temporary directory and inspects the result cell by
cell, so the suite needs no real export and no real knowledge base.
"""
from __future__ import annotations

import hashlib

import pytest

from akasha import feed, feed_store, knowledge
from akasha.config import load_config
from akasha.db import connect
from akasha.markdown import parse_frontmatter
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

    with pytest.raises(ValueError, match="unknown column 'item'"):
        feed.feed(conn, cfg, source(rows="sku,price\nbolt,4\n"), "prices", None,
                  repo="sample-repo")

    assert _copy(cfg).read_bytes() == copy_before
    assert _doc(cfg).read_text() == doc_before
    assert feed_store.load_units(conn, "prices") == units_before
    assert feed_store.stored_state(conn, "prices") == state_before


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
    assert list((cfg.knowledge_dir.parent / "files").rglob("*.tmp")) == []
    assert feed_store.load_units(conn, "prices") == []


def test_feed_switching_extension_trashes_old_copy(env, source, tmp_path):
    """The copy's extension follows the incoming file, so the previous copy has no name to
    live at any more and goes to the trash instead of being overwritten."""
    openpyxl = pytest.importorskip("openpyxl")
    conn, cfg = env
    settings = FeedSettings(key="item")
    feed.feed(conn, cfg, source(), "prices", settings, repo="sample-repo")
    old_bytes = _copy(cfg).read_bytes()

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["item", "price"])
    sheet.append(["bolt", 4])
    sheet.append(["nut", 2])
    book = tmp_path / "prices.xlsx"
    workbook.save(book)

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
    openpyxl = pytest.importorskip("openpyxl")
    conn, cfg = env
    settings = FeedSettings(key="item")
    feed.feed(conn, cfg, source(), "prices", settings, repo="sample-repo")
    copy_before = _copy(cfg).read_bytes()
    doc_before = _doc(cfg).read_text()

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["item", "price"])
    sheet.append(["bolt", 4])
    book = tmp_path / "prices.xlsx"
    workbook.save(book)

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

    result = feed.feed(
        conn, cfg, source(rows="note\nAuthorization: Bearer abcdef1234567890abcdef\n"),
        "prices", FeedSettings(), repo="sample-repo")

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
