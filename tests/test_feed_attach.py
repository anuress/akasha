"""Indexing a dataset document attaches one search chunk per cell of its units.

The dataset doc and its copied file are written by hand here, through the frontmatter
contract `feed` will use, so these tests pin the index-time half of `akasha feed`
without depending on the command that produces the doc.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import pytest

from akasha import feed, feed_store, splitters
from akasha.config import load_config
from akasha.db import connect
from akasha.events import recent
from akasha.index import index_all
from akasha.markdown import parse_frontmatter
from akasha.splitters import FeedSettings, Unit

ROWS = "item,price\nbolt,4\nnut,2\n"

DATASET_DOC = """---
title: Dataset: {name}
kind: dataset
feature: {name}
repo: {repo}
dataset: {name}
file: {file}
sha256: {sha}
settings_sha256: {settings_sha}
settings: {settings_json}
---

# Dataset: {name}

Copied from a table export.
"""


@pytest.fixture
def env(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")
    cfg.knowledge_dir = tmp_path / "knowledge"
    cfg.knowledge_dir.mkdir()
    # `dataset` is not a built-in kind, so without this the doc indexes as reference
    # and the attach hook never sees it.
    cfg.custom_kinds = ["dataset"]
    return connect(tmp_path / "s.db"), cfg


@pytest.fixture
def make_dataset(env):
    _, cfg = env

    def build(name="prices", rows=ROWS, settings=None, sha=None, repo="sample-repo"):
        settings = settings or FeedSettings()
        copy = cfg.knowledge_dir.parent / "files" / repo / f"{name}.csv"
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_text(rows, encoding="utf-8")
        doc = cfg.knowledge_dir / repo / "datasets" / f"{name}.md"
        doc.parent.mkdir(parents=True, exist_ok=True)
        doc.write_text(
            DATASET_DOC.format(
                name=name, repo=repo, file=copy,
                sha=sha or hashlib.sha256(copy.read_bytes()).hexdigest(),
                settings_sha=settings.sha256(), settings_json=settings.to_json()),
            encoding="utf-8")
        return doc, copy

    return build


def _doc_id(conn, name="prices"):
    row = conn.execute("SELECT id FROM documents WHERE kind = 'dataset' AND path LIKE ?",
                       (f"%{name}.md",)).fetchone()
    return row["id"] if row else None


def _unit_chunks(conn, doc_id):
    return conn.execute(
        "SELECT c.id, c.heading, c.body, c.ord FROM chunks c"
        " JOIN feed_chunks f ON f.chunk_id = c.id WHERE c.document_id = ? ORDER BY c.ord",
        (doc_id,)).fetchall()


def _boom(*args, **kwargs):
    raise AssertionError("the copy was split again")


def test_indexing_a_dataset_doc_splits_the_copy_and_adds_one_chunk_per_cell(
        env, make_dataset):
    """A dataset is only searchable cell by cell, so indexing its doc has to turn the
    copied file into one chunk per non-empty cell and record which cell each came from."""
    conn, cfg = env
    _doc, copy = make_dataset()
    index_all(conn, cfg)

    doc_id = _doc_id(conn)
    chunks = _unit_chunks(conn, doc_id)
    assert [(c["heading"], c["body"]) for c in chunks] == [
        ("row:2 · item", "prices row:2 · item: bolt"),
        ("row:2 · price", "prices row:2 · price: 4"),
        ("row:3 · item", "prices row:3 · item: nut"),
        ("row:3 · price", "prices row:3 · price: 2"),
    ]
    # In unit and field order, after the doc's own markdown chunk.
    assert [c["ord"] for c in chunks] == [1, 2, 3, 4]
    assert [tuple(r) for r in conn.execute(
        "SELECT dataset, locator, field FROM feed_chunks ORDER BY locator, field")] == [
        ("prices", "row:2", "item"),
        ("prices", "row:2", "price"),
        ("prices", "row:3", "item"),
        ("prices", "row:3", "price"),
    ]
    assert [u.locator for u in feed_store.load_units(conn, "prices")] == ["row:2", "row:3"]
    sha = hashlib.sha256(copy.read_bytes()).hexdigest()
    assert feed_store.stored_state(conn, "prices") == f"{sha}:{FeedSettings().sha256()}"


def test_reindex_with_current_state_does_not_resplit(env, make_dataset, monkeypatch):
    """The copy is the expensive part of an index run; when neither it nor its settings
    changed, the stored units are what the chunks are rebuilt from."""
    conn, cfg = env
    make_dataset()
    index_all(conn, cfg)
    doc_id = _doc_id(conn)
    monkeypatch.setattr(splitters, "split", _boom)

    stats = index_all(conn, cfg, force=True)

    assert stats.dataset_errors == []
    assert len(_unit_chunks(conn, doc_id)) == 4


def test_attach_uses_given_units_without_splitting(env, make_dataset, monkeypatch):
    """`feed add` holds the units it just diffed; splitting the copy again could pick up a
    file newer than the units whose change report was already written."""
    conn, cfg = env
    doc, _copy = make_dataset()
    index_all(conn, cfg)
    doc_id = _doc_id(conn)
    meta, _body = parse_frontmatter(doc.read_text())
    conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
    monkeypatch.setattr(splitters, "split", _boom)

    given = [Unit(locator="row:9", key=None, label="given row:9", fields={"note": "v"})]
    assert feed.attach(conn, cfg, doc_id, meta, units=given) == 1
    assert [(c["heading"], c["body"]) for c in _unit_chunks(conn, doc_id)] == [
        ("row:9 · note", "given row:9 · note: v")]


def test_unchanged_cells_keep_chunk_ids_across_reindex(env, make_dataset):
    """Chunk ids are content-derived so a reindex of an untouched copy keeps each cell's
    vector instead of re-embedding the whole dataset."""
    conn, cfg = env
    make_dataset()
    index_all(conn, cfg)
    doc_id = _doc_id(conn)
    first = sorted(c["id"] for c in _unit_chunks(conn, doc_id))

    index_all(conn, cfg, force=True)

    assert sorted(c["id"] for c in _unit_chunks(conn, doc_id)) == first
    assert len(first) == 4


def test_secret_shaped_cell_is_redacted_in_units_and_chunks(env, make_dataset):
    """A cell holds the same secrets a paragraph does, and the stored units are reused by
    every later reindex, so redaction has to happen once, before either is written."""
    conn, cfg = env
    make_dataset(rows="note\nAuthorization: Bearer abcdef1234567890abcdef\n")
    index_all(conn, cfg)

    doc_id = _doc_id(conn)
    stored = feed_store.load_units(conn, "prices")
    assert stored[0].fields["note"] == "[REDACTED:bearer]"
    body = _unit_chunks(conn, doc_id)[0]["body"]
    assert body == "prices row:2 · note: [REDACTED:bearer]"
    assert "abcdef1234567890abcdef" not in body


def test_missing_copy_keeps_last_units_and_reports_error(env, make_dataset):
    """A copy deleted behind akasha's back must not empty a dataset from search, and the
    run must say so rather than leaving a silently stale dataset behind."""
    conn, cfg = env
    make_dataset()
    index_all(conn, cfg)
    doc_id = _doc_id(conn)

    doc, copy = make_dataset(sha="0" * 64)
    copy.unlink()
    os.utime(doc, (time.time() + 10, time.time() + 10))
    stats = index_all(conn, cfg)

    assert len(_unit_chunks(conn, doc_id)) == 4
    assert [name for name, _message in stats.dataset_errors] == ["prices"]
    assert "prices.csv" in stats.dataset_errors[0][1]
    events = recent(conn, kind="feed.attach_failed")
    assert [json.loads(e["payload"])["dataset"] for e in events] == ["prices"]


def test_index_all_continues_past_broken_dataset(env, make_dataset):
    """One unreadable dataset is one dataset; the rest of the corpus still indexes."""
    conn, cfg = env
    _doc, broken_copy = make_dataset(name="broken", sha="0" * 64)
    broken_copy.unlink()
    notes = cfg.knowledge_dir / "sample-repo" / "notes.md"
    notes.write_text("## Shelving\nLoanService new\n", encoding="utf-8")

    stats = index_all(conn, cfg)

    assert stats.indexed == 2
    assert [name for name, _message in stats.dataset_errors] == ["broken"]
    assert conn.execute(
        "SELECT COUNT(*) c FROM chunks c JOIN documents d ON d.id = c.document_id"
        " WHERE d.path = ?", (str(notes),)).fetchone()["c"] == 1


def test_force_index_rebuilds_units_after_they_are_dropped(env, make_dataset):
    """`--force` is the escape hatch when the units are gone but the doc's stat says it is
    unchanged, so it has to re-split and re-attach as well."""
    conn, cfg = env
    make_dataset()
    index_all(conn, cfg)
    doc_id = _doc_id(conn)
    feed_store.drop_dataset(conn, "prices")
    assert feed_store.load_units(conn, "prices") == []

    index_all(conn, cfg, force=True)

    assert [u.locator for u in feed_store.load_units(conn, "prices")] == ["row:2", "row:3"]
    assert len(_unit_chunks(conn, doc_id)) == 4


def test_non_dataset_docs_unaffected(env):
    """Only a dataset doc owns feed rows; an ordinary note must not grow a feed_chunks row
    that a later `feed add` would delete or report as changed."""
    conn, cfg = env
    notes = cfg.knowledge_dir / "sample-repo" / "notes.md"
    notes.parent.mkdir(parents=True)
    notes.write_text("## Shelving\nLoanService new\n", encoding="utf-8")

    stats = index_all(conn, cfg)

    assert stats.indexed == 1
    assert stats.dataset_errors == []
    assert conn.execute("SELECT COUNT(*) c FROM feed_chunks").fetchone()["c"] == 0
