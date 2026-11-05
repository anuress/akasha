"""The feed tables and the pure diff between two lists of units.

The store is what a later `feed add` writes and what a rebuild reads back; the diff is
what it tells the caller changed. Both are compared to the units a splitter produces, so
synthetic units are the fixtures and no file is read here.
"""
from __future__ import annotations

import numpy as np
import pytest

from akasha import feed_store
from akasha.db import SCHEMA_VERSION, connect
from akasha.splitters import Unit


def _conn(tmp_path):
    return connect(tmp_path / "s.db")


def _unit(locator, key=None, label=None, **fields):
    return Unit(locator=locator, key=key, label=label or f"sample {locator}", fields=fields)


def _add_chunk(conn, chunk_id, body):
    conn.execute(
        "INSERT INTO chunks (id, document_id, title, heading, body, ord)"
        " VALUES (?, ?, ?, ?, ?, ?)", (chunk_id, "d_sample", "sample", "", body, 0))


def _add_feed_chunk(conn, chunk_id, dataset, locator, field):
    conn.execute(
        "INSERT INTO feed_chunks (chunk_id, dataset, locator, field) VALUES (?, ?, ?, ?)",
        (chunk_id, dataset, locator, field))


def test_feed_tables_exist_without_schema_version_bump(tmp_path):
    """A bump makes every existing akasha.db unusable, and the runs and events in it
    cannot be rebuilt from markdown, so the feed tables must be additive only."""
    assert SCHEMA_VERSION == 1
    tables = {r["name"] for r in _conn(tmp_path).execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"feed_units", "feed_chunks"} <= tables


def test_replace_then_load_round_trips_units_in_order(tmp_path):
    """Row order is what a rebuild compares and what the caller lists, so the round trip
    must preserve it, and the JSON column must not mangle a non-ASCII cell."""
    conn = _conn(tmp_path)
    units = [
        _unit("row:2", key="b", name="Bäckerei", city="Köln"),
        _unit("row:3", key="a", name="Anfang"),
    ]
    feed_store.replace_units(conn, "prices", units, "fresh")

    assert feed_store.load_units(conn, "prices") == units


def test_replace_units_replaces_previous_rows_and_state(tmp_path):
    """A second import of the same file must not leave the previous import's rows behind
    or report its state, or the diff would compare against a mix of two exports."""
    conn = _conn(tmp_path)
    feed_store.replace_units(conn, "prices", [_unit("row:2", a="1")], "fresh")
    feed_store.replace_units(conn, "prices", [_unit("row:5", b="2")], "stale")

    assert [unit.locator for unit in feed_store.load_units(conn, "prices")] == ["row:5"]
    assert feed_store.stored_state(conn, "prices") == "stale"


def test_diff_matches_by_key_when_rows_move():
    """A re-export that only reorders rows is not a change; pairing by locator would
    report every row as removed and re-added."""
    old = [_unit("row:2", key="7", label="Werk", name="Werk"),
           _unit("row:3", key="8", label="Haus", name="Haus")]
    new = [_unit("row:9", key="8", label="Haus", name="Haus"),
           _unit("row:4", key="7", label="Werk", name="Werk")]

    assert feed_store.diff(old, new).empty()


def test_diff_falls_back_to_locator_without_key():
    """A row without a key can only be recognised by where it sits, so editing it is a
    change, not a removal plus an addition."""
    result = feed_store.diff([_unit("row:2", name="Werk")], [_unit("row:2", name="Werkstatt")])

    assert not result.added and not result.removed
    assert [(change.identity, change.fields) for change in result.changed] == [
        ("row:2", ["name"])]


def test_diff_key_becoming_duplicate_pairs_by_locator():
    """The splitter drops a key that became a duplicate, so the same row arrives with a
    None key; it must pair by locator rather than read as removed plus added."""
    result = feed_store.diff(
        [_unit("row:3", key="7", name="Werk")], [_unit("row:3", key=None, name="Werk")])

    assert not result.added and not result.removed
    assert [change.identity for change in result.changed] == []


def test_diff_reports_changed_fields_label_and_capped_lines():
    """The lines go into the answer an agent reads, so a long or multi-line cell would
    flood its context; the cap is the only thing that bounds it, and 0 must mean no cap
    because `feed` asks for the full diff on demand."""
    old = [_unit("row:2", key="7", label="old label",
                 note="line one\nline two\nline three", size="10")]
    new = [_unit("row:2", key="7", label="new label",
                 note="line one\nline TWO\nline three", mode="x")]

    capped = feed_store.diff(old, new, line_cap=2).changed[0]
    assert capped.label == "new label"
    # `size` is gone from the new import and `mode` is new; a field on one side only is
    # a changed field, or a dropped column would vanish from the report.
    assert capped.fields == ["mode", "note", "size"]
    assert capped.lines == ["mode: +x", "note: -line two"]

    uncapped = feed_store.diff(old, new, line_cap=0).changed[0]
    assert uncapped.lines == [
        "mode: +x", "note: -line two", "note: +line TWO", "size: -10"]


def test_unit_chunk_id_changes_only_with_value():
    """The id is what carries a stored vector across a re-import, so it must depend on
    the cell's identity and value and on nothing else."""
    base = feed_store.unit_chunk_id("prices", "row:2", "name", "Werk")

    assert base.startswith("c_") and len(base) == 14
    assert base == feed_store.unit_chunk_id("prices", "row:2", "name", "Werk")
    assert base != feed_store.unit_chunk_id("prices", "row:2", "name", "Werk 2")
    assert base != feed_store.unit_chunk_id("prices", "row:3", "name", "Werk")
    assert base != feed_store.unit_chunk_id("prices", "row:2", "city", "Werk")
    assert base != feed_store.unit_chunk_id("houses", "row:2", "name", "Werk")


def test_drop_dataset_removes_units_chunks_vectors_and_meta(tmp_path):
    """An orphan vector spends a slot in nearest()'s k and returns nothing, and a chunk
    left behind stays searchable, so the drop has to reach all five tables."""
    from akasha import vectors

    conn = _conn(tmp_path)
    if not vectors.available(conn):
        pytest.skip("sqlite-vec not loadable here")

    feed_store.replace_units(conn, "prices", [_unit("row:2", key="7", name="Werk")], "fresh")
    chunk_id = feed_store.unit_chunk_id("prices", "row:2", "name", "Werk")
    _add_chunk(conn, chunk_id, "Werk")
    _add_feed_chunk(conn, chunk_id, "prices", "row:2", "name")
    vectors.ensure_table(conn, 8)
    conn.execute(
        "INSERT INTO vec_chunks (chunk_id, repo, embedding) VALUES (?, ?, ?)",
        (chunk_id, "", np.zeros(8, dtype=np.float32).tobytes()))

    feed_store.drop_dataset(conn, "prices")

    assert feed_store.load_units(conn, "prices") == []
    assert feed_store.stored_state(conn, "prices") is None
    assert conn.execute("SELECT COUNT(*) c FROM feed_chunks").fetchone()["c"] == 0
    assert conn.execute(
        "SELECT COUNT(*) c FROM chunks WHERE id = ?", (chunk_id,)).fetchone()["c"] == 0
    assert conn.execute("SELECT COUNT(*) c FROM vec_chunks").fetchone()["c"] == 0


def test_drop_dataset_leaves_other_datasets_alone(tmp_path):
    """Dropping one dataset must not take another's units, chunks or state with it;
    otherwise `feed add` of one file silently deletes the search of another."""
    conn = _conn(tmp_path)
    feed_store.replace_units(conn, "prices", [_unit("row:2", name="Werk")], "fresh")
    feed_store.replace_units(conn, "houses", [_unit("row:2", name="Haus")], "fresh")
    kept = feed_store.unit_chunk_id("houses", "row:2", "name", "Haus")
    _add_chunk(conn, kept, "Haus")
    _add_feed_chunk(conn, kept, "houses", "row:2", "name")
    doomed = feed_store.unit_chunk_id("prices", "row:2", "name", "Werk")
    _add_chunk(conn, doomed, "Werk")
    _add_feed_chunk(conn, doomed, "prices", "row:2", "name")

    feed_store.drop_dataset(conn, "prices")

    assert [unit.label for unit in feed_store.load_units(conn, "houses")] == ["sample row:2"]
    assert feed_store.stored_state(conn, "houses") == "fresh"
    assert [r["chunk_id"] for r in conn.execute("SELECT chunk_id FROM feed_chunks")] == [kept]
    assert conn.execute(
        "SELECT COUNT(*) c FROM chunks WHERE id = ?", (kept,)).fetchone()["c"] == 1
