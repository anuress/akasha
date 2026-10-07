import json

import pytest

from akasha.db import connect
from akasha.events import emit, recent


@pytest.fixture
def conn(tmp_path):
    return connect(tmp_path / "s.db")


def test_emit_records_kind_and_payload(conn):
    emit(conn, "knowledge.searched", count=3, repo="sample-repo")
    row = recent(conn)[0]
    assert row["kind"] == "knowledge.searched"
    assert json.loads(row["payload"])["count"] == 3


def test_recent_is_newest_first_and_limited(conn):
    for i in range(5):
        emit(conn, "tick", n=i)
    rows = recent(conn, limit=3)
    assert len(rows) == 3
    assert json.loads(rows[0]["payload"])["n"] == 4


def test_recent_filters_by_kind(conn):
    emit(conn, "knowledge.searched")
    emit(conn, "tool.failed")
    assert len(recent(conn, kind="tool.failed")) == 1


def test_payload_never_stores_large_bodies(conn):
    """events.payload holds identifiers and outcomes, never document text."""
    emit(conn, "knowledge.written", title="x" * 5000)
    stored = recent(conn)[0]["payload"]
    assert len(stored) < 600


def test_payload_fields_are_trimmed_exactly_at_the_boundary(conn):
    """An identifier up to and including MAX_FIELD characters passes untouched, one over
    is cut. The cut marks the spot, so a reader can see a field was trimmed."""
    from akasha.events import MAX_FIELD

    emit(conn, "k", at_boundary="a" * MAX_FIELD, one_over="b" * (MAX_FIELD + 1),
         number=7, flag=True)
    payload = json.loads(recent(conn)[0]["payload"])
    assert payload["at_boundary"] == "a" * MAX_FIELD
    assert payload["one_over"] == "b" * MAX_FIELD + "…"
    assert payload["number"] == 7 and payload["flag"] is True, \
        "only strings are bounded"
