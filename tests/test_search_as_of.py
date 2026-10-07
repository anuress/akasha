"""As-of search: a superseded document stays answerable for the dates before it fell.

Supersession archives the predecessor so current recall stays true, but the database is
a derived index of the files: without a recorded interval end there is no way to answer
what was believed on date X. `invalid_at` is the interval's end, written into the
document's frontmatter like `status`; `search(as_of=)` answers from the interval instead
of from current status.

The supersession date is known in every test: a real write and supersede stamp today.
"""
from datetime import date, timedelta
from pathlib import Path

import pytest

from akasha.config import load_config
from akasha.db import connect
from akasha.knowledge import archive, write
from akasha.markdown import parse_frontmatter, render
from akasha.search import search


@pytest.fixture
def env(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")
    cfg.knowledge_dir = tmp_path / "knowledge"
    cfg.knowledge_dir.mkdir()
    cfg.index_roots = []
    return connect(tmp_path / "s.db"), cfg


def _old_policy(conn, cfg):
    return write(conn, cfg, "Old TTL policy",
                 "## TTL\nthe cache ttl is sixty seconds, glacialflow style\n", repo="r")


def _supersede(conn, cfg, old):
    return write(conn, cfg, "New TTL policy",
                 "## TTL\nthe cache ttl is ten seconds now\n", repo="r", supersedes=[old])


@pytest.fixture
def superseded(env):
    """One document superseded today, one never superseded, both written today.

    Each carries a token that appears in no other document, so a search for it is a
    question only that document can answer.
    """
    conn, cfg = env
    old = _old_policy(conn, cfg)
    keeper = write(conn, cfg, "Standing policy",
                   "## Standing\nhedgehog checks run nightly\n", repo="r")
    _supersede(conn, cfg, old)
    before = (date.today() - timedelta(days=1)).isoformat()
    after = (date.today() + timedelta(days=1)).isoformat()
    return conn, cfg, old, keeper, before, after


def test_superseded_document_absent_from_current_search(superseded):
    """(a) The document superseded on a known date is gone from default search."""
    conn, _, old, _, _, _ = superseded
    hits = search(conn, "glacialflow", all_repos=True)
    assert old not in {h.document_id for h in hits}


def test_superseded_document_present_in_as_of_search_before_supersession(superseded):
    """(b) One day before the supersession the document was believed; search says so."""
    conn, _, old, _, before, _ = superseded
    hits = search(conn, "glacialflow", all_repos=True, as_of=before)
    assert old in {h.document_id for h in hits}, \
        "as-of search before the supersession must still answer from the old document"


def test_superseded_document_absent_in_as_of_search_after_supersession(superseded):
    """(c) One day after, the interval has closed and the document is out again."""
    conn, _, old, _, _, after = superseded
    hits = search(conn, "glacialflow", all_repos=True, as_of=after)
    assert old not in {h.document_id for h in hits}


def test_never_superseded_document_unaffected_on_all_three_paths(superseded):
    """(d) A document with no invalid_at behaves exactly as it does now, everywhere."""
    conn, _, _, keeper, before, after = superseded
    for as_of in (None, before, after):
        hits = search(conn, "hedgehog", all_repos=True, as_of=as_of)
        assert keeper in {h.document_id for h in hits}, \
            f"never-superseded document lost on as_of={as_of!r}"


def test_interval_end_lives_in_the_file_and_a_reindex_reads_it_back(env):
    """The row is a derived index: the interval end must live in the document file, like
    status, or one reindex undoes the supersession. The file is edited (not the row) and
    then reindexed, so what search answers by is whatever the frontmatter produced."""
    import os
    import time

    from akasha.index import index_path

    conn, cfg = env
    old = _old_policy(conn, cfg)
    _supersede(conn, cfg, old)

    path = Path(conn.execute("SELECT path FROM documents WHERE id=?", (old,))
                .fetchone()["path"])
    meta, body = parse_frontmatter(path.read_text())
    known = (date.today() - timedelta(days=30)).isoformat()
    meta["invalid_at"] = known
    path.write_text(render(meta, body))
    os.utime(path, (time.time() + 10, time.time() + 10))
    index_path(conn, cfg, path, "native", root=None)

    stored = conn.execute("SELECT invalid_at FROM documents WHERE id=?", (old,)).fetchone()
    assert stored["invalid_at"] == known, \
        "index did not read invalid_at from the file's frontmatter"

    recent = (date.today() - timedelta(days=1)).isoformat()
    distant = (date.today() - timedelta(days=60)).isoformat()
    recent_hits = {h.document_id for h in
                   search(conn, "glacialflow", all_repos=True, as_of=recent)}
    distant_hits = {h.document_id for h in
                    search(conn, "glacialflow", all_repos=True, as_of=distant)}
    assert old not in recent_hits, \
        "the reindexed interval end did not reach search: invalid after the date it fell"
    assert old in distant_hits, \
        "the reindexed interval did not reach search: still valid before the date it fell"


def test_as_of_reaches_a_document_only_the_dense_side_found(env, monkeypatch):
    """The lexical filter and the dense-side `_hit_for_chunk` must give one answer to one
    flag. The query matches no chunk lexically, so vectors are the only way in."""
    from akasha import vectors

    conn, cfg = env
    old = _old_policy(conn, cfg)
    _supersede(conn, cfg, old)
    chunk = conn.execute(
        "SELECT c.id FROM chunks c WHERE c.document_id = ?", (old,)).fetchone()["id"]
    monkeypatch.setattr(vectors, "enabled", lambda cfg: True)
    monkeypatch.setattr(vectors, "nearest", lambda *a, **k: [chunk])

    before = (date.today() - timedelta(days=1)).isoformat()
    after = (date.today() + timedelta(days=1)).isoformat()
    asked = search(conn, "unmatchableqqq", all_repos=True, cfg=cfg, as_of=before)
    assert [h.id for h in asked] == [chunk], \
        "as-of view must reach a document only vectors found"
    assert search(conn, "unmatchableqqq", all_repos=True, cfg=cfg, as_of=after) == []
    assert search(conn, "unmatchableqqq", all_repos=True, cfg=cfg) == [], \
        "the default path still has to mean something when as_of is not passed"


def test_as_of_rejects_a_date_that_is_not_iso(env):
    conn, _ = env
    with pytest.raises(ValueError, match="ISO"):
        search(conn, "anything", all_repos=True, as_of="next tuesday")

def test_a_plain_archive_closes_the_interval_the_same_way_supersession_does(env):
    """Archive is the other way out of search, and it must record when, or as-of search
    could not tell a document archived last week from one never written."""
    conn, cfg = env
    old = _old_policy(conn, cfg)
    archive(conn, cfg, old)
    row = conn.execute("SELECT status, invalid_at FROM documents WHERE id=?", (old,)).fetchone()
    assert (row["status"], row["invalid_at"]) == ("archived", date.today().isoformat())

    before = (date.today() - timedelta(days=1)).isoformat()
    after = (date.today() + timedelta(days=1)).isoformat()
    assert old not in {h.document_id for h in search(conn, "glacialflow", all_repos=True)}
    assert old in {h.document_id for h in
                   search(conn, "glacialflow", all_repos=True, as_of=before)}
    assert old not in {h.document_id for h in
                       search(conn, "glacialflow", all_repos=True, as_of=after)}


def test_supersession_writes_the_interval_end_into_the_file(env):
    """The row is derived from the file, so the file must carry the end of the interval."""
    conn, cfg = env
    old = _old_policy(conn, cfg)
    _supersede(conn, cfg, old)
    path = Path(conn.execute("SELECT path FROM documents WHERE id=?", (old,)).fetchone()["path"])
    meta, _ = parse_frontmatter(path.read_text())
    assert meta["status"] == "archived"
    assert str(meta["invalid_at"]) == date.today().isoformat()
