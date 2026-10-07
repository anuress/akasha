"""Concurrent-writer safety: an update must refuse, never clobber, when the file on
disk changed behind akasha's back. The guard is in the update path only; append stays atomic and
order-independent by design, so it is not covered here."""
import time

import pytest

from akasha.config import load_config
from akasha.db import connect
from akasha.knowledge import DriftRefusal, update, write


@pytest.fixture
def env(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")
    cfg.knowledge_dir = tmp_path / "knowledge"
    cfg.knowledge_dir.mkdir()
    cfg.index_roots = []
    return connect(tmp_path / "s.db"), cfg


def _path(conn, doc_id):
    from pathlib import Path

    return Path(conn.execute("SELECT path FROM documents WHERE id=?", (doc_id,)).fetchone()["path"])


def test_update_refuses_when_file_drifted_behind_the_indexers_back(env):
    """A sibling agent wrote the file after the index; an update must not overwrite it.

    The refusal leaves the on-disk version untouched, copies it to a sibling backup,
    and the message names the backup and orders the only correct recovery: re-read the
    file and re-apply the change on top of the other writer's version."""
    conn, cfg = env
    doc_id = write(conn, cfg, "T", "## A\noriginal\n", repo="r")
    path = _path(conn, doc_id)
    indexed_mtime = conn.execute("SELECT mtime FROM documents WHERE id=?", (doc_id,)).fetchone()["mtime"]

    time.sleep(0.02)                     # a coarse- or fine-grained fs must see a new mtime
    path.write_text("## A\nsomeone else's edit\n")
    assert path.stat().st_mtime != indexed_mtime

    with pytest.raises(DriftRefusal) as exc_info:
        update(conn, cfg, doc_id, body="## A\nthe overwrite attempt\n")

    assert "re-read" in str(exc_info.value).lower()
    assert path.read_text() == "## A\nsomeone else's edit\n", \
        "the on-disk version must survive; the update must not replace it"

    backups = list((path.parent).glob(path.name + ".drift*"))
    assert len(backups) == 1, "a sibling backup of the on-disk version must exist"
    assert backups[0].read_text() == "## A\nsomeone else's edit\n"
    assert backups[0].name in str(exc_info.value), "the refusal must name the backup"


def test_own_write_then_update_does_not_trip_the_guard(env):
    """akasha's own write-then-reindex stores the mtime it wrote, so its own next update
    must pass the drift check — the guard is for other writers, not for akasha itself."""
    conn, cfg = env
    doc_id = write(conn, cfg, "T", "## A\nfirst\n", repo="r")
    path = _path(conn, doc_id)

    update(conn, cfg, doc_id, body="## A\nsecond\n")
    assert "## A\nsecond\n" in path.read_text()

    # The update's own reindex recorded the fresh mtime; a second update is untouched.
    update(conn, cfg, doc_id, body="## A\nthird\n")
    assert "## A\nthird\n" in path.read_text()