"""Editing a document by naming the part that changes.

Correcting one sentence should not cost the whole body: the caller is an agent, and
context is the scarce resource. A short unique substring is what it already has in front
of it.
"""
import pytest

from akasha.config import load_config
from akasha.db import connect
from akasha.knowledge import update, write


@pytest.fixture
def env(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")
    cfg.knowledge_dir = tmp_path / "knowledge"
    cfg.knowledge_dir.mkdir()
    cfg.index_roots = []
    return connect(tmp_path / "s.db"), cfg


def _body(cfg, doc_id, conn):
    from pathlib import Path

    from akasha.markdown import parse_frontmatter
    path = Path(conn.execute("SELECT path FROM documents WHERE id=?",
                             (doc_id,)).fetchone()["path"])
    return parse_frontmatter(path.read_text())[1]


def test_a_unique_substring_is_replaced_and_the_rest_survives(env):
    conn, cfg = env
    doc_id = write(conn, cfg, "Finding", "## Body\nthe cap is 5\nand the rest stands\n",
                   repo="r")
    update(conn, cfg, doc_id, match="the cap is 5", replacement="the cap is 12")
    body = _body(cfg, doc_id, conn)
    assert "the cap is 12" in body
    assert "and the rest stands" in body


def test_a_substring_that_matches_nothing_fails_loudly(env):
    """Silence here is the dangerous outcome: the caller believes it corrected a
    document that still says the wrong thing."""
    conn, cfg = env
    doc_id = write(conn, cfg, "Finding", "## Body\nthe cap is 5\n", repo="r")
    with pytest.raises(ValueError, match="0"):
        update(conn, cfg, doc_id, match="the cap is 9", replacement="the cap is 12")
    assert "the cap is 5" in _body(cfg, doc_id, conn)


def test_an_ambiguous_substring_is_refused_with_the_count(env):
    """Two matches means the caller is not describing what it thinks it is. Replacing
    the first would be a guess, and replacing both would be a different edit."""
    conn, cfg = env
    doc_id = write(conn, cfg, "Finding", "## Body\nthe cap is 5\nthe cap is 5 again\n",
                   repo="r")
    with pytest.raises(ValueError, match="2"):
        update(conn, cfg, doc_id, match="the cap is 5", replacement="the cap is 12")
    assert _body(cfg, doc_id, conn).count("the cap is 5") == 2


def test_a_replacement_without_a_match_is_refused(env):
    """Half the pair is not an edit anyone can act on, and defaulting the other half
    would silently mean something."""
    conn, cfg = env
    doc_id = write(conn, cfg, "Finding", "## Body\nthe cap is 5\n", repo="r")
    with pytest.raises(ValueError):
        update(conn, cfg, doc_id, replacement="the cap is 12")


def test_a_whole_body_and_a_substring_edit_cannot_be_asked_for_at_once(env):
    """They disagree about what the document should end up saying. Picking one would be
    picking for the caller."""
    conn, cfg = env
    doc_id = write(conn, cfg, "Finding", "## Body\nthe cap is 5\n", repo="r")
    with pytest.raises(ValueError):
        update(conn, cfg, doc_id, body="## Body\nrewritten\n",
               match="the cap is 5", replacement="the cap is 12")


def test_the_path_form_still_works_untouched(env):
    """Existing callers pass a whole body. The substring form is an addition, not a
    replacement of the interface."""
    conn, cfg = env
    doc_id = write(conn, cfg, "Finding", "## Body\nthe cap is 5\n", repo="r")
    update(conn, cfg, doc_id, body="## Body\nrewritten whole\n")
    assert "rewritten whole" in _body(cfg, doc_id, conn)
