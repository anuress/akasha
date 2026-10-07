"""The post-tool hook: reindex the one file a tool just wrote."""
import json
import time

import pytest

from akasha import hooks
from akasha.config import IndexRoot, load_config
from akasha.db import connect


@pytest.fixture
def env(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")
    cfg.db_path = tmp_path / "s.db"
    cfg.knowledge_dir = tmp_path / "knowledge"
    cfg.knowledge_dir.mkdir()
    root = tmp_path / "notes"
    (root / "sample-repo").mkdir(parents=True)
    cfg.index_roots = [IndexRoot(path=str(root), source="notes")]
    return connect(cfg.db_path), cfg, root


def _count(conn, path):
    return conn.execute("SELECT COUNT(*) c FROM documents WHERE path=?",
                        (str(path),)).fetchone()["c"]


def _payload(path, key="file_path"):
    return json.dumps({"tool_name": "Write", "tool_input": {key: str(path)}})


def test_extract_path_from_claude_shape():
    assert hooks.extract_path({"tool_name": "Edit",
                               "tool_input": {"file_path": "/repo/a.md"}}) == "/repo/a.md"


def test_extract_path_from_gemini_shape():
    assert hooks.extract_path({"toolName": "write_file",
                               "args": {"absolute_path": "/repo/b.md"}}) == "/repo/b.md"


def test_extract_path_from_generic_shape():
    assert hooks.extract_path(
        {"tool": {"name": "edit", "input": {"path": "/repo/c.md"}}}) == "/repo/c.md"


def test_extract_path_is_none_when_absent():
    assert hooks.extract_path({"tool_name": "Bash", "tool_input": {"command": "ls"}}) is None


def test_a_file_inside_a_root_is_indexed(env):
    conn, cfg, root = env
    doc = root / "sample-repo" / "n.md"
    doc.write_text("## A\nthe catalog cache was globally off\n")
    assert hooks.run_post_tool(_payload(doc), conn, cfg) == 0
    assert _count(conn, doc) == 1


def test_a_file_the_root_excludes_is_not_indexed(env):
    """The walk honours a root's exclude patterns; the hook must agree, or a tool writing
    into an excluded folder puts it in the knowledge base anyway."""
    conn, cfg, root = env
    cfg.index_roots[0].exclude = ["**/cache/**"]
    doc = root / "cache" / "n.md"
    doc.parent.mkdir()
    doc.write_text("## A\ngenerated cache entry\n")
    hooks.run_post_tool(_payload(doc), conn, cfg)
    assert _count(conn, doc) == 0


def test_a_file_outside_the_roots_include_is_not_indexed(env):
    conn, cfg, root = env
    cfg.index_roots[0].include = ["sample-repo/**"]
    doc = root / "other" / "n.md"
    doc.parent.mkdir()
    doc.write_text("## A\nnot in the include\n")
    hooks.run_post_tool(_payload(doc), conn, cfg)
    assert _count(conn, doc) == 0


def test_an_edit_to_an_indexed_file_is_picked_up(env):
    conn, cfg, root = env
    doc = root / "sample-repo" / "n.md"
    doc.write_text("## A\nfirst body\n")
    hooks.run_post_tool(_payload(doc), conn, cfg)
    doc.write_text("## A\nthe catalog cache was globally off\n")
    hooks.run_post_tool(_payload(doc), conn, cfg)
    row = conn.execute("SELECT body FROM chunks").fetchone()
    assert "catalog cache" in row["body"]


def test_a_native_knowledge_file_is_indexed(env):
    conn, cfg, _ = env
    doc = cfg.knowledge_dir / "sample-repo" / "n.md"
    doc.parent.mkdir()
    doc.write_text("## A\nnative note\n")
    hooks.run_post_tool(_payload(doc), conn, cfg)
    assert _count(conn, doc) == 1


def test_a_file_outside_every_root_is_ignored(env, tmp_path):
    """Editing a markdown file elsewhere must not put it in the knowledge base."""
    conn, cfg, _ = env
    stray = tmp_path / "code" / "README.md"
    stray.parent.mkdir()
    stray.write_text("## Notes\nstray\n")
    assert hooks.run_post_tool(_payload(stray), conn, cfg) == 0
    assert _count(conn, stray) == 0


def test_a_symlink_out_of_a_root_is_ignored(env, tmp_path):
    conn, cfg, root = env
    target = tmp_path / "outside.md"
    target.write_text("## A\nsecret-ish\n")
    link = root / "sample-repo" / "link.md"
    link.symlink_to(target)
    hooks.run_post_tool(_payload(link), conn, cfg)
    assert _count(conn, target) == 0 and _count(conn, link) == 0


def test_a_denied_file_inside_a_root_is_ignored(env):
    conn, cfg, root = env
    secret = root / "sample-repo" / ".env.md"
    secret.write_text("KEY=abc\n")
    hooks.run_post_tool(_payload(secret), conn, cfg)
    assert _count(conn, secret) == 0


def test_a_non_markdown_file_inside_a_root_is_ignored(env):
    conn, cfg, root = env
    other = root / "sample-repo" / "run.trace"
    other.write_text("binary-ish")
    hooks.run_post_tool(_payload(other), conn, cfg)
    assert _count(conn, other) == 0


@pytest.mark.parametrize("raw", ["", "   ", "{not json", "[]", "42", '{"tool_input": 3}',
                                 '{"tool_input": {"file_path": ""}}'])
def test_a_bad_payload_exits_zero_and_does_nothing(env, raw):
    """A hook must never block a tool call over a payload it cannot read."""
    conn, cfg, _ = env
    assert hooks.run_post_tool(raw, conn, cfg) == 0
    assert conn.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"] == 0


def test_a_deleted_path_exits_zero(env):
    conn, cfg, root = env
    assert hooks.run_post_tool(_payload(root / "gone.md"), conn, cfg) == 0


def test_an_indexing_failure_exits_zero(env, monkeypatch):
    conn, cfg, root = env
    doc = root / "sample-repo" / "n.md"
    doc.write_text("## A\nx\n")

    def boom(*a, **k):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(hooks, "index_path", boom)
    assert hooks.run_post_tool(_payload(doc), conn, cfg) == 0


def test_post_tool_never_walks_the_corpus(env, monkeypatch):
    """A full walk on every tool call would put a corpus scan in front of each edit."""
    conn, cfg, root = env
    doc = root / "sample-repo" / "n.md"
    doc.write_text("## A\nx\n")
    monkeypatch.setattr(hooks, "index_all",
                        lambda *a, **k: pytest.fail("post-tool walked the corpus"))
    hooks.run_post_tool(_payload(doc), conn, cfg)
    assert _count(conn, doc) == 1


def test_a_root_reached_through_a_symlink_does_not_duplicate_documents(env, tmp_path):
    """The walk records paths as the root is configured; a tool may report the resolved
    spelling. Both must land on one document."""
    from akasha.index import index_all

    conn, cfg, root = env
    alias = tmp_path / "alias"
    alias.symlink_to(root)
    cfg.index_roots = [IndexRoot(path=str(alias), source="notes")]
    doc = root / "sample-repo" / "n.md"
    doc.write_text("## A\nbody\n")
    index_all(conn, cfg)
    hooks.run_post_tool(_payload(doc), conn, cfg)
    assert conn.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"] == 1


def test_a_non_markdown_path_returns_before_any_root_lookup(env, monkeypatch):
    """Most tool calls write source files; resolving paths for each is wasted work."""
    conn, cfg, root = env
    monkeypatch.setattr(hooks, "_owning_root",
                        lambda *a, **k: pytest.fail("looked up roots for a non-markdown path"))
    assert hooks.run_post_tool(_payload(root / "sample-repo" / "main.py"), conn, cfg) == 0


def test_a_locked_database_never_blocks_the_tool_for_seconds(env):
    """The host waits on this hook after every write; a contended database must cost it a
    fraction of a second, and the file is simply picked up at the next session start."""
    import sqlite3

    conn, cfg, root = env
    doc = root / "sample-repo" / "n.md"
    doc.write_text("## A\nx\n")
    locker = sqlite3.connect(cfg.db_path, isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    try:
        start = time.monotonic()
        assert hooks.run_post_tool(_payload(doc), cfg=cfg) == 0
        assert time.monotonic() - start < 2
    finally:
        locker.execute("ROLLBACK")
        locker.close()


def test_post_tool_closes_the_connection_it_opened(env, monkeypatch):
    from akasha import db

    conn, cfg, root = env
    doc = root / "sample-repo" / "n.md"
    doc.write_text("## A\nx\n")
    real = db.connect
    opened = []

    class Spy:
        def __init__(self, *a, **k):
            self.conn = real(*a, **k)
            self.closed = False
            opened.append(self)

        def __getattr__(self, name):
            return getattr(self.conn, name)

        def close(self):
            self.closed = True
            self.conn.close()

    monkeypatch.setattr(db, "connect", lambda *a, **k: Spy(*a, **k))
    hooks.run_post_tool(_payload(doc), cfg=cfg)
    assert opened and all(s.closed for s in opened)


def test_a_failed_post_tool_rolls_back_the_connection_it_was_given(env, monkeypatch):
    conn, cfg, root = env
    doc = root / "sample-repo" / "n.md"
    doc.write_text("## A\nx\n")

    def half_done(conn, *a, **k):
        conn.execute("BEGIN")
        conn.execute("INSERT INTO meta (key, value) VALUES ('half', 'done')")
        raise RuntimeError("boom")

    monkeypatch.setattr(hooks, "index_path", half_done)
    hooks.run_post_tool(_payload(doc), conn, cfg)
    assert not conn.in_transaction
    assert conn.execute("SELECT 1 FROM meta WHERE key='half'").fetchone() is None


def test_a_symlink_to_another_file_in_the_root_is_indexed_under_its_own_spelling(env):
    """The walk records each spelling it meets, so the hook must too: re-anchoring the link
    to its target would index a path the walk never recorded for that name, and the next
    walk would add the link as a second document."""
    from akasha.index import index_all

    conn, cfg, root = env
    target = root / "sample-repo" / "a.md"
    target.write_text("## A\nbody\n")
    link = root / "sample-repo" / "link.md"
    link.symlink_to(target)
    hooks.run_post_tool(_payload(link), conn, cfg)
    assert _count(conn, link) == 1 and _count(conn, target) == 0
    index_all(conn, cfg)
    paths = {r["path"] for r in conn.execute("SELECT path FROM documents")}
    assert paths == {str(target), str(link)}
