import pytest

from akasha.config import load_config
from akasha.db import connect
from akasha.knowledge import archive, timeline, write


@pytest.fixture
def env(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")
    cfg.knowledge_dir = tmp_path / "knowledge"
    cfg.knowledge_dir.mkdir()
    cfg.index_roots = []
    return connect(tmp_path / "s.db"), cfg


def test_timeline_is_chronological(env):
    conn, cfg = env
    first = write(conn, cfg, "Overview", "## A\nstart\n", repo="r", feature="loan-limits")
    second = write(conn, cfg, "Findings", "## A\nmiddle\n", repo="r", feature="loan-limits")
    third = write(conn, cfg, "Result", "## A\nend\n", repo="r", feature="loan-limits")
    assert [d["id"] for d in timeline(conn, feature="loan-limits")] == [first, second, third]


def test_timeline_scopes_to_a_feature(env):
    conn, cfg = env
    write(conn, cfg, "A", "## A\nx\n", repo="r", feature="loan-limits")
    write(conn, cfg, "B", "## A\ny\n", repo="r", feature="catalog-cache")
    assert len(timeline(conn, feature="loan-limits")) == 1


def test_timeline_scopes_to_a_repo(env):
    conn, cfg = env
    write(conn, cfg, "A", "## A\nx\n", repo="sample-repo")
    write(conn, cfg, "B", "## A\ny\n", repo="sample-repo-b")
    assert len(timeline(conn, repo="sample-repo-b")) == 1


def test_timeline_excludes_deleted_documents(env):
    conn, cfg = env
    doc_id = write(conn, cfg, "Gone", "## A\nx\n", repo="r", feature="f")
    conn.execute("UPDATE documents SET deleted_at=datetime('now') WHERE id=?", (doc_id,))
    conn.commit()
    assert timeline(conn, feature="f") == []


def test_timeline_on_an_unknown_feature_is_empty(env):
    conn, cfg = env
    assert timeline(conn, feature="never-existed") == []


def test_timeline_includes_the_kind(env):
    conn, cfg = env
    write(conn, cfg, "Spec", "## A\nx\n", repo="r", feature="f", kind="spec")
    entry = timeline(conn, feature="f")[0]
    assert entry["kind"] == "spec"


def test_timeline_entries_carry_no_path(env):
    """knowledge_get returns the path; repeating it on every entry costs the caller
    context for nothing."""
    conn, cfg = env
    write(conn, cfg, "Spec", "## A\nx\n", repo="r", feature="f")
    assert "path" not in timeline(conn, feature="f")[0]


def test_timeline_omits_default_valued_fields(env):
    """An active status and an absent repo are the defaults; only departures are said."""
    conn, cfg = env
    doc_id = write(conn, cfg, "Plain", "## A\nx\n", feature="f")
    plain = timeline(conn, feature="f")[0]
    assert "status" not in plain and "repo" not in plain
    archive(conn, cfg, doc_id)
    assert timeline(conn, feature="f")[0]["status"] == "archived"
    write(conn, cfg, "Scoped", "## A\ny\n", repo="r", feature="f")
    assert timeline(conn, feature="f")[1]["repo"] == "r"


def test_timeline_limit_keeps_the_oldest_entries(env):
    conn, cfg = env
    ids = [write(conn, cfg, f"Step {i}", "## A\nx\n", repo="r", feature="f") for i in range(3)]
    assert [d["id"] for d in timeline(conn, feature="f", limit=2)] == ids[:2]
