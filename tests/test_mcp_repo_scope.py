"""gitctx.resolve_repo is the one place repo scope is decided. An omitted `repo` argument
means "this checkout" on every tool, an explicit one wins, and '*' widens to every repo."""
import subprocess

import pytest

from akasha.config import load_config
from akasha.db import connect
from akasha.mcp_server import call_tool


@pytest.fixture
def env(tmp_path, monkeypatch):
    cfg = load_config(tmp_path / "absent.toml")
    cfg.knowledge_dir = tmp_path / "knowledge"
    cfg.knowledge_dir.mkdir()
    cfg.index_roots = []
    conn = connect(tmp_path / "s.db")
    monkeypatch.setattr("akasha.mcp_server._ctx", lambda: (conn, cfg))
    return conn, cfg


def _git_repo(path):
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    return path


def test_write_derives_repo_from_cwd_when_omitted(env, tmp_path, monkeypatch):
    conn, _ = env
    monkeypatch.chdir(_git_repo(tmp_path / "sample-repo"))
    out = call_tool("knowledge_write", {"title": "T", "body": "## H\nbody"})
    assert out["repo"] == "sample-repo"
    assert conn.execute("SELECT repo FROM documents WHERE id=?",
                        (out["id"],)).fetchone()["repo"] == "sample-repo"


def test_write_explicit_repo_still_wins(env, tmp_path, monkeypatch):
    monkeypatch.chdir(_git_repo(tmp_path / "sample-repo"))
    out = call_tool("knowledge_write", {"title": "T", "body": "## H\nbody", "repo": "other"})
    assert out["repo"] == "other"


def test_write_star_files_under_no_repo(env, tmp_path, monkeypatch):
    monkeypatch.chdir(_git_repo(tmp_path / "sample-repo"))
    assert call_tool("knowledge_write", {"title": "T", "body": "## H\nbody",
                                         "repo": "*"})["repo"] is None


def test_write_outside_any_checkout_files_repo_less(env, tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)
    out = call_tool("knowledge_write", {"title": "T", "body": "## H\nbody"})
    assert "error" not in out and out["repo"] is None


def test_search_prefers_the_derived_repo_when_omitted(env, tmp_path, monkeypatch):
    """The other repo's document repeats the term, so it would win on raw score; only a
    repo-scoped attempt returns this checkout's weaker one."""
    call_tool("knowledge_write", {"title": "Beta", "body": "## H\ncap-zzq cap-zzq here",
                                  "repo": "other-repo"})
    call_tool("knowledge_write", {"title": "Alpha", "body": "## H\ncap-zzq here",
                                  "repo": "sample-repo"})
    monkeypatch.chdir(_git_repo(tmp_path / "sample-repo"))
    hit = call_tool("knowledge_search", {"q": "cap-zzq"})[0]
    assert hit["repo"] == "sample-repo" and "cross_repo" not in hit


def _two_repo_timeline(monkeypatch, tmp_path):
    call_tool("knowledge_write", {"title": "In", "body": "## H\nx", "repo": "sample-repo"})
    call_tool("knowledge_write", {"title": "Out", "body": "## H\nx", "repo": "other-repo"})
    monkeypatch.chdir(_git_repo(tmp_path / "sample-repo"))


def test_timeline_derives_repo_when_omitted(env, tmp_path, monkeypatch):
    _two_repo_timeline(monkeypatch, tmp_path)
    assert [r["title"] for r in call_tool("knowledge_timeline", {})] == ["In"]


def test_timeline_explicit_repo_wins(env, tmp_path, monkeypatch):
    _two_repo_timeline(monkeypatch, tmp_path)
    assert [r["title"] for r in call_tool("knowledge_timeline", {"repo": "other-repo"})] == ["Out"]


def test_timeline_star_means_every_repo(env, tmp_path, monkeypatch):
    _two_repo_timeline(monkeypatch, tmp_path)
    assert {r["title"] for r in call_tool("knowledge_timeline", {"repo": "*"})} == {"In", "Out"}
