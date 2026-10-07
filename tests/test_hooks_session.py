import subprocess
import sys
import types

import pytest

from akasha import hooks
from akasha.config import load_config
from akasha.db import connect
from akasha.knowledge import write


@pytest.fixture
def env(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")
    cfg.knowledge_dir = tmp_path / "knowledge"
    cfg.knowledge_dir.mkdir()
    cfg.index_roots = []
    cfg.db_path = tmp_path / "s.db"
    return connect(cfg.db_path), cfg, tmp_path


def test_brief_points_at_the_tools_even_on_an_empty_base(env):
    """A silent brief leaves the agent with no pointer to the tools."""
    conn, cfg, _ = env
    assert "knowledge_search" in hooks.session_brief(conn)


def test_brief_stays_short_and_one_line(env):
    conn, cfg, _ = env
    brief = hooks.session_brief(conn)
    assert "\n" not in brief and len(brief) < 300


def test_brief_names_the_search_and_write_tools(env):
    conn, cfg, _ = env
    brief = hooks.session_brief(conn)
    assert "knowledge_search before investigating" in brief
    assert "knowledge_write" in brief


def test_integrity_note_reads_the_cached_count_and_runs_no_fsck(env, monkeypatch):
    """Running every check at each session start is too slow; the last recorded error
    count answers the same question."""
    from akasha import fsck

    conn, cfg, _ = env
    fsck.record_errors(conn, 2)

    def boom(*a, **k):
        raise AssertionError("fsck ran during session start")

    monkeypatch.setattr(fsck, "check", boom)
    brief = hooks.session_brief(conn)
    assert "2 fsck error(s)" in brief


def test_integrity_note_is_silent_when_clean_or_never_run(env):
    from akasha import fsck

    conn, cfg, _ = env
    assert "fsck" not in hooks.session_brief(conn)          # never run
    fsck.record_errors(conn, 0)
    assert "fsck" not in hooks.session_brief(conn)          # clean


def test_a_failing_cached_read_does_not_take_the_brief_down(env, monkeypatch):
    conn, cfg, _ = env

    def boom(*a, **k):
        raise RuntimeError("meta unreadable")

    monkeypatch.setattr("akasha.fsck.cached_error_count", boom)
    assert "knowledge_search" in hooks.session_brief(conn)


def _rule(conn, cfg, repo, text="Run the linter before committing."):
    write(conn, cfg, "House rules", f"## R\n{text}\n", repo=repo, kind="convention")


def test_conventions_reach_the_session(env):
    conn, cfg, _ = env
    _rule(conn, cfg, "sample-repo")
    assert "Run the linter before committing" in hooks.session_conventions(conn, repo="sample-repo")


def test_default_scope_applies_to_a_session_with_no_repo(env):
    conn, cfg, _ = env
    _rule(conn, cfg, "default")
    assert "Run the linter before committing" in hooks.session_conventions(conn, repo=None)


def test_conventions_are_silent_when_there_are_none(env):
    conn, cfg, _ = env
    write(conn, cfg, "Not a rule", "## A\nreference\n", repo="sample-repo")
    assert hooks.session_conventions(conn, repo="sample-repo") == ""


def test_conventions_say_they_outrank_a_conflicting_note(env):
    """The block lands beside whatever memory file the harness loaded; without stated
    precedence the tie is broken by reading order."""
    conn, cfg, _ = env
    _rule(conn, cfg, "sample-repo")
    block = hooks.session_conventions(conn, repo="sample-repo")
    assert "Follow them." in block and "outrank" in block and "conflict" in block


def test_an_archived_convention_is_not_injected(env):
    from akasha.knowledge import archive

    conn, cfg, _ = env
    _rule(conn, cfg, "sample-repo")
    doc = conn.execute("SELECT id FROM documents").fetchone()["id"]
    archive(conn, cfg, doc)
    assert hooks.session_conventions(conn, repo="sample-repo") == ""


def test_a_failing_convention_render_does_not_take_the_session_down(env, monkeypatch):
    conn, _, _ = env

    def boom(*a, **k):
        raise RuntimeError("index is wedged")

    monkeypatch.setattr(hooks, "_conventions", boom)
    assert hooks.session_conventions(conn, repo="sample-repo") == ""


def test_an_over_budget_convention_is_truncated_with_a_pointer(env):
    conn, cfg, _ = env
    _rule(conn, cfg, "sample-repo", "rule. " * 3000)
    block = hooks.session_conventions(conn, repo="sample-repo")
    assert "truncated; knowledge_get" in block
    assert len(block) < hooks.CONVENTION_BUDGET + 600
    assert hooks.conventions_over_budget(conn)[0]["title"] == "House rules"


# --- the hook never waits for an index refresh ---------------------------------------

def test_defer_refresh_replies_without_the_refresh_and_spawns_a_detached_child(
        env, monkeypatch):
    """Hosts run this hook synchronously and block their startup on it. The reply must
    not wait for the refresh; the refresh still happens, in a child nobody waits for."""
    conn, cfg, tmp = env
    spawned = []

    class FakePopen:
        def __init__(self, argv, **kwargs):
            spawned.append((argv, kwargs))

    def boom(*a, **k):
        raise AssertionError("index refresh ran on the reply path")

    monkeypatch.setattr(hooks, "index_all", boom)
    # A stub module, not Popen itself: gitctx shells out through the same module.
    monkeypatch.setattr(hooks, "subprocess", types.SimpleNamespace(
        Popen=FakePopen, DEVNULL=subprocess.DEVNULL))
    out = hooks.session_start(conn, cfg, tmp, defer_refresh=True)
    assert "knowledge_search" in out
    argv, kwargs = spawned[0]
    assert argv[-3:] == ["hook", "session-start", "--refresh-only"]
    assert kwargs["start_new_session"] is True


def test_a_missing_binary_skips_the_deferred_refresh_quietly(env, monkeypatch):
    conn, cfg, tmp = env

    def gone(*a, **k):
        raise FileNotFoundError("no akasha")

    monkeypatch.setattr(hooks, "subprocess", types.SimpleNamespace(
        Popen=gone, DEVNULL=subprocess.DEVNULL))
    assert "knowledge_search" in hooks.session_start(conn, cfg, tmp, defer_refresh=True)


def test_without_defer_the_refresh_runs_inline(env, monkeypatch):
    conn, cfg, tmp = env
    calls = []
    monkeypatch.setattr(hooks, "index_all", lambda *a, **k: calls.append(1))
    hooks.session_start(conn, cfg, tmp)
    assert calls == [1]


def test_refresh_only_refreshes_and_prints_nothing(env, monkeypatch):
    conn, cfg, tmp = env
    calls = []
    monkeypatch.setattr(hooks, "index_all", lambda *a, **k: calls.append(1))
    assert hooks.session_start(conn, cfg, tmp, refresh_only=True) == ""
    assert calls == [1]


def test_a_failing_refresh_never_breaks_the_hook(env, monkeypatch):
    conn, cfg, tmp = env

    def boom(*a, **k):
        raise sqlite3_locked()

    import sqlite3

    def sqlite3_locked():
        return sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(hooks, "index_all", boom)
    assert "knowledge_search" in hooks.session_start(conn, cfg, tmp)


def test_the_refresh_actually_makes_new_notes_searchable(env):
    from akasha.search import search

    conn, cfg, tmp = env
    (cfg.knowledge_dir / "sample-repo").mkdir()
    (cfg.knowledge_dir / "sample-repo" / "n.md").write_text("## A\nthe catalog cache was globally off\n")
    hooks.session_start(conn, cfg, tmp)
    assert search(conn, "catalog cache", cfg=cfg, all_repos=True)


# --- the vendor-neutral contract: `akasha hook session-start` ------------------------

def test_the_hook_works_with_no_payload_on_stdin(tmp_path):
    """An external adapter may have nothing to send. Exit 0, brief on stdout."""
    code, out = hooks.run_session_start(tmp_path, stdin_text="")
    assert code == 0 and "knowledge_search" in out


def test_the_hook_ignores_a_garbage_payload(tmp_path):
    code, out = hooks.run_session_start(tmp_path, stdin_text="{not json")
    assert code == 0 and "knowledge_search" in out


def test_the_hook_exits_zero_even_when_the_database_cannot_open(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("schema mismatch")

    monkeypatch.setattr("akasha.db.connect", boom)
    assert hooks.run_session_start(tmp_path) == (0, "")


def test_concurrent_refreshes_run_one_at_a_time(env, monkeypatch):
    """Every session start spawns a detached refresh; a burst of starts must not pile up
    full-corpus walks that fight over the same database."""
    import fcntl

    conn, cfg, tmp = env
    calls = []
    monkeypatch.setattr(hooks, "index_all", lambda *a, **k: calls.append(1))
    with open(cfg.db_path.parent / hooks.REFRESH_LOCK, "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert hooks.session_start(conn, cfg, tmp, refresh_only=True) == ""
        assert calls == []
    hooks.session_start(conn, cfg, tmp, refresh_only=True)
    assert calls == [1], "the lock must be released once a refresh ends"


def test_a_hook_closes_the_connection_it_opened(tmp_path, monkeypatch):
    """The hook process is short-lived, but a leaked handle holds the WAL open."""
    spy = _spy_connect(monkeypatch)
    hooks.run_session_start(tmp_path)
    assert spy.closed


def test_a_hook_rolls_back_before_closing_when_it_fails(tmp_path, monkeypatch):
    spy = _spy_connect(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("wedged")

    monkeypatch.setattr(hooks, "session_start", boom)
    assert hooks.run_session_start(tmp_path) == (0, "")
    assert spy.rolled_back and spy.closed


def _spy_connect(monkeypatch):
    from akasha import db

    real = db.connect

    class Spy:
        closed = rolled_back = False

        def __init__(self, *a, **k):
            self.conn = real(*a, **k)

        def __getattr__(self, name):
            return getattr(self.conn, name)

        def rollback(self):
            Spy.rolled_back = True
            self.conn.rollback()

        def close(self):
            Spy.closed = True
            self.conn.close()

    monkeypatch.setattr(db, "connect", lambda *a, **k: Spy(*a, **k))
    return Spy


def test_the_refresh_child_argv_is_accepted_by_the_cli(tmp_path):
    """The detached child runs this exact argv: the CLI must accept it, run the refresh
    and print nothing."""
    done = subprocess.run(hooks.refresh_command(), capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, timeout=60, cwd=tmp_path)
    assert done.returncode == 0 and done.stdout == ""


def test_the_deferred_child_runs_the_same_interpreter_as_a_module():
    """A bare `akasha` may not be on the hook's PATH; the running interpreter always is."""
    assert hooks.refresh_command() == [
        sys.executable, "-m", "akasha", "hook", "session-start", "--refresh-only"]
