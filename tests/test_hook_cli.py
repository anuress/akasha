"""The `akasha hook` commands: thin wiring over hooks.run_session_start and run_post_tool.
Drives the CLI because the hook logic is tested directly elsewhere; what these pin is the
argv, stdin and stdout contract a vendor sees."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from akasha.cli import main


@pytest.fixture
def home(isolated_home):
    assert main(["init"]) == 0
    return isolated_home


def _stdin(monkeypatch, payload) -> None:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    monkeypatch.setattr("sys.stdin", io.StringIO(text))


def _notes_root(home) -> Path:
    root = home / "notes"
    root.mkdir()
    config = home / ".akasha" / "config.toml"
    config.write_text(config.read_text() + f'\n[[index]]\npath = "{root}"\nsource = "notes"\n')
    return root


def test_session_start_prints_the_brief_and_ignores_stdin(home, capsys, monkeypatch):
    _stdin(monkeypatch, "not json at all")
    assert main(["hook", "session-start"]) == 0
    assert "knowledge_search" in capsys.readouterr().out


def test_session_start_works_with_nothing_on_stdin(home, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert main(["hook", "session-start"]) == 0
    assert "knowledge_search" in capsys.readouterr().out


def test_session_start_refreshes_the_index_inline(home, capsys):
    root = _notes_root(home)
    (root / "n.md").write_text("## A\nthe catalog cache was off\n")
    main(["hook", "session-start"])
    capsys.readouterr()
    main(["knowledge", "search", "catalog cache", "--all"])
    assert "n.md" in capsys.readouterr().out


def test_defer_refresh_replies_without_walking_and_spawns_the_child(home, capsys, monkeypatch):
    spawned = []
    monkeypatch.setattr("akasha.hooks.subprocess.Popen",
                        lambda argv, **kw: spawned.append(argv))
    monkeypatch.setattr("akasha.hooks.index_all",
                        lambda *a, **k: pytest.fail("the walk ran on the reply path"))
    assert main(["hook", "session-start", "--defer-refresh"]) == 0
    assert spawned == [__import__("akasha.hooks", fromlist=["x"]).refresh_command()]
    assert "knowledge_search" in capsys.readouterr().out


def test_refresh_only_walks_and_stays_silent(home, capsys, monkeypatch):
    walks = []
    monkeypatch.setattr("akasha.hooks.index_all", lambda *a, **k: walks.append(1))
    assert main(["hook", "session-start", "--refresh-only"]) == 0
    assert walks == [1] and capsys.readouterr().out == ""


def test_a_failing_index_never_breaks_the_hook(home, monkeypatch):
    import sqlite3

    def boom(*a, **k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr("akasha.hooks.index_all", boom)
    assert main(["hook", "session-start"]) == 0


def test_post_tool_indexes_only_the_edited_file(home, capsys, monkeypatch):
    root = _notes_root(home)
    doc = root / "n.md"
    doc.write_text("## A\nthe catalog cache was off\n")
    monkeypatch.setattr("akasha.hooks.index_all",
                        lambda *a, **k: pytest.fail("a per-edit hook must not walk"))
    _stdin(monkeypatch, {"tool_name": "Edit", "tool_input": {"file_path": str(doc)}})
    assert main(["hook", "post-tool"]) == 0
    capsys.readouterr()
    main(["knowledge", "search", "catalog cache", "--all"])
    assert "n.md" in capsys.readouterr().out


def test_post_tool_ignores_a_file_outside_every_root(home, monkeypatch):
    _notes_root(home)
    stray = home / "code" / "README.md"
    stray.parent.mkdir()
    stray.write_text("## Notes\ncatalog\n")
    _stdin(monkeypatch, {"tool_input": {"file_path": str(stray)}})
    assert main(["hook", "post-tool"]) == 0
    from akasha.config import load_config
    from akasha.db import connect

    conn = connect(load_config().db_path)
    assert conn.execute("SELECT COUNT(*) c FROM documents WHERE path=?",
                        (str(stray),)).fetchone()["c"] == 0


@pytest.mark.parametrize("payload", ["", "{broken", "[]", '{"tool_input": {"command": "ls"}}'])
def test_post_tool_fails_open_on_any_payload(home, monkeypatch, capsys, payload):
    _stdin(monkeypatch, payload)
    assert main(["hook", "post-tool"]) == 0
    assert capsys.readouterr().out == ""
