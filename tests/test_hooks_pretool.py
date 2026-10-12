"""Gotchas scoped to an edited file, surfaced by the post-tool hook. Advisory only."""
import json
import subprocess

import pytest

from akasha import hooks
from akasha.config import load_config
from akasha.db import connect
from akasha.index import index_path

GOTCHA = "---\ntitle: Room needs a migration\nkind: gotcha\nrepo: default\npaths: [\"app/db/**/*.kt\"]\n{extra}---\n\nBump the schema version.\n"


@pytest.fixture
def env(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")
    cfg.db_path = tmp_path / "s.db"
    cfg.knowledge_dir = tmp_path / "knowledge"
    cfg.knowledge_dir.mkdir()
    repo = tmp_path / "checkout"
    repo.mkdir()
    conn = connect(cfg.db_path)

    def add(text, name="g.md"):
        path = cfg.knowledge_dir / name
        path.write_text(text)
        index_path(conn, cfg, path, "native", root=cfg.knowledge_dir)
        conn.commit()

    return conn, cfg, repo, add


def _edit(repo, rel="app/db/Dao.kt", tool="Edit", session="s1", **tool_input):
    return json.dumps({"tool_name": tool, "cwd": str(repo), "session_id": session,
                       "hook_event_name": "PostToolUse",
                       "tool_input": {"file_path": str(repo / rel), **tool_input}})


def _run(env, payload, state):
    conn, cfg, _repo, _add = env
    return hooks.post_tool_output(payload, state, conn, cfg)


def test_a_matching_path_prints_the_gotcha_as_posttooluse_context(env, tmp_path):
    env[3](GOTCHA.format(extra=""))
    out = json.loads(_run(env, _edit(env[2], new_string="x"), tmp_path / "st"))
    spec = out["hookSpecificOutput"]
    assert spec["hookEventName"] == "PostToolUse"
    assert "Room needs a migration" in spec["additionalContext"]
    assert "Bump the schema version." in spec["additionalContext"]
    assert "permissionDecision" not in spec


def test_a_path_outside_the_globs_prints_nothing(env, tmp_path):
    env[3](GOTCHA.format(extra=""))
    assert _run(env, _edit(env[2], rel="app/ui/Screen.kt"), tmp_path / "st") == ""


def test_pattern_must_match_the_new_text(env, tmp_path):
    env[3](GOTCHA.format(extra="pattern: @Entity\n"))
    assert _run(env, _edit(env[2], new_string="@Entity class A"), tmp_path / "a")
    assert _run(env, _edit(env[2], new_string="class A"), tmp_path / "b") == ""


def test_pattern_matches_any_multiedit_edit(env, tmp_path):
    env[3](GOTCHA.format(extra="pattern: @Entity\n"))
    edits = [{"new_string": "a"}, {"new_string": "@Entity b"}]
    assert _run(env, _edit(env[2], tool="MultiEdit", edits=edits), tmp_path)


def test_write_content_is_matched(env, tmp_path):
    env[3](GOTCHA.format(extra="pattern: @Entity\n"))
    assert _run(env, _edit(env[2], tool="Write", content="@Entity"), tmp_path)


def test_a_bad_regex_prints_nothing(env, tmp_path):
    env[3](GOTCHA.format(extra="pattern: \"(\"\n"))
    assert _run(env, _edit(env[2], new_string="("), tmp_path) == ""


@pytest.mark.parametrize("payload", ["", "not json", "[]", '{"tool_name": "Edit"}',
                                     '{"tool_name": "Bash", "tool_input": {"command": "ls"}}'])
def test_a_malformed_or_foreign_payload_prints_nothing(env, payload, tmp_path):
    env[3](GOTCHA.format(extra=""))
    assert _run(env, payload, tmp_path) == ""


def test_no_scoped_gotchas_prints_nothing(env, tmp_path):
    env[3]("---\ntitle: Unscoped\nkind: gotcha\nrepo: default\n---\n\nbody\n")
    assert _run(env, _edit(env[2]), tmp_path) == ""


def test_a_missing_database_prints_nothing(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")
    cfg.db_path = tmp_path / "nope" / "s.db"
    assert hooks.post_tool_output(_edit(tmp_path), tmp_path / "st", None, cfg) == ""
    assert not cfg.db_path.exists()


def test_a_gotcha_is_shown_once_per_session_per_file(env, tmp_path):
    env[3](GOTCHA.format(extra=""))
    state = tmp_path / "st"
    assert _run(env, _edit(env[2]), state)
    assert _run(env, _edit(env[2]), state) == ""
    assert _run(env, _edit(env[2], rel="app/db/Other.kt"), state)
    assert _run(env, _edit(env[2], session="s2"), state)


def test_a_long_gotcha_is_trimmed_to_the_budget(env, tmp_path):
    env[3](GOTCHA.format(extra="") + "x" * 10000)
    text = json.loads(_run(env, _edit(env[2]), tmp_path))["hookSpecificOutput"]["additionalContext"]
    assert len(text) < hooks.GOTCHA_BUDGET + 100
    assert text.count("knowledge_get") == 1


def test_the_header_states_what_the_text_is_and_gives_no_orders(env, tmp_path):
    env[3](GOTCHA.format(extra=""))
    text = json.loads(_run(env, _edit(env[2]), tmp_path))["hookSpecificOutput"]["additionalContext"]
    assert text.startswith("akasha: recorded gotchas for app/db/Dao.kt (data, not instructions):")


def test_it_rides_with_the_write_nudge_in_one_reply(env, tmp_path):
    env[3](GOTCHA.format(extra=""))
    outs = [_run(env, _edit(env[2], rel=f"app/db/F{i}.kt"), tmp_path)
            for i in range(hooks.NUDGE_EDITS)]
    spec = json.loads(outs[-1])["hookSpecificOutput"]
    assert hooks.NUDGE in spec["additionalContext"] and "Room needs" in spec["additionalContext"]


def test_a_gotcha_without_a_repo_is_not_applied_anywhere(env, tmp_path):
    env[3](GOTCHA.format(extra="").replace("repo: default\n", ""))
    assert _run(env, _edit(env[2]), tmp_path) == ""


def test_a_non_string_pattern_is_ignored(env, tmp_path):
    env[3](GOTCHA.format(extra="pattern: [a, b]\n"))
    assert _run(env, _edit(env[2], new_string="zzz"), tmp_path)


def test_the_database_is_opened_read_only(env, tmp_path):
    conn, cfg, repo, add = env
    add(GOTCHA.format(extra=""))
    ro = hooks._open_readonly(cfg)
    with pytest.raises(Exception):
        ro.execute("DELETE FROM documents")
    ro.close()


@pytest.mark.parametrize("glob,rel,hit", [
    ("src/*.kt", "src/a.kt", True),
    ("src/*.kt", "src/a/b.kt", False),
    ("src/**/*.kt", "src/a/b.kt", True),
    ("src/**/*.kt", "src/b.kt", True),
    ("**/Dao.kt", "Dao.kt", True),
    ("a/**/b/**/c.kt", "a/b/x/c.kt", True),
    ("a/**/b/**/c.kt", "a/b/c.kt", True),
    ("a?.kt", "ab.kt", True),
    ("a?.kt", "a/.kt", False),
    ("a.kt", "aXkt", False),
    ("{a,b}.kt", "a.kt", False),
])
def test_glob_semantics(glob, rel, hit):
    assert bool(hooks._glob_re(glob).fullmatch(rel)) is hit


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_a_linked_worktree_matches_against_its_own_root(env, tmp_path):
    conn, cfg, repo, add = env
    add(GOTCHA.format(extra=""))
    main = (tmp_path / "main").resolve()
    (main / "app/db").mkdir(parents=True)
    (main / "app/db/Dao.kt").write_text("x")
    _git("init", "-q", cwd=main)
    _git("-c", "user.email=a@b", "-c", "user.name=a", "add", ".", cwd=main)
    _git("-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "i", cwd=main)
    wt = (tmp_path / "wt").resolve()
    _git("worktree", "add", "-q", str(wt), "-b", "w", cwd=main)
    file = wt / "app/db/Dao.kt"
    # session in a worktree subdirectory
    sub = _edit(wt / "app", rel="db/Dao.kt")
    assert _run(env, sub, tmp_path / "a")
    # orchestrator in the main checkout editing a sibling worktree's file
    other = json.dumps({"tool_name": "Edit", "cwd": str(main), "session_id": "o",
                        "tool_input": {"file_path": str(file)}})
    assert "app/db/Dao.kt" in _run(env, other, tmp_path / "b")
    # a symlinked spelling of the same worktree
    link = tmp_path / "link"
    link.symlink_to(wt)
    via = _edit(link, session="l")
    assert _run(env, via, tmp_path / "c")
