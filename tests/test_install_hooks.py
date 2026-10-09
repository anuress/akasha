import json
import os

import pytest

from akasha.install import (HOOK_COMMAND, INSTALLED_HOOK_COMMAND, VENDORS,
                            allow_akasha_tools,
                            hook_vendors, install_hooks, post_tool_hook_command,
                            session_hook_command)

POST = "akasha hook post-tool"


def _settings(home, vendor="claude"):
    return home / f".{vendor}" / "settings.json"


def _seed(home, data, vendor="claude"):
    path = _settings(home, vendor)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data))
    return path


def _hooks(home, vendor="claude"):
    return json.loads(_settings(home, vendor).read_text())["hooks"]


def test_the_installed_command_defers_the_index_refresh(tmp_path):
    """Hosts block their startup on this hook; the installed command must pass
    --defer-refresh so a session never waits on an index."""
    install_hooks("claude", tmp_path)
    assert f"{HOOK_COMMAND} --defer-refresh" in json.dumps(_hooks(tmp_path))


def test_session_start_and_post_tool_are_the_only_hooks(tmp_path):
    install_hooks("claude", tmp_path)
    assert sorted(_hooks(tmp_path)) == ["PostToolUse", "SessionStart"]
    assert "pre-tool" not in json.dumps(_hooks(tmp_path))


def test_post_tool_runs_only_for_file_writing_and_recording_tools(tmp_path):
    """A process per Read or Bash call would cost more than the reindex is worth. The
    recording tools are matched so the write nudge sees a session record something;
    Gemini names MCP tools mcp_<server>_<tool>, Claude mcp__<server>__<tool>."""
    install_hooks("claude", tmp_path)
    install_hooks("gemini", tmp_path)
    claude = _hooks(tmp_path)["PostToolUse"][0]
    gemini = _hooks(tmp_path, "gemini")["AfterTool"][0]
    assert claude["matcher"] == (
        "Write|Edit|MultiEdit|mcp__akasha__knowledge_(write|append|update)")
    assert gemini["matcher"] == "write_file|replace|mcp_akasha_knowledge_(write|append|update)"
    assert claude["hooks"][0]["command"] == POST


def test_gemini_hooks_use_its_own_event_names(tmp_path):
    """Gemini fires AfterTool, not PostToolUse; an entry under the wrong name never runs."""
    install_hooks("gemini", tmp_path)
    assert sorted(_hooks(tmp_path, "gemini")) == ["AfterTool", "SessionStart"]
    assert session_hook_command("gemini", tmp_path) == INSTALLED_HOOK_COMMAND
    assert post_tool_hook_command("gemini", tmp_path) == POST


def test_only_vendors_with_a_hook_schema_are_listed():
    assert hook_vendors() == ["claude", "gemini"]


def test_existing_settings_and_hooks_survive(tmp_path):
    """settings.json holds the user's own rules and hooks; merge, never replace."""
    path = _seed(tmp_path, {
        "model": "opus",
        "hooks": {"SessionStart": [{"matcher": "Bash", "hooks": [
            {"type": "command", "command": "existing.sh"}]}]},
    })
    install_hooks("claude", tmp_path)
    data = json.loads(path.read_text())
    assert data["model"] == "opus"
    assert "existing.sh" in json.dumps(data["hooks"])
    assert INSTALLED_HOOK_COMMAND in json.dumps(data["hooks"])


def test_reinstalling_is_a_no_op(tmp_path):
    install_hooks("claude", tmp_path)
    assert "already" in install_hooks("claude", tmp_path)
    text = _settings(tmp_path).read_text()
    assert text.count(HOOK_COMMAND) == 1 and text.count(POST) == 1


def test_a_session_hook_without_defer_is_upgraded_in_place(tmp_path):
    """An entry that waits on the index blocks every session start; reinstalling fixes
    it rather than adding a second entry."""
    path = _seed(tmp_path, {"hooks": {"SessionStart": [{"matcher": "*", "hooks": [
        {"type": "command", "command": HOOK_COMMAND}]}]}})
    install_hooks("claude", tmp_path)
    flat = path.read_text()
    assert flat.count(HOOK_COMMAND) == 1 and "--defer-refresh" in flat


def test_the_upgrade_matches_only_akashas_exact_command(tmp_path):
    """`session-start-foo` is some other program that shares a prefix."""
    foreign = "akasha hook session-start-foo"
    _seed(tmp_path, {"hooks": {"SessionStart": [{"matcher": "*", "hooks": [
        {"type": "command", "command": foreign}]}]}})
    install_hooks("claude", tmp_path)
    commands = [h["command"] for e in _hooks(tmp_path)["SessionStart"] for h in e["hooks"]]
    assert foreign in commands and INSTALLED_HOOK_COMMAND in commands


def test_the_upgrade_keeps_extra_arguments_after_the_exact_command(tmp_path):
    """Exact command followed by a space-separated argument is still akasha's own."""
    custom = f"{HOOK_COMMAND} --repo my-repo"
    _seed(tmp_path, {"hooks": {"SessionStart": [{"matcher": "*", "hooks": [
        {"type": "command", "command": custom}]}]}})
    install_hooks("claude", tmp_path)
    commands = [h["command"] for e in _hooks(tmp_path)["SessionStart"] for h in e["hooks"]]
    assert commands == [INSTALLED_HOOK_COMMAND]


def test_a_stale_post_tool_matcher_on_akashas_entry_is_upgraded(tmp_path):
    """A matcher that misses Edit and MultiEdit leaves edits unindexed; one that misses
    the recording tools leaves the write nudge blind to writes."""
    _seed(tmp_path, {"hooks": {"PostToolUse": [{"matcher": "Write", "hooks": [
        {"type": "command", "command": POST}]}]}})
    install_hooks("claude", tmp_path)
    entries = _hooks(tmp_path)["PostToolUse"]
    assert len(entries) == 1
    assert entries[0]["matcher"] == VENDORS["claude"].write_matcher


def test_a_matcher_shared_with_someone_elses_hook_is_left_alone(tmp_path):
    _seed(tmp_path, {"hooks": {"PostToolUse": [{"matcher": "Bash", "hooks": [
        {"type": "command", "command": POST}, {"type": "command", "command": "other.sh"}]}]}})
    install_hooks("claude", tmp_path)
    assert _hooks(tmp_path)["PostToolUse"][0]["matcher"] == "Bash"


def test_dry_run_writes_nothing(tmp_path):
    assert "would" in install_hooks("claude", tmp_path, dry_run=True)
    assert not (tmp_path / ".claude").exists()


@pytest.mark.parametrize("raw", [b"{not json", b"[]", b"\xff\xfe"])
def test_unusable_settings_are_left_alone(tmp_path, raw):
    path = _settings(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    assert "left untouched" in install_hooks("claude", tmp_path)
    assert path.read_bytes() == raw


@pytest.mark.parametrize("settings", [
    {"hooks": {"SessionStart": {}}},
    {"hooks": {"PostToolUse": "x"}},
    {"hooks": {"SessionStart": [{"hooks": {}}]}},
    {"hooks": {"SessionStart": [{"hooks": "x"}]}},
    {"hooks": []},
])
def test_a_wrongly_shaped_hooks_section_is_left_untouched_with_a_reason(tmp_path, settings):
    """Mutating a shape we do not understand would corrupt the user's settings."""
    path = _seed(tmp_path, settings)
    before = path.read_text()
    assert "left untouched" in install_hooks("claude", tmp_path)
    assert path.read_text() == before


def test_a_failure_mid_write_leaves_settings_intact(tmp_path, monkeypatch):
    path = _seed(tmp_path, {"model": "opus"})
    before = path.read_text()

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr("os.fsync", boom)
    assert "failed" in install_hooks("claude", tmp_path)
    assert path.read_text() == before


# --- permission entry ----------------------------------------------------------------

def _allow(home):
    return json.loads(_settings(home).read_text())["permissions"]["allow"]


def test_only_akasha_is_allowed_never_a_wildcard(tmp_path):
    allow_akasha_tools(tmp_path)
    assert _allow(tmp_path) == ["mcp__akasha"]


def test_allowing_is_idempotent(tmp_path):
    allow_akasha_tools(tmp_path)
    assert "already" in allow_akasha_tools(tmp_path)
    assert _allow(tmp_path).count("mcp__akasha") == 1


def test_existing_permissions_survive(tmp_path):
    path = _seed(tmp_path, {
        "permissions": {"allow": ["Bash(git status)"], "deny": ["Read(./secrets/**)"]},
        "model": "opus"})
    allow_akasha_tools(tmp_path)
    data = json.loads(path.read_text())
    assert "Bash(git status)" in data["permissions"]["allow"]
    assert data["permissions"]["deny"] == ["Read(./secrets/**)"]
    assert data["model"] == "opus"


@pytest.mark.parametrize("section", ["deny", "ask"])
def test_an_explicit_deny_or_ask_rule_is_never_overridden(tmp_path, section):
    path = _seed(tmp_path, {"permissions": {section: ["mcp__akasha__knowledge_write"]}})
    result = allow_akasha_tools(tmp_path)
    assert "allow" not in json.loads(path.read_text())["permissions"]
    assert section in result


def test_a_deliberately_narrow_allow_is_not_widened(tmp_path):
    """Allowing one tool is a choice; adding the whole server would undo it."""
    path = _seed(tmp_path, {"permissions": {"allow": ["mcp__akasha__knowledge_search"]}})
    before = path.read_text()
    assert "already" in allow_akasha_tools(tmp_path)
    assert path.read_text() == before


def test_no_allow_skips_the_permission_entry(tmp_path):
    assert "skipped" in allow_akasha_tools(tmp_path, no_allow=True)
    assert not _settings(tmp_path).exists()


def test_allow_dry_run_writes_nothing(tmp_path):
    assert "would" in allow_akasha_tools(tmp_path, dry_run=True)
    assert not _settings(tmp_path).exists()


@pytest.mark.parametrize("permissions", [
    {"allow": "mcp__other"}, {"allow": {}}, {"deny": "x"}, {"ask": 3}, [],
])
def test_a_wrongly_shaped_permissions_section_is_left_untouched(tmp_path, permissions):
    """A non-list `allow` used to crash the append."""
    path = _seed(tmp_path, {"permissions": permissions})
    before = path.read_text()
    assert "left untouched" in allow_akasha_tools(tmp_path)
    assert path.read_text() == before


def test_unparseable_permission_settings_are_left_alone(tmp_path):
    path = _seed(tmp_path, "{not json")
    assert "left untouched" in allow_akasha_tools(tmp_path)
    assert path.read_text() == "{not json"


def test_a_rewrite_keeps_the_settings_file_mode(tmp_path):
    path = _seed(tmp_path, {})
    os.chmod(path, 0o640)
    allow_akasha_tools(tmp_path)
    assert path.stat().st_mode & 0o777 == 0o640
