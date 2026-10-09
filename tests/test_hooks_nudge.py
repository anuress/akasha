"""The post-tool nudge: after a run of edits with nothing recorded, remind the agent once.

Measured before building: in long sessions agents edited code for dozens of tool calls
without a single write, and the session-start brief is far behind them by then. The
post-tool hook already runs on every edit in every harness, so the reminder rides on it.
"""
import json
import os
import time

import pytest

from akasha import hooks


def _edit(session="s1", tool="Edit", event="PostToolUse"):
    return json.dumps({"session_id": session, "hook_event_name": event, "tool_name": tool,
                       "tool_input": {"file_path": "/repo/main.py"}})


def _edits(state, n, **kw):
    return [hooks.post_tool_output(_edit(**kw), state) for _ in range(n)]


def test_the_nudge_fires_on_the_threshold_edit_and_not_before(tmp_path):
    out = _edits(tmp_path, hooks.NUDGE_EDITS)
    assert out[:-1] == [""] * (hooks.NUDGE_EDITS - 1)
    assert "knowledge_write" in json.loads(out[-1])["hookSpecificOutput"]["additionalContext"]


def test_the_nudge_fires_once_per_run_of_edits(tmp_path):
    """A reminder repeated on every edit after the threshold becomes noise the agent
    learns to skip, or pressure that produces junk writes."""
    out = _edits(tmp_path, hooks.NUDGE_EDITS * 2)
    assert sum(1 for o in out if o) == 1


def test_the_nudge_says_doing_nothing_is_fine(tmp_path):
    """Without an explicit way out, a nudge pushes the agent to write something to
    satisfy it."""
    text = json.loads(_edits(tmp_path, hooks.NUDGE_EDITS)[-1])
    assert "otherwise carry on" in text["hookSpecificOutput"]["additionalContext"]


@pytest.mark.parametrize("tool", ["mcp__akasha__knowledge_write",       # claude, pi
                                  "mcp__akasha__knowledge_append",
                                  "mcp__akasha__knowledge_update",
                                  "mcp_akasha_knowledge_write"])        # gemini
def test_recording_restarts_the_count(tmp_path, tool):
    _edits(tmp_path, hooks.NUDGE_EDITS - 1)
    hooks.post_tool_output(_edit(tool=tool), tmp_path)
    assert _edits(tmp_path, hooks.NUDGE_EDITS - 1) == [""] * (hooks.NUDGE_EDITS - 1)
    assert hooks.post_tool_output(_edit(), tmp_path)


def test_a_session_nudged_once_is_nudged_again_after_recording_and_editing_on(tmp_path):
    """A long session should be reminded again later, not only once in its life."""
    _edits(tmp_path, hooks.NUDGE_EDITS)
    hooks.post_tool_output(_edit(tool="mcp__akasha__knowledge_append"), tmp_path)
    assert _edits(tmp_path, hooks.NUDGE_EDITS)[-1]


def test_sessions_are_counted_apart(tmp_path):
    _edits(tmp_path, hooks.NUDGE_EDITS - 1, session="a")
    assert hooks.post_tool_output(_edit(session="b"), tmp_path) == ""


def test_the_reply_names_the_event_the_host_sent(tmp_path):
    """Claude reads PostToolUse, Gemini AfterTool; a mismatched name may be dropped."""
    out = _edits(tmp_path, hooks.NUDGE_EDITS, event="AfterTool")[-1]
    assert json.loads(out)["hookSpecificOutput"]["hookEventName"] == "AfterTool"


def test_no_session_id_means_no_nudge_and_no_state(tmp_path):
    payload = json.dumps({"tool_name": "Edit", "tool_input": {"file_path": "/repo/a.py"}})
    for _ in range(hooks.NUDGE_EDITS):
        assert hooks.post_tool_output(payload, tmp_path) == ""
    assert not tmp_path.exists() or not any(tmp_path.iterdir())


def test_a_hostile_session_id_stays_inside_the_state_dir(tmp_path):
    state = tmp_path / "sessions"
    hooks.post_tool_output(_edit(session="../../escape"), state)
    assert not (tmp_path / "escape").exists()
    assert [p.parent for p in state.iterdir()] == [state]


def test_an_unwritable_state_dir_fails_open(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("")
    assert _edits(blocker / "sessions", hooks.NUDGE_EDITS) == [""] * hooks.NUDGE_EDITS


def test_state_from_old_sessions_is_pruned(tmp_path):
    """One file per session would otherwise grow without bound."""
    old = tmp_path / "old"
    old.write_text("3")
    stale = time.time() - hooks.NUDGE_STATE_DAYS * 86400 - 60
    os.utime(old, (stale, stale))
    hooks.post_tool_output(_edit(session="new"), tmp_path)
    assert not old.exists()


@pytest.mark.parametrize("raw", ["", "{broken", "[]", "42"])
def test_a_bad_payload_says_nothing(tmp_path, raw):
    assert hooks.post_tool_output(raw, tmp_path) == ""
