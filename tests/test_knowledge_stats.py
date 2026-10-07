"""Read-only aggregates over the events the codebase already writes.

`knowledge.searched` exists so retrieval quality is answerable. Each test below asserts a
rate or a grouping computed over the window, because a total or an unfiltered join would
answer a different, less useful question than the one asked.
"""
from datetime import datetime, timedelta, timezone

import pytest

from akasha.config import load_config
from akasha.db import connect
from akasha.events import emit
from akasha.knowledge import stats, write


@pytest.fixture
def env(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")
    cfg.knowledge_dir = tmp_path / "knowledge"
    cfg.knowledge_dir.mkdir()
    cfg.index_roots = []
    return connect(tmp_path / "s.db"), cfg


def _old_ts(days=200):
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def _search(conn, query, ids, count=None, ts=None):
    """Emit a knowledge.searched event, then optionally backdate it — emit() always
    stamps `now()`, so a window-boundary test has to move the row after the fact."""
    count = len(ids) if count is None else count
    event_id = emit(conn, "knowledge.searched", query=query, ids=ids, count=count,
                    mode="default", match_mode="auto")
    if ts is not None:
        conn.execute("UPDATE events SET ts=? WHERE id=?", (ts, event_id))
        conn.commit()
    return event_id


def test_zero_hit_share_counts_only_the_window_not_the_whole_table(env):
    """A zero-hit search from 200 days ago must not dilute or inflate today's rate —
    windowing is the entire reason to ask for a rate instead of a running total."""
    conn, cfg = env
    _search(conn, "old miss", [], count=0, ts=_old_ts())
    _search(conn, "recent miss", [], count=0)
    _search(conn, "recent hit", ["d_1"], count=1)

    result = stats(conn, days=30)
    assert result["zero_hit"]["total"] == 2
    assert result["zero_hit"]["count"] == 1


def test_surfaced_id_whose_document_row_is_gone_does_not_crash(env):
    """Documents get deleted and knowledge.searched is append-only history: an id
    outliving its document row must be skipped, never raise."""
    conn, cfg = env
    _search(conn, "ghost", ["d_missing"], count=1)

    result = stats(conn, days=30)
    assert result["surfaced_never_opened"] == []
    assert result["surfaced_never_opened_total"] == 0


def test_surfaced_never_opened_finds_a_live_untouched_document(env):
    conn, cfg = env
    doc_id = write(conn, cfg, "Untouched", "## Body\nx\n", repo="r")
    _search(conn, "q", [doc_id], count=1)

    result = stats(conn, days=30)
    assert [row["id"] for row in result["surfaced_never_opened"]] == [doc_id]
    assert result["surfaced_never_opened_total"] == 1


def test_surfaced_never_opened_excludes_a_document_that_was_since_read(env):
    conn, cfg = env
    doc_id = write(conn, cfg, "Read later", "## Body\nx\n", repo="r")
    conn.execute("UPDATE documents SET access_count=1 WHERE id=?", (doc_id,))
    conn.commit()
    _search(conn, "q", [doc_id], count=1)

    result = stats(conn, days=30)
    assert result["surfaced_never_opened"] == []


def test_repeated_queries_groups_identical_query_text(env):
    """Same question twice means the answer is missing or unfindable — the count is
    what separates that from two unrelated single questions."""
    conn, cfg = env
    _search(conn, "how does X work", ["d_1"], count=1)
    _search(conn, "how does X work", ["d_1"], count=1)
    _search(conn, "unrelated question", ["d_2"], count=1)

    result = stats(conn, days=30)
    assert result["repeated_queries"] == [{"query": "how does X work", "count": 2}]


def test_window_boundary_excludes_events_older_than_the_requested_days(env):
    conn, cfg = env
    _search(conn, "ancient", [], count=0, ts=_old_ts())

    result = stats(conn, days=30)
    assert result["zero_hit"]["total"] == 0
    assert result["repeated_queries"] == []


def test_window_past_retention_is_flagged_as_truncated(env):
    """Events are purged after EVENT_RETENTION_DAYS; a window above that is
    silently reporting on data that no longer fully exists unless this says so."""
    conn, cfg = env
    assert stats(conn, days=120)["window_truncated"] is True
    assert stats(conn, days=30)["window_truncated"] is False


def test_tool_failed_counts_empty_when_no_events(env):
    """When there are no tool.failed events, the section reports zero."""
    conn, cfg = env
    result = stats(conn, days=30)
    assert result["tool_failed"]["count"] == 0
    assert result["tool_failed"]["rows"] == []


def test_tool_failed_groups_by_tool_and_phase(env):
    """tool.failed events are grouped by (tool, phase) and sorted by count descending."""
    conn, cfg = env
    # Emit two failure events for the same tool+phase
    emit(conn, "tool.failed", tool="foo", phase="handler", error="Something went wrong")
    emit(conn, "tool.failed", tool="foo", phase="handler", error="Another error")
    # Emit one failure for a different phase
    emit(conn, "tool.failed", tool="foo", phase="validate", error="Bad args")
    # Emit one failure for a different tool
    emit(conn, "tool.failed", tool="bar", phase="handler", error="Timeout")

    result = stats(conn, days=30)
    assert result["tool_failed"]["count"] == 4
    assert len(result["tool_failed"]["rows"]) == 3  # 3 unique (tool, phase) pairs
    # First row should be foo+handler with count=2
    assert result["tool_failed"]["rows"][0]["tool"] == "foo"
    assert result["tool_failed"]["rows"][0]["phase"] == "handler"
    assert result["tool_failed"]["rows"][0]["count"] == 2


def test_tool_failed_shows_most_recent_error(env):
    """Each row includes the most recent error text truncated to ~100 chars."""
    conn, cfg = env
    emit(conn, "tool.failed", tool="test", phase="handler", error="First error text")
    # Backdate this so the second one is more recent
    conn.execute("UPDATE events SET ts=? WHERE kind='tool.failed'",
                 (_old_ts(days=1),))
    conn.commit()
    emit(conn, "tool.failed", tool="test", phase="handler", error="Most recent error")

    result = stats(conn, days=30)
    assert len(result["tool_failed"]["rows"]) == 1
    assert result["tool_failed"]["rows"][0]["recent_error"] == "Most recent error"


def test_tool_failed_truncates_long_errors(env):
    """Error text longer than ~100 chars is truncated."""
    conn, cfg = env
    long_error = "x" * 150
    emit(conn, "tool.failed", tool="test", phase="handler", error=long_error)

    result = stats(conn, days=30)
    row = result["tool_failed"]["rows"][0]
    assert len(row["recent_error"]) <= 105  # Some reasonable truncation limit


def test_tool_failed_respects_window_boundary(env):
    """tool.failed events older than the window are excluded."""
    conn, cfg = env
    emit(conn, "tool.failed", tool="test", phase="handler", error="Old error")
    conn.execute("UPDATE events SET ts=? WHERE kind='tool.failed'",
                 (_old_ts(days=200),))
    conn.commit()

    result = stats(conn, days=30)
    assert result["tool_failed"]["count"] == 0
    assert result["tool_failed"]["rows"] == []


def test_tool_failed_drops_readonly_source_path_prefix(env):
    """Path-led refusal errors have their path replaced with basename."""
    conn, cfg = env
    error = "/home/someone/.notes/projects/a-very-long-slug/memory/reference_x.md belongs to source 'claude-memory'. Record the change in a native document instead."
    emit(conn, "tool.failed", tool="knowledge_update", phase="handler", error=error)

    result = stats(conn, days=30)
    row = result["tool_failed"]["rows"][0]
    assert "belongs to source" in row["recent_error"]
    assert "reference_x.md" in row["recent_error"]
    assert "/home/someone/.notes" not in row["recent_error"]


def test_tool_failed_cleans_pydantic_error_prefix(env):
    """Pydantic errors drop 'Error executing tool' and '1 validation error' prefixes, preserve field names."""
    conn, cfg = env
    error = "Error executing tool knowledge_write: 1 validation error for knowledge_writeArguments\nbody\n  Field required [type=missing, input_value=<stripped>, input_type=dict]"
    emit(conn, "tool.failed", tool="knowledge_write", phase="handler", error=error)

    result = stats(conn, days=30)
    row = result["tool_failed"]["rows"][0]
    assert "body" in row["recent_error"]
    assert "Field required" in row["recent_error"]
    assert "Error executing tool" not in row["recent_error"]
    assert "knowledge_writeArguments" not in row["recent_error"]


def test_tool_failed_keeps_a_slash_inside_a_word(env):
    """Only a path token loses its directories: 'update/append' in prose is not a path,
    and collapsing it would garble the very reason the display cleanup exists to keep."""
    conn, cfg = env
    emit(conn, "tool.failed", tool="knowledge_append", phase="handler",
         error="refused: update/append need a native document")

    row = stats(conn, days=30)["tool_failed"]["rows"][0]
    assert "update/append" in row["recent_error"]
