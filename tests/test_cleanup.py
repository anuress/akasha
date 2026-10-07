import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from akasha.cleanup import last_run, purge_events, run, run_if_due
from akasha.config import load_config
from akasha.db import connect
from akasha.events import recent
from akasha.knowledge import EVENT_RETENTION_DAYS

REF = datetime(2026, 1, 10, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def env(tmp_path):
    return connect(tmp_path / "s.db"), load_config(tmp_path / "absent.toml")


def _ts(days_ago: float) -> str:
    return (REF - timedelta(days=days_ago)).isoformat()


def _event(conn, event_id, ts):
    conn.execute("INSERT INTO events (id, ts, kind, payload) VALUES (?, ?, 'a.b', '{}')",
                 (event_id, ts))
    conn.commit()


def test_events_older_than_the_window_are_removed_and_fresh_ones_spared(env):
    """Inside the window an event must survive, just outside it must go."""
    conn, _ = env
    _event(conn, "e_old", _ts(EVENT_RETENTION_DAYS + 1))
    _event(conn, "e_edge", _ts(EVENT_RETENTION_DAYS - 1))

    assert purge_events(conn, reference=REF) == 1
    ids = {r["id"] for r in conn.execute("SELECT id FROM events")}
    assert ids == {"e_edge"}


def test_the_window_is_the_one_stats_reports_against():
    """knowledge.stats and the purge must read the same constant, or the numbers a caller
    sees would describe events already deleted."""
    import akasha.cleanup as cleanup
    import akasha.knowledge as knowledge

    assert cleanup.EVENT_RETENTION_DAYS is knowledge.EVENT_RETENTION_DAYS


def test_run_reports_what_it_removed_and_records_when_it_ran(env):
    conn, cfg = env
    _event(conn, "e_old", _ts(EVENT_RETENTION_DAYS + 1))

    assert run(conn, reference=REF) == {"events": 1}
    ran = [json.loads(e["payload"]) for e in recent(conn) if e["kind"] == "cleanup.ran"]
    assert ran and ran[-1]["events"] == 1
    assert last_run(conn) is not None


def test_a_pass_inside_the_interval_is_skipped(env):
    """housekeeping_interval_min is the cadence: a second pass right after the first
    must not run, or a hook calling it would purge on every start."""
    conn, cfg = env
    assert run_if_due(conn, cfg) == {"events": 0}
    old = datetime.now(timezone.utc) - timedelta(days=EVENT_RETENTION_DAYS + 1)
    _event(conn, "e_old", old.isoformat())

    assert run_if_due(conn, cfg) is None
    assert conn.execute("SELECT COUNT(*) c FROM events WHERE id='e_old'").fetchone()["c"] == 1


def test_an_interval_of_zero_runs_every_time(env):
    conn, cfg = env
    cfg.housekeeping_interval_min = 0
    assert run_if_due(conn, cfg) is not None
    assert run_if_due(conn, cfg) is not None


def test_a_failing_purge_is_named_and_does_not_raise(env, monkeypatch):
    conn, cfg = env

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr("akasha.cleanup.purge_events", boom)
    assert run_if_due(conn, cfg) == {"events": 0}
    failed = [json.loads(e["payload"]) for e in recent(conn) if e["kind"] == "cleanup.failed"]
    assert failed and "disk full" in failed[0]["error"]


def test_a_failing_failure_record_still_does_not_raise(env, monkeypatch):
    """run_if_due runs from hooks: even recording the failure may fail, and that must not
    escape either."""
    conn, cfg = env

    def boom(*a, **k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr("akasha.cleanup.purge_events", boom)
    monkeypatch.setattr("akasha.cleanup.emit", boom)
    assert run_if_due(conn, cfg) == {"events": 0}


def test_a_failed_purge_is_rolled_back_before_recording_the_failure(env, monkeypatch):
    conn, cfg = env

    def half_done(*a, **k):
        conn.execute("BEGIN")
        conn.execute("INSERT INTO meta (key, value) VALUES ('half', 'done')")
        raise RuntimeError("disk full")

    monkeypatch.setattr("akasha.cleanup.purge_events", half_done)
    run_if_due(conn, cfg)
    assert conn.execute("SELECT 1 FROM meta WHERE key='half'").fetchone() is None


def test_an_unreadable_due_check_does_not_raise(env, monkeypatch):
    conn, cfg = env

    def boom(*a, **k):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr("akasha.cleanup._due", boom)
    assert run_if_due(conn, cfg) == {"events": 0}
