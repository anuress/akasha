"""`akasha housekeeping` is the retention pass's driver: nothing else bounds the events
table. It runs the pass when the cadence permits and --now forces one."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from akasha.cli import main
from akasha.config import load_config
from akasha.db import connect


@pytest.fixture
def conn(isolated_home):
    main(["init"])
    return connect(load_config().db_path)


def _old(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def _add_event(conn, id_, days):
    conn.execute("INSERT INTO events (id, ts, kind) VALUES (?, ?, 'x')", (id_, _old(days)))
    conn.commit()


def _ids(conn):
    return {r["id"] for r in conn.execute("SELECT id FROM events")}


def test_housekeeping_purges_out_of_window_events_and_reports(conn, capsys):
    _add_event(conn, "e_old", 91)
    _add_event(conn, "e_new", 1)
    assert main(["housekeeping"]) == 0
    assert "purged 1 events" in capsys.readouterr().out
    assert "e_old" not in _ids(conn) and "e_new" in _ids(conn)
    assert json.loads(conn.execute(
        "SELECT value FROM meta WHERE key = 'cleanup.last_run'").fetchone()["value"])


def test_housekeeping_skips_a_pass_still_inside_the_cadence(conn, capsys):
    main(["housekeeping"])
    capsys.readouterr()
    _add_event(conn, "e_old", 91)
    assert main(["housekeeping"]) == 0
    assert "not due" in capsys.readouterr().out
    assert "e_old" in _ids(conn)


def test_housekeeping_now_forces_the_pass(conn, capsys):
    main(["housekeeping"])
    capsys.readouterr()
    _add_event(conn, "e_old", 91)
    assert main(["housekeeping", "--now"]) == 0
    assert "purged 1 events" in capsys.readouterr().out
    assert "e_old" not in _ids(conn)
