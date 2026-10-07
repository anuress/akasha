import json
import subprocess
import sys

import pytest

from akasha import install
from akasha.cli import main


@pytest.fixture
def home(isolated_home):
    assert main(["init"]) == 0
    return isolated_home


def _out(capsys):
    return capsys.readouterr().out


def _write(capsys, *argv):
    assert main(["knowledge", "write", *argv]) == 0
    return _out(capsys).strip()


# --- init ------------------------------------------------------------------------------

def test_init_stays_inside_the_patched_home(home):
    assert (home / ".akasha" / "config.toml").exists()
    assert (home / ".akasha" / "knowledge").is_dir()
    assert (home / ".akasha").stat().st_mode & 0o777 == 0o700


def test_init_dry_run_creates_nothing(isolated_home, capsys):
    assert main(["init", "--dry-run"]) == 0
    assert not (isolated_home / ".akasha").exists()
    assert "dry-run" in _out(capsys)


@pytest.fixture
def vendor_cli(monkeypatch):
    """Vendor CLIs are never run for real: every subprocess call is recorded."""
    calls = []

    def fake(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(install.subprocess, "run", fake)
    monkeypatch.setattr(install, "detect_vendors", lambda: ["claude"])
    return calls


def test_init_all_registers_the_server_the_hooks_and_the_permission(home, vendor_cli):
    assert main(["init", "--all"]) == 0
    assert any(c[:3] == ["claude", "mcp", "add"] for c in vendor_cli)
    settings = json.loads((home / ".claude" / "settings.json").read_text())
    assert "SessionStart" in settings["hooks"] and "PostToolUse" in settings["hooks"]
    assert "mcp__akasha" in settings["permissions"]["allow"]


def test_init_all_no_allow_leaves_permissions_alone(home, vendor_cli):
    assert main(["init", "--all", "--no-allow"]) == 0
    settings = json.loads((home / ".claude" / "settings.json").read_text())
    assert "permissions" not in settings


def test_init_all_dry_run_runs_nothing_and_writes_nothing(home, vendor_cli, capsys):
    assert main(["init", "--all", "--dry-run"]) == 0
    assert vendor_cli == []
    assert not (home / ".claude").exists()
    assert "would" in _out(capsys)


def test_init_without_all_only_mentions_detected_vendors(home, vendor_cli, capsys):
    capsys.readouterr()
    assert main(["init"]) == 0
    assert vendor_cli == [] and "--all" in _out(capsys)


# --- knowledge -------------------------------------------------------------------------

def test_write_then_search_roundtrip(home, capsys):
    _write(capsys, "--title", "Limit sweep", "--body", "## Result\nlimit 30 wins", "--repo", "sample-repo")
    assert main(["knowledge", "search", "limit sweep", "--all"]) == 0
    assert "limit 30 wins" in _out(capsys)


def test_search_json_uses_the_caller_shape(home, capsys):
    _write(capsys, "--title", "T", "--body", "## A\njsonable", "--repo", "r")
    main(["knowledge", "search", "jsonable", "--all", "--json"])
    payload = json.loads(_out(capsys))
    assert payload[0]["text"].strip().endswith("jsonable")
    assert "path" not in payload[0] and "relaxed" not in payload[0]


def test_search_works_on_a_non_trunk_branch(home, capsys, monkeypatch):
    """A branch name is not a feature: deriving one with no documents filtered every query
    to nothing on any feature branch."""
    monkeypatch.setattr("akasha.gitctx.git_context",
                        lambda cfg, cwd: ("my-repo", "some-branch-name"))
    _write(capsys, "--title", "T", "--body", "## A\nfindable", "--repo", "my-repo")
    main(["knowledge", "search", "findable", "--json"])
    assert json.loads(_out(capsys))


def test_an_explicit_feature_is_still_honoured(home, capsys):
    _write(capsys, "--title", "T", "--body", "## A\nfindable", "--repo", "r",
           "--feature", "real-feature")
    main(["knowledge", "search", "findable", "--feature", "no-such-feature", "--json"])
    assert json.loads(_out(capsys)) == []


def test_terminal_search_output_truncates_a_long_body(home, capsys):
    """Bounds a human's scrollback; --json and the MCP tool carry the full text."""
    _write(capsys, "--title", "wildebeest", "--body", "## R\n" + "x" * 900, "--repo", "r")
    main(["knowledge", "search", "wildebeest", "--all"])
    out = _out(capsys)
    assert "x" * 500 not in out


def test_get_prints_the_document(home, capsys):
    doc_id = _write(capsys, "--title", "A finding", "--body", "## Why\nthe guard was wrong",
                    "--repo", "r")
    assert main(["knowledge", "get", doc_id]) == 0
    assert "the guard was wrong" in _out(capsys)


def test_get_unknown_id_is_a_message_not_a_traceback(home, capsys):
    assert main(["knowledge", "get", "k_nope"]) == 1
    assert "unknown document" in _out(capsys)


def test_write_derives_the_repo_like_the_tool_does(home, capsys, monkeypatch):
    monkeypatch.setattr("akasha.gitctx.git_context", lambda cfg, cwd: ("sample-repo", None))
    monkeypatch.setattr("akasha.cli.resolve_repo", lambda cfg, cwd, explicit: ("sample-repo", "cwd"))
    doc_id = _write(capsys, "--title", "T", "--body", "## A\nx")
    from akasha.config import load_config
    from akasha.db import connect

    row = connect(load_config().db_path).execute(
        "SELECT repo FROM documents WHERE id=?", (doc_id,)).fetchone()
    assert row["repo"] == "sample-repo"


def test_write_reads_the_body_from_stdin(home, capsys, monkeypatch):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("## S\nfrom stdin\n"))
    _write(capsys, "--title", "T", "--body", "-", "--repo", "r")
    main(["knowledge", "search", "from stdin", "--all"])
    assert "from stdin" in _out(capsys)


def test_write_without_a_body_says_what_is_needed(home, capsys):
    assert main(["knowledge", "write", "--title", "T"]) == 1
    assert "--body" in _out(capsys)


def test_write_can_supersede_from_the_cli(home, capsys):
    old = _write(capsys, "--title", "Old", "--body", "## C\nexpires in 15m", "--repo", "r")
    _write(capsys, "--title", "New", "--body", "## C\nexpires in 5m", "--repo", "r",
           "--supersedes", old)
    main(["knowledge", "get", old])
    assert "archived" in _out(capsys)


def test_an_unknown_kind_is_refused_with_the_allowed_list(home, capsys):
    assert main(["knowledge", "write", "--title", "T", "--body", "b", "--kind", "notes"]) == 1
    assert "finding" in _out(capsys)


def test_update_append_and_archive(home, capsys):
    doc_id = _write(capsys, "--title", "T", "--body", "## A\nalpha", "--repo", "r")
    assert main(["knowledge", "update", doc_id, "--title", "T2"]) == 0
    assert main(["knowledge", "append", doc_id, "--body", "beta-token"]) == 0
    assert main(["knowledge", "archive", doc_id]) == 0
    capsys.readouterr()
    main(["knowledge", "search", "beta-token", "--all", "--include-archived"])
    assert "beta-token" in _out(capsys)


def test_rm_restore_and_purge_need_confirmation(home, capsys):
    doc_id = _write(capsys, "--title", "T", "--body", "## A\nx", "--repo", "r")
    assert main(["knowledge", "purge"]) == 1
    assert "--yes" in _out(capsys)
    assert main(["knowledge", "rm", doc_id]) == 0
    assert main(["knowledge", "restore", doc_id]) == 0
    assert main(["knowledge", "rm", doc_id]) == 0
    assert main(["knowledge", "purge", "--yes"]) == 0
    assert "purged 1" in _out(capsys)


def test_a_convention_over_the_budget_warns_on_stderr(home, capsys):
    from akasha.hooks import CONVENTION_BUDGET

    body = "## Rules\n" + "rule. " * (CONVENTION_BUDGET + 100)
    assert main(["knowledge", "write", "--kind", "convention", "--title", "Big rules",
                 "--body", body, "--repo", "sample-repo"]) == 0
    err = capsys.readouterr().err
    assert "Big rules" in err and str(CONVENTION_BUDGET) in err


def test_a_convention_that_fits_is_silent(home, capsys):
    main(["knowledge", "write", "--kind", "convention", "--title", "Small",
          "--body", "## Rules\nkeep", "--repo", "sample-repo"])
    assert capsys.readouterr().err == ""


def test_stale_defaults_to_the_configured_age(home, capsys):
    from akasha.config import load_config
    from akasha.db import connect

    doc_id = _write(capsys, "--title", "T", "--body", "## A\nx", "--repo", "r")
    conn = connect(load_config().db_path)
    from datetime import datetime, timedelta, timezone

    sixty_days_ago = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
    conn.execute("UPDATE documents SET created_at=? WHERE id=?", (sixty_days_ago, doc_id))
    conn.commit()
    capsys.readouterr()
    main(["knowledge", "stale"])
    assert "0 stale" in _out(capsys)
    assert main(["config", "set", "knowledge.stale_after_days", "30"]) == 0
    capsys.readouterr()
    main(["knowledge", "stale"])
    assert "1 stale" in _out(capsys)


def test_stats_reports_searches_and_failures(home, capsys):
    main(["knowledge", "search", "nothing here", "--all"])
    capsys.readouterr()
    assert main(["knowledge", "stats"]) == 0
    assert "zero-hit rate" in _out(capsys)


def test_fsck_in_full_records_the_error_count(home, capsys):
    from akasha import fsck
    from akasha.config import load_config
    from akasha.db import connect

    assert main(["knowledge", "fsck"]) == 0
    assert fsck.cached_error_count(connect(load_config().db_path)) == 0


def test_fsck_json_is_the_core_report(home, capsys):
    capsys.readouterr()
    main(["knowledge", "fsck", "--json"])
    assert set(json.loads(_out(capsys))) == {"total", "counts", "findings", "withheld"}


def test_timeline_and_related(home, capsys):
    from akasha.config import load_config
    from akasha.db import connect
    from akasha.knowledge import write

    cfg = load_config()
    conn = connect(cfg.db_path)
    c = write(conn, cfg, "C", "## A\nend\n", repo="r")
    b = write(conn, cfg, "B", f"## A\n[[{c}]]\n", repo="r")
    capsys.readouterr()
    assert main(["knowledge", "timeline", "--repo", "r"]) == 0
    assert b in _out(capsys)
    assert main(["knowledge", "related", b]) == 0
    assert f"outbound  {c}  C" in _out(capsys)


# --- feature ---------------------------------------------------------------------------

def test_feature_show_alias_and_list(home, capsys):
    _write(capsys, "--title", "T", "--body", "## A\nx", "--repo", "r",
           "--feature", "summer-reading-program")
    assert main(["feature", "alias", "summer-reading-program", "summer-reading"]) == 0
    capsys.readouterr()
    assert main(["feature", "show", "summer-reading"]) == 0
    assert "docs: 1" in _out(capsys)
    assert main(["feature", "list"]) == 0
    assert "summer-reading-program" in _out(capsys)


def test_feature_unknown_slug_fails_with_a_message(home, capsys):
    assert main(["feature", "alias", "never-seen", "ns"]) == 1
    assert "unknown feature" in _out(capsys)
    assert main(["feature", "show", "never-seen"]) == 1


# --- index -----------------------------------------------------------------------------

def test_index_picks_up_a_source_added_with_index_add(home, capsys, tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.md").write_text("## A\ncatalog cache was off\n")
    assert main(["index", "add", str(notes), "--source", "notes"]) == 0
    capsys.readouterr()
    assert main(["index"]) == 0
    assert "indexed 1" in _out(capsys)
    main(["knowledge", "search", "catalog cache", "--all"])
    assert "a.md" in _out(capsys)


def test_index_add_a_missing_directory_is_a_message(home, capsys, tmp_path):
    assert main(["index", "add", str(tmp_path / "nope")]) == 1
    assert "does not exist" in _out(capsys)


def test_index_force_reprocesses_unchanged_files_and_refreshes_the_fsck_count(
        home, capsys, monkeypatch):
    calls = []
    real = __import__("akasha.index", fromlist=["index_all"]).index_all
    monkeypatch.setattr("akasha.cli.index_all",
                        lambda conn, cfg, force=False: calls.append(force) or real(conn, cfg, force))
    from akasha import fsck

    recorded = []
    real_record = fsck.record_errors
    monkeypatch.setattr(fsck, "record_errors", lambda c, n: recorded.append(n) or real_record(c, n))
    assert main(["index", "--force"]) == 0
    assert main(["index"]) == 0
    assert calls == [True, False] and len(recorded) == 2


def test_index_forget_removes_a_path(home, capsys, tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    doc = notes / "a.md"
    doc.write_text("## A\ncatalog\n")
    main(["index", "add", str(notes)])
    main(["index"])
    capsys.readouterr()
    assert main(["index", "--forget", str(doc)]) == 0
    assert "forgot 1" in _out(capsys)


# --- config ----------------------------------------------------------------------------

def test_config_set_get_unset(home, capsys):
    assert main(["config", "set", "security.scan_secrets", "false"]) == 0
    capsys.readouterr()
    assert main(["config", "get", "security.scan_secrets"]) == 0
    assert _out(capsys).strip() == "False"
    assert main(["config", "unset", "security.scan_secrets"]) == 0
    assert main(["config", "get", "security.scan_secrets"]) == 1


def test_config_get_a_whole_section(home, capsys):
    assert main(["config", "set", "housekeeping.interval_min", "30"]) == 0
    capsys.readouterr()
    assert main(["config", "get", "housekeeping"]) == 0
    assert "housekeeping.interval_min = 30" in _out(capsys)


def test_config_key_without_a_section_is_a_message(home, capsys):
    assert main(["config", "set", "novalue", "1"]) == 1
    assert "section.key" in _out(capsys)


# --- doctor ----------------------------------------------------------------------------

def test_doctor_prints_findings_and_exits_by_severity(home, capsys, monkeypatch):
    monkeypatch.setattr("akasha.doctor.detect_vendors", lambda: [])
    capsys.readouterr()
    assert main(["doctor"]) == 0
    out = _out(capsys)
    assert "knowledge" in out and len(out.splitlines()) < 20

    from akasha.config import load_config

    monkeypatch.setattr("akasha.vectors.available", lambda conn: False)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    assert main(["config", "set", "embeddings.provider", "model2vec"]) == 0
    capsys.readouterr()
    assert main(["doctor"]) == 1
    assert "sqlite-vec did not load" in _out(capsys)


# --- shape -----------------------------------------------------------------------------

def test_unknown_command_exits_nonzero(home):
    with pytest.raises(SystemExit):
        main(["nonsense"])


def test_python_dash_m_runs_the_cli(tmp_path):
    done = subprocess.run([sys.executable, "-m", "akasha", "--help"], capture_output=True,
                          text=True, cwd=tmp_path, timeout=60)
    assert done.returncode == 0 and "knowledge" in done.stdout


# --- failures and passthroughs ---------------------------------------------------------

def test_get_of_a_document_whose_file_is_gone_is_one_line_not_a_traceback(home, capsys):
    doc_id = _write(capsys, "--title", "T", "--body", "## H\nb", "--repo", "r")
    path = next((home / ".akasha" / "knowledge").rglob("*.md"))
    path.unlink()
    assert main(["knowledge", "get", doc_id]) != 0
    out = _out(capsys)
    assert len(out.strip().splitlines()) == 1 and "Traceback" not in out


def test_update_passes_a_match_and_replacement_through(home, capsys):
    doc_id = _write(capsys, "--title", "T", "--body", "## H\nthe cat sat\n", "--repo", "r")
    assert main(["knowledge", "update", doc_id, "--match", "cat", "--replacement", "dog"]) == 0
    capsys.readouterr()
    assert main(["knowledge", "get", doc_id]) == 0
    assert "the dog sat" in _out(capsys)


def test_update_passes_status_and_expected_updated_through(home, capsys):
    doc_id = _write(capsys, "--title", "T", "--body", "## H\nb\n", "--repo", "r")
    assert main(["knowledge", "update", doc_id, "--status", "archived",
                 "--expected-updated", "1999-01-01"]) == 1
    assert "changed since you read it" in _out(capsys)


def test_append_passes_a_heading_through(home, capsys):
    doc_id = _write(capsys, "--title", "T", "--body", "## H\nb\n", "--repo", "r")
    assert main(["knowledge", "append", doc_id, "--body", "more", "--heading", "## Custom"]) == 0
    capsys.readouterr()
    assert main(["knowledge", "get", doc_id]) == 0
    assert "## Custom" in _out(capsys)


def test_get_pages_with_offset_and_max_chars(home, capsys):
    doc_id = _write(capsys, "--title", "T", "--body", "## H\n0123456789\n", "--repo", "r")
    assert main(["knowledge", "get", doc_id, "--offset", "0", "--max-chars", "10"]) == 0
    first = capsys.readouterr()
    assert len(first.out.rstrip("\n")) == 10 and "withheld" in first.err


def test_search_max_chars_caps_the_listing(home, capsys):
    _write(capsys, "--title", "T", "--body", "## H\n" + "word " * 100, "--repo", "r")
    assert main(["knowledge", "search", "word", "--all", "--max-chars", "20"]) == 0
    text_line = [l for l in _out(capsys).splitlines() if l.startswith("    ") and "word" in l]
    assert len(text_line[0].strip()) <= 20


def test_index_add_writes_the_repo(home, tmp_path):
    import tomllib

    notes = tmp_path / "notes"
    notes.mkdir()
    assert main(["index", "add", str(notes), "--repo", "my-repo"]) == 0
    roots = tomllib.loads((home / ".akasha" / "config.toml").read_text())["index"]
    assert roots[-1]["repo"] == "my-repo"


def test_hook_survives_non_utf8_stdin(home, monkeypatch):
    import io

    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(b"\xff\xfe\x00"), encoding="utf-8"))
    assert main(["hook", "post-tool"]) == 0


def test_init_creates_the_config_private_from_the_start(isolated_home, monkeypatch):
    """No world-readable window: the mode is set at creation, not by a later chmod."""
    import os

    os.umask(0)
    monkeypatch.setattr(os, "chmod", lambda *a, **k: None)
    assert main(["init"]) == 0
    assert (isolated_home / ".akasha" / "config.toml").stat().st_mode & 0o777 == 0o600


def test_append_takes_body_from_a_file_and_refuses_a_positional(home, capsys, tmp_path):
    """append reads its text like write does; the old positional is gone."""
    doc_id = _write(capsys, "--title", "T", "--body", "## H\nb\n", "--repo", "r")
    f = tmp_path / "sec.md"
    f.write_text("from-file-token")
    assert main(["knowledge", "append", doc_id, "--body-file", str(f)]) == 0
    capsys.readouterr()
    main(["knowledge", "get", doc_id])
    assert "from-file-token" in _out(capsys)
    with pytest.raises(SystemExit):
        main(["knowledge", "append", doc_id, "positional"])
    assert main(["knowledge", "append", doc_id]) == 1


# --- init chooses the embeddings provider ------------------------------------------------

def _config_text(home):
    return (home / ".akasha" / "config.toml").read_text()


def test_init_enables_dense_search_when_the_extra_is_installed(isolated_home, monkeypatch, capsys):
    from akasha import vectors

    monkeypatch.setattr(vectors, "extra_installed", lambda: True)
    assert main(["init"]) == 0
    assert 'provider = "model2vec"' in _config_text(isolated_home)
    assert "dense search is off" not in _out(capsys)


def test_init_without_the_extra_stays_keyword_only_and_says_so(isolated_home, monkeypatch, capsys):
    from akasha import vectors

    monkeypatch.setattr(vectors, "extra_installed", lambda: False)
    assert main(["init"]) == 0
    assert "[embeddings]" not in _config_text(isolated_home)
    out = _out(capsys)
    assert "dense search is off" in out and "akasha-mcp[vectors]" in out


def test_init_never_changes_an_existing_config(isolated_home, monkeypatch):
    from akasha import vectors

    monkeypatch.setattr(vectors, "extra_installed", lambda: False)
    assert main(["init"]) == 0
    before = _config_text(isolated_home)
    monkeypatch.setattr(vectors, "extra_installed", lambda: True)
    assert main(["init"]) == 0
    assert _config_text(isolated_home) == before


def test_init_dry_run_says_which_provider_it_would_choose(isolated_home, monkeypatch, capsys):
    from akasha import vectors

    monkeypatch.setattr(vectors, "extra_installed", lambda: True)
    assert main(["init", "--dry-run"]) == 0
    assert "model2vec" in _out(capsys)
