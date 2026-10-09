import json
import os

import pytest

from akasha import doctor, install, vectors
from akasha.config import IndexRoot, load_config
from akasha.db import connect


@pytest.fixture
def env(tmp_path, isolated_home):
    """A scratch HOME with a 0700 data home, a config pointing inside it."""
    data = isolated_home / ".akasha"
    data.mkdir(mode=0o700)
    os.chmod(data, 0o700)
    cfg = load_config(data / "config.toml")
    cfg.db_path = data / "akasha.db"
    cfg.knowledge_dir = data / "knowledge"
    cfg.knowledge_dir.mkdir()
    cfg.index_roots = []
    return connect(cfg.db_path), cfg, isolated_home


def _kinds(findings, severity=None):
    return [f.kind for f in findings if severity is None or f.severity == severity]


def _run(env):
    conn, cfg, home = env
    return doctor.check(cfg, conn, home=home)


@pytest.fixture(autouse=True)
def no_vendors(monkeypatch):
    """Which agent CLIs happen to be installed must not decide a test's outcome."""
    monkeypatch.setattr("akasha.doctor.detect_vendors", lambda: [])


def test_a_fresh_install_is_healthy(env):
    findings = _run(env)
    assert doctor.exit_code(findings) == 0
    assert "knowledge" in _kinds(findings)


def test_a_missing_data_home_is_a_failure(env):
    conn, cfg, home = env
    findings = doctor.check(cfg, conn, home=home / "elsewhere")
    assert "permissions" in _kinds(findings, "error")


def test_loose_permissions_are_a_failure(env):
    conn, cfg, home = env
    os.chmod(home / ".akasha", 0o755)
    assert "permissions" in _kinds(_run(env), "error")


# --- sources -------------------------------------------------------------------------

def test_a_root_with_nothing_to_index_is_a_note_not_a_failure(env, tmp_path):
    """A health check that can never go green teaches you to stop reading it."""
    conn, cfg, _ = env
    empty = tmp_path / "notes-not-written-yet"
    empty.mkdir()
    cfg.index_roots = [IndexRoot(path=str(empty), source="notes")]
    findings = _run(env)
    assert [f for f in findings if "nothing to index" in f.detail][0].severity == "note"
    assert doctor.exit_code(findings) == 0


def test_a_root_whose_files_were_all_skipped_is_a_failure(env, tmp_path):
    """Files present and none indexed is the include-pattern fault; it stays loud."""
    conn, cfg, _ = env
    root = tmp_path / "has-notes"
    root.mkdir()
    (root / "note.md").write_text("## A\nbody\n")
    cfg.index_roots = [IndexRoot(path=str(root), source="notes")]
    findings = _run(env)
    assert any("indexed nothing" in f.detail for f in findings)
    assert doctor.exit_code(findings) != 0


def test_a_root_that_does_not_exist_is_a_failure(env, tmp_path):
    conn, cfg, _ = env
    cfg.index_roots = [IndexRoot(path=str(tmp_path / "typo"), source="notes")]
    findings = _run(env)
    assert any("does not exist" in f.detail for f in findings)
    assert doctor.exit_code(findings) != 0


# --- conventions ---------------------------------------------------------------------

def test_a_convention_longer_than_the_budget_is_named(env):
    from akasha.hooks import CONVENTION_BUDGET
    from akasha.knowledge import write

    conn, cfg, _ = env
    write(conn, cfg, "House rules", "## R\n" + "rule. " * (CONVENTION_BUDGET + 100),
          repo="sample-repo", kind="convention")
    findings = _run(env)
    assert any("House rules" in f.detail and str(CONVENTION_BUDGET) in f.detail
               for f in findings)
    assert doctor.exit_code(findings) != 0


# --- vectors -------------------------------------------------------------------------

def _enable(cfg):
    cfg.embeddings_provider = "model2vec"


def _vec(findings):
    return [f for f in findings if f.kind == "vectors"][0]


def test_vectors_off_is_said_not_silent(env):
    finding = _vec(_run(env))
    assert finding.severity == "note" and "lexical only" in finding.detail


def test_a_missing_vectors_extra_is_an_error_naming_the_fix(env, monkeypatch):
    """Provider configured but the extra not installed: search silently drops to lexical.
    That must be a red line that says how to fix it."""
    conn, cfg, _ = env
    _enable(cfg)
    real = vectors.importlib.util.find_spec
    monkeypatch.setattr(vectors.importlib.util, "find_spec",
                        lambda name, *a: None if name == "sqlite_vec" else real(name, *a))
    finding = _vec(_run(env))
    assert finding.severity == "error"
    assert "sqlite_vec" in finding.detail and "[vectors]" in finding.detail
    assert doctor.exit_code([finding]) == 1


def test_a_dead_encoder_is_an_error_carrying_its_message(env, monkeypatch):
    conn, cfg, _ = env
    _enable(cfg)
    monkeypatch.setattr(vectors, "_encoder", lambda: (_ for _ in ()).throw(
        ImportError("No module named 'rich.table'")))
    if not vectors.available(conn):
        pytest.skip("sqlite-vec not loadable here")
    finding = _vec(_run(env))
    assert finding.severity == "error" and "rich.table" in finding.detail


def _stub_encode(texts):
    import numpy as np

    return np.ones((len(texts), 8), dtype=np.float32)


def _seed_vector(conn, cfg):
    import numpy as np

    if not vectors.available(conn):
        pytest.skip("sqlite-vec not loadable here")
    vectors.ensure_table(conn, dim=8)
    conn.execute("INSERT INTO vec_chunks(chunk_id, repo, embedding) VALUES (?, ?, ?)",
                 ("c1", "r", np.zeros(8, dtype=np.float32).tobytes()))
    conn.execute("INSERT INTO meta (key, value) VALUES (?, ?)",
                 (vectors.SIGNATURE_KEY, vectors._signature(cfg, vectors.DIM)))
    conn.commit()


def test_a_working_encoder_reports_count_and_signature(env, monkeypatch):
    conn, cfg, _ = env
    _enable(cfg)
    _seed_vector(conn, cfg)
    monkeypatch.setattr(vectors, "_encoder", lambda: _stub_encode)
    finding = _vec(_run(env))
    assert finding.severity == "info"
    assert "1 vectors" in finding.detail and "model2vec|" in finding.detail


def test_vectors_built_by_another_model_are_flagged_for_reindex(env, monkeypatch):
    conn, cfg, _ = env
    _enable(cfg)
    _seed_vector(conn, cfg)
    conn.execute("UPDATE meta SET value='other|model|8' WHERE key=?", (vectors.SIGNATURE_KEY,))
    conn.commit()
    monkeypatch.setattr(vectors, "_encoder", lambda: _stub_encode)
    finding = _vec(_run(env))
    assert finding.severity == "note" and "akasha index" in finding.detail


# --- integrations --------------------------------------------------------------------

def test_an_unregistered_cli_is_a_note(env, monkeypatch):
    monkeypatch.setattr("akasha.doctor.detect_vendors", lambda: ["claude"])
    findings = _run(env)
    assert any(f.kind == "mcp" and "claude" in f.detail and f.severity == "note"
               for f in findings)


def test_a_registered_cli_is_reported_once_and_healthy(env, monkeypatch):
    _, _, home = env
    monkeypatch.setattr("akasha.doctor.detect_vendors", lambda: ["claude"])
    (home / ".claude.json").write_text(json.dumps({"mcpServers": {"akasha": {}}}))
    install.install_hooks("claude", home)
    findings = _run(env)
    assert [f.detail for f in findings if f.kind == "mcp"] == ["registered: claude"]
    assert "hook" not in _kinds(findings)
    assert doctor.exit_code(findings) == 0


def test_a_hook_without_defer_refresh_is_an_error(env, monkeypatch):
    _, _, home = env
    monkeypatch.setattr("akasha.doctor.detect_vendors", lambda: ["claude"])
    path = home / ".claude" / "settings.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"hooks": {"SessionStart": [{"matcher": "*", "hooks": [
        {"type": "command", "command": install.HOOK_COMMAND}]}]}}))
    findings = _run(env)
    assert any(f.kind == "hook" and "--defer-refresh" in f.detail and f.severity == "error"
               for f in findings)


def test_a_post_tool_hook_blind_to_recording_tools_is_a_note(env, monkeypatch):
    """An install from before the write nudge still reindexes edits, but never sees a
    write, so it would nudge sessions that did record something."""
    _, _, home = env
    monkeypatch.setattr("akasha.doctor.detect_vendors", lambda: ["claude"])
    install.install_hooks("claude", home)
    path = home / ".claude" / "settings.json"
    data = json.loads(path.read_text())
    data["hooks"]["PostToolUse"][0]["matcher"] = "Write|Edit|MultiEdit"
    path.write_text(json.dumps(data))
    notes = [f for f in _run(env) if f.kind == "hook"]
    assert len(notes) == 1 and notes[0].severity == "note"
    assert "akasha init" in notes[0].detail


def test_missing_hooks_are_notes(env, monkeypatch):
    monkeypatch.setattr("akasha.doctor.detect_vendors", lambda: ["gemini"])
    notes = [f for f in _run(env) if f.kind == "hook"]
    assert len(notes) == 2 and all(f.severity == "note" for f in notes)
    assert any("post-tool" in f.detail for f in notes)


def test_an_unsupported_cli_on_path_is_never_checked(env, monkeypatch):
    """An agent CLI with no entry in the vendor table adds no line."""
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/othercli" if n == "othercli" else None)
    monkeypatch.setattr("akasha.doctor.detect_vendors", install.detect_vendors)
    assert not [f for f in _run(env) if f.kind in ("mcp", "hook")]


# --- fsck and failures ---------------------------------------------------------------

def test_fsck_never_run_is_a_note_and_errors_fail(env):
    from akasha import fsck

    conn, _, _ = env
    assert [f.severity for f in _run(env) if f.kind == "fsck"] == ["note"]
    fsck.record_errors(conn, 2)
    findings = _run(env)
    assert [f.severity for f in findings if f.kind == "fsck"] == ["error"]
    assert doctor.exit_code(findings) != 0
    fsck.record_errors(conn, 0)
    assert "fsck" not in _kinds(_run(env), "error")


def test_doctor_does_not_run_fsck(env, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("fsck ran")

    monkeypatch.setattr("akasha.fsck.check", boom)
    _run(env)


def test_recent_tool_failures_are_a_note(env):
    from akasha.events import emit

    conn, _, _ = env
    emit(conn, "tool.failed", tool="knowledge_get")
    findings = _run(env)
    assert any(f.kind == "tool_failures" for f in findings)
    assert doctor.exit_code(findings) == 0


def test_the_report_stays_short_on_a_healthy_install(env):
    assert len(_run(env)) <= 4


# --- per-root counts -----------------------------------------------------------------

def _doc(conn, path):
    conn.execute("INSERT INTO documents (id, source, path, created_at, updated_at)"
                 " VALUES (?, 'notes', ?, 't', 't')", (f"d_{path}", str(path)))
    conn.commit()


def _root_counts(findings):
    return {f.detail.split(" ")[1].rstrip(":"): f.detail for f in findings if f.kind == "index_root"}


def test_a_root_does_not_count_a_sibling_sharing_its_prefix(env, tmp_path):
    """/a/b must not claim documents under /a/bc."""
    conn, cfg, _ = env
    short, long = tmp_path / "a" / "b", tmp_path / "a" / "bc"
    for root in (short, long):
        root.mkdir(parents=True)
    _doc(conn, long / "n.md")
    _doc(conn, long / "m.md")
    _doc(conn, short / "n.md")
    cfg.index_roots = [IndexRoot(path=str(short), source="notes")]
    assert "1 docs" in _root_counts(_run(env))[str(short)]


def test_a_root_containing_like_wildcards_counts_only_its_own_documents(env, tmp_path):
    conn, cfg, _ = env
    odd, other = tmp_path / "x%y_z", tmp_path / "xAAyBz"
    for root in (odd, other):
        root.mkdir()
        _doc(conn, root / "n.md")
    cfg.index_roots = [IndexRoot(path=str(odd), source="notes")]
    assert "1 docs" in _root_counts(_run(env))[str(odd)]


# --- isolated checks -----------------------------------------------------------------

def _boom(*a, **k):
    raise RuntimeError("check exploded")


@pytest.mark.parametrize("target", ["_vectors", "_sources", "_integrations",
                                    "conventions_over_budget", "check_permissions",
                                    "cached_error_count"])
def test_a_check_that_raises_becomes_an_unchecked_finding(env, monkeypatch, target):
    """One broken check must not hide every other result."""
    monkeypatch.setattr(doctor, target, _boom)
    findings = _run(env)
    unchecked = [f for f in findings if "unchecked" in f.detail]
    assert unchecked and "check exploded" in unchecked[0].detail
    assert unchecked[0].severity == "error"
    assert "knowledge" in _kinds(findings)


def test_a_database_error_in_one_check_is_isolated(env):
    conn, cfg, _ = env
    conn.execute("DROP TABLE events")
    findings = _run(env)
    assert any(f.kind == "tool_failures" and "unchecked" in f.detail for f in findings)
    assert "knowledge" in _kinds(findings)


def test_an_unreadable_data_home_is_unchecked_not_fatal(env, monkeypatch):
    monkeypatch.setattr("pathlib.Path.stat", _boom)
    findings = _run(env)
    assert any("unchecked" in f.detail for f in findings)


def test_a_corrupt_fsck_meta_value_reads_as_never_run(env):
    from akasha import fsck

    conn, _, _ = env
    conn.execute("INSERT INTO meta (key, value) VALUES (?, 'not-a-number')", (fsck.ERRORS_KEY,))
    assert fsck.cached_error_count(conn) is None
    assert [f.severity for f in _run(env) if f.kind == "fsck"] == ["note"]


# --- config and integration config ---------------------------------------------------

def test_an_unparsable_config_is_an_error(env, tmp_path):
    conn, cfg, home = env
    bad = tmp_path / "config.toml"
    bad.write_text("[paths\nbroken")
    findings = doctor.check(cfg, conn, home=home, config_path=bad)
    assert any(f.kind == "config" and f.severity == "error" for f in findings)


def test_an_unreadable_config_is_an_error(env, tmp_path):
    conn, cfg, home = env
    bad = tmp_path / "config.toml"
    bad.write_bytes(b"\xff\xfe")
    findings = doctor.check(cfg, conn, home=home, config_path=bad)
    assert any(f.kind == "config" and f.severity == "error" for f in findings)


def test_an_unparsable_vendor_config_is_an_error_not_a_missing_registration(env, monkeypatch):
    _, _, home = env
    monkeypatch.setattr("akasha.doctor.detect_vendors", lambda: ["claude"])
    (home / ".claude.json").write_text("{not json")
    findings = _run(env)
    assert any(f.kind == "mcp" and f.severity == "error" and "not valid JSON" in f.detail
               for f in findings)
    assert not any("not registered" in f.detail for f in findings)


def test_unparsable_hook_settings_are_an_error(env, monkeypatch):
    _, _, home = env
    monkeypatch.setattr("akasha.doctor.detect_vendors", lambda: ["claude"])
    path = home / ".claude" / "settings.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    findings = _run(env)
    assert any(f.kind == "hook" and f.severity == "error" for f in findings)
