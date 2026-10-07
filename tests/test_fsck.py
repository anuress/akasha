import pytest

from akasha.config import load_config
from akasha.db import connect
from akasha.fsck import check
from akasha.knowledge import write


@pytest.fixture
def env(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")
    cfg.knowledge_dir = tmp_path / "knowledge"
    cfg.knowledge_dir.mkdir()
    cfg.index_roots = []
    return connect(tmp_path / "s.db"), cfg


def _kinds(findings):
    return {f.kind for f in findings}


def _repo(env, tmp_path, name="r"):
    """A configured repo root with a real file to anchor against."""
    conn, cfg = env
    root = tmp_path / name
    (root / "src").mkdir(parents=True)
    (root / "src" / "util.py").write_text("def u():\n    pass\n")
    cfg.projects = {name: {"path": str(root)}}
    return conn, cfg, root


def test_a_clean_base_reports_nothing(env):
    conn, cfg = env
    write(conn, cfg, "Only doc", "## A\ncontent here\n", repo="r")
    assert check(conn, cfg) == []


def test_duplicate_titles_are_flagged(env):
    conn, cfg = env
    write(conn, cfg, "Catalog cache miss", "## A\nfirst take\n", repo="r")
    write(conn, cfg, "Catalog cache miss", "## A\nsecond take\n", repo="r")
    assert "duplicate_title" in _kinds(check(conn, cfg))


def test_dangling_links_are_flagged(env):
    conn, cfg = env
    write(conn, cfg, "Source", "## A\nsee [[k_does_not_exist]]\n", repo="r")
    findings = check(conn, cfg)
    assert "dangling_link" in _kinds(findings)
    assert any("k_does_not_exist" in f.detail for f in findings)


def test_an_archived_documents_dangling_link_is_not_flagged(env):
    """An archived document is hidden from default search on purpose — fsck reporting
    its dangling link forever (with no way to silence it short of deleting the doc)
    defeats the point of archiving. Only active documents' links are worth reporting."""
    conn, cfg = env
    doc_id = write(conn, cfg, "Source", "## A\nsee [[k_does_not_exist]]\n", repo="r")
    conn.execute("UPDATE documents SET status='archived' WHERE id=?", (doc_id,))
    conn.commit()
    assert "dangling_link" not in _kinds(check(conn, cfg))


def test_supersede_pointing_at_a_missing_document_is_flagged(env):
    conn, cfg = env
    doc_id = write(conn, cfg, "New", "## A\nnew conclusion\n", repo="r")
    conn.execute("UPDATE documents SET supersedes=? WHERE id=?", ("k_missing", doc_id))
    conn.commit()
    assert "broken_supersede" in _kinds(check(conn, cfg))


def test_a_superseded_document_still_active_is_flagged(env):
    conn, cfg = env
    old = write(conn, cfg, "Old", "## A\nold\n", repo="r")
    write(conn, cfg, "New", "## A\nnew\n", repo="r", supersedes=[old])
    conn.execute("UPDATE documents SET status='active' WHERE id=?", (old,))
    conn.commit()
    assert "superseded_but_active" in _kinds(check(conn, cfg))


def test_an_empty_document_is_flagged(env):
    conn, cfg = env
    doc_id = write(conn, cfg, "Empty", "## A\nplaceholder\n", repo="r")
    conn.execute("DELETE FROM chunks WHERE document_id=?", (doc_id,))
    conn.commit()
    assert "no_chunks" in _kinds(check(conn, cfg))


def test_a_missing_file_on_disk_is_flagged(env):
    conn, cfg = env
    from pathlib import Path

    doc_id = write(conn, cfg, "Vanished", "## A\ncontent\n", repo="r")
    path = conn.execute("SELECT path FROM documents WHERE id=?", (doc_id,)).fetchone()["path"]
    Path(path).unlink()
    findings = check(conn, cfg)
    assert "missing_file" in _kinds(findings)
    assert any(f.severity == "error" for f in findings)


def test_findings_carry_severity(env):
    conn, cfg = env
    write(conn, cfg, "Source", "## A\nsee [[k_nope]]\n", repo="r")
    findings = check(conn, cfg)
    assert all(f.severity in ("error", "warning", "info") for f in findings)


def test_fsck_never_modifies_anything(env):
    conn, cfg = env
    write(conn, cfg, "Dup", "## A\na\n", repo="r")
    write(conn, cfg, "Dup", "## A\nb\n", repo="r")
    before = conn.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"]
    check(conn, cfg)
    assert conn.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"] == before


def test_same_title_in_different_repos_is_not_a_duplicate(env):
    """A note named 'record' in several repos is not a duplicate."""
    conn, cfg = env
    write(conn, cfg, "record", "## A\nfirst notes\n", repo="tool-api")
    write(conn, cfg, "record", "## A\nsecond notes\n", repo="tool-cli")
    assert "duplicate_title" not in _kinds(check(conn, cfg))


def test_same_title_in_different_features_is_not_a_duplicate(env):
    """Every feature has a overview document. Distinct documents."""
    conn, cfg = env
    write(conn, cfg, "Overview", "## A\none\n", repo="r", feature="feat-a")
    write(conn, cfg, "Overview", "## A\ntwo\n", repo="r", feature="feat-b")
    assert "duplicate_title" not in _kinds(check(conn, cfg))


def test_same_title_in_the_same_feature_is_a_duplicate(env):
    conn, cfg = env
    write(conn, cfg, "findings", "## A\none\n", repo="r", feature="feat-a")
    write(conn, cfg, "findings", "## A\ntwo\n", repo="r", feature="feat-a")
    assert "duplicate_title" in _kinds(check(conn, cfg))


def test_a_missing_file_from_an_indexed_root_is_flagged(env):
    """The check covers every source, not just native: index_all does not reconcile
    between runs, so fsck is the only place a vanished indexed file is reported."""
    conn, cfg = env
    conn.execute(
        "INSERT INTO documents (id, source, path, title, kind, status, mtime,"
        " indexed_at, created_at, updated_at)"
        " VALUES ('k_ext', 'shared-notes', '/nowhere/gone.md', 'Gone', 'reference',"
        " 'active', 0, '', '', '')")
    conn.commit()
    findings = check(conn, cfg)
    assert "missing_file" in _kinds(findings)
    assert any(f.document_id == "k_ext" for f in findings)


def test_a_missing_external_file_is_a_warning_not_an_error(env):
    """A native file that vanished is corruption: akasha owns that directory. An indexed
    root belongs to someone else, where a file disappearing is ordinary."""
    conn, cfg = env
    conn.execute(
        "INSERT INTO documents (id, source, path, title, kind, status, mtime,"
        " indexed_at, created_at, updated_at)"
        " VALUES ('k_ext', 'shared-notes', '/nowhere/gone.md', 'Gone', 'reference',"
        " 'active', 0, '', '', '')")
    conn.commit()
    finding = next(f for f in check(conn, cfg) if f.kind == "missing_file")
    assert finding.severity == "warning"


def test_a_pruned_document_is_not_reported_missing(env):
    """index_all tombstones what it reconciles; fsck must not re-report it as a problem."""
    conn, cfg = env
    conn.execute(
        "INSERT INTO documents (id, source, path, title, kind, status, mtime,"
        " indexed_at, created_at, updated_at, deleted_at)"
        " VALUES ('k_ext', 'shared-notes', '/nowhere/gone.md', 'Gone', 'reference',"
        " 'active', 0, '', '', '', '2026-01-01')")
    conn.commit()
    assert "missing_file" not in _kinds(check(conn, cfg))


def test_an_existing_anchor_path_produces_no_finding(env, tmp_path):
    """A document naming a path that exists is not stale — no finding."""
    conn, cfg, _ = _repo(env, tmp_path)
    write(conn, cfg, "Notes", "## A\nUses `src/util.py` here.\n", repo="r")
    assert [f for f in check(conn, cfg) if f.kind == "stale_anchor"] == []


def test_a_missing_anchor_path_produces_exactly_one_finding(env, tmp_path):
    """A document naming a path that does not exist reports exactly one stale_anchor."""
    conn, cfg, _ = _repo(env, tmp_path)
    doc_id = write(conn, cfg, "Notes", "## A\nUses `src/gone.py` here.\n", repo="r")
    findings = [f for f in check(conn, cfg) if f.kind == "stale_anchor"]
    assert len(findings) == 1
    assert findings[0].document_id == doc_id
    assert findings[0].severity == "info"
    assert "src/gone.py" in findings[0].detail


def test_a_bare_filename_anchor_produces_no_finding_either_way(env, tmp_path):
    """A bare filename is ambiguous across repos, so its existence decides nothing."""
    conn, cfg, _ = _repo(env, tmp_path)
    # util.py exists on disk, context.py does not — neither may produce a finding.
    write(conn, cfg, "Bare", "## A\nsee `util.py` and `context.py`.\n", repo="r")
    assert [f for f in check(conn, cfg) if f.kind == "stale_anchor"] == []


def test_an_anchor_with_a_line_number_is_judged_on_the_path_alone(env, tmp_path):
    """A line number drifts on every commit, so only the path decides the finding."""
    conn, cfg, _ = _repo(env, tmp_path)
    write(conn, cfg, "Anchored", "## A\n`src/util.py:41` and `src/gone.py:7`.\n", repo="r")
    findings = [f for f in check(conn, cfg) if f.kind == "stale_anchor"]
    assert len(findings) == 1
    assert "src/gone.py" in findings[0].detail
    assert "src/util.py" not in findings[0].detail


def test_an_anchor_in_a_repo_without_a_configured_path_is_not_guessed(env, tmp_path):
    """Resolving the anchor needs a real checkout; nothing configured means skip."""
    conn, cfg = env
    write(conn, cfg, "Orphan", "## A\n`src/gone.py`.\n", repo="unconfigured")
    assert [f for f in check(conn, cfg) if f.kind == "stale_anchor"] == []


def test_a_command_shaped_anchor_produces_no_finding(env, tmp_path):
    """A shell command names a program, not a file — whitespace marks it as no anchor."""
    conn, cfg, _ = _repo(env, tmp_path)
    write(conn, cfg, "Ops",
          "## A\nrun `python3 scripts/lint.py --files` before committing.\n",
          repo="r")
    assert [f for f in check(conn, cfg) if f.kind == "stale_anchor"] == []


def test_a_glob_shaped_anchor_produces_no_finding(env, tmp_path):
    """A glob names a pattern, not one file; it must not be reported stale."""
    conn, cfg, _ = _repo(env, tmp_path)
    write(conn, cfg, "Globs", "## A\ncovers `src/*` and `./run-tests *`.\n", repo="r")
    assert [f for f in check(conn, cfg) if f.kind == "stale_anchor"] == []


def test_a_directory_anchor_produces_no_finding(env, tmp_path):
    """A trailing slash marks a directory, which resolves to nothing file-shaped."""
    conn, cfg, _ = _repo(env, tmp_path)
    write(conn, cfg, "Dir", "## A\nsee `.cache/`.\n", repo="r")
    assert [f for f in check(conn, cfg) if f.kind == "stale_anchor"] == []


def test_an_extensionless_anchor_produces_no_finding(env, tmp_path):
    """`scripts/run` is a command shape, not a file; extensionless spans are skipped."""
    conn, cfg, _ = _repo(env, tmp_path)
    write(conn, cfg, "NoExt", "## A\nrun `scripts/run` or `bin/install`.\n", repo="r")
    assert [f for f in check(conn, cfg) if f.kind == "stale_anchor"] == []


def test_a_symbol_selecting_anchor_is_judged_on_the_file_alone(env, tmp_path):
    """`file.py::func` selects a symbol, out of scope like a line number — the file alone decides."""
    conn, cfg, _ = _repo(env, tmp_path)
    write(conn, cfg, "NodeIds", "## A\n`src/util.py::helper` and `src/gone.py::gone`.\n", repo="r")
    findings = [f for f in check(conn, cfg) if f.kind == "stale_anchor"]
    assert len(findings) == 1
    assert "src/gone.py" in findings[0].detail
    assert "src/util.py" not in findings[0].detail


def test_an_archived_documents_stale_anchor_is_not_flagged(env, tmp_path):
    """Same reasoning as the archived dangling_link case: an archived document is
    deliberately hidden, so its stale anchors should not keep surfacing either."""
    conn, cfg, _ = _repo(env, tmp_path)
    doc_id = write(conn, cfg, "Notes", "## A\nUses `src/gone.py` here.\n", repo="r")
    conn.execute("UPDATE documents SET status='archived' WHERE id=?", (doc_id,))
    conn.commit()
    assert [f for f in check(conn, cfg) if f.kind == "stale_anchor"] == []


def test_a_tilde_project_root_is_expanded(env, tmp_path, monkeypatch):
    """A [projects.*] path may carry ~; without expanduser every anchor reads stale."""
    monkeypatch.setenv("HOME", str(tmp_path))
    conn, cfg = env
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "util.py").write_text("def u():\n    pass\n")
    cfg.projects = {"r": {"path": "~/repo"}}
    write(conn, cfg, "Tilde", "## A\nUses `src/util.py` here.\n", repo="r")
    assert [f for f in check(conn, cfg) if f.kind == "stale_anchor"] == []


# Near-duplicates. duplicate_title groups on LOWER(title), so two documents recording the
# same finding under different titles are invisible to it.

def _vector_env(env, encode, dim=8):
    """The fsck env with vectors indexed by a stub encoder, or a skip when unavailable."""
    from akasha import vectors
    conn, cfg = env
    cfg.embeddings_provider = "model2vec"
    if not vectors.available(conn):
        pytest.skip("sqlite-vec unavailable on this build")
    vectors.index_chunks(conn, cfg, encode=encode, dim=dim)
    return conn, cfg


def _stub_embedder(dim: int = 8):
    """Deterministic vectors from word overlap. """
    import numpy as np
    vocab = ["cache", "catalog", "record", "member", "test", "router", "worker", "review"]

    def encode(texts):
        out = []
        for text in texts:
            low = text.lower()
            v = np.array([low.count(w) for w in vocab[:dim]], dtype=np.float32)
            if not v.any():
                v[0] = 0.01
            out.append(v / np.linalg.norm(v))
        return np.array(out, dtype=np.float32)

    return encode


def test_near_duplicates_under_different_titles_are_flagged(env):
    """The gap duplicate_title leaves: same finding, two titles, no exact match."""
    conn, cfg = env
    write(conn, cfg, "Catalog cache empty after a restart",
          "## Why\nthe catalog cache was empty after a restart\n", repo="r")
    write(conn, cfg, "Restart leaves the catalog cache unpopulated",
          "## Why\nthe catalog cache went empty after a restart\n", repo="r")
    conn, cfg = _vector_env((conn, cfg), _stub_embedder())
    findings = check(conn, cfg)
    assert "near_duplicate" in _kinds(findings)
    # One finding per document, matching how duplicate_title names each offender.
    assert len([f for f in findings if f.kind == "near_duplicate"]) == 2
    # The disposition must be a merge: each document gets its own warning, so advice
    # like "supersede" is actionable as drop-this-copy — and one-at-a-time drops are
    # how both twins vanish with no single step looking wrong.
    for f in [f for f in findings if f.kind == "near_duplicate"]:
        assert "knowledge_append" in f.detail
        assert "supersede" not in f.detail


def test_unrelated_documents_are_not_near_duplicates(env):
    """A check that fires on ordinary distinct documents trains readers to skip fsck."""
    conn, cfg = env
    write(conn, cfg, "Catalog cache", "## Why\nthe catalog cache was empty\n", repo="r")
    write(conn, cfg, "Router worker", "## Why\nthe router worker runs alone\n", repo="r")
    conn, cfg = _vector_env((conn, cfg), _stub_embedder())
    assert [f for f in check(conn, cfg) if f.kind == "near_duplicate"] == []


def test_near_duplicates_are_scoped_to_one_repo(env):
    """Same reasoning duplicate_title gives for titles: two repos' writeups of
    one change can score as near-identical and are two documents, not a duplicated one."""
    conn, cfg = env
    write(conn, cfg, "Catalog cache empty", "## Why\nthe catalog cache was empty\n",
          repo="tool-api")
    write(conn, cfg, "Catalog cache blank", "## Why\nthe catalog cache was empty\n",
          repo="tool-cli")
    conn, cfg = _vector_env((conn, cfg), _stub_embedder())
    assert [f for f in check(conn, cfg) if f.kind == "near_duplicate"] == []


def test_an_archived_document_is_not_a_near_duplicate(env):
    """Consistent with the archived dangling_link and stale_anchor cases."""
    conn, cfg = env
    write(conn, cfg, "Catalog cache empty", "## Why\nthe catalog cache was empty\n", repo="r")
    doc_id = write(conn, cfg, "Catalog cache blank", "## Why\nthe catalog cache was empty\n",
                   repo="r")
    conn.execute("UPDATE documents SET status='archived' WHERE id=?", (doc_id,))
    conn.commit()
    conn, cfg = _vector_env((conn, cfg), _stub_embedder())
    assert [f for f in check(conn, cfg) if f.kind == "near_duplicate"] == []


def test_an_unusable_vector_index_is_reported_not_silent(env, monkeypatch):
    """An operator who asked for
    vectors must not read a clean fsck when the near-duplicate half never ran."""
    from akasha import vectors
    conn, cfg = env
    cfg.embeddings_provider = "model2vec"
    write(conn, cfg, "Only doc", "## A\ncontent here\n", repo="r")
    monkeypatch.setattr(vectors, "available", lambda conn: False)
    assert "near_duplicate_unchecked" in _kinds(check(conn, cfg))


def test_vectors_turned_off_is_not_a_degradation(env):
    """`embeddings_provider = "none"` is a choice, not a failure — reporting it every
    run would be noise on a lexical-only install."""
    conn, cfg = env
    cfg.embeddings_provider = "none"
    write(conn, cfg, "Only doc", "## A\ncontent here\n", repo="r")
    assert check(conn, cfg) == []


def test_imported_documents_are_not_near_duplicate_candidates(env):
    """The ownership line missing_file already draws. An indexed root belongs to another
    tool whose documents refuse archive and update, so a fold-or-supersede finding is
    advice its owner cannot take, and unscoped it would bury the native findings."""
    conn, cfg = env
    write(conn, cfg, "Catalog cache empty", "## Why\nthe catalog cache was empty\n", repo="r")
    doc_id = write(conn, cfg, "Catalog cache blank", "## Why\nthe catalog cache was empty\n",
                   repo="r")
    conn.execute("UPDATE documents SET source='serena' WHERE id=?", (doc_id,))
    conn.commit()
    conn, cfg = _vector_env((conn, cfg), _stub_embedder())
    assert [f for f in check(conn, cfg) if f.kind == "near_duplicate"] == []


def _many_findings(conn, cfg, n=5):
    for i in range(n):
        write(conn, cfg, f"Source {i}", f"## A\nsee [[k_missing_{i}]]\n", repo="r")


def test_a_limited_report_still_counts_everything(env):
    """The caller reads counts to decide whether to dig; a limit trims the detail it is
    paid for in context, never the totals."""
    from akasha.fsck import report

    conn, cfg = env
    _many_findings(conn, cfg, 5)
    out = report(conn, cfg, limit=2)
    assert len(out["findings"]) == 2
    assert out["total"] == 5
    assert out["counts"] == {"dangling_link": 5}
    assert out["withheld"] == 3


def test_an_unlimited_report_withholds_nothing(env):
    from akasha.fsck import report

    conn, cfg = env
    _many_findings(conn, cfg, 3)
    out = report(conn, cfg, limit=None)
    assert len(out["findings"]) == 3 and out["withheld"] == 0


def test_a_full_run_records_its_error_count_for_the_session_brief(env):
    """The session-start note reads this instead of running fsck on every session."""
    from pathlib import Path

    from akasha.fsck import cached_error_count, report

    conn, cfg = env
    assert cached_error_count(conn) is None, "never run is not the same as zero errors"
    doc_id = write(conn, cfg, "Vanished", "## A\ncontent\n", repo="r")
    Path(conn.execute("SELECT path FROM documents WHERE id=?", (doc_id,)).fetchone()["path"]).unlink()

    report(conn, cfg)
    assert cached_error_count(conn) == 1


def test_the_recorded_error_count_ignores_the_display_limit(env):
    """A limit of zero shows no findings, and the recorded count must still be the truth."""
    from pathlib import Path

    from akasha.fsck import cached_error_count, report

    conn, cfg = env
    for i in range(3):
        doc_id = write(conn, cfg, f"Gone {i}", "## A\ncontent\n", repo="r")
        Path(conn.execute("SELECT path FROM documents WHERE id=?",
                          (doc_id,)).fetchone()["path"]).unlink()
    report(conn, cfg, limit=0)
    assert cached_error_count(conn) == 3


def test_a_clean_run_records_zero(env):
    from akasha.fsck import cached_error_count, report

    conn, cfg = env
    write(conn, cfg, "Only doc", "## A\ncontent here\n", repo="r")
    report(conn, cfg)
    assert cached_error_count(conn) == 0
