import hashlib
import os
from pathlib import Path

import pytest

from akasha import feed
from akasha.config import IndexRoot, load_config
from akasha.db import connect
from akasha.fsck import MAX_READ_CHARS, check, dataset_documents
from akasha.index import index_path
from akasha.markdown import parse_frontmatter
from akasha.splitters import FeedSettings
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


def test_a_native_document_over_the_get_window_is_flagged(env):
    """A document that needs multiple get calls should be visible to fsck."""
    conn, cfg = env
    doc_id = write(conn, cfg, "Large", "x" * (MAX_READ_CHARS + 1), repo="r")

    findings = [f for f in check(conn, cfg) if f.kind == "oversized"]

    assert len(findings) == 1
    finding = findings[0]
    path = conn.execute("SELECT path FROM documents WHERE id=?", (doc_id,)).fetchone()["path"]
    chars = len(Path(path).read_text(errors="replace"))
    assert finding.document_id == doc_id
    assert finding.severity == "warning"
    assert finding.detail == (
        f"{chars} chars, over the {MAX_READ_CHARS}-char knowledge_get window; "
        f"write a summary with supersedes=[{doc_id}] instead of appending further")


def test_a_native_document_under_the_get_window_is_not_flagged(env):
    """A normal-sized document should not add noise to fsck."""
    conn, cfg = env
    write(conn, cfg, "Small", "short body", repo="r")

    assert "oversized" not in _kinds(check(conn, cfg))


def test_a_multibyte_file_under_the_get_char_window_is_not_flagged(env):
    """Filesystem bytes can exceed the limit while decoded characters remain readable."""
    conn, cfg = env
    doc_id = write(conn, cfg, "Multibyte", "é" * (MAX_READ_CHARS // 2), repo="r")
    path = Path(conn.execute("SELECT path FROM documents WHERE id=?",
                             (doc_id,)).fetchone()["path"])
    chars = len(path.read_text(errors="replace"))

    assert path.stat().st_size > MAX_READ_CHARS
    assert chars <= MAX_READ_CHARS
    assert not any(f.kind == "oversized" and f.document_id == doc_id for f in check(conn, cfg))


def test_a_document_exactly_at_the_get_window_is_not_flagged(env):
    """Only files with more characters than the retrieval window should be reported."""
    conn, cfg = env
    doc_id = write(conn, cfg, "Boundary", "small body", repo="r")
    path = Path(conn.execute("SELECT path FROM documents WHERE id=?",
                             (doc_id,)).fetchone()["path"])
    text = path.read_text(errors="replace")
    path.write_text(text + "x" * (MAX_READ_CHARS - len(text)))

    assert len(path.read_text(errors="replace")) == MAX_READ_CHARS
    assert not any(f.kind == "oversized" and f.document_id == doc_id for f in check(conn, cfg))


def test_a_native_file_grown_without_reindexing_is_flagged(env):
    """The indexed byte size must not hide a file edited past the get window."""
    conn, cfg = env
    doc_id = write(conn, cfg, "Grown", "small body", repo="r")
    row = conn.execute("SELECT path, size FROM documents WHERE id=?", (doc_id,)).fetchone()
    path = Path(row["path"])
    text = path.read_text(errors="replace")
    path.write_text(text + "x" * (MAX_READ_CHARS + 1 - len(text)))

    assert row["size"] < MAX_READ_CHARS
    assert len(path.read_text(errors="replace")) == MAX_READ_CHARS + 1
    assert any(f.kind == "oversized" and f.document_id == doc_id for f in check(conn, cfg))


def test_a_non_native_oversized_document_is_not_flagged(tmp_path, env):
    """The oversized warning is scoped to native documents only."""
    conn, cfg = env
    root = tmp_path / "indexed"
    doc = root / "sample-repo" / "notes" / "large.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("x" * (MAX_READ_CHARS + 1))
    cfg.index_roots = [IndexRoot(path=str(root), source="sample-notes")]
    doc_id = index_path(conn, cfg, doc, "sample-notes", root=root)[0]

    assert conn.execute("SELECT source FROM documents WHERE id=?", (doc_id,)).fetchone()["source"] == "sample-notes"
    assert not any(f.kind == "oversized" and f.document_id == doc_id for f in check(conn, cfg))


def test_an_archived_oversized_document_is_not_flagged(env):
    """Archived documents are intentionally hidden and should not keep surfacing."""
    conn, cfg = env
    doc_id = write(conn, cfg, "Archived", "x" * (MAX_READ_CHARS + 1), repo="r")
    conn.execute("UPDATE documents SET status='archived' WHERE id=?", (doc_id,))
    conn.commit()

    assert "oversized" not in _kinds(check(conn, cfg))


def test_a_missing_oversized_file_reports_missing_not_oversized(env):
    """A cached large size must not make fsck read a file that has vanished."""
    conn, cfg = env
    doc_id = write(conn, cfg, "Vanished", "x" * (MAX_READ_CHARS + 1), repo="r")
    path = Path(conn.execute("SELECT path FROM documents WHERE id=?",
                             (doc_id,)).fetchone()["path"])
    path.unlink()

    findings = [f for f in check(conn, cfg) if f.document_id == doc_id]

    assert any(f.kind == "missing_file" for f in findings)
    assert not any(f.kind == "oversized" for f in findings)


def test_fsck_skips_a_file_removed_during_oversize_check(env, monkeypatch):
    """A concurrent removal between file check and read should not abort fsck."""
    conn, cfg = env
    doc_id = write(conn, cfg, "Vanished", "x" * (MAX_READ_CHARS + 1), repo="r")
    path = Path(conn.execute("SELECT path FROM documents WHERE id=?",
                             (doc_id,)).fetchone()["path"])
    read_text = Path.read_text

    def remove_before_read(file, *args, **kwargs):
        if file == path:
            file.unlink()
        return read_text(file, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", remove_before_read)

    findings = [f for f in check(conn, cfg) if f.document_id == doc_id]

    assert not any(f.kind == "oversized" for f in findings)


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
    from akasha.fsck import cached_error_count, report

    conn, cfg = env
    assert cached_error_count(conn) is None, "never run is not the same as zero errors"
    doc_id = write(conn, cfg, "Vanished", "## A\ncontent\n", repo="r")
    Path(conn.execute("SELECT path FROM documents WHERE id=?", (doc_id,)).fetchone()["path"]).unlink()

    report(conn, cfg)
    assert cached_error_count(conn) == 1


def test_the_recorded_error_count_ignores_the_display_limit(env):
    """A limit of zero shows no findings, and the recorded count must still be the truth."""
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


# --- datasets ------------------------------------------------------------------------

def _dataset(env, tmp_path, name="prices"):
    """One fed dataset: the state every dataset finding is about."""
    conn, cfg = env
    source = tmp_path / f"{name}.csv"
    source.write_text("item,price\nbolt,4\nnut,2\n", encoding="utf-8")
    return feed.feed(conn, cfg, source, name, settings=FeedSettings(key="item"),
                     repo="sample-repo")


def _dataset_doc(cfg, name="prices"):
    return cfg.knowledge_dir / "sample-repo" / "datasets" / f"{name}.md"


def _dataset_findings(conn, cfg):
    return [f for f in check(conn, cfg) if f.kind.startswith("dataset_")]


def test_clean_dataset_has_no_dataset_findings(env, tmp_path):
    """A healthy dataset must add no noise, or the whole report gets skipped as noise."""
    conn, cfg = env
    _dataset(env, tmp_path)

    assert _dataset_findings(conn, cfg) == []


def test_a_dataset_whose_copy_vanished_is_an_error(env, tmp_path):
    """The document promises cells it can no longer re-read, and only a re-feed restores it."""
    conn, cfg = env
    result = _dataset(env, tmp_path)
    result.copy.unlink()

    findings = [f for f in _dataset_findings(conn, cfg) if f.kind == "dataset_copy_missing"]

    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert findings[0].document_id == result.doc_id
    assert "akasha feed add" in findings[0].detail


def test_a_copy_whose_bytes_changed_is_an_error_naming_both_shas(env, tmp_path):
    """A rebuild trusts the frontmatter sha, so a silent edit leaves the units unexplained."""
    conn, cfg = env
    result = _dataset(env, tmp_path)
    result.copy.write_text("item,price\nbolt,9\n", encoding="utf-8")
    actual = hashlib.sha256(result.copy.read_bytes()).hexdigest()
    recorded = parse_frontmatter(_dataset_doc(cfg).read_text())[0]["sha256"]

    findings = [f for f in _dataset_findings(conn, cfg) if f.kind == "dataset_copy_mismatch"]

    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert findings[0].document_id == result.doc_id
    assert actual[:12] in findings[0].detail
    assert recorded[:12] in findings[0].detail
    assert "feed the original file again" in findings[0].detail


def test_units_that_are_not_the_copy_are_stale_and_say_to_force_a_reindex(env, tmp_path):
    """Stale units answer searches with cells the document no longer describes."""
    conn, cfg = env
    result = _dataset(env, tmp_path)
    conn.execute("UPDATE meta SET value = 'older' WHERE key = 'feed_state:prices'")

    findings = [f for f in _dataset_findings(conn, cfg) if f.kind == "dataset_units_stale"]

    assert len(findings) == 1
    assert findings[0].severity == "warning"
    assert findings[0].document_id == result.doc_id
    assert "akasha index --force" in findings[0].detail


def test_a_file_no_dataset_names_is_an_orphan(env, tmp_path):
    """An orphan costs disk and reads as a dataset copy to anyone looking at the files dir."""
    conn, cfg = env
    _dataset(env, tmp_path)
    stray = cfg.knowledge_dir.parent / "files" / "sample-repo" / "notes.csv"
    stray.write_text("a\n1\n", encoding="utf-8")

    findings = [f for f in _dataset_findings(conn, cfg) if f.kind == "dataset_copy_orphan"]

    assert len(findings) == 1
    assert findings[0].severity == "warning"
    assert findings[0].document_id is None
    assert str(stray) in findings[0].detail


def test_staging_and_backup_files_are_leftovers_not_orphans(env, tmp_path):
    """Both are what an interrupted feed leaves, and neither is a copy worth keeping."""
    conn, cfg = env
    _dataset(env, tmp_path)
    repo_dir = cfg.knowledge_dir.parent / "files" / "sample-repo"
    staging = repo_dir / "prices.staging.csv"
    backup = Path(f"{repo_dir / 'prices.csv'}.bak")
    staging.write_text("item,price\n", encoding="utf-8")
    backup.write_text("item,price\n", encoding="utf-8")

    findings = _dataset_findings(conn, cfg)

    leftovers = [f for f in findings if f.kind == "dataset_leftover_temp"]
    assert len(leftovers) == 2
    assert all(f.severity == "warning" and f.document_id is None for f in leftovers)
    assert any(str(staging) in f.detail for f in leftovers)
    assert any(str(backup) in f.detail for f in leftovers)
    assert not [f for f in findings if f.kind == "dataset_copy_orphan"]


def test_trash_and_locks_are_not_orphans(env, tmp_path):
    """akasha moves a replaced copy to the trash and keeps its locks; both are deliberate."""
    conn, cfg = env
    _dataset(env, tmp_path)
    files = cfg.knowledge_dir.parent / "files"
    (files / ".trash" / "sample-repo").mkdir(parents=True)
    (files / ".trash" / "sample-repo" / "old.csv").write_text("item\n", encoding="utf-8")

    assert _dataset_findings(conn, cfg) == []


def test_dataset_doc_with_bad_frontmatter_is_reported(env, tmp_path):
    """A damaged document must not vanish from every check while its cells still answer."""
    conn, cfg = env
    result = _dataset(env, tmp_path)
    doc = _dataset_doc(cfg)
    doc.write_text("".join(line for line in doc.read_text().splitlines(keepends=True)
                           if not line.startswith("sha256:")))

    findings = [f for f in _dataset_findings(conn, cfg) if f.kind == "dataset_metadata_invalid"]

    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert findings[0].document_id == result.doc_id
    assert findings[0].detail == (
        "dataset metadata sha256 missing or unusable; "
        "feed the original file again with `akasha feed add`")
    assert doc.is_file(), "fsck reports a damaged document, it never deletes one"
    assert conn.execute("SELECT COUNT(*) c FROM chunks WHERE document_id = ?",
                        (result.doc_id,)).fetchone()["c"] > 0


def test_the_copy_of_a_document_with_bad_metadata_is_not_called_orphan(env, tmp_path):
    """The re-feed needs that copy, so the document is the finding, never the file."""
    conn, cfg = env
    result = _dataset(env, tmp_path)
    doc = _dataset_doc(cfg)
    doc.write_text("".join(line for line in doc.read_text().splitlines(keepends=True)
                           if not line.startswith(("sha256:", "settings_sha256:"))))

    findings = _dataset_findings(conn, cfg)

    assert [f.kind for f in findings] == ["dataset_metadata_invalid"]
    assert findings[0].document_id == result.doc_id
    assert "sha256, settings_sha256" in findings[0].detail
    assert result.copy.is_file()


def test_the_named_copy_of_a_broken_document_is_not_an_orphan(env, tmp_path):
    """One bad key still leaves `file` readable, and that copy is the one to re-feed."""
    conn, cfg = env
    _dataset(env, tmp_path)
    doc = _dataset_doc(cfg)
    doc.write_text("".join(line for line in doc.read_text().splitlines(keepends=True)
                           if not line.startswith("settings_sha256:")))

    kinds = [f.kind for f in _dataset_findings(conn, cfg)]

    assert kinds == ["dataset_metadata_invalid"]


def test_a_document_with_no_frontmatter_keeps_its_copy_out_of_the_orphan_list(env, tmp_path):
    """Frontmatter gone, the copy is still what a re-feed reads: deleting it is wrong."""
    conn, cfg = env
    result = _dataset(env, tmp_path)
    _dataset_doc(cfg).write_text("## no frontmatter at all\ncells still answer\n")

    assert _dataset_findings(conn, cfg) == []
    assert result.copy.is_file()


def test_an_orphan_is_described_as_unclaimed(env, tmp_path):
    """Nothing here knows who owns an orphan, so no advice calls it safe to delete."""
    conn, cfg = env
    _dataset(env, tmp_path)
    stray = cfg.knowledge_dir.parent / "files" / "sample-repo" / "notes.csv"
    stray.write_text("a\n1\n", encoding="utf-8")

    orphans = [f for f in _dataset_findings(conn, cfg) if f.kind == "dataset_copy_orphan"]

    assert len(orphans) == 1
    assert orphans[0].detail == f"{stray} is named by no readable dataset document"


def test_a_note_with_kind_dataset_outside_the_datasets_folder_is_ignored(env):
    """The path makes a dataset document; a note that claims the kind is still a note."""
    conn, cfg = env
    # Written by hand: `knowledge.write` refuses the kind now, and this is the frontmatter
    # an older akasha or another writer can still leave on disk.
    cfg.custom_kinds = ["dataset"]          # registered, so the kind survives indexing
    note = cfg.knowledge_dir / "r" / "meeting-notes.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("---\nid: k_meeting\ntitle: Meeting notes\nrepo: r\nkind: dataset\n"
                    "status: active\ncreated: 2026-01-01\nupdated: 2026-01-01\n---\n\n"
                    "## A\nordinary prose\n", encoding="utf-8")
    index_path(conn, cfg, note, "native", root=None)

    row = conn.execute("SELECT kind FROM documents WHERE path = ?", (str(note),)).fetchone()
    assert row["kind"] == "dataset", "the fixture must reach the check under test"
    assert _dataset_findings(conn, cfg) == []


def test_a_dataset_doc_recorded_with_backslash_paths_is_still_checked(env, tmp_path):
    """Paths are stored with str(path), so a Windows row holds backslashes; one separator
    must not hide every dataset document from the checks."""
    conn, cfg = env
    result = _dataset(env, tmp_path)
    conn.execute("UPDATE documents SET path = ? WHERE id = ?",
                 (str(_dataset_doc(cfg)).replace("/", "\\"), result.doc_id))
    result.copy.unlink()

    findings = [f for f in _dataset_findings(conn, cfg) if f.kind == "dataset_copy_missing"]

    assert [f.document_id for f in findings] == [result.doc_id]


@pytest.mark.skipif(os.name == "nt", reason="a backslash is a separator there")
def test_a_backslash_in_a_posix_directory_name_is_not_a_separator(env):
    """A backslash is legal in a POSIX name; reading it as a separator hides a real dataset."""
    conn, cfg = env
    doc = cfg.knowledge_dir / "sample\\repo" / "datasets" / "prices.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("---\ndataset: prices\nfile: /nowhere/prices.csv\n"
                   "sha256: " + "a" * 64 + "\nsettings_sha256: " + "b" * 64
                   + "\nsettings: {\"key\": \"item\"}\n---\n\nitem,price\nbolt,4\n")
    indexed = index_path(conn, cfg, doc, "native", root=cfg.knowledge_dir)

    findings = [f for f in _dataset_findings(conn, cfg) if f.kind == "dataset_copy_missing"]

    assert [f.document_id for f in findings] == [indexed[0]]


def test_a_row_outside_datasets_is_rejected_by_shape_before_any_stat(env, monkeypatch):
    """A row that cannot be a dataset document must not cost a stat, whatever separator."""
    conn, cfg = env
    doc_id = write(conn, cfg, "Meeting notes", "## A\nordinary prose\n", repo="r")
    stored = str(cfg.knowledge_dir / "r" / "meeting-notes.md").replace("/", "\\")
    conn.execute("UPDATE documents SET path = ? WHERE id = ?", (stored, doc_id))
    calls: list[Path] = []
    real = Path.is_file

    def counting(self):
        calls.append(self)
        return real(self)

    monkeypatch.setattr(Path, "is_file", counting)
    dataset_documents(conn, cfg)

    assert calls == []


def test_a_plain_note_inside_datasets_is_not_a_broken_dataset(env, tmp_path):
    """A user may keep notes beside their datasets; without a `dataset` key it is a note,
    and feeding the original file again could never repair it."""
    conn, cfg = env
    _dataset(env, tmp_path)
    datasets = cfg.knowledge_dir / "sample-repo" / "datasets"
    (datasets / "shopping.md").write_text(
        "---\ntitle: Shopping notes\n---\n\nbolt is cheap\n")
    (datasets / "loose.md").write_text("## no frontmatter at all\nbolt is cheap\n")
    for note in ("shopping.md", "loose.md"):
        index_path(conn, cfg, datasets / note, "native", root=cfg.knowledge_dir)

    assert _dataset_findings(conn, cfg) == []


def test_a_dataset_key_with_nothing_else_is_still_a_broken_dataset(env, tmp_path):
    """Only an absent `dataset` key makes a note: a document that kept the key and lost the
    rest is damaged metadata."""
    conn, cfg = env
    result = _dataset(env, tmp_path)
    _dataset_doc(cfg).write_text("---\ndataset: prices\n---\n\nthe cells are gone\n")

    findings = [f for f in _dataset_findings(conn, cfg) if f.kind == "dataset_metadata_invalid"]

    assert [f.document_id for f in findings] == [result.doc_id]
    assert "file, sha256, settings_sha256" in findings[0].detail
    assert result.copy.is_file()


def test_a_dataset_doc_without_settings_is_reported(env, tmp_path):
    """Attach cannot re-split the copy without settings, so the document is not usable."""
    conn, cfg = env
    result = _dataset(env, tmp_path)
    doc = _dataset_doc(cfg)
    doc.write_text("".join(line for line in doc.read_text().splitlines(keepends=True)
                           if not line.startswith("settings:")))

    findings = [f for f in _dataset_findings(conn, cfg) if f.kind == "dataset_metadata_invalid"]

    assert len(findings) == 1
    assert findings[0].document_id == result.doc_id
    assert "settings" in findings[0].detail


def test_a_dataset_doc_with_unreadable_settings_is_reported(env, tmp_path):
    """A settings value that parses to no object would fail every future attach."""
    conn, cfg = env
    result = _dataset(env, tmp_path)
    doc = _dataset_doc(cfg)
    doc.write_text("".join("settings: {not json\n" if line.startswith("settings:") else line
                           for line in doc.read_text().splitlines(keepends=True)))

    findings = [f for f in _dataset_findings(conn, cfg) if f.kind == "dataset_metadata_invalid"]

    assert len(findings) == 1
    assert findings[0].document_id == result.doc_id
    assert "settings" in findings[0].detail


def test_a_damaged_doc_in_one_repo_does_not_exempt_another_repos_copy(env, tmp_path):
    """The exemption is per (repo, name): another repo's same-named copy has no owner here."""
    conn, cfg = env
    _dataset(env, tmp_path)
    _dataset_doc(cfg).write_text("---\ndataset: prices\n---\n\nthe cells are gone\n")
    other = cfg.knowledge_dir.parent / "files" / "repo-b" / "prices.csv"
    other.parent.mkdir(parents=True)
    other.write_text("item\n1\n", encoding="utf-8")

    findings = _dataset_findings(conn, cfg)

    orphans = [f for f in findings if f.kind == "dataset_copy_orphan"]
    assert len(orphans) == 1
    assert str(other) in orphans[0].detail
    assert [f.kind for f in findings if f.kind == "dataset_metadata_invalid"]
