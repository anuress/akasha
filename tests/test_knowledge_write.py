import pytest

from akasha.config import IndexRoot, load_config
from akasha.db import connect
from akasha.index import index_all
from akasha.knowledge import ReadOnlySource, StaleWrite, append, update, write


@pytest.fixture
def env(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")
    cfg.knowledge_dir = tmp_path / "knowledge"
    cfg.knowledge_dir.mkdir()
    cfg.index_roots = []
    return connect(tmp_path / "s.db"), cfg


def test_write_creates_file_with_frontmatter(env):
    conn, cfg = env
    doc_id = write(conn, cfg, "Catalog ownership", "## Taxonomy\nFineCalculator\n",
                   repo="sample-repo", feature="fines", kind="spec")
    row = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
    from pathlib import Path
    text = Path(row["path"]).read_text()
    assert text.startswith("---")
    assert "title: Catalog ownership" in text
    assert "feature: fines" in text
    assert row["source"] == "native"


def test_write_is_searchable_immediately(env):
    conn, cfg = env
    write(conn, cfg, "T", "## Body\nunique-token-zzz\n", repo="r")
    from akasha.search import search
    assert search(conn, "unique-token-zzz", all_repos=True)


def test_supersede_archives_the_old_document(env):
    conn, cfg = env
    old = write(conn, cfg, "Old", "## A\nold conclusion\n", repo="r")
    write(conn, cfg, "New", "## A\nnew conclusion\n", repo="r", supersedes=[old])
    assert conn.execute("SELECT status FROM documents WHERE id=?", (old,)).fetchone()["status"] == "archived"


def test_update_changes_body_and_bumps_updated(env):
    conn, cfg = env
    doc_id = write(conn, cfg, "T", "## A\nfirst\n", repo="r")
    before = conn.execute("SELECT updated_at FROM documents WHERE id=?", (doc_id,)).fetchone()["updated_at"]
    update(conn, cfg, doc_id, body="## A\nsecond\n")
    row = conn.execute("SELECT updated_at FROM documents WHERE id=?", (doc_id,)).fetchone()
    assert row["updated_at"] >= before
    from akasha.search import search
    assert search(conn, "second", all_repos=True)
    assert not search(conn, "first", all_repos=True)


def test_update_metadata_only_does_not_require_body(env):
    conn, cfg = env
    doc_id = write(conn, cfg, "T", "## A\nkeep me\n", repo="r")
    update(conn, cfg, doc_id, kind="findings")
    row = conn.execute("SELECT kind FROM documents WHERE id=?", (doc_id,)).fetchone()
    assert row["kind"] == "finding"     # alias canonicalised on the update path too
    from akasha.search import search
    assert search(conn, "keep", all_repos=True)


def test_update_with_stale_expected_updated_is_refused(env):
    conn, cfg = env
    doc_id = write(conn, cfg, "T", "## A\nx\n", repo="r")
    with pytest.raises(StaleWrite):
        update(conn, cfg, doc_id, body="## A\ny\n", expected_updated="1999-01-01T00:00:00+00:00")


def test_update_on_external_source_is_refused(env, tmp_path):
    conn, cfg = env
    ext = tmp_path / "ws" / "repo" / "feat"
    ext.mkdir(parents=True)
    doc = ext / "external.md"
    doc.write_text("## A\nnot ours\n")
    cfg.index_roots = [IndexRoot(path=str(tmp_path / "ws"), source="shared-notes")]
    index_all(conn, cfg)
    doc_id = conn.execute("SELECT id FROM documents WHERE source='shared-notes'").fetchone()["id"]
    with pytest.raises(ReadOnlySource):
        update(conn, cfg, doc_id, body="## A\nmine now\n")


def test_append_adds_dated_section_and_preserves_original(env):
    conn, cfg = env
    doc_id = write(conn, cfg, "T", "## A\noriginal text\n", repo="r")
    append(conn, cfg, doc_id, "limit 30 shipped")
    from pathlib import Path
    path = conn.execute("SELECT path FROM documents WHERE id=?", (doc_id,)).fetchone()["path"]
    text = Path(path).read_text()
    assert "original text" in text
    assert "limit 30 shipped" in text
    assert "## UPDATE" in text


def test_two_appends_both_survive(env):
    conn, cfg = env
    doc_id = write(conn, cfg, "T", "## A\nbase\n", repo="r")
    append(conn, cfg, doc_id, "first addition")
    append(conn, cfg, doc_id, "second addition")
    from pathlib import Path
    path = conn.execute("SELECT path FROM documents WHERE id=?", (doc_id,)).fetchone()["path"]
    text = Path(path).read_text()
    assert "first addition" in text and "second addition" in text


def test_canonical_kind_resolves_aliases_and_names_the_vocabulary_on_refusal():
    """The refusal lists what is allowed: the next guess is no better informed."""
    from akasha.knowledge import KINDS, canonical_kind

    assert canonical_kind("findings") == "finding"
    assert canonical_kind("spec") == "spec"
    with pytest.raises(ValueError) as exc:
        canonical_kind("essay")
    assert all(kind in str(exc.value) for kind in KINDS)


def test_the_indexer_falls_back_for_a_kind_write_would_refuse(env):
    """Same vocabulary, different handling: bulk indexing counts an unknown kind and
    carries on, where an interactive write refuses it."""
    from akasha.index import index_path
    from akasha.markdown import render

    conn, cfg = env
    path = cfg.knowledge_dir / "odd.md"
    path.write_text(render({"title": "Odd", "kind": "essay"}, "## A\nbody\n"))
    doc_id, _, fallback = index_path(conn, cfg, path, "native", root=cfg.knowledge_dir)
    assert fallback == 1
    assert conn.execute("SELECT kind FROM documents WHERE id=?", (doc_id,)).fetchone()[0] == "reference"
    with pytest.raises(ValueError):
        write(conn, cfg, "Odd", "## A\nbody\n", kind="essay")


# --- repo names become a directory name, so they are validated -------------------------

@pytest.mark.parametrize("repo", ["../../x", "/tmp/elsewhere", "a/b", "..", ".", "a b"])
def test_write_refuses_a_repo_that_is_not_a_plain_name(env, repo, tmp_path):
    """The repo is joined onto the knowledge dir; anything but a plain name could write
    outside it."""
    conn, cfg = env
    with pytest.raises(ValueError, match="repo"):
        write(conn, cfg, "T", "## H\nb", repo=repo)
    assert list(tmp_path.rglob("*.md")) == []


def test_write_accepts_a_plain_repo_name(env):
    conn, cfg = env
    doc_id = write(conn, cfg, "T", "## H\nb", repo="my-repo_1.x")
    path = conn.execute("SELECT path FROM documents WHERE id=?", (doc_id,)).fetchone()["path"]
    assert path.startswith(str(cfg.knowledge_dir / "my-repo_1.x"))


def test_write_refuses_a_repo_folder_that_resolves_outside_the_knowledge_dir(env, tmp_path):
    """A symlinked folder with a valid name must not lead outside either."""
    conn, cfg = env
    outside = tmp_path / "outside"
    outside.mkdir()
    (cfg.knowledge_dir / "linked").symlink_to(outside)
    with pytest.raises(ValueError, match="repo"):
        write(conn, cfg, "T", "## H\nb", repo="linked")
    assert list(outside.iterdir()) == []
