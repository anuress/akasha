import pytest

from akasha.config import load_config
from akasha.db import connect
from akasha.knowledge import write
from akasha.index import index_all
from akasha.links import graph, parse_refs, related


@pytest.fixture
def env(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")
    cfg.knowledge_dir = tmp_path / "knowledge"
    cfg.knowledge_dir.mkdir()
    cfg.index_roots = []
    return connect(tmp_path / "s.db"), cfg


def test_parse_refs_finds_wiki_links():
    refs = parse_refs("see [[k_8f3a2c]] and [[shelf-layout-notes]] for detail")
    assert refs == ["k_8f3a2c", "shelf-layout-notes"]


def test_parse_refs_ignores_code_fenced_brackets():
    assert parse_refs("```\n[[not-a-link]]\n```\n") == []


def test_link_is_created_and_resolves_both_directions(env):
    conn, cfg = env
    target = write(conn, cfg, "Target", "## A\ntarget body\n", repo="r")
    source = write(conn, cfg, "Source", f"## A\nsee [[{target}]]\n", repo="r")

    forward = [r["id"] for r in related(conn, source)]
    assert target in forward

    backward = [r["id"] for r in related(conn, target)]
    assert source in backward


def test_dangling_link_is_recorded_not_an_error(env):
    conn, cfg = env
    doc_id = write(conn, cfg, "Source", "## A\nsee [[k_does_not_exist]]\n", repo="r")
    row = conn.execute(
        "SELECT to_document_id, to_ref FROM links WHERE from_document_id=?", (doc_id,)).fetchone()
    assert row["to_document_id"] is None
    assert row["to_ref"] == "k_does_not_exist"


def test_graph_traverses_two_hops(env):
    conn, cfg = env
    c = write(conn, cfg, "C", "## A\nend\n", repo="r")
    b = write(conn, cfg, "B", f"## A\n[[{c}]]\n", repo="r")
    a = write(conn, cfg, "A", f"## A\n[[{b}]]\n", repo="r")
    ids = {n["id"] for n in graph(conn, a, depth=2)}
    assert b in ids and c in ids


def test_graph_depth_one_stops_early(env):
    conn, cfg = env
    c = write(conn, cfg, "C", "## A\nend\n", repo="r")
    b = write(conn, cfg, "B", f"## A\n[[{c}]]\n", repo="r")
    a = write(conn, cfg, "A", f"## A\n[[{b}]]\n", repo="r")
    ids = {n["id"] for n in graph(conn, a, depth=1)}
    assert b in ids and c not in ids


def test_forward_reference_is_resolved_after_the_full_walk(tmp_path):
    """A link to a document indexed later must not stay dangling."""
    from akasha.config import IndexRoot, load_config
    from akasha.db import connect
    from akasha.index import index_all

    cfg = load_config(tmp_path / "absent.toml")
    cfg.knowledge_dir = tmp_path / "k"
    cfg.knowledge_dir.mkdir()
    cfg.db_path = tmp_path / "s.db"

    early = tmp_path / "aaa"
    late = tmp_path / "zzz"
    early.mkdir(); late.mkdir()
    (early / "referrer.md").write_text("## A\nsee [[target-note]]\n")
    (late / "target-note.md").write_text("## A\nthe target\n")
    cfg.index_roots = [IndexRoot(path=str(early), source="a"),
                       IndexRoot(path=str(late), source="b")]

    conn = connect(tmp_path / "s.db")
    stats = index_all(conn, cfg)
    assert stats.links_resolved >= 1
    row = conn.execute(
        "SELECT to_document_id FROM links WHERE to_ref='target-note'").fetchone()
    assert row["to_document_id"] is not None


@pytest.fixture
def indexed(tmp_path):
    """A real index root, unlike `env` — these tests exercise the walk's link pass."""
    from akasha.config import IndexRoot

    cfg = load_config(tmp_path / "absent.toml")
    cfg.knowledge_dir = tmp_path / "knowledge"
    cfg.knowledge_dir.mkdir()
    root = tmp_path / "ws"
    (root / "sample-repo" / "proj-100").mkdir(parents=True)
    cfg.index_roots = [IndexRoot(path=str(root), source="shared-notes")]
    return connect(tmp_path / "s.db"), cfg, root


def test_a_ref_resolves_across_underscore_and_hyphen(indexed):
    """A convention that writes `name: my-note` in frontmatter but stores the file as
    my_note.md produced links that dangled forever: the ref matched neither the title
    (taken from the filename stem, underscores intact) nor the path. Both spellings name
    one document, so both must resolve to it."""
    conn, cfg, root = indexed
    (root / "sample-repo" / "proj-100").mkdir(parents=True, exist_ok=True)
    (root / "sample-repo" / "proj-100" / "my_note.md").write_text("## A\ntarget\n")
    (root / "sample-repo" / "proj-100" / "source.md").write_text("## A\nsee [[my-note]]\n")
    index_all(conn, cfg)

    row = conn.execute(
        "SELECT to_document_id FROM links WHERE to_ref = 'my-note'").fetchone()
    assert row["to_document_id"] is not None


def test_a_ref_to_something_absent_still_dangles(indexed):
    """The looser match must not invent a target: a reference to a document that does not
    exist is the signal fsck reports, and resolving it to anything would hide that."""
    conn, cfg, root = indexed
    (root / "sample-repo" / "proj-100").mkdir(parents=True, exist_ok=True)
    (root / "sample-repo" / "proj-100" / "source.md").write_text("## A\nsee [[nothing-here]]\n")
    index_all(conn, cfg)

    row = conn.execute(
        "SELECT to_document_id FROM links WHERE to_ref = 'nothing-here'").fetchone()
    assert row["to_document_id"] is None


def test_a_forward_ref_across_spellings_resolves_on_the_second_pass(indexed):
    """resolve_dangling is the pass that fixes references recorded before their target was
    indexed; it needs the same matching rule as sync_links or the two disagree."""
    conn, cfg, root = indexed
    (root / "sample-repo" / "proj-100").mkdir(parents=True, exist_ok=True)
    (root / "sample-repo" / "proj-100" / "aaa-source.md").write_text("## A\nsee [[zzz-target]]\n")
    index_all(conn, cfg)
    assert conn.execute(
        "SELECT to_document_id FROM links WHERE to_ref='zzz-target'").fetchone()[0] is None

    (root / "sample-repo" / "proj-100" / "zzz_target.md").write_text("## A\nhere\n")
    index_all(conn, cfg)
    assert conn.execute(
        "SELECT to_document_id FROM links WHERE to_ref='zzz-target'").fetchone()[0] is not None


def _agreement(conn, ids):
    """Per-document neighbour count from the search path versus the length of related()."""
    from akasha.links import neighbour_counts

    counts = neighbour_counts(conn, ids)
    return {i: (counts.get(i, 0), len(related(conn, i))) for i in ids}


def test_related_and_search_neighbour_counts_agree_on_a_mutual_link(env):
    """The hit's `neighbours` number promises what a follow-up knowledge_related returns.
    A pair that cites each other is two edges in each direction, and both must count."""
    conn, cfg = env
    a = write(conn, cfg, "A", "## A\nfirst\n", repo="r")
    b = write(conn, cfg, "B", f"## A\n[[{a}]]\n", repo="r")
    conn.execute("INSERT INTO links (id, from_document_id, to_document_id, to_ref, kind,"
                 " created_at) VALUES ('l_back', ?, ?, ?, 'cites', '')", (a, b, b))
    conn.commit()
    for seen, listed in _agreement(conn, [a, b]).values():
        assert seen == listed == 2


def test_related_and_search_neighbour_counts_agree_when_two_refs_hit_one_target(env):
    """`[[id]]` and `[[Title]]` are different refs that resolve to one document: two link
    rows, one neighbour. A count that follows rows would promise a hop that returns less."""
    conn, cfg = env
    target = write(conn, cfg, "Target", "## A\nbody\n", repo="r")
    source = write(conn, cfg, "Source", f"## A\n[[{target}]] and [[Target]]\n", repo="r")
    assert conn.execute("SELECT COUNT(*) FROM links WHERE from_document_id=?",
                        (source,)).fetchone()[0] == 2
    for seen, listed in _agreement(conn, [source, target]).values():
        assert seen == listed == 1


def test_related_and_search_neighbour_counts_ignore_a_deleted_neighbour(env):
    conn, cfg = env
    target = write(conn, cfg, "Target", "## A\nbody\n", repo="r")
    source = write(conn, cfg, "Source", f"## A\n[[{target}]]\n", repo="r")
    conn.execute("UPDATE documents SET deleted_at='x' WHERE id=?", (target,))
    conn.commit()
    assert _agreement(conn, [source])[source] == (0, 0)


def test_related_entries_carry_no_path_and_no_default_fields(env):
    """The caller gets knowledge_get for the path; every entry is paid for in context."""
    conn, cfg = env
    target = write(conn, cfg, "Target", "## A\nbody\n", repo="r")
    source = write(conn, cfg, "Source", f"## A\n[[{target}]]\n", repo="r")
    entry, = related(conn, source)
    assert entry == {"id": target, "title": "Target", "direction": "outbound"}
    assert all("path" not in n for n in graph(conn, source, depth=2))
