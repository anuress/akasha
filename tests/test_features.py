import pytest

from akasha.db import connect
from akasha.features import add_alias, list_features, resolve_feature


@pytest.fixture
def conn(tmp_path):
    return connect(tmp_path / "s.db")


def test_resolve_creates_feature_once(conn):
    a = resolve_feature(conn, "proj-100", repo="sample-repo")
    b = resolve_feature(conn, "proj-100", repo="sample-repo")
    assert a == b
    assert len(list_features(conn)) == 1


def test_resolve_is_case_insensitive(conn):
    a = resolve_feature(conn, "PROJ-101")
    b = resolve_feature(conn, "proj-101")
    assert a == b


def test_alias_resolves_to_canonical_feature(conn):
    canonical = resolve_feature(conn, "summer-reading-program")
    add_alias(conn, "summer-reading-program", "summer-reading")
    assert resolve_feature(conn, "summer-reading", create=False) == canonical
    assert len(list_features(conn)) == 1


def test_resolve_without_create_returns_none(conn):
    assert resolve_feature(conn, "never-seen", create=False) is None


def test_list_features_filters_by_status(conn):
    resolve_feature(conn, "alive")
    fid = resolve_feature(conn, "old")
    conn.execute("UPDATE features SET status='shelved' WHERE id=?", (fid,))
    assert [f["slug"] for f in list_features(conn, status="active")] == ["alive"]


def test_show_counts_live_documents_only(conn):
    from akasha.features import show

    fid = resolve_feature(conn, "fines")
    for i, deleted in enumerate((None, None, "2026-01-01")):
        conn.execute(
            "INSERT INTO documents (id, source, path, feature_id, deleted_at, created_at,"
            " updated_at) VALUES (?, 'native', ?, ?, ?, '', '')",
            (f"k_{i}", f"/p/{i}.md", fid, deleted))
    assert show(conn, "fines") == {"slug": "fines", "documents": 2}


def test_show_follows_an_alias_and_names_the_canonical_slug(conn):
    from akasha.features import show

    resolve_feature(conn, "fines-rewrite")
    add_alias(conn, "fines-rewrite", "fines")
    assert show(conn, "fines")["slug"] == "fines-rewrite"


def test_show_on_an_unknown_feature_is_refused(conn):
    from akasha.features import show

    with pytest.raises(ValueError, match="unknown feature"):
        show(conn, "never-seen")
