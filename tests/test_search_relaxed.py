import pytest

from akasha.config import load_config
from akasha.db import connect
from akasha.search import search
from docs import add_doc


@pytest.fixture
def env(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")
    cfg.knowledge_dir = tmp_path / "knowledge"
    cfg.knowledge_dir.mkdir()
    cfg.index_roots = []
    conn = connect(tmp_path / "s.db")
    add_doc(conn, cfg, "Session cache miss", "## Cause\ncaching was globally off\n",
            repo="sample-repo", feature="catalog-cache")
    add_doc(conn, cfg, "Loan limit sweep", "## Result\nlimit 30 wins on low end\n",
            repo="sample-repo", feature="loan-limits")
    return conn, cfg


def test_strict_match_still_wins_when_all_terms_present(env):
    conn, _ = env
    hits = search(conn, "caching globally off", all_repos=True)
    assert hits
    assert hits[0].relaxed is False


def test_zero_result_and_query_falls_back_to_or(env):
    conn, _ = env
    # "sweep" and "caching" never co-occur in one chunk
    hits = search(conn, "caching sweep", all_repos=True)
    assert hits, "auto mode should degrade rather than return nothing"
    assert all(h.relaxed for h in hits)


def test_explicit_all_mode_does_not_fall_back(env):
    conn, _ = env
    assert search(conn, "caching sweep", all_repos=True, match_mode="all") == []


def test_explicit_any_mode_is_relaxed_from_the_start(env):
    conn, _ = env
    hits = search(conn, "caching sweep", all_repos=True, match_mode="any")
    assert hits and all(h.relaxed for h in hits)


def test_a_repo_scope_widens_but_source_and_kind_stay_hard(env):
    """The repo scope is a preference: knowledge is deliberately cross-repo, and an agent
    asking from the wrong checkout should get the answer, marked, rather than silence.
    source and kind are genuine filters and still return nothing."""
    conn, _ = env
    widened = search(conn, "caching sweep", repo="sample-repo-b")
    assert widened, "the repo scope widens rather than returning nothing"
    assert all(h.cross_repo for h in widened), "and every widened hit says so"

    assert search(conn, "caching sweep", source="nonexistent-source") == []
    assert search(conn, "caching sweep", kind="nonexistent-kind") == []


def test_a_genuinely_absent_term_returns_nothing_even_relaxed(env):
    conn, _ = env
    assert search(conn, "kubernetes helm istio", all_repos=True) == []


def test_relaxed_hits_rank_below_strict_ones_for_the_same_query(env):
    conn, _ = env
    strict = search(conn, "limit sweep", all_repos=True)
    assert strict and strict[0].relaxed is False


def test_a_strict_hit_anywhere_beats_a_loose_hit_in_this_repo(env):
    """A relaxed in-scope match must not be preferred over a strict match recorded under
    another repo: loose matches in the current repo are what an agent cannot tell apart
    from real answers."""
    conn, cfg = env
    # Shares one query term ("off"), so a relaxed in-scope match exists and is wrong.
    add_doc(conn, cfg, "Stale records", "## Why\noverdue notices are hard to head off\n",
            repo="other-repo", feature="tests")

    hits = search(conn, "caching globally off", repo="other-repo")

    assert hits[0].relaxed is False, "a strict match elsewhere outranks a loose one here"
    assert hits[0].repo == "sample-repo"
    assert hits[0].cross_repo is True
