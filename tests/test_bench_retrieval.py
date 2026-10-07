"""The retrieval benchmark's query set and harness must be self-consistent.

These tests guard the parts that would otherwise rot: every expected title resolves to
exactly one document in the fixture corpus, the paraphrase case is present, and the
harness degrades loudly rather than reporting a missing mode as a number.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location(
    "bench_retrieval", ROOT / "scripts" / "bench-retrieval.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


@pytest.fixture
def env(tmp_path):
    conn, cfg = bench.build_env(tmp_path)
    yield conn, cfg
    conn.close()


def test_query_set_starts_with_the_paraphrase_case():
    """The case where fusion pays is the first entry, so it is never dropped when someone
    trims the set."""
    queries = bench.load_queries()
    assert queries[0] == ("who puts books back in their proper place", "shelving rules")


def test_every_expected_title_resolves_to_exactly_one_document(env):
    """Ground truth is by document title (ids are assigned at index time). A title
    that resolves to zero documents would silently measure nothing; two would be
    ambiguous."""
    conn, _ = env
    queries = bench.load_queries()
    expected = bench.resolve_expected(conn, queries)
    assert len(expected) == len(queries)
    assert len(set(expected)) == len(expected)


def test_paraphrase_target_is_indexed(env):
    conn, _ = env
    ids = bench.resolve_expected(conn, bench.load_queries())
    title = conn.execute("SELECT title FROM documents WHERE id=?", (ids[0],)).fetchone()
    assert title["title"] == "shelving rules"


def test_lexical_ranks_are_one_based_with_miss_none(env):
    conn, _ = env
    queries = bench.load_queries()
    expected = bench.resolve_expected(conn, queries)
    ranked = bench.run_lexical(conn, queries)
    r = bench.ranks(ranked, expected)
    assert all(v is None or 1 <= v <= bench.DEPTH for v in r)
    assert bench.mrr(ranked, expected) <= 1.0


def test_mrr_is_zero_when_nothing_is_found():
    assert bench.mrr([["a"], ["b"]], ["x", "y"]) == 0.0


def test_mrr_awards_reciprocal_rank():
    assert bench.mrr([["a", "b", "c"]], ["c"]) == 1 / 3


def test_index_vectors_reports_a_reason_when_unavailable(env, monkeypatch):
    """Degrade loudly: a machine without the extension must be told the vector modes
    were skipped, not shown a number that pretends to be fused."""
    conn, cfg = env
    from akasha import vectors

    monkeypatch.setattr(vectors, "available", lambda conn: False)
    reason = bench.index_vectors(conn, cfg)
    assert reason and "unavailable" in reason


def test_main_runs_end_to_end_without_vectors(monkeypatch, tmp_path, capsys):
    """The whole script must run on a machine with no vector support and say so."""
    from akasha import vectors

    monkeypatch.setattr(vectors, "available", lambda conn: False)
    rc = bench.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "SKIPPED" in out
    assert "MRR lexical" in out
    assert "MRR   fused  skipped" in out