#!/usr/bin/env python3
"""Run the retrieval MRR benchmark: lexical only, vectors only, and fused.

It builds a fresh database from the committed fixture corpus
(tests/fixtures/retrieval-bench/corpus), runs the committed query set
(tests/fixtures/retrieval-bench/queries.py) through search() three ways, and prints MRR
per mode. Ground truth is by document title, because document ids are assigned at index
time.

Vectors degrade loudly: when the extension or model is unavailable, the vector modes are
skipped with the reason stated, never reported as a lower number pretending to be fused.
Run with the vectors extra:

    uv run --extra dev --extra vectors python3 scripts/bench-retrieval.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from akasha.config import IndexRoot, load_config
from akasha.db import connect
from akasha.index import index_all

CORPUS = ROOT / "tests" / "fixtures" / "retrieval-bench" / "corpus"
QUERIES_PY = ROOT / "tests" / "fixtures" / "retrieval-bench" / "queries.py"

# Deep enough that a paraphrase target ranked low lexically still shows up, which is the
# case fusion exists for.
DEPTH = 10


def load_queries(path: Path = QUERIES_PY) -> list[tuple[str, str]]:
    import importlib.util

    spec = importlib.util.spec_from_file_location("retrieval_bench_queries", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(mod.QUERIES)


def build_env(workdir: Path, corpus: Path = CORPUS):
    """Fresh database from the fixture corpus, lexical index only."""
    cfg = load_config(workdir / "absent.toml")
    cfg.knowledge_dir = workdir / "knowledge"
    cfg.knowledge_dir.mkdir()
    cfg.index_roots = [IndexRoot(path=str(corpus), source="fixture")]
    conn = connect(workdir / "bench.db")
    index_all(conn, cfg)
    return conn, cfg


def resolve_expected(conn, queries: list[tuple[str, str]]) -> list[str]:
    """Ground truth by title: each expected title must resolve to exactly one document."""
    ids: list[str] = []
    for _query, title in queries:
        rows = conn.execute(
            "SELECT id FROM documents WHERE title = ?", (title,)).fetchall()
        if len(rows) != 1:
            raise ValueError(
                f"expected title {title!r} resolves to {len(rows)} documents, not one")
        ids.append(rows[0]["id"])
    return ids


def index_vectors(conn, cfg) -> str | None:
    """Embed the fixture corpus. None when the vectors are ready, else the reason."""
    from akasha import vectors

    if not vectors.available(conn):
        return "sqlite-vec extension unavailable on this python"
    cfg.embeddings_provider = "model2vec"
    try:
        count = vectors.index_chunks(conn, cfg)
    except Exception as exc:                    # noqa: BLE001 - report, never crash
        return f"embedding failed: {exc}"
    if count == 0:
        return "no chunks were embedded"
    return None


def run_lexical(conn, queries: list[tuple[str, str]], depth: int = DEPTH) -> list[list[str]]:
    from akasha.search import search

    out = []
    for q, _ in queries:
        hits = search(conn, q, all_repos=True, limit=depth, cfg=None)
        out.append([h.document_id for h in hits])
    return out


def run_vectors(conn, queries: list[tuple[str, str]], depth: int = DEPTH) -> list[list[str]]:
    from akasha import vectors

    out = []
    for q, _ in queries:
        chunk_ids = vectors.nearest(conn, q, limit=depth)
        docs: list[str] = []
        for cid in chunk_ids:
            row = conn.execute(
                "SELECT document_id FROM chunks WHERE id = ?", (cid,)).fetchone()
            if row and row["document_id"] not in docs:
                docs.append(row["document_id"])
        out.append(docs)
    return out


def run_fused(conn, cfg, queries: list[tuple[str, str]], depth: int = DEPTH) -> list[list[str]]:
    from akasha.search import search

    out = []
    for q, _ in queries:
        hits = search(conn, q, all_repos=True, limit=depth, cfg=cfg)
        out.append([h.document_id for h in hits])
    return out


def mrr(ranked: list[list[str]], expected: list[str]) -> float:
    """Mean reciprocal rank. A target missing from the list contributes zero."""
    total = 0.0
    for rank_list, exp in zip(ranked, expected):
        try:
            total += 1.0 / (rank_list.index(exp) + 1)
        except ValueError:
            pass
    return total / len(ranked)


def ranks(ranked: list[list[str]], expected: list[str]) -> list[int | None]:
    return [lst.index(exp) + 1 if exp in lst else None
            for lst, exp in zip(ranked, expected)]


def main() -> int:
    queries = load_queries()

    with tempfile.TemporaryDirectory() as tmp:
        conn, cfg = build_env(Path(tmp))
        expected = resolve_expected(conn, queries)
        reason = index_vectors(conn, cfg)

        results = {
            "lexical": run_lexical(conn, queries),
            "vectors": run_vectors(conn, queries) if reason is None else None,
            "fused": run_fused(conn, cfg, queries) if reason is None else None,
        }
        conn.close()

    if reason is not None:
        print(f"VECTOR MODES SKIPPED: {reason}")
        print("lexical runs alone; vectors-only and fused are NOT reported.\n")

    print(f"depth {DEPTH}  |  {len(queries)} queries  |  ground truth by document title\n")
    headers = ["query", "lexical", "vectors", "fused"]
    print(f"{headers[0]:<38} {headers[1]:>8} {headers[2]:>8} {headers[3]:>8}")
    per_mode = {m: (ranks(results[m], expected) if results[m] is not None
                    else [None] * len(queries))
                for m in ("lexical", "vectors", "fused")}
    for (q, title), *cells in zip(
            queries, per_mode["lexical"], per_mode["vectors"], per_mode["fused"]):
        row = [title[:38]] + [str(r) if r is not None else "-" for r in cells]
        print(f"{row[0]:<38} {row[1]:>8} {row[2]:>8} {row[3]:>8}")

    print()
    for mode in ("lexical", "vectors", "fused"):
        if results[mode] is None:
            print(f"MRR {mode:>7}  skipped")
        else:
            print(f"MRR {mode:>7}  {mrr(results[mode], expected):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())