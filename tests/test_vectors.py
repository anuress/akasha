"""Vector search is optional everywhere: absent extension, absent model, absent config.

The suite must not download a 30MB model, so every test injects a deterministic stub
embedder. What is exercised here is the plumbing — storage, retrieval, fusion and the
degradation path — not the quality of any particular embedding model.
"""
from __future__ import annotations

import numpy as np
import pytest

from akasha.config import load_config
from akasha.db import connect
from docs import add_doc


def _stub_embedder(dim: int = 8):
    """Deterministic vectors from word overlap, so 'nearest' is predictable in a test."""
    vocab = ["cache", "session", "record", "renewal", "test", "queue", "worker", "review"]

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


@pytest.fixture
def env(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")
    cfg.knowledge_dir = tmp_path / "knowledge"
    cfg.knowledge_dir.mkdir()
    cfg.index_roots = []
    cfg.embeddings_provider = "model2vec"
    conn = connect(tmp_path / "s.db")
    add_doc(conn, cfg, "Session cache miss", "## Why\nthe session cache was empty after a restart",
            repo="sample-repo")
    add_doc(conn, cfg, "Renewal guard", "## Why\nthe renewal request needs a limit check",
            repo="sample-repo")
    add_doc(conn, cfg, "Queue worker notes", "## Why\nthe queue worker runs in its own process",
            repo="other-repo")
    return conn, cfg


def test_search_still_works_when_vectors_are_unavailable(env, monkeypatch):
    """A machine whose python was built without extension support must still search:
    degraded, not broken."""
    from akasha import vectors
    from akasha.search import search

    monkeypatch.setattr(vectors, "available", lambda conn: False)
    conn, cfg = env
    hits = search(conn, "session cache", all_repos=True)
    assert hits, "FTS must answer even with no vector support"


def test_available_never_raises_on_a_build_without_extension_support():
    """The failure modes are AttributeError (python built without support) and
    OperationalError (binary missing). Neither may reach the caller."""
    from akasha import vectors

    class NoExtensions:
        def enable_load_extension(self, _flag):
            raise AttributeError("built without --enable-loadable-sqlite-extensions")

    assert vectors.available(NoExtensions()) is False


def test_indexing_stores_one_vector_per_chunk(env):
    from akasha import vectors

    conn, cfg = env
    if not vectors.available(conn):
        pytest.skip("sqlite-vec not loadable here")

    count = vectors.index_chunks(conn, cfg, encode=_stub_embedder(), dim=8)
    chunks = conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]
    assert count == chunks
    stored = conn.execute("SELECT COUNT(*) c FROM vec_chunks").fetchone()["c"]
    assert stored == chunks


def test_forget_drops_the_vectors_of_the_chunks_it_deleted(env):
    """nearest() reads chunk ids out of vec_chunks without joining chunks, and the fusion
    in search.py drops the ones that resolve to nothing, so each orphan would spend a
    slot in the k nearest and contribute no result."""
    from pathlib import Path

    from akasha import vectors
    from akasha.index import forget

    conn, cfg = env
    if not vectors.available(conn):
        pytest.skip("sqlite-vec not loadable here")

    vectors.index_chunks(conn, cfg, encode=_stub_embedder(), dim=8)
    doomed = conn.execute(
        "SELECT path FROM documents WHERE title = 'Session cache miss'").fetchone()["path"]
    live = conn.execute(
        "SELECT COUNT(*) c FROM chunks WHERE document_id NOT IN"
        " (SELECT id FROM documents WHERE path = ?)", (doomed,)).fetchone()["c"]

    assert forget(conn, Path(doomed)) == 1
    stored = conn.execute("SELECT COUNT(*) c FROM vec_chunks").fetchone()["c"]
    assert stored == live


def test_indexing_is_idempotent(env):
    """Re-indexing must not duplicate points, or every rebuild doubles the store."""
    from akasha import vectors

    conn, cfg = env
    if not vectors.available(conn):
        pytest.skip("sqlite-vec not loadable here")

    vectors.index_chunks(conn, cfg, encode=_stub_embedder(), dim=8)
    vectors.index_chunks(conn, cfg, encode=_stub_embedder(), dim=8)
    stored = conn.execute("SELECT COUNT(*) c FROM vec_chunks").fetchone()["c"]
    assert stored == conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]


def test_nearest_finds_the_semantically_closest_chunk(env):
    from akasha import vectors

    conn, cfg = env
    if not vectors.available(conn):
        pytest.skip("sqlite-vec not loadable here")

    vectors.index_chunks(conn, cfg, encode=_stub_embedder(), dim=8)
    ids = vectors.nearest(conn, "session cache", limit=1, encode=_stub_embedder(), dim=8)
    assert ids
    body = conn.execute("SELECT body FROM chunks WHERE id=?", (ids[0],)).fetchone()["body"]
    assert "session" in body.lower()


def test_hybrid_search_returns_documents_from_both_halves(env, monkeypatch):
    """RRF fuses ranks, so a document either retriever found can surface. The point of
    the hybrid is the paraphrase case FTS misses entirely."""
    from akasha import vectors
    from akasha.search import search

    conn, cfg = env
    if not vectors.available(conn):
        pytest.skip("sqlite-vec not loadable here")
    vectors.index_chunks(conn, cfg, encode=_stub_embedder(), dim=8)
    monkeypatch.setattr(vectors, "_encoder", lambda: _stub_embedder())
    monkeypatch.setattr(vectors, "DIM", 8)

    hits = search(conn, "renewal guard", all_repos=True, cfg=cfg)
    assert hits
    assert any("renewal" in h.text.lower() for h in hits)


def test_vector_search_is_off_when_the_provider_is_none(env):
    """Embedding costs time and disk. `provider = "none"` must mean no vector work at
    all, not a silent download."""
    from akasha import vectors

    conn, cfg = env
    cfg.embeddings_provider = "none"
    assert vectors.enabled(cfg) is False


def test_nearest_returns_empty_when_the_encoder_cannot_load(env, monkeypatch):
    """A machine whose model2vec chain is broken must not kill search: `nearest`
    returns nothing and records the cause for doctor, it does not propagate."""
    from akasha import vectors

    conn, cfg = env
    if not vectors.available(conn):
        pytest.skip("sqlite-vec not loadable here")
    vectors.index_chunks(conn, cfg, encode=_stub_embedder(), dim=8)

    monkeypatch.setattr(vectors, "_encoder",
                        lambda: (_ for _ in ()).throw(ImportError("No module named 'missing.module'")))
    assert vectors.nearest(conn, "session cache", encode=None, dim=8) == []
    assert vectors.degradation(conn) == "ImportError: No module named 'missing.module'"


def test_search_degrades_to_lexical_when_the_encoder_cannot_load(env, monkeypatch):
    """A dead encoder must return the same lexical hits as vectors being disabled, not
    raise out of the MCP tool."""
    from akasha import vectors
    from akasha.search import search

    conn, cfg = env
    if not vectors.available(conn):
        pytest.skip("sqlite-vec not loadable here")
    vectors.index_chunks(conn, cfg, encode=_stub_embedder(), dim=8)
    monkeypatch.setattr(vectors, "DIM", 8)
    monkeypatch.setattr(vectors, "_encoder",
                        lambda: (_ for _ in ()).throw(ImportError("No module named 'missing.module'")))

    degraded = search(conn, "session cache", all_repos=True, cfg=cfg)
    assert degraded, "FTS must answer even with a broken encoder"

    monkeypatch.setattr(vectors, "available", lambda conn: False)
    lexical = search(conn, "session cache", all_repos=True, cfg=cfg)
    assert [h.document_id for h in degraded] == [h.document_id for h in lexical]


def test_index_chunks_returns_zero_when_the_encoder_cannot_load(env, monkeypatch):
    """`index_chunks` must report "nothing stored" rather than raising when the
    embedder cannot load, or a reindex kills the whole command."""
    from akasha import vectors

    conn, cfg = env
    if not vectors.available(conn):
        pytest.skip("sqlite-vec not loadable here")
    monkeypatch.setattr(vectors, "_encoder",
                        lambda: (_ for _ in ()).throw(ImportError("No module named 'missing.module'")))

    assert vectors.index_chunks(conn, cfg, dim=8) == 0
    assert "ImportError" in vectors.degradation(conn)


def test_a_successful_encode_clears_the_degradation_record(env, monkeypatch):
    """A record that survives a fix would make doctor report a degradation that no longer
    exists, so the first successful run must forget the old failure."""
    from akasha import vectors

    conn, cfg = env
    if not vectors.available(conn):
        pytest.skip("sqlite-vec not loadable here")
    vectors.index_chunks(conn, cfg, encode=_stub_embedder(), dim=8)

    monkeypatch.setattr(vectors, "_encoder",
                        lambda: (_ for _ in ()).throw(ImportError("No module named 'missing.module'")))
    assert vectors.nearest(conn, "session cache", encode=None, dim=8) == []
    assert vectors.degradation(conn) is not None

    monkeypatch.setattr(vectors, "_encoder", lambda: _stub_embedder())
    assert vectors.nearest(conn, "session cache", encode=None, dim=8)
    assert vectors.degradation(conn) is None


def test_include_archived_reaches_a_document_only_the_dense_side_found(env, monkeypatch):
    """`include_archived` must mean the same on both sides: the lexical filter applies it
    in SQL, and `_hit_for_chunk`, the path a document only vectors found comes back
    through, must not reject an archived document unconditionally. The dense side is
    where an archived document is most likely to be the only answer."""
    from akasha import vectors
    from akasha.search import search

    conn, cfg = env
    conn.execute("UPDATE documents SET status='archived' WHERE title='Renewal guard'")
    conn.commit()
    chunk = conn.execute(
        "SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document_id"
        " WHERE d.title = 'Renewal guard'").fetchone()["id"]
    monkeypatch.setattr(vectors, "enabled", lambda cfg: True)
    monkeypatch.setattr(vectors, "nearest", lambda *a, **k: [chunk])

    # A query no chunk matches lexically, so the dense side is the only way in.
    asked = search(conn, "unmatchableqqq", all_repos=True, cfg=cfg, include_archived=True)
    assert [h.id for h in asked] == [chunk]

    unasked = search(conn, "unmatchableqqq", all_repos=True, cfg=cfg)
    assert unasked == [], "the flag still has to mean something when it is not passed"


def _counting(encode):
    """Wrap an embedder so a test can see how many texts it was asked to embed."""
    calls = []

    def wrapped(texts):
        calls.append(len(texts))
        return encode(texts)

    return wrapped, calls


def test_a_second_index_with_no_changed_chunk_embeds_nothing(env):
    """Vectors are keyed by chunk id and chunk ids survive until a file changes, so a
    no-op reindex has nothing to embed and must not pay for a rewrite."""
    from akasha import vectors

    conn, cfg = env
    if not vectors.available(conn):
        pytest.skip("sqlite-vec not loadable here")
    encode, calls = _counting(_stub_embedder())
    vectors.index_chunks(conn, cfg, encode=encode, dim=8)
    calls.clear()

    assert vectors.index_chunks(conn, cfg, encode=encode, dim=8) == 3
    assert calls == []


def test_only_new_chunks_are_embedded_and_removed_ones_are_dropped(env):
    from akasha import vectors

    conn, cfg = env
    if not vectors.available(conn):
        pytest.skip("sqlite-vec not loadable here")
    encode, calls = _counting(_stub_embedder())
    vectors.index_chunks(conn, cfg, encode=encode, dim=8)
    calls.clear()

    add_doc(conn, cfg, "Review notes", "## Why\nthe review needs a test", repo="sample-repo")
    conn.execute("DELETE FROM chunks WHERE document_id IN"
                 " (SELECT id FROM documents WHERE title='Renewal guard')")
    conn.commit()
    vectors.index_chunks(conn, cfg, encode=encode, dim=8)

    assert calls == [1]
    stored = {r["chunk_id"] for r in conn.execute("SELECT chunk_id FROM vec_chunks")}
    assert stored == {r["id"] for r in conn.execute("SELECT id FROM chunks")}


def test_a_provider_change_rebuilds_every_vector(env):
    """Vectors from one model mean nothing to another, so the diff path is only valid
    while the model that wrote them is still the one in use."""
    from akasha import vectors

    conn, cfg = env
    if not vectors.available(conn):
        pytest.skip("sqlite-vec not loadable here")
    encode, calls = _counting(_stub_embedder())
    vectors.index_chunks(conn, cfg, encode=encode, dim=8)
    calls.clear()

    cfg.embeddings_provider = "fastembed"
    vectors.index_chunks(conn, cfg, encode=encode, dim=8)
    assert calls == [3]


def test_a_dimension_change_rebuilds_the_table(env):
    from akasha import vectors

    conn, cfg = env
    if not vectors.available(conn):
        pytest.skip("sqlite-vec not loadable here")
    vectors.index_chunks(conn, cfg, encode=_stub_embedder(8), dim=8)

    assert vectors.index_chunks(conn, cfg, encode=_stub_embedder(4), dim=4) == 3
    assert conn.execute("SELECT COUNT(*) c FROM vec_chunks").fetchone()["c"] == 3


def test_deleting_the_last_live_document_empties_the_vector_table(env, tmp_path, monkeypatch):
    """The early return on "no live chunks" used to skip orphan cleanup, so the vectors of
    the final deleted document stayed behind for nearest() to spend its k on."""
    from akasha import vectors
    from akasha.config import IndexRoot
    from akasha.index import index_all

    conn, cfg = env
    if not vectors.available(conn):
        pytest.skip("sqlite-vec not loadable here")
    narrow = _stub_embedder()
    # index_all embeds at the default width, so pad the stub up to it.
    monkeypatch.setattr(
        vectors, "_encoder",
        lambda: lambda texts: np.pad(narrow(texts), ((0, 0), (0, vectors.DIM - 8))))
    conn.execute("DELETE FROM chunks")
    conn.execute("DELETE FROM documents")
    conn.commit()
    for fixture_doc in cfg.knowledge_dir.glob("*.md"):
        fixture_doc.unlink()
    root = tmp_path / "notes"
    root.mkdir()
    note = root / "only.md"
    note.write_text("## Why\nthe session cache was empty\n")
    cfg.index_roots = [IndexRoot(path=str(root), source="notes")]

    index_all(conn, cfg)
    assert conn.execute("SELECT COUNT(*) c FROM vec_chunks").fetchone()["c"] == 1
    note.unlink()
    index_all(conn, cfg)

    assert conn.execute("SELECT COUNT(*) c FROM vec_chunks").fetchone()["c"] == 0


def test_a_failed_rebuild_does_not_leave_old_model_vectors_searchable(env):
    """After a provider change the stored vectors belong to the old model. If the new
    encoder fails, serving them would rank by a meaning the query no longer shares."""
    from akasha import vectors

    conn, cfg = env
    if not vectors.available(conn):
        pytest.skip("sqlite-vec not loadable here")
    vectors.index_chunks(conn, cfg, encode=_stub_embedder(), dim=8)
    assert vectors.nearest(conn, "session cache", encode=_stub_embedder(), dim=8)

    def broken(_texts):
        raise RuntimeError("model unavailable")

    cfg.embeddings_provider = "fastembed"
    vectors.index_chunks(conn, cfg, encode=broken, dim=8)

    assert vectors.nearest(conn, "session cache", encode=_stub_embedder(), dim=8) == []


def test_a_dimension_change_rebuilds_every_vector(env):
    """The dimension is part of the signature: vectors of another width cannot be mixed
    into the same table."""
    from akasha import vectors

    conn, cfg = env
    if not vectors.available(conn):
        pytest.skip("sqlite-vec not loadable here")
    encode, calls = _counting(_stub_embedder(8))
    vectors.index_chunks(conn, cfg, encode=encode, dim=8)
    calls.clear()

    vectors.index_chunks(conn, cfg, encode=_counting(_stub_embedder(4))[0], dim=4)
    stored = conn.execute("SELECT COUNT(*) c FROM vec_chunks").fetchone()["c"]
    assert stored == conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]


def test_status_reports_count_stored_and_expected_signature_without_a_table(env):
    """Doctor reads vector state through this, not through private names."""
    from akasha import vectors

    conn, cfg, *_ = env
    cfg.embeddings_provider = "model2vec"
    state = vectors.status(conn, cfg)
    assert state["count"] == 0 and state["stored"] is None
    assert state["expected"].startswith("model2vec|")


def _fake_model2vec(monkeypatch, cached, delay=0.0):
    """Stand in for model2vec: record every load, and say whether the hub cache has it."""
    import time

    import model2vec
    import model2vec.persistence.hf as hf

    from akasha import vectors

    loads = []
    monkeypatch.setattr(vectors, "_model", None)
    monkeypatch.setattr(hf, "maybe_get_cached_model_path", lambda model_id: cached)
    monkeypatch.setattr(
        model2vec.StaticModel, "from_pretrained",
        classmethod(lambda cls, path, **kw: loads.append((str(path), kw)) or time.sleep(delay) or
                    type("M", (), {"encode": staticmethod(_stub_embedder())})()))
    return loads


def test_cached_model_loads_from_disk_without_the_network(monkeypatch, tmp_path):
    """The library default force-downloads on every load; a cached model must be read
    from its snapshot directory so a start costs no hub request."""
    from akasha import vectors

    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    loads = _fake_model2vec(monkeypatch, snapshot)
    vectors._encoder()
    assert loads == [(str(snapshot), {"force_download": False})]


def test_uncached_model_downloads_once_and_says_so(env, monkeypatch):
    from akasha import events, vectors

    conn, cfg = env
    loads = _fake_model2vec(monkeypatch, None)
    assert vectors._embed(conn, None, ["x"]) is not None
    vectors._embed(conn, None, ["y"])
    assert loads == [(vectors.MODEL, {"force_download": False})]
    assert [e["kind"] for e in events.recent(conn, kind="model.downloaded")] == ["model.downloaded"]


def test_two_threads_load_the_model_once(monkeypatch):
    """The warm thread and a first search can reach _encoder together."""
    import threading

    from akasha import vectors

    loads = _fake_model2vec(monkeypatch, None, delay=0.2)
    threads = [threading.Thread(target=vectors._encoder) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(loads) == 1


def test_model_load_keeps_hub_progress_bars_off_stderr(monkeypatch):
    """A download draws a tqdm bar on stderr; hook callers would see it as noise."""
    from huggingface_hub.utils import are_progress_bars_disabled, enable_progress_bars

    from akasha import vectors

    enable_progress_bars()
    _fake_model2vec(monkeypatch, None)
    vectors._encoder()
    assert are_progress_bars_disabled()
