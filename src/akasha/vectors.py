"""Optional dense retrieval, stored in the same SQLite file.

Vectors are a supplement, not a replacement: BM25 is very strong on engineering notes
because they are full of rare exact identifiers, which is what IDF rewards. What lexical
cannot do is paraphrase, and fusing the two rankings closes that gap.

Everything here is optional and degrades to lexical-only, because the extension can be
unavailable on a healthy machine. `doctor` reports when that happens, since a silent drop
in retrieval quality cannot be seen from outside.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import sqlite3
import threading
import warnings

from akasha.config import Config

MODEL = "minishlab/potion-base-8M"
DIM = 256
# sqlite-vec is brute force, which is the right algorithm at this scale: tens of thousands
# of vectors are one pass over a small matrix. An ANN index only pays off in the millions.
TABLE = "vec_chunks"

# Where an encoder failure is recorded, for doctor to read back through degradation().
DEGRADED_KEY = "vectors.degraded"

# What the stored vectors were made by; a mismatch forces a full rebuild.
SIGNATURE_KEY = "vectors.signature"

_model = None
# Set when the model had to be fetched from the hub; _embed records it once as an event.
_downloaded = False
_model_lock = threading.Lock()


def enabled(cfg: Config) -> bool:
    """Whether the operator asked for vectors at all. Embedding costs time and disk."""
    return (cfg.embeddings_provider or "none") != "none"


def extra_installed() -> bool:
    """Whether the `[vectors]` extra's packages import. Says nothing about the sqlite
    extension loading; `available` covers that."""
    return not missing_modules()


def missing_modules() -> list[str]:
    return [m for m in ("sqlite_vec", "model2vec", "numpy")
            if importlib.util.find_spec(m) is None]


def available(conn: sqlite3.Connection) -> bool:
    """Whether this connection can run vector search. Never raises.

    Two independent things can be missing and both happen on healthy machines: a python
    built without `--enable-loadable-sqlite-extensions` (it is off in several common
    builds, because loading an extension is arbitrary native code in-process), and the
    platform binary itself.
    """
    try:
        import sqlite_vec
    except ImportError:
        return False
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
    except (AttributeError, sqlite3.OperationalError, sqlite3.DatabaseError):
        return False
    finally:
        # Close it again at once: while open, any SQL on this connection can load native
        # code, and agents reach this database through MCP.
        try:
            conn.enable_load_extension(False)
        except (AttributeError, sqlite3.OperationalError):
            pass
    return True


def _encoder():
    """The default embedder, loaded once per process and only when used."""
    global _model
    if _model is None:
        # The warm thread and a first search can arrive together; loading twice would
        # double the cost this is meant to hide.
        with _model_lock:
            if _model is None:
                from model2vec import StaticModel

                from huggingface_hub.utils import disable_progress_bars
                from model2vec.persistence.hf import maybe_get_cached_model_path

                # A download's progress bar goes to stderr, which hook callers read.
                disable_progress_bars()

                # from_pretrained force-downloads by default, one hub request per load.
                # A cached snapshot is loaded by path; only a missing model is fetched.
                cached = maybe_get_cached_model_path(MODEL)
                # The hub client warns about unauthenticated requests; a hook's stderr
                # is read by agents, who cannot act on it. Level restored afterwards so
                # other errors from the library still surface.
                hub_log = logging.getLogger("huggingface_hub")
                previous = hub_log.level
                hub_log.setLevel(logging.ERROR)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _model = StaticModel.from_pretrained(
                            cached or MODEL, force_download=False)
                finally:
                    hub_log.setLevel(previous)
                global _downloaded
                _downloaded = cached is None
    return _model.encode


def _record_degradation(conn: sqlite3.Connection, exc: BaseException) -> None:
    """Persist why the encoder failed, so a later doctor run can say it. Without it,
    fusion quietly becoming lexical-only has no symptom but "results feel worse"."""
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?)"
        " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (DEGRADED_KEY, json.dumps({"class": type(exc).__name__, "message": str(exc)})))
    conn.commit()


def _record_download(conn: sqlite3.Connection) -> None:
    """Say once that the model came from the network rather than the local cache."""
    global _downloaded
    if _downloaded:
        _downloaded = False
        from akasha.events import emit

        emit(conn, "model.downloaded", model=MODEL)


def _clear_degradation(conn: sqlite3.Connection) -> None:
    """Forget a recorded failure once the encoder runs again; it is no longer true."""
    if conn.execute("SELECT 1 FROM meta WHERE key=?", (DEGRADED_KEY,)).fetchone():
        conn.execute("DELETE FROM meta WHERE key=?", (DEGRADED_KEY,))
        conn.commit()


def probe(conn: sqlite3.Connection) -> str | None:
    """Run the real encode chain on one word. Returns the failure message, or None.

    `enabled` and `available` say nothing about the encoder itself: sqlite-vec can load
    while the model chain is broken. One encode is the cost of the check being real.
    """
    if _embed(conn, None, ["probe"]) is None:
        return degradation(conn) or "encoder failed"
    return None


def degradation(conn: sqlite3.Connection) -> str | None:
    """Why the encoder last failed, "Class: message", for doctor. None while healthy."""
    row = conn.execute("SELECT value FROM meta WHERE key=?", (DEGRADED_KEY,)).fetchone()
    if row is None:
        return None
    try:
        record = json.loads(row["value"])
    except (json.JSONDecodeError, TypeError):
        return None
    return f"{record.get('class')}: {record.get('message')}"


def _embed(conn: sqlite3.Connection, encode, texts):
    """Normalised vectors, or None when the encoder chain is broken.

    Loading the default embedder, downloading its model and running it are one
    optional chain; any failure must cost the caller at most the lexical fallback.
    The failure is recorded so it stays falsifiable, and the record clears on the
    first success so a fix is not reported forever.
    """
    try:
        encode = encode or _encoder()
        vectors = _normalise(encode(texts))
    except Exception as exc:
        _record_degradation(conn, exc)
        return None
    _clear_degradation(conn)
    _record_download(conn)
    return vectors


def _normalise(vectors):
    import numpy as np

    array = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / (norms + 1e-9)


def ensure_table(conn: sqlite3.Connection, dim: int = DIM) -> None:
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {TABLE} USING vec0("
        f" chunk_id TEXT PRIMARY KEY, repo TEXT, embedding float[{dim}])")


def _signature(cfg: Config, dim: int) -> str:
    """What the stored vectors were made by. Vectors from one model mean nothing to
    another, so a change here invalidates every one of them."""
    return f"{cfg.embeddings_provider}|{MODEL}|{dim}"


def status(conn: sqlite3.Connection, cfg: Config) -> dict:
    """What is stored against what the config wants: {count, stored, expected}. `stored`
    is None before the first index."""
    try:
        count = conn.execute(f"SELECT COUNT(*) c FROM {TABLE}").fetchone()["c"]
    except sqlite3.OperationalError:
        count = 0
    row = conn.execute("SELECT value FROM meta WHERE key=?", (SIGNATURE_KEY,)).fetchone()
    return {"count": count, "stored": row["value"] if row else None,
            "expected": _signature(cfg, DIM)}


def _stored_ids(conn: sqlite3.Connection) -> set[str] | None:
    """Chunk ids that have a vector, or None when the table does not exist yet."""
    try:
        return {r["chunk_id"] for r in conn.execute(f"SELECT chunk_id FROM {TABLE}")}
    except sqlite3.OperationalError:
        return None


def index_chunks(conn: sqlite3.Connection, cfg: Config, encode=None, dim: int = DIM) -> int:
    """Bring the vector table in line with the chunks. Returns the number stored.

    Chunk ids are fresh on every index of a document (see `new_id("c")` in index.py), so
    any edit re-embeds that whole document; unchanged documents keep their ids. Only chunks
    without a vector are embedded and vectors without a chunk are dropped, so a no-op
    reindex writes nothing. A change of provider, model or dimension rebuilds every
    vector instead. Vectors are derived data: the markdown is still the truth.
    """
    if not enabled(cfg) or not available(conn):
        return 0
    rows = conn.execute(
        "SELECT c.id, d.repo FROM chunks c"
        " JOIN documents d ON d.id = c.document_id WHERE d.deleted_at IS NULL").fetchall()

    signature = _signature(cfg, dim)
    marker = conn.execute("SELECT value FROM meta WHERE key=?", (SIGNATURE_KEY,)).fetchone()
    stored = _stored_ids(conn)
    rebuild = stored is None or marker is None or marker["value"] != signature
    if rebuild:
        stored = set()
    live = {r["id"] for r in rows}
    todo = [r for r in rows if r["id"] not in stored]

    vectors = None
    if todo:
        # Text is read only for what needs embedding; a no-op run never loads a body.
        text_of: dict[str, str] = {}
        for i in range(0, len(todo), 500):
            batch = [r["id"] for r in todo[i:i + 500]]
            for r in conn.execute(
                    "SELECT id, heading, body FROM chunks WHERE id IN"
                    f" ({','.join('?' * len(batch))})", batch):
                text_of[r["id"]] = ((r["heading"] or "") + "\n" + (r["body"] or ""))[:2000]
        texts = [text_of[r["id"]] for r in todo]
        vectors = _embed(conn, encode, texts)
        if vectors is None:
            if rebuild:
                # The stored vectors belong to a model that is no longer in use; serving
                # them would rank by a meaning the query does not share.
                conn.execute(f"DROP TABLE IF EXISTS {TABLE}")
                conn.execute("DELETE FROM meta WHERE key=?", (SIGNATURE_KEY,))
                conn.commit()
            return 0

    if rebuild:
        conn.execute(f"DROP TABLE IF EXISTS {TABLE}")
    ensure_table(conn, dim)
    conn.executemany(f"DELETE FROM {TABLE} WHERE chunk_id = ?",
                     [(i,) for i in stored - live])
    if todo:
        conn.executemany(
            f"INSERT INTO {TABLE}(chunk_id, repo, embedding) VALUES (?,?,?)",
            [(r["id"], r["repo"] or "", v.tobytes()) for r, v in zip(todo, vectors)])
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?)"
        " ON CONFLICT(key) DO UPDATE SET value=excluded.value", (SIGNATURE_KEY, signature))
    conn.commit()
    return len(rows)


def nearest(conn: sqlite3.Connection, query: str, limit: int = 10,
            repo: str | None = None, encode=None, dim: int = DIM) -> list[str]:
    """Chunk ids closest to the query, nearest first. Empty when unavailable.

    The repo filter is applied inside the query rather than after it — that is the
    reason this lives in SQLite rather than a numpy matrix held in the process.
    """
    if not available(conn):
        return []
    try:
        conn.execute(f"SELECT 1 FROM {TABLE} LIMIT 1")
    except sqlite3.OperationalError:
        return []                       # not indexed yet

    embedding = _embed(conn, encode, [query])
    if embedding is None:
        return []
    vector = embedding[0]
    sql = (f"SELECT chunk_id FROM {TABLE} WHERE embedding MATCH ? AND k = ?")
    params: list = [vector.tobytes(), limit]
    if repo:
        sql += " AND repo = ?"
        params.append(repo)
    try:
        return [r["chunk_id"] for r in conn.execute(sql + " ORDER BY distance", params)]
    except sqlite3.OperationalError:
        return []
