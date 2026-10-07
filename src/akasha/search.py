"""Retrieval. FTS5 always runs; vector results merge in by RRF when they exist."""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, replace
from datetime import date

from akasha.features import resolve_feature
from akasha.links import neighbour_counts
from akasha.markdown import HEADING

# The statuses something writes: `active` is the default, `stale` comes from the stale
# readout, `archived` from archive() — which is also what a supersession does to the
# document it replaces.
#
# An allowlist rather than a blocklist, because a blocklist fails in the wrong direction:
# an unrecognised status (a typo in a document's frontmatter) would pass the filter and
# score at STATUS_FACTOR's 1.0 default, ranking above every correctly-marked document.
SEARCHABLE_STATUSES = ("active", "unsorted", "stale")

# stale sits between unsorted and archived: still found, because nobody has said it is
# wrong, but ranked under anything still in use, since disuse is the only evidence it has
# stopped mattering.
STATUS_FACTOR = {"active": 1.0, "unsorted": 0.9, "stale": 0.6, "archived": 0.3}


@dataclass
class Hit:
    id: str
    document_id: str
    score: float
    repo: str | None
    feature: str | None
    path: str
    heading: str
    text: str
    relaxed: bool = False
    cross_repo: bool = False
    # Resolved links in either direction, so the caller knows when a follow-up
    # knowledge_related hop would pay. Zero says the document sits alone in the graph.
    neighbours: int = 0

    def as_dict(self, withheld_chars: int = 0) -> dict:
        """The shape a caller receives: default-valued fields are omitted, and so is
        `path`, which knowledge_get returns. The attributes themselves stay, because
        in-process callers (the injection event, the terminal listing) read them."""
        out = {"id": self.id, "document_id": self.document_id,
               "score": float(f"{self.score:.3g}"),
               "heading": self.heading, "text": self.text}
        for name, value in (("repo", self.repo), ("feature", self.feature), ("relaxed", self.relaxed),
                            ("cross_repo", self.cross_repo),
                            ("withheld_chars", withheld_chars),
                            ("neighbours", self.neighbours)):
            if value:
                out[name] = value
        return out


def _escape(q: str, mode: str = "all") -> str:
    """FTS5 treats punctuation as syntax, so quote each bare term."""
    terms = [f'"{t}"' for t in q.replace('"', " ").split() if t]
    if not terms:
        return ""
    return (" OR " if mode == "any" else " ").join(terms)


def _run(
    conn: sqlite3.Connection,
    match_expr: str,
    *,
    repo: str | None = None,
    feature: str | None = None,
    source: str | None = None,
    kind: str | None = None,
    limit: int = 5,
    include_archived: bool = False,
    all_repos: bool = False,
    relaxed: bool = False,
    exclude_document_id: str | None = None,
    as_of: str | None = None,
) -> list[Hit]:
    sql = """
      SELECT c.id, c.document_id, c.heading, c.body,
             d.path, d.repo, d.status, d.updated_at,
             f.slug AS feature_slug,
             bm25(chunks_fts, 20.0, 10.0, 1.0) AS raw
      FROM chunks_fts
      JOIN chunks c ON c.rowid = chunks_fts.rowid
      JOIN documents d ON d.id = c.document_id
      LEFT JOIN features f ON f.id = d.feature_id
      WHERE chunks_fts MATCH ? AND d.deleted_at IS NULL
    """
    params: list = [match_expr]
    if exclude_document_id:
        # In SQL, not on the returned hits: filtering afterwards is too late, because the
        # self-match would already have taken the slot that stops search() from trying
        # the OR fallback.
        sql += " AND d.id != ?"
        params.append(exclude_document_id)
    if repo and not all_repos:
        sql += " AND d.repo = ?"
        params.append(repo)
    if source:
        sql += " AND d.source = ?"
        params.append(source)
    if kind:
        sql += " AND d.kind = ?"
        params.append(kind)
    if feature:
        fid = resolve_feature(conn, feature, create=False)
        if fid is None:
            return []
        sql += " AND d.feature_id = ?"
        params.append(fid)
    if as_of is not None:
        # The as-of view replaces the status filter rather than adding to it: status says
        # what is true now, the interval says when it stopped being true. A document
        # archived after `as_of` was believed on `as_of`, so it answers; one archived on
        # or before it does not. NULL invalid_at means the interval is still open,
        # including hand-archived documents with no known end.
        sql += " AND (d.invalid_at IS NULL OR d.invalid_at > ?)"
        params.append(as_of)
    else:
        allowed = list(SEARCHABLE_STATUSES) + (["archived"] if include_archived else [])
        sql += f" AND d.status IN ({','.join('?' * len(allowed))})"
        params.extend(allowed)
    sql += " ORDER BY raw LIMIT ?"
    params.append(limit * 5)

    rows = list(conn.execute(sql, params))
    hits: list[Hit] = []
    for r in rows:
        base = -r["raw"]
        score = base * STATUS_FACTOR.get(r["status"], 1.0)
        hits.append(Hit(
            id=r["id"], document_id=r["document_id"], score=score,
            repo=r["repo"], feature=r["feature_slug"], path=r["path"],
            heading=r["heading"] or "", text=r["body"], relaxed=relaxed,
        ))
    hits.sort(key=lambda h: h.score, reverse=True)
    # One chunk per document: a limit of 5 should surface five documents, not five
    # excerpts of one. knowledge_get fetches the rest.
    best: list[Hit] = []
    seen: set[str] = set()
    for hit in hits:
        if hit.document_id in seen:
            continue
        seen.add(hit.document_id)
        best.append(hit)
    return best[:limit]


def _supersession_order(conn: sqlite3.Connection, hits: list[Hit]) -> list[Hit]:
    """Pull a replacement above the document it superseded, whatever relevance said.

    The status factor is a multiplier, not an order: a short dense predecessor can
    outscore a diluted replacement by more than the 0.3 factor takes away. A superseded
    document may still surface, since the correction rarely restates the context, but
    whoever passed `supersedes` already said which of the two is true, and an agent reads
    the first hit. So this is an invariant in code rather than one more weight to tune.

    Runs on the merged list before the caller's limit, so a replacement that fusion
    recovered is not cut before it can be promoted. It cannot promote a replacement that
    never matched the query.
    """
    pos: dict[str, int] = {}
    for i, hit in enumerate(hits):
        pos.setdefault(hit.document_id, i)
    if len(pos) < 2:
        return hits

    ids = list(pos)
    rows = conn.execute(
        "SELECT id, supersedes FROM documents WHERE supersedes IS NOT NULL"
        " AND supersedes NOT IN ('', '[]')"
        f" AND id IN ({','.join('?' * len(ids))})", ids)
    # Known limit: pairwise. A supersession chain with all three documents in one result
    # set can still misorder the far ends; walk the edges transitively if that shows up.
    anchor: dict[str, int] = {}
    for row in rows:
        try:
            targets = json.loads(row["supersedes"])
        except (TypeError, ValueError):
            # The column may hold a bare id rather than a JSON list.
            targets = [row["supersedes"]]
        if isinstance(targets, str):
            targets = [targets]
        above = [pos[t] for t in targets if t in pos and pos[t] < pos[row["id"]]]
        if above:
            anchor[row["id"]] = min(above)

    if not anchor:
        return hits
    # A replacement adopts its predecessor's rank and breaks the tie upward; every other
    # hit keeps its own. Sorting on the original index last keeps the rest stable.
    def key(pair: tuple[int, Hit]) -> tuple[int, int, int]:
        i, hit = pair
        return (anchor.get(hit.document_id, pos[hit.document_id]),
                0 if hit.document_id in anchor else 1, i)

    return [hit for _, hit in sorted(enumerate(hits), key=key)]


def iso_date(text: str) -> str:
    """`text` unchanged if it is a strict YYYY-MM-DD date. Validity is compared as text,
    so a looser spelling (20240101) would compare wrongly rather than fail."""
    try:
        if len(text) != 10:
            raise ValueError
        date.fromisoformat(text)
    except ValueError:
        raise ValueError(f"as_of must be an ISO date (YYYY-MM-DD), got {text!r}") from None
    return text


def search(
    conn: sqlite3.Connection,
    q: str,
    repo: str | None = None,
    feature: str | None = None,
    source: str | None = None,
    kind: str | None = None,
    limit: int = 5,
    include_archived: bool = False,
    all_repos: bool = False,
    match_mode: str = "auto",
    cfg=None,
    exclude_document_id: str | None = None,
    as_of: str | None = None,
) -> list[Hit]:
    """Search chunks. `auto` tries strict AND, then degrades to OR rather than
    returning nothing — a four-word question rarely has every word in one chunk.

    `as_of` answers from a validity interval instead of current status: pass an ISO date
    (YYYY-MM-DD) and a document is in view exactly when its `invalid_at` is unset or later
    than that date, whatever its status is today. This is how a superseded document stays
    reachable for the dates it was believed on. `include_archived` is meaningless
    alongside `as_of`, which decides on the interval alone.

    `exclude_document_id` drops one document at the SQL level, lexical and dense both,
    for `link_candidates` re-scanning a document that is already indexed: a query built
    from its own text would otherwise self-match and never reach a real neighbour.
    """
    # Validated once here so every entry point fails the same way; a bad value silently
    # matching nothing or falling through to another strategy would be silent degradation.
    if match_mode not in ("auto", "all", "any"):
        raise ValueError(f"match_mode must be auto, all or any, got {match_mode!r}")
    if as_of is not None:
        iso_date(as_of)
    if not q.strip():
        return []

    # Fusion needs depth on both sides: a document fusion rescues is one neither retriever
    # ranked highly, so cutting the lexical list to `limit` before merging throws away
    # exactly the candidates RRF exists to promote.
    fusing = cfg is not None
    fetch = limit * 3 if fusing else limit
    kwargs = dict(repo=repo, feature=feature, source=source, kind=kind,
                  limit=fetch, include_archived=include_archived, all_repos=all_repos,
                  exclude_document_id=exclude_document_id, as_of=as_of)

    # Attempts in order of trustworthiness. The repo scope is a preference and a strict
    # match beats a loose one, so a real answer recorded under another repo outranks a
    # doc here that merely shares a word. An agent cannot tell those apart.
    scoped = repo and not all_repos
    attempts = [(_escape(q, "all"), False, False)]
    if scoped:
        attempts.append((_escape(q, "all"), False, True))
    if match_mode != "all":
        attempts.append((_escape(q, "any"), True, False))
        if scoped:
            attempts.append((_escape(q, "any"), True, True))

    hits: list[Hit] = []
    for expr, relaxed, widen in attempts:
        found = _run(conn, expr, relaxed=relaxed, **dict(kwargs, all_repos=widen or all_repos))
        if found:
            hits = [replace(h, cross_repo=True) for h in found] if widen else found
            break

    # Dense retrieval is a supplement, never a replacement: BM25 is strong on engineering
    # notes because they are full of rare exact identifiers; what it cannot do is
    # paraphrase, and that is the gap fusion closes.
    if fusing:
        hits = _fuse_with_vectors(conn, cfg, q, hits, repo, all_repos, limit,
                                   exclude_document_id=exclude_document_id,
                                   include_archived=include_archived, as_of=as_of)
    hits = _supersession_order(conn, hits)
    hits = hits[:limit]

    counts = neighbour_counts(conn, [h.document_id for h in hits])
    hits = [replace(h, neighbours=counts.get(h.document_id, 0)) for h in hits]

    if hits:
        from akasha.knowledge import touch
        touch(conn, [h.document_id for h in hits])
    return hits


def _strip_headings(text: str) -> str:
    """Drop heading lines. A heading is structure, not vocabulary -- letting it
    into a search query adds noise no author typed as prose."""
    return "\n".join(line for line in text.splitlines() if not HEADING.match(line))


def _first_paragraph(body: str) -> str:
    """The first block of prose in a document body, heading lines stripped."""
    stripped = _strip_headings(body).strip()
    if not stripped:
        return ""
    return re.split(r"\n\s*\n", stripped, maxsplit=1)[0].strip()


def link_candidates(
    conn: sqlite3.Connection,
    cfg,
    title: str,
    body: str,
    limit: int = 5,
    exclude_id: str | None = None,
    strategy: str = "body",
) -> list[Hit]:
    """Fold candidates for a document identified by title + body alone. It need not be
    indexed yet, and normally should not be.

    Two query shapes. `strategy="body"` (the default) searches with the whole body,
    headings stripped; `strategy="lede"` searches with the title plus the first paragraph.

    A whole-body query finds nothing under strict-AND, so search degrades to OR, and OR
    rewards vocabulary breadth: length can beat topic. The lede query counters that. The
    body default favours recall, because near-duplicates share vocabulary heavily and an
    agent reads a ranked list rather than only rank 1. A missed fold is a depth problem,
    not a query-shape one: raise `limit`.

    Call this before the caller's document is written and indexed, when possible; no
    exclusion is needed then. `exclude_id` covers re-scanning a document that is already
    indexed: its own text would otherwise take the one slot search()'s first successful
    attempt returns and starve the OR fallback that finds a real neighbour. The exclusion
    happens in `search()` at the SQL level, lexical and dense both. `limit + 1` is kept
    as a cheap second line of defence.

    No blanket `except Exception: return []`: a failure here surfaces like any other.
    """
    if strategy == "body":
        query = _strip_headings(body)
    elif strategy == "lede":
        query = f"{title}\n\n{_first_paragraph(body)}"
    else:
        raise ValueError(f"unknown strategy {strategy!r}; expected 'body' or 'lede'")
    hits = search(conn, query, all_repos=True, limit=limit + 1, cfg=cfg,
                  exclude_document_id=exclude_id)
    if exclude_id is not None:
        hits = [h for h in hits if h.document_id != exclude_id]
    return hits[:limit]


def _fuse_with_vectors(conn, cfg, q, hits, repo, all_repos, limit,
                        exclude_document_id: str | None = None,
                        include_archived: bool = False,
                        as_of: str | None = None) -> list[Hit]:
    """Reciprocal-rank fusion of the lexical hits with the nearest vectors.

    Ranks, not scores: BM25 is unbounded and query-dependent while cosine is [0,1], so
    adding them would let BM25 decide everything. RRF discards magnitude and rewards
    agreement between the two retrievers instead.
    """
    from akasha import vectors

    if not vectors.enabled(cfg):
        return hits
    dense = vectors.nearest(conn, q, limit=limit * 3,
                            repo=None if all_repos else repo)
    if not dense:
        return hits                      # unavailable, unindexed, or nothing near

    # Fuse DOCUMENTS, not chunks. The lexical side is already one chunk per document
    # while the dense side returns raw chunks, so fusing chunk ids compares two different
    # units and can demote a document lexical ranked first.
    best_chunk: dict[str, str] = {}
    lexical_docs: list[str] = []
    for hit in hits:
        if hit.document_id not in best_chunk:
            best_chunk[hit.document_id] = hit.id
            lexical_docs.append(hit.document_id)

    dense_docs: list[str] = []
    for chunk_id in dense:
        row = conn.execute("SELECT document_id FROM chunks WHERE id=?", (chunk_id,)).fetchone()
        if row is None:
            continue
        doc = row["document_id"]
        if doc == exclude_document_id:
            # nearest() has no exclude-by-document filter, and a query built from a
            # document's own text is also its own nearest embedding.
            continue
        if doc not in best_chunk:
            best_chunk[doc] = chunk_id
        if doc not in dense_docs:
            dense_docs.append(doc)

    by_id = {h.id: h for h in hits}
    fused: list[Hit] = []
    for doc in rrf_merge([lexical_docs, dense_docs]):
        chunk_id = best_chunk.get(doc)
        hit = by_id.get(chunk_id) or _hit_for_chunk(
            conn, chunk_id, include_archived=include_archived, as_of=as_of)
        if hit is None:
            continue
        fused.append(hit)
        if len(fused) >= limit:
            break
    return fused


def _hit_for_chunk(conn, chunk_id: str, include_archived: bool = False,
                   as_of: str | None = None) -> Hit | None:
    """Materialise a Hit for a chunk only vector search found.

    Takes `include_archived` because the lexical side already applied it in SQL: without
    it the flag would mean one thing for a document both retrievers found and another for
    one only vectors found. An archived document is most often the only answer on the
    dense side, reachable by paraphrase rather than by the query's words.
    """
    row = conn.execute(
        "SELECT c.id, c.document_id, c.heading, c.body, d.path, d.repo, d.status,"
        " d.invalid_at, f.slug AS feature_slug FROM chunks c"
        " JOIN documents d ON d.id = c.document_id"
        " LEFT JOIN features f ON f.id = d.feature_id"
        " WHERE c.id = ? AND d.deleted_at IS NULL", (chunk_id,)).fetchone()
    if row is None:
        return None
    if as_of is not None:
        # The dense side's half of the as-of filter, so one query cannot answer two ways
        # depending on which retriever hit it.
        if row["invalid_at"] is not None and row["invalid_at"] <= as_of:
            return None
    else:
        allowed = SEARCHABLE_STATUSES + (("archived",) if include_archived else ())
        if row["status"] not in allowed:
            return None
    return Hit(id=row["id"], document_id=row["document_id"], score=0.0,
               repo=row["repo"], feature=row["feature_slug"], path=row["path"],
               heading=row["heading"] or "", text=row["body"])


def rrf_merge(lists: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal-rank fusion. No weights to tune; presence in both lists wins."""
    scores: dict[str, float] = {}
    for ranked in lists:
        for rank, item in enumerate(ranked):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda i: scores[i], reverse=True)
