"""Link graph. Explicit [[refs]] only — guessed relationships are noise."""
from __future__ import annotations

import re
import sqlite3

from akasha.db import new_id, now

WIKI_LINK = re.compile(r"\[\[([^\]\n]+)\]\]")
FENCE_BLOCK = re.compile(r"```.*?```", re.DOTALL)


# A ref and its target may disagree on word separator: a note named `my_note.md` is
# linked as [[my-note]]. Normalising both sides resolves that without a looser match that
# would invent targets and hide what fsck exists to report. SQL rather than Python
# because both call sites are single statements over the whole table.
def _norm(expr: str) -> str:
    return f"LOWER(REPLACE({expr}, '_', '-'))"


def _match_clause(ref: str) -> str:
    """The one rule for resolving a ref to a document, shared by both call sites so a
    link does not appear or vanish depending on index order."""
    return (
        "SELECT d.id FROM documents d WHERE d.deleted_at IS NULL AND ("
        f" d.id = {ref}"
        f" OR {_norm('d.title')} = {_norm(ref)}"
        f" OR {_norm('d.path')} LIKE '%/' || {_norm(ref)} || '.md'"
        ") LIMIT 1"
    )


def parse_refs(body: str) -> list[str]:
    """Extract [[refs]], ignoring anything inside fenced code."""
    stripped = FENCE_BLOCK.sub("", body)
    seen: list[str] = []
    for match in WIKI_LINK.finditer(stripped):
        ref = match.group(1).strip()
        if ref not in seen:
            seen.append(ref)
    return seen


def resolve_dangling(conn: sqlite3.Connection) -> int:
    """Re-resolve links whose target was not indexed yet when they were recorded.

    Links are written during the walk, so a reference to a document indexed later would
    stay dangling without this pass.
    """
    match = _match_clause("links.to_ref")
    cursor = conn.execute(
        f"UPDATE links SET to_document_id = ({match})"
        f" WHERE to_document_id IS NULL AND to_ref IS NOT NULL AND ({match}) IS NOT NULL"
    )
    conn.commit()
    return cursor.rowcount


def sync_links(conn: sqlite3.Connection, doc_id: str, body: str) -> int:
    """Replace this document's outbound links. Unresolvable refs are kept as dangling."""
    conn.execute("DELETE FROM links WHERE from_document_id = ? AND kind = 'cites'", (doc_id,))
    count = 0
    for ref in parse_refs(body):
        row = conn.execute(_match_clause("?"), (ref, ref, ref)).fetchone()
        conn.execute(
            "INSERT INTO links (id, from_document_id, to_document_id, to_ref, kind, created_at)"
            " VALUES (?,?,?,?,'cites',?)",
            (new_id("l"), doc_id, row["id"] if row else None, ref, now()),
        )
        count += 1
    return count

def _edges(marks: str) -> str:
    """Resolved neighbours of the documents in `marks`, as (doc, other, direction) rows.

    The single definition behind both `related` and `neighbour_counts`, so the number a
    search hit advertises is the number a follow-up related call returns. UNION, not
    UNION ALL: two refs that resolve to one document are one neighbour. A link counts only
    when the other end resolved and is not deleted.
    """
    return f"""
        SELECT l.from_document_id AS doc, d.id AS other, 'outbound' AS direction
        FROM links l JOIN documents d ON d.id = l.to_document_id
        WHERE l.from_document_id IN ({marks}) AND d.deleted_at IS NULL AND d.id != l.from_document_id
        UNION
        SELECT l.to_document_id AS doc, d.id AS other, 'backlink' AS direction
        FROM links l JOIN documents d ON d.id = l.from_document_id
        WHERE l.to_document_id IN ({marks}) AND d.deleted_at IS NULL AND d.id != l.to_document_id
    """

def related(conn: sqlite3.Connection, doc_id: str) -> list[dict]:
    """Neighbours in both directions: outbound citations and backlinks.

    No `path`: knowledge_get returns it, and every entry is paid for in the caller's
    context.
    """
    rows = conn.execute(
        f"SELECT d.id, d.title, e.direction FROM ({_edges('?')}) e"
        " JOIN documents d ON d.id = e.other",
        (doc_id, doc_id),
    )
    return [dict(r) for r in rows]

def neighbour_counts(conn: sqlite3.Connection, doc_ids: list[str]) -> dict[str, int]:
    """Resolved neighbours per document, for many documents in one query.

    Retrieval is on the hot path, so one query over the final documents beats one per hit.
    """
    if not doc_ids:
        return {}
    marks = ",".join("?" * len(doc_ids))
    rows = conn.execute(
        f"SELECT doc, COUNT(*) AS n FROM ({_edges(marks)}) GROUP BY doc", doc_ids + doc_ids)
    return {r["doc"]: r["n"] for r in rows}

def graph(conn: sqlite3.Connection, doc_id: str, depth: int = 2) -> list[dict]:
    """Breadth-first traversal to `depth` hops, in both directions."""
    seen = {doc_id}
    frontier = [doc_id]
    out: list[dict] = []
    for _ in range(depth):
        next_frontier: list[str] = []
        for node in frontier:
            for neighbour in related(conn, node):
                if neighbour["id"] in seen:
                    continue
                seen.add(neighbour["id"])
                out.append(neighbour)
                next_frontier.append(neighbour["id"])
        frontier = next_frontier
        if not frontier:
            break
    return out
