"""Deterministic integrity checks. Reports only; never modifies, never deletes.

akasha calls no model, so it reports what is provably wrong and leaves judgement to
whoever reads the output. The one check that needs more than string equality,
near_duplicate, reuses the embeddings search already keeps in vec_chunks, so it stays a
pure query with no inference in it.
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from akasha.config import Config
from akasha.links import FENCE_BLOCK
from akasha.markdown import parse_frontmatter

NEAR_DUPLICATE_COSINE = 0.92
DEFAULT_LIMIT = 50
ERRORS_KEY = "fsck.errors"

INLINE_CODE = re.compile(r"`([^`\n]+)`")
LINE_NUMBER = re.compile(r"(:\d+|#L\d+|::[\w.]+)$")
COMMAND_CHARS = re.compile(r"[\s*?<>|&;()\[\]{}\"'$]")

def _anchor_path(span: str) -> str | None:
    """The path an inline code span anchors to, or None when it is not an anchor.

    `src/foo.py`, `./scripts/run.sh` and `src/foo.py:41` anchor; `foo.py` is bare,
    `/abs/src/foo.py` and `~/src` sit outside the repo-root model, `../x` escapes it,
    a URL or a fragment (`#`, `?`) resolves to nothing file-shaped, and a command,
    glob or directory names nothing that could exist as one file.
    """
    path = LINE_NUMBER.sub("", span)
    if not path or "://" in path or "#" in path or "?" in path:
        return None
    if path.startswith(("/", "~", "..")):
        return None
    if "/" not in path:
        return None            # bare filename: ambiguous across repos
    if COMMAND_CHARS.search(path) or path.endswith("/"):
        return None            # a command, a glob or a directory, not a path
    if "." not in path.rsplit("/", 1)[-1]:
        return None            # no file extension: command-shaped, not file-shaped
    return path

def _anchor_candidates(body: str) -> list[str]:
    """Path-shaped code anchors in a body, deduplicated, first-occurrence order.

    Inline code spans only: a fenced block is code being quoted, not a reference, so
    the line links.py already draws for [[refs]] holds here too.
    """
    seen: list[str] = []
    for span in INLINE_CODE.findall(FENCE_BLOCK.sub("", body)):
        path = _anchor_path(span)
        if path is not None and path not in seen:
            seen.append(path)
    return seen

@dataclass
class Finding:
    kind: str
    severity: str          # error | warning | info
    document_id: str | None
    detail: str

def duplicate_groups(conn: sqlite3.Connection) -> list[dict]:
    """Documents that share a title within one scope, grouped.

    Scoped to repo AND feature. A title alone is not a duplicate signal: a note called
    "record" exists in several repos, and every feature has an overview document. Those are
    distinct documents that share a name.
    """
    return [
        {"title": row["title"], "repo": row["repo"], "feature_id": row["feature_id"],
         "count": row["n"], "ids": row["ids"].split(",")}
        for row in conn.execute(
            "SELECT title, repo, feature_id, COUNT(*) n, GROUP_CONCAT(id) ids FROM documents"
            " WHERE deleted_at IS NULL AND status != 'archived'"
            " GROUP BY LOWER(title), IFNULL(repo,''), IFNULL(feature_id,'')"
            " HAVING n > 1")
    ]

def _document_vectors(conn: sqlite3.Connection):
    """Mean chunk embedding per active document, grouped by repo. numpy only here.

    A document is one point, not one point per chunk: chunk-level pairs fire on a shared
    boilerplate heading while the documents around them differ, which is noise dressed as
    a finding. The same reason search.py fuses documents rather than chunks.
    """
    import numpy as np

    rows = conn.execute(
        "SELECT v.embedding, v.repo, c.document_id, d.title FROM vec_chunks v"
        " JOIN chunks c ON c.id = v.chunk_id"
        " JOIN documents d ON d.id = c.document_id"
        " WHERE d.deleted_at IS NULL AND d.status != 'archived'"
        " AND d.source = 'native'").fetchall()

    acc: dict[str, dict] = {}
    for row in rows:
        entry = acc.setdefault(row["document_id"], {
            "repo": row["repo"] or "", "title": row["title"] or "", "vecs": []})
        entry["vecs"].append(np.frombuffer(row["embedding"], dtype=np.float32))

    by_repo: dict[str, list[tuple[str, str, "np.ndarray"]]] = {}
    for doc_id, entry in acc.items():
        mean = np.mean(entry["vecs"], axis=0)
        norm = float(np.linalg.norm(mean))
        if norm == 0.0:
            continue
        by_repo.setdefault(entry["repo"], []).append((doc_id, entry["title"], mean / norm))
    return by_repo

def _near_duplicates(conn: sqlite3.Connection, cfg: Config) -> list[Finding]:
    """Documents that record the same thing under different titles.

    duplicate_groups() matches on LOWER(title) and cannot see these. The embeddings are
    already in the database for search, so this needs no model of its own.

    Scoped to one repo, for the reason duplicate_groups() gives about titles: two repos'
    writeups of one change can read as near-identical and are two documents.

    Native documents only, on the ownership line missing_file already draws. Indexed
    roots are other tools' files: they refuse archive and update, so "fold them together
    with knowledge_append" is advice their owner cannot take, and including them would
    bury the native findings under a report readers learn to skip.
    """
    from akasha import vectors

    if not vectors.enabled(cfg):
        return []                # a lexical-only install is a choice, not a degradation
    if not vectors.available(conn):
        return [Finding("near_duplicate_unchecked", "info", None,
                        "vectors are configured but unavailable here, so near-duplicate "
                        "detection did not run; `akasha doctor` has the reason")]
    try:
        by_repo = _document_vectors(conn)
    except sqlite3.OperationalError:
        return [Finding("near_duplicate_unchecked", "info", None,
                        "the vector index is not built, so near-duplicate detection did "
                        "not run; reindex to include it")]

    import numpy as np

    findings: list[Finding] = []
    for repo, docs in by_repo.items():
        if len(docs) < 2:
            continue
        matrix = np.stack([v for _, _, v in docs])
        scores = matrix @ matrix.T
        np.fill_diagonal(scores, -1.0)
        for i, (doc_id, _, _) in enumerate(docs):
            partners = np.flatnonzero(scores[i] >= NEAR_DUPLICATE_COSINE)
            if not partners.size:
                continue
            best = int(partners[np.argmax(scores[i][partners])])
            findings.append(Finding(
                "near_duplicate", "warning", doc_id,
                f"reads as {scores[i][best]:.2f} cosine to '{docs[best][1]}' "
                f"({docs[best][0]}) in the same repo ({repo or '-'}); "
                f"{partners.size} near match(es) total — fold them together "
                f"with knowledge_append; never just drop one"))
    return findings

def check(conn: sqlite3.Connection, cfg: Config) -> list[Finding]:
    findings: list[Finding] = []

    for row in duplicate_groups(conn):
        scope = f"{row['repo'] or '-'}"
        # One finding per document, not per group: naming each offender is what a caller
        # can act on.
        for doc_id in row["ids"]:
            findings.append(Finding("duplicate_title", "warning", doc_id,
                                    f"'{row['title']}' duplicates {row['count'] - 1} other "
                                    f"active documents in the same feature ({scope})"))

    findings.extend(_near_duplicates(conn, cfg))

    for row in conn.execute(
        "SELECT l.from_document_id, l.to_ref FROM links l"
        " JOIN documents d ON d.id = l.from_document_id"
        " WHERE l.to_document_id IS NULL AND l.to_ref IS NOT NULL"
        " AND d.deleted_at IS NULL AND d.status != 'archived'"
    ):
        findings.append(Finding("dangling_link", "info", row["from_document_id"],
                                f"link to '{row['to_ref']}' resolves to nothing"))

    for row in conn.execute(
        "SELECT id, supersedes FROM documents"
        " WHERE deleted_at IS NULL AND supersedes IS NOT NULL AND supersedes != '[]'"
    ):
        try:
            targets = json.loads(row["supersedes"])
        except (json.JSONDecodeError, TypeError):
            targets = [row["supersedes"]]
        for target in targets:
            other = conn.execute(
                "SELECT id, status FROM documents WHERE id=?", (target,)).fetchone()
            if other is None:
                findings.append(Finding("broken_supersede", "warning", row["id"],
                                        f"supersedes '{target}', which does not exist"))
            elif other["status"] == "active":
                findings.append(Finding("superseded_but_active", "warning", target,
                                        f"superseded by {row['id']} but still active"))

    for row in conn.execute(
        "SELECT d.id, d.path FROM documents d"
        " LEFT JOIN chunks c ON c.document_id = d.id"
        " WHERE d.deleted_at IS NULL GROUP BY d.id HAVING COUNT(c.id) = 0"
    ):
        findings.append(Finding("no_chunks", "warning", row["id"],
                                f"indexed but has no content: {row['path']}"))

    # Every source, not just native: index_all does not reconcile between runs, so this is
    # the only place a vanished indexed file is reported. Severity splits on ownership:
    # akasha owns the native directory, so a file missing there is corruption, while an
    # indexed root belongs to someone else, where a deleted file is ordinary work.
    for row in conn.execute(
        "SELECT id, path, source FROM documents WHERE deleted_at IS NULL"
    ):
        if not Path(row["path"]).exists():
            severity = "error" if row["source"] == "native" else "warning"
            findings.append(Finding("missing_file", severity, row["id"],
                                    f"indexed but absent from disk: {row['path']}"))

    # Code anchors that no longer resolve. `info`: a moved file is ordinary work, not
    # corruption. The body is read from disk rather than rebuilt from the chunks, which
    # drop heading lines. The repo root comes from the project's configured path, and a
    # repo with no configured path is skipped, never guessed at.
    for row in conn.execute(
        "SELECT id, path, repo FROM documents"
        " WHERE deleted_at IS NULL AND source = 'native' AND status != 'archived'"
    ):
        file = Path(row["path"])
        if not file.is_file():
            continue             # missing_file already reports this document
        root = (cfg.projects.get(row["repo"]) or {}).get("path")
        if not root:
            continue
        root = Path(root).expanduser()
        body = parse_frontmatter(file.read_text(errors="replace"))[1]
        for anchor in _anchor_candidates(body):
            if not (root / anchor).exists():
                findings.append(Finding(
                    "stale_anchor", "info", row["id"],
                    f"anchor '{anchor}' resolves to nothing under the repo root"))

    return findings

def report(conn: sqlite3.Connection, cfg: Config, limit: int | None = DEFAULT_LIMIT) -> dict:
    """The caller's shape: full counts, and at most `limit` findings in detail.

    The check always runs in full, so `total`, `counts` and the recorded error count are
    never shortened by a display limit; only the findings list is. `None` withholds nothing.
    """
    findings = check(conn, cfg)
    record_errors(conn, sum(f.severity == "error" for f in findings))
    counts: dict[str, int] = {}
    for f in findings:
        counts[f.kind] = counts.get(f.kind, 0) + 1
    shown = findings if limit is None else findings[:limit]
    return {"total": len(findings), "counts": counts,
            "findings": [{"kind": f.kind, "severity": f.severity,
                          "document_id": f.document_id, "detail": f.detail} for f in shown],
            "withheld": len(findings) - len(shown)}

def record_errors(conn: sqlite3.Connection, errors: int) -> None:
    """Store the latest full-run error count, so a session start can mention integrity
    without running every check."""
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?)"
        " ON CONFLICT(key) DO UPDATE SET value=excluded.value", (ERRORS_KEY, str(errors)))

def cached_error_count(conn: sqlite3.Connection) -> int | None:
    """The error count of the last full run, or None if fsck has never run or the stored
    value is unreadable: unknown is not the same as clean."""
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (ERRORS_KEY,)).fetchone()
    try:
        return int(row["value"]) if row else None
    except ValueError:
        return None
