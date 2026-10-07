"""Indexer. Walks configured roots and mirrors markdown into the derived tables."""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from akasha.config import Config
from akasha.db import new_id, now
from akasha.events import emit
from akasha.features import resolve_feature
from akasha.knowledge import canonical_kind
from akasha.links import resolve_dangling, sync_links
from akasha.markdown import chunk as chunk_body
from akasha.markdown import first_heading, parse_frontmatter, strip_data_uris
from akasha.security import (INJECTION_PATTERNS, INJECTION_STRIP_RULES, SECRET_PATTERNS,
                             is_denied, redact, scrub_injection)

SCAN_CONFIG_KEY = "index.scan_config"

# Bump when the chunker, the kind hints or strip_data_uris change: stored chunks then no
# longer match what indexing the same file now produces, and unchanged files must be
# processed again.
INDEX_FORMAT = 1

# Path hints map to canonical kinds, not to a second vocabulary: "findings" indexes as
# "finding", because a hint that re-creates a kind the write path bans reopens the split.
# The filename and the parent directory both draw from this one table.
KIND_HINTS = {
    "plan": "plan",
    "findings": "finding", "spec": "spec", "design": "spec",
    "decision": "decision", "notes": "reference",
}


class IdCollision(ValueError):
    """A derived document id already belongs to a different file."""


@dataclass
class IndexStats:
    indexed: int = 0
    skipped: int = 0
    denied: int = 0
    pruned: int = 0
    kind_fallback: int = 0
    redacted: list[str] = field(default_factory=list)
    redacted_files: list[tuple[str, list[str]]] = field(default_factory=list)
    links_resolved: int = 0
    vectors: int = 0
    collisions: list[tuple[str, str]] = field(default_factory=list)


def _guess_kind(path: Path) -> str:
    """Kind from the filename, then the parent directory: a plan filed at
    .../plans/<name>.md carries its kind in the folder. The filename is more specific, so
    it wins when both match."""
    lowered = path.name.lower()
    for hint, kind in KIND_HINTS.items():
        if hint in lowered:
            return kind
    lowered_dir = path.parent.name.lower()
    for hint, kind in KIND_HINTS.items():
        if hint in lowered_dir:
            return kind
    return "reference"


def _matches(path: Path, root: Path, patterns: list[str]) -> bool:
    rel = path.relative_to(root).as_posix()
    # fnmatch needs a literal "/" for "**/", so a leading "**/" must also match with
    # the prefix removed, or files directly in the root never match.
    return any(fnmatch.fnmatch(rel, p) or (p.startswith("**/") and fnmatch.fnmatch(rel, p[3:]))
               for p in patterns)


def index_path(
    conn: sqlite3.Connection, cfg: Config, path: Path, source: str, root: Path | None = None,
    root_repo: str | None = None, force: bool = False,
) -> tuple[str, list[str], int] | None:
    """Index one markdown file. Returns (document id, redaction hits, kind fallback
    count), or None when refused or unchanged.

    The unchanged check comes first and needs only a stat, so a no-op run never reads,
    redacts or scans a file. `force` skips it, for when the scan rules changed and an
    unchanged file must still be processed again.
    """
    if is_denied(path, cfg) or path.suffix.lower() != ".md":
        return None

    stat = path.stat()
    mtime, size = stat.st_mtime, stat.st_size
    existing = conn.execute(
        "SELECT id, mtime, size, deleted_at FROM documents WHERE path = ?", (str(path),)
    ).fetchone()
    # Known limit: same mtime and same size counts as unchanged, so an edit that keeps both
    # is missed; `index --force` is the escape hatch.
    # A tombstoned row is never unchanged, whatever the mtime says: restoring a file from
    # a backup that preserves mtime would otherwise leave the document invisible for good.
    if (existing and not force and existing["deleted_at"] is None
            and existing["mtime"] == mtime and existing["size"] == size):
        return None

    text = path.read_text(errors="replace")
    meta, body = parse_frontmatter(text)
    # A base64 payload is worthless to both retrievers, so it goes before the scanners,
    # which would otherwise spend their time on megabytes of image data.
    body = strip_data_uris(body)
    if cfg.scan_secrets:
        body, hits = redact(body)
        body, inj = scrub_injection(body)
    else:
        hits = []
        inj = []
    if inj:
        # Same channel as redaction: the stats field is "rule names that fired", and the
        # event below carries the document.
        hits = hits + [h for h in inj if h not in hits]

    repo = meta.get("repo")
    feature_slug = meta.get("feature")
    if root is not None:
        rel = path.relative_to(root).parts
        # Frontmatter first, then the root's declared repo, then the first path segment.
        # A declared repo beats the segment, which under a root inside a repo is a
        # subfolder name rather than a repo.
        repo = repo or root_repo or (rel[0] if len(rel) > 1 else None)
        if not feature_slug and len(rel) > 2:
            candidate = rel[1]
            if not candidate.startswith("."):
                siblings = {p.name for p in root.iterdir() if p.is_dir()}
                if candidate not in siblings:
                    feature_slug = candidate

    feature_id = resolve_feature(conn, feature_slug, repo=repo) if feature_slug else None
    supersedes = json.dumps(meta.get("supersedes") or [])
    if existing:
        doc_id = existing["id"]
    elif meta.get("id"):
        doc_id = meta["id"]
    else:
        # Derived from where the file lives, so a rebuilt database reproduces the ids that
        # links name. The root is part of the key because several roots share a source and
        # can hold files of the same name. Moving a file or its root changes the id.
        if root is not None:
            key = f"{source}:{os.path.abspath(root)}:{path.relative_to(root).as_posix()}"
        else:
            key = f"{source}:{path}"
        doc_id = "k_" + hashlib.sha256(key.encode()).hexdigest()[:12]
    if not existing:
        clash = conn.execute("SELECT path FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if clash:
            raise IdCollision(f"document id {doc_id} for {path} collides with {clash['path']}")
    title = meta.get("title") or first_heading(body) or path.stem.replace("-", " ")
    # Same vocabulary as write(), but indexing is bulk: one bad file must not stop a run,
    # so an unknown kind becomes reference and is counted.
    try:
        kind, kind_fallback = canonical_kind(meta.get("kind") or _guess_kind(path)), 0
    except ValueError:
        kind, kind_fallback = "reference", 1
    status = meta.get("status") or "active"
    # Set unconditionally: a date removed from the frontmatter must leave the column, or
    # a rebuild keeps an invalidation the file no longer claims.
    invalid_at = meta.get("invalid_at")

    if existing:
        # deleted_at is cleared unconditionally: reaching here means the file is on disk,
        # which is the only fact the tombstone ever asserted.
        conn.execute(
            "UPDATE documents SET title=?, repo=?, feature_id=?, kind=?, status=?, supersedes=?,"
            " invalid_at=?, mtime=?, size=?, indexed_at=?, updated_at=?, deleted_at=NULL"
            " WHERE id=?",
            (title, repo, feature_id, kind, status, supersedes, invalid_at, mtime, size,
             now(), now(), doc_id),
        )
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
    else:
        conn.execute(
            "INSERT INTO documents (id, source, path, title, repo, feature_id, kind, status,"
            " supersedes, invalid_at, mtime, size, indexed_at, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (doc_id, source, str(path), title, repo, feature_id, kind, status,
             supersedes, invalid_at, mtime, size, now(), now(), now()),
        )

    for c in chunk_body(body):
        # Fresh ids on every index: vector diffing relies on a changed document getting
        # new chunk ids, so its old vectors are dropped and the new text re-embedded.
        conn.execute(
            "INSERT INTO chunks (id, document_id, title, heading, body, ord) VALUES (?,?,?,?,?,?)",
            (new_id("c"), doc_id, title, c.heading, c.body, c.ord),
        )
    sync_links(conn, doc_id, body)
    if inj:
        emit(conn, "security.injection", document=doc_id, path=str(path), rules=inj)
    return doc_id, hits, kind_fallback


def _prune(conn: sqlite3.Connection, root: Path, seen: set[str]) -> int:
    """Tombstone documents under `root` whose file is no longer there.

    Soft-delete rather than forget: search filters on deleted_at, the id other documents
    link to survives, and a file that comes back is revived rather than re-created under
    a new id.

    Containment is tested in Python instead of a LIKE on the path, because a root
    containing '%' or '_' would make that pattern match paths under a different root.
    """
    pruned = 0
    for row in conn.execute(
        "SELECT id, path FROM documents WHERE deleted_at IS NULL"
    ).fetchall():
        path = Path(row["path"])
        if path.is_relative_to(root) and row["path"] not in seen:
            conn.execute("UPDATE documents SET deleted_at=?, updated_at=? WHERE id=?",
                         (now(), now(), row["id"]))
            pruned += 1
    return pruned


def _scan_config_marker(cfg: Config) -> str:
    """A digest of everything that decides what a file's stored text looks like: whether
    scanning is on, the deny lists, the redaction and injection rules, the index format and
    each root's path and repo. A change to any of them means stored chunks may no longer match what indexing the
    same file now would produce."""
    state = {
        "scan_secrets": cfg.scan_secrets,
        "deny_files": sorted(cfg.deny_files),
        "deny_extensions": sorted(cfg.deny_extensions),
        "secrets": [(n, p.pattern) for n, p in SECRET_PATTERNS],
        "injection": [(n, p.pattern) for n, p in INJECTION_PATTERNS],
        "strip": sorted(INJECTION_STRIP_RULES),
        "format": INDEX_FORMAT,
        # Stored on every document, so a changed repo must re-derive them.
        "roots": [(str(r.path), r.repo) for r in cfg.index_roots],
    }
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


def index_all(conn: sqlite3.Connection, cfg: Config, force: bool = False) -> IndexStats:
    """Walk every configured root plus the owned knowledge dir. `force` reprocesses
    files whose stat is unchanged."""
    stats = IndexStats()
    marker = _scan_config_marker(cfg)
    stored = conn.execute("SELECT value FROM meta WHERE key=?", (SCAN_CONFIG_KEY,)).fetchone()
    force = force or stored is None or stored["value"] != marker
    roots: list[tuple[Path, str, list[str], list[str], str | None]] = [
        (Path(r.path).expanduser(), r.source, r.include, r.exclude, r.repo)
        for r in cfg.index_roots
    ]
    roots.append((cfg.knowledge_dir, "native", ["**/*.md"], [], None))

    for root, source, include, exclude, root_repo in roots:
        if not root.exists():
            continue
        seen: set[str] = set()
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if is_denied(path, cfg):
                stats.denied += 1
                continue
            if path.suffix.lower() != ".md":
                continue
            if exclude and _matches(path, root, exclude):
                continue
            if include and not _matches(path, root, include):
                continue
            # Recorded before indexing: index_path returns None for a refused file and for
            # an unchanged one alike, so its result cannot mean "seen".
            seen.add(str(path))
            try:
                result = index_path(conn, cfg, path, source, root=root,
                                    root_repo=root_repo, force=force)
            except OSError:
                # Removed or unreadable since the listing; the next run sees the truth.
                result = None
            except IdCollision as exc:
                # One clash must not stop the corpus; the caller prints it.
                stats.collisions.append((str(path), str(exc)))
                result = None
            if result is None:
                stats.skipped += 1
            else:
                _, hits, kind_fallback = result
                stats.indexed += 1
                stats.kind_fallback += kind_fallback
                stats.redacted.extend(h for h in hits if h not in stats.redacted)
                if hits:
                    stats.redacted_files.append((str(path), hits))
        # Inside the loop: a root that does not exist was skipped above, so an unmounted
        # disk or a renamed directory is never read as "every document was deleted".
        stats.pruned += _prune(conn, root, seen)
    # Recorded only once the walk finished, so an interrupted forced pass is forced again.
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?)"
        " ON CONFLICT(key) DO UPDATE SET value=excluded.value", (SCAN_CONFIG_KEY, marker))
    conn.commit()
    # Targets indexed after their referrer were recorded dangling; fix them now.
    stats.links_resolved = resolve_dangling(conn)

    # Zero when the provider is "none" or the extension will not load.
    from akasha import vectors as _vectors

    stats.vectors = _vectors.index_chunks(conn, cfg)
    return stats


def forget(conn: sqlite3.Connection, path: Path) -> int:
    """Drop index rows for a path. The file on disk is never touched."""
    row = conn.execute("SELECT id FROM documents WHERE path = ?", (str(path),)).fetchone()
    if not row:
        return 0
    # Vectors first: they are keyed by chunk id, so the chunks must still be here to name
    # them. An orphan would spend a slot in nearest()'s k and return nothing.
    from akasha import vectors as _vectors

    if _vectors.available(conn):
        try:
            conn.execute(
                f"DELETE FROM {_vectors.TABLE} WHERE chunk_id IN"
                " (SELECT id FROM chunks WHERE document_id = ?)", (row["id"],))
        except sqlite3.OperationalError:
            pass                        # nothing embedded yet
    conn.execute("DELETE FROM chunks WHERE document_id = ?", (row["id"],))
    # Both directions: fsck reads `links` without joining `documents`, so a row left
    # behind would be reported dangling with no document left to correct it from.
    conn.execute("DELETE FROM links WHERE from_document_id = ? OR to_document_id = ?",
                 (row["id"], row["id"]))
    conn.execute("DELETE FROM documents WHERE id = ?", (row["id"],))
    conn.commit()
    return 1
