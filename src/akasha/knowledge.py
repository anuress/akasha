"""Native document lifecycle. Files are written first; the index follows."""
from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from akasha.config import Config
from akasha.db import new_id, now
from akasha.markdown import parse_frontmatter, render

# The events-retention pass imports this, so the window `stats` reports against and the
# window events are actually kept for cannot disagree.
EVENT_RETENTION_DAYS = 90

_REPO_NAME = re.compile(r"[A-Za-z0-9._-]+")


class StaleWrite(RuntimeError):
    """The document changed since the caller last read it."""

class ReadOnlySource(RuntimeError):
    """akasha only writes documents it owns."""

class DriftRefusal(RuntimeError):
    """The file on disk changed since akasha indexed it: another writer got there first."""

KINDS = ("reference", "plan", "spec", "finding", "review", "progress",
         "decision", "convention", "gotcha", "log")
ALIASES = {"findings": "finding"}

def canonical_kind(kind: str) -> str:
    """Resolve an alias and reject anything outside the vocabulary.

    The error names what is allowed, because the next guess is no better informed than
    the last. The index path is bulk and unattended, so it uses this too but decides its
    own handling for an unknown kind.
    """
    kind = ALIASES.get(kind, kind)
    if kind not in KINDS:
        raise ValueError(f"unknown kind: {kind!r}. Allowed kinds: {', '.join(KINDS)}")
    return kind

def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "untitled"

def _doc_row(conn: sqlite3.Connection, doc_id: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    if row is None:
        raise KeyError(f"unknown document: {doc_id}")
    return row

def get(conn: sqlite3.Connection, doc_id: str) -> sqlite3.Row:
    """Return the full document row."""
    return _doc_row(conn, doc_id)

_READONLY_HINT = ("It belongs to a read-only source: record the change in a native "
                  "document with knowledge_write instead of editing it.")

def readonly_hint(source: str) -> str | None:
    """None for a native document; otherwise the instruction update and archive raise.

    Both go through _require_native, which ignores a root's `writable` opt-in; only
    append checks that (_require_appendable). A document on a writable, non-native root
    therefore still gets a hint here, because update and archive would refuse it.
    """
    return None if source == "native" else _READONLY_HINT

def _require_native(row: sqlite3.Row) -> None:
    if row["source"] != "native":
        raise ReadOnlySource(f"{row['path']} belongs to source '{row['source']}'. "
                             f"{_READONLY_HINT}")

def _owning_root(cfg: Config, path: Path):
    """The configured root that contains `path`, or None."""
    for root in cfg.index_roots:
        # Resolve both sides: a symlink inside a writable root can point outside it.
        if path.resolve().is_relative_to(Path(root.path).expanduser().resolve()):
            return root
    return None

def _require_appendable(row: sqlite3.Row, cfg: Config) -> None:
    """Native always; an indexed root only when it has opted in with `writable`."""
    if row["source"] == "native":
        return
    root = _owning_root(cfg, Path(row["path"]))
    if root is not None and root.writable:
        return
    raise ReadOnlySource(
        f"{row['path']} belongs to source '{row['source']}', which is not writable. "
        "Set `writable = true` on that [[index]] root to append to it, or record the "
        "addition in a native document with knowledge_write."
    )

# `documents.mtime` is the exact double `st_mtime` returns, but a coarse-granularity
# filesystem can record a value a notch off the later stat, and a nanosecond one can
# round the stored double on its last bit. 1e-5 s sits well above that float noise and
# well below the gap between two real rewrites, so an external edit always exceeds it
# while akasha's own write-then-reindex, which records the mtime it just wrote, never does.
MTIME_TOLERANCE = 1e-5

def _drift_backup(path: Path) -> Path:
    """The first free sibling backup name for a drifted file.

    The `.drift` / `.drift.N` suffix is not `.md`, so the indexer never picks a backup up
    as a document, and the counter means a second drift never destroys the first backup.
    """
    candidate = path.with_name(path.name + ".drift")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.drift.{counter}")
        counter += 1
    return candidate

def _check_no_drift(row: sqlite3.Row, path: Path) -> None:
    """Refuse a write to a file another writer changed since the index, backing it up.

    When the file's mtime no longer matches the one recorded at index time, something
    else wrote it and this caller's edit would silently clobber that work. The on-disk
    version is copied to a sibling backup, the original is left untouched, and the
    refusal names the recovery: read the current file and re-apply the change on top.
    """
    indexed_mtime = row["mtime"]
    if indexed_mtime is None:
        return            # no recorded mtime is not evidence of drift
    try:
        current_mtime = path.stat().st_mtime
    except OSError:
        raise DriftRefusal(
            f"{path} is gone from disk since akasha indexed it, so the update cannot be "
            "applied and there is nothing to back up. Find out what deleted it and "
            "re-create the document rather than editing this path.")
    if abs(current_mtime - indexed_mtime) <= MTIME_TOLERANCE:
        return
    backup = _drift_backup(path)
    shutil.copy2(path, backup)
    raise DriftRefusal(
        f"{path} changed on disk since akasha indexed it "
        f"(mtime {indexed_mtime} -> {current_mtime}): another writer got there first, "
        "so the update was NOT applied: it would have overwritten their work. The "
        "other writer's version still sits at the path above and is also backed up at "
        f"{backup}. Re-read the file now and re-apply your change on top of it; that "
        "is the only correct recovery.")

def _reindex(conn: sqlite3.Connection, cfg: Config, path: Path) -> None:
    # Lazy: index.py imports canonical_kind from here, so a module-level import of
    # index_path would be a cycle.
    from akasha.index import index_path

    # Reindex as whatever the document already is: passing "native" and no root would
    # relabel an indexed-root document and drop the repo and feature, both of which
    # index_path derives from the path relative to its root.
    row = conn.execute("SELECT source FROM documents WHERE path = ?", (str(path),)).fetchone()
    source = row["source"] if row else "native"
    root = _owning_root(cfg, path)

    conn.execute("DELETE FROM chunks WHERE document_id IN "
                 "(SELECT id FROM documents WHERE path = ?)", (str(path),))
    conn.execute("UPDATE documents SET mtime = NULL WHERE path = ?", (str(path),))
    index_path(conn, cfg, path, source,
               root=Path(root.path).expanduser() if root else None,
               root_repo=root.repo if root else None)
    conn.commit()

def write(
    conn: sqlite3.Connection,
    cfg: Config,
    title: str,
    body: str,
    repo: str | None = None,
    feature: str | None = None,
    kind: str = "reference",
    supersedes: list[str] | None = None,
) -> str:
    kind = canonical_kind(kind)
    folder = cfg.knowledge_dir / (repo or "_global")
    # The repo names a directory, so anything but a plain name could write outside the
    # knowledge dir; the resolve check also catches a symlinked folder.
    if repo and (not _REPO_NAME.fullmatch(repo) or repo in (".", "..")
                 or not folder.resolve().is_relative_to(cfg.knowledge_dir.resolve())):
        raise ValueError(
            f"invalid repo {repo!r}: use a plain name of letters, digits, '.', '_' and '-'")
    # Checked before any file exists: a bad id must not leave a document behind.
    for old_id in supersedes or []:
        old = _doc_row(conn, old_id)
        if old["deleted_at"] is not None:
            raise ValueError(f"cannot supersede {old_id}: it is deleted")
        _require_native(old)
    doc_id = new_id("k")
    folder.mkdir(parents=True, exist_ok=True)

    meta = {
        "id": doc_id, "title": title, "repo": repo, "feature": feature,
        "kind": kind, "status": "active", "supersedes": supersedes or [],
        "created": date.today().isoformat(), "updated": date.today().isoformat(),
    }
    text = render(meta, body)
    stem = f"{date.today().isoformat()}-{_slugify(title)}"
    counter = 1
    while True:
        path = folder / (f"{stem}.md" if counter == 1 else f"{stem}-{counter}.md")
        try:
            # Exclusive create: a name taken since it was chosen bumps the counter
            # instead of overwriting the other document.
            with path.open("x") as handle:
                handle.write(text)
            break
        except FileExistsError:
            counter += 1
    # Lazy for the same cycle as _reindex.
    from akasha.index import index_path

    index_path(conn, cfg, path, "native", root=None)

    for old_id in supersedes or []:
        archive(conn, cfg, old_id)
    conn.commit()
    return doc_id

SETTABLE_FIELDS = frozenset({"title", "status", "kind", "feature", "invalid_at", "supersedes"})

def update(
    conn: sqlite3.Connection,
    cfg: Config,
    doc_id: str,
    expected_updated: str | None = None,
    body: str | None = None,
    match: str | None = None,
    replacement: str | None = None,
    **fields,
) -> str:
    """Correct a document: a whole new body, or one substring named by `match`.

    The substring form exists because the caller pays for every token it resends:
    correcting one sentence should not cost the whole body.

    Ambiguity never resolves silently. Zero matches means the caller is describing a
    document that does not say what it thinks; more than one means the span does not
    identify the edit, and picking the first would be a guess. Both refuse with the count.
    """
    if (match is None) != (replacement is None):
        raise ValueError(
            "match and replacement go together: half the pair is not an edit, and "
            "defaulting the other half would silently mean something")
    if match is not None and body is not None:
        raise ValueError(
            "pass either body (the whole document) or match/replacement (one span), "
            "not both: they disagree about what the document should end up saying")
    unsettable = sorted(set(fields) - SETTABLE_FIELDS)
    if unsettable:
        raise ValueError(
            f"cannot set {', '.join(unsettable)}: they are identity or provenance. "
            f"Settable fields: {', '.join(sorted(SETTABLE_FIELDS))}")
    row = _doc_row(conn, doc_id)
    _require_native(row)
    if expected_updated is not None and expected_updated != row["updated_at"]:
        raise StaleWrite(
            f"{doc_id} changed since you read it (now {row['updated_at']}); re-read and retry")

    path = Path(row["path"])
    _check_no_drift(row, path)
    meta, current_body = parse_frontmatter(path.read_text())
    if fields.get("kind") is not None:
        fields["kind"] = canonical_kind(fields["kind"])
    for key, value in fields.items():
        if value is not None:
            meta[key] = value
    if match is not None:
        found = current_body.count(match)
        if found != 1:
            raise ValueError(
                f"{doc_id}: match occurs {found} times, and an edit must name exactly "
                f"one span. " + ("Nothing was changed. Re-read the document, since it "
                                 "does not say what you expected."
                                 if found == 0 else
                                 "Nothing was changed. Extend the match with "
                                 "surrounding text until it is unique."))
        body = current_body.replace(match, replacement, 1)
    meta["updated"] = date.today().isoformat()
    path.write_text(render(meta, body if body is not None else current_body))
    _reindex(conn, cfg, path)
    return doc_id

def append(
    conn: sqlite3.Connection, cfg: Config, doc_id: str, text: str, heading: str | None = None
) -> str:
    row = _doc_row(conn, doc_id)
    _require_appendable(row, cfg)
    path = Path(row["path"])
    # No drift check: append-only open cannot clobber what another writer put there.
    section = heading or f"## UPDATE {date.today().isoformat()}"
    with path.open("a") as handle:          # atomic append, order-independent
        handle.write(f"\n\n{section}\n\n{text}\n")
    _reindex(conn, cfg, path)
    return doc_id

def archive(conn: sqlite3.Connection, cfg: Config, doc_id: str) -> str:
    """Hide from default search. Reversible, and the only delete-shaped tool agents get.

    Written through `update`, because the column alone is a cache of the file: an archive
    that never reached the frontmatter is undone the moment anything touches the document,
    since index.py re-reads `status` from the file and skips one whose mtime has not moved.

    Also stamps `invalid_at`, the end of the validity interval `search(as_of=...)` answers
    by, through the same write for the same reason. Supersession reaches this path too:
    write() archives each document it supersedes.
    """
    return update(conn, cfg, doc_id, status="archived",
                  invalid_at=date.today().isoformat())

def touch(conn: sqlite3.Connection, document_ids: list[str]) -> None:
    """Reinforcement: a document that keeps being found is not cold. Best-effort.

    The bump rides the end of a read that has already succeeded, so a broken counter
    must never fail that read. The failure is recorded as an event rather than swallowed,
    so a counter that stopped counting is visible.
    """
    try:
        for document_id in set(document_ids):
            conn.execute(
                "UPDATE documents SET last_accessed=?, access_count=access_count+1 WHERE id=?",
                (now(), document_id))
        conn.commit()
    except Exception as exc:                                     # noqa: BLE001
        # If the connection is the thing that failed, even the event cannot be written.
        try:
            from akasha.events import emit
            emit(conn, "knowledge.touch_failed", error=str(exc),
                 count=len(set(document_ids)))
        except Exception:                                        # noqa: BLE001
            pass

def stale(
    conn: sqlite3.Connection, older_than_days: int = 180, never_accessed_only: bool = False
) -> list[dict]:
    """Documents nobody has read. A report: nothing is moved or deleted.

    Disuse is absence of evidence either way, so it never takes a document out of search.
    Leaving search takes evidence: supersession, or an explicit archive.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat()
    sql = ("SELECT id, title, path, repo, kind, source, created_at, last_accessed,"
           " access_count FROM documents WHERE deleted_at IS NULL AND created_at < ?"
           " AND IFNULL(last_accessed, '') < ?")
    if never_accessed_only:
        sql += " AND last_accessed IS NULL"
    return [dict(r) for r in conn.execute(sql + " ORDER BY created_at", (cutoff, cutoff))]

def _display_error(error_text: str) -> str:
    """Stats shows ~100 chars per failure. A read-only refusal leads with an absolute
    path and a validation error with framework boilerplate, so either would fill the
    whole cut and hide the reason. Display only; the stored event is untouched."""
    error = re.sub(r"^Error executing tool [^:]+:\s*", "", error_text)
    error = re.sub(r"^(\d+\s+)?validation errors? for \w+Arguments\s*", "", error)
    return re.sub(r"(?<![^\s'\"(])/\S+", lambda m: m.group(0).rsplit("/", 1)[-1], error)

def stats(conn: sqlite3.Connection, days: int = 30, limit: int = 20) -> dict:
    """Retrieval quality, read from the `knowledge.searched` and `tool.failed` events.

    Answers what share of searches came back empty, which surfaced documents nobody then
    opened, and which questions were asked more than once.

    Events are purged after EVENT_RETENTION_DAYS; a wider window is silently missing the
    tail it claims to cover, so `window_truncated` says so.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    searches = []
    for row in conn.execute(
            "SELECT payload FROM events WHERE kind='knowledge.searched' AND ts >= ?",
            (cutoff,)):
        try:
            searches.append(json.loads(row["payload"] or "{}"))
        except json.JSONDecodeError:
            continue

    zero_hits = sum(1 for p in searches if p.get("count") == 0)

    surfaced_counts: dict[str, int] = {}
    for p in searches:
        for doc_id in p.get("ids") or []:
            surfaced_counts[doc_id] = surfaced_counts.get(doc_id, 0) + 1

    never_opened = []
    if surfaced_counts:
        placeholders = ",".join("?" * len(surfaced_counts))
        rows = conn.execute(
            "SELECT id, path, access_count FROM documents"
            f" WHERE id IN ({placeholders}) AND deleted_at IS NULL AND access_count = 0",
            tuple(surfaced_counts))
        for row in rows:
            never_opened.append({"id": row["id"], "path": row["path"],
                                 "surfaced": surfaced_counts[row["id"]]})
        never_opened.sort(key=lambda r: -r["surfaced"])

    query_counts: dict[str, int] = {}
    for p in searches:
        query = p.get("query")
        if query:
            query_counts[query] = query_counts.get(query, 0) + 1
    repeated = sorted(
        ({"query": q, "count": n} for q, n in query_counts.items() if n > 1),
        key=lambda r: -r["count"])

    tool_failed_groups: dict[tuple[str, str], dict] = {}
    for row in conn.execute(
            "SELECT payload, ts FROM events WHERE kind='tool.failed' AND ts >= ? "
            "ORDER BY ts DESC", (cutoff,)):
        try:
            payload = json.loads(row["payload"] or "{}")
        except json.JSONDecodeError:
            continue
        tool = payload.get("tool") or "unknown"
        phase = payload.get("phase") or "unknown"
        key = (tool, phase)
        if key not in tool_failed_groups:
            # Rows arrive newest first, so the first error seen is the latest.
            flattened = " ".join(_display_error(payload.get("error") or "").split())
            truncated = (flattened[:100] + "...") if len(flattened) > 100 else flattened
            tool_failed_groups[key] = {"tool": tool, "phase": phase, "count": 0,
                                       "recent_error": truncated}
        tool_failed_groups[key]["count"] += 1

    tool_failed_rows = sorted(tool_failed_groups.values(), key=lambda r: -r["count"])

    return {
        "days": days,
        "window_truncated": days > EVENT_RETENTION_DAYS,
        "zero_hit": {"count": zero_hits, "total": len(searches)},
        "surfaced_never_opened": never_opened[:limit],
        "surfaced_never_opened_total": len(never_opened),
        "repeated_queries": repeated[:limit],
        "tool_failed": {"count": sum(r["count"] for r in tool_failed_rows),
                        "rows": tool_failed_rows[:limit]},
    }

def timeline(
    conn: sqlite3.Connection,
    feature: str | None = None,
    repo: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Documents in the order they were written: how an investigation is reconstructed.

    Entries carry no `path` (knowledge_get returns it) and omit the default-valued
    `status` and an absent `repo`, the same economy as `Hit.as_dict`.
    """
    from akasha.features import resolve_feature

    sql = ("SELECT id, title, kind, status, repo, created_at FROM documents"
           " WHERE deleted_at IS NULL")
    params: list = []
    if feature:
        feature_id = resolve_feature(conn, feature, create=False)
        if feature_id is None:
            return []
        sql += " AND feature_id = ?"
        params.append(feature_id)
    if repo:
        sql += " AND repo = ?"
        params.append(repo)
    sql += " ORDER BY created_at, rowid LIMIT ?"
    params.append(limit)
    entries = []
    for row in conn.execute(sql, params):
        entry = dict(row)
        if entry["status"] == "active":
            del entry["status"]
        if entry["repo"] is None:
            del entry["repo"]
        entries.append(entry)
    return entries

def remove(conn: sqlite3.Connection, cfg: Config, doc_id: str) -> Path:
    """Move the file to trash and soft-delete the row. Nothing is unlinked."""
    row = _doc_row(conn, doc_id)
    if row["deleted_at"] is not None:
        raise ValueError(f"{doc_id} is already deleted; restore or purge it")
    _require_native(row)
    trash = cfg.knowledge_dir.parent / "trash"
    trash.mkdir(parents=True, exist_ok=True)
    source = Path(row["path"])
    target = trash / f"{doc_id}__{source.name}"
    # A row whose file is already gone is what fsck reports as `missing_file`, and this is
    # the command that clears it, so a missing file must not make it crash. The row still
    # becomes a tombstone; there is simply nothing to put in the trash.
    if source.exists():
        source.rename(target)

    conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
    conn.execute(
        "UPDATE documents SET deleted_at=?, path=?, updated_at=? WHERE id=?",
        (now(), str(target), now(), doc_id))
    conn.commit()
    return target

def restore(conn: sqlite3.Connection, cfg: Config, doc_id: str) -> Path:
    row = _doc_row(conn, doc_id)
    if row["deleted_at"] is None:
        raise ValueError(f"{doc_id} is not deleted, so there is nothing to restore")
    trashed = Path(row["path"])
    if not trashed.exists():
        raise ValueError(f"{doc_id} has no file in the trash to restore; purge the row instead")
    original_name = trashed.name.partition("__")[2] or trashed.name
    folder = cfg.knowledge_dir / (row["repo"] or "_global")
    folder.mkdir(parents=True, exist_ok=True)
    stem, suffix = Path(original_name).stem, Path(original_name).suffix
    target = folder / original_name
    counter = 2
    while True:
        try:
            # A hard link fails if the name is taken, where rename would silently
            # replace the document that claimed it while this one was in the trash.
            os.link(trashed, target)
            break
        except FileExistsError:
            target = folder / f"{stem}-{counter}{suffix}"
            counter += 1
    trashed.unlink()
    conn.execute(
        "UPDATE documents SET deleted_at=NULL, path=?, mtime=NULL, updated_at=? WHERE id=?",
        (str(target), now(), doc_id))
    conn.commit()
    _reindex(conn, cfg, target)
    return target

def purge(conn: sqlite3.Connection, cfg: Config, older_than_days: int | None = None) -> int:
    """Permanently delete trashed documents. The only destructive operation in akasha.

    Only files inside the trash directory are unlinked; a row pointing elsewhere is
    skipped and reported as a `knowledge.purge_skipped` event.
    """
    from akasha.events import emit

    trash = (cfg.knowledge_dir.parent / "trash").resolve()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days or 0)).isoformat()
    purged = 0
    for row in conn.execute(
            "SELECT id, path, deleted_at FROM documents WHERE deleted_at IS NOT NULL"
    ).fetchall():
        # Age is the deletion time: rename keeps the file's mtime, which says how old
        # the document is, not how long it has been in the trash.
        if older_than_days is not None and row["deleted_at"] > cutoff:
            continue
        path = Path(row["path"])
        if not path.resolve().is_relative_to(trash):
            emit(conn, "knowledge.purge_skipped", id=row["id"], path=row["path"])
            continue
        path.unlink(missing_ok=True)
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (row["id"],))
        conn.execute("DELETE FROM links WHERE from_document_id = ? OR to_document_id = ?",
                     (row["id"], row["id"]))
        conn.execute("DELETE FROM documents WHERE id = ?", (row["id"],))
        purged += 1
    conn.commit()
    return purged
