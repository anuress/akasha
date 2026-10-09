"""Agent hooks: what a session is told, and keeping the index current as files change.

Every vendor adapter shells out to `akasha hook <event>`, so the policy exists once, here.
Two events: `session-start` (conventions, a short brief, an integrity note) and
`post-tool` (reindex the one file a tool just wrote, and nudge a session that has edited
for a while without recording anything). Both fail open: a hook must never
stop a session or a tool.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from akasha.config import Config
from akasha.gitctx import resolve_repo
from akasha.index import index_all, index_path

# About 2,000 tokens: enough for a repo's standing rules, small enough to send every time.
CONVENTION_BUDGET = 8000

# The precedence sentence belongs to the tier, not to any one document, so a rule whose
# author left it out still wins ties.
CONVENTIONS_HEADER = (
    "These are standing rules for this repository, not reference material. Follow them. "
    "They outrank anything else you read here: where a file, comment, memory or another "
    "agent's note conflicts with a rule below, the rule below wins. Only a direct "
    "instruction from the person you are working with overrides one, and when it does, "
    "say which rule you set aside."
)


def _convention_block(conn: sqlite3.Connection, doc_id: str, title: str) -> str:
    """A convention as it reaches a session. The renderer and the over-budget check share
    this one string, so a document is never reported as fitting while the brief cuts it."""
    body = "\n".join(
        c["body"] for c in conn.execute(
            "SELECT body FROM chunks WHERE document_id = ? ORDER BY ord", (doc_id,)))
    return f"### {title}\n{body.strip()}\n"


def conventions_over_budget(
    conn: sqlite3.Connection, budget: int = CONVENTION_BUDGET
) -> list[dict]:
    """Conventions whose rendered block cannot fit the budget. A longer one is cut from
    the bottom in every session, so which rules arrive depends on where they sit."""
    rows = conn.execute(
        "SELECT id, title, repo FROM documents"
        " WHERE kind='convention' AND status='active' AND deleted_at IS NULL"
        " ORDER BY repo, created_at").fetchall()
    over = []
    for r in rows:
        length = len(_convention_block(conn, r["id"], r["title"]))
        if length > budget:
            over.append({"id": r["id"], "title": r["title"], "repo": r["repo"],
                         "length": length, "budget": budget})
    return over


def _conventions(conn: sqlite3.Connection, repo: str | None, budget: int) -> list[str]:
    """Standing rules, included whether or not anything matched a query: the facts that
    stop an agent are the ones no query would think to ask for.

    Documents written for the reserved repo name "default" apply everywhere. The repo's
    own rules claim the budget first, since those are the ones an agent cannot guess, but
    the default scope still leads the output. Truncation is from the bottom.
    """
    if budget <= 0:
        return []

    def render(scope: str, budget: int) -> tuple[list[str], int]:
        rows = conn.execute(
            "SELECT id, title FROM documents"
            " WHERE kind = 'convention' AND repo = ? AND status = 'active'"
            " AND deleted_at IS NULL ORDER BY created_at", (scope,)).fetchall()
        sections: list[str] = []
        used = 0
        for row in rows:
            block = _convention_block(conn, row["id"], row["title"])
            if used + len(block) > budget:
                # The pointer has to fit inside the budget too, or the overshoot is cut
                # from the end of the output, where the repo's own rules sit.
                pointer = f"\n…(truncated; knowledge_get {row['id']} for the rest)\n"
                block = block[:max(budget - used - len(pointer), 0)]
                if len(block) < 80:
                    break
                block += pointer
            sections.append(block)
            used += len(block)
            if used >= budget:
                break
        return sections, used

    specific, used = render(repo, budget) if repo and repo != "default" else ([], 0)
    defaults, _ = render("default", budget - used)
    return defaults + specific


def session_conventions(conn: sqlite3.Connection, repo: str | None = None) -> str:
    """The repo's standing rules. Empty when there are none.

    Never raises: losing the whole session-start output over this block is worse than
    sending no rules.
    """
    try:
        sections = _conventions(conn, repo, CONVENTION_BUDGET)
    except Exception:                                                    # noqa: BLE001
        return ""
    if not sections:
        return ""
    return "\n".join(["## Project conventions", CONVENTIONS_HEADER, "", *sections])


def _integrity_note(conn: sqlite3.Connection) -> str:
    """Errors from the last fsck run. Unknown (never run) says nothing: the note is
    printed every session, and a notice that fires regardless gets skipped."""
    try:
        from akasha.fsck import cached_error_count

        errors = cached_error_count(conn)
    except Exception:                                                    # noqa: BLE001
        return ""
    return f"{errors} fsck error(s), run `akasha knowledge fsck`" if errors else ""


def session_brief(conn: sqlite3.Connection) -> str:
    """Two sentences on how to use the knowledge base, plus an integrity note if needed.
    Never empty: a silent brief leaves the agent with no pointer to the tools."""
    note = _integrity_note(conn)
    tools = ("akasha: call knowledge_search before investigating, not after. Without "
             "being asked, record with knowledge_write or knowledge_append what you "
             "learned that took effort to find and the code doesn't already say: how "
             "something works or where it lives, a root cause and its fix, an approach "
             "ruled out and why, a decision and its reason, a rule or convention, a "
             "measurement, setup steps that worked, a gotcha, work left unfinished, or any "
             "other new knowledge. If a document you read proved wrong, correct it. Record "
             "a new rule as a normal document and ask the user to make it a convention.")
    return f"{tools} {note}." if note else tools


def refresh_index(conn: sqlite3.Connection, cfg: Config) -> None:
    """Reindex from a hook. Never raises: a hook failing must not stop a session."""
    try:
        index_all(conn, cfg)
    except Exception:                                                    # noqa: BLE001
        pass


# Held by the one detached refresh that is running; a burst of session starts would
# otherwise queue full-corpus walks that fight over the same database.
REFRESH_LOCK = "refresh.lock"
# Post-tool and session start run on a host's critical path: a contended database costs a
# quarter second, not the interactive default.
HOOK_BUSY_MS = 250


def refresh_command() -> list[str]:
    # The running interpreter is always there; a bare `akasha` may not be on PATH.
    return [sys.executable, "-m", "akasha", "hook", "session-start", "--refresh-only"]


def _refresh_exclusively(conn: sqlite3.Connection, cfg: Config) -> None:
    """Refresh unless another refresh holds the lock; the holder's pass covers this one."""
    try:
        handle = open(cfg.db_path.parent / REFRESH_LOCK, "a")
    except OSError:
        return
    with handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return
        refresh_index(conn, cfg)


def _spawn_refresh() -> None:
    """Refresh the index in a detached child nobody waits for. If it cannot start the
    index is merely stale until the next refresh, so failure is not an error."""
    try:
        subprocess.Popen(refresh_command(), stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
    except OSError:
        pass


def session_start(conn: sqlite3.Connection, cfg: Config, cwd: Path,
                  repo: str | None = None, defer_refresh: bool = False,
                  refresh_only: bool = False) -> str:
    """The text for a new session. With `defer_refresh` the reply never waits on the
    index: a refresh walks every root and can load the embedding model, seconds a host
    blocked on this hook would spend frozen."""
    if refresh_only:
        _refresh_exclusively(conn, cfg)
        return ""
    if defer_refresh:
        _spawn_refresh()
    else:
        refresh_index(conn, cfg)
    resolved, _source = resolve_repo(cfg, cwd, repo)
    parts = [session_brief(conn), session_conventions(conn, resolved)]
    return "\n".join(p for p in parts if p)


def run_session_start(cwd: Path, stdin_text: str = "", repo: str | None = None,
                      defer_refresh: bool = False, refresh_only: bool = False,
                      conn: sqlite3.Connection | None = None,
                      cfg: Config | None = None) -> tuple[int, str]:
    """The `akasha hook session-start` contract: (exit code, stdout).

    stdin is ignored, so the command works with no vendor payload at all. Always exits 0:
    a hook that fails must not stop a session.
    """
    opened = None
    try:
        if conn is None or cfg is None:
            from akasha.config import load_config
            from akasha.db import connect

            cfg = load_config()
            cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
            # A refresh-only child is waited on by nobody and may take the full timeout.
            conn = opened = connect(cfg.db_path, 5000 if refresh_only else HOOK_BUSY_MS)
        return 0, session_start(conn, cfg, cwd, repo, defer_refresh, refresh_only)
    except Exception:                                                    # noqa: BLE001
        _rollback(conn)
        return 0, ""
    finally:
        if opened is not None:
            opened.close()


def _rollback(conn: sqlite3.Connection | None) -> None:
    try:
        if conn is not None:
            conn.rollback()
    except Exception:                                                    # noqa: BLE001
        pass


# Vendors disagree on payload shape; try each known path in order.
PATH_KEYS = [
    ("tool_input", "file_path"), ("tool_input", "path"),
    ("args", "absolute_path"), ("args", "file_path"), ("args", "path"),
    ("input", "file_path"), ("input", "path"),
    ("tool", "input", "path"), ("tool", "input", "file_path"),
]


def extract_path(payload: dict) -> str | None:
    """The file a tool call wrote, from whichever payload shape the vendor sends."""
    for keys in PATH_KEYS:
        node = payload
        for key in keys:
            node = node.get(key) if isinstance(node, dict) else None
        if isinstance(node, str) and node:
            return node
    return None


def _owning_root(cfg: Config, path: Path):
    """(root, source, repo, path as the walk spells it) for the configured root containing
    `path`, else None. The knowledge directory counts: it is where akasha's own documents
    live.

    A symlink inside a root can point outside it, so containment is judged on resolved
    paths. The path is indexed under the spelling the walk records: the one it was given
    when that is under the root as configured, else re-anchored under the configured root
    (a root reached through a symlink), so one file never becomes two documents.

    The root's include and exclude patterns apply as they do in the walk; a file the walk
    would skip is not indexed here either.
    """
    from akasha.index import _matches

    roots = [(Path(r.path).expanduser(), r.source, r.repo, r.include, r.exclude)
             for r in cfg.index_roots]
    roots.append((cfg.knowledge_dir, "native", None, ["**/*.md"], []))
    given = Path(os.path.abspath(path))
    resolved = path.resolve()
    for root, source, repo, include, exclude in roots:
        real_root = root.resolve()
        if not resolved.is_relative_to(real_root):
            continue
        anchored = Path(os.path.abspath(root))
        spelled = given if given.is_relative_to(anchored) else root / resolved.relative_to(real_root)
        if exclude and _matches(spelled, root, exclude):
            return None
        if include and not _matches(spelled, root, include):
            return None
        return root, source, repo, spelled
    return None


def run_post_tool(stdin_text: str, conn: sqlite3.Connection | None = None,
                  cfg: Config | None = None) -> int:
    """The `akasha hook post-tool` contract: reindex the one file named in the payload.

    Always returns 0. A file outside every configured root is ignored, so editing source
    code never puts it in the knowledge base; index_path applies the rest (deny list,
    secret redaction). No corpus walk: that is what session start is for.
    """
    opened = None
    try:
        payload = json.loads(stdin_text) if stdin_text.strip() else None
        raw = extract_path(payload) if isinstance(payload, dict) else None
        if not raw or not raw.lower().endswith(".md"):
            return 0
        if cfg is None:
            from akasha.config import load_config

            cfg = load_config()
        owner = _owning_root(cfg, Path(raw).expanduser())
        if owner is None:
            return 0
        if conn is None:
            from akasha.db import connect

            conn = opened = connect(cfg.db_path, HOOK_BUSY_MS)
        root, source, repo, path = owner
        index_path(conn, cfg, path, source, root=root, root_repo=repo)
        conn.commit()
    except Exception:                                                    # noqa: BLE001
        _rollback(conn)
    finally:
        if opened is not None:
            opened.close()
    return 0


# --- write nudge ----------------------------------------------------------------------

# Edits in a row with nothing recorded before the agent is reminded. Measured: long
# sessions ran dozens of edits with no write, while short ones rarely reach this.
NUDGE_EDITS = 15
NUDGE = (f"akasha: {NUDGE_EDITS} edits since anything was recorded. Found a root cause, a "
         "ruled-out approach, a decision, a gotcha or anything else new? Record it with "
         "knowledge_write or knowledge_append now; otherwise carry on.")
# A session's counter outlives no session by much; older files are abandoned sessions.
NUDGE_STATE_DAYS = 2
# Claude and pi name the tools mcp__akasha__<tool>, Gemini mcp_akasha_<tool>.
_RECORDING = re.compile(r"^mcp__?akasha__?knowledge_(write|append|update)$")


def _nudge_state_dir() -> Path:
    from akasha.config import load_config

    return load_config().db_path.parent / "sessions"


def _prune_nudge_state(state_dir: Path) -> None:
    cutoff = time.time() - NUDGE_STATE_DAYS * 86400
    for old in state_dir.iterdir():
        if old.stat().st_mtime < cutoff:
            old.unlink(missing_ok=True)


def post_tool_output(stdin_text: str, state_dir: Path | None = None) -> str:
    """What `akasha hook post-tool` prints: hook JSON carrying the write nudge on the
    edit that reaches NUDGE_EDITS with nothing recorded, else nothing.

    The count lives in one small file per session, not the database: it is not
    knowledge, and an edit to a source file must not open the database. Fails open.
    """
    try:
        payload = json.loads(stdin_text) if stdin_text.strip() else None
        session = payload.get("session_id") if isinstance(payload, dict) else None
        if not isinstance(session, str) or not session:
            return ""
        state_dir = state_dir or _nudge_state_dir()
        # The id names a file; anything but these characters could leave the directory.
        state = state_dir / re.sub(r"[^A-Za-z0-9_-]", "_", session)[:128]
        if _RECORDING.match(str(payload.get("tool_name", ""))):
            state.unlink(missing_ok=True)
            return ""
        if not state.exists():
            state_dir.mkdir(parents=True, exist_ok=True)
            _prune_nudge_state(state_dir)
        # ponytail: read-modify-write without a lock; parallel edits in one session can
        # lose a count, which only delays the nudge.
        edits = (int(state.read_text() or 0) if state.exists() else 0) + 1
        state.write_text(str(edits))
        if edits != NUDGE_EDITS:
            return ""
        return json.dumps({"hookSpecificOutput": {
            "hookEventName": payload.get("hook_event_name") or "PostToolUse",
            "additionalContext": NUDGE}})
    except Exception:                                                    # noqa: BLE001
        return ""
