"""MCP surface: thin wrappers over the same core functions the CLI calls.

Deleting is deliberately absent. An unattended agent should not hold a delete button;
rm, restore and purge stay at the CLI.
"""
from __future__ import annotations

import contextvars
import dataclasses
import json
import sqlite3
import sys
import threading
from collections.abc import Mapping
from pathlib import Path

from akasha import fsck
from akasha import knowledge as kb
from akasha import links as lk
from akasha import vectors
from akasha.config import Config, load_config
from akasha.db import SchemaMismatch, connect
from akasha.events import emit
from akasha.features import show as feature_show_core
from akasha.gitctx import resolve_repo
from akasha.search import iso_date, link_candidates, search as do_search
from akasha.security import redact, scrub_injection


def _ctx():
    cfg = load_config()
    return connect(cfg.db_path), cfg


def _resolved_repo(cfg: Config, args: dict) -> str | None:
    return resolve_repo(cfg, Path.cwd(), args.get("repo"))[0]


# The caller is an agent with a context window, so how much a tool returns is part of its
# contract. These ceilings only bite on the pathological case (a single chunk can be
# megabytes); ordinary results pass untouched.
MAX_HIT_CHARS = 2000
MAX_DOC_CHARS = 20000
MAX_FINDINGS = 50
FSCK_LIMIT = 10
TIMELINE_LIMIT = 15
# A near-duplicate a few places down the list is common: recall at 15 is noticeably
# better than at 10 for a few dozen extra tokens.
FOLD_LIMIT = 15


def _cap(text: str, limit: int, offset: int = 0) -> tuple[str, int]:
    """At most `limit` characters from `offset`, and how many were withheld. The count is
    returned rather than implied by a marker: "there is more" is something a caller can
    act on."""
    text = text or ""
    window = text[offset:offset + limit] if limit > 0 else ""
    return window, max(len(text) - offset - len(window), 0)


def _at_least(args: dict, name: str, default: int, low: int = 1, high: int | None = None) -> int:
    value = args.get(name, default)
    if not isinstance(value, int) or value < low or (high is not None and value > high):
        bound = f"{low} to {high}" if high is not None else f"at least {low}"
        raise ValueError(f"{name} must be {bound}, got {value!r}")
    return value


def _knowledge_search(args: dict):
    limit = _at_least(args, "limit", 5)
    if args.get("as_of") is not None:
        iso_date(args["as_of"])
    conn, cfg = _ctx()
    repo = _resolved_repo(cfg, args)
    hits = do_search(
        conn, args["q"], cfg=cfg, repo=repo, feature=args.get("feature"),
        source=args.get("source"), kind=args.get("kind"), limit=limit,
        include_archived=args.get("include_archived", False),
        all_repos=args.get("all_repos", False), match_mode=args.get("match_mode", "auto"),
        as_of=args.get("as_of"))
    max_chars = args.get("max_chars", MAX_HIT_CHARS)

    # A Hit has no source, so one lookup covers every hit.
    sources: dict[str, str] = {}
    if hits:
        ids = [h.document_id for h in hits]
        marks = ",".join("?" * len(ids))
        sources = {r["id"]: r["source"] for r in conn.execute(
            f"SELECT id, source FROM documents WHERE id IN ({marks})", ids)}

    out = []
    for h in hits:
        # Chunks indexed before the scan existed sit raw in the table, so the body is
        # scrubbed on the way out: a live instruction-shaped span must not reach the caller.
        text, fired = scrub_injection(h.text)
        if fired:
            emit(conn, "security.injection", document=h.document_id, path=h.path, rules=fired)
        text, withheld = _cap(text, max_chars)
        hit = dataclasses.replace(h, text=text).as_dict(withheld)
        if kb.readonly_hint(sources.get(h.document_id, "native")) is not None:
            hit["writable"] = False
        out.append(hit)

    # The query is user text, so it passes through the same redaction as other output.
    query_text, _ = redact(args["q"])
    emit(conn, "knowledge.searched", query=query_text, ids=[h.document_id for h in hits],
         count=len(hits),
         mode="all_repos" if args.get("all_repos") else ("repo" if repo else "default"),
         match_mode=args.get("match_mode", "auto"))
    return out


def _knowledge_get(args: dict):
    _at_least(args, "offset", 0, low=0)
    conn, cfg = _ctx()
    row = conn.execute("SELECT * FROM documents WHERE id = ?", (args["id"],)).fetchone()
    if row is None:
        return {"error": f"unknown document: {args['id']}"}
    text = Path(row["path"]).read_text()
    # Indexing redacts the chunks; the file itself must not be a way around that. Applied
    # on the way out only, so the user's own notes are never rewritten.
    redacted: list[str] = []
    if cfg.scan_secrets:
        text, redacted = redact(text)
    text, fired = scrub_injection(text)
    if fired:
        emit(conn, "security.injection", document=row["id"], path=row["path"], rules=fired)
    # Redaction runs on the whole text before the window is taken: after the cut, a secret
    # could straddle the boundary and escape.
    offset = args.get("offset", 0)
    text, withheld = _cap(text, args.get("max_chars", MAX_DOC_CHARS), offset)
    result = {"id": row["id"], "title": row["title"], "path": row["path"],
              "status": row["status"], "text": text}
    for name, value in (("redacted", redacted), ("injection", fired),
                        ("withheld_chars", withheld), ("offset", offset)):
        if value:
            result[name] = value
    hint = kb.readonly_hint(row["source"])
    if hint is not None:
        result["writable"] = False
        result["readonly_hint"] = hint
    return result


def _fold_candidates(conn, cfg, title: str, body: str) -> dict:
    """Documents the new one may belong in. Looked up before the write: once the document
    exists, a query drawn from its own text matches itself and starves the real neighbours.

    A failure here must neither lose the document nor claim the base held no near
    duplicate, so the reason is returned for the response to carry."""
    try:
        hits = link_candidates(conn, cfg, title, body, limit=FOLD_LIMIT)
    except Exception as exc:
        return {"fold_candidates_unavailable": f"{type(exc).__name__}: {exc}"}
    ranked: list[str] = []
    for h in hits:
        if h.document_id not in ranked:
            ranked.append(h.document_id)
    if not ranked:
        return {}
    titles = {r["id"]: r["title"] for r in conn.execute(
        f"SELECT id, title FROM documents WHERE id IN ({','.join('?' * len(ranked))})", ranked)}
    return {"fold_candidates": [{"id": d, "title": titles.get(d, "")} for d in ranked]}


def _refuse_convention(kind: str | None) -> None:
    # A convention is injected into every session brief for its repo as an instruction.
    if (kind or "").lower() == "convention":
        raise PermissionError(
            "kind='convention' is CLI-only: it becomes a standing instruction in every "
            "session. Record findings with kind='finding' instead.")


def _refuse_convention_edit(conn, doc_id: str) -> None:
    # Reading the kind from the row, not the argument: the target is what matters.
    row = conn.execute("SELECT kind FROM documents WHERE id = ?", (doc_id,)).fetchone()
    if row is not None and row["kind"] == "convention":
        raise PermissionError(
            f"{doc_id} is a convention: conventions are CLI-only, because each is a "
            "standing instruction in every session. Record new findings with "
            "knowledge_write instead.")


def _written(conn, tool: str, doc_id: str, **extra) -> None:
    emit(conn, "knowledge.written", tool=tool, document=doc_id, **extra)


def _knowledge_write(args: dict):
    conn, cfg = _ctx()
    kind = args.get("kind", "reference")
    _refuse_convention(kind)
    for old_id in args.get("supersedes") or []:
        _refuse_convention_edit(conn, old_id)
    repo = _resolved_repo(cfg, args)
    fold = _fold_candidates(conn, cfg, args["title"], args["body"])
    doc_id = kb.write(conn, cfg, args["title"], args["body"], repo=repo,
                      feature=args.get("feature"), kind=kind,
                      supersedes=args.get("supersedes"))
    _written(conn, "knowledge_write", doc_id, repo=repo)
    # repo is echoed because it is derived, not always what the caller typed.
    return {"id": doc_id, "repo": repo, **fold}


def _knowledge_update(args: dict):
    conn, cfg = _ctx()
    _refuse_convention(args.get("kind"))
    _refuse_convention_edit(conn, args["id"])
    kb.update(conn, cfg, args["id"], expected_updated=args.get("expected_updated"),
              body=args.get("body"), match=args.get("match"),
              replacement=args.get("replacement"), title=args.get("title"),
              kind=args.get("kind"), status=args.get("status"))
    _written(conn, "knowledge_update", args["id"])
    return {"id": args["id"]}


def _knowledge_append(args: dict):
    conn, cfg = _ctx()
    _refuse_convention_edit(conn, args["id"])
    kb.append(conn, cfg, args["id"], args["text"], heading=args.get("heading"))
    _written(conn, "knowledge_append", args["id"])
    return {"id": args["id"]}


def _knowledge_archive(args: dict):
    conn, cfg = _ctx()
    _refuse_convention_edit(conn, args["id"])
    doc_id = kb.archive(conn, cfg, args["id"])
    _written(conn, "knowledge_archive", doc_id)
    return {"id": doc_id}


def _knowledge_fsck(args: dict):
    limit = _at_least(args, "limit", FSCK_LIMIT)
    conn, cfg = _ctx()
    return fsck.report(conn, cfg, limit)


def _knowledge_timeline(args: dict):
    limit = _at_least(args, "limit", TIMELINE_LIMIT)
    conn, cfg = _ctx()
    return kb.timeline(conn, feature=args.get("feature"), repo=_resolved_repo(cfg, args),
                       limit=limit)


def _knowledge_related(args: dict):
    depth = _at_least(args, "depth", 1, high=5)
    conn, _ = _ctx()
    if depth > 1:
        return lk.graph(conn, args["id"], depth=depth)
    return lk.related(conn, args["id"])


def _feature_show(args: dict):
    conn, _ = _ctx()
    return feature_show_core(conn, args["slug"])


def doctor_report(findings: list, limit: int = MAX_FINDINGS) -> dict:
    """Degraded and noteworthy findings, errors first so the cap cannot bury them. Healthy
    counts are the report, not the signal."""
    from akasha.doctor import exit_code

    actionable = sorted((f for f in findings if f.severity != "info"),
                        key=lambda f: 0 if f.severity == "error" else 1)
    return {"exit_code": exit_code(findings), "total": len(actionable),
            "findings": [{"kind": f.kind, "severity": f.severity, "detail": f.detail}
                         for f in actionable[:limit]],
            "withheld": max(len(actionable) - limit, 0)}


def _doctor(args: dict):
    from akasha.doctor import check

    limit = _at_least(args, "limit", MAX_FINDINGS)
    conn, cfg = _ctx()
    return doctor_report(check(cfg, conn), limit)


TOOLS = {
    "knowledge_search": _knowledge_search,
    "knowledge_get": _knowledge_get,
    "knowledge_write": _knowledge_write,
    "knowledge_update": _knowledge_update,
    "knowledge_append": _knowledge_append,
    "knowledge_archive": _knowledge_archive,
    "knowledge_fsck": _knowledge_fsck,
    "knowledge_timeline": _knowledge_timeline,
    "knowledge_related": _knowledge_related,
    "feature_show": _feature_show,
    "doctor": _doctor,
}

# Written for the agent that reads them in every session: short, and about what to pass.
DESCRIPTIONS = {
    "knowledge_search": (
        "Search the knowledge base; returns ranked chunks. Prefers the current repo but "
        "falls back to all repos (cross_repo marks those); pass all_repos=true when the "
        "question is not about this checkout. A strict match that finds nothing is retried "
        "as OR (relaxed marks those). Text is capped at max_chars; withheld_chars says how "
        "much was cut. neighbours counts linked documents: when above 0, knowledge_related "
        "may help. as_of (YYYY-MM-DD) answers from what was valid then."),
    "knowledge_get": (
        "Fetch one document by id, with its path. Long text is paged: pass offset to "
        "continue where withheld_chars says it stopped."),
    "knowledge_write": (
        "Write a new document. Search first and use knowledge_append on an existing "
        "document rather than writing a near-duplicate. Record the decision rule, the "
        "numbers, what was ruled out and why, not a summary of what you read. If an earlier "
        "conclusion proved wrong, pass its id in supersedes. kind: reference, plan, spec, "
        "finding, review, progress, decision, gotcha, log. The response lists "
        "fold_candidates, documents this may belong in; read them before treating the write "
        "as done. repo defaults to the current checkout and the response says which."),
    "knowledge_update": (
        "Correct a document in place; for new results use knowledge_append. To change one "
        "span, pass match (must occur exactly once) and replacement instead of the whole "
        "body. A match found zero or several times changes nothing and reports the count."),
    "knowledge_append": "Append a dated section to a document. Cheap and cannot overwrite.",
    "knowledge_archive": "Hide a document from default search. Reversible.",
    "knowledge_fsck": (
        "Integrity check of the knowledge base; changes nothing. Returns counts per kind, "
        "the first `limit` findings, and withheld, the number not listed."),
    "knowledge_timeline": (
        "Documents in the order they were written, optionally for one feature. repo "
        "defaults to the current checkout; '*' means every repo."),
    "knowledge_related": (
        "Documents linked to this one: citations and backlinks. depth above 1 walks further."),
    "feature_show": "How many documents carry a feature.",
    "doctor": (
        "What is degraded about this installation. exit_code 0 means healthy; findings are "
        "the errors and notes, errors first."),
}

# A handler failure the caller can act on is returned as a message; anything else is a
# bug and propagates (the framework reports it) after being recorded.
EXPECTED_ERRORS = (ValueError, KeyError, PermissionError, kb.StaleWrite,
                   kb.ReadOnlySource, kb.DriftRefusal)

# Set by the validate middleware before the call and mutated in place by
# _record_tool_failure, so a failure recorded in a handler is not counted again when the
# framework turns it into an error result. A dict rather than a bool because the call runs
# in a worker thread, where a ContextVar.set() would not be visible to the caller.
_CURRENT_TOOL_CALL: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "akasha_current_tool_call", default=None)

def _field_names(error: object) -> str:
    """What a validate-phase event may keep of a framework error. The message echoes the
    offending argument values, so an exception reduces to its class and a message to its
    unindented lines, which are the field names; the indented detail is dropped."""
    if isinstance(error, BaseException):
        return type(error).__name__
    return "\n".join(line for line in str(error).splitlines() if line and not line[0].isspace())


def _record_tool_failure(tool: str, phase: str, error: object, **extra) -> None:
    """Best-effort audit of a failed call. Never raises: a broken counter must not mask
    the real error."""
    marker = _CURRENT_TOOL_CALL.get()
    if marker is not None:
        marker["recorded"] = True
    try:
        conn, _cfg = _ctx()
        text, _ = redact(_field_names(error) if phase == "validate" else str(error))
        emit(conn, "tool.failed", tool=tool, phase=phase, error=text, **extra)
    except Exception:                                          # noqa: BLE001
        pass


def _is_infrastructure(exc: Exception) -> bool:
    """Disk, database or schema trouble: not something the caller can fix, and its message
    carries paths. An OSError without an errno is a refusal of ours, not the OS's."""
    return (isinstance(exc, (sqlite3.Error, SchemaMismatch))
            or (isinstance(exc, OSError) and exc.errno is not None))


def call_tool(name: str, args: dict):
    if name not in TOOLS:
        raise KeyError(f"unknown tool: {name}")
    try:
        result = TOOLS[name](args)
    except Exception as exc:
        if _is_infrastructure(exc):
            _record_tool_failure(name, "handler", f"{type(exc).__name__}: {exc}")
            return {"error": f"{name} failed ({type(exc).__name__}); run akasha doctor"}
        if isinstance(exc, EXPECTED_ERRORS):
            message = str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else str(exc)
            _record_tool_failure(name, "result", message)
            return {"error": message}
        _record_tool_failure(name, "handler", exc)
        raise
    if isinstance(result, dict) and "error" in result:
        _record_tool_failure(name, "result", result["error"])
    return result


def _is_error_result(result) -> bool:
    # The result a middleware sees is the wire dict (camelCase isError), though a typed
    # model is possible too.
    if isinstance(result, Mapping):
        return bool(result.get("isError", False))
    return bool(getattr(result, "is_error", False))


def _error_text(result) -> str:
    content = (result.get("content") if isinstance(result, Mapping)
               else getattr(result, "content", None))
    parts = []
    for block in content or []:
        text = block.get("text") if isinstance(block, Mapping) else getattr(block, "text", None)
        if text:
            parts.append(text)
    return " ".join(parts) or "tool call failed"


# One spelling per concept: a wrong spelling fails loudly naming the right one, instead of
# being silently aliased. Checked on the raw arguments, before the framework builds the
# tool's model; the schema error alone names only the field it wanted, not the one sent.
_WRONG_SPELLINGS: dict[str, dict[str, tuple[str, ...]]] = {
    "knowledge_write": {"body": ("content", "text")},
    "knowledge_update": {"body": ("content", "text")},
    "knowledge_append": {"body": ("text", "content")},
    "knowledge_get": {"id": ("document_id",)},
    "knowledge_search": {"query": ("q",)},
}


def _wrong_spelling_refusal(tool: str, arguments: Mapping) -> str | None:
    complaints = []
    for canonical, wrongs in _WRONG_SPELLINGS.get(tool, {}).items():
        sent = [w for w in wrongs if w in arguments]
        if sent:
            listed = ", ".join(f"`{w}`" for w in sent)
            complaints.append(f"{listed} {'is' if len(sent) == 1 else 'are'} not a "
                              f"parameter; the parameter is `{canonical}`")
    return f"{tool}: " + ". ".join(complaints) + "." if complaints else None


async def _validate_failure_middleware(ctx, call_next):
    """Records request-side failures that die inside the framework before any handler
    runs, and refuses known wrong spellings. Notifications are never failed tool calls."""
    is_tool_call = ctx.request_id is not None and ctx.method == "tools/call"
    tool, arg_names, marker, token = "", [], None, None
    if is_tool_call:
        params = ctx.params if isinstance(ctx.params, Mapping) else {}
        tool = params.get("name", "")
        arguments = params.get("arguments")
        arg_names = sorted(arguments) if isinstance(arguments, Mapping) else []
        if isinstance(arguments, Mapping):
            refusal = _wrong_spelling_refusal(tool, arguments)
            if refusal is not None:
                _record_tool_failure(tool, "validate", refusal, args=arg_names)
                return {"content": [{"type": "text", "text": refusal}], "isError": True}
        marker = {"recorded": False}
        token = _CURRENT_TOOL_CALL.set(marker)
    try:
        try:
            result = await call_next(ctx)
        except Exception as exc:
            if is_tool_call:
                _record_tool_failure(tool, "validate", exc, args=arg_names)
            raise
    finally:
        if token is not None:
            _CURRENT_TOOL_CALL.reset(token)
    if is_tool_call and not marker["recorded"] and _is_error_result(result):
        _record_tool_failure(tool, "validate", _error_text(result), args=arg_names)
    return result


def _json(result) -> str:
    # Compact: whitespace is tokens with no reader.
    return json.dumps(result, separators=(",", ":"), default=str)


def _strip_titles(node):
    """Drop the `title` keyword pydantic generates for every property. A property that is
    itself named title is a key of `properties`, so it is kept."""
    if isinstance(node, dict):
        for key in [k for k, v in node.items() if k == "title" and isinstance(v, str)]:
            del node[key]
        for key, value in node.items():
            if key == "properties" and isinstance(value, dict):
                for sub in value.values():
                    _strip_titles(sub)
            else:
                _strip_titles(value)
    elif isinstance(node, list):
        for item in node:
            _strip_titles(item)


INSTRUCTIONS = "Local knowledge base. Search before investigating; record findings unprompted."


def build_server():
    """The MCP server with every tool declared by an explicit signature: the generated
    input schema is how an agent learns what to pass."""
    from mcp.server import MCPServer

    server = MCPServer("akasha", version="0.0.1", instructions=INSTRUCTIONS,
                       middleware=[_validate_failure_middleware])

    @server.tool(description=DESCRIPTIONS["knowledge_search"])
    def knowledge_search(query: str, repo: str | None = None, feature: str | None = None,
                         source: str | None = None, kind: str | None = None,
                         limit: int = 5, include_archived: bool = False,
                         all_repos: bool = False, match_mode: str = "auto",
                         max_chars: int = MAX_HIT_CHARS, as_of: str | None = None) -> str:
        return _json(call_tool("knowledge_search", {
            "q": query, "repo": repo, "feature": feature, "source": source, "kind": kind,
            "limit": limit, "include_archived": include_archived, "all_repos": all_repos,
            "match_mode": match_mode, "max_chars": max_chars, "as_of": as_of}))

    @server.tool(description=DESCRIPTIONS["knowledge_get"])
    def knowledge_get(id: str, max_chars: int = MAX_DOC_CHARS, offset: int = 0) -> str:
        return _json(call_tool("knowledge_get", {
            "id": id, "max_chars": max_chars, "offset": offset}))

    @server.tool(description=DESCRIPTIONS["knowledge_write"])
    def knowledge_write(title: str, body: str, repo: str | None = None,
                        feature: str | None = None, kind: str = "reference",
                        supersedes: list[str] | None = None) -> str:
        return _json(call_tool("knowledge_write", {
            "title": title, "body": body, "repo": repo, "feature": feature, "kind": kind,
            "supersedes": supersedes}))

    @server.tool(description=DESCRIPTIONS["knowledge_update"])
    def knowledge_update(id: str, body: str | None = None, match: str | None = None,
                         replacement: str | None = None, title: str | None = None,
                         kind: str | None = None, status: str | None = None,
                         expected_updated: str | None = None) -> str:
        return _json(call_tool("knowledge_update", {
            "id": id, "body": body, "match": match, "replacement": replacement,
            "title": title, "kind": kind, "status": status,
            "expected_updated": expected_updated}))

    @server.tool(description=DESCRIPTIONS["knowledge_append"])
    def knowledge_append(id: str, body: str, heading: str | None = None) -> str:
        # Exposed as `body`, the name every other content-taking tool uses; the core
        # function calls it `text`.
        return _json(call_tool("knowledge_append", {"id": id, "text": body,
                                                    "heading": heading}))

    @server.tool(description=DESCRIPTIONS["knowledge_archive"])
    def knowledge_archive(id: str) -> str:
        return _json(call_tool("knowledge_archive", {"id": id}))

    @server.tool(description=DESCRIPTIONS["knowledge_fsck"])
    def knowledge_fsck(limit: int = FSCK_LIMIT) -> str:
        return _json(call_tool("knowledge_fsck", {"limit": limit}))

    @server.tool(description=DESCRIPTIONS["knowledge_timeline"])
    def knowledge_timeline(feature: str | None = None, repo: str | None = None,
                           limit: int = TIMELINE_LIMIT) -> str:
        return _json(call_tool("knowledge_timeline", {
            "feature": feature, "repo": repo, "limit": limit}))

    @server.tool(description=DESCRIPTIONS["knowledge_related"])
    def knowledge_related(id: str, depth: int = 1) -> str:
        return _json(call_tool("knowledge_related", {"id": id, "depth": depth}))

    @server.tool(description=DESCRIPTIONS["feature_show"])
    def feature_show(feature: str) -> str:
        return _json(call_tool("feature_show", {"slug": feature}))

    @server.tool(description=DESCRIPTIONS["doctor"])
    def doctor(limit: int = MAX_FINDINGS) -> str:
        return _json(call_tool("doctor", {"limit": limit}))

    # Reaches into the framework's private tool registry; losing it only makes schemas
    # larger, so it degrades instead of stopping the server.
    try:
        tools = server._tool_manager.list_tools()
    except AttributeError:
        print("akasha: warning: could not strip schema titles; the framework's internals "
              "changed, so tool schemas are larger than intended", file=sys.stderr)
    else:
        for tool in tools:
            _strip_titles(tool.parameters)
    return server


def _warm_embedder() -> threading.Thread | None:
    """Load the embedding model in the background so the first hybrid search does not pay
    for it. The checks run in the thread too: the server's first response never waits."""
    def load() -> None:
        try:
            conn, cfg = _ctx()
            if vectors.available(conn):
                vectors._encoder()
        except Exception:                                      # noqa: BLE001
            pass                    # search degrades visibly through doctor, not here

    try:
        cfg = load_config()
    except Exception:                                          # noqa: BLE001
        return None
    if not vectors.enabled(cfg):
        return None
    thread = threading.Thread(target=load, name="akasha-warm", daemon=True)
    thread.start()
    return thread


def serve() -> None:
    """Run the stdio MCP server."""
    server = build_server()
    _warm_embedder()
    server.run("stdio")
