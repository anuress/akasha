"""argparse CLI: a thin wrapper over the core modules. No logic lives here."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

from akasha import cleanup, feed as feed_core, fsck, install, markdown, vectors
from akasha import knowledge as kb
from akasha import links as lk
from akasha.config import load_config
from akasha.config_edit import (ConfigWriteError, _split_section, get_key, set_key,
                                unset_key, write_config)
from akasha.db import connect
from akasha.features import add_alias, list_features, resolve_feature, show as feature_show
from akasha.gitctx import git_context, resolve_repo
from akasha.index import forget, index_all
from akasha.search import iso_date, search as do_search
from akasha.splitters import FeedSettings, clean_headers

# Failures a person can act on are printed as one line, never a traceback.
EXPECTED_ERRORS = (ValueError, KeyError, PermissionError, kb.StaleWrite, kb.ReadOnlySource,
                   kb.DriftRefusal, ConfigWriteError, OSError)


def _config_path() -> Path:
    """Resolved per call, never at import: a constant would bind the real home before a
    test can redirect it."""
    return Path.home() / ".akasha" / "config.toml"


def _ctx():
    cfg = load_config()
    return connect(cfg.db_path), cfg


def _message(exc: Exception) -> str:
    return str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else str(exc)


def cmd_init(args) -> int:
    home = Path.home() / ".akasha"
    cfg_path = home / "config.toml"
    if not args.dry_run:
        home.mkdir(mode=0o700, exist_ok=True)
        os.chmod(home, 0o700)
    if not cfg_path.exists():
        from akasha.discover import config_from_discovery, discover_sources

        code_roots = [Path(p).expanduser() for p in args.code_root] or [Path.home()]
        roots = discover_sources(Path.home(), code_roots)
        print(f"discovered {len(roots)} source(s):")
        for r in roots:
            print(f"  {r.source:16} {r.path}")
        dense = vectors.extra_installed()
        if dense:
            print('dense search: on ([embeddings] provider = "model2vec")')
        else:
            print("dense search is off: the vectors extra isn't installed; "
                  "reinstall with `uv tool install --force --python 3.12 'akasha-mcp[vectors]'`")
        if args.dry_run:
            print("[dry-run] would write config.toml; nothing created")
        else:
            # Private from creation: a write-then-chmod leaves a readable window.
            with os.fdopen(os.open(cfg_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
                           "w") as handle:
                handle.write(config_from_discovery(roots)
                             + ('[embeddings]\nprovider = "model2vec"\n' if dense else ""))
    if args.dry_run:
        print(f"[dry-run] {home} would be initialised")
    else:
        os.chmod(cfg_path, 0o600)
        load_config(cfg_path).knowledge_dir.mkdir(parents=True, exist_ok=True)
        print(f"initialised {home}")

    found = install.detect_vendors()
    if not args.all_vendors:
        if found:
            print(f"  detected: {', '.join(found)}; re-run with --all to register akasha")
        return 0
    for vendor, message in install.register_all(found, dry_run=args.dry_run).items():
        print(f"  {vendor}: {message}")
    for vendor in found:
        if vendor in install.hook_vendors():
            print(f"  {vendor} hooks: {install.install_hooks(vendor, dry_run=args.dry_run)}")
    if "claude" in found:
        # Registering the server is not enough: its tools still need a permission, or they
        # prompt interactively and are denied when unattended.
        print(f"  claude permissions: "
              f"{install.allow_akasha_tools(dry_run=args.dry_run, no_allow=args.no_allow)}")
    return 0


def cmd_doctor(args) -> int:
    from akasha.doctor import check, exit_code

    conn, cfg = _ctx()
    findings = check(cfg, conn)
    for f in findings:
        marker = "" if f.severity == "info" else f"{f.severity.upper()}: "
        print(f"{f.kind:12} {marker}{f.detail}")
    return exit_code(findings)


def cmd_serve(args) -> int:
    from akasha.mcp_server import serve

    serve()
    return 0


def cmd_index(args) -> int:
    conn, cfg = _ctx()
    if args.forget:
        print(f"forgot {forget(conn, Path(args.forget).expanduser().resolve())} document(s)")
        return 0
    stats = index_all(conn, cfg, force=args.force)
    print(f"indexed {stats.indexed}, skipped {stats.skipped}, denied {stats.denied}, "
          f"pruned {stats.pruned}")
    if stats.kind_fallback:
        print(f"kind fallback {stats.kind_fallback}: unknown kind, stored as reference")
    for _path, message in stats.collisions:
        print(f"not indexed: {message}")
    for name, message in stats.dataset_errors:
        print(f"dataset error: {name}: {message}")
    if stats.vectors:
        print(f"vectors {stats.vectors:,}")
    if stats.redacted:
        print(f"redacted: {', '.join(stats.redacted)}")
    for path, rules in stats.redacted_files:
        print(f"  redacted [{', '.join(rules)}] {path}")
    # Keeps the count the session brief reads current: a full check, none of it listed.
    fsck.report(conn, cfg, limit=0)
    conn.commit()
    return 0


def cmd_index_add(args) -> int:
    from akasha.register import add_index

    print(add_index(_config_path(), args.path, source=args.source,
                    include=args.include, exclude=args.exclude, repo=args.repo))
    print("  run `akasha index` to pull it in")
    return 0


def _repo_feature(cfg, args, conn):
    """CLI flags win; otherwise derive from the current checkout. A branch-derived feature
    applies only when that feature exists: branch names are not features, and filtering
    every query to an empty one would silently return nothing."""
    repo, _source = resolve_repo(cfg, Path.cwd(), args.repo)
    feature = args.feature
    if feature is None:
        _, derived = git_context(cfg, Path.cwd())
        if derived and resolve_feature(conn, derived, create=False):
            feature = derived
    return repo, feature


def _iso_date(text: str) -> str:
    try:
        return iso_date(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an ISO date (YYYY-MM-DD)")


def _capped(hit, cap: int | None) -> dict:
    if cap is None or len(hit.text) <= cap:
        return hit.as_dict()
    return dataclasses.replace(hit, text=hit.text[:cap]).as_dict(len(hit.text) - cap)


def cmd_search(args) -> int:
    conn, cfg = _ctx()
    repo, feature = _repo_feature(cfg, args, conn)
    hits = do_search(conn, args.query, cfg=cfg, repo=repo, feature=feature,
                     source=args.source, kind=args.kind, limit=args.limit,
                     include_archived=args.include_archived, all_repos=args.all,
                     match_mode=args.match_mode, as_of=args.as_of)
    cap = args.max_chars
    if args.json:
        print(json.dumps([_capped(h, cap) for h in hits], indent=2))
        return 0
    for h in hits:
        marker = ("~" if h.relaxed else "") + ("+" if h.cross_repo else "")
        print(f"{marker}[{h.score:.3g}] {h.repo or '-'}/{h.feature or '-'} {h.heading}")
        print(f"    {h.path}")
        # Bounds a terminal's scrollback; --json and the MCP tool carry the full text.
        print(f"    {h.text[:400 if cap is None else cap]}")
    return 0


def cmd_kb_get(args) -> int:
    conn, cfg = _ctx()
    text = Path(kb.get(conn, args.id)["path"]).read_text()
    if cfg.scan_secrets:
        from akasha.security import redact

        text, _ = redact(text)
    window = text[args.offset:]
    if args.max_chars is not None and len(window) > args.max_chars:
        print(f"{len(window) - args.max_chars} chars withheld; continue with --offset "
              f"{args.offset + args.max_chars}", file=sys.stderr)
        window = window[:args.max_chars]
    print(window)
    return 0


def _warn_convention_over_budget(conn, doc_id: str) -> None:
    """A convention reaches every session cut from the bottom; the author is told at the
    command that made it too long."""
    from akasha.hooks import conventions_over_budget

    for over in conventions_over_budget(conn):
        if over["id"] == doc_id:
            print(f"WARNING: convention '{over['title']}' is {over['length']} chars, over "
                  f"the {over['budget']}-char budget. It is cut from the bottom of every "
                  "session brief; split it, or move the rules that must reach every "
                  "session to the top.", file=sys.stderr)


def _body_arg(args) -> str | None:
    """Body text from --body-file, --body - (stdin) or --body; None when none was given."""
    if args.body_file:
        return Path(args.body_file).read_text()
    if args.body == "-":
        return sys.stdin.read()
    return args.body or None


def cmd_kb_write(args) -> int:
    conn, cfg = _ctx()
    body = _body_arg(args)
    if body is None:
        print("--body, --body-file, or --body - (stdin) is required")
        return 1
    repo, _source = resolve_repo(cfg, Path.cwd(), args.repo)
    doc_id = kb.write(conn, cfg, args.title, body, repo=repo, feature=args.feature,
                      kind=args.kind, supersedes=args.supersedes, new_kind=args.new_kind)
    print(doc_id)
    if (args.kind or "").lower() == "convention":
        _warn_convention_over_budget(conn, doc_id)
    return 0


def _after_edit(conn, doc_id: str) -> None:
    print(doc_id)
    row = conn.execute("SELECT kind FROM documents WHERE id = ?", (doc_id,)).fetchone()
    if row and row["kind"] == "convention":
        _warn_convention_over_budget(conn, doc_id)


def cmd_kb_update(args) -> int:
    conn, cfg = _ctx()
    kb.update(conn, cfg, args.id, expected_updated=args.expected_updated, body=args.body,
              match=args.match, replacement=args.replacement, title=args.title,
              kind=args.kind, status=args.status)
    _after_edit(conn, args.id)
    return 0


def cmd_kb_append(args) -> int:
    conn, cfg = _ctx()
    text = _body_arg(args)
    if text is None:
        print("--body, --body-file, or --body - (stdin) is required")
        return 1
    kb.append(conn, cfg, args.id, text, heading=args.heading)
    _after_edit(conn, args.id)
    return 0


def cmd_kb_archive(args) -> int:
    conn, cfg = _ctx()
    print(kb.archive(conn, cfg, args.id))
    return 0


def cmd_kb_rm(args) -> int:
    conn, cfg = _ctx()
    trashed = kb.remove(conn, cfg, args.id)
    print(f"moved to {trashed}" if trashed.exists() else
          f"{args.id} cleared from the index; its file was already gone")
    return 0


def cmd_kb_restore(args) -> int:
    conn, cfg = _ctx()
    print(f"restored to {kb.restore(conn, cfg, args.id)}")
    return 0


def cmd_kb_purge(args) -> int:
    conn, cfg = _ctx()
    if not args.yes:
        print("purge permanently deletes trashed documents. Re-run with --yes to confirm.")
        return 1
    print(f"purged {kb.purge(conn, cfg, args.older_than)} document(s)")
    return 0


def cmd_kb_stale(args) -> int:
    conn, cfg = _ctx()
    days = args.older_than or cfg.knowledge_stale_after_days or 180
    listed = kb.stale(conn, older_than_days=days, never_accessed_only=args.never_accessed)
    for d in listed:
        print(f"  {d['created_at'][:10]}  a={d['access_count']:3d}  {d['path']}")
    print(f"{len(listed)} stale document(s) (no read in {days} days)")
    return 0


def cmd_kb_stats(args) -> int:
    conn, _ = _ctx()
    data = kb.stats(conn, days=args.days)
    if data["window_truncated"]:
        print(f"note: events are purged at {kb.EVENT_RETENTION_DAYS} days; --days "
              f"{data['days']} reports on a truncated window\n")
    zero = data["zero_hit"]
    if zero["total"]:
        print(f"zero-hit rate: {zero['count'] / zero['total'] * 100:.1f}%  "
              f"({zero['count']}/{zero['total']} searches, last {data['days']}d)")
    else:
        print(f"zero-hit rate: no searches in the last {data['days']}d")
    print(f"successful reads: {data['reads']['count']} (last {data['days']}d)")
    if data["reads_since"] is None:
        print("note: read tracking has no events yet; never-opened counts are unavailable")
    else:
        print(f"note: never-opened counts searches since {data['reads_since'][:10]}")
    print(f"\nsurfaced but never opened: {data['surfaced_never_opened_total']}")
    for row in data["surfaced_never_opened"]:
        print(f"  surfaced={row['surfaced']:<3d} {row['path']}")
    print(f"\nrepeated queries: {len(data['repeated_queries'])}")
    for row in data["repeated_queries"]:
        print(f"  {row['count']:3d}x  {row['query']}")
    failed = data["tool_failed"]
    print(f"\ntool failures: {failed['count']}")
    for row in failed["rows"]:
        print(f"  {row['count']:3d}x  {row['tool']}({row['phase']})  {row['recent_error']}")
    return 0


def cmd_kb_fsck(args) -> int:
    conn, cfg = _ctx()
    # Always the full run: it also records the error count the session brief reads.
    report = fsck.report(conn, cfg, limit=None)
    conn.commit()
    errors = any(f["severity"] == "error" for f in report["findings"])
    if args.json:
        print(json.dumps(report, indent=2))
    elif not report["findings"]:
        print("no problems found")
    else:
        for f in report["findings"]:
            print(f"  [{f['severity']}] {f['kind']}: {f['detail']}")
    return 1 if errors else 0


def cmd_kb_timeline(args) -> int:
    conn, cfg = _ctx()
    repo = resolve_repo(cfg, Path.cwd(), args.repo)[0] if args.repo else None
    entries = kb.timeline(conn, feature=args.feature, repo=repo, limit=args.limit)
    for e in entries:
        print(f"  {e['created_at'][:10]}  {e['kind']:10} {e['id']}  {e['title']}")
    if not entries:
        print("no documents found")
    return 0


def cmd_kb_related(args) -> int:
    conn, _ = _ctx()
    if args.depth == 1:
        for node in lk.related(conn, args.id):
            print(f"{node['direction']:9} {node['id']}  {node['title']}")
        return 0
    for node in lk.graph(conn, args.id, depth=args.depth):
        print(f"{node['id']}  {node['title']}")
    return 0


def cmd_feature_list(args) -> int:
    conn, _ = _ctx()
    for f in list_features(conn, repo=args.repo, status=args.status):
        print(f"{f['status']:9} {f['slug']:40} {f['repo'] or '-'}")
    return 0


def cmd_feature_show(args) -> int:
    conn, _ = _ctx()
    info = feature_show(conn, args.slug)
    print(info["slug"])
    print(f"  docs: {info['documents']}")
    return 0


def cmd_feature_alias(args) -> int:
    conn, _ = _ctx()
    add_alias(conn, args.slug, args.alias)
    conn.commit()
    print(f"{args.alias} -> {args.slug}")
    return 0


def cmd_hook(args) -> int:
    from akasha import hooks

    # A hook must never block the agent's session, whatever the payload or the machine.
    try:
        if args.event == "post-tool":
            stdin_text = sys.stdin.read()
            hooks.run_post_tool(stdin_text)
            reply = hooks.post_tool_output(stdin_text)
            if reply:
                print(reply)
            return 0
        code, text = hooks.run_session_start(
            Path.cwd(), repo=args.repo, defer_refresh=args.defer_refresh,
            refresh_only=args.refresh_only)
        if text:
            print(text)
        return code
    except Exception:                                          # noqa: BLE001
        return 0


def _split_dotted(dotted: str) -> tuple[str, str]:
    """`section[.sub].key` as (section, key); the last part is always the key."""
    parts = _split_section(dotted)
    if len(parts) < 2:
        raise ValueError(
            f"'{dotted}' names no key: give section.key, e.g. security.scan_secrets")
    return ".".join(parts[:-1]), parts[-1]


def _parse_value(raw: str):
    """Only literals that cannot be misread: true/false and numbers. Everything else stays
    text, so a path is never reinterpreted."""
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    return raw


def _config_text() -> str:
    path = _config_path()
    return path.read_text() if path.exists() else ""


def cmd_config_get(args) -> int:
    text = _config_text()
    parts = _split_section(args.key)
    if len(parts) < 2:
        import tomllib

        node = tomllib.loads(text)
        for part in parts:
            node = node.get(part) if isinstance(node, dict) else None
        if not isinstance(node, dict):
            print(f"no section '{args.key}'")
            return 1
        for k, v in node.items():
            print(f"{args.key}.{k} = {v}")
        return 0
    section, key = _split_dotted(args.key)
    value = get_key(text, section, key)
    if value is None:
        print(f"{args.key} is not set")
        return 1
    if isinstance(value, dict):
        for k, v in value.items():
            print(f"{args.key}.{k} = {v}")
        return 0
    print(value)
    return 0


def cmd_config_set(args) -> int:
    section, key = _split_dotted(args.key)
    updated = set_key(_config_text(), section, key, _parse_value(args.value))
    write_config(_config_path(), updated)
    print(f"{args.key} = {get_key(updated, section, key)}")
    return 0


def cmd_config_unset(args) -> int:
    section, key = _split_dotted(args.key)
    text = _config_text()
    updated = unset_key(text, section, key)
    if updated == text:
        print(f"{args.key} was not set; nothing to remove")
        return 0
    write_config(_config_path(), updated)
    print(f"removed {args.key}")
    return 0


def cmd_housekeeping(args) -> int:
    conn, cfg = _ctx()
    counts = cleanup.run(conn) if args.now else cleanup.run_if_due(conn, cfg)
    if counts is None:
        cleaned = cleanup.last_run(conn)
        print(f"not due; last cleanup {cleaned}" if cleaned
              else "not due; retention has never run")
        return 0
    print(f"purged {counts['events']} events")
    return 0


def _header_row(text: str) -> int:
    """A 1-based row number: 0 or less would silently feed a row holding no headers."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"header_row must be 1 or more, got {value}")
    return value


def _selected_settings(args) -> FeedSettings | None:
    """The settings the setting flags describe, or None to reuse what the last feed stored."""
    given = (args.key, args.sheet, args.header_row, args.require, args.exclude, args.label)
    if all(value is None for value in given):
        return None
    # Any flag replaces the whole stored object: the flags not given take their defaults.
    return FeedSettings(key=args.key, sheet=args.sheet, header_row=args.header_row or 1,
                        require=args.require or [], exclude=args.exclude or [],
                        label=args.label)


def _scrub(cfg, text: str) -> str:
    """The scan policy, applied to output: rows fed while scanning was off sit raw in the
    table, so feed's lines have to be defused the way `knowledge get` defuses its text."""
    if not cfg.scan_secrets:
        return text
    from akasha.security import redact

    return redact(text)[0]


def _replaced_settings(cfg, name: str, settings: FeedSettings) -> tuple[list[str], bool]:
    """Which setting names the flags changed, and whether the stored ones were unreadable.

    A partial object replaces the rest, key included, which changes how every row is
    addressed, so neither a replace nor an unreadable predecessor may be silent. Unreadable
    still counts as none: `feed` reports the invalid dataset, and refusing here would block
    the repair re-feed that fixes it.
    """
    doc = feed_core._dataset_doc(cfg, name)
    if doc is None:
        return [], False
    try:
        stored = feed_core._settings_from(markdown.parse_frontmatter(doc.read_text())[0], name)
    except (ValueError, OSError):
        return [], True
    changed = [field.name for field in dataclasses.fields(FeedSettings)
               if getattr(stored, field.name) != getattr(settings, field.name)]
    return changed, False


def _identity(unit) -> str:
    return f"key:{unit.key}" if unit.key is not None else unit.locator


def _row_line(cfg, identity: str, label: str, fields) -> str:
    names = ", ".join(_scrub(cfg, name) for name in fields)
    return f"  {_scrub(cfg, identity)} {_scrub(cfg, label)}" + (f": {names}" if names else "")


def cmd_feed_add(args) -> int:
    conn, cfg = _ctx()
    name = args.as_name
    settings = _selected_settings(args)
    # The replace is reported from outside the lock on purpose: a race here can only make
    # the warning wrong, never the feed. The key rule needs the lock, so it lives in core.
    replaced, unreadable = (_replaced_settings(cfg, name, settings) if settings is not None
                            else ([], False))
    result = feed_core.feed(conn, cfg, Path(args.file).expanduser().resolve(), name,
                            settings, args.repo,
                            line_cap=0 if args.full else feed_core.CHANGE_LINE_CAP)
    for old_header, new_header in result.renamed:
        print(f"renamed: {_scrub(cfg, old_header)} -> {_scrub(cfg, new_header)}")
    if replaced:
        print(f"settings replaced: {', '.join(replaced)}")
    elif unreadable:
        print("stored settings were unreadable and were replaced")
    for warning in result.warnings:
        print(_scrub(cfg, warning))
    if result.redacted:
        print(f"redacted: {result.redacted}")
    if result.unchanged:
        print("unchanged")
        return 0
    diff = result.diff
    if diff is None:
        print(f"first feed: {result.units} units")
        return 0
    print(f"+{len(diff.added)} -{len(diff.removed)} ~{len(diff.changed)}")
    entries = ([(_identity(unit), unit.label, sorted(unit.fields), []) for unit in diff.added]
               + [(_identity(unit), unit.label, sorted(unit.fields), [])
                  for unit in diff.removed]
               + [(change.identity, change.label, change.fields, change.lines)
                  for change in diff.changed])
    shown = entries if args.full else entries[:feed_core.CHANGE_LINE_CAP]
    for identity, label, fields, lines in shown:
        print(_row_line(cfg, identity, label, fields))
        for line in lines:
            print(f"    {_scrub(cfg, line)}")
    if len(entries) > len(shown):
        print(f"  … {len(entries) - len(shown)} more (use --full)")
    return 0


def cmd_feed_list(args) -> int:
    conn, _cfg = _ctx()
    rows = conn.execute("SELECT path FROM documents WHERE kind = 'dataset'"
                        " AND deleted_at IS NULL ORDER BY path").fetchall()
    if not rows:
        print("no datasets")
        return 0
    for row in rows:
        path = Path(row["path"])
        meta, body = markdown.parse_frontmatter(path.read_text())
        name = meta.get("dataset") or path.stem
        units = conn.execute("SELECT COUNT(*) AS n FROM feed_units WHERE dataset = ?",
                             (name,)).fetchone()["n"]
        # The newest change-log stamp is the last feed's date; `updated_at` moves with
        # every index run and would not be a feed date. Split on the heading's own line,
        # so a field named `### x` cannot answer before the log does.
        fed = body.split("\n### ", 1)[1][:10] if "\n### " in body else "-"
        print(f"{name}  {meta.get('repo') or '-'}  {units} units  fed {fed}  "
              f"{meta.get('file')}")
    return 0


def _unit_row(cfg, unit) -> dict:
    """One row as it is printed: defused by the scan policy, with the locator as stored
    because `row:N` is the row, not export data."""
    # Two headers can redact to one placeholder; `clean_headers` numbers the later one the
    # way it numbers a duplicate column, where a dict keyed by the scrubbed name would
    # drop a whole field.
    names, _renamed = clean_headers([_scrub(cfg, field) for field in unit.fields])
    return {"locator": unit.locator,
            "key": _scrub(cfg, unit.key) if unit.key is not None else None,
            "label": _scrub(cfg, unit.label),
            "fields": {name: _scrub(cfg, value)
                       for name, value in zip(names, unit.fields.values())}}


def _print_unit(row: dict) -> None:
    print(f"locator: {row['locator']}")
    print(f"key: {row['key'] if row['key'] is not None else '-'}")
    print(f"label: {row['label']}")
    for field, value in row["fields"].items():
        print(f"{field}: {value}")


def cmd_feed_get(args) -> int:
    conn, cfg = _ctx()
    row = _unit_row(cfg, feed_core.get_unit(conn, args.name, args.locator))
    if args.json:
        print(json.dumps(row, indent=2))
        return 0
    _print_unit(row)
    return 0


def cmd_feed_query(args) -> int:
    conn, cfg = _ctx()
    fields = [name.strip() for name in args.fields.split(",")] if args.fields else None
    found = feed_core.query_units(conn, args.name, where=args.where or (),
                                  fields=fields, limit=args.limit)
    rows = [_unit_row(cfg, unit) for unit in found]
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    if not rows:
        print("no units")
    for index, row in enumerate(rows):
        if index:
            print()
        _print_unit(row)
    return 0


def cmd_feed_rm(args) -> int:
    conn, cfg = _ctx()
    doc_trash, copy_trash = feed_core.remove_dataset(conn, cfg, args.name)
    print(f"document: {doc_trash}")
    if copy_trash.exists():
        # A copy deleted behind akasha's back leaves nothing to trash, and naming a path
        # that was never written would read as a successful move.
        print(f"copy: {copy_trash}")
    # The trash name is `<doc id>__<name>.md`, as `restore` reads it.
    print(f"restore with: akasha knowledge restore {doc_trash.name.partition('__')[0]}")
    return 0


def _error_text(args, exc) -> str:
    """The line `main` prints for an expected failure.

    A feed error quotes export text — a stored field name, a column header — so it passes
    the same redaction the rest of that command's output does. Once, here, rather than at
    every place a feed command can fail.
    """
    message = _message(exc)
    if getattr(args, "feed_command", None) is None:
        return message
    try:
        cfg = load_config()
    except EXPECTED_ERRORS:
        # An unloadable config is one of these errors, and re-reading it here would raise
        # out of the handler that prints this line. An unreadable policy is unknown, not
        # off: a secret on the line matters more than a false positive.
        from akasha.security import redact

        return redact(message)[0]
    return _scrub(cfg, message)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="akasha",
        description="A local knowledge base for your coding agents.",
        epilog=(
            "examples:\n"
            "  akasha knowledge search \"cache expiry\" --limit 3\n"
            "  akasha knowledge write --title \"Findings\" --body-file notes.md --repo my-repo\n"
            "  akasha feature show my-feature\n"
            "  akasha doctor\n"
            "\n"
            "destructive, nothing else deletes data:\n"
            "  akasha knowledge rm ID     move a document to trash (recoverable)\n"
            "  akasha knowledge purge     permanently delete trashed documents\n"
            "  akasha feed rm NAME        move a dataset to trash (recoverable)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    p = sub.add_parser("init", help="create ~/.akasha; with --all, register with agent CLIs")
    p.add_argument("--all", dest="all_vendors", action="store_true",
                   help="register the server and hooks with every detected agent CLI")
    p.add_argument("--no-allow", action="store_true",
                   help="with --all, do not pre-approve akasha's tools in claude's settings")
    p.add_argument("--dry-run", action="store_true",
                   help="print what would be written without writing anything")
    p.add_argument("--code-root", action="append", default=[],
                   help="directory whose immediate subdirectories are scanned for repos with "
                        "serena memories and graphify output (repeatable, defaults to ~)")
    p.set_defaults(func=cmd_init)

    sub.add_parser("doctor", help="diagnose config, sources, registration and permissions"
                   ).set_defaults(func=cmd_doctor)
    sub.add_parser("serve", help="run the stdio MCP server; invoked by agents, not by you"
                   ).set_defaults(func=cmd_serve)

    p = sub.add_parser("housekeeping", help="delete events past their retention window (when due)")
    p.add_argument("--now", action="store_true", help="run the pass now, regardless of cadence")
    p.set_defaults(func=cmd_housekeeping)

    p_index = sub.add_parser("index", help="(re)build the search index from configured sources")
    p_index.add_argument("--force", action="store_true",
                         help="reprocess every file, including unchanged ones")
    p_index.add_argument("--forget", metavar="PATH",
                         help="forget one file's document; it returns on the next index unless its "
                              "root is removed from config")
    p_index.set_defaults(func=cmd_index)
    # Optional subcommand: bare `akasha index` keeps rebuilding.
    idx_sub = p_index.add_subparsers(dest="index_command", metavar="command")
    p = idx_sub.add_parser("add", help="register a directory of markdown as a source")
    p.add_argument("path", help="directory of markdown to index")
    p.add_argument("--source", help="source label; defaults to the directory name")
    p.add_argument("--repo", help="repo every document in this root belongs to")
    p.add_argument("--include", action="append", default=[],
                   help="only index files matching this glob (repeatable)")
    p.add_argument("--exclude", action="append", default=[],
                   help="skip files matching this glob (repeatable)")
    p.set_defaults(func=cmd_index_add)

    p_feed = sub.add_parser("feed", help="feed a .csv/.xlsx export as a searchable dataset")
    fd_sub = p_feed.add_subparsers(dest="feed_command", required=True, metavar="command")

    p = fd_sub.add_parser("add", help="feed a file; reports what changed since the last feed")
    p.add_argument("file", metavar="FILE", help="the .csv or .xlsx export to feed")
    p.add_argument("--as", dest="as_name", required=True, metavar="NAME",
                   help="dataset name; feeding it again replaces the stored rows")
    p.add_argument("--repo", help="repo the dataset belongs to; defaults to the current checkout")
    p.add_argument("--key", help="column whose value addresses a row across re-feeds")
    p.add_argument("--sheet", help="xlsx sheet name; defaults to the first sheet")
    p.add_argument("--header-row", dest="header_row", type=_header_row, metavar="N",
                   help="row holding the column headers (default 1)")
    p.add_argument("--require", action="extend", nargs="+", metavar="C",
                   help="keep only rows with a value in this column (repeatable)")
    p.add_argument("--exclude", action="extend", nargs="+", metavar="C",
                   help="drop this column from the stored rows (repeatable)")
    p.add_argument("--label", help="row label template, e.g. '{id} {city}'")
    p.add_argument("--full", action="store_true",
                   help="print every row and change line, not the first 50")
    p.set_defaults(func=cmd_feed_add)

    p = fd_sub.add_parser("list", help="the fed datasets, with repo, units, date and copy")
    p.set_defaults(func=cmd_feed_list)

    p = fd_sub.add_parser("get", help="print one row, addressed as `row:N` or `key:V`")
    p.add_argument("name", help="dataset the row belongs to")
    p.add_argument("locator", metavar="LOCATOR", help="the row to print, e.g. row:2 or key:bolt")
    p.add_argument("--json", action="store_true", help="print the row as JSON")
    p.set_defaults(func=cmd_feed_get)

    p = fd_sub.add_parser("query", help="filter rows; every --where item must match")
    p.add_argument("name", help="dataset to filter")
    p.add_argument("--where", action="extend", nargs="+", metavar="EXPR",
                   help="FIELD=VALUE (exact) or FIELD~VALUE (contains), case-insensitive; "
                        "repeatable, and all must match")
    p.add_argument("--fields", metavar="F,F", help="comma-separated fields to return")
    p.add_argument("--limit", type=int, default=20, help="how many rows to return (default 20)")
    p.add_argument("--json", action="store_true", help="print the rows as JSON")
    p.set_defaults(func=cmd_feed_query)

    p = fd_sub.add_parser("rm", help="move a dataset's document and copy to trash; recoverable")
    p.add_argument("name", help="dataset to remove")
    p.set_defaults(func=cmd_feed_rm)

    kb_sub = sub.add_parser("knowledge", help="search, write and maintain the knowledge base"
                            ).add_subparsers(dest="kb_command", required=True, metavar="command")

    p = kb_sub.add_parser("search", help="search indexed documents; retries as OR when strict finds nothing")
    p.add_argument("query", help="what to look for")
    p.add_argument("--repo", help="only results from this repo; defaults to the current checkout")
    p.add_argument("--feature", help="only results from this feature; defaults to the "
                                     "current branch when it names an existing one")
    p.add_argument("--source", help="only results from this source label")
    p.add_argument("--kind", help="only results of this document kind (reference, finding, plan, ...)")
    p.add_argument("--limit", type=int, default=5, help="how many hits to return (default 5)")
    p.add_argument("--include-archived", action="store_true",
                   help="include documents hidden from default search")
    p.add_argument("--as-of", type=_iso_date, metavar="DATE",
                   help="answer from documents valid on this ISO date (YYYY-MM-DD)")
    p.add_argument("--all", action="store_true", help="search every repo, not just this one")
    p.add_argument("--match", dest="match_mode", choices=["auto", "all", "any", "or", "strict"], default="auto",
                   help="auto retries as OR when strict AND finds nothing")
    p.add_argument("--max-chars", type=int, dest="max_chars",
                   help="cut each hit's text to this many characters (default 400 in the "
                        "listing, uncut with --json)")
    p.add_argument("--json", action="store_true", help="print hits as JSON")
    p.set_defaults(func=cmd_search)

    p = kb_sub.add_parser("get", help="print one document in full")
    p.add_argument("id", help="document to print")
    p.add_argument("--offset", type=int, default=0, help="first character to print (default 0)")
    p.add_argument("--max-chars", type=int, dest="max_chars",
                   help="print at most this many characters; the rest is reported on stderr")
    p.set_defaults(func=cmd_kb_get)

    p = kb_sub.add_parser("write", help="write a new document; use --supersedes when a conclusion changed")
    p.add_argument("--title", required=True, help="heading the document is stored under")
    p.add_argument("--body", help="markdown body; pass '-' to read it from stdin")
    p.add_argument("--body-file", dest="body_file", help="file whose contents become the body")
    p.add_argument("--repo", help="repo the document belongs to; defaults to the current checkout")
    p.add_argument("--feature", help="feature the document belongs to")
    p.add_argument("--kind", default="reference",
                   help="document kind: reference, finding, plan, ... (default reference)")
    p.add_argument("--supersedes", nargs="+", metavar="ID",
                   help="documents this one replaces; each is archived")
    p.add_argument("--new-kind", action="store_true",
                   help="register an unknown kind before writing")
    p.set_defaults(func=cmd_kb_write)

    p = kb_sub.add_parser("update", help="correct a document in place")
    p.add_argument("id", help="document to correct")
    p.add_argument("--body", help="replacement markdown body")
    p.add_argument("--title", help="replacement title")
    p.add_argument("--kind", help="replacement kind")
    p.add_argument("--status", help="replacement status, e.g. active or archived")
    p.add_argument("--match", help="one span of the body to change; must occur exactly once")
    p.add_argument("--replacement", help="text that replaces --match")
    p.add_argument("--expected-updated", dest="expected_updated",
                   help="refuse if the document changed since this updated date")
    p.set_defaults(func=cmd_kb_update)

    p = kb_sub.add_parser("append", help="append a dated section; cheap, cannot clobber")
    p.add_argument("id", help="document to extend")
    p.add_argument("--body", help="the section to add, dated with today; pass '-' to read it from stdin")
    p.add_argument("--body-file", dest="body_file", help="file whose contents become the section")
    p.add_argument("--heading", help="section heading, e.g. '## Result'; defaults to a dated one")
    p.set_defaults(func=cmd_kb_append)

    p = kb_sub.add_parser("archive", help="hide from default search; reversible")
    p.add_argument("id", help="document to hide from default search")
    p.set_defaults(func=cmd_kb_archive)

    p = kb_sub.add_parser("rm", help="move a document to trash; recoverable with restore")
    p.add_argument("id", help="document to move to trash")
    p.set_defaults(func=cmd_kb_rm)

    p = kb_sub.add_parser("restore", help="bring a trashed document back")
    p.add_argument("id", help="trashed document to bring back")
    p.set_defaults(func=cmd_kb_restore)

    p = kb_sub.add_parser("purge", help="permanently delete trashed documents; needs --yes")
    p.add_argument("--older-than", type=int, dest="older_than",
                   help="only purge documents trashed this many days ago; omit to purge all")
    p.add_argument("--yes", action="store_true", help="confirm: permanently deletes, no undo")
    p.set_defaults(func=cmd_kb_purge)

    p = kb_sub.add_parser("stale", help="documents nobody has read; reports only, deletes nothing")
    p.add_argument("--older-than", type=int, dest="older_than",
                   help="age in days past which a document is stale "
                        "(default: [knowledge] stale_after_days, else 180)")
    p.add_argument("--never-accessed", action="store_true",
                   help="only documents that have never been read")
    p.set_defaults(func=cmd_kb_stale)

    p = kb_sub.add_parser("stats", help="retrieval quality over a window; reports only. Events, including "
                   "redacted query heads, are kept 90 days locally")
    p.add_argument("--days", type=int, default=30,
                   help="window in days (default 30); events are kept for 90")
    p.set_defaults(func=cmd_kb_stats)

    p = kb_sub.add_parser("fsck", help="integrity check; reports only")
    p.add_argument("--json", action="store_true", help="print the report as JSON")
    p.set_defaults(func=cmd_kb_fsck)

    p = kb_sub.add_parser("timeline", help="documents in the order written")
    p.add_argument("--feature", help="only documents of this feature")
    p.add_argument("--repo", help="only documents of this repo")
    p.add_argument("--limit", type=int, default=50, help="how many to list (default 50)")
    p.set_defaults(func=cmd_kb_timeline)

    p = kb_sub.add_parser("related", help="neighbouring documents, including backlinks")
    p.add_argument("id", help="document whose neighbours to list")
    p.add_argument("--depth", type=int, default=1,
                   help="how many hops to walk; 1 lists direct neighbours (default 1)")
    p.set_defaults(func=cmd_kb_related)

    f_sub = sub.add_parser("feature", help="group documents under a feature tag"
                           ).add_subparsers(dest="feature_command", required=True, metavar="command")
    p = f_sub.add_parser("list", help="list all features")
    p.add_argument("--repo", help="only features of this repo")
    p.add_argument("--status", help="only features with this status")
    p.set_defaults(func=cmd_feature_list)
    p = f_sub.add_parser("show", help="how many documents carry a feature")
    p.add_argument("slug", help="feature to summarise")
    p.set_defaults(func=cmd_feature_show)
    p = f_sub.add_parser("alias", help="point an alias at a feature so shorthand resolves to it")
    p.add_argument("slug", help="feature the alias should resolve to")
    p.add_argument("alias", help="shorthand that resolves to it")
    p.set_defaults(func=cmd_feature_alias)

    p = sub.add_parser("hook", help="agent hook entrypoint; post-tool reads a JSON payload on stdin")
    p.add_argument("event", choices=["session-start", "post-tool"],
                   help="session-start prints a brief; post-tool reindexes the file just written "
                        "and may print a write nudge and the gotchas scoped to that file")
    p.add_argument("--repo", help="session-start: repo the session is in; derived from the checkout if unset")
    p.add_argument("--defer-refresh", action="store_true",
                   help="session-start: reply without waiting for the index walk")
    p.add_argument("--refresh-only", action="store_true",
                   help="session-start internal: refresh the index and exit")
    p.set_defaults(func=cmd_hook)

    cf_sub = sub.add_parser("config", help="read and edit config.toml by key"
                            ).add_subparsers(dest="config_command", required=True, metavar="command")
    p = cf_sub.add_parser("get", help="print one key, or a whole section")
    p.add_argument("key", help="section.key, or a section on its own")
    p.set_defaults(func=cmd_config_get)
    p = cf_sub.add_parser("set", help="set one key, creating its section if needed")
    p.add_argument("key", help="section.key, e.g. security.scan_secrets")
    p.add_argument("value", help="true/false, a number, or text")
    p.set_defaults(func=cmd_config_set)
    p = cf_sub.add_parser("unset", help="remove one key")
    p.add_argument("key", help="section.key")
    p.set_defaults(func=cmd_config_unset)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Python ignores SIGPIPE, so `akasha ... | head` would raise BrokenPipeError instead of
    # exiting quietly like other unix tools.
    try:
        import signal

        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (AttributeError, ValueError):
        pass                      # not POSIX, or not on the main thread
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except EXPECTED_ERRORS as exc:
        print(_error_text(args, exc))
        return 1
