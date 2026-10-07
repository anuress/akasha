"""Installation and configuration diagnosis, the single source for `akasha doctor`.

The CLI and the MCP tool consume the same findings, so they cannot disagree about whether
the index is healthy. Severity is what the exit status does with a line:

  error   exits non-zero
  note    worth saying, not a red line
  info    the healthy state: a count, an "on"

There is deliberately no printer here; rendering is the caller's job.
"""
from __future__ import annotations

import sqlite3
import stat
import tomllib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from akasha import vectors
from akasha.config import Config
from akasha.db import SCHEMA_VERSION
from akasha.fsck import cached_error_count
from akasha.hooks import conventions_over_budget
from akasha.install import (INSTALLED_HOOK_COMMAND, ConfigError, detect_vendors,
                            hook_vendors, mcp_problem, mcp_registered,
                            post_tool_hook_command, session_hook_command)


@dataclass
class Finding:
    kind: str          # machine key: index_root, vectors, permissions, ...
    severity: str      # error | note | info
    detail: str        # what a caller can act on, in one line


def check_permissions(home: Path) -> list[str]:
    """Permission problems under the data home. Empty means healthy."""
    if not home.exists():
        return [f"{home} does not exist; run `akasha init`"]
    problems: list[str] = []
    if stat.S_IMODE(home.stat().st_mode) != 0o700:
        problems.append(f"{home} should be 0700")
    config = home / "config.toml"
    if config.exists() and stat.S_IMODE(config.stat().st_mode) != 0o600:
        problems.append(f"{config} should be 0600; it may hold credentials")
    return problems


def _sources(cfg: Config, conn: sqlite3.Connection) -> list[Finding]:
    findings = []
    for root in cfg.index_roots:
        path = Path(root.path).expanduser()
        # Count by path prefix, not source: several roots can share a source name. The
        # trailing "/" keeps /a/b from claiming /a/bc, and substr avoids LIKE wildcards
        # in a directory name.
        prefix = f"{path}/"
        count = conn.execute(
            "SELECT COUNT(*) c FROM documents WHERE deleted_at IS NULL"
            " AND (path = ? OR substr(path, 1, ?) = ?)",
            (str(path), len(prefix), prefix)).fetchone()["c"]
        # Three outcomes, not two: an empty root is normal and must not keep doctor red
        # forever, but files present with none indexed is the include-pattern fault and
        # stays loud. The presence test ignores include/exclude on purpose: sharing the
        # indexer's filter would let a wrong filter report "nothing to index" and pass.
        if not path.exists():
            state, severity = "does not exist", "error"
        elif count:
            state, severity = "", "info"
        elif any(p.suffix.lower() == ".md" and p.is_file() for p in path.rglob("*")):
            state, severity = "indexed nothing", "error"
        else:
            state, severity = "nothing to index yet", "note"
        findings.append(Finding("index_root", severity,
                                f"{root.source} {root.path}: {count} docs"
                                + (f" ({state})" if state else "")))
    return findings


def _vectors(cfg: Config, conn: sqlite3.Connection) -> Finding:
    """Say out loud when search is lexical only: the drop in quality has no other symptom."""
    if not vectors.enabled(cfg):
        return Finding("vectors", "note", 'off ([embeddings] provider = "none"); lexical only')
    missing = vectors.missing_modules()
    if missing:
        return Finding("vectors", "error",
                       f"extra not installed ({', '.join(missing)} missing); search is "
                       "lexical only. Reinstall with `uv tool install --force "
                       "'akasha-mcp[vectors]'`")
    if not vectors.available(conn):
        return Finding("vectors", "error", "sqlite-vec did not load; search is lexical only")
    # available() proves the extension loads, not that the encoder runs.
    failure = vectors.probe(conn)
    if failure:
        return Finding("vectors", "error", f"encoder failed: {failure}; search is lexical only")
    state = vectors.status(conn, cfg)
    count, stored, expected = state["count"], state["stored"], state["expected"]
    if not count:
        return Finding("vectors", "note", "on, 0 vectors; run `akasha index`")
    if stored != expected:
        return Finding("vectors", "note",
                       f"on, {count:,} vectors from {stored}, config wants {expected}; "
                       "run `akasha index`")
    return Finding("vectors", "info", f"on, {count:,} vectors ({stored})")


def _integrations(home: Path) -> list[Finding]:
    findings: list[Finding] = []
    registered: list[str] = []
    present = detect_vendors()
    for vendor in present:
        state = mcp_registered(vendor, home)
        if state is None:
            # Unreadable config: say so rather than claim the server is missing.
            findings.append(Finding("mcp", "error",
                                    f"{vendor}: {mcp_problem(vendor, home) or 'unchecked'}"))
        elif state:
            registered.append(vendor)
        else:
            findings.append(Finding("mcp", "note",
                                    f"{vendor}: akasha server not registered; run `akasha init`"))
    if registered:
        findings.append(Finding("mcp", "info", "registered: " + ", ".join(registered)))
    for vendor in present:
        if vendor not in hook_vendors():
            continue
        try:
            command = session_hook_command(vendor, home)
            post = post_tool_hook_command(vendor, home)
        except ConfigError as exc:
            findings.append(Finding("hook", "error", f"{vendor}: {exc}"))
            continue
        if command is None:
            findings.append(Finding("hook", "note",
                                    f"{vendor}: session-start hook not installed; run `akasha init`"))
        elif command != INSTALLED_HOOK_COMMAND:
            findings.append(Finding(
                "hook", "error",
                f"{vendor}: session-start hook lacks --defer-refresh, so every session waits "
                "on an index refresh; run `akasha init`"))
        if post is None:
            findings.append(Finding(
                "hook", "note",
                f"{vendor}: post-tool hook not installed, so edits are indexed only at the "
                "next session start; run `akasha init`"))
    return findings


def _config(config_path: Path) -> list[Finding]:
    if not config_path.exists():
        return [Finding("config", "note", f"{config_path} missing; defaults in use")]
    try:
        tomllib.loads(config_path.read_text())
    except (OSError, ValueError) as exc:
        return [Finding("config", "error", f"{config_path} cannot be read: {exc}")]
    return []


def _knowledge(conn: sqlite3.Connection) -> list[Finding]:
    docs = conn.execute(
        "SELECT COUNT(*) c FROM documents WHERE deleted_at IS NULL").fetchone()["c"]
    chunks = conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]
    return [Finding("knowledge", "info",
                    f"{docs} documents, {chunks:,} chunks, schema v{SCHEMA_VERSION}")]


def _conventions(conn: sqlite3.Connection) -> list[Finding]:
    # A convention over budget loses its tail in every session, and nothing else says so.
    return [Finding(
        "conventions", "error",
        f"{c['title']} ({c['repo'] or 'default'}): {c['length']} chars > "
        f"{c['budget']}-char budget; rules past the budget never reach a session")
        for c in conventions_over_budget(conn)]


def _permissions(data_home: Path) -> list[Finding]:
    return [Finding("permissions", "error", p) for p in check_permissions(data_home)]


def _fsck(conn: sqlite3.Connection) -> list[Finding]:
    errors = cached_error_count(conn)
    if errors is None:
        return [Finding("fsck", "note", "never run; `akasha knowledge fsck`")]
    if errors:
        return [Finding("fsck", "error",
                        f"{errors} error(s) at the last run; `akasha knowledge fsck`")]
    return []


def _failures(conn: sqlite3.Connection) -> list[Finding]:
    # A short window: doctor flags failures happening now; stats owns the history.
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    failed = conn.execute(
        "SELECT COUNT(*) c FROM events WHERE kind='tool.failed' AND ts >= ?",
        (cutoff,)).fetchone()["c"]
    if not failed:
        return []
    return [Finding("tool_failures", "note",
                    f"{failed} tool failure(s) in 7 days; `akasha knowledge stats`")]


def _isolated(kind: str, run) -> list[Finding]:
    """Run one check. A check that cannot run is a finding, not the end of the report: an
    aborted report would hide every check after it."""
    try:
        return run()
    except Exception as exc:                                             # noqa: BLE001
        return [Finding(kind, "error", f"unchecked: {type(exc).__name__}: {exc}")]


def check(cfg: Config, conn: sqlite3.Connection, home: Path | None = None,
          config_path: Path | None = None) -> list[Finding]:
    """Every check `akasha doctor` runs, as findings an agent can act on."""
    home = home or Path.home()
    data_home = home / ".akasha"
    config_path = config_path or data_home / "config.toml"
    checks = [
        ("config", lambda: _config(config_path)),
        ("index_root", lambda: _sources(cfg, conn)),
        ("knowledge", lambda: _knowledge(conn)),
        ("conventions", lambda: _conventions(conn)),
        ("vectors", lambda: [_vectors(cfg, conn)]),
        ("mcp", lambda: _integrations(home)),
        ("permissions", lambda: _permissions(data_home)),
        ("fsck", lambda: _fsck(conn)),
        ("tool_failures", lambda: _failures(conn)),
    ]
    findings: list[Finding] = []
    for kind, run in checks:
        findings += _isolated(kind, run)
    return findings


def exit_code(findings: list[Finding]) -> int:
    return 1 if any(f.severity == "error" for f in findings) else 0
