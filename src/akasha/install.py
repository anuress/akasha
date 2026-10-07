"""Register akasha's MCP server and hooks with each installed agent CLI.

Everything that differs between agent CLIs lives in the VENDORS table; nothing else in
the package names one. A vendor's own CLI registers the server wherever it has an add
command, so its config format stays its own. Settings files are merged, never replaced,
and written atomically: they hold the user's other servers and rules. A file that does
not parse or has an unexpected shape is left untouched and named.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

HOOK_COMMAND = "akasha hook session-start"
# --defer-refresh: the host waits on this hook, so it must not wait on an index.
INSTALLED_HOOK_COMMAND = f"{HOOK_COMMAND} --defer-refresh"
POST_TOOL_COMMAND = "akasha hook post-tool"

# A vendor CLI that prompts or hangs must not stall `akasha init`.
CLI_TIMEOUT = 30

_FILE_ENTRY = {"command": "akasha", "args": ["serve"]}


@dataclass(frozen=True)
class Vendor:
    binary: str                        # found on PATH means installed
    mcp_file: tuple[str, ...]          # under HOME: where registration is checked, and
    mcp_key: str                       # written when mcp_argv is None; key holds servers
    mcp_argv: tuple[str, ...] | None = None   # the vendor's own add command
    mcp_entry: dict | None = None      # the server entry for a file edit
    hook_file: tuple[str, ...] | None = None  # under HOME: settings holding hooks
    session_event: str = "SessionStart"
    post_event: str = "PostToolUse"
    # Only tools that write a file: a process per Read or Bash call would cost more than
    # the reindex is worth. Tool names are the vendor's own.
    write_matcher: str = ""


# A vendor is added only after its registration and config shape are verified against a
# real install.
VENDORS: dict[str, Vendor] = {
    "claude": Vendor(
        binary="claude", mcp_file=(".claude.json",), mcp_key="mcpServers",
        mcp_argv=("claude", "mcp", "add", "--scope", "user", "akasha", "--", "akasha", "serve"),
        hook_file=(".claude", "settings.json"), write_matcher="Write|Edit|MultiEdit"),
    # --trust is scoped to this server: without it every tool call needs confirmation.
    "gemini": Vendor(
        binary="gemini", mcp_file=(".gemini", "settings.json"), mcp_key="mcpServers",
        mcp_argv=("gemini", "mcp", "add", "--scope", "user", "--transport", "stdio", "--trust",
                  "--description", "akasha knowledge base", "akasha", "akasha", "serve"),
        hook_file=(".gemini", "settings.json"), post_event="AfterTool",
        write_matcher="write_file|replace"),
    "copilot": Vendor(
        binary="copilot", mcp_file=(".copilot", "mcp-config.json"), mcp_key="mcpServers",
        mcp_entry=_FILE_ENTRY),
}


class ConfigError(Exception):
    """A config file akasha must not touch; the message says why."""


def detect_vendors() -> list[str]:
    """Which supported agent CLIs are on PATH."""
    return [name for name, v in VENDORS.items() if shutil.which(v.binary)]


def hook_vendors() -> list[str]:
    """Vendors with a verified hook schema. The rest get the MCP server only."""
    return [name for name, v in VENDORS.items() if v.hook_file]


def _home(home: Path | None) -> Path:
    return home or Path.home()


# --- reading and writing JSON config --------------------------------------------------

def _load(path: Path) -> dict:
    """The parsed object, {} when the file is absent. Anything else raises ConfigError."""
    try:
        text = path.read_text()
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        raise ConfigError(f"{path} is unreadable ({exc}); left untouched") from exc
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ConfigError(f"{path} is not valid JSON ({exc}); left untouched") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} has an unexpected shape (not an object); left untouched")
    return data


def _section(data: dict, path: Path, key: str, kind: type) -> object:
    """data[key], required to be a `kind` when present, so nothing is mutated blindly."""
    value = data.get(key, kind())
    if not isinstance(value, kind):
        raise ConfigError(f"{path} has an unexpected shape ({key} is not a "
                          f"{kind.__name__}); left untouched")
    return value


def _write_json(path: Path, data: dict) -> None:
    """Replace `path` atomically: a crash mid-write must not leave half a config."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = path.stat().st_mode & 0o7777
    except FileNotFoundError:
        mode = None
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(data, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _save(path: Path, data: dict, done: str) -> str:
    try:
        _write_json(path, data)
    except OSError as exc:
        return f"failed: could not write {path} ({exc}); left unchanged"
    return f"{done} {path}"


# --- MCP registration -----------------------------------------------------------------

def mcp_registered(vendor: str, home: Path | None = None) -> bool | None:
    """Whether the vendor's own config names the akasha server. None when that cannot be
    told: an unknown vendor, or a config that cannot be read."""
    spec = VENDORS.get(vendor)
    if spec is None:
        return None
    try:
        servers = _load(_home(home).joinpath(*spec.mcp_file)).get(spec.mcp_key, {})
    except ConfigError:
        return None
    return isinstance(servers, dict) and "akasha" in servers


def mcp_problem(vendor: str, home: Path | None = None) -> str | None:
    """Why the vendor's MCP config cannot be read, or None when it can."""
    try:
        _load(_home(home).joinpath(*VENDORS[vendor].mcp_file))
    except ConfigError as exc:
        return str(exc)
    return None


def _run_cli(vendor: str, argv: tuple[str, ...]) -> str:
    try:
        result = subprocess.run(list(argv), capture_output=True, text=True,
                                stdin=subprocess.DEVNULL, timeout=CLI_TIMEOUT)
    except subprocess.TimeoutExpired:
        return f"failed: {vendor} did not answer within {CLI_TIMEOUT}s (timed out)"
    except OSError as exc:
        return f"failed: could not run {vendor} ({exc})"
    if result.returncode != 0:
        return f"failed: {(result.stderr or '').strip() or f'exit {result.returncode}'}"
    return f"registered with {vendor}"


def _register_file(spec: Vendor, home: Path, dry_run: bool) -> str:
    path = home.joinpath(*spec.mcp_file)
    try:
        data = _load(path)
        servers = _section(data, path, spec.mcp_key, dict)
    except ConfigError as exc:
        return f"{exc}; add akasha manually"
    if servers.get("akasha") == spec.mcp_entry:
        return f"{path} already registered"
    if dry_run:
        return f"would update {path}"
    data[spec.mcp_key] = {**servers, "akasha": dict(spec.mcp_entry)}
    return _save(path, data, "updated")


def register(vendor: str, home: Path | None = None, dry_run: bool = False) -> str:
    """Register the akasha MCP server with one vendor."""
    spec = VENDORS[vendor]
    home = _home(home)
    if spec.mcp_argv is None:
        return _register_file(spec, home, dry_run)
    if mcp_registered(vendor, home):
        return f"{vendor} already registered"
    if dry_run:
        return f"would run: {' '.join(spec.mcp_argv)}"
    return _run_cli(vendor, spec.mcp_argv)


def register_all(vendors: list[str], dry_run: bool = False,
                 home: Path | None = None) -> dict[str, str]:
    return {v: register(v, home, dry_run) for v in vendors if v in VENDORS}


# --- hooks ----------------------------------------------------------------------------

def _is_ours(command: object, ours: str) -> bool:
    """akasha's own command: exactly `ours`, or `ours` followed by arguments. A longer
    command that merely starts with the same characters is someone else's."""
    return isinstance(command, str) and (command == ours or command.startswith(ours + " "))


def _ensure_hook(path: Path, hooks: dict, event: str, ours: str, command: str,
                 matcher: str) -> bool:
    """Make `event` run `command`, upgrading akasha's own entry in place. True if changed.
    Entries that are not ours are never touched."""
    entries = hooks.get(event, [])
    if not isinstance(entries, list):
        raise ConfigError(f"{path} has an unexpected shape (hooks.{event} is not a list); "
                          "left untouched")
    for entry in entries:
        if isinstance(entry, dict) and not isinstance(entry.get("hooks", []), list):
            raise ConfigError(f"{path} has an unexpected shape (a hooks.{event} entry's "
                              "hooks is not a list); left untouched")
    found = changed = False
    for entry in entries:
        inner = entry.get("hooks", []) if isinstance(entry, dict) else []
        mine = [h for h in inner if isinstance(h, dict) and _is_ours(h.get("command"), ours)]
        for hook in mine:
            found = True
            if hook["command"] != command:
                hook["command"] = command
                changed = True
        # A matcher shared with someone else's hook is theirs to choose.
        if mine and len(mine) == len(inner) and entry.get("matcher") != matcher:
            entry["matcher"] = matcher
            changed = True
    if not found:
        entries.append({"matcher": matcher, "hooks": [{"type": "command", "command": command}]})
        changed = True
    hooks[event] = entries
    return changed


def install_hooks(vendor: str, home: Path | None = None, dry_run: bool = False) -> str:
    """Write the session-start and post-tool hooks into the vendor's settings."""
    spec = VENDORS[vendor]
    if spec.hook_file is None:
        return f"{vendor} has no hook support; MCP server only"
    path = _home(home).joinpath(*spec.hook_file)
    try:
        data = _load(path)
        hooks = _section(data, path, "hooks", dict)
        changed = _ensure_hook(path, hooks, spec.session_event, HOOK_COMMAND,
                               INSTALLED_HOOK_COMMAND, "*")
        changed |= _ensure_hook(path, hooks, spec.post_event, POST_TOOL_COMMAND,
                                POST_TOOL_COMMAND, spec.write_matcher)
    except ConfigError as exc:
        return f"{exc}; add the hooks manually"
    if not changed:
        return f"{path} already has the akasha hooks"
    if dry_run:
        return f"would update {path}"
    data["hooks"] = hooks
    return _save(path, data, "updated")


def hook_command(vendor: str, event: str, home: Path | None = None) -> str | None:
    """The akasha command installed for `event` ("session" or "post") under `vendor`, or
    None when absent. Raises ConfigError when the settings cannot be read."""
    spec = VENDORS.get(vendor)
    if spec is None or spec.hook_file is None:
        return None
    path = _home(home).joinpath(*spec.hook_file)
    name = spec.session_event if event == "session" else spec.post_event
    entries = _load(path).get("hooks", {})
    entries = entries.get(name) if isinstance(entries, dict) else None
    for entry in entries if isinstance(entries, list) else []:
        for hook in entry.get("hooks", []) if isinstance(entry, dict) else []:
            command = hook.get("command", "") if isinstance(hook, dict) else ""
            if isinstance(command, str) and command.startswith("akasha hook"):
                return command
    return None


def session_hook_command(vendor: str, home: Path | None = None) -> str | None:
    return hook_command(vendor, "session", home)


def post_tool_hook_command(vendor: str, home: Path | None = None) -> str | None:
    return hook_command(vendor, "post", home)


# --- permission entry -----------------------------------------------------------------

# The narrowest string that covers akasha's tools and nothing else. Never a wildcard: this
# grants permission, so it must not reach past the server it is meant to cover.
ALLOW_ENTRY = "mcp__akasha"


def allow_akasha_tools(home: Path | None = None, dry_run: bool = False,
                       no_allow: bool = False) -> str:
    """Pre-approve akasha's MCP tools in claude's settings so calls do not prompt.

    Only ever adds the one entry. A deny or ask rule naming akasha wins, and any allow
    entry already naming akasha stands: a user who allowed one tool meant only that.
    """
    if no_allow:
        return "skipped the permission entry (--no-allow)"
    path = _home(home).joinpath(*VENDORS["claude"].hook_file)
    try:
        data = _load(path)
        permissions = _section(data, path, "permissions", dict)
        lists = {k: _section(permissions, path, k, list) for k in ("allow", "deny", "ask")}
    except ConfigError as exc:
        return f"{exc}; allow {ALLOW_ENTRY} manually"
    for kind in ("deny", "ask"):
        if any(ALLOW_ENTRY in str(rule) for rule in lists[kind]):
            return f"{ALLOW_ENTRY} is named in {kind} rules in {path}; leaving that decision alone"
    if any(str(rule).startswith(ALLOW_ENTRY) for rule in lists["allow"]):
        return f"{path} already allows {ALLOW_ENTRY}"
    if dry_run:
        return f"would allow {ALLOW_ENTRY} in {path}"
    permissions["allow"] = [*lists["allow"], ALLOW_ENTRY]
    data["permissions"] = permissions
    return _save(path, data, f"allowed {ALLOW_ENTRY} in")
