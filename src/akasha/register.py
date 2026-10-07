"""Add an index root to config.toml without hand-editing TOML.

The block is appended and the result parsed before it replaces the original, the same
guarantee the config editor gives: a stray quote in a file the user also edits by hand
would break every other command.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

from akasha.config_edit import ConfigWriteError, write_config


def _toml_string(value: str) -> str:
    """A TOML basic string. Paths can contain quotes and backslashes."""
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _load(path: Path) -> tuple[str, dict]:
    text = Path(path).read_text() if Path(path).exists() else ""
    try:
        return text, tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigWriteError(f"{path} does not parse; fix it first: {exc}") from exc


def add_index(config_path: Path, root: Path | str, source: str | None = None,
              include: list[str] | None = None,
              exclude: list[str] | None = None, repo: str | None = None) -> str:
    """Register a directory of markdown as a knowledge source.

    A glob is stored verbatim: expand_roots resolves `*` at load time, so writing the
    expansion here would freeze today's matches into the config.
    """
    raw = str(root)
    if "*" in raw:
        stored = str(Path(raw).expanduser())
        probe = Path(stored.split("*")[0])
    else:
        stored = str(Path(raw).expanduser().resolve())
        probe = Path(stored)
    if not probe.exists():
        raise ConfigWriteError(f"{probe} does not exist")

    source = source or Path(stored.rstrip("/*")).name
    text, parsed = _load(config_path)
    if any(entry.get("path") == stored for entry in parsed.get("index", [])):
        return f"{stored} is already an index source"

    lines = ["", "[[index]]", f"path    = {_toml_string(stored)}",
             f"source  = {_toml_string(source)}"]
    if repo:
        lines.append(f"repo    = {_toml_string(repo)}")
    # include is omitted by default: a pattern like "**/*.md" needs a literal "/" under
    # fnmatch and so misses flat directories.
    if include:
        lines.append("include = [" + ", ".join(_toml_string(p) for p in include) + "]")
    if exclude:
        lines.append("exclude = [" + ", ".join(_toml_string(p) for p in exclude) + "]")

    write_config(config_path, text.rstrip() + "\n" + "\n".join(lines) + "\n")
    return f"added index source '{source}' at {stored}"
