"""Targeted edits to config.toml, scoped to one section, preserving everything else.

Every config command routes through this one editor. Two properties matter more than the
mechanism:

**Scoping.** A key is matched inside its own section's line span, never across the file,
because the same key name (`enabled`, say) can appear under several sections.

**Both spellings of a name.** `[projects."alpha"]` and `[projects.alpha]` are the same
section, so section names are compared as parsed parts. An editor that matched only one
spelling would silently do nothing.

Regex over a TOML round-tripper is a deliberate trade: the shape needed is "assign a
scalar inside a named section", and every write is parsed before it replaces anything, so
a botched edit raises rather than lands. That last property is what makes the trade safe,
so do not remove it.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path


class ConfigWriteError(RuntimeError):
    """The edit would have produced a config that does not parse."""


_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")


def _split_section(section: str) -> tuple[str, ...]:
    """A dotted section name as its parts, with any quoting removed.

    `projects."alpha"`, `projects.alpha` and `[projects."alpha"]`'s inside all become
    `("projects", "alpha")`, so the caller may spell it either way and so may the file.
    """
    parts: list[str] = []
    current: list[str] = []
    quote: str | None = None
    for char in section.strip():
        if quote:
            if char == quote:
                quote = None
            else:
                current.append(char)
        elif char in "\"'":
            quote = char
        elif char == ".":
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    parts.append("".join(current).strip())
    return tuple(p for p in parts if p)


def _render_name(parts: tuple[str, ...]) -> str:
    """The parts as a header body, quoting only what TOML cannot take bare."""
    return ".".join(p if _BARE_KEY.fullmatch(p) else f'"{p}"' for p in parts)


def _render_value(value) -> str:
    if isinstance(value, bool):                    # before int: bool is an int
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_render_value(v) for v in value) + "]"
    if isinstance(value, dict):
        # An inline table stays one line, so it is a single value this editor can replace
        # rather than a section of its own.
        inner = ", ".join(
            f"{k if _BARE_KEY.fullmatch(str(k)) else _render_value(str(k))} = "
            f"{_render_value(v)}" for k, v in value.items())
        return "{ " + inner + " }"
    # json.dumps produces TOML's basic-string escaping, including embedded quotes and
    # backslashes.
    import json

    return json.dumps(str(value))


def _section_span(lines: list[str], parts: tuple[str, ...]) -> tuple[int, int] | None:
    """(header index, end index) for a section, or None when it is absent.

    End is the index of the next section header at column 0, or len(lines). Array-of-
    table headers (`[[index]]`) are section boundaries too, or a key would be inserted
    into whichever one followed.
    """
    start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not (stripped.startswith("[") and stripped.endswith("]")):
            continue
        if start is not None:
            return start, i
        body = stripped[1:-1]
        if body.startswith("[") and body.endswith("]"):    # [[array]] of tables
            body = body[1:-1]
            if _split_section(body) == parts:
                continue     # an array-of-tables is never a plain section
        if _split_section(body) == parts:
            start = i
    if start is None:
        return None
    return start, len(lines)


def _key_line(lines: list[str], span: tuple[int, int], key: str) -> int | None:
    pattern = re.compile(rf"^\s*(?:{re.escape(key)}|\"{re.escape(key)}\")\s*=")
    for i in range(span[0] + 1, span[1]):
        if pattern.match(lines[i]):
            return i
    return None


def _parsed_or_raise(text: str) -> dict:
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigWriteError(f"refusing an edit that would not parse: {exc}") from exc


def set_key(text: str, section: str, key: str, value) -> str:
    """Return `text` with `section`.`key` set to `value`.

    The section is created when absent; the key is inserted directly under the header
    when the section exists without it, so it lands in the section it names rather than
    at the end of the file.
    """
    parts = _split_section(section)
    if not parts:
        raise ConfigWriteError("no section given")
    # A key with a space is legal TOML once quoted, so it would be written happily and
    # read by nothing: a typo would become a silent no-op. Every real key is bare.
    if not _BARE_KEY.fullmatch(key):
        raise ConfigWriteError(
            f"'{key}' is not a valid config key — expected letters, digits, - or _")
    line = f"{key} = {_render_value(value)}"

    lines = text.splitlines()
    span = _section_span(lines, parts)
    if span is None:
        prefix = text if text.endswith("\n") or not text else text + "\n"
        joiner = "\n" if prefix.strip() else ""
        out = f"{prefix}{joiner}[{_render_name(parts)}]\n{line}\n"
    else:
        at = _key_line(lines, span, key)
        if at is not None:
            lines[at] = line
        else:
            lines.insert(span[0] + 1, line)
        out = "\n".join(lines) + "\n"

    _parsed_or_raise(out)
    return out


def unset_key(text: str, section: str, key: str) -> str:
    """Return `text` with `section`.`key` removed. Absent key or section: unchanged."""
    parts = _split_section(section)
    lines = text.splitlines()
    span = _section_span(lines, parts) if parts else None
    if span is None:
        return text
    at = _key_line(lines, span, key)
    if at is None:
        return text
    del lines[at]
    out = "\n".join(lines) + "\n"
    _parsed_or_raise(out)
    return out


def get_key(text: str, section: str, key: str):
    """The parsed value, or None when the section or key is absent.

    Read from the parsed document rather than the text: the value a caller acts on must
    be what TOML says it is, not what a regex thinks it sees.
    """
    node = _parsed_or_raise(text)
    for part in _split_section(section):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    if not isinstance(node, dict):
        return None
    return node.get(key)


def write_config(path: Path, text: str) -> None:
    """Parse before replacing. A config that does not load breaks every command."""
    _parsed_or_raise(text)
    Path(path).write_text(text)
