"""Markdown frontmatter and heading-based chunking."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

FENCE = re.compile(r"^\s*(```|~~~)")
HEADING = re.compile(r"^(#{1,6})\s+\S")

# A data URI is `data:[<mediatype>][;base64],<payload>`. The payload is one
# token — base64 has no whitespace, parens, angle brackets, quotes or backticks —
# so the mediatype runs to the first comma and the payload to the first terminator.
DATA_URI = re.compile(r"data:[^,\s)>\"'`]*,[^\s)>\"'`]*")
DATA_URI_PLACEHOLDER = "[image]"

# The largest chunk a heading section may span: long enough to keep sections whole, short
# enough that BM25 length normalisation and the embedding input truncation still reach
# the tail.
CHUNK_CEILING = 8961


def strip_data_uris(text: str) -> str:
    """Replace data URIs with a short placeholder so the sentence still reads.

    A base64 payload is worthless to both retrievers, yet it dilutes term statistics and
    eats the embedding truncation budget before the encoder reaches a real sentence. Only
    the URI is replaced; labels and alt text stay.
    """
    return DATA_URI.sub(DATA_URI_PLACEHOLDER, text)


@dataclass
class Chunk:
    heading: str
    body: str
    ord: int


def first_heading(body: str) -> str | None:
    """First H1 line in the body, or None. Files that share a filename still have distinct
    H1s, which `path.stem` would ignore."""
    for line in body.splitlines():
        m = re.match(r"^#\s+(.+)", line)
        if m:
            return m.group(1).strip()
    return None


def _unquote(value: str) -> str:
    """Drop quotes only when they are a matched pair around the whole value.

    Stripping any quote from either end truncates content that merely ends in one, such
    as a signature ending in `'lxml'`.
    """
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def _coerce(value: str):
    value = value.strip()
    # Structured values (objects, lists of objects) are stored as JSON, which render()
    # writes. The bracket-list branch below splits on every comma with no notion of
    # nesting, so it would corrupt them. Bare-word lists like `[k_1, k_2]` are not valid
    # JSON and fall through to it unchanged.
    if value[:1] in "[{":
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_unquote(v.strip()) for v in inner.split(",")]
    if value in ("null", "~", ""):
        return None
    if value in ("true", "false"):
        return value == "true"
    return _unquote(value)


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Return (metadata, body). Missing or malformed frontmatter yields ({}, text)."""
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines(keepends=True)
    end = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end = i
            break
    if end is None:
        return {}, text
    meta: dict = {}
    for line in lines[1:end]:
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = _coerce(value)
    return meta, "".join(lines[end + 1:]).lstrip("\n")


def _split_section(text: str, ceiling: int) -> list[str]:
    """Split a heading section over the ceiling on paragraph breaks.

    Greedy: paragraphs are packed up to the ceiling, and only a single paragraph over the
    limit is hard-cut. Pieces concatenate back to `text` in order, so ord still
    reconstructs the document.
    """
    if len(text) <= ceiling:
        return [text]
    # Keep the blank-line separators in the stream so no character is lost: parts are
    # paragraph, separator, paragraph, ... and each piece is a run of consecutive parts.
    parts = re.split(r"(\n\s*\n)", text)
    pieces: list[str] = []
    current = ""
    for part in parts:
        if not part:
            continue
        if current and len(current) + len(part) > ceiling:
            pieces.append(current)
            current = ""
        if len(part) > ceiling:
            pieces.extend(part[i:i + ceiling] for i in range(0, len(part), ceiling))
        else:
            current += part
    if current:
        pieces.append(current)
    return pieces


def chunk(body: str) -> list[Chunk]:
    """Split at markdown headings. Headings inside fenced code are not boundaries."""
    if not body.strip():
        return []
    chunks: list[Chunk] = []
    heading = ""
    buffer: list[str] = []
    in_fence = False

    def flush() -> None:
        # A heading with nothing under it carries no content but would still match via the
        # indexed title column, surfacing as a hit with a blank body.
        text = "".join(buffer).strip()
        if text:
            for piece in _split_section(text, CHUNK_CEILING):
                chunks.append(Chunk(heading=heading, body=piece, ord=len(chunks)))

    for line in body.splitlines(keepends=True):
        if FENCE.match(line):
            in_fence = not in_fence
            buffer.append(line)
            continue
        if not in_fence and HEADING.match(line):
            flush()
            heading = line.strip()
            buffer = []
            continue
        buffer.append(line)
    flush()
    return chunks


def render(meta: dict, body: str) -> str:
    """Serialise a document back to markdown with frontmatter."""
    lines = ["---"]
    for key, value in meta.items():
        if isinstance(value, dict) or (
                isinstance(value, list) and value and isinstance(value[0], dict)):
            # Real JSON: the bracket-list branch below joins with str(v), which on a dict
            # gives a Python repr that does not parse back.
            rendered = json.dumps(value)
        elif isinstance(value, list):
            rendered = "[" + ", ".join(str(v) for v in value) + "]"
        elif value is None:
            rendered = "null"
        elif isinstance(value, bool):
            rendered = "true" if value else "false"
        else:
            rendered = str(value)
        lines.append(f"{key}: {rendered}")
    lines.append("---")
    lines.append("")
    return "\n".join(lines) + body if body.startswith("\n") else "\n".join(lines) + "\n" + body
