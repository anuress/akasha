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


@dataclass
class Section:
    number: int
    heading: str
    text: str


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


def _starts_section(heading: str) -> bool:
    m = HEADING.match(heading)
    return bool(m) and len(m.group(1)) <= 2


def section_numbers(headings: list[str]) -> list[int]:
    """Number chunk headings in ord order: 0 before the first `#`/`##`, then 1 each."""
    numbers: list[int] = []
    number = 0
    prev = ""
    for heading in headings:
        # A ceiling-split chunk repeats its heading; only a change starts a section.
        if _starts_section(heading) and heading != prev:
            number += 1
        numbers.append(number)
        prev = heading
    return numbers


def sections(body: str) -> list[Section]:
    """Group a body into sections addressed by number, since headings repeat.

    # ponytail: two adjacent `#`/`##` headings with identical text merge into one section
    # and the second heading line is lost; upgrade if a document needs both.
    """
    chunks = chunk(body)
    numbers = section_numbers([c.heading for c in chunks])
    grouped: list[Section] = []
    prev = ""
    for c, number in zip(chunks, numbers):
        if not grouped or grouped[-1].number != number:
            heading = "" if number == 0 else c.heading
            grouped.append(Section(number=number, heading=heading, text=""))
        # A chunk leaves its heading out of body, so the first one puts it back.
        piece = f"{c.heading}\n\n{c.body}" if c.heading and c.heading != prev else c.body
        # Ceiling-split pieces are one run of text: joining them with "" reads the
        # section back without blank lines the file never had.
        joint = "\n\n" if grouped[-1].text and c.heading != prev else ""
        grouped[-1].text += joint + piece
        prev = c.heading
    return grouped


def sections_of(text: str) -> list[Section]:
    """Sections of a document as stored, so MCP and CLI number one file the same way."""
    return sections(strip_data_uris(parse_frontmatter(text)[1]))


def _size(chars: int) -> str:
    return f"{chars / 1000:.1f}k" if chars >= 1000 else str(chars)


def outline(secs: list[Section], cap: int) -> str:
    """One line per section, capped, ending with the count it left out."""
    lines = [f"{s.number} · {s.heading or '(opening)'} · {_size(len(s.text))}" for s in secs]
    if not lines:
        return ""
    last = secs[-1].number
    shown = 0
    for i in range(len(lines)):
        omitted = len(lines) - i - 1
        tail = [f"… {omitted} more sections, up to {last}"] if omitted else []
        # The truncation line is part of the budget, or the cap overflows by it.
        if len("\n".join(lines[:i + 1] + tail)) > cap:
            break
        shown = i + 1
    if shown == len(lines):
        return "\n".join(lines)
    left_out = len(lines) - shown
    return "\n".join(lines[:shown] + [f"… {left_out} more sections, up to {last}"])


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
