"""Split a table export into one Unit per kept row, ready for indexing."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterator, NamedTuple

# Headers reach placeholders verbatim and may contain `:`, `!` and parentheses, so
# `str.format` would misread them; only the placeholder braces themselves are syntax.
_PLACEHOLDER = re.compile(r"\{([^{}]+)\}")

_SPACES = re.compile(r"\s+")


@dataclass
class FeedSettings:
    key: str | None = None
    sheet: str | None = None
    header_row: int = 1
    require: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    label: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> FeedSettings:
        return cls(**json.loads(s))

    def sha256(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()


class Unit(NamedTuple):
    locator: str
    key: str | None
    label: str
    fields: dict[str, str]


@dataclass
class SplitResult:
    units: list[Unit]
    headers: list[str]
    renamed: list[tuple[str, str]]
    warnings: list[str]


def _letters(position: int) -> str:
    """Spreadsheet column letters for a 1-based position: A..Z, AA, AB, ..."""
    letters = ""
    while position:
        position, remainder = divmod(position - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return letters


def _clean(name: str) -> str:
    return name.replace("\r\n", " ").replace("\n", " ").strip()


def clean_headers(raw: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """Name every column, making unnamed and duplicate headers addressable."""
    cleaned = [_clean(name) for name in raw]
    bases = [name or f"col_{_letters(i)}" for i, name in enumerate(cleaned, start=1)]
    # A real header named `note (2)` claims that name before any duplicate reaches it, so
    # every base name counts as taken up front rather than only ones already visited.
    taken = {base.casefold() for base in bases}
    used: set[str] = set()
    headers: list[str] = []
    renamed: list[tuple[str, str]] = []
    for original, base, plain in zip(raw, bases, cleaned):
        name = base
        if name.casefold() in used:
            number = 2
            while f"{base} ({number})".casefold() in taken:
                number += 1
            name = f"{base} ({number})"
            taken.add(name.casefold())
        used.add(name.casefold())
        headers.append(name)
        if name != plain:
            renamed.append((original, name))
    return headers, renamed


def _rows(path: Path, settings: FeedSettings) -> Iterator[list[str]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as fh:
            yield from csv.reader(fh)
    except UnicodeDecodeError:
        raise ValueError(f"{path.name} is not UTF-8; re-export as UTF-8 CSV") from None


def _requested(settings: FeedSettings) -> list[str]:
    columns = [settings.key] if settings.key is not None else []
    columns += settings.require + settings.exclude
    if settings.label is not None:
        columns += _PLACEHOLDER.findall(settings.label)
    return columns


def split(path: Path, name: str, settings: FeedSettings) -> SplitResult:
    suffix = path.suffix.lower()
    if suffix not in SPLITTERS:
        raise ValueError(f"unsupported file type '{suffix}'; accepted: .csv, .xlsx")
    return SPLITTERS[suffix](path, name, settings)


def split_table(path: Path, name: str, settings: FeedSettings) -> SplitResult:
    records = enumerate(_rows(path, settings), start=1)
    header = next(((n, row) for n, row in records if n >= settings.header_row), None)
    if header is None:
        return SplitResult(units=[], headers=[], renamed=[], warnings=[])
    _, raw = header
    headers, renamed = clean_headers(raw)
    for column in _requested(settings):
        if column not in headers:
            raise ValueError(f"unknown column '{column}'; columns: {', '.join(headers)}")
    width = len(headers)
    require = [headers.index(column) for column in settings.require]
    exclude = set(settings.exclude)
    key_index = headers.index(settings.key) if settings.key is not None else None

    rows: list[tuple[int, list[str]]] = []
    for number, cells in records:
        cells = list(cells)[:width] + [""] * (width - len(cells))
        if all(not cell.strip() for cell in cells):
            continue
        if any(not cells[i].strip() for i in require):
            continue
        rows.append((number, cells))

    keys = [cells[key_index].strip() if key_index is not None else None for _, cells in rows]
    warnings: list[str] = []
    if key_index is not None:
        counts = Counter(key for key in keys if key)
        dropped = {i for i, key in enumerate(keys) if not key or counts[key] > 1}
        if dropped:
            warnings.append(
                f"key '{settings.key}': {len(dropped)} rows have empty or duplicate values;"
                " addressed by row"
            )
        keys = [None if i in dropped else key for i, key in enumerate(keys)]

    units: list[Unit] = []
    for (number, cells), key in zip(rows, keys):
        locator = f"row:{number}"
        if settings.label is not None:
            values = dict(zip(headers, cells))
            label = _SPACES.sub(" ", _PLACEHOLDER.sub(lambda m: values[m.group(1)], settings.label))
            label = label.strip()
        else:
            label = f"{name} {key or locator}"
        units.append(
            Unit(
                locator=locator,
                key=key,
                label=label,
                fields={
                    column: cell
                    for column, cell in zip(headers, cells)
                    if cell.strip() and column not in exclude
                },
            )
        )

    return SplitResult(units=units, headers=headers, renamed=renamed, warnings=warnings)


SPLITTERS: dict[str, Callable[[Path, str, FeedSettings], SplitResult]] = {".csv": split_table}
