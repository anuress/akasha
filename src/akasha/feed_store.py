"""Dataset units in SQLite, the chunks made from them, and a pure diff of two unit lists.

The units are what a rebuild re-splits the copy on disk into, and the diff is what `feed
add` tells the caller changed. Nothing here commits: a dataset swap must be one
transaction with the chunk rows the caller writes, so the caller owns it.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import sqlite3
from dataclasses import dataclass

from akasha import vectors
from akasha.splitters import Unit

__all__ = [
    "Change",
    "Diff",
    "diff",
    "drop_dataset",
    "load_units",
    "replace_units",
    "stored_state",
    "unit_chunk_id",
]


def _state_key(dataset: str) -> str:
    return f"feed_state:{dataset}"


def load_units(conn: sqlite3.Connection, dataset: str) -> list[Unit]:
    rows = conn.execute(
        "SELECT locator, key, label, fields FROM feed_units WHERE dataset = ? ORDER BY ord",
        (dataset,))
    return [Unit(locator=r["locator"], key=r["key"], label=r["label"],
                 fields=json.loads(r["fields"])) for r in rows]


def replace_units(conn: sqlite3.Connection, dataset: str, units: list[Unit],
                  state: str) -> None:
    conn.execute("DELETE FROM feed_units WHERE dataset = ?", (dataset,))
    conn.executemany(
        "INSERT INTO feed_units (dataset, locator, key, label, fields, ord)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        [(dataset, unit.locator, unit.key, unit.label,
          json.dumps(unit.fields, ensure_ascii=False), position)
         for position, unit in enumerate(units)])
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?)"
        " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (_state_key(dataset), state))


def stored_state(conn: sqlite3.Connection, dataset: str) -> str | None:
    row = conn.execute(
        "SELECT value FROM meta WHERE key = ?", (_state_key(dataset),)).fetchone()
    return row["value"] if row else None


def drop_dataset(conn: sqlite3.Connection, dataset: str) -> None:
    ids = [r["chunk_id"] for r in conn.execute(
        "SELECT chunk_id FROM feed_chunks WHERE dataset = ?", (dataset,))]
    # Vectors first, as index.forget does: they are keyed by chunk id, so the chunk rows
    # must still name them. An orphan would spend a slot in nearest()'s k and return
    # nothing.
    if vectors.available(conn):
        try:
            conn.execute(
                f"DELETE FROM {vectors.TABLE} WHERE chunk_id IN"
                " (SELECT chunk_id FROM feed_chunks WHERE dataset = ?)", (dataset,))
        except sqlite3.OperationalError:
            pass  # nothing embedded yet
    conn.executemany("DELETE FROM chunks WHERE id = ?", [(chunk_id,) for chunk_id in ids])
    conn.execute("DELETE FROM feed_chunks WHERE dataset = ?", (dataset,))
    conn.execute("DELETE FROM feed_units WHERE dataset = ?", (dataset,))
    conn.execute("DELETE FROM meta WHERE key = ?", (_state_key(dataset),))


def unit_chunk_id(dataset: str, locator: str, field: str, value: str) -> str:
    """Content-derived, so re-importing an unchanged cell keeps its chunk and its vector."""
    digest = hashlib.sha256(f"{dataset}\0{locator}\0{field}\0{value}".encode()).hexdigest()
    return "c_" + digest[:12]


@dataclass
class Change:
    identity: str
    label: str
    fields: list[str]
    lines: list[str]


@dataclass
class Diff:
    added: list[Unit]
    removed: list[Unit]
    changed: list[Change]

    def empty(self) -> bool:
        return not (self.added or self.removed or self.changed)


def _pair(old: list[Unit], new: list[Unit]) -> list[tuple[int, int, str]]:
    """(old index, new index, identity) matches, key first and locator as the fallback.

    The splitter guarantees a stored key is unique on its side, but only a key present on
    both sides can pair a row; a key that became a duplicate arrives as None and lands on
    the locator fallback.
    """
    pairs: list[tuple[int, int, str]] = []
    used_old: set[int] = set()
    used_new: set[int] = set()
    by_key: dict[str, list[int]] = {}
    for i, unit in enumerate(old):
        if unit.key is not None:
            by_key.setdefault(unit.key, []).append(i)
    for j, unit in enumerate(new):
        queue = by_key.get(unit.key) if unit.key is not None else None
        if queue:
            i = queue.pop(0)
            used_old.add(i)
            used_new.add(j)
            pairs.append((i, j, f"key:{unit.key}"))
    by_locator = {unit.locator: i for i, unit in enumerate(old) if i not in used_old}
    for j, unit in enumerate(new):
        i = by_locator.get(unit.locator) if j not in used_new else None
        if i is not None and i not in used_old:
            used_old.add(i)
            used_new.add(j)
            pairs.append((i, j, unit.locator))
    return pairs


def _change(old: Unit, new: Unit, identity: str, line_cap: int) -> Change | None:
    names = sorted(name for name in set(old.fields) | set(new.fields)
                   if old.fields.get(name) != new.fields.get(name))
    if not names and old.label == new.label:
        return None
    lines: list[str] = []
    for name in names:
        diff = difflib.unified_diff(
            (old.fields.get(name) or "").splitlines(),
            (new.fields.get(name) or "").splitlines(), lineterm="", n=0)
        # The two header lines are skipped by position, not by text: `unified_diff`
        # always yields them first when the values differ, and a cell line of exactly
        # `-- ` renders as `--- `, indistinguishable from the header by content. Hunk
        # headers can be skipped by text because every content line starts `+` or `-`.
        body = list(diff)[2:]
        lines.extend(f"{name}: {line}" for line in body if not line.startswith("@@"))
    if line_cap:
        lines = lines[:line_cap]
    return Change(identity=identity, label=new.label, fields=names, lines=lines)


def diff(old: list[Unit], new: list[Unit], line_cap: int = 20) -> Diff:
    """What changed between two imports. `line_cap` is 0 for no cap."""
    pairs = _pair(old, new)
    matched_old = {i for i, _, _ in pairs}
    matched_new = {j for _, j, _ in pairs}
    changes = [change for change in
               (_change(old[i], new[j], identity, line_cap)
                for i, j, identity in sorted(pairs, key=lambda pair: pair[1]))
               if change is not None]
    return Diff(
        added=[unit for j, unit in enumerate(new) if j not in matched_new],
        removed=[unit for i, unit in enumerate(old) if i not in matched_old],
        changed=changes)
