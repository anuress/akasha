"""Reading one dataset row by locator, and filtering rows by field.

Every test feeds a synthetic CSV through the real write path, so what it reads back is
what a feed actually stored, and the suite needs no real export.
"""
from __future__ import annotations

import pytest

from akasha import feed
from akasha.config import load_config
from akasha.db import connect
from akasha.splitters import FeedSettings

ROWS = "item,price\nbolt,4\nnut,2\n"
NOTED = "item,note\nbolt,Steel fastener\nnut,Brass fastener\nscrew,steel pin\n"


@pytest.fixture
def fed(tmp_path):
    """The dataset rows under test, fed through the same call `akasha feed` makes."""
    cfg = load_config(tmp_path / "config.toml")
    cfg.knowledge_dir = tmp_path / "knowledge"
    cfg.knowledge_dir.mkdir()
    cfg.index_roots = []
    conn = connect(tmp_path / "s.db")

    def write(rows=ROWS, name="prices"):
        path = tmp_path / f"{name}.csv"
        path.write_text(rows, encoding="utf-8")
        feed.feed(conn, cfg, path, name, FeedSettings(key="item"), repo="sample-repo")

    return conn, write


def test_get_unit_by_row_and_by_key(fed):
    """A locator is how a caller addresses a row it saw in a search hit, so both the row
    number and the stored key must reach the same unit, fields intact."""
    conn, write = fed
    write()

    by_row = feed.get_unit(conn, "prices", "row:2")
    by_key = feed.get_unit(conn, "prices", "key:bolt")

    assert by_row == by_key
    assert (by_row.locator, by_row.key) == ("row:2", "bolt")
    assert by_row.fields == {"item": "bolt", "price": "4"}
    assert feed.get_unit(conn, "prices", "row:3").fields["price"] == "2"


def test_get_unknown_locator_and_unknown_dataset_are_keyerrors(fed):
    """A locator that matches nothing and a dataset that was never fed are different
    mistakes, and the message has to say which one the caller made."""
    conn, write = fed
    write()

    with pytest.raises(KeyError, match=r"no unit row:9 in 'prices'"):
        feed.get_unit(conn, "prices", "row:9")
    with pytest.raises(KeyError, match=r"no unit key:tofu in 'prices'"):
        feed.get_unit(conn, "prices", "key:tofu")
    with pytest.raises(KeyError, match=r"no dataset 'absent'"):
        feed.get_unit(conn, "absent", "row:2")


def test_query_exact_and_contains_are_case_insensitive_and_anded(fed):
    """`=` means the whole cell and `~` means part of it, neither cares about case, and
    two items must both hold; a query that ORed them would return nearly everything."""
    conn, write = fed
    write(rows=NOTED)

    assert [u.key for u in feed.query_units(conn, "prices", ["item=bOlT"])] == ["bolt"]
    assert [u.key for u in feed.query_units(conn, "prices", ["note~FASTEN"])] == [
        "bolt", "nut"]
    assert [u.key for u in feed.query_units(conn, "prices", ["note~steel"])] == [
        "bolt", "screw"]
    assert [u.key for u in feed.query_units(
        conn, "prices", ["note~steel", "note~fasten"])] == ["bolt"]


def test_query_splits_on_first_operator(fed):
    """A cell may hold `=` or `~` itself, so an item splits once: `a=b=c` is field `a`,
    value `b=c`, not field `a=b`."""
    conn, write = fed
    write(rows="item,note\nbolt,b=c\nnut,a=b\n")

    assert [u.key for u in feed.query_units(conn, "prices", ["note=b=c"])] == ["bolt"]
    assert [u.key for u in feed.query_units(conn, "prices", ["note~b=c"])] == ["bolt"]
    assert [u.key for u in feed.query_units(conn, "prices", ["note=a=b"])] == ["nut"]


def test_query_unknown_field_or_missing_operator_lists_fields(fed):
    """The caller cannot see the columns from here, so a refusal has to name the fields
    that exist; an item with no operator fails the same way rather than matching none."""
    conn, write = fed
    write(rows=NOTED)

    with pytest.raises(ValueError, match=r"unknown field 'colour'; fields: item, note"):
        feed.query_units(conn, "prices", ["colour=red"])
    with pytest.raises(ValueError, match=r"fields: item, note"):
        feed.query_units(conn, "prices", ["item bolt"])
    with pytest.raises(ValueError, match=r"unknown field 'colour'; fields: item, note"):
        feed.query_units(conn, "prices", fields=["colour"])


def test_query_projects_fields_and_clamps_limit(fed):
    """A query result is context the caller spends, so the projection drops the columns
    nobody asked for and the limit stays inside the range the tool promises."""
    conn, write = fed
    write(rows="item,price\n" + "".join(f"part-{n},{n}\n" for n in range(150)))

    projected = feed.query_units(conn, "prices", ["item~part-1"], fields=["price"], limit=2)
    assert [u.fields for u in projected] == [{"price": "1"}, {"price": "10"}]

    assert len(feed.query_units(conn, "prices", limit=0)) == 1
    assert len(feed.query_units(conn, "prices", limit=1000)) == 100
    assert [u.key for u in feed.query_units(conn, "prices", limit=2)] == ["part-0", "part-1"]


def test_cell_with_separator_newlines_and_heading_round_trips_verbatim(fed):
    """The chunk body joins a label, ` · ` and the field name around the value, so a cell
    holding that separator, a newline or a heading line must still come back byte for
    byte: the chunk text is a rendering, the stored unit is the truth."""
    conn, write = fed
    cell = "a · b\n# heading\nc"
    write(rows='item,note\nbolt,"a · b\n# heading\nc"\n')

    assert feed.get_unit(conn, "prices", "row:2").fields["note"] == cell
