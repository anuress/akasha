"""One section-scoped TOML editor, so every section is reachable by command.

The scoping is not a nicety: matching a key across the whole file edits whichever
section happens to come first when the same key name appears under several.
"""
from __future__ import annotations

import tomllib

import pytest

from akasha.config_edit import ConfigWriteError, get_key, set_key, unset_key


def _load(text: str) -> dict:
    return tomllib.loads(text)


TWO_ENABLED = """\
# a comment that must survive
[first]
enabled          = false
max_chars        = 900

[second]
enabled            = true
max_items     = 2
"""


def test_a_key_is_matched_inside_its_own_section_only():
    """A whole-file match would flip whichever `enabled` came first. [first] precedes
    [second] here, so that behaviour would be the wrong one."""
    out = set_key(TWO_ENABLED, "second", "enabled", False)
    parsed = _load(out)
    assert parsed["second"]["enabled"] is False
    assert parsed["first"]["enabled"] is False, "first was already false; it must not move"

    out2 = set_key(TWO_ENABLED, "first", "enabled", True)
    parsed2 = _load(out2)
    assert parsed2["first"]["enabled"] is True
    assert parsed2["second"]["enabled"] is True, "second was already true; it must not move"


def test_comments_and_neighbours_survive_an_edit():
    out = set_key(TWO_ENABLED, "second", "max_items", 4)
    assert "# a comment that must survive" in out
    assert _load(out)["second"]["max_items"] == 4
    assert _load(out)["first"]["max_chars"] == 900


def test_a_missing_key_is_inserted_into_the_section_that_exists():
    out = set_key(TWO_ENABLED, "first", "min_ratio", 0.5)
    assert _load(out)["first"]["min_ratio"] == 0.5
    # Inserted under [first], not appended to the end of the file where it would
    # land inside whichever section happens to be last.
    assert out.index("min_ratio") < out.index("[second]")


def test_a_missing_section_is_appended():
    out = set_key(TWO_ENABLED, "catalog", "refresh_min", 30)
    assert _load(out)["catalog"]["refresh_min"] == 30


def test_a_quoted_section_name_is_the_same_section_as_the_bare_one():
    """[projects."alpha"] and [projects.alpha] are one section. An editor that matched
    only one spelling would silently do nothing."""
    quoted = '[projects."alpha"]\npath = "/tmp/alpha"\n'
    out = set_key(quoted, "projects.alpha", "weight", 0.5)
    assert _load(out)["projects"]["alpha"]["weight"] == 0.5
    assert _load(out)["projects"]["alpha"]["path"] == "/tmp/alpha"

    bare = "[projects.beta]\npath = \"/tmp/beta\"\n"
    out2 = set_key(bare, 'projects."beta"', "weight", 0.5)
    assert _load(out2)["projects"]["beta"]["weight"] == 0.5


def test_a_name_that_needs_quoting_round_trips():
    """A colon is not valid in a bare key, so the name must be quoted."""
    text = '[sources."main:books"]\nfont = "x"\n'
    out = set_key(text, "sources.main:books", "mode", "write")
    assert _load(out)["sources"]["main:books"]["mode"] == "write"

    fresh = set_key("", "sources.main:serials", "mode", "write")
    assert _load(fresh)["sources"]["main:serials"]["mode"] == "write"


@pytest.mark.parametrize("value,expected", [
    (True, True), (False, False), (3, 3), (0.5, 0.5), ("text", "text"),
    (["a", "b"], ["a", "b"]),
])
def test_values_round_trip_by_type(value, expected):
    out = set_key("[second]\n", "second", "k", value)
    assert _load(out)["second"]["k"] == expected


def test_an_inline_table_round_trips_with_keys_that_need_quoting():
    """An inline table's keys may need quoting, as `main:serials` does."""
    options = {"strict": True, "main:serials": False}
    out = set_key("[second]\nenabled = true\n", "second", "options", options)
    assert _load(out)["second"]["options"] == options
    assert _load(out)["second"]["enabled"] is True


def test_a_string_with_quotes_does_not_break_the_file():
    out = set_key("[projects.a]\n", "projects.a", "cmd", 'python -c "import x"')
    assert _load(out)["projects"]["a"]["cmd"] == 'python -c "import x"'


def test_a_key_that_is_not_a_bare_identifier_is_refused():
    """Quoting would make `not a key` legal TOML, which is the trap: it would be written
    happily and read by nothing, so a typo becomes a silent no-op rather than an error."""
    with pytest.raises(ConfigWriteError) as exc:
        set_key("[second]\nenabled = true\n", "second", "not a key", 1)
    assert "not a key" in str(exc.value)


def test_an_edit_onto_a_file_that_does_not_parse_is_refused():
    """Parse-before-replace is what makes a regex editor safe to keep. Proven against
    text that is already broken, since the writer itself should never produce any."""
    broken = "[second\nenabled = true\n"          # unterminated header
    with pytest.raises(ConfigWriteError):
        set_key(broken, "second", "enabled", False)


def test_unset_removes_only_that_key():
    out = unset_key(TWO_ENABLED, "second", "enabled")
    parsed = _load(out)
    assert "enabled" not in parsed["second"]
    assert parsed["second"]["max_items"] == 2
    assert parsed["first"]["enabled"] is False, "the same key elsewhere must survive"


def test_unset_of_something_absent_changes_nothing():
    assert unset_key(TWO_ENABLED, "second", "nope") == TWO_ENABLED
    assert unset_key(TWO_ENABLED, "nosuch", "nope") == TWO_ENABLED


def test_get_key_reads_the_parsed_value_not_the_text():
    assert get_key(TWO_ENABLED, "second", "enabled") is True
    assert get_key(TWO_ENABLED, "first", "max_chars") == 900
    assert get_key(TWO_ENABLED, "second", "absent") is None
    assert get_key(TWO_ENABLED, "nosuch", "k") is None
