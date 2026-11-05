"""Splitters turn a table export into one Unit per kept row."""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, time
from pathlib import Path

import pytest

from akasha.splitters import FeedSettings, clean_headers, split


def _csv(tmp_path: Path, text: str, name: str = "data.csv") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _xlsx(tmp_path: Path, sheets: dict[str, list[list[object]]], name: str = "data.xlsx") -> Path:
    """Build a workbook from sheet title -> rows, so a test states only what it exercises."""
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    for index, (title, rows) in enumerate(sheets.items()):
        if index:
            ws = wb.create_sheet()
        ws.title = title
        for row in rows:
            ws.append(row)
    path = tmp_path / name
    wb.save(path)
    return path


def test_clean_headers_letters_empty_and_suffixes_case_insensitive_duplicates():
    """An unnamed column needs a stable name, and note/NOTE are one column to a caller."""
    headers, renamed = clean_headers(["Name", "", "note", "NOTE", "Multi\nline"])

    assert headers == ["Name", "col_B", "note", "NOTE (2)", "Multi line"]
    assert renamed == [("", "col_B"), ("NOTE", "NOTE (2)")]


def test_clean_headers_skips_suffix_taken_by_real_header():
    """Renaming a duplicate must not collide with a header that already carries that name."""
    headers, renamed = clean_headers(["NOTE", "NOTE (2)", "note"])

    assert headers == ["NOTE", "NOTE (2)", "note (3)"]
    assert renamed == [("note", "note (3)")]


def test_clean_headers_duplicate_suffix_does_not_rename_a_later_real_header():
    """A duplicate's suffix must skip a name that a real header further along still needs."""
    headers, _ = clean_headers(["note", "note", "NOTE (2)"])

    assert headers == ["note", "note (3)", "NOTE (2)"]


def test_column_letters_past_z_keep_counting():
    """Past column 26 the generated name must keep going, not restart at A."""
    headers, _ = clean_headers([""] * 27)

    assert headers[:2] == ["col_A", "col_B"]
    assert headers[26] == "col_AA"


def test_locator_is_record_number_with_multiline_cells(tmp_path):
    """A locator has to match the spreadsheet row number even when a cell spans lines."""
    path = _csv(
        tmp_path,
        'City,Note\nParis,"line one\nline two\nline three"\nLima,ok\n',
    )

    result = split(path, "places", FeedSettings())

    assert [u.locator for u in result.units] == ["row:2", "row:3"]
    assert result.units[0].fields["Note"] == "line one\nline two\nline three"


def test_header_row_skips_title_rows(tmp_path):
    """An export that starts with a title record must not turn that title into the header."""
    path = _csv(tmp_path, "Quarterly report,,\nCity,Country\nParis,France\n")

    result = split(path, "places", FeedSettings(header_row=2))

    assert result.headers == ["City", "Country"]
    assert [u.locator for u in result.units] == ["row:3"]


def test_header_row_past_last_record_raises(tmp_path):
    """An empty result would read as "the table now has 0 rows" and wipe stored units."""
    path = _csv(tmp_path, "City\nParis\n")

    with pytest.raises(ValueError) as exc:
        split(path, "places", FeedSettings(header_row=5))

    assert str(exc.value) == "data.csv: header_row 5 is past the last record (2 records)"


def test_empty_file_raises(tmp_path):
    """A truncated export must fail loudly rather than look like a table with no rows."""
    path = _csv(tmp_path, "")

    with pytest.raises(ValueError) as exc:
        split(path, "places", FeedSettings())

    assert str(exc.value) == "data.csv: header_row 1 is past the last record (0 records)"


def test_blank_header_record_raises(tmp_path):
    """Generated col_A.. names would hide the column names every caller addresses."""
    path = _csv(tmp_path, ",,\nParis,France,Lima\n")

    with pytest.raises(ValueError) as exc:
        split(path, "places", FeedSettings())

    assert str(exc.value) == "data.csv: header row 1 is empty"


def test_require_skips_rows_with_empty_required_column(tmp_path):
    """A unit without its required column is not addressable, so it must not be indexed."""
    path = _csv(tmp_path, "City,Country\nParis,France\nLima,\nBogota,Colombia\n")

    result = split(path, "places", FeedSettings(require=["Country"]))

    assert [u.locator for u in result.units] == ["row:2", "row:4"]


def test_require_naming_excluded_column_still_filters(tmp_path):
    """A required column can be excluded from fields and still decide what is kept."""
    path = _csv(tmp_path, "City,Country\nParis,France\nLima,\n")
    settings = FeedSettings(require=["Country"], exclude=["Country"])

    result = split(path, "places", settings)

    assert [u.locator for u in result.units] == ["row:2"]
    assert result.units[0].fields == {"City": "Paris"}


def test_short_row_missing_trailing_required_cell_counts_as_empty(tmp_path):
    """A short row must read as an empty required cell, not crash the filter."""
    path = _csv(tmp_path, "City,Country,Motto\nParis,France\nLima,Peru,Sunny\n")

    result = split(path, "places", FeedSettings(require=["Motto"]))

    assert [u.locator for u in result.units] == ["row:3"]


def test_row_whose_only_value_is_past_header_width_is_skipped(tmp_path):
    """A cell past the header width has no column, so it cannot make an empty row real."""
    path = _csv(tmp_path, "City,Country\n,,extra\nParis,France\n")

    result = split(path, "places", FeedSettings())

    assert [u.locator for u in result.units] == ["row:3"]


def test_whitespace_only_cells_are_left_out_of_fields(tmp_path):
    """A space-only cell carries nothing, so it must not become a searchable field."""
    path = _csv(tmp_path, "City,Country\nParis,   \n")

    result = split(path, "places", FeedSettings())

    assert result.units[0].fields == {"City": "Paris"}


def test_duplicate_keys_fall_back_to_row_locator_with_one_warning(tmp_path):
    """A key shared by two rows cannot address either, and one warning beats one per row."""
    path = _csv(
        tmp_path,
        "City,Code\nParis,P1\nLima,P1\nBogota,\nQuito,Q9\n",
    )

    result = split(path, "cities", FeedSettings(key="Code"))

    assert [u.key for u in result.units] == [None, None, None, "Q9"]
    assert [u.label for u in result.units] == [
        "cities row:2",
        "cities row:3",
        "cities row:4",
        "cities Q9",
    ]
    assert result.warnings == [
        "key 'Code': 3 rows have empty or duplicate values; addressed by row"
    ]


def test_exclude_drops_columns_from_fields(tmp_path):
    """An excluded column stays out of what callers read, but its header is still reported."""
    path = _csv(tmp_path, 'City,Internal,Country\n" Paris ",x,France\n')

    result = split(path, "places", FeedSettings(exclude=["Internal"]))

    assert result.headers == ["City", "Internal", "Country"]
    assert result.units[0].fields == {"City": " Paris ", "Country": "France"}


def test_key_in_exclude_column_is_still_computed(tmp_path):
    """Excluding a column hides it from fields; the key still has to address the row."""
    path = _csv(tmp_path, "City,Country\nParis,France\nLima,Peru\n")

    result = split(path, "places", FeedSettings(key="City", exclude=["City"]))

    assert [u.key for u in result.units] == ["Paris", "Lima"]
    assert all("City" not in u.fields for u in result.units)


@pytest.mark.parametrize(
    "settings",
    [
        FeedSettings(key="Missing"),
        FeedSettings(require=["Missing"]),
        FeedSettings(exclude=["Missing"]),
        FeedSettings(label="{Missing}"),
    ],
)
def test_unknown_key_require_exclude_or_label_column_lists_cleaned_headers(tmp_path, settings):
    """A column typo must fail loudly, naming the columns that do exist."""
    path = _csv(tmp_path, "City,Country\nParis,France\n")

    with pytest.raises(ValueError) as exc:
        split(path, "places", settings)

    assert str(exc.value) == "unknown column 'Missing'; columns: City, Country"


def test_label_template_fills_columns_and_default_is_name_and_key_or_locator(tmp_path):
    """A label is what a search hit shows, so it needs a usable default and a fillable shape."""
    path = _csv(tmp_path, "City,Country\nParis,France\nLima,\n")

    templated = split(path, "places", FeedSettings(label="{Country}  /  {City}"))
    by_key = split(path, "places", FeedSettings(key="City"))
    no_key = split(path, "places", FeedSettings())

    assert [u.label for u in templated.units] == ["France / Paris", "/ Lima"]
    assert [u.label for u in by_key.units] == ["places Paris", "places Lima"]
    assert [u.label for u in no_key.units] == ["places row:2", "places row:3"]


def test_label_placeholder_naming_excluded_column_still_renders(tmp_path):
    """A label may name a column that is deliberately not a field."""
    path = _csv(tmp_path, "City,Country\nParis,France\n")
    settings = FeedSettings(label="{City} - {Country}", exclude=["Country"])

    result = split(path, "places", settings)

    assert result.units[0].label == "Paris - France"
    assert result.units[0].fields == {"City": "Paris"}


def test_csv_with_bom_reads_first_header_cleanly(tmp_path):
    """A BOM left in the first header would make it unaddressable as a column."""
    path = tmp_path / "data.csv"
    path.write_text("City,Country\nParis,France\n", encoding="utf-8-sig")

    result = split(path, "places", FeedSettings())

    assert result.headers == ["City", "Country"]
    assert result.units[0].fields == {"City": "Paris", "Country": "France"}


def test_non_utf8_csv_raises_clear_error(tmp_path):
    """A latin-1 export decodes to mojibake silently; the caller needs to be told to re-export."""
    path = tmp_path / "data.csv"
    path.write_bytes("City\nMontréal\n".encode("latin-1"))

    with pytest.raises(ValueError) as exc:
        split(path, "places", FeedSettings())

    assert str(exc.value) == "data.csv is not UTF-8; re-export as UTF-8 CSV"


def test_unsupported_extension_names_accepted_ones(tmp_path):
    """The error must say what is accepted, or the caller cannot pick a working format."""
    path = tmp_path / "data.txt"
    path.write_text("City\nParis\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        split(path, "places", FeedSettings())

    assert str(exc.value) == "unsupported file type '.txt'; accepted: .csv, .xlsx"


def test_settings_json_round_trip_and_sha_stable():
    """The sha identifies a feed across runs, so equal settings must hash equal."""
    settings = FeedSettings(
        key="City",
        header_row=2,
        require=["Country"],
        exclude=["Internal"],
        label="{City}",
    )
    compact = json.dumps(json.loads(settings.to_json()), sort_keys=True, separators=(",", ":"))
    reordered = FeedSettings.from_json(
        '{"label":"{City}","exclude":["Internal"],"require":["Country"],'
        '"header_row":2,"sheet":null,"key":"City"}'
    )

    assert settings.to_json() == compact
    assert FeedSettings.from_json(settings.to_json()) == settings
    assert reordered.sha256() == settings.sha256()
    assert len(settings.sha256()) == 64


def test_xlsx_splits_like_csv_with_worksheet_row_locators(tmp_path):
    """An xlsx export is the same table, so a caller must not see a different result shape."""
    path = _xlsx(
        tmp_path,
        {"Sheet1": [["City", "Country"], ["Paris", "France"], ["Lima", "Peru"]]},
    )

    result = split(path, "places", FeedSettings())

    assert result.headers == ["City", "Country"]
    assert [u.locator for u in result.units] == ["row:2", "row:3"]
    assert result.units[0] == (
        "row:2",
        None,
        "places row:2",
        {"City": "Paris", "Country": "France"},
    )


def test_xlsx_values_render_dates_ints_bools_and_keep_error_strings(tmp_path):
    """A cell value has to reach search as readable text, not as a repr of its Python type."""
    path = _xlsx(
        tmp_path,
        {
            "Sheet1": [
                ["Name", "Day", "Seen", "Qty", "Active", "Ref", "At"],
                [
                    "Paris",
                    date(2024, 3, 5),
                    datetime(2024, 3, 5, 14, 30),
                    42.0,
                    True,
                    "#REF!",
                    time(9, 30),
                ],
                ["Lima", None, None, 7, False, "", None],
            ]
        },
    )

    result = split(path, "places", FeedSettings())

    assert result.units[0].fields == {
        "Name": "Paris",
        "Day": "2024-03-05",
        "Seen": "2024-03-05T14:30:00",
        "Qty": "42",
        "Active": "TRUE",
        "Ref": "#REF!",
        "At": "09:30:00",
    }
    assert result.units[1].fields == {"Name": "Lima", "Qty": "7", "Active": "FALSE"}


def test_xlsx_uncached_formula_is_empty_not_None(tmp_path):
    """A formula saved without a cached value reads as None, and "None" text would be a lie."""
    path = _xlsx(tmp_path, {"Sheet1": [["Total", "Note"], ["=1+2", "ok"]]})

    result = split(path, "places", FeedSettings())

    assert result.units[0].fields == {"Note": "ok"}
    assert "None" not in result.units[0].label


def test_xlsx_sheet_setting_picks_tab_and_unknown_tab_lists_tabs(tmp_path):
    """A workbook holds many tables; picking the wrong tab silently must not be possible."""
    path = _xlsx(
        tmp_path,
        {"First": [["City"], ["Paris"]], "Second": [["Code"], ["P1"]]},
    )

    picked = split(path, "places", FeedSettings(sheet="Second"))
    with pytest.raises(ValueError) as exc:
        split(path, "places", FeedSettings(sheet="Nope"))

    assert picked.headers == ["Code"]
    assert picked.units[0].fields == {"Code": "P1"}
    assert str(exc.value) == "data.xlsx: no sheet 'Nope'; sheets: First, Second"


def test_xlsx_leading_empty_row_keeps_row_numbers(tmp_path):
    """Read-only mode can start at the first filled row, which would shift every locator."""
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws["A2"] = "City"
    ws["A3"] = "Paris"
    ws["A4"] = "Lima"
    path = tmp_path / "data.xlsx"
    wb.save(path)

    result = split(path, "places", FeedSettings(header_row=2))

    assert result.headers == ["City"]
    assert [u.locator for u in result.units] == ["row:3", "row:4"]


def test_xlsx_without_extra_raises_install_hint(tmp_path, monkeypatch):
    """A missing optional dependency must name the install command, not raise a traceback."""
    path = tmp_path / "data.xlsx"
    path.write_bytes(b"")
    monkeypatch.setitem(sys.modules, "openpyxl", None)

    with pytest.raises(ValueError) as exc:
        split(path, "places", FeedSettings())

    assert str(exc.value) == "xlsx needs the extra: uv tool install 'akasha-mcp[xlsx]'"
