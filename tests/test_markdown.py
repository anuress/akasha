from akasha.markdown import CHUNK_CEILING, chunk, parse_frontmatter, render, strip_data_uris


def test_parse_frontmatter_extracts_scalars_and_lists():
    text = (
        "---\n"
        "id: k_8f3a2c\n"
        "title: Catalog ownership design\n"
        "feature: proj-100\n"
        "tags: [record, owner]\n"
        "supersedes: []\n"
        "---\n"
        "\n"
        "# Body\n"
        "text here\n"
    )
    meta, body = parse_frontmatter(text)
    assert meta["id"] == "k_8f3a2c"
    assert meta["title"] == "Catalog ownership design"
    assert meta["tags"] == ["record", "owner"]
    assert meta["supersedes"] == []
    assert body.startswith("# Body")


def test_parse_frontmatter_absent_returns_empty_meta():
    meta, body = parse_frontmatter("# Just a doc\n\nno frontmatter\n")
    assert meta == {}
    assert body.startswith("# Just a doc")


def test_chunk_splits_on_headings():
    body = (
        "intro text\n"
        "## Shelving\n"
        "LoanService is new\n"
        "## Exceptions\n"
        "oversize moves\n"
    )
    chunks = chunk(body)
    assert [c.heading for c in chunks] == ["", "## Shelving", "## Exceptions"]
    assert "LoanService" in chunks[1].body
    assert [c.ord for c in chunks] == [0, 1, 2]


def test_chunk_ignores_headings_inside_fenced_code():
    body = "## Real\n```\n## not a heading\n```\ntail\n"
    chunks = chunk(body)
    assert len(chunks) == 1
    assert "not a heading" in chunks[0].body


def test_chunk_empty_body_yields_nothing():
    assert chunk("   \n") == []


def test_headingless_body_over_ceiling_splits_and_reassembles_in_ord_order():
    """A chunk over the ceiling cannot be retrieved by either path, so the section must
    split on paragraph breaks, and the pieces must reassemble to the original text in
    ord order."""
    paragraphs = "\n\n".join(f"para {i}: " + "x" * 400 for i in range(30))
    chunks = chunk(paragraphs)
    assert len(chunks) > 1
    assert all(len(c.body) <= CHUNK_CEILING for c in chunks)
    assert [c.ord for c in chunks] == list(range(len(chunks)))
    assert "".join(c.body for c in chunks) == paragraphs


def test_ordinary_document_under_ceiling_is_not_split():
    """The ceiling must not move boundaries for a heading-led body under it."""
    body = (
        "intro text\n"
        "## Shelving\n"
        "LoanService is new\n"
        "## Exceptions\n"
        "oversize moves\n"
    )
    chunks = chunk(body)
    assert [c.heading for c in chunks] == ["", "## Shelving", "## Exceptions"]
    assert [c.ord for c in chunks] == [0, 1, 2]


def test_single_paragraph_over_ceiling_falls_back_to_hard_cut():
    """A paragraph with no internal blank line cannot split on paragraph breaks, so it
    falls back to a hard character cut — the piece must still stay under the ceiling
    and reassemble."""
    text = "z" * (CHUNK_CEILING * 2 + 100)
    chunks = chunk(text)
    assert len(chunks) > 1
    assert all(len(c.body) <= CHUNK_CEILING for c in chunks)
    assert "".join(c.body for c in chunks) == text


def test_structured_frontmatter_values_roundtrip_through_parse():
    """A dict, and a list of dicts, must survive a rewrite. The bracket-list branch joins
    with str(v), which on a dict gives a Python repr (single-quoted, not JSON), and then
    splits on every comma including the ones inside each object."""
    meta = {
        "id": "k_1", "title": "T",
        "custom_field": {"by": "alice", "at": "2026-01-01T00:00:00+00:00"},
        "reviewed_by": [
            {"by": "bob", "at": "2026-01-01T01:00:00+00:00"},
            {"by": "carol", "at": "2026-01-01T02:00:00+00:00"},
        ],
    }
    text = render(meta, "# Hi\n\nbody\n")
    back, _ = parse_frontmatter(text)
    assert back["custom_field"] == meta["custom_field"]
    assert back["reviewed_by"] == meta["reviewed_by"]


def test_a_bracket_list_of_bare_words_still_parses_as_before():
    """The JSON-first branch must not swallow the bare-word list format
    (`supersedes: [k_1, k_2]`): unquoted identifiers are not valid JSON, so json.loads
    must fail and fall through to the comma-split."""
    meta, _ = parse_frontmatter("---\nsupersedes: [k_1, k_2]\n---\n\nbody\n")
    assert meta["supersedes"] == ["k_1", "k_2"]


def test_render_roundtrips_through_parse():
    meta = {"id": "k_1", "title": "T", "tags": ["a", "b"]}
    text = render(meta, "# Hi\n\nbody\n")
    back, body = parse_frontmatter(text)
    assert back["id"] == "k_1"
    assert back["tags"] == ["a", "b"]
    assert body.strip() == "# Hi\n\nbody".strip()


def test_a_value_ending_in_a_quote_keeps_it():
    """Stripping any quote from either end silently truncates content: a signature ending
    in `'lxml'` would come back as `'lxml`. Only a matched pair is quoting; anything else
    is text."""
    meta, _ = parse_frontmatter(
        "---\nsignature: ModuleNotFoundError: No module named 'lxml'\n"
        "title: \"the quoted one\"\n---\n\nbody\n")
    assert meta["signature"] == "ModuleNotFoundError: No module named 'lxml'"
    assert meta["title"] == "the quoted one"


_URI = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ" \
    "AAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="


def test_strip_data_uris_replaces_the_payload_with_a_placeholder():
    """A data URI is a base64 blob that neither retriever can use, yet it dilutes term
    statistics and eats the embedding truncation budget. It must not reach the index, so
    it is replaced by a short placeholder and the surrounding prose stays."""
    body = f"The flow: ![][image1]\n\n[image1]: <{_URI}>\n"
    out = strip_data_uris(body)
    assert "iVBORw0KGgo" not in out
    assert "[image]" in out
    assert "The flow:" in out


def test_stripped_body_chunks_smaller_than_the_raw_body():
    """The payload would consume the whole 2,000-char truncation budget before the encoder
    reached a real sentence. After stripping, the chunk is the prose, and it is smaller
    than the raw body."""
    body = f"## Screenshot\nThe report page:\n![][image1]\n\n[image1]: <{_URI}>\n"
    chunks = chunk(strip_data_uris(body))
    assert len(chunks) == 1
    assert len(chunks[0].body) < len(body)
    assert "The report page:" in chunks[0].body
    assert "iVBORw0KGgo" not in chunks[0].body


def test_strip_data_uris_handles_inline_images():
    body = f"The result: ![result]({_URI})\n"
    out = strip_data_uris(body)
    assert "iVBORw0KGgo" not in out
    assert "[image]" in out


def test_strip_data_uris_leaves_a_clean_body_unchanged():
    """Stripping is a no-op for a document that never embedded an image, so nothing
    already indexed shifts. `:data:catalog:` is a label, not a URI, and must
    survive untouched."""
    body = "## Notes\nplain prose\nsee [[k_1]] and :data:catalog:export\n"
    assert strip_data_uris(body) == body


def test_strip_data_uris_does_not_eat_the_backtick_after_a_mention():
    """A prose mention of the scheme, 'Strips `data:application/pdf;base64,` prefix', has
    its closing backtick flush against the comma. The payload class must not include
    backticks, or the sentence loses its code-span marker along with the URI."""
    body = "(see below). Strips `data:application/pdf;base64,` prefix, decodes\n"
    out = strip_data_uris(body)
    assert "[image]" in out
    assert "`[image]`" in out, "the code-span backtick pair must survive"
    assert out == "(see below). Strips `[image]` prefix, decodes\n"
