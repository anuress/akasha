import pytest

from akasha.discover import config_from_discovery, discover_sources


@pytest.fixture
def fake_home(tmp_path):
    (tmp_path / "notes" / "repo-a").mkdir(parents=True)
    (tmp_path / ".claude" / "projects" / "-home-user-code" / "memory").mkdir(parents=True)
    (tmp_path / "code" / "repo-a" / ".serena" / "memories").mkdir(parents=True)
    (tmp_path / "code" / "repo-b" / ".serena" / "memories").mkdir(parents=True)
    (tmp_path / "code" / "repo-a" / "graphify-out" / "cache").mkdir(parents=True)
    return tmp_path


def test_a_generic_notes_directory_is_not_discovered(fake_home):
    """Only the known tool locations are discovered; a notes folder is added explicitly."""
    roots = discover_sources(fake_home, [fake_home / "code"])
    assert {r.source for r in roots} == {"claude-memory", "serena", "graphify"}


def test_discovers_every_serena_directory(fake_home):
    roots = discover_sources(fake_home, [fake_home / "code"])
    serena = [r for r in roots if r.source == "serena"]
    assert len(serena) == 2


def test_discovers_claude_memory(fake_home):
    roots = discover_sources(fake_home, [fake_home / "code"])
    assert any(r.source == "claude-memory" for r in roots)


def test_graphify_root_excludes_the_cache_directory(fake_home):
    roots = discover_sources(fake_home, [fake_home / "code"])
    graphify = [r for r in roots if r.source == "graphify"]
    assert graphify and any("cache" in pattern for pattern in graphify[0].exclude)


def test_discovery_finds_nothing_on_an_empty_home(tmp_path):
    assert discover_sources(tmp_path, []) == []


def test_generated_config_is_parseable_and_contains_sources(fake_home):
    import tomllib

    text = config_from_discovery(discover_sources(fake_home, [fake_home / "code"]))
    parsed = tomllib.loads(text)
    assert len(parsed["index"]) >= 4


def test_a_serena_root_is_labelled_with_the_repo_that_owns_it(tmp_path):
    """The repo name sits above the root, so the walk cannot infer it later: discovery
    must record it, or notes are labelled with a subfolder name or nothing."""
    from akasha.discover import discover_sources

    home = tmp_path / "home"
    (home / "code" / "sample-repo" / ".serena" / "memories").mkdir(parents=True)
    (home / "code" / "sample-repo" / "graphify-out").mkdir(parents=True)

    roots = {r.source: r for r in discover_sources(home, [home / "code"])}

    assert roots["serena"].repo == "sample-repo"
    assert roots["graphify"].repo == "sample-repo"


def test_a_path_with_a_quote_and_backslash_round_trips_through_the_config(tmp_path):
    """Paths are written into TOML strings, so they must be escaped."""
    from akasha.config import IndexRoot, load_config
    from akasha.discover import config_from_discovery

    odd = tmp_path / 'we"ird\\dir'
    config = tmp_path / "config.toml"
    config.write_text(config_from_discovery(
        [IndexRoot(path=str(odd), source='s"rc', repo='r\\epo', exclude=['a"b'])]))
    [root] = load_config(config).index_roots
    assert (root.path, root.source, root.repo, root.exclude) == (
        str(odd), 's"rc', 'r\\epo', ['a"b'])
