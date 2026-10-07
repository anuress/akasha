"""Adding an index root without hand-editing TOML."""
import tomllib

import pytest

from akasha.config_edit import ConfigWriteError


@pytest.fixture
def cfg_path(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[paths]\ndb = "~/.akasha/akasha.db"\n\n# keep me\n[security]\n'
                    'scan_secrets = true\n')
    return path


def test_adding_an_index_root_writes_a_block(cfg_path, tmp_path):
    from akasha.register import add_index

    notes = tmp_path / "notes"
    notes.mkdir()
    add_index(cfg_path, notes, source="notes")
    roots = tomllib.loads(cfg_path.read_text())["index"]
    assert roots[-1]["path"] == str(notes)
    assert roots[-1]["source"] == "notes"


def test_an_index_root_defaults_its_source_to_the_directory_name(cfg_path, tmp_path):
    from akasha.register import add_index

    notes = tmp_path / "notes-dir"
    notes.mkdir()
    add_index(cfg_path, notes)
    assert tomllib.loads(cfg_path.read_text())["index"][-1]["source"] == "notes-dir"


def test_an_index_root_omits_include_by_default(cfg_path, tmp_path):
    """A default include of '**/*.md' misses flat directories under fnmatch."""
    from akasha.register import add_index

    notes = tmp_path / "notes"
    notes.mkdir()
    add_index(cfg_path, notes)
    assert "include" not in tomllib.loads(cfg_path.read_text())["index"][-1]


def test_adding_the_same_index_root_twice_is_idempotent(cfg_path, tmp_path):
    from akasha.register import add_index

    notes = tmp_path / "notes"
    notes.mkdir()
    add_index(cfg_path, notes)
    assert "already" in add_index(cfg_path, notes)
    assert len(tomllib.loads(cfg_path.read_text())["index"]) == 1


def test_a_glob_index_path_is_kept_verbatim(cfg_path, tmp_path):
    """expand_roots resolves '*' at load; storing the expansion would freeze it."""
    from akasha.register import add_index

    (tmp_path / "repos" / "a" / "docs").mkdir(parents=True)
    add_index(cfg_path, f"{tmp_path}/repos/*/docs", source="docs")
    assert tomllib.loads(cfg_path.read_text())["index"][-1]["path"].endswith("/*/docs")


def test_a_glob_whose_anchor_is_missing_is_refused(cfg_path, tmp_path):
    """A typo in a glob would otherwise index nothing and say nothing."""
    from akasha.register import add_index

    with pytest.raises(ConfigWriteError, match="does not exist"):
        add_index(cfg_path, f"{tmp_path}/typo/*/docs", source="docs")


def test_registration_never_writes_an_unparseable_config(cfg_path, tmp_path):
    from akasha.register import add_index

    notes = tmp_path / 'weird"name'
    notes.mkdir()
    add_index(cfg_path, notes, source="weird")
    tomllib.loads(cfg_path.read_text())      # must not raise


def test_existing_config_survives_registration(cfg_path, tmp_path):
    from akasha.register import add_index

    notes = tmp_path / "notes"
    notes.mkdir()
    add_index(cfg_path, notes)
    text = cfg_path.read_text()
    assert "# keep me" in text
    assert tomllib.loads(text)["security"]["scan_secrets"] is True


def test_a_missing_directory_is_refused(cfg_path, tmp_path):
    from akasha.register import add_index

    with pytest.raises(ConfigWriteError, match="does not exist"):
        add_index(cfg_path, tmp_path / "nope")

