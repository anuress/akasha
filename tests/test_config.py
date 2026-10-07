import tomllib
from pathlib import Path

from akasha.config import default_config_toml, expand_roots, IndexRoot, load_config


def test_load_config_uses_defaults_when_file_missing(tmp_path):
    cfg = load_config(tmp_path / "nope.toml")
    assert cfg.knowledge_dir.name == "knowledge"
    assert cfg.db_path.name == "akasha.db"
    assert cfg.embeddings_provider == "none"
    assert "local.properties" in cfg.deny_files
    assert ".keystore" in cfg.deny_extensions
    assert "id_rsa" in cfg.deny_files
    assert ".pem" in cfg.deny_extensions
    assert cfg.scan_secrets is True


def test_load_config_reads_values_and_expands_home(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(
        '[paths]\n'
        'db = "~/custom/akasha.db"\n'
        'knowledge = "~/custom/knowledge"\n'
        '\n'
        '[[index]]\n'
        'path = "~/notes"\n'
        'source = "notes"\n'
        'include = ["**/*.md"]\n'
        'exclude = ["**/cache/**"]\n'
        '\n'
        '[embeddings]\n'
        'provider = "fastembed"\n'
    )
    cfg = load_config(p)
    assert cfg.db_path == Path.home() / "custom" / "akasha.db"
    assert cfg.embeddings_provider == "fastembed"
    assert len(cfg.index_roots) == 1
    assert cfg.index_roots[0].source == "notes"
    assert cfg.index_roots[0].exclude == ["**/cache/**"]


def test_default_config_toml_is_parseable_and_matches_the_defaults(tmp_path):
    """The generated file is what `init` writes, so it must load to the same values the
    loader falls back to when the file is absent."""
    p = tmp_path / "config.toml"
    p.write_text(default_config_toml())
    tomllib.loads(p.read_text())
    written, absent = load_config(p), load_config(tmp_path / "absent.toml")
    assert written.embeddings_provider == absent.embeddings_provider
    assert written.scan_secrets == absent.scan_secrets
    assert written.housekeeping_interval_min == absent.housekeeping_interval_min
    assert written.knowledge_stale_after_days is None


def test_retention_and_staleness_defaults(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")
    assert cfg.housekeeping_interval_min == 60
    assert cfg.knowledge_stale_after_days is None


def test_stale_after_days_is_read_when_set(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[knowledge]\nstale_after_days = 90\n")
    assert load_config(p).knowledge_stale_after_days == 90


def test_index_root_without_include_has_no_filter(tmp_path):
    """A default include of '**/*.md' silently skips flat directories under fnmatch."""
    p = tmp_path / "config.toml"
    p.write_text('[[index]]\npath = "~/x"\nsource = "s"\n')
    assert load_config(p).index_roots[0].include == []


def test_projects_expand_their_paths(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('[projects.app]\npath = "~/code/app"\nfeature = "f1"\n')
    project = load_config(p).projects["app"]
    assert project["path"] == str(Path.home() / "code" / "app")
    assert project["feature"] == "f1"


def test_a_glob_root_expands_to_each_matching_directory_and_keeps_every_field(tmp_path):
    """Without expansion a glob silently matches nothing. The expanded roots must carry
    every field of the original, `writable` above all, where losing it would fail open."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "file.txt").write_text("x")
    root = IndexRoot(path=str(tmp_path / "*"), source="s", repo="r", writable=True,
                     exclude=["x"])
    out = expand_roots([root])
    assert [r.path for r in out] == [str(tmp_path / "a"), str(tmp_path / "b")]
    assert all(r.repo == "r" and r.writable and r.exclude == ["x"] for r in out)


def test_a_plain_root_passes_through_untouched(tmp_path):
    root = IndexRoot(path=str(tmp_path), source="s")
    assert expand_roots([root]) == [root]


def test_a_glob_matching_nothing_expands_to_nothing(tmp_path):
    assert expand_roots([IndexRoot(path=str(tmp_path / "nope/*/x"), source="s")]) == []
