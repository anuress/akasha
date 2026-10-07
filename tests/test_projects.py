"""Projects are config the user edits: a name keyed to a checkout path, with an optional
feature override. gitctx reads them to name a repo and a feature."""
import subprocess

from akasha.config import load_config


def _repo(path, branch="main"):
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", "-b", branch], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@e.st"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    (path / "README.md").write_text("x")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=path, check=True)
    return path


def test_git_context_names_the_project_that_owns_a_path(tmp_path):
    """The directory is deliberately not the project name: the configured name wins over
    the checkout's basename."""
    from akasha.gitctx import git_context

    repo = _repo(tmp_path / "catalog-web")
    path = tmp_path / "config.toml"
    path.write_text(f'[projects.library-site]\npath = "{repo}"\n')
    name, _feature = git_context(load_config(path), repo)
    assert name == "library-site"


def test_the_feature_override_still_works(tmp_path):
    from akasha.gitctx import git_context

    repo = _repo(tmp_path / "catalog-web")
    path = tmp_path / "config.toml"
    path.write_text(f'[projects.library-site]\npath = "{repo}"\nfeature = "circulation"\n')
    name, feature = git_context(load_config(path), repo)
    assert (name, feature) == ("library-site", "circulation")
