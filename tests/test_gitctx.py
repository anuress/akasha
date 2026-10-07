import subprocess
from pathlib import Path

import pytest

from akasha import gitctx
from akasha.config import load_config
from akasha.gitctx import feature_from_branch, git_context, repo_root, resolve_repo


def _git_repo(path, branch="main"):
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", "-b", branch], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@e.st"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    (path / "README.md").write_text("x")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=path, check=True)
    return path


def _linked_worktree(main, name, branch):
    path = main.parent / name
    subprocess.run(["git", "worktree", "add", "-q", "-b", branch, str(path)],
                   cwd=main, check=True)
    return path


@pytest.fixture
def git_calls(monkeypatch):
    """Count git subprocesses started by gitctx."""
    calls: list[list[str]] = []
    real_run = subprocess.run

    def counting_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(gitctx.subprocess, "run", counting_run)
    return calls


def test_feature_from_ticket_branch():
    assert feature_from_branch("proj-123-add-login") == "proj-123"
    assert feature_from_branch("PROJ-100") == "proj-100"
    assert feature_from_branch("feature/PROJ-124-add-login") == "proj-124"


def test_feature_from_descriptive_branch():
    assert feature_from_branch("catalog-cache-experiment") == "catalog-cache-experiment"


def test_trunk_branches_yield_no_feature():
    for branch in ("main", "master", "develop", "HEAD"):
        assert feature_from_branch(branch) is None


def test_repo_root_from_a_subdirectory(tmp_path):
    root = _git_repo(tmp_path / "sample-repo")
    nested = root / "app" / "src"
    nested.mkdir(parents=True)
    assert repo_root(nested) == root


def test_repo_root_outside_a_repo_is_none(tmp_path):
    assert repo_root(tmp_path) is None


def test_git_context_derives_repo_and_feature(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")
    root = _git_repo(tmp_path / "sample-repo", branch="proj-100-extract-loans")
    assert git_context(cfg, root) == ("sample-repo", "proj-100")


def test_git_context_on_trunk_gives_repo_only(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")
    root = _git_repo(tmp_path / "sample-repo-b", branch="develop")
    assert git_context(cfg, root) == ("sample-repo-b", None)


def test_config_override_wins_over_derivation(tmp_path):
    root = _git_repo(tmp_path / "sample-repo", branch="develop")
    p = tmp_path / "config.toml"
    p.write_text(
        '[projects.renamed-repo]\n'
        f'path = "{root}"\n'
        'feature = "proj-100"\n'
    )
    cfg = load_config(p)
    assert git_context(cfg, root) == ("renamed-repo", "proj-100")


def test_outside_a_repo_returns_nothing(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")
    assert git_context(cfg, tmp_path) == (None, None)


def test_no_dotfile_is_ever_written(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")
    root = _git_repo(tmp_path / "repo", branch="proj-1-x")
    git_context(cfg, root)
    assert not (root / ".akasha").exists()
    assert list(root.glob(".akasha*")) == []


def test_repo_root_from_inside_a_worktree_is_the_main_checkout(tmp_path):
    """`--show-toplevel` inside a linked worktree names the worktree's own path, which is
    not the project. A real `git worktree add` proves the fix against git's actual
    output rather than an assumption about it."""
    main = _git_repo(tmp_path / "sample-repo")
    worktree = _linked_worktree(main, "some-slug", "some-slug")
    assert repo_root(worktree) == main


def test_git_context_from_a_worktree_uses_main_repo_name_and_the_worktrees_own_branch(tmp_path):
    """Repo identity comes from the main checkout, but the feature comes from the
    worktree's own branch: they are different HEADs sharing one `.git`."""
    main = _git_repo(tmp_path / "sample-repo", branch="main")
    cfg = load_config(tmp_path / "absent.toml")
    worktree = _linked_worktree(main, "wt", "work/proj-100-fines")
    assert git_context(cfg, worktree) == ("sample-repo", "proj-100")


def test_git_context_from_a_worktree_matches_a_project_keyed_by_the_main_checkouts_path(tmp_path):
    """cfg.projects is keyed by the main checkout's path, so a repo derived inside a
    linked worktree must equal it or a configured rename or feature override stops
    applying."""
    main = _git_repo(tmp_path / "sample-repo", branch="develop")
    p = tmp_path / "config.toml"
    p.write_text('[projects.renamed-repo]\n' f'path = "{main}"\n' 'feature = "proj-100"\n')
    cfg = load_config(p)
    worktree = _linked_worktree(main, "wt", "scratch")
    assert git_context(cfg, worktree) == ("renamed-repo", "proj-100")


def test_resolve_repo_explicit_wins_over_derivation(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")
    root = _git_repo(tmp_path / "sample-repo")
    assert resolve_repo(cfg, root, "other-repo") == ("other-repo", "explicit")


def test_resolve_repo_star_means_no_filter(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")
    root = _git_repo(tmp_path / "sample-repo")
    assert resolve_repo(cfg, root, "*") == (None, "all")


def test_resolve_repo_falls_back_to_the_checkout_at_cwd(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")
    root = _git_repo(tmp_path / "sample-repo")
    assert resolve_repo(cfg, root, None) == ("sample-repo", "cwd")


def test_resolve_repo_outside_any_checkout_derives_nothing(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")
    assert resolve_repo(cfg, tmp_path, None) == (None, "none")


# --- per-process cache of the repo derivation ----------------------------------------

def test_repo_derivation_runs_git_once_across_repeated_calls(tmp_path, git_calls):
    """Every tool call derives the repo from cwd, and cwd does not change within a
    process. Re-running git each time costs tens of milliseconds per call for an answer
    that cannot change: the first derivation is one git call, later ones are none."""
    cfg = load_config(tmp_path / "absent.toml")
    root = _git_repo(tmp_path / "sample-repo")
    git_calls.clear()

    assert resolve_repo(cfg, root, None) == ("sample-repo", "cwd")
    assert resolve_repo(cfg, root, None) == ("sample-repo", "cwd")

    assert len(git_calls) == 1


def test_branch_is_never_cached(tmp_path, git_calls):
    """The feature follows the checked-out branch, which changes mid-process, so only the
    repo identity may be cached."""
    cfg = load_config(tmp_path / "absent.toml")
    root = _git_repo(tmp_path / "sample-repo", branch="main")
    assert git_context(cfg, root) == ("sample-repo", None)

    subprocess.run(["git", "checkout", "-q", "-b", "proj-7-next"], cwd=root, check=True)
    assert git_context(cfg, root) == ("sample-repo", "proj-7")


def test_repo_derivation_outside_a_repo_is_not_cached(tmp_path):
    """A directory that later becomes a repo must be seen as one."""
    cfg = load_config(tmp_path / "absent.toml")
    where = tmp_path / "later"
    where.mkdir()
    assert resolve_repo(cfg, where, None) == (None, "none")

    _git_repo(where)
    assert resolve_repo(cfg, where, None) == ("later", "cwd")


def test_relative_cwd_follows_the_checkout_it_currently_names(tmp_path, monkeypatch):
    """The cache key must be the resolved path: "." names a different checkout after chdir."""
    a = _git_repo(tmp_path / "a")
    b = _git_repo(tmp_path / "b")
    monkeypatch.chdir(a)
    assert repo_root(Path(".")) == a
    monkeypatch.chdir(b)
    assert repo_root(Path(".")) == b


def test_short_rev_parse_output_means_not_a_repo(tmp_path, monkeypatch):
    """Fewer than three lines from rev-parse is not a repo, not a crash."""
    monkeypatch.setattr(gitctx, "_git", lambda args, cwd: "only\ntwo")
    assert repo_root(tmp_path / "short") is None
