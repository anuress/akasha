"""Derive repo and feature from git. Nothing is ever written into a repository."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from akasha.config import Config

TRUNK_BRANCHES = {"main", "master", "develop", "release", "head", "trunk"}
TICKET = re.compile(r"^([a-z]{2,6}-\d+)", re.IGNORECASE)


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                                text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


# Repo identity by cwd. A process's cwd does not change, and neither does which checkout
# it is in, so one git call answers every later derivation. Only hits are stored: a
# directory outside a repo may become one. The branch is deliberately not part of this,
# because it changes while the process runs.
# Known limit: a hit lives for the process, so a repo created inside an already-cached
# directory, or a worktree removed, is not seen until restart.
_ROOTS: dict[str, Path] = {}


def repo_root(cwd: Path) -> Path | None:
    """The checkout's identity root: for a linked worktree, the *main* checkout —
    never the worktree's own path.

    `--show-toplevel` inside a linked worktree names the worktree's own directory, which
    is not the project, and `cfg.projects` is keyed by the main checkout's path.
    `--git-common-dir` is shared by every worktree and always names the main checkout's
    `.git`; for a plain checkout it equals `--git-dir`. `--path-format=absolute` matters
    because some git versions print the common dir relative to cwd, which would never
    compare equal to `--git-dir`. All three come from one `rev-parse`.
    """
    cwd = Path(cwd).resolve()
    key = str(cwd)
    if key in _ROOTS:
        return _ROOTS[key]
    out = _git(["rev-parse", "--path-format=absolute", "--git-common-dir", "--git-dir",
                "--show-toplevel"], cwd)
    lines = out.splitlines() if out else []
    if len(lines) != 3:
        return None
    common, git_dir, top = lines
    # Linked worktree: identity is the main checkout.
    root = Path(common).parent if Path(git_dir) != Path(common) else Path(top)
    _ROOTS[key] = root
    return root


def feature_from_branch(branch: str | None) -> str | None:
    """A ticket prefix if present, else the branch name. Trunk branches mean no feature."""
    if not branch:
        return None
    name = branch.strip().split("/")[-1]
    if name.lower() in TRUNK_BRANCHES:
        return None
    match = TICKET.match(name)
    if match:
        return match.group(1).lower()
    return name.lower() or None


def _project(cfg: Config, cwd: Path) -> tuple[str, dict] | None:
    """(repo name, its config block) for the checkout at `cwd`, or None outside one."""
    root = repo_root(cwd)
    if root is None:
        return None
    # Projects are keyed by name, so the path lookup is a scan; there are a handful.
    for name, settings in cfg.projects.items():
        if (settings or {}).get("path") and Path(settings["path"]) == root:
            return name, settings
    return root.name, {}


def git_context(cfg: Config, cwd: Path) -> tuple[str | None, str | None]:
    cwd = Path(cwd)
    found = _project(cfg, cwd)
    if found is None:
        return None, None
    repo, override = found
    if "feature" in override:
        return repo, override["feature"]

    # HEAD is read from the caller's own checkout, not from the repo root: inside a
    # linked worktree the root is the main checkout, whose HEAD is a different branch
    # from the worktree's own, which is the one the feature lives on.
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
    return repo, feature_from_branch(branch)


def resolve_repo(cfg: Config, cwd: Path, explicit: str | None) -> tuple[str | None, str]:
    """The one place repo scope is decided, for every read and write.

    `explicit` (a CLI --repo or an MCP `repo` argument) always wins. `"*"` means no
    filter, returned as `(None, "all")` so a caller can tell a deliberate wide-open scope
    from one that failed to derive. Otherwise the checkout at `cwd` derives the repo;
    `source` is `"cwd"` when that worked and `"none"` when `cwd` is not inside a checkout.
    """
    if explicit == "*":
        return None, "all"
    if explicit:
        return explicit, "explicit"
    found = _project(cfg, Path(cwd))
    if found:
        return found[0], "cwd"
    return None, "none"
