"""Configuration loading. Everything is path-driven so a fresh machine works."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

# Matched against the file name with fnmatch, so globs work.
DEFAULT_DENY_FILES = [
    ".env.*",
    "id_rsa",
    "id_ed25519",
    ".netrc",
    ".npmrc",
    ".pypirc",
    ".git-credentials",
    "credentials.json",
    "service-account*.json",
    "secrets.properties",
    "keystore.properties",
    "local.properties",
    "gradle.properties",
    "google-services.json",
    "GoogleService-Info.plist",
]
# Matched as a case-insensitive suffix of the file name, so ".env" also covers a bare
# ".env" file.
DEFAULT_DENY_EXTENSIONS = [".env", ".pem", ".key", ".p12", ".pfx", ".kdbx", ".keystore", ".jks"]

@dataclass
class IndexRoot:
    path: str
    source: str
    # Empty by default: index_all already restricts to .md, and a pattern like "**/*.md"
    # misses flat directories because fnmatch needs a literal "/".
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    # The repo the whole root belongs to. Needed when the root sits inside a repo, because
    # repo is otherwise the first path segment below the root, and the repo name is above.
    repo: str | None = None
    # Opt-in write access for knowledge_append. Off by default and per root, because a
    # root's owner may not expect akasha to write into it.
    writable: bool = False


@dataclass
class Config:
    db_path: Path
    knowledge_dir: Path
    index_roots: list[IndexRoot]
    deny_files: list[str]
    deny_extensions: list[str]
    scan_secrets: bool
    embeddings_provider: str
    # One block per project, keyed by name: `path` and an optional `feature` override.
    projects: dict = field(default_factory=dict)
    # Minutes between retention passes; read by cleanup.py.
    housekeeping_interval_min: int = 60
    # Days of disuse before a document is reported stale. Unset means never: disuse alone
    # is not evidence a document is wrong.
    knowledge_stale_after_days: int | None = None


def _expand(value: str) -> Path:
    return Path(value).expanduser()


def default_config_toml() -> str:
    return '''[paths]
db        = "~/.akasha/akasha.db"
knowledge = "~/.akasha/knowledge"

# One block per directory of markdown to index.
# [[index]]
# path    = "~/notes"
# source  = "notes"
# include = ["**/*.md"]
# exclude = ["**/cache/**"]

[embeddings]
provider = "none"   # none | fastembed | model2vec | api

[security]
scan_secrets = true

[housekeeping]
# Minutes `akasha housekeeping` waits before it repeats the retention pass.
# 0 runs the pass on every invocation.
interval_min = 60

[knowledge]
# Days of disuse before `akasha knowledge stale` reports a document. Unset means never.
# stale_after_days = 90
'''


def expand_roots(roots: list[IndexRoot]) -> list[IndexRoot]:
    """Expand `*` in configured paths. Without this a glob silently matches nothing."""
    expanded: list[IndexRoot] = []
    for root in roots:
        path = str(Path(root.path).expanduser())
        if "*" not in path:
            expanded.append(root)
            continue
        anchor = Path(path.split("*")[0]).parent
        pattern = str(Path(path).relative_to(anchor)) if anchor != Path(path) else path
        for match in sorted(anchor.glob(pattern)):
            if match.is_dir():
                # replace() keeps every field, including `writable`, where losing it
                # would fail open.
                expanded.append(replace(root, path=str(match)))
    return expanded


def load_config(path: Path | None = None) -> Config:
    path = Path(path) if path else Path.home() / ".akasha" / "config.toml"
    raw: dict = {}
    if path.exists():
        raw = tomllib.loads(path.read_text())

    paths = raw.get("paths", {})
    security = raw.get("security", {})
    roots = expand_roots([
        IndexRoot(
            path=r["path"],
            source=r["source"],
            include=r.get("include", []),
            exclude=r.get("exclude", []),
            repo=r.get("repo"),
            writable=r.get("writable", False),
        )
        for r in raw.get("index", [])
    ])
    return Config(
        db_path=_expand(paths.get("db", "~/.akasha/akasha.db")),
        knowledge_dir=_expand(paths.get("knowledge", "~/.akasha/knowledge")),
        index_roots=roots,
        deny_files=security.get("deny_files", DEFAULT_DENY_FILES),
        deny_extensions=security.get("deny_extensions", DEFAULT_DENY_EXTENSIONS),
        scan_secrets=security.get("scan_secrets", True),
        embeddings_provider=raw.get("embeddings", {}).get("provider", "none"),
        projects=_projects(raw.get("projects", {})),
        housekeeping_interval_min=int(raw.get("housekeeping", {}).get("interval_min", 60)),
        knowledge_stale_after_days=_optional_int(
            raw.get("knowledge", {}).get("stale_after_days")),
    )


def _optional_int(value) -> int | None:
    """None stays None — for a setting whose absence is itself the policy."""
    return None if value is None else int(value)


def _projects(raw: dict) -> dict:
    """Project blocks with their paths expanded. The name is the key, always."""
    projects = {}
    for name, settings in (raw or {}).items():
        entry = dict(settings or {})
        if entry.get("path"):
            entry["path"] = str(_expand(str(entry["path"])))
        projects[name] = entry
    return projects
