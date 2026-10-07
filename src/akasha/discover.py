"""First-run source discovery, so the default config already points at real notes."""
from __future__ import annotations

import json
from pathlib import Path

from akasha.config import IndexRoot


def discover_sources(home: Path, code_roots: list[Path]) -> list[IndexRoot]:
    """Find knowledge sources that actually exist on this machine."""
    found: list[IndexRoot] = []
    home = Path(home)

    for project in sorted((home / ".claude" / "projects").glob("*/memory")):
        if project.is_dir():
            found.append(IndexRoot(path=str(project), source="claude-memory"))

    for code_root in code_roots:
        code_root = Path(code_root)
        if not code_root.is_dir():
            continue
        for repo in sorted(p for p in code_root.iterdir() if p.is_dir()):
            memories = repo / ".serena" / "memories"
            if memories.is_dir():
                # The repo name is above the root, so the walk cannot infer it later.
                found.append(IndexRoot(path=str(memories), source="serena",
                                       repo=repo.name))
            graphify = repo / "graphify-out"
            if graphify.is_dir():
                found.append(IndexRoot(path=str(graphify), source="graphify",
                                       include=["*.md"], exclude=["cache/**"],
                                       repo=repo.name))
    return found


def config_from_discovery(roots: list[IndexRoot]) -> str:
    lines = ['[paths]', 'db        = "~/.akasha/akasha.db"',
             'knowledge = "~/.akasha/knowledge"', '']
    if not roots:
        lines += ["# No sources detected. Add [[index]] blocks pointing at your notes.", ""]
    for root in roots:
        lines.append("[[index]]")
        # JSON string escaping is TOML basic-string escaping (kept unicode, since TOML
        # rejects surrogate escapes): paths can hold quotes and backslashes.
        lines.append(f"path    = {json.dumps(root.path, ensure_ascii=False)}")
        lines.append(f"source  = {json.dumps(root.source, ensure_ascii=False)}")
        if root.include != ["**/*.md"]:
            lines.append(f"include = {json.dumps(root.include, ensure_ascii=False)}")
        if root.exclude:
            lines.append(f"exclude = {json.dumps(root.exclude, ensure_ascii=False)}")
        if root.repo:
            lines.append(f"repo    = {json.dumps(root.repo, ensure_ascii=False)}")
        lines.append("")
    return "\n".join(lines)
