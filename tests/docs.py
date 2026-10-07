"""Test helper: put a native markdown document on disk and index it."""
from akasha.index import index_path
from akasha.markdown import render


def add_doc(conn, cfg, title, body, repo="sample-repo", feature="f", **meta):
    """Write `title`/`body` under the knowledge dir with extra frontmatter in `meta`,
    index it, and return the document id."""
    slug = "-".join(title.lower().split())[:50]
    path = cfg.knowledge_dir / f"{slug}.md"
    front = {"title": title, "repo": repo, "feature": feature, **meta}
    path.write_text(render(front, body))
    doc_id, _, _ = index_path(conn, cfg, path, "native", root=cfg.knowledge_dir)
    return doc_id
