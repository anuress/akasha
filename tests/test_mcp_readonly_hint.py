"""A caller should learn a document is read-only before it edits, not by having update,
append or archive refuse it. knowledge_get carries the flag and how to edit; search hits
carry the flag only."""
from __future__ import annotations

import pytest

from akasha.config import IndexRoot, load_config
from akasha.db import connect
from akasha.index import index_path
from akasha.mcp_server import call_tool


@pytest.fixture
def env(tmp_path, monkeypatch):
    cfg = load_config(tmp_path / "absent.toml")
    cfg.knowledge_dir = tmp_path / "knowledge"
    cfg.knowledge_dir.mkdir()
    cfg.index_roots = []
    conn = connect(tmp_path / "s.db")
    monkeypatch.setattr("akasha.mcp_server._ctx", lambda: (conn, cfg))
    return conn, cfg


def _external_doc(conn, cfg, tmp_path):
    root = tmp_path / "imported"
    root.mkdir()
    doc = root / "notes.md"
    doc.write_text("## Findings\nthe import runs nightly\n")
    doc_id, _, _ = index_path(conn, cfg, doc, "serena", root=None)
    cfg.index_roots.append(IndexRoot(path=str(root), source="serena"))
    return doc_id


def test_get_flags_an_external_document_with_the_readonly_hint(env, tmp_path):
    conn, cfg = env
    result = call_tool("knowledge_get", {"id": _external_doc(conn, cfg, tmp_path)})
    assert result["writable"] is False
    assert "knowledge_write" in result["readonly_hint"]
    assert "to_edit" not in result


def test_get_omits_the_hint_for_a_native_document(env):
    doc_id = call_tool("knowledge_write", {"title": "T", "body": "## H\nbody", "repo": "r"})["id"]
    result = call_tool("knowledge_get", {"id": doc_id})
    assert "writable" not in result and "readonly_hint" not in result


def test_search_flags_an_external_hit_without_the_how_to(env, tmp_path):
    conn, cfg = env
    _external_doc(conn, cfg, tmp_path)
    hit = call_tool("knowledge_search", {"q": "import runs nightly", "all_repos": True})[0]
    assert hit["writable"] is False and "readonly_hint" not in hit


def test_search_omits_the_flag_for_a_native_hit(env):
    call_tool("knowledge_write", {"title": "T", "body": "## H\nlimit 30 wins", "repo": "r"})
    hit = call_tool("knowledge_search", {"q": "limit 30 wins", "all_repos": True})[0]
    assert "writable" not in hit
