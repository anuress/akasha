import pytest

from akasha.config import IndexRoot, load_config
from akasha.db import connect
from akasha.index import index_all
from akasha.search import link_candidates, rrf_merge, search
from docs import add_doc


@pytest.fixture
def env(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")
    cfg.knowledge_dir = tmp_path / "knowledge"
    cfg.knowledge_dir.mkdir()
    root = tmp_path / "ws"
    d = root / "sample-repo" / "catalog-cache"
    d.mkdir(parents=True)
    (d / "findings.md").write_text("## Cache\nthe cache layer is warm after a restart\n")
    (d / "other.md").write_text("## Loans\nrenewal and overdue reminders\n")
    other = root / "sample-repo-b" / "home"
    other.mkdir(parents=True)
    (other / "notes.md").write_text("## Cache\nweb caches sessions differently\n")
    cfg.index_roots = [IndexRoot(path=str(root), source="notes")]
    conn = connect(tmp_path / "s.db")
    index_all(conn, cfg)
    return conn, cfg


def test_search_returns_matching_chunk_text(env):
    conn, _ = env
    hits = search(conn, "cache", all_repos=True)
    assert hits
    assert any("restart" in h.text for h in hits)
    assert hits[0].path.endswith(".md")


def test_search_stems_so_caching_matches_cache(env):
    conn, _ = env
    assert search(conn, "caching", all_repos=True)


def test_search_filters_by_repo(env):
    conn, _ = env
    hits = search(conn, "cache", repo="sample-repo-b")
    assert hits
    assert all(h.repo == "sample-repo-b" for h in hits)


def test_heading_hit_outranks_body_hit(env):
    conn, _ = env
    conn.execute(
        "INSERT INTO documents (id, source, path, title, repo, status, created_at, updated_at)"
        " VALUES ('d9','native','/tmp/z.md','Z','sample-repo','active',"
        " datetime('now'), datetime('now'))")
    conn.execute(
        "INSERT INTO chunks (id, document_id, heading, body, ord)"
        " VALUES ('c9','d9','## Slotting','mentions slotting once',0)")
    conn.execute(
        "INSERT INTO chunks (id, document_id, heading, body, ord)"
        " VALUES ('c10','d9','## Other','slotting appears in the body only',1)")
    conn.commit()
    hits = search(conn, "slotting", all_repos=True, limit=2)
    assert hits[0].heading == "## Slotting"


def test_archived_ranks_below_active(env):
    conn, _ = env
    conn.execute("UPDATE documents SET status='archived' WHERE path LIKE '%web%'")
    conn.commit()
    hits = search(conn, "cache", all_repos=True, include_archived=True)
    statuses = [
        conn.execute("SELECT status FROM documents WHERE id=?", (h.document_id,)).fetchone()["status"]
        for h in hits
    ]
    assert statuses[0] == "active"


def test_archived_excluded_by_default(env):
    conn, _ = env
    conn.execute("UPDATE documents SET status='archived' WHERE path LIKE '%web%'")
    conn.commit()
    hits = search(conn, "cache", all_repos=True)
    assert all("web" not in h.path for h in hits)


def test_alias_query_returns_canonical_feature_docs(env):
    conn, _ = env
    conn.execute("UPDATE features SET aliases='[\"old-alias\"]' WHERE slug='catalog-cache'")
    conn.commit()
    hits = search(conn, "cache", feature="old-alias", all_repos=True)
    assert hits


def test_search_with_no_embeddings_never_errors(env):
    conn, _ = env
    assert isinstance(search(conn, "nonexistent-term-xyz", all_repos=True), list)


def test_rrf_merge_ranks_item_present_in_both_lists_first():
    merged = rrf_merge([["a", "b", "c"], ["c", "a", "z"]])
    assert merged[0] == "a"
    assert "z" in merged


def test_results_are_one_per_document(env):
    """A limit of N should surface N documents, not N excerpts of one."""
    conn, cfg = env
    add_doc(conn, cfg, "Multi", "## A\ncache flow one\n## B\ncache flow two\n"
          "## C\ncache flow three\n", repo="sample-repo", feature="f")
    hits = search(conn, "cache flow", all_repos=True, limit=5)
    assert len(hits) == len({h.document_id for h in hits})


def test_heading_only_sections_do_not_become_empty_hits(env):
    conn, cfg = env
    add_doc(conn, cfg, "Evidence doc", "## Evidence\n## Detail\nunique-body-token\n",
          repo="r", feature="f")
    for hit in search(conn, "unique-body-token", all_repos=True, limit=5):
        assert hit.text.strip(), "a hit with no body text is noise"


def test_a_repo_scope_that_finds_nothing_falls_back_to_every_repo(env):
    """The scope is a preference, not a filter that may return nothing: an answer recorded
    under another repo outranks a loose match in this one, and the hit says it widened."""
    conn, _ = env
    # "sessions" exists only under sample-repo-b; the scope names sample-repo.
    hits = search(conn, "sessions", repo="sample-repo")

    assert hits, "a scope with no match must not return nothing"
    assert hits[0].repo == "sample-repo-b"
    assert hits[0].cross_repo is True, "and must say it widened"


def test_hit_reports_neighbour_count_matching_related(env):
    """A hit on a linked document carries its resolved-link count in both directions, so
    the caller can tell when a follow-up hop pays without a call per hit."""
    conn, cfg = env
    a = add_doc(conn, cfg, "Hop doc", "## A\nshared-token alpha\n", repo="sample-repo", feature="f")
    b = add_doc(conn, cfg, "Leaf doc", "## B\nshared-token beta\n", repo="sample-repo", feature="f")
    conn.execute(
        "INSERT INTO links (id, from_document_id, to_document_id, to_ref, kind, created_at)"
        " VALUES ('l1', ?, ?, 'Hop doc', 'cites', datetime('now'))", (b, a))
    conn.commit()

    hits = search(conn, "shared-token", all_repos=True, limit=5)
    by_doc = {h.document_id: h for h in hits}
    assert by_doc[a].neighbours == 1, "backlink must count"
    assert by_doc[b].neighbours == 1, "outbound citation must count"


def test_dangling_link_is_not_a_neighbour(env):
    """A ref whose target never resolved must not promise a hop: only a resolved
    to_document_id counts."""
    conn, cfg = env
    a = add_doc(conn, cfg, "Only doc", "## A\nsolo-token\n", repo="sample-repo", feature="f")
    conn.execute(
        "INSERT INTO links (id, from_document_id, to_document_id, to_ref, kind, created_at)"
        " VALUES ('l1', ?, NULL, 'vanished target', 'cites', datetime('now'))", (a,))
    conn.commit()

    hits = search(conn, "solo-token", all_repos=True, limit=5)
    assert hits[0].neighbours == 0


def test_a_failed_telemetry_bump_never_fails_the_search(env, monkeypatch):
    """The access_count / last_accessed bump is best-effort: a broken counter must not
    swallow the results the read already produced, and the failure is recorded as an
    event so the breakage is visible."""
    import akasha.knowledge as kb

    def broken_now():
        raise RuntimeError("telemetry clock is broken")

    monkeypatch.setattr(kb, "now", broken_now)
    hits = search(env[0], "cache", all_repos=True)
    assert hits, "the search result must still be returned"
    row = env[0].execute(
        "SELECT payload FROM events WHERE kind='knowledge.touch_failed' ORDER BY rowid DESC LIMIT 1"
    ).fetchone()
    assert row, "the swallowed failure must be visible as an event"
    assert "telemetry clock is broken" in row["payload"]


def test_link_candidates_finds_the_fold_target_written_in_different_vocabulary(env):
    """A whole-body query cannot find a near-duplicate written in different words once a
    long, broadly-worded document sits in the base. A short, focused query (title + first
    paragraph, `strategy="lede"`) still surfaces the real fold target past a
    length-rewarding distractor."""
    conn, cfg = env
    target_id = add_doc(
        conn, cfg, "handling transient failures with backoff",
        "## Backoff schedule\n"
        "When a call to a downstream dependency fails with a transient error, "
        "wait longer before each subsequent attempt instead of hammering it "
        "immediately again.\n\n"
        "## Jitter\n"
        "Add randomness to each wait so many callers do not retry at once.\n",
        repo="sample-repo", feature="f")
    # A long, broadly-worded distractor: no topical overlap, but wide generic
    # vocabulary -- the shape that made OR-mode BM25 reward length over topic.
    add_doc(
        conn, cfg, "a long design proposal touching many unrelated systems",
        "## Overview\n" + "system process request change desk shelf work service. " * 60,
        repo="sample-repo", feature="f")

    hits = link_candidates(
        conn, None,
        "why a flaky dependency needs a delay, not an instant repeat",
        "## The trouble with retrying immediately\n"
        "A dependency that occasionally times out gets worse, not better, when "
        "every caller notices the failure and fires the same request again "
        "right away -- the repeats themselves become the load spike.\n\n"
        "## Spacing attempts apart\n"
        "Each subsequent try should wait longer than the last.\n",
        limit=5, strategy="lede")

    assert target_id in [h.document_id for h in hits]


def test_link_candidates_default_finds_a_fold_target_whose_signal_is_past_the_lede(env):
    """The default query is the whole body (headings stripped), because the caller is a
    ranked list an agent reads and real near-duplicates share vocabulary throughout. A
    fold target whose shared vocabulary lives past the first paragraph is invisible to a
    lede query.

    The fixture speaks in rare terms only, deliberately: chunks_fts has no stopword
    removal, so a function word in the query matches every document that has one and its
    document frequency, not the topic, decides the rank."""
    conn, cfg = env
    target_id = add_doc(
        conn, cfg, "shard maintenance notes",
        "## Housekeeping\n"
        "Calendar hygiene memo, meeting-note conventions.\n\n"
        "## The actual procedure\n"
        "Quiescent shard rebalancing runbook: drain quiescent shard, rebalancing "
        "replicas inside rebalancing window, verify shard placement.\n",
        repo="sample-repo", feature="f")

    # Enough distractors that a top-5 is a real cut. They own the query's lede
    # vocabulary -- the terms a lede query would search with -- and none of them
    # touches the deep signal the two real documents share.
    for i in range(6):
        add_doc(
            conn, cfg, f"calibration handbook chapter {i}",
            "## Cycles\n"
            "Quorum ledger calibration cycles recalibrate ledger quorum.\n",
            repo="sample-repo", feature="f")

    query_title = "quorum ledger calibration worry"
    query_body = (
        "## The routine\n"
        "Routine quorum ledger calibration drift, recalibration lag, audit blindness.\n\n"
        "## What finally helped\n"
        "Quiescent shard rebalancing runbook: drain quiescent shard, rebalancing "
        "replicas inside rebalancing window, verify shard placement.\n")
    hits = link_candidates(conn, None, query_title, query_body, limit=5)

    assert target_id in [h.document_id for h in hits], (
        "the fold target shares its vocabulary only past the first paragraph and was "
        "not offered -- the default query is back to a lede shape")

    # The fixture must genuinely discriminate the two shapes, or the assertion above
    # proves nothing about the default.
    lede_hits = link_candidates(conn, None, query_title, query_body, limit=5,
                                strategy="lede")
    assert target_id not in [h.document_id for h in lede_hits], (
        "the lede query found a target whose shared signal is past its first "
        "paragraph -- the fixture no longer reaches the line under test")


def test_whole_body_query_reproduces_the_recorded_failure(env):
    """Guard for the design trade-off: a raw whole-body query, the default shape, fails
    the case `strategy="lede"` solves. The default accepts that cost on adversarial
    corpora in exchange for recall; this test notices if the cost disappears or the
    fixture regresses to the easy case. Asserts only that the true neighbour is not
    ranked first, the effect that reproduces reliably."""
    conn, cfg = env
    target_id = add_doc(
        conn, cfg, "handling transient failures with backoff",
        "## Backoff schedule\n"
        "When a call to a downstream dependency fails with a transient error, "
        "wait longer before each subsequent attempt instead of hammering it "
        "immediately again.\n\n"
        "## Jitter\n"
        "Add randomness to each wait so many callers do not retry at once.\n",
        repo="sample-repo", feature="f")
    # Broad, cross-topic vocabulary rather than one repeated phrase: breadth is what
    # makes BM25-OR reward length over topic, while a repeated phrase saturates.
    filler_paragraphs = [
        "One system calls a downstream dependency that occasionally fails with a "
        "transient error, and callers were retrying immediately and repeatedly; "
        "the fix agreed on was to wait longer before each subsequent attempt, add "
        "jitter so many callers do not retry in the same instant, and give up "
        "after a fixed number of attempts.",
        "A second system had a read cache that was not cleared when its "
        "underlying record changed and kept serving the old value well past the "
        "write that made it wrong; the fix was for the write path to invalidate "
        "the cache entry for that key.",
        "A third system still exported its nightly report as one monolithic job "
        "months after the data outgrew it; the recommendation was the usual "
        "incremental approach -- small batches first, compare totals before "
        "widening, full cutover, then remove the monolithic job.",
        "A fourth system's database connection pool was sized for last year's "
        "traffic: too small, so requests queue waiting for a free connection; "
        "too large, and the pressure just lands on the database's own ceiling.",
        "A fifth system's log file was rotating on a schedule far looser than "
        "its actual growth rate, slowly filling the disk it ran on; moving the "
        "current file aside on size or age fixed it.",
    ]
    for i, para in enumerate(filler_paragraphs):
        add_doc(
            conn, cfg, f"a quarterly shelving inventory, item {i}",
            f"## Overview\n{para}\n",
            repo="sample-repo", feature="f")

    whole_body_query = (
        "why a flaky dependency needs a delay, not an instant repeat\n"
        "The trouble with retrying immediately\n"
        "A dependency that occasionally times out gets worse, not better, when "
        "every caller notices the failure and fires the same request again "
        "right away -- the repeats themselves become the load spike.\n"
        "Spacing attempts apart\n"
        "Each subsequent try should wait longer than the last.\n")
    hits = search(conn, whole_body_query, all_repos=True, limit=5, cfg=None)

    assert hits, "the query should still match something"
    assert hits[0].document_id != target_id, (
        "whole-body query ranked the fold target first -- either the fixture "
        "regressed to the easy case, or BM25-OR stopped rewarding length in this "
        "corpus; either way this test needs a look before trusting recall numbers")


def test_link_candidates_fetches_limit_plus_one_so_self_cannot_starve_the_result(env):
    """exclude_id filters the caller's own document out of the results. Filtering a
    limit-sized fetch after the fact would return one candidate short whenever the
    self-document scores inside the top `limit`."""
    conn, cfg = env
    self_id = add_doc(conn, cfg, "Session guide",
                     "## A\nsession token shared here\n", repo="sample-repo", feature="f")
    for i in range(5):
        add_doc(conn, cfg, f"Session note {i}",
              f"## B\nsession token shared here also {i}\n", repo="sample-repo", feature="f")

    hits = link_candidates(conn, None, "Session guide", "session token shared here",
                            limit=5, exclude_id=self_id)

    assert len(hits) == 5
    assert self_id not in [h.document_id for h in hits]


def test_a_replacement_outranks_the_document_it_superseded(env):
    """A superseded document may surface with include_archived, but never above the
    document that replaced it: whoever wrote `supersedes` already said which of the two
    is true, and an agent reads the first hit.

    The predecessor is built to win on relevance alone — term in the heading, three
    times in a short body — so only the supersession edge can put the replacement first.
    """
    conn, _ = env
    conn.execute(
        "INSERT INTO documents (id, source, path, title, repo, status, created_at, updated_at)"
        " VALUES ('d_old','native','/tmp/old.md','Old','sample-repo','archived',"
        " datetime('now'), datetime('now'))")
    conn.execute(
        "INSERT INTO chunks (id, document_id, heading, body, ord)"
        " VALUES ('c_old','d_old','## Slotting','slotting slotting slotting',0)")
    conn.execute(
        "INSERT INTO documents (id, source, path, title, repo, status, supersedes,"
        " created_at, updated_at)"
        " VALUES ('d_new','native','/tmp/new.md','New','sample-repo','active','[\"d_old\"]',"
        " datetime('now'), datetime('now'))")
    conn.execute(
        "INSERT INTO chunks (id, document_id, heading, body, ord) VALUES"
        " ('c_new','d_new','## Correction',?,0)",
        ("the earlier note about slotting was wrong " + "filler words here " * 40,))
    conn.commit()

    hits = search(conn, "slotting", all_repos=True, include_archived=True, limit=5)
    order = [h.document_id for h in hits]
    assert {"d_old", "d_new"} <= set(order), order
    assert order.index("d_new") < order.index("d_old"), order


def test_promoting_a_replacement_leaves_every_other_hit_where_it_was(env):
    """The promotion is one pair moving, not a re-rank: reordering around a supersession
    edge must not drop a hit, duplicate one, or disturb documents that carry no edge."""
    conn, _ = env
    before = search(conn, "cache", all_repos=True, include_archived=True, limit=5)
    assert len(before) > 1

    conn.execute(
        "INSERT INTO documents (id, source, path, title, repo, status, supersedes,"
        " created_at, updated_at)"
        " VALUES ('d_repl','native','/tmp/repl.md','Repl','sample-repo','active','[\"nope\"]',"
        " datetime('now'), datetime('now'))")
    conn.execute(
        "INSERT INTO chunks (id, document_id, heading, body, ord) VALUES"
        " ('c_repl','d_repl','## Cache','the cache note that supersedes a document which"
        " is not in this result set',0)")
    conn.commit()

    after = search(conn, "cache", all_repos=True, include_archived=True, limit=5)
    assert len(after) == len(set(h.document_id for h in after))
    # A supersedes target that never matched the query changes nothing about the order.
    assert [h.document_id for h in after if h.document_id != "d_repl"] == \
        [h.document_id for h in before]


def test_a_document_at_a_status_nothing_writes_stays_out_of_search(env):
    """The filter is an allowlist of the statuses something writes — active, unsorted,
    stale, and archived when asked — so a row at any other status is unreachable. Such a
    row comes from a markdown file whose frontmatter carries a hand-typed status."""
    conn, _ = env
    conn.execute("UPDATE documents SET status='superseded' WHERE path LIKE '%web%'")
    conn.commit()

    assert all("web" not in h.path for h in search(conn, "cache", all_repos=True))
    assert all("web" not in h.path
               for h in search(conn, "cache", all_repos=True, include_archived=True))


def test_an_unknown_status_fails_closed_rather_than_ranking_at_full_weight(env):
    """An unrecognised status must not pass the filter and score at the default weight: a
    typo in one document's frontmatter would rank it above every correctly-marked one."""
    conn, _ = env
    conn.execute("UPDATE documents SET status='activ' WHERE path LIKE '%web%'")
    conn.commit()

    assert all("web" not in h.path for h in search(conn, "cache", all_repos=True))


def test_the_caller_shape_omits_default_valued_fields_and_the_path(env):
    """Every hit used to repeat relaxed=false, cross_repo=false, a zero count and a null
    feature, plus a path knowledge_get already returns: context spent on nothing."""
    conn, cfg = env
    add_doc(conn, cfg, "Plain", "## A\nplainshape token\n", repo="sample-repo", feature="")
    hit = search(conn, "plainshape", all_repos=True)[0]

    shape = hit.as_dict()
    assert shape == {"id": hit.id, "document_id": hit.document_id,
                     "score": round(hit.score, 2), "repo": "sample-repo",
                     "heading": "## A", "text": hit.text}
    assert hit.path, "in-process callers still read the path from the hit itself"


def test_the_caller_shape_keeps_a_field_that_carries_information(env):
    conn, cfg = env
    add_doc(conn, cfg, "Marked", "## A\nmarkedshape token\n", repo="sample-repo", feature="f1")
    hit = search(conn, "markedshape", repo="elsewhere")[0]

    shape = hit.as_dict(withheld_chars=7)
    assert shape["cross_repo"] is True
    assert shape["feature"] == "f1"
    assert shape["withheld_chars"] == 7
    assert "relaxed" not in shape and "neighbours" not in shape
    assert "path" not in shape


def test_as_dict_omits_repo_when_there_is_none():
    """A null field costs the caller tokens and says nothing."""
    from akasha.search import Hit

    bare = Hit(id="c1", document_id="d1", score=1.0, repo=None, feature=None,
               path="/p.md", heading="h", text="t")
    assert "repo" not in bare.as_dict()
    assert Hit(id="c1", document_id="d1", score=1.0, repo="sample-repo", feature=None,
               path="/p.md", heading="h", text="t").as_dict()["repo"] == "sample-repo"


def test_an_unknown_match_mode_is_refused(tmp_path):
    """A typo in match_mode must not silently fall through to a different strategy."""
    from akasha.db import connect

    with pytest.raises(ValueError, match="match_mode"):
        search(connect(tmp_path / "s.db"), "x", match_mode="fuzzy")


@pytest.mark.parametrize("text", ["20240101", "2024-1-1", "garbage", "2024-01-01T00:00"])
def test_as_of_must_be_a_strict_iso_date(text, tmp_path):
    """The comparison against invalid_at is textual, so only YYYY-MM-DD compares right."""
    from akasha.db import connect

    with pytest.raises(ValueError, match="as_of"):
        search(connect(tmp_path / "s.db"), "x", as_of=text)


def test_lexical_hits_keep_a_visible_score_on_a_small_corpus(tmp_path):
    """BM25 on a few short documents is tiny; rounding it to 2 places showed a real
    match as 0.0, which reads as no match."""

    cfg = load_config(tmp_path / "absent.toml")
    cfg.knowledge_dir = tmp_path / "k"
    cfg.knowledge_dir.mkdir()
    cfg.index_roots = []
    conn = connect(tmp_path / "s.db")
    add_doc(conn, cfg, "Cache notes", "## Why\nthe session cache was empty")
    add_doc(conn, cfg, "Queue notes", "## Why\nthe queue worker runs alone")
    hits = search(conn, "cache", cfg=cfg)
    assert hits and hits[0].score > 0
    assert hits[0].as_dict()["score"] > 0
