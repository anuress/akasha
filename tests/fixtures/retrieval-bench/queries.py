"""The retrieval benchmark's query set.

Ground truth is by document title: document ids are assigned at index time, so a
committed query set can only name a document by the thing that is stable.

The first entry is the paraphrase case. "shelving rules" talks about the sorting cart
and Dewey order and shares almost no vocabulary with the query, so lexical BM25 ranks
it low while dense retrieval, and therefore fusion, finds it: the case that justifies
fusion.

Each other query shares a rare identifier or distinctive phrase with its target, so BM25
should rank most targets first.
"""
from __future__ import annotations

QUERIES: list[tuple[str, str]] = [
    # Paraphrase case: the query shares almost no words with the target's body.
    ("who puts books back in their proper place", "shelving rules"),
    ("fine calculator daily rate cap", "overdue fines"),
    ("hold queue pickup window", "hold queue"),
    ("renewal policy limit twice", "renewal limits"),
    ("membership tier loan length", "membership tiers"),
    ("facet counts author subject", "search facets"),
    ("room booking overlap slot", "room reservations"),
    ("reminder digest due soon", "reminder emails"),
    ("isbn check digit validation", "isbn validation"),
    ("bulk import batch rollback", "bulk import"),
    ("duplicate record merge candidates", "duplicate records"),
    ("author name authority variants", "author authority"),
    ("metadata export marc csv", "metadata export"),
    ("barcode label sheet printing", "barcode labels"),
    ("inventory audit missing items", "inventory audit"),
]
