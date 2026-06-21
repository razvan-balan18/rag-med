"""Retrieval skeleton tests (week2 day5) — mock everything.

No real models, no FAISS file, no BM25 pickle, no torch, no DB. Every seam
(`_load_faiss`, `_load_bm25`, `_embed_query`, `_cross_encode`, `_chunk_text_map`,
`_load_chunks`) is monkeypatched. Tests exercise observable behavior through the
public stage functions + `retrieve`.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np

from rag_med.serving import retrieve as r
from rag_med.shared.models import Chunk


class _FakeFaiss:
    """Stand-in FAISS index: `.search` returns canned (scores, idxs) 2D arrays."""

    def __init__(self, scores: list[float], idxs: list[int]):
        self._scores = np.array([scores], dtype=np.float32)
        self._idxs = np.array([idxs], dtype=np.int64)

    def search(self, q, top_k):
        return self._scores[:, :top_k], self._idxs[:, :top_k]


class _FakeBM25:
    """Stand-in BM25: `.get_scores` returns a fixed per-doc score array."""

    def __init__(self, scores: list[float]):
        self._scores = np.array(scores, dtype=float)

    def get_scores(self, tokens):
        return self._scores


def _chunk(cid: str, section: str = "methods") -> Chunk:
    return Chunk(
        chunk_id=cid,
        pmid="111",
        section_type=section,
        ordinal=0,
        text=f"text-{cid}",
        n_deberta_tokens=1,
        n_medcpt_tokens=1,
    )


def test_fuse_rrf_math_chunk_in_both_lists():
    # chunk "A" is rank 1 in both lists -> 1/(60+1) + 1/(60+1) = 2/61
    dense = [("A", 9.0), ("B", 8.0)]
    lexical = [("A", 5.0), ("C", 4.0)]
    fused = dict(r.fuse(dense, lexical, rrf_k=60))
    assert fused["A"] == 2 / 61


def test_fuse_rrf_math_single_list_rank():
    # chunk only at rank 5 of dense -> 1/(60+5)
    dense = [("z1", 9), ("z2", 8), ("z3", 7), ("z4", 6), ("E", 5)]
    lexical = [("other", 1)]
    fused = dict(r.fuse(dense, lexical, rrf_k=60))
    assert fused["E"] == 1 / 65


def test_fuse_dedupes_and_sorts_desc():
    dense = [("A", 9.0), ("B", 8.0)]
    lexical = [("A", 5.0)]  # A appears in both -> summed, listed once
    fused = r.fuse(dense, lexical, rrf_k=60)
    ids = [cid for cid, _ in fused]
    assert ids.count("A") == 1
    # A (2/61) outranks B (1/61); list is sorted by rrf_score desc
    assert ids[0] == "A"
    assert [s for _, s in fused] == sorted((s for _, s in fused), reverse=True)


# --------------------------------------------------------------------------- #
# dense_search / lexical_search
# --------------------------------------------------------------------------- #


def test_dense_search_returns_top_k_in_score_order(monkeypatch):
    chunk_ids = ["c0", "c1", "c2", "c3"]
    # FAISS hands back rows already sorted by descending IP score.
    fake = _FakeFaiss(scores=[0.5, 0.25, 0.125, 0.0625], idxs=[0, 1, 2, 3])
    monkeypatch.setattr(r, "_load_faiss", lambda: (fake, chunk_ids))

    hits = r.dense_search(np.zeros(r.EMBED_DIM, dtype=np.float32), top_k=3)

    assert [cid for cid, _ in hits] == ["c0", "c1", "c2"]
    assert [s for _, s in hits] == [0.5, 0.25, 0.125]


def test_dense_search_skips_negative_padding_idx(monkeypatch):
    chunk_ids = ["c0", "c1"]
    fake = _FakeFaiss(scores=[0.5, 0.25, 0.0], idxs=[0, 1, -1])  # -1 = FAISS pad
    monkeypatch.setattr(r, "_load_faiss", lambda: (fake, chunk_ids))

    hits = r.dense_search(np.zeros(r.EMBED_DIM, dtype=np.float32), top_k=3)

    assert [cid for cid, _ in hits] == ["c0", "c1"]


def test_lexical_search_returns_top_k_in_score_order(monkeypatch):
    chunk_ids = ["d0", "d1", "d2", "d3"]
    monkeypatch.setattr(r, "_load_bm25", lambda: (_FakeBM25([0.1, 5.0, 2.0, 3.0]), chunk_ids))

    hits = r.lexical_search(["copd"], top_k=2)

    assert [cid for cid, _ in hits] == ["d1", "d3"]  # 5.0 then 3.0


# --------------------------------------------------------------------------- #
# rerank
# --------------------------------------------------------------------------- #


def test_rerank_returns_top_k_by_rerank_score(monkeypatch):
    fused = [("a", 0.10), ("b", 0.09), ("c", 0.08), ("d", 0.07)]
    monkeypatch.setattr(r, "_chunk_text_map", lambda ids: {i: f"text-{i}" for i in ids})
    monkeypatch.setattr(r, "_cross_encode", lambda q, texts: np.array([0.4, 0.9, 0.1, 0.6]))

    out = r.rerank("q", fused, top_k=2)

    assert out == [("b", 0.9), ("d", 0.6)]


def test_rerank_empty_candidates_short_circuits(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("cross-encoder must not run on empty candidates")

    monkeypatch.setattr(r, "_cross_encode", _boom)
    assert r.rerank("q", []) == []


# --------------------------------------------------------------------------- #
# embed — LRU cache
# --------------------------------------------------------------------------- #


def test_embed_lru_cache_skips_second_call(monkeypatch):
    r.embed.cache_clear()
    calls: list[str] = []

    def _fake_embed_query(question):
        calls.append(question)
        return np.ones(r.EMBED_DIM, dtype=np.float32)

    monkeypatch.setattr(r, "_embed_query", _fake_embed_query)

    v1 = r.embed("same question")
    v2 = r.embed("same question")

    assert len(calls) == 1  # second call served from cache
    assert np.array_equal(v1, v2)
    r.embed.cache_clear()


def test_embed_normalizes_to_unit_length(monkeypatch):
    r.embed.cache_clear()
    monkeypatch.setattr(r, "_embed_query", lambda q: np.full(r.EMBED_DIM, 3.0, dtype=np.float32))

    vec = r.embed("q")

    assert np.isclose(float(np.linalg.norm(vec)), 1.0)
    r.embed.cache_clear()


# --------------------------------------------------------------------------- #
# retrieve — rerank_floor (two effects) + histogram
# --------------------------------------------------------------------------- #


def _patch_retrieve(monkeypatch, *, reranked, floor, chunks):
    """Wire `retrieve`'s collaborators: stage outputs + floor + chunk load."""
    monkeypatch.setattr(r, "embed", lambda q: np.zeros(r.EMBED_DIM, dtype=np.float32))
    monkeypatch.setattr(r, "dense_search", lambda v: [])
    monkeypatch.setattr(r, "lexical_search", lambda toks: [])
    monkeypatch.setattr(r, "rerank", lambda q, cands, top_k=r.DEFAULT_TOP_K: reranked)
    monkeypatch.setattr(r, "get_settings", lambda: SimpleNamespace(rerank_floor=floor))
    monkeypatch.setattr(r, "_load_chunks", lambda ids: [c for c in chunks if c.chunk_id in ids])


def test_retrieve_floor_drops_per_chunk_below_threshold(monkeypatch):
    reranked = [("a", 0.9), ("b", 0.6), ("c", 0.4), ("d", 0.3)]
    chunks = [_chunk("a"), _chunk("b"), _chunk("c"), _chunk("d")]
    _patch_retrieve(monkeypatch, reranked=reranked, floor=0.5, chunks=chunks)

    res = asyncio.run(r.retrieve("q"))

    assert res.hard_refusal is False
    assert len(res.final_chunks) == 2
    assert [c.chunk_id for c in res.final_chunks] == ["a", "b"]
    assert res.rerank_scores == [0.9, 0.6]


def test_retrieve_hard_refusal_when_top1_below_floor(monkeypatch):
    reranked = [("a", 0.4), ("b", 0.3)]
    _patch_retrieve(monkeypatch, reranked=reranked, floor=0.5, chunks=[_chunk("a"), _chunk("b")])

    res = asyncio.run(r.retrieve("q"))

    assert res.hard_refusal is True
    assert res.final_chunks == []
    assert res.rerank_scores == []
    assert res.section_type_histogram == {}


def test_retrieve_section_type_histogram(monkeypatch):
    reranked = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    chunks = [_chunk("a", "methods"), _chunk("b", "methods"), _chunk("c", "results")]
    _patch_retrieve(monkeypatch, reranked=reranked, floor=0.0, chunks=chunks)

    res = asyncio.run(r.retrieve("q"))

    assert res.section_type_histogram == {"methods": 2, "results": 1}
