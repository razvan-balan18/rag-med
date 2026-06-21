"""Retrieval pipeline — stages 1–5 of architecture.md §3.1.

    question
      ├─ embed (MedCPT-Query) → dense_search (FAISS) ─┐
      │                                               ├─ fuse (RRF) → rerank → top-k
      └─ bm25_tokenize → lexical_search (BM25) ───────┘

Phase 2 (serving): must not import `rag_med.indexing` — FAISS/BM25 are read via
the third-party libs directly; `Chunk` comes from `shared/`. Every model/FAISS/
BM25/DB access sits behind a `_load_*`/`_embed_query`/`_cross_encode` seam so
unit tests monkeypatch them and never touch torch/faiss/sqlite.
"""

from __future__ import annotations

import asyncio
import json
import pickle
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import structlog

from rag_med.config import get_settings
from rag_med.shared.db import connect
from rag_med.shared.models import Chunk
from rag_med.shared.tokenize import bm25_tokenize

log = structlog.get_logger()

QUERY_MODEL_NAME = "ncbi/MedCPT-Query-Encoder"
CROSS_ENCODER_NAME = "ncbi/MedCPT-Cross-Encoder"
EMBED_DIM = 768
MAX_TOKENS = 512

RRF_K = 60
DENSE_TOP_K = 50
LEXICAL_TOP_K = 50
RERANK_CANDIDATES = 50
DEFAULT_TOP_K = 10

Hit = tuple[str, float]  # (chunk_id, score)


@dataclass
class RetrievalResult:
    final_chunks: list[Chunk]
    rerank_scores: list[float]
    hard_refusal: bool
    section_type_histogram: dict[str, int]


_query_model = None
_query_tokenizer = None
_cross_model = None
_cross_tokenizer = None
_faiss: tuple[object, list[str]] | None = None
_bm25: tuple[object, list[str]] | None = None


def _device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    log.warning("mps_unavailable_cpu_fallback")
    return "cpu"


def _load_query_encoder():
    global _query_model, _query_tokenizer
    if _query_model is None:
        import torch  # noqa: F401
        from transformers import AutoModel, AutoTokenizer

        _query_tokenizer = AutoTokenizer.from_pretrained(QUERY_MODEL_NAME)
        _query_model = AutoModel.from_pretrained(QUERY_MODEL_NAME).to(_device()).eval()
    return _query_tokenizer, _query_model


def _load_cross_encoder():
    global _cross_model, _cross_tokenizer
    if _cross_model is None:
        import torch  # noqa: F401
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        _cross_tokenizer = AutoTokenizer.from_pretrained(CROSS_ENCODER_NAME)
        _cross_model = (
            AutoModelForSequenceClassification.from_pretrained(CROSS_ENCODER_NAME)
            .to(_device())
            .eval()
        )
    return _cross_tokenizer, _cross_model


def _load_faiss() -> tuple[object, list[str]]:
    global _faiss
    if _faiss is None:
        import faiss

        settings = get_settings()
        index = faiss.read_index(str(settings.faiss_index_path))
        chunk_ids = json.loads(settings.faiss_chunk_ids_path.read_text())
        _faiss = (index, chunk_ids)
    return _faiss


def _load_bm25() -> tuple[object, list[str]]:
    # TRUSTED-INPUT ONLY: bm25.pkl is built by our own Phase-1 pipeline and
    # loaded only here. Never unpickle one from an external source.
    global _bm25
    if _bm25 is None:
        settings = get_settings()
        with settings.bm25_path.open("rb") as f:
            bm25 = pickle.load(f)  # noqa: S301 — trusted, our-own-pipeline artifact
        chunk_ids = json.loads(settings.bm25_chunk_ids_path.read_text())
        _bm25 = (bm25, chunk_ids)
    return _bm25


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def _embed_query(question: str) -> np.ndarray:
    import torch

    tokenizer, model = _load_query_encoder()
    enc = tokenizer(
        [question],
        truncation=True,
        max_length=MAX_TOKENS,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        vec = model(**enc).last_hidden_state[:, 0, :]
    return vec.cpu().numpy()[0].astype(np.float32)


def _cross_encode(question: str, texts: list[str]) -> np.ndarray:
    import torch

    tokenizer, model = _load_cross_encoder()
    pairs = [[question, t] for t in texts]
    enc = tokenizer(
        pairs,
        truncation=True,
        max_length=MAX_TOKENS,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        logits = model(**enc).logits.squeeze(-1)
    return logits.cpu().numpy().reshape(-1).astype(np.float32)


def _chunk_text_map(chunk_ids: list[str]) -> dict[str, str]:
    if not chunk_ids:
        return {}
    settings = get_settings()
    conn = connect(settings.sqlite_path)
    try:
        placeholders = ",".join("?" * len(chunk_ids))
        rows = conn.execute(
            f"SELECT chunk_id, text FROM chunks WHERE chunk_id IN ({placeholders})",  # noqa: S608
            chunk_ids,
        ).fetchall()
    finally:
        conn.close()
    return {cid: text for cid, text in rows}


def _load_chunks(chunk_ids: list[str]) -> list[Chunk]:
    """Resolve chunk_ids → Chunk objects, preserving input order."""
    if not chunk_ids:
        return []
    settings = get_settings()
    conn = connect(settings.sqlite_path)
    try:
        placeholders = ",".join("?" * len(chunk_ids))
        rows = conn.execute(
            "SELECT chunk_id, pmid, section_type, ordinal, text, "
            "n_deberta_tokens, n_medcpt_tokens "
            f"FROM chunks WHERE chunk_id IN ({placeholders})",  # noqa: S608
            chunk_ids,
        ).fetchall()
    finally:
        conn.close()
    by_id = {row[0]: Chunk(*row) for row in rows}
    return [by_id[cid] for cid in chunk_ids if cid in by_id]


@lru_cache(maxsize=1000)
def embed(question: str) -> np.ndarray:
    return _l2_normalize(_embed_query(question))


def dense_search(query_vector: np.ndarray, top_k: int = DENSE_TOP_K) -> list[Hit]:
    index, chunk_ids = _load_faiss()
    q = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
    scores, idxs = index.search(q, top_k)
    hits: list[Hit] = []
    for score, idx in zip(scores[0], idxs[0], strict=False):
        if idx < 0:  # FAISS pads with -1 when fewer than top_k vectors exist
            continue
        hits.append((chunk_ids[idx], float(score)))
    return hits


def lexical_search(bm25_tokens: list[str], top_k: int = LEXICAL_TOP_K) -> list[Hit]:
    bm25, chunk_ids = _load_bm25()
    scores = np.asarray(bm25.get_scores(bm25_tokens))
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(chunk_ids[i], float(scores[i])) for i in top_idx]


def fuse(dense_hits: list[Hit], lexical_hits: list[Hit], rrf_k: int = RRF_K) -> list[Hit]:
    """RRF (Q8): Σ 1/(rrf_k + rank) across both lists, deduped, score-desc."""
    rrf_scores: dict[str, float] = {}
    for hits in (dense_hits, lexical_hits):
        for rank, (chunk_id, _score) in enumerate(hits, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)


def rerank(
    question: str,
    fused_candidates: list[Hit],
    top_k: int = DEFAULT_TOP_K,
) -> list[Hit]:
    if not fused_candidates:
        return []
    chunk_ids = [cid for cid, _ in fused_candidates]
    text_map = _chunk_text_map(chunk_ids)
    texts = [text_map.get(cid, "") for cid in chunk_ids]
    scores = _cross_encode(question, texts)
    ranked = sorted(
        ((cid, float(s)) for cid, s in zip(chunk_ids, scores, strict=False)),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return ranked[:top_k]


def _section_histogram(chunks: list[Chunk]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for ch in chunks:
        hist[ch.section_type] = hist.get(ch.section_type, 0) + 1
    return hist


async def retrieve(question: str, top_k: int = DEFAULT_TOP_K) -> RetrievalResult:
    """embed → (dense ∥ lexical) → fuse → rerank → rerank_floor.

    dense_search ∥ lexical_search run via asyncio.gather + to_thread so day-7
    SSE streams without a refactor. rerank_floor (Q9b) gates both the hard-
    refusal short-circuit and per-chunk drops.
    """
    query_vector = embed(question)
    bm25_tokens = bm25_tokenize(question)

    dense_hits, lexical_hits = await asyncio.gather(
        asyncio.to_thread(dense_search, query_vector),
        asyncio.to_thread(lexical_search, bm25_tokens),
    )

    fused_candidates = fuse(dense_hits, lexical_hits)[:RERANK_CANDIDATES]
    reranked = rerank(question, fused_candidates, top_k=top_k)

    floor = get_settings().rerank_floor

    if not reranked or reranked[0][1] < floor:
        log.info("retrieve_hard_refusal", question_len=len(question))
        return RetrievalResult([], [], hard_refusal=True, section_type_histogram={})

    kept = [(cid, score) for cid, score in reranked if score >= floor]
    chunks = _load_chunks([cid for cid, _ in kept])
    return RetrievalResult(
        final_chunks=chunks,
        rerank_scores=[score for _, score in kept],
        hard_refusal=False,
        section_type_histogram=_section_histogram(chunks),
    )
