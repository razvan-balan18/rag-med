"""FAISS IndexFlatIP build over chunk embeddings (Q23d, week2 day3).

End-to-end indexing stage: read ``chunks.text`` ordered by ``chunk_id``,
embed via MedCPT-Article, L2-normalize (inner product on unit vectors ==
cosine), build a flat exact index, persist alongside a sidecar JSON that
maps FAISS row idx → ``chunk_id`` (FAISS stores only int row positions).

The embedder is injected (``embed_fn``) so unit tests feed deterministic
vectors and never touch torch; the CLI binds the real ``embed.embed_chunks``.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from pathlib import Path

import faiss
import numpy as np
import structlog

from rag_med.indexing.embed import EMBED_DIM, embed_chunks

log = structlog.get_logger()

EmbedFn = Callable[[list[str]], np.ndarray]


def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    """Unit-length each row; inner product then equals cosine similarity.

    Zero vectors stay zero (guard the divide); they score 0 against anything.

    unit vectors so IP == cosine
    """
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vecs / norms).astype(np.float32)


def build_index(texts: list[str], *, embed_fn: EmbedFn = embed_chunks) -> faiss.Index:
    """Embed ``texts``, L2-normalize, add to a fresh ``IndexFlatIP(768)``.

    embed -> normalize -> indexflatip.add
    """
    index = faiss.IndexFlatIP(EMBED_DIM)
    if not texts:
        return index
    vecs = l2_normalize(embed_fn(texts))
    index.add(vecs)
    return index


def write_index(
    index: faiss.Index,
    chunk_ids: list[str],
    index_path: Path,
    sidecar_path: Path,
) -> None:
    """Persist the index + the ordered row-idx → ``chunk_id`` sidecar JSON.

    guards len(chunk_ids) == ntotal
    """
    if len(chunk_ids) != index.ntotal:
        raise ValueError(f"chunk_ids ({len(chunk_ids)}) != index.ntotal ({index.ntotal})")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    sidecar_path.write_text(json.dumps(chunk_ids))


def read_index(index_path: Path, sidecar_path: Path) -> tuple[faiss.Index, list[str]]:
    """Load the index + sidecar. Returns ``(index, chunk_ids)``.

    load both back
    """
    index = faiss.read_index(str(index_path))
    chunk_ids = json.loads(sidecar_path.read_text())
    return index, chunk_ids


def run_embed(
    *,
    conn: sqlite3.Connection,
    index_path: Path,
    sidecar_path: Path,
    embed_fn: EmbedFn = embed_chunks,
) -> dict[str, int]:
    """
    orchestrator
    Read every chunk ordered by ``chunk_id``, embed, build + persist FAISS.

    Whole-corpus rebuild (not incremental) — the toy corpus is ~2k chunks and
    a flat index has no merge story. Returns ``{"chunks": N}``.
    """
    rows = conn.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id").fetchall()
    chunk_ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]

    index = build_index(texts, embed_fn=embed_fn)
    write_index(index, chunk_ids, index_path, sidecar_path)
    log.info("run_embed_done", n_chunks=index.ntotal, index_path=str(index_path))
    return {"chunks": index.ntotal}
