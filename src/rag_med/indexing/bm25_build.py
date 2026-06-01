"""rank_bm25 inverted-index build over chunk text (Q7, week2 day4).

End-to-end indexing stage mirroring ``faiss_build``: read ``chunks.text``
ordered by ``chunk_id``, tokenize each with the shared biomedical tokenizer,
build a ``BM25Okapi`` index, and persist it as a pickle alongside a sidecar
JSON mapping BM25 doc idx -> ``chunk_id`` (BM25Okapi stores positional docs
only, like FAISS row idx).

SECURITY (architecture.md §11.1): ``data/bm25.pkl`` is pickle. It is built by
this pipeline and loaded only by our own server. NEVER unpickle a ``bm25.pkl``
from any external source — pickle load executes arbitrary code.

The tokenizer is injected (``tokenize_fn``) so unit tests stay off the regex
internals; the CLI binds the real ``shared.tokenize.bm25_tokenize``.
"""

from __future__ import annotations

import json
import pickle
import sqlite3
from collections.abc import Callable
from pathlib import Path

import structlog
from rank_bm25 import BM25Okapi

from rag_med.shared.tokenize import bm25_tokenize

log = structlog.get_logger()

TokenizeFn = Callable[[str], list[str]]


def build_index(texts: list[str], *, tokenize_fn: TokenizeFn = bm25_tokenize) -> BM25Okapi:
    """Tokenize ``texts`` and build a ``BM25Okapi`` over the corpus.

    tokenize each -> BM25Okapi(corpus_tokens)
    """
    corpus_tokens = [tokenize_fn(t) for t in texts]
    return BM25Okapi(corpus_tokens)


def write_index(
    bm25: BM25Okapi,
    chunk_ids: list[str],
    index_path: Path,
    sidecar_path: Path,
) -> None:
    """Pickle the index + write the ordered doc-idx -> ``chunk_id`` sidecar.

    guards len(chunk_ids) == corpus_size
    """
    if len(chunk_ids) != bm25.corpus_size:
        raise ValueError(f"chunk_ids ({len(chunk_ids)}) != bm25.corpus_size ({bm25.corpus_size})")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("wb") as f:
        pickle.dump(bm25, f)
    sidecar_path.write_text(json.dumps(chunk_ids))


def read_index(index_path: Path, sidecar_path: Path) -> tuple[BM25Okapi, list[str]]:
    """Load the index + sidecar. Returns ``(bm25, chunk_ids)``.

    TRUSTED-INPUT ONLY — see module docstring. Never call on a file from
    outside our own pipeline.
    """
    with index_path.open("rb") as f:
        bm25 = pickle.load(f)  # noqa: S301 — trusted, our-own-pipeline artifact
    chunk_ids = json.loads(sidecar_path.read_text())
    return bm25, chunk_ids


def run_bm25(
    *,
    conn: sqlite3.Connection,
    index_path: Path,
    sidecar_path: Path,
    tokenize_fn: TokenizeFn = bm25_tokenize,
) -> dict[str, int]:
    """
    orchestrator
    Read every chunk ordered by ``chunk_id``, tokenize, build + persist BM25.

    Whole-corpus rebuild (not incremental) — BM25Okapi precomputes IDF over
    the full corpus, so there is no add-one story. Returns ``{"chunks": N}``.
    """
    rows = conn.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id").fetchall()
    chunk_ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]

    bm25 = build_index(texts, tokenize_fn=tokenize_fn)
    write_index(bm25, chunk_ids, index_path, sidecar_path)
    log.info("run_bm25_done", n_chunks=bm25.corpus_size, index_path=str(index_path))
    return {"chunks": bm25.corpus_size}
