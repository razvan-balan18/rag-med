"""FAISS IndexFlatIP build tests (week2 day3) — mock the embedder.

No real model, no torch. ``build_index`` takes an injected ``embed_fn`` so
tests feed deterministic vectors. Round-trip + self-query prove the index +
sidecar mapping survive write→read intact.
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np

from rag_med.indexing import faiss_build
from rag_med.shared.db import init_schema


def _fake_embed_factory(vecs: np.ndarray):
    """Return an embed_fn that ignores its texts and yields ``vecs``."""

    def _embed(texts, batch_size=32):
        return vecs[: len(texts)].astype(np.float32)

    return _embed


def test_build_index_size_and_dim():
    vecs = np.random.rand(10, 768).astype(np.float32)
    index = faiss_build.build_index(["t"] * 10, embed_fn=_fake_embed_factory(vecs))

    assert index.ntotal == 10
    assert index.d == 768


def test_write_read_roundtrip_identical_search(tmp_path):
    vecs = np.random.rand(10, 768).astype(np.float32)
    chunk_ids = [f"111_methods_{i:02d}" for i in range(10)]
    index = faiss_build.build_index(["t"] * 10, embed_fn=_fake_embed_factory(vecs))

    index_path = tmp_path / "faiss.index"
    sidecar_path = tmp_path / "faiss.chunk_ids.json"
    faiss_build.write_index(index, chunk_ids, index_path, sidecar_path)

    index2, chunk_ids2 = faiss_build.read_index(index_path, sidecar_path)

    q = faiss_build.l2_normalize(vecs[:1])
    scores1, idxs1 = index.search(q, 5)
    scores2, idxs2 = index2.search(q, 5)
    assert np.array_equal(idxs1, idxs2)
    assert np.allclose(scores1, scores2)
    assert chunk_ids2 == chunk_ids


def test_sidecar_length_matches_index(tmp_path):
    vecs = np.random.rand(7, 768).astype(np.float32)
    chunk_ids = [f"222_results_{i:02d}" for i in range(7)]
    index = faiss_build.build_index(["t"] * 7, embed_fn=_fake_embed_factory(vecs))

    index_path = tmp_path / "faiss.index"
    sidecar_path = tmp_path / "faiss.chunk_ids.json"
    faiss_build.write_index(index, chunk_ids, index_path, sidecar_path)

    sidecar = json.loads(sidecar_path.read_text())
    assert len(sidecar) == index.ntotal == 7


def test_self_query_returns_own_chunk_id_score_one(tmp_path):
    vecs = np.random.rand(10, 768).astype(np.float32)
    chunk_ids = [f"333_discussion_{i:02d}" for i in range(10)]
    index = faiss_build.build_index(["t"] * 10, embed_fn=_fake_embed_factory(vecs))

    index_path = tmp_path / "faiss.index"
    sidecar_path = tmp_path / "faiss.chunk_ids.json"
    faiss_build.write_index(index, chunk_ids, index_path, sidecar_path)
    index2, chunk_ids2 = faiss_build.read_index(index_path, sidecar_path)

    q = faiss_build.l2_normalize(vecs[3:4])
    scores, idxs = index2.search(q, 1)
    assert chunk_ids2[idxs[0][0]] == chunk_ids[3]
    assert scores[0][0] == np.float32(1.0) or np.isclose(scores[0][0], 1.0)


def _mem_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    init_schema(conn)
    return conn


def _seed_chunk(conn: sqlite3.Connection, chunk_id: str, pmid: str) -> None:
    conn.execute(
        "INSERT INTO papers (pmid, title, fetched_at) VALUES (?, ?, ?)",
        (pmid, "t", "2026-01-01T00:00:00+00:00"),
    )
    conn.execute(
        "INSERT INTO chunks (chunk_id, pmid, section_type, ordinal, text, "
        "n_deberta_tokens, n_medcpt_tokens) VALUES (?, ?, 'methods', 0, ?, 1, 1)",
        (chunk_id, pmid, f"text-{chunk_id}"),
    )


def test_run_embed_reads_chunks_ordered_by_chunk_id(tmp_path):
    conn = _mem_conn()
    # insert out of order; expect output sorted by chunk_id
    _seed_chunk(conn, "999_methods_00", "999")
    _seed_chunk(conn, "111_methods_00", "111")
    _seed_chunk(conn, "555_methods_00", "555")
    conn.commit()

    seen: list[list[str]] = []

    def _embed(texts, batch_size=32):
        seen.append(list(texts))
        return np.random.rand(len(texts), 768).astype(np.float32)

    index_path = tmp_path / "faiss.index"
    sidecar_path = tmp_path / "faiss.chunk_ids.json"
    counters = faiss_build.run_embed(
        conn=conn,
        index_path=index_path,
        sidecar_path=sidecar_path,
        embed_fn=_embed,
    )

    assert counters == {"chunks": 3}
    _, chunk_ids = faiss_build.read_index(index_path, sidecar_path)
    assert chunk_ids == ["111_methods_00", "555_methods_00", "999_methods_00"]
    # embed_fn saw the same texts in the same chunk_id order
    assert seen[0] == ["text-111_methods_00", "text-555_methods_00", "text-999_methods_00"]
