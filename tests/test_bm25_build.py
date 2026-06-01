"""BM25 build tests (week2 day4) — mock the chunks, no model.

Mirrors test_faiss_build: an injected ``tokenize_fn`` keeps tests off the
real regex internals, round-trip proves the pickle + sidecar survive
write->read, and a known-query top-1 proves the index is actually searchable.
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np

from rag_med.indexing import bm25_build
from rag_med.shared.db import init_schema


def _top1_chunk_id(bm25, chunk_ids, query_tokens):
    scores = bm25.get_scores(query_tokens)
    return chunk_ids[int(np.argmax(scores))]


def test_known_query_top1_exact_match_wins():
    texts = [f"common filler text doc {i}" for i in range(10)]
    texts[4] = "common filler unobtainium marker doc 4"
    chunk_ids = [f"111_methods_{i:02d}" for i in range(10)]

    bm25 = bm25_build.build_index(texts)

    assert _top1_chunk_id(bm25, chunk_ids, ["unobtainium"]) == "111_methods_04"


def test_write_read_roundtrip_identical_search(tmp_path):
    texts = [f"common filler text doc {i}" for i in range(10)]
    texts[4] = "common filler unobtainium marker doc 4"
    chunk_ids = [f"111_methods_{i:02d}" for i in range(10)]
    bm25 = bm25_build.build_index(texts)

    index_path = tmp_path / "bm25.pkl"
    sidecar_path = tmp_path / "bm25.chunk_ids.json"
    bm25_build.write_index(bm25, chunk_ids, index_path, sidecar_path)

    bm25_2, chunk_ids_2 = bm25_build.read_index(index_path, sidecar_path)

    scores1 = bm25.get_scores(["unobtainium"])
    scores2 = bm25_2.get_scores(["unobtainium"])
    assert np.allclose(scores1, scores2)
    assert chunk_ids_2 == chunk_ids


def test_sidecar_length_matches_corpus(tmp_path):
    texts = [f"doc number {i}" for i in range(7)]
    chunk_ids = [f"222_results_{i:02d}" for i in range(7)]
    bm25 = bm25_build.build_index(texts)

    index_path = tmp_path / "bm25.pkl"
    sidecar_path = tmp_path / "bm25.chunk_ids.json"
    bm25_build.write_index(bm25, chunk_ids, index_path, sidecar_path)

    sidecar = json.loads(sidecar_path.read_text())
    assert len(sidecar) == bm25.corpus_size == 7


def _mem_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    init_schema(conn)
    return conn


def _seed_chunk(conn: sqlite3.Connection, chunk_id: str, pmid: str, text: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO papers (pmid, title, fetched_at) VALUES (?, ?, ?)",
        (pmid, "t", "2026-01-01T00:00:00+00:00"),
    )
    conn.execute(
        "INSERT INTO chunks (chunk_id, pmid, section_type, ordinal, text, "
        "n_deberta_tokens, n_medcpt_tokens) VALUES (?, ?, 'methods', 0, ?, 1, 1)",
        (chunk_id, pmid, text),
    )


def test_run_bm25_reads_chunks_ordered_by_chunk_id(tmp_path):
    conn = _mem_conn()
    # insert out of order; expect output sorted by chunk_id
    _seed_chunk(conn, "999_methods_00", "999", "alpha text")
    _seed_chunk(conn, "111_methods_00", "111", "beta text")
    _seed_chunk(conn, "555_methods_00", "555", "gamma text")
    conn.commit()

    index_path = tmp_path / "bm25.pkl"
    sidecar_path = tmp_path / "bm25.chunk_ids.json"
    counters = bm25_build.run_bm25(
        conn=conn,
        index_path=index_path,
        sidecar_path=sidecar_path,
    )

    assert counters == {"chunks": 3}
    _, chunk_ids = bm25_build.read_index(index_path, sidecar_path)
    assert chunk_ids == ["111_methods_00", "555_methods_00", "999_methods_00"]
