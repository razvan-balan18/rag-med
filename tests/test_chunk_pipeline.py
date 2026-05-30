"""Tests for pipeline.run_chunk — parse → chunk → INSERT OR IGNORE chunks.

Real parse + real chunk_paper over fixtures; token counters mocked to
word-split (mirrors test_chunk.py) so no HF tokenizer loads in CI.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from rag_med.indexing import chunk as chunk_mod
from rag_med.indexing.pipeline import run_chunk
from rag_med.shared.db import init_schema

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _mock_tokenizers(monkeypatch):
    monkeypatch.setattr(chunk_mod, "count_deberta_tokens", lambda t: len(t.split()))
    monkeypatch.setattr(chunk_mod, "count_medcpt_tokens", lambda t: len(t.split()))


def _mem_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    init_schema(conn)
    return conn


def _seed_paper(conn: sqlite3.Connection, pmid: str, xml_name: str) -> None:
    xml = (FIXTURES / xml_name).read_bytes()
    conn.execute(
        "INSERT INTO papers (pmid, title, fetched_at) VALUES (?, ?, ?)",
        (pmid, "seed title", "2026-01-01T00:00:00+00:00"),
    )
    conn.execute(
        "INSERT INTO paper_xml (pmid, raw_xml, parsed_at) VALUES (?, ?, ?)",
        (pmid, xml, "2026-01-01T00:00:00+00:00"),
    )
    conn.commit()


def test_happy_path_inserts_chunks():
    conn = _mem_conn()
    _seed_paper(conn, "11111111", "pmc_full_text.xml")

    counters = run_chunk(conn=conn)

    n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert n_chunks > 0
    assert counters == {"papers": 1, "chunks": n_chunks}
    # chunks carry the row pmid (parser returns empty pmid; pipeline backfills).
    assert (
        conn.execute("SELECT COUNT(*) FROM chunks WHERE pmid='11111111'").fetchone()[0] == n_chunks
    )


def test_idempotent_on_repeat_run():
    conn = _mem_conn()
    _seed_paper(conn, "11111111", "pmc_full_text.xml")

    run_chunk(conn=conn)
    n_after_first = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    counters = run_chunk(conn=conn)

    assert counters == {"papers": 0, "chunks": 0}
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == n_after_first


def test_skips_already_chunked_papers():
    conn = _mem_conn()
    _seed_paper(conn, "11111111", "pmc_full_text.xml")
    _seed_paper(conn, "22222222", "pmc_abstract_only.xml")

    # Pre-chunk only the first paper.
    conn.execute(
        "INSERT INTO chunks (chunk_id, pmid, section_type, ordinal, text, "
        "n_deberta_tokens, n_medcpt_tokens) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("11111111_abstract_00", "11111111", "abstract", 0, "x", 1, 1),
    )
    conn.commit()

    counters = run_chunk(conn=conn)

    assert counters["papers"] == 1
    assert conn.execute("SELECT COUNT(*) FROM chunks WHERE pmid='22222222'").fetchone()[0] > 0
    # untouched paper keeps its single pre-seeded chunk
    assert conn.execute("SELECT COUNT(*) FROM chunks WHERE pmid='11111111'").fetchone()[0] == 1


def test_no_papers_returns_zero_counters():
    conn = _mem_conn()
    assert run_chunk(conn=conn) == {"papers": 0, "chunks": 0}
