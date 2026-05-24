"""M1 ingest smoke gates — Q22e, week1.md day 7.

Real SQLite at `settings.sqlite_path`. Prereq:
    python -m rag_med.indexing.pipeline fetch --query-preset copd-m1 --limit 100

Tests skip with a clear message if the DB is absent. The idempotency test
re-runs the pipeline (hits NCBI) and is therefore slow (~30s).
"""

from __future__ import annotations

import asyncio
import sqlite3

import pytest

from rag_med.config import get_settings
from rag_med.indexing import pipeline as pipe

VOLUME_GATE = 95
FULLTEXT_GATE = 80
FAILED_BUDGET = 5


@pytest.fixture(scope="module")
def conn():
    db = get_settings().sqlite_path
    if not db.exists():
        pytest.skip(f"populate DB first: run pipeline fetch (missing {db})")
    c = sqlite3.connect(str(db))
    yield c
    c.close()


def _count(conn: sqlite3.Connection, sql: str) -> int:
    return conn.execute(sql).fetchone()[0]


def test_volume_gate(conn):
    assert _count(conn, "SELECT COUNT(*) FROM papers") >= VOLUME_GATE


def test_all_titles_present(conn):
    missing = _count(conn, "SELECT COUNT(*) FROM papers WHERE title IS NULL OR title = ''")
    assert missing == 0


def test_pmcid_coverage(conn):
    assert _count(conn, "SELECT COUNT(*) FROM papers WHERE pmcid IS NOT NULL") >= FULLTEXT_GATE


def test_body_xml_present(conn):
    # source_type='full_text' iff parse() returned ≥1 body section -> raw_xml
    # has body content. paper_xml row is inserted alongside, so this is the
    # honest "body XML present" check.
    assert _count(conn, "SELECT COUNT(*) FROM papers WHERE source_type='full_text'") >= FULLTEXT_GATE
    assert _count(conn, "SELECT COUNT(*) FROM paper_xml") >= FULLTEXT_GATE


def test_failure_budget(conn):
    assert _count(conn, "SELECT COUNT(*) FROM failed_papers") < FAILED_BUDGET


def test_idempotent_rerun():
    db = get_settings().sqlite_path
    if not db.exists():
        pytest.skip(f"populate DB first: run pipeline fetch (missing {db})")
    c = sqlite3.connect(str(db))
    try:
        before = (
            _count(c, "SELECT COUNT(*) FROM papers"),
            _count(c, "SELECT COUNT(*) FROM paper_xml"),
            _count(c, "SELECT COUNT(*) FROM failed_papers"),
        )
        asyncio.run(
            pipe.run_fetch(query=pipe.QUERY_PRESETS["copd-m1"], limit=100, conn=c)
        )
        after = (
            _count(c, "SELECT COUNT(*) FROM papers"),
            _count(c, "SELECT COUNT(*) FROM paper_xml"),
            _count(c, "SELECT COUNT(*) FROM failed_papers"),
        )
        assert before == after
    finally:
        c.close()
