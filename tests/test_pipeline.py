"""Tests for pipeline.run_fetch — pure orchestration over injected fns + conn.

Real NCBI / real network never hit; that's reserved for test_smoke_ingest.py
on Day 7.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from rag_med.indexing.pipeline import run_fetch
from rag_med.shared.db import init_schema

FIXTURES = Path(__file__).parent / "fixtures"


def _xml(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _mem_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    init_schema(conn)
    return conn


def _fake_esearch_factory(pmids: list[str]):
    async def _esearch(query: str, retmax: int) -> list[str]:
        return pmids[:retmax]

    return _esearch


def _fake_elink_factory(mapping: dict[str, str]):
    async def _elink(pmids: list[str]) -> dict[str, str]:
        return {p: mapping[p] for p in pmids if p in mapping}

    return _elink


def _fake_efetch_factory(by_pmcid: dict[str, bytes]):
    async def _efetch(pmcids: list[str]) -> bytes:
        # Pipeline calls one PMCID at a time.
        assert len(pmcids) == 1
        return by_pmcid[pmcids[0]]

    return _efetch


def test_happy_path_inserts_paper_and_xml():
    conn = _mem_conn()
    pmids = ["11111111"]
    mapping = {"11111111": "PMC1111111"}
    xml = _xml("pmc_full_text.xml")

    counters = asyncio.run(
        run_fetch(
            query="q",
            limit=1,
            conn=conn,
            esearch_fn=_fake_esearch_factory(pmids),
            elink_fn=_fake_elink_factory(mapping),
            efetch_pmc_fn=_fake_efetch_factory({"PMC1111111": xml}),
        )
    )
    assert counters == {"fetched": 1, "parsed": 1, "salvaged": 0, "failed": 0}

    row = conn.execute(
        "SELECT pmid, pmcid, title, year, journal FROM papers WHERE pmid='11111111'"
    ).fetchone()
    assert row is not None
    assert row[0] == "11111111"
    assert row[1] == "PMC1111111"
    assert row[2].startswith("Long-acting bronchodilators")
    assert row[3] == 2023
    assert row[4] == "Test Journal of Pulmonology"

    xml_row = conn.execute("SELECT raw_xml FROM paper_xml WHERE pmid='11111111'").fetchone()
    assert xml_row is not None
    assert xml_row[0] == xml

    assert conn.execute("SELECT COUNT(*) FROM failed_papers").fetchone()[0] == 0


def test_salvage_failure_goes_to_failed_papers():
    conn = _mem_conn()
    pmids = ["33333333"]
    mapping = {"33333333": "PMC3333333"}
    xml = _xml("pmc_no_title.xml")

    counters = asyncio.run(
        run_fetch(
            query="q",
            limit=1,
            conn=conn,
            esearch_fn=_fake_esearch_factory(pmids),
            elink_fn=_fake_elink_factory(mapping),
            efetch_pmc_fn=_fake_efetch_factory({"PMC3333333": xml}),
        )
    )
    assert counters == {"fetched": 1, "parsed": 0, "salvaged": 1, "failed": 0}

    assert conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0
    row = conn.execute(
        "SELECT pmid, failure_reason FROM failed_papers WHERE pmid='33333333'"
    ).fetchone()
    assert row == ("33333333", "missing_title")


def test_unparseable_xml_goes_to_failed_papers_with_xml_parse_error():
    conn = _mem_conn()
    pmids = ["55555555"]
    mapping = {"55555555": "PMC5555555"}

    counters = asyncio.run(
        run_fetch(
            query="q",
            limit=1,
            conn=conn,
            esearch_fn=_fake_esearch_factory(pmids),
            elink_fn=_fake_elink_factory(mapping),
            efetch_pmc_fn=_fake_efetch_factory({"PMC5555555": b"not xml at all"}),
        )
    )
    assert counters["salvaged"] == 1
    row = conn.execute("SELECT failure_reason FROM failed_papers WHERE pmid='55555555'").fetchone()
    assert row == ("xml_parse_error",)


def test_pmid_without_pmc_mapping_goes_to_failed_papers():
    conn = _mem_conn()
    pmids = ["66666666"]

    counters = asyncio.run(
        run_fetch(
            query="q",
            limit=1,
            conn=conn,
            esearch_fn=_fake_esearch_factory(pmids),
            elink_fn=_fake_elink_factory({}),  # no mapping
            efetch_pmc_fn=_fake_efetch_factory({}),
        )
    )
    assert counters["failed"] == 1
    row = conn.execute("SELECT failure_reason FROM failed_papers WHERE pmid='66666666'").fetchone()
    assert row == ("no_content",)


def test_idempotent_on_repeat_run():
    conn = _mem_conn()
    pmids = ["11111111"]
    mapping = {"11111111": "PMC1111111"}
    xml = _xml("pmc_full_text.xml")

    kwargs = dict(
        query="q",
        limit=1,
        conn=conn,
        esearch_fn=_fake_esearch_factory(pmids),
        elink_fn=_fake_elink_factory(mapping),
        efetch_pmc_fn=_fake_efetch_factory({"PMC1111111": xml}),
    )
    asyncio.run(run_fetch(**kwargs))
    asyncio.run(run_fetch(**kwargs))

    assert conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM paper_xml").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM failed_papers").fetchone()[0] == 0


def test_mixed_batch_counters_and_rows():
    conn = _mem_conn()
    pmids = ["11111111", "22222222", "33333333", "66666666"]
    mapping = {
        "11111111": "PMC1111111",
        "22222222": "PMC2222222",
        "33333333": "PMC3333333",
        # 66666666 absent -> no_content
    }
    xmls = {
        "PMC1111111": _xml("pmc_full_text.xml"),
        "PMC2222222": _xml("pmc_abstract_only.xml"),
        "PMC3333333": _xml("pmc_no_title.xml"),
    }

    counters = asyncio.run(
        run_fetch(
            query="q",
            limit=4,
            conn=conn,
            esearch_fn=_fake_esearch_factory(pmids),
            elink_fn=_fake_elink_factory(mapping),
            efetch_pmc_fn=_fake_efetch_factory(xmls),
        )
    )
    assert counters == {"fetched": 3, "parsed": 2, "salvaged": 1, "failed": 1}
    assert conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM failed_papers").fetchone()[0] == 2


def test_efetch_exception_marks_failed_with_xml_parse_error():
    conn = _mem_conn()
    pmids = ["77777777"]
    mapping = {"77777777": "PMC7777777"}

    async def _boom(pmcids: list[str]) -> bytes:
        raise RuntimeError("network down")

    counters = asyncio.run(
        run_fetch(
            query="q",
            limit=1,
            conn=conn,
            esearch_fn=_fake_esearch_factory(pmids),
            elink_fn=_fake_elink_factory(mapping),
            efetch_pmc_fn=_boom,
        )
    )
    assert counters["failed"] == 1
    row = conn.execute("SELECT failure_reason FROM failed_papers WHERE pmid='77777777'").fetchone()
    assert row == ("xml_parse_error",)


def test_empty_esearch_returns_zero_counters():
    conn = _mem_conn()
    counters = asyncio.run(
        run_fetch(
            query="q",
            limit=10,
            conn=conn,
            esearch_fn=_fake_esearch_factory([]),
            elink_fn=_fake_elink_factory({}),
            efetch_pmc_fn=_fake_efetch_factory({}),
        )
    )
    assert counters == {"fetched": 0, "parsed": 0, "salvaged": 0, "failed": 0}


@pytest.mark.parametrize("limit,expected_attempted", [(1, 1), (2, 2), (5, 3)])
def test_limit_respected_against_esearch_supply(limit, expected_attempted):
    conn = _mem_conn()
    pmids = ["11111111", "22222222", "33333333"]
    mapping = {p: f"PMC{p}" for p in pmids}
    xmls = {
        "PMC11111111": _xml("pmc_full_text.xml"),
        "PMC22222222": _xml("pmc_abstract_only.xml"),
        "PMC33333333": _xml("pmc_no_title.xml"),
    }

    counters = asyncio.run(
        run_fetch(
            query="q",
            limit=limit,
            conn=conn,
            esearch_fn=_fake_esearch_factory(pmids),
            elink_fn=_fake_elink_factory(mapping),
            efetch_pmc_fn=_fake_efetch_factory(xmls),
        )
    )
    # `attempted` = fetched + failed-without-fetch (no_content); fetched only when XML retrieved.
    attempted = counters["fetched"] + counters["failed"]
    assert attempted == expected_attempted
