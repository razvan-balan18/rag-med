"""indexing/pipeline.py — orchestrate esearch -> elink -> efetch_pmc -> parse -> insert.

`run_fetch` is pure async orchestration with every network call and the DB
handle injected. Unit tests pass fakes; the CLI binds the real `pubmed.*`
functions and a real SQLite connection at `settings.sqlite_path`.

this wires together the parts in parse and pubmed

Per-paper flow:
  1. esearch -> PMIDs (capped at `limit`)
  2. elink   -> {pmid: pmcid}; PMIDs without a PMC mapping go to failed_papers
                with reason `no_content` (no full-text retrievable).
  3. efetch_pmc per PMCID -> raw JATS XML bytes. Network exceptions land in
     failed_papers as `xml_parse_error` (the upstream `_get_with_retry` already
     burned the 3-attempt budget).
  4. parse() -> (paper, None) | (None, reason). Salvage failures go to
     failed_papers with the reason returned by the parser.
  5. Success -> INSERT OR IGNORE into `papers` + `paper_xml`. PMID/PMCID are
     backfilled from the esearch/elink results (parser returns empty strings
     for both against real NCBI XML; see decisions.md Q22c drift).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

import structlog

from rag_med.config import get_settings
from rag_med.indexing.ingest import parse as parse_mod
from rag_med.indexing.ingest import pubmed
from rag_med.shared.db import connect, init_schema

log = structlog.get_logger()

QUERY_PRESETS: dict[str, str] = {"copd-m1": pubmed.M1_QUERY}

EsearchFn = Callable[[str, int], Awaitable[list[str]]]
ElinkFn = Callable[[list[str]], Awaitable[dict[str, str]]]
EfetchPmcFn = Callable[[list[str]], Awaitable[bytes]]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _ms_since(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def _insert_failed(conn: sqlite3.Connection, pmid: str, reason: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO failed_papers (pmid, failure_reason, attempted_at) "
        "VALUES (?, ?, ?)",
        (pmid, reason, _now()),
    )


async def run_fetch(
    *,
    query: str,
    limit: int,
    conn: sqlite3.Connection,
    esearch_fn: EsearchFn = pubmed.esearch,
    elink_fn: ElinkFn = pubmed.elink_pubmed_to_pmc,
    efetch_pmc_fn: EfetchPmcFn = pubmed.efetch_pmc,
) -> dict[str, int]:
    """Run one ingest pass against `conn`. Returns counters."""
    counters = {"fetched": 0, "parsed": 0, "salvaged": 0, "failed": 0}

    t0 = time.perf_counter()
    pmids = await esearch_fn(query, limit)
    log.info("esearch_done", n_pmids=len(pmids), elapsed_ms=_ms_since(t0))
    if not pmids:
        return counters

    t1 = time.perf_counter()
    pmid_to_pmcid = await elink_fn(pmids)
    log.info(
        "elink_done",
        n_in=len(pmids),
        n_mapped=len(pmid_to_pmcid),
        elapsed_ms=_ms_since(t1),
    )

    for pmid in pmids:
        t_paper = time.perf_counter()
        pmcid = pmid_to_pmcid.get(pmid)

        if not pmcid:
            counters["failed"] += 1
            _insert_failed(conn, pmid, "no_content")
            conn.commit()
            log.info(
                "paper_processed",
                pmid=pmid,
                status="failed",
                reason="no_pmcid_mapping",
                elapsed_ms=_ms_since(t_paper),
            )
            continue

        try:
            xml = await efetch_pmc_fn([pmcid])
        except Exception as e:
            counters["failed"] += 1
            _insert_failed(conn, pmid, "xml_parse_error")
            conn.commit()
            log.warning(
                "paper_processed",
                pmid=pmid,
                status="failed",
                reason="efetch_exception",
                error=str(e),
                elapsed_ms=_ms_since(t_paper),
            )
            continue

        counters["fetched"] += 1
        paper, fail_reason = parse_mod.parse(xml)

        if paper is None:
            counters["salvaged"] += 1
            _insert_failed(conn, pmid, fail_reason or "xml_parse_error")
            conn.commit()
            log.info(
                "paper_processed",
                pmid=pmid,
                status="salvaged",
                reason=fail_reason,
                elapsed_ms=_ms_since(t_paper),
            )
            continue

        source_type = "full_text" if paper["sections"] else "abstract_only"
        conn.execute(
            "INSERT OR IGNORE INTO papers "
            "(pmid, pmcid, doi, title, journal, year, source_type, mesh_terms_json, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                pmid,
                pmcid,
                None,
                paper["title"],
                paper.get("journal") or None,
                paper.get("year"),
                source_type,
                json.dumps(paper.get("mesh_terms") or []),
                _now(),
            ),
        )
        conn.execute(
            "INSERT OR IGNORE INTO paper_xml (pmid, raw_xml, parsed_at) VALUES (?, ?, ?)",
            (pmid, xml, _now()),
        )
        conn.commit()
        counters["parsed"] += 1
        log.info(
            "paper_processed",
            pmid=pmid,
            pmcid=pmcid,
            status="parsed",
            sections=len(paper["sections"]),
            source_type=source_type,
            elapsed_ms=_ms_since(t_paper),
        )

    return counters


def _configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
    )


async def _cli_fetch(query: str, limit: int) -> None:
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    conn = connect(settings.sqlite_path)
    init_schema(conn)
    try:
        counters = await run_fetch(query=query, limit=limit, conn=conn)
        log.info("run_fetch_done", **counters)
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    parser = argparse.ArgumentParser(prog="rag_med.indexing.pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    fetch_p = sub.add_parser("fetch", help="fetch + parse + insert M1 corpus")
    fetch_p.add_argument(
        "--query-preset", choices=list(QUERY_PRESETS), default="copd-m1"
    )
    fetch_p.add_argument("--limit", type=int, default=100)
    args = parser.parse_args(argv)

    if args.cmd == "fetch":
        asyncio.run(_cli_fetch(query=QUERY_PRESETS[args.query_preset], limit=args.limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
