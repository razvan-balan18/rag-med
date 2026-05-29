"""NCBI E-utilities client: esearch + efetch for PubMed/PMC.

Direct httpx wrappers

Rate limit: 10 req/s with API key; api_key + email on every request (politeness).

Retry 3x with exponential backoff on 5xx + network errors. - decided in decisions.md
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import httpx

from rag_med.config import get_settings

logger = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

M1_QUERY = (
    '("Pulmonary Disease, Chronic Obstructive"[MeSH] OR "COPD"[Title/Abstract]) '
    'AND ("2020"[Date - Publication] : "3000"[Date - Publication]) '
    'AND "pubmed pmc open access"[filter]'
)

MAX_CONCURRENT = 10  # Q22b: 10 req/s with API key
ELINK_BATCH = 50  # NCBI elink GETs with >~50 ids stream-close mid-response

RETRY_ATTEMPTS = 3
RETRY_BACKOFF_S = 1.0
TIMEOUT_S = 30.0


def _politeness_params() -> dict[str, str]:
    s = get_settings()
    return {"api_key": s.ncbi_api_key, "email": str(s.ncbi_email)}


def _client_factory() -> httpx.AsyncClient:
    limits = httpx.Limits(
        max_connections=MAX_CONCURRENT,
        max_keepalive_connections=MAX_CONCURRENT,
    )
    return httpx.AsyncClient(base_url=EUTILS_BASE, timeout=TIMEOUT_S, limits=limits)


async def _get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    params: dict[str, str] | list[tuple[str, str]],
    sem: asyncio.Semaphore,
) -> httpx.Response:
    last_exc: Exception | None = None
    for attempt in range(RETRY_ATTEMPTS):
        async with sem:
            try:
                r = await client.get(url, params=params)
                if r.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"5xx {r.status_code}", request=r.request, response=r
                    )
                r.raise_for_status()
                return r
            except (httpx.HTTPError, httpx.NetworkError) as e:
                last_exc = e
                if attempt == RETRY_ATTEMPTS - 1:
                    break
                wait = RETRY_BACKOFF_S * (2**attempt)
                logger.warning("retry %d/%d after %.1fs: %s", attempt + 1, RETRY_ATTEMPTS, wait, e)
                await asyncio.sleep(wait)
    assert last_exc is not None
    raise last_exc


async def esearch(query: str, retmax: int) -> list[str]:
    """Return PMIDs matching `query`, capped at `retmax`."""
    # pmid - pk for paper schema

    params = {
        "db": "pubmed",
        "term": query,
        "retmax": str(retmax),
        "retmode": "json",
        **_politeness_params(),
    }
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    async with _client_factory() as client:
        r = await _get_with_retry(client, "/esearch.fcgi", params, sem)
    data = r.json()
    return list(data["esearchresult"]["idlist"])


async def _efetch(db: str, ids: list[str]) -> bytes:
    if not ids:
        return b""
    params = {
        "db": db,
        "id": ",".join(ids),
        "retmode": "xml",
        **_politeness_params(),
    }
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    async with _client_factory() as client:
        r = await _get_with_retry(client, "/efetch.fcgi", params, sem)
    return r.content


# pmid - pubmed, can be just abstract and stuff like that   -> pubmed
# pcmid - full files only ids, can be null if just abstract -> pubmed central
async def efetch_pubmed(pmids: list[str]) -> bytes:
    """Return raw PubMed XML for the given PMIDs (metadata + abstract)."""
    return await _efetch("pubmed", pmids)


async def efetch_pmc(pmcids: list[str]) -> bytes:
    """Return raw PMC XML for the given PMCIDs (full-text body)."""
    return await _efetch("pmc", pmcids)


async def _elink_batch(
    client: httpx.AsyncClient, sem: asyncio.Semaphore, pmids: list[str]
) -> dict[str, str]:
    params: list[tuple[str, str]] = [
        ("dbfrom", "pubmed"),
        ("db", "pmc"),
        ("retmode", "json"),
    ]
    params.extend(("id", p) for p in pmids)
    params.extend(_politeness_params().items())

    r = await _get_with_retry(client, "/elink.fcgi", params, sem)
    data = r.json()

    mapping: dict[str, str] = {}
    for linkset in data.get("linksets", []):
        ids = linkset.get("ids") or []
        if not ids:
            continue
        pmid = str(ids[0])
        for lsdb in linkset.get("linksetdbs") or []:
            if lsdb.get("linkname") != "pubmed_pmc":
                continue
            links = lsdb.get("links") or []
            if links:
                mapping[pmid] = f"PMC{links[0]}"
            break
    return mapping


async def elink_pubmed_to_pmc(pmids: list[str]) -> dict[str, str]:
    """Map PMIDs -> PMCIDs (with 'PMC' prefix) via NCBI elink.

    PMIDs with no PMC counterpart are omitted from the returned dict.
    Batched: NCBI closes the response stream when ~100 ids are passed in one GET
    (observed 2026-05-24). Chunk to ELINK_BATCH per request.
    """
    if not pmids:
        return {}
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    mapping: dict[str, str] = {}
    async with _client_factory() as client:
        for i in range(0, len(pmids), ELINK_BATCH):
            chunk = pmids[i : i + ELINK_BATCH]
            mapping.update(await _elink_batch(client, sem, chunk))
    return mapping


async def _smoke() -> None:
    """Manual smoke: 5 PMIDs via M1 query, dump XML to data/raw/."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    raw_dir: Path = get_settings().data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    pmids = await esearch(M1_QUERY, retmax=5)
    logger.info("esearch returned %d PMIDs: %s", len(pmids), pmids)

    pubmed_xml = await efetch_pubmed(pmids)
    (raw_dir / "smoke_pubmed.xml").write_bytes(pubmed_xml)
    logger.info("wrote smoke_pubmed.xml: %d bytes", len(pubmed_xml))

    print(json.dumps({"pmids": pmids, "pubmed_xml_bytes": len(pubmed_xml)}, indent=2))


if __name__ == "__main__":
    asyncio.run(_smoke())
