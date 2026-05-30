"""Parse PMC OA JATS XML bytes into a structured paper dict.

wraps `pubmed_parser.parse_pubmed_xml` (metadata + abstract)
wraps `pubmed_parser.parse_pubmed_paragraph` (body sections)

Returns (paper_dict, None) on success.
Returns (None, failure_reason) on salvage failure, where reason matches
the `failed_papers.failure_reason` enum (db.py FAILURE_REASONS):
  - "xml_parse_error" — pubmed_parser raised on the metadata XML
  - "missing_title"   — title element empty / absent
  - "no_content"      — both abstract empty AND no body sections parsed

Per-paragraph forgiveness: a paragraph that fails to stringify is
dropped silently; the rest of the paper is kept.

dict shape: {"pmid", "pmcid", "title", "abstract", "sections": [{"section_name", "text"}], "mesh_terms",
  "journal", "year", "authors"}
"""

from __future__ import annotations

import io
import logging

import pubmed_parser as pp
import pubmed_parser.pubmed_oa_parser as _oa

logger = logging.getLogger(__name__)


# Upstream bug (pubmed_parser 0.5.1): parse_pubmed_xml does
#   `int(pub_date_dict["year"])` guarded by `except TypeError`, which lets
# `KeyError` escape when XML has neither `ppub` nor `collection` pub-date with
# a <year> child (common on epub-only papers — ~40% of recent PMC OA hits).
# Patch parse_date to always seed year=None so int(None) -> TypeError -> None.
_orig_parse_date = _oa.parse_date


def _parse_date_with_year_default(tree, date_type):  # type: ignore[no-untyped-def]
    d = _orig_parse_date(tree, date_type)
    # Only seed year=None on the *last* fallback attempt ("collection") so the
    # upstream ppub->collection probing still fires. int(None) -> TypeError ->
    # pub_year=None in the caller (which is what we want).
    if date_type == "collection":
        d.setdefault("year", None)
    return d


_oa.parse_date = _parse_date_with_year_default


def parse(xml: bytes) -> tuple[dict | None, str | None]:
    """Parse JATS PMC XML. Returns (dict, None) or (None, failure_reason)."""
    try:
        meta = pp.parse_pubmed_xml(io.BytesIO(xml))
    except Exception as e:
        logger.warning("parse_pubmed_xml failed: %s", e)
        return None, "xml_parse_error"

    title = (meta.get("full_title") or "").strip()
    abstract = (meta.get("abstract") or "").strip()
    sections = _parse_sections(xml)

    if not title:
        return None, "missing_title"
    if not abstract and not sections:
        return None, "no_content"

    paper = {
        "pmid": meta.get("pmid") or "",
        "pmcid": meta.get("pmc") or "",
        "title": title,
        "abstract": abstract,
        "sections": sections,
        "mesh_terms": _split_subjects(meta.get("subjects") or ""),
        "journal": (meta.get("journal") or "").strip(),
        "year": meta.get("publication_year"),
        "authors": meta.get("author_list") or [],
    }
    return paper, None


def _parse_sections(xml: bytes) -> list[dict[str, str]]:
    try:
        paragraphs = pp.parse_pubmed_paragraph(io.BytesIO(xml), all_paragraph=True)
    except Exception as e:
        logger.warning("parse_pubmed_paragraph failed: %s", e)
        return []

    grouped: dict[str, list[str]] = {}
    order: list[str] = []
    for para in paragraphs:
        try:
            name = (para.get("section") or "").strip()
            text = (para.get("text") or "").strip()
            if not text:
                continue
            if name not in grouped:
                grouped[name] = []
                order.append(name)
            grouped[name].append(text)
        except Exception as e:
            logger.debug("dropped paragraph: %s", e)
            continue
    return [{"section_name": n, "text": "\n\n".join(grouped[n])} for n in order]


def _split_subjects(s: str) -> list[str]:
    return [t.strip() for t in s.split(";") if t.strip()]
