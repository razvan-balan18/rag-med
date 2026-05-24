"""Parse PMC OA JATS XML bytes into a structured paper dict.

wraps `pubmed_parser.parse_pubmed_xml` (metadata + abstract) 
wraps `pubmed_parser.parse_pubmed_paragraph` (body sections)

Salvage rule: returns None iff
  - title missing, OR
  - abstract missing AND no body sections, OR
  - XML unparseable.
Caller writes a `failed_papers` row with the appropriate reason.

Per-paragraph forgiveness: a paragraph that fails to stringify is
dropped silently; the rest of the paper is kept.
"""

from __future__ import annotations

import io
import logging

import pubmed_parser as pp

logger = logging.getLogger(__name__)


def parse(xml: bytes) -> dict | None:
    """Parse JATS PMC XML bytes. Returns structured dict or None on salvage failure."""
    try:
        meta = pp.parse_pubmed_xml(io.BytesIO(xml))
    except Exception as e:
        logger.warning("parse_pubmed_xml failed: %s", e)
        return None

    title = (meta.get("full_title") or "").strip()
    abstract = (meta.get("abstract") or "").strip()
    sections = _parse_sections(xml)

    if not title:
        return None
    if not abstract and not sections:
        return None

    return {
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
