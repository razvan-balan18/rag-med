"""IMRaD chunker — splits a parsed paper into retrievable chunks (Q5).

Greedy-packs pysbd sentences to ~300 DeBERTa tokens (soft ceiling 400).
Tables and figure captions emit one chunk each, regardless of length.
Abstract emits exactly one chunk.

chunk_id format: ``{pmid}_{section_type}_{ordinal:02d}`` (glossary).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rag_med.shared.db import SECTION_TYPES

logger = logging.getLogger(__name__)

TARGET_TOKENS = 300
CEILING_TOKENS = 400


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    pmid: str
    section_type: str
    ordinal: int
    text: str
    n_deberta_tokens: int
    n_medcpt_tokens: int


_deberta_tok = None
_medcpt_tok = None
_pysbd_seg = None


def _load_deberta():
    global _deberta_tok
    if _deberta_tok is None:
        from transformers import AutoTokenizer

        _deberta_tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    return _deberta_tok


def _load_medcpt():
    global _medcpt_tok
    if _medcpt_tok is None:
        from transformers import AutoTokenizer

        _medcpt_tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
    return _medcpt_tok


def _load_pysbd():
    global _pysbd_seg
    if _pysbd_seg is None:
        import pysbd

        _pysbd_seg = pysbd.Segmenter(language="en", clean=False)
    return _pysbd_seg


def count_deberta_tokens(text: str) -> int:
    return len(_load_deberta().encode(text, add_special_tokens=False))


def count_medcpt_tokens(text: str) -> int:
    return len(_load_medcpt().encode(text, add_special_tokens=False))


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _load_pysbd().segment(text) if s.strip()]


def _section_type_for(name: str) -> str:
    n = (name or "").lower()
    if "table" in n:
        return "table"
    if "fig" in n or "caption" in n:
        return "caption"
    if "intro" in n or "background" in n:
        return "introduction"
    if "method" in n:
        return "methods"
    if "result" in n:
        return "results"
    if "discuss" in n or "conclus" in n:
        return "discussion"
    if "abstract" in n:
        return "abstract"
    return "other"


assert set(SECTION_TYPES) >= {
    "abstract",
    "introduction",
    "methods",
    "results",
    "discussion",
    "table",
    "caption",
    "other",
}


def _make_chunk(pmid: str, st: str, ordinal: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{pmid}_{st}_{ordinal:02d}",
        pmid=pmid,
        section_type=st,
        ordinal=ordinal,
        text=text,
        n_deberta_tokens=count_deberta_tokens(text),
        n_medcpt_tokens=count_medcpt_tokens(text),
    )


def _pack(text: str) -> list[str]:
    sentences = split_sentences(text)
    out: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for s in sentences:
        st = count_deberta_tokens(s)
        if not buf and st > CEILING_TOKENS:
            logger.warning("oversize sentence (%d tokens) emitted alone", st)
            out.append(s)
            continue
        if buf_tokens + st > CEILING_TOKENS:
            out.append(" ".join(buf))
            buf, buf_tokens = [s], st
            if buf_tokens >= TARGET_TOKENS:
                out.append(" ".join(buf))
                buf, buf_tokens = [], 0
        else:
            buf.append(s)
            buf_tokens += st
            if buf_tokens >= TARGET_TOKENS:
                out.append(" ".join(buf))
                buf, buf_tokens = [], 0
    if buf:
        out.append(" ".join(buf))
    return out


def chunk_paper(paper: dict) -> list[Chunk]:
    """Convert one parsed-paper dict to a list of Chunks."""
    pmid = str(paper.get("pmid") or "")
    chunks: list[Chunk] = []
    counters: dict[str, int] = {}

    def _next(st: str) -> int:
        n = counters.get(st, 0)
        counters[st] = n + 1
        return n

    abstract = (paper.get("abstract") or "").strip()
    if abstract:
        chunks.append(_make_chunk(pmid, "abstract", _next("abstract"), abstract))

    for section in paper.get("sections") or []:
        text = (section.get("text") or "").strip()
        if not text:
            continue
        st = _section_type_for(section.get("section_name") or "")
        if st in {"table", "caption"}:
            chunks.append(_make_chunk(pmid, st, _next(st), text))
            continue
        for piece in _pack(text):
            chunks.append(_make_chunk(pmid, st, _next(st), piece))

    return chunks
