"""Cross-phase data models (decisions.md Q11d repo layout).

Entities shared across phases live here so ``indexing`` and ``serving`` never
import each other (phase-isolation hard rule). ``Chunk`` is produced by the
indexing chunker and consumed by the serving retriever + prompt builder.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A retrievable text unit derived from a paper (glossary `chunk`).

    chunk_id format: ``{pmid}_{section_type}_{ordinal:02d}``. Both token counts
    recorded — ``n_deberta_tokens`` canonical, ``n_medcpt_tokens`` sanity check.
    """

    chunk_id: str
    pmid: str
    section_type: str
    ordinal: int
    text: str
    n_deberta_tokens: int
    n_medcpt_tokens: int
