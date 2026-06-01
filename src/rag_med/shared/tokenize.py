"""Biomedical tokenizer for BM25 lexical_search (Q7).

Lives in ``shared/`` so the ``eval`` phase reuses the exact tokenizer the
``indexing`` build and ``serving`` query path use — the three must agree or
BM25 scores are meaningless. One public function: ``bm25_tokenize``.

Design (Q7): a naive word split shreds medical vocabulary
(``IL-4`` -> ``il``, ``4``). The regex below treats ``-``, ``+`` and ``/``
as *intra*-token characters when glued to alphanumerics, so ``IL-4``,
``CD8+`` and ``FEV1/FVC`` survive as single tokens, while sentence
punctuation still separates words.
"""

from __future__ import annotations

import re

# A token is a run of letters/digits, optionally with intra-token -, +, /
# (and . for decimals like 2.5) glued between alphanumerics. Leading/trailing
# punctuation is excluded by requiring alphanumeric anchors on the connectors.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-+/.][a-z0-9]+)*\+?")

# Small hand-list of English stopwords (Q7: "~30 words"). Deliberately tiny —
# medical terms are never stopwords, so we err toward keeping.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "not",
        "no",
    }
)


def bm25_tokenize(text: str) -> list[str]:
    """Tokenize ``text`` into BM25 terms, biomedical-aware.

    lowercase -> regex tokens (keeps IL-4 / CD8+ / FEV1/FVC) -> drop stopwords
    """
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]
