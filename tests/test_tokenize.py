"""Biomedical BM25 tokenizer tests (week2 day4) — pure, no model.

Per Q7: keep hyphens / + / inside-slash tokens (``IL-4``, ``CD8+``,
``FEV1/FVC``), keep digit-letter combos (``FEV1``, ``25mg``), lowercase,
drop a small English stopword list. Same tokenizer runs at build time and
query time, so these cases pin the contract both sides depend on.
"""

from __future__ import annotations

from rag_med.shared.tokenize import bm25_tokenize


def test_keeps_biomedical_terms_intact():
    tokens = bm25_tokenize("IL-4 induces CD8+ T-cell proliferation in FEV1/FVC patients")
    assert "il-4" in tokens
    assert "cd8+" in tokens
    assert "fev1/fvc" in tokens


def test_drops_english_stopwords():
    tokens = bm25_tokenize("the drug is effective and safe")
    assert "the" not in tokens
    assert "is" not in tokens
    assert "and" not in tokens
    assert "drug" in tokens
    assert "effective" in tokens


def test_keeps_digit_letter_combos():
    tokens = bm25_tokenize("25mg dose raised FEV1 in the cohort")
    assert "25mg" in tokens
    assert "fev1" in tokens


def test_empty_and_whitespace_yield_empty_list():
    assert bm25_tokenize("") == []
    assert bm25_tokenize("   \n\t ") == []
