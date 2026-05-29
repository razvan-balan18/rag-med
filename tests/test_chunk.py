"""IMRaD chunker tests — mock token counters, real pysbd."""

from __future__ import annotations

import re

import pytest

from rag_med.indexing import chunk as chunk_mod
from rag_med.indexing.chunk import Chunk, chunk_paper


@pytest.fixture(autouse=True)
def _mock_tokenizers(monkeypatch):
    """Replace HF tokenizer counts with word-split counts for predictability."""
    monkeypatch.setattr(chunk_mod, "count_deberta_tokens", lambda t: len(t.split()))
    monkeypatch.setattr(chunk_mod, "count_medcpt_tokens", lambda t: len(t.split()))


def _paper(**overrides) -> dict:
    base = {
        "pmid": "12345678",
        "pmcid": "PMC1",
        "title": "T",
        "abstract": "",
        "sections": [],
        "mesh_terms": [],
        "journal": "J",
        "year": 2023,
        "authors": [],
    }
    base.update(overrides)
    return base


CHUNK_ID_RE = re.compile(
    r"^\d+_(abstract|introduction|methods|results|discussion|table|caption|other)_\d{2}$"
)


def test_empty_paper_no_chunks():
    chunks = chunk_paper(_paper())
    assert chunks == []


def test_abstract_only_one_chunk():
    chunks = chunk_paper(_paper(abstract="Short abstract about COPD."))
    assert len(chunks) == 1
    c = chunks[0]
    assert c.section_type == "abstract"
    assert c.ordinal == 0
    assert c.chunk_id == "12345678_abstract_00"
    assert c.text == "Short abstract about COPD."
    assert c.pmid == "12345678"


def test_long_abstract_splits_under_ceiling():
    """Q5: abstract is one chunk unless >400 tokens, then split like a section."""
    sentence = (" ".join(["word"] * 50)) + "."
    text = " ".join([sentence] * 12)  # ~600 mock-tokens, 12 sentences
    chunks = chunk_paper(_paper(abstract=text))
    abstracts = [c for c in chunks if c.section_type == "abstract"]
    assert len(abstracts) >= 2
    for c in abstracts:
        assert c.n_deberta_tokens <= 400, c.n_deberta_tokens
    assert [c.ordinal for c in abstracts] == list(range(len(abstracts)))


def test_chunk_id_format_regex():
    chunks = chunk_paper(
        _paper(
            abstract="A.",
            sections=[
                {"section_name": "Introduction", "text": "Intro one. Intro two."},
                {"section_name": "Methods", "text": "Methods text."},
            ],
        )
    )
    assert len(chunks) >= 3
    for c in chunks:
        assert CHUNK_ID_RE.match(c.chunk_id), c.chunk_id


def test_section_name_lowercase_substring_mapping():
    chunks = chunk_paper(
        _paper(
            sections=[
                {"section_name": "Introduction", "text": "i."},
                {"section_name": "Materials and Methods", "text": "m."},
                {"section_name": "RESULTS", "text": "r."},
                {"section_name": "Discussion", "text": "d."},
                {"section_name": "Acknowledgements", "text": "a."},
            ]
        )
    )
    by_section = {c.section_type for c in chunks}
    assert "introduction" in by_section
    assert "methods" in by_section
    assert "results" in by_section
    assert "discussion" in by_section
    assert "other" in by_section


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Statistical analysis", "methods"),
        ("Statistical Analysis", "methods"),
        ("Study design", "methods"),
        ("Study population", "methods"),
        ("Eligibility criteria", "methods"),
        ("Search strategy", "methods"),
        ("Data Collection", "methods"),
        ("Covariates", "methods"),
        ("Outcomes", "results"),
        ("Baseline Characteristics", "results"),
        ("Limitations", "discussion"),
        ("Strengths and limitations", "discussion"),
        ("Objectives", "introduction"),
        # regression: originals still map correctly
        ("Materials and Methods", "methods"),
        ("Conclusion", "discussion"),
        ("Acknowledgements", "other"),
    ],
)
def test_subsection_headings_map_to_imrad(name, expected):
    assert chunk_mod._section_type_for(name) == expected


def test_unknown_section_name_maps_to_other():
    chunks = chunk_paper(_paper(sections=[{"section_name": "Funding statement", "text": "f."}]))
    assert chunks[0].section_type == "other"


def test_short_section_one_chunk():
    text = " ".join(["word"] * 200)
    chunks = chunk_paper(_paper(sections=[{"section_name": "Methods", "text": f"{text}."}]))
    methods = [c for c in chunks if c.section_type == "methods"]
    assert len(methods) == 1
    assert methods[0].n_deberta_tokens == 200


def test_long_section_splits_into_300_to_400_token_chunks():
    sentence = (" ".join(["word"] * 50)) + "."
    text = " ".join([sentence] * 30)
    chunks = chunk_paper(_paper(sections=[{"section_name": "Methods", "text": text}]))
    methods = [c for c in chunks if c.section_type == "methods"]
    assert len(methods) >= 3
    for c in methods[:-1]:
        assert 300 <= c.n_deberta_tokens <= 400, c.n_deberta_tokens
    assert methods[-1].n_deberta_tokens <= 400


def test_ordinals_are_sequential_per_section():
    sentence = (" ".join(["word"] * 50)) + "."
    text = " ".join([sentence] * 30)
    chunks = chunk_paper(_paper(sections=[{"section_name": "Methods", "text": text}]))
    methods = [c for c in chunks if c.section_type == "methods"]
    ordinals = [c.ordinal for c in methods]
    assert ordinals == list(range(len(methods)))


def test_table_section_is_own_single_chunk():
    long_table = " ".join(["cell"] * 800) + "."
    chunks = chunk_paper(_paper(sections=[{"section_name": "Table 1", "text": long_table}]))
    tables = [c for c in chunks if c.section_type == "table"]
    assert len(tables) == 1
    assert tables[0].n_deberta_tokens == 800


def test_caption_section_is_own_chunk():
    chunks = chunk_paper(
        _paper(sections=[{"section_name": "Figure 1 caption", "text": "FEV1 vs time."}])
    )
    captions = [c for c in chunks if c.section_type == "caption"]
    assert len(captions) == 1


def test_both_token_counts_populated():
    chunks = chunk_paper(_paper(abstract="One two three four five."))
    c = chunks[0]
    assert c.n_deberta_tokens == 5
    assert c.n_medcpt_tokens == 5


def test_oversize_single_sentence_emitted_alone(monkeypatch, caplog):
    monkeypatch.setattr(chunk_mod, "count_deberta_tokens", lambda t: 500)
    monkeypatch.setattr(chunk_mod, "count_medcpt_tokens", lambda t: 500)
    chunks = chunk_paper(
        _paper(sections=[{"section_name": "Methods", "text": "Giant sentence here."}])
    )
    methods = [c for c in chunks if c.section_type == "methods"]
    assert len(methods) == 1
    assert methods[0].n_deberta_tokens == 500


def test_pysbd_does_not_break_medical_abbreviations(monkeypatch):
    """Real pysbd: should not split on Fig. / et al. / p < 0.05 / 2.5 mg."""
    monkeypatch.setattr(chunk_mod, "count_deberta_tokens", lambda t: 1000)
    monkeypatch.setattr(chunk_mod, "count_medcpt_tokens", lambda t: 1000)
    text = "Patients received 2.5 mg daily (Fig. 1). Smith et al. reported p < 0.05 across groups."
    sentences = chunk_mod.split_sentences(text)
    assert len(sentences) == 2
    assert "2.5 mg" in sentences[0]
    assert "Fig. 1" in sentences[0]
    assert "et al." in sentences[1]
    assert "p < 0.05" in sentences[1]


@pytest.mark.parametrize("marker", ["(DOCX)", "(TIF)", "(PDF)", "(XLSX)", "TIF", "(TIFF)", "(PNG)"])
def test_supplementary_file_markers_dropped(marker):
    chunks = chunk_paper(_paper(sections=[{"section_name": "Methods", "text": marker}]))
    assert chunks == []


def test_marker_inside_real_text_is_kept():
    chunks = chunk_paper(
        _paper(
            sections=[{"section_name": "Methods", "text": "See the supplement (DOCX) for details."}]
        )
    )
    methods = [c for c in chunks if c.section_type == "methods"]
    assert len(methods) == 1
    assert "DOCX" in methods[0].text


def test_noise_table_chunk_dropped():
    chunks = chunk_paper(_paper(sections=[{"section_name": "Table 1", "text": "(DOCX)"}]))
    assert chunks == []


def test_returns_chunk_dataclass_instances():
    chunks = chunk_paper(_paper(abstract="Some abstract."))
    assert all(isinstance(c, Chunk) for c in chunks)
