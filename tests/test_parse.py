from pathlib import Path

from rag_med.indexing.ingest.parse import parse

FIXTURES = Path(__file__).parent / "fixtures"


def _xml(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def test_full_text_paper_returns_dict_with_sections():
    out, reason = parse(_xml("pmc_full_text.xml"))
    assert reason is None
    assert out is not None
    assert out["pmid"] == "11111111"
    assert out["pmcid"] == "PMC1111111"
    assert out["title"].startswith("Long-acting bronchodilators")
    assert "COPD patients benefit" in out["abstract"]
    section_names = [s["section_name"] for s in out["sections"]]
    assert "Introduction" in section_names
    assert "Methods" in section_names
    assert "Results" in section_names
    intro = next(s for s in out["sections"] if s["section_name"] == "Introduction")
    assert "leading cause of mortality" in intro["text"]
    assert "first-line therapy" in intro["text"]
    assert out["journal"] == "Test Journal of Pulmonology"
    assert out["year"] == 2023
    assert "Pulmonary Disease, Chronic Obstructive" in out["mesh_terms"]
    assert "Spirometry" in out["mesh_terms"]


def test_abstract_only_paper_returns_dict_with_empty_sections():
    out, reason = parse(_xml("pmc_abstract_only.xml"))
    assert reason is None
    assert out is not None
    assert out["pmid"] == "22222222"
    assert out["pmcid"] == ""
    assert out["title"].startswith("Inhaled corticosteroids")
    assert "12 RCTs" in out["abstract"]
    assert out["sections"] == []
    assert out["year"] == 2024


def test_salvage_drops_paper_without_title():
    out, reason = parse(_xml("pmc_no_title.xml"))
    assert out is None
    assert reason == "missing_title"


def test_salvage_drops_paper_without_abstract_and_body():
    out, reason = parse(_xml("pmc_no_content.xml"))
    assert out is None
    assert reason == "no_content"


def test_unparseable_xml_returns_xml_parse_error():
    out, reason = parse(b"not xml at all")
    assert out is None
    assert reason == "xml_parse_error"
