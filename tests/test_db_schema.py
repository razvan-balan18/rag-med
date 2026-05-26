import sqlite3

import pytest

from rag_med.shared.db import FAILURE_REASONS, connect, init_schema


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def test_schema_creates_expected_tables(tmp_path):
    conn = connect(tmp_path / "t.db")
    init_schema(conn)
    assert {"papers", "paper_xml", "failed_papers"} <= _table_names(conn)


def test_pragmas_set_on_file_connection(tmp_path):
    conn = connect(tmp_path / "t.db")
    init_schema(conn)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_in_memory_ddl_runs(tmp_path):
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    init_schema(conn)
    assert {"papers", "paper_xml", "failed_papers"} <= _table_names(conn)


def test_paper_xml_foreign_key_enforced(tmp_path):
    conn = connect(tmp_path / "t.db")
    init_schema(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO paper_xml (pmid, raw_xml, parsed_at) VALUES (?, ?, ?)",
            ("999", b"<xml/>", "2026-01-01T00:00:00Z"),
        )
        conn.commit()


def test_failure_reason_check_constraint(tmp_path):
    conn = connect(tmp_path / "t.db")
    init_schema(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO failed_papers (pmid, failure_reason, attempted_at) VALUES (?, ?, ?)",
            ("1", "bogus_reason", "2026-01-01T00:00:00Z"),
        )
        conn.commit()
    for reason in FAILURE_REASONS:
        conn.execute(
            "INSERT INTO failed_papers (pmid, failure_reason, attempted_at) VALUES (?, ?, ?)",
            (f"pmid-{reason}", reason, "2026-01-01T00:00:00Z"),
        )
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM failed_papers").fetchone()[0]
    assert count == len(FAILURE_REASONS)


def test_papers_round_trip(tmp_path):
    conn = connect(tmp_path / "t.db")
    init_schema(conn)
    conn.execute(
        "INSERT INTO papers (pmid, pmcid, title, year, source_type, fetched_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("12345", "PMC1", "Test paper", 2024, "full_text", "2026-05-24T00:00:00Z"),
    )
    conn.commit()
    row = conn.execute("SELECT pmid, pmcid, title, year FROM papers WHERE pmid='12345'").fetchone()
    assert row == ("12345", "PMC1", "Test paper", 2024)


def test_papers_title_not_null(tmp_path):
    conn = connect(tmp_path / "t.db")
    init_schema(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO papers (pmid, fetched_at) VALUES (?, ?)",
            ("777", "2026-01-01T00:00:00Z"),
        )
        conn.commit()


def test_chunks_table_created(tmp_path):
    conn = connect(tmp_path / "t.db")
    init_schema(conn)
    assert "chunks" in _table_names(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(chunks)")}
    assert {
        "chunk_id",
        "pmid",
        "section_type",
        "ordinal",
        "text",
        "n_deberta_tokens",
        "n_medcpt_tokens",
    } <= cols


def test_chunks_foreign_key_to_papers(tmp_path):
    conn = connect(tmp_path / "t.db")
    init_schema(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO chunks (chunk_id, pmid, section_type, ordinal, text, "
            "n_deberta_tokens, n_medcpt_tokens) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("999_methods_00", "999", "methods", 0, "x", 1, 1),
        )
        conn.commit()


def test_chunks_indices_exist(tmp_path):
    conn = connect(tmp_path / "t.db")
    init_schema(conn)
    indices = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='chunks'"
        )
    }
    assert any("pmid" in i for i in indices)
    assert any("section_type" in i for i in indices)


def test_chunks_round_trip(tmp_path):
    conn = connect(tmp_path / "t.db")
    init_schema(conn)
    conn.execute(
        "INSERT INTO papers (pmid, title, fetched_at) VALUES (?, ?, ?)",
        ("42", "P", "2026-05-25T00:00:00Z"),
    )
    conn.execute(
        "INSERT INTO chunks (chunk_id, pmid, section_type, ordinal, text, "
        "n_deberta_tokens, n_medcpt_tokens) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("42_methods_00", "42", "methods", 0, "body", 10, 12),
    )
    conn.commit()
    row = conn.execute(
        "SELECT chunk_id, pmid, section_type, ordinal, n_deberta_tokens FROM chunks "
        "WHERE chunk_id='42_methods_00'"
    ).fetchone()
    assert row == ("42_methods_00", "42", "methods", 0, 10)


def test_insert_or_ignore_idempotent(tmp_path):
    conn = connect(tmp_path / "t.db")
    init_schema(conn)
    for _ in range(3):
        conn.execute(
            "INSERT OR IGNORE INTO papers (pmid, title, fetched_at) VALUES (?, ?, ?)",
            ("42", "Same paper", "2026-05-24T00:00:00Z"),
        )
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 1
