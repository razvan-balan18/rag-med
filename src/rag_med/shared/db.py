"""SQLite connection helper + schema DDL.

db schema, made by tdd
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

FAILURE_REASONS: tuple[str, ...] = (
    "missing_title",
    "no_content",
    "xml_parse_error",
    "encoding_error",
)

_DDL = """
CREATE TABLE IF NOT EXISTS papers (
    pmid             TEXT PRIMARY KEY,
    pmcid            TEXT,
    doi              TEXT,
    title            TEXT NOT NULL,
    journal          TEXT,
    year             INTEGER,
    source_type      TEXT,
    mesh_terms_json  TEXT,
    fetched_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_xml (
    pmid       TEXT PRIMARY KEY,
    raw_xml    BLOB NOT NULL,
    parsed_at  TEXT,
    FOREIGN KEY (pmid) REFERENCES papers(pmid) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS failed_papers (
    pmid            TEXT PRIMARY KEY,
    failure_reason  TEXT NOT NULL CHECK (failure_reason IN (
        'missing_title', 'no_content', 'xml_parse_error', 'encoding_error'
    )),
    attempted_at    TEXT NOT NULL
);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection with WAL + FK pragmas applied."""
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create tables idempotently."""
    conn.executescript(_DDL)
    conn.commit()
