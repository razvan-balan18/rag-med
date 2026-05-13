# Week 1 — Pre-commit chores + M1 ingest scaffold

**Goal of week:** clear the 8 pre-commit chores (Q23i), then land commits 1–7 of the M1 ingest sequence (Q22e). End-state: `tests/test_smoke_ingest.py` green against 100 COPD papers in local SQLite.

**Budget:** ~15–20 hr (Q23b half-time pace). Buffer day at end.

**Anchors:** `.claude/research/decisions.md` Q22e (commit sequence), Q23i (chore list), Q23a (dev HW), Q23j (risk flags). `.claude/research/architecture.md` §8 indexing, §13 config, §15 versioning.

---

## Day 0 — Pre-commit chores (Q23i)

**Goal:** unblock commit 1. Cheap, ~5 min each, blocking. Do all before writing any code.

| # | Chore | Where | Verify |
|---|---|---|---|
| 1 | NCBI account + API key + email | ncbi.nlm.nih.gov → Account → API Key Management | Key string in hand; email confirmed |
| 2 | Anthropic paid tier + customer spend limit **$50/mo** on dev account | console.anthropic.com → buy ≥$5 credit (gets Tier 1, $100/mo ceiling) → Settings → Limits → "Set spend limit" = $50 | Both Tier 1 status + $50 customer limit visible |
| 3 | HuggingFace account | huggingface.co → Sign up | Username confirmed; needed for model auto-download (Q23f) + M5 bundle host |
| 4 | `git init` + new public GitHub repo `rag-med` | `git init` in project root; `gh repo create rag-med --public` | `git remote -v` shows origin |
| 5 | Confirm ≥ 25 GB free disk on dev | `df -h /` | ≥ 25 GB free |
| 6 | Lock brother's labeling slot ~week 7–8 calendar | Calendar invite, 30 min slot | Invite accepted |
| 7 | Brother does same as #2 on his own account: ≥$5 credit deposit → Tier 1 → set customer spend limit **$30/mo** | Ask brother to mirror chore #2 | Confirmation from him with both values visible |

**Gold-set drafting deferred to week 7** (work-style preference: batch in one block rather than incremental). Q23i recommended incremental drafting — risk is cold-start 4–6 hr block at week 7 colliding with brother's labeling slot. Budget the time explicitly when week 7 plan is written.

**Anti-pattern:** "I'll set up the NCBI key when I need it." Commit 3 needs it. Do all 8 now.

---

## Day 1 — Commit 1: scaffolding

**LOC:** ~80. **Time:** ~2 hr.

- `pyproject.toml`: project name `rag-med`, Python ≥3.11, deps stubbed empty for now (`fastapi`, `httpx`, `pydantic-settings`, `pubmed_parser`, `pysbd`, `faiss-cpu`, `rank_bm25` come in later commits)
- `ruff` config: line length 100, ignore `E501` in tests
- `pytest` config: `tests/` discovery, `-q`
- `.env.example`: `NCBI_API_KEY=`, `NCBI_EMAIL=`, `ANTHROPIC_API_KEY=`, `HF_HOME=./data/hf_cache`
- `.gitignore`: `data/`, `.env`, `.venv/`, `__pycache__/`, `*.egg-info/`, `dist/`
- Folder skeleton per decisions.md §Q11d: `src/rag_med/{indexing,serving,eval,shared}/__init__.py`, `tests/`, `scripts/`, `static/`, `data/` (empty)
- `README.md`: one paragraph + install ritual placeholder

**Verify:** `uv venv && uv pip install -e .` succeeds; `ruff check src/` clean; `pytest` finds 0 tests.
**Commit:** `chore: scaffold project layout`.

---

## Day 2 — Commit 2: config

**LOC:** ~80. **Time:** ~2 hr.

`src/rag_med/config.py` — Pydantic Settings class loading from `.env` + `config.yaml`. Required fields:

- `ncbi_api_key: str`, `ncbi_email: EmailStr`
- `anthropic_api_key: SecretStr`
- `monthly_cap_usd: float = 15.0` (Q16 layer 1)
- `per_query_ceiling_usd: float = 0.10` (Q16 layer 3)
- `max_tokens: int = 1024` (Q16 layer 4)
- `rerank_floor: float = 0.0` (Q9b — calibrated in M4, placeholder now)
- `hf_home: Path = Path("./data/hf_cache")`
- `data_dir: Path = Path("./data")`
- `sqlite_path: Path` (computed from `data_dir`)

`tests/test_config.py`: loads with fake env, asserts defaults + types. No real keys in tests.
**Verify:** `pytest tests/test_config.py` green.
**Commit:** `feat(config): pydantic settings with cost-defense knobs`.

---

## Day 3 — Commit 3: fetch (NCBI E-utilities)

**LOC:** ~120. **Time:** ~3 hr.

`src/rag_med/indexing/ingest/pubmed.py` — `httpx` wrappers (Q22b, NOT Biopython):

- `esearch(query: str, retmax: int) -> list[str]` returns PMIDs
- `efetch_pubmed(pmids: list[str]) -> bytes` raw PubMed XML
- `efetch_pmc(pmcids: list[str]) -> bytes` raw PMC XML
- Politeness: include `api_key` + `email` in every URL. Rate-limit guard: `httpx` client with `httpx.Limits` + `asyncio.Semaphore(10)` (Q22b: 10 req/s with key).
- Retry-3x on 5xx + network errors (decisions.md §5 retry strategy).
- **Hardcoded M1 esearch query** per Q22b:
  ```
  ("Pulmonary Disease, Chronic Obstructive"[MeSH] OR "COPD"[Title/Abstract])
    AND ("2020"[Date - Publication] : "3000"[Date - Publication])
    AND "open access"[filter]
  ```

No SQLite yet — Q22e says print/dump JSON. Save raw XML to `data/raw/{pmid}.xml` for inspection.

**Smoke (manual, not in pytest yet):** `python -m rag_med.indexing.ingest.pubmed` fetches 5 PMIDs, dumps XML. Confirm files non-empty, well-formed.
**Commit:** `feat(ingest): NCBI esearch + efetch via httpx`.

---

## Day 4 — Commit 4: SQLite schema

**LOC:** ~100. **Time:** ~2 hr.

`src/rag_med/shared/db.py` — connection helper + DDL. Tables per architecture.md §11.1 (papers, paper_xml, failed_papers — chunks/embeddings come later):

- `papers (pmid PK, pmcid, doi, title, journal, year, source_type, mesh_terms_json, fetched_at)`
- `paper_xml (pmid PK FK, raw_xml BLOB, parsed_at)`
- `failed_papers (pmid PK, failure_reason TEXT, attempted_at)` — `failure_reason` enum: `missing_title | no_content | xml_parse_error | encoding_error` (Q22d)
- PRAGMAs (Q23h): `journal_mode=WAL`, `foreign_keys=ON`
- Indices: none needed yet (chunks table comes later)

`tests/test_db_schema.py`: spin up in-memory SQLite, run DDL, assert tables/pragmas present.
**Verify:** `pytest tests/test_db_schema.py` green.
**Commit:** `feat(db): SQLite schema for papers + paper_xml + failed_papers`.

---

## Day 5 — Commit 5: parse (pubmed_parser) + MPS smoke

**LOC:** ~100. **Time:** ~3 hr.

`src/rag_med/indexing/ingest/parse.py`:
- Wraps `pubmed_parser.parse_pubmed_xml` and `pubmed_parser.parse_pubmed_paragraph` (Q22c)
- Returns `dict` with `{pmid, pmcid, title, abstract, sections: [{section_name, text}], mesh_terms, journal, year, authors}`
- Salvage rule (Q22d): function returns `None` (caller writes to `failed_papers`) iff title missing OR (abstract missing AND no body sections)
- Per-chunk forgiveness: try/except around each section parse, drop failing section, keep paper

`tests/test_parse.py`: 2 fixture XMLs in `tests/fixtures/` (one PMC OA full-text, one abstract-only). Assert salvage rule + section list.

**MPS smoke (Q23j risk flag):** quick standalone script — load `microsoft/deberta-v3-large-mnli` on MPS, run dummy forward pass `("Patients had COPD.", "Patients had a respiratory condition.")`. If any op falls back to CPU loudly, document in decisions.md before pipeline commit. ~10 min, not a commit.

**Commit:** `feat(parse): pubmed_parser wrapper + salvage rule`.

---

## Day 6 — Commit 6: ingest pipeline

**LOC:** ~150. **Time:** ~3 hr.

`src/rag_med/indexing/pipeline.py` — `fetch` subcommand wiring everything:

```
esearch(query, retmax=100)
  → for each PMID:
      efetch → save paper_xml row
      parse → if salvage rule fails: insert failed_papers, continue
      else: INSERT OR IGNORE into papers (idempotent per Q22e smoke)
```

- CLI: `python -m rag_med.indexing.pipeline fetch --query-preset copd-m1 --limit 100`
- Logging: `structlog` JSON to stdout (architecture.md §13) — one log line per paper with `{pmid, status: fetched|parsed|salvaged|failed, elapsed_ms}`
- Idempotency: `INSERT OR IGNORE` so re-running adds 0 rows on second pass
- Retry-3x on network already in fetch module; nothing extra here

**Commit:** `feat(pipeline): fetch+parse+insert with salvage rule`.

---

## Day 7 — Commit 7: smoke run + verify

**LOC:** ~50 (test only). **Time:** ~2 hr.

Run the full M1 fetch:
```
python -m rag_med.indexing.pipeline fetch --query-preset copd-m1 --limit 100
```

Expected ~30 s wall time at 10 req/s with NCBI key. Watch SQLite fill in real time:
```
watch -n 2 "sqlite3 data/sqlite.db 'SELECT COUNT(*) FROM papers, (SELECT COUNT(*) FROM failed_papers)'"
```

Write `tests/test_smoke_ingest.py` per Q22e smoke gates:

| Gate | Assert |
|---|---|
| Volume | `SELECT COUNT(*) FROM papers >= 95` (≤ 5% salvage loss) |
| Title | All rows `title IS NOT NULL` |
| Full-text | `>= 80` rows have `pmcid IS NOT NULL` |
| Body XML | `>= 80` rows have body XML in `paper_xml` |
| Failure budget | `SELECT COUNT(*) FROM failed_papers < 5` |
| Idempotency | Run pipeline twice; row count identical |

`pytest tests/test_smoke_ingest.py` green = **M1 ingest done.** Commit 7 = `test(smoke): M1 ingest gate green on 100 COPD papers`.

**If any gate fails:** triage `failed_papers.failure_reason` distribution. Decide whether to tighten salvage rule (Q22d) or accept the loss honestly and document in decisions.md. Do not skip the gate.

---

## Week 1 done — what's true at end

- ✅ All 8 pre-commit chores cleared.
- ✅ 100 COPD papers ingested locally, ≥ 80 with PMC full-text, salvage budget respected.
- ✅ Smoke test green; idempotent.
- ✅ Repo public on GitHub; commits 1–7 pushed.
- ✅ MPS smoke for DeBERTa confirmed (or fallback documented).

## Anti-patterns to refuse

- Vertical-slicing chunking + embedding + retrieval into week 1 (Q22e last paragraph). M1 builds linearly: ingest first, prove it solid, then chunk next week.
- Indexing + serving simultaneously on the 24 GB M5 Pro (Q23a). Don't.
- Skipping the smoke gates "because the count looks right." Run them.
- Mocking the database in `test_smoke_ingest.py`. Real SQLite, real schema, real rows.

## Next week preview

Week 2 = commits 8–14 of M1: chunker (IMRaD + 350-DeBERTa-token target), embedder (MedCPT-Article), FAISS `IndexFlatIP` build, BM25 build, retrieval skeleton, end-to-end `/ask` with **mocked verifier returning all-green** (M1 acceptance per Q14). Then M2 starts: real NLI + Haiku judge.
