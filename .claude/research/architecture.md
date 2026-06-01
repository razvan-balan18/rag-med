# Pneumology RAG — Architecture

Companion to `decisions.md`. Where `decisions.md` answers **what** we picked (FastAPI, MedCPT, FAISS, NLI hybrid verifier), this document answers **how** the system behaves at runtime: process shape, data flow, error handling, concurrency, caching, startup, indexing, eval, frontend, config, logging, versioning, testing, dev loop.

Last updated: 2026-05-11 (Q23 closures from final grill: dev HW, M3/M5 reorder, FAISS=FlatIP, pysbd, model-weight auto-download)

---

## 1. System overview

Three runtime phases, three distinct programs sharing the same Python codebase:

| Phase | Program | Where it runs | Frequency | Purpose |
|---|---|---|---|---|
| 1. Indexing | `python -m rag_med.indexing.pipeline` | Dev machine (**M5 Pro / 24 GB / 15 cores, MPS**) | Once + occasional refresh | Build SQLite + FAISS + BM25 artifacts from PubMed/PMC |
| 2. Serving | FastAPI process inside Docker | Brother's machine (**i7-155H / 32 GB, CPU-only**) | Always-on | Answer questions with verified citations |
| 3. Eval | `python -m eval` | Dev machine | On demand | Measure retrieval + faithfulness against gold set |

Phases share `src/rag_med/shared/` (tokenizer, schemas, DB helpers).

---

## 2. Process & runtime model — Phase 2 (Serving)

**Single FastAPI process. Everything in-memory at startup.**

One Python process inside one container. On boot, loads:

| Resource | Size | Load time (cached) |
|---|---|---|
| FAISS index | ~600 MB | ~3s |
| BM25 serialized index | ~200 MB | ~1s |
| MedCPT-Query-Encoder | ~400 MB | ~1s |
| MedCPT-Cross-Encoder | ~400 MB | ~5s |
| NLI deberta-v3-large | ~1.5 GB | ~10s |
| SQLite connection | — | instant |
| **Total RAM** | **~3–4 GB** (of 32 GB available on i7-155H, Q13) | **~12–15s parallel cold start** |

All models stay loaded for process lifetime. No lazy load. No model-server processes. No GPU assumptions (covered separately in deferred Q15 — hardware).

**Concurrency model: fully serialized.** One query at a time across the whole process. Single-user app, simpler reasoning, second tab waits for first to finish. Upgrade to free-flowing asyncio later if real contention appears.

---

## 3. Request lifecycle — `/ask`

Brother types question → backend runs the following pipeline → SSE stream back to browser.

### 3.1 Pipeline stages (with parallelism)

```
question
   │
   ├──→ MedCPT-Query embed ──→ FAISS search ──┐
   │     (asyncio.to_thread)                   │
   │                                           ├──→ RRF fuse ──→ Cross-Encoder rerank ──→ top 10
   └──→ BM25 search ──────────────────────────┘   (one batch)
        (asyncio.to_thread)
                                                                                 │
                                                                                 ▼
                                                                    build prompt (string)
                                                                                 │
                                                                                 ▼
                                                                Claude streaming call
                                                                                 │
                                                                                 ▼
                                                              tokens accumulate locally
                                                              + emit `token` SSE events
                                                                                 │
                                                                                 ▼
                                                              full answer text complete
                                                                                 │
                                                                                 ▼
                                                                  sentence split (regex)
                                                                                 │
                                                                                 ▼
                                                              NLI batch over all sentences
                                                                                 │
                                                                                 ▼
                                                              judge LLM (parallel, only borderline)
                                                                                 │
                                                                                 ▼
                                                              emit `verified` SSE event
                                                                                 │
                                                                                 ▼
                                                                       emit `done`
```

**Parallelism decisions:**
- FAISS + BM25 run in parallel via `asyncio.gather(to_thread(faiss), to_thread(bm25))`.
- NLI runs as one batched forward pass over all sentences (not loop).
- Judge LLM calls (only sentences in borderline NLI band 0.3–0.9) run in parallel via `asyncio.gather`.

### 3.2 SSE event schema

Streaming protocol = **token-streaming with verify-at-end** (not incremental sentence verification). v1 known limitation: brother reads streamed answer before dots arrive (~5–8 s gap). Acceptable for v1; **post-v1 path is incremental verify per sentence** (revisit if a GPU shows up or if user feedback says trust-timing matters).

```
event: retrieved   data: {chunks: [{chunk_id, paper_pmid, section_type, text_preview}, ...]}
event: token       data: {text: "Patients "}
event: token       data: {text: "with COPD..."}
...
event: verified    data: {sentences: [{idx, text, citations, status, confidence}, ...]}
event: done        data: {}
```

On any fatal error:
```
event: error  data: {stage, code, message, retryable}
event: done   data: {}
```
Single error event ends stream. No mid-stream warning events. Frontend shows banner + retry button.

### 3.3 Stage-by-stage rationale

| Step | Type | Why |
|---|---|---|
| 1. MedCPT-Query embed | Standard RAG | Convert question to vector for FAISS |
| 2. FAISS search | Standard RAG | Semantic top-50 |
| 3. BM25 search | Hybrid retrieval | Exact-token recall (rare medical terms FAISS misses) |
| 4. RRF fuse | Hybrid retrieval | Combine two ranked lists, score-scale-agnostic |
| 5. Cross-Encoder rerank | Two-stage retrieval | Refine top-50 → top-10 via question-chunk joint scoring |
| 6. Build prompt | Standard | Stuff chunks + question + system rules into template |
| 7. Claude stream | Standard | Generate answer with `[n]` citations |
| 8. Sentence split | Verifier prep | Each sentence = one verifiable claim. **Library = `pysbd` (Q23e locked)** — handles `2.5 mg`, `Fig. 1`, `et al.`, `e.g.`, `p < 0.05`, `vs.` out of the box; ~2 ms/answer |
| 9. NLI batch | **Differentiator** | Cheap entailment check; confident verdicts settled here. **Multi-cite (Q1) = AND-of-singles**: NLI runs once per `(sentence, cited_chunk)` pair |
| 10. Judge LLM | **Differentiator** | Borderline NLI cases escalated to Claude **Haiku 4.5** judge (Q16) |
| 11. Emit verified | Differentiator | Frontend paints green/yellow/red dots; `failure_kind` tag on each non-supported sentence (Q6) |

---

## 4. API surface

Four routes total. Minimal but enables full citation UX.

```
POST /ask                  → SSE stream (Section 3.2)
  body: {question: str, top_k: int = 10}

GET /chunks/{chunk_id}     → JSON
  response: {chunk_id, paper_pmid, section_type, text,
             prev_chunk_id, next_chunk_id}

GET /papers/{pmid}         → JSON
  response: {pmid, pmcid, doi, title, authors, journal, year,
             abstract, mesh_terms, source_type}

GET /health                → JSON
  response: {status: "ok"|"loading"|"error", paper_count,
             chunk_count, index_built_at, models_loaded: bool,
             git_sha}
```

`/chunks` and `/papers` exist so the frontend citation panel can fetch full chunk text + paper metadata on click — without these, brother would have to manually search PubMed for every `[n]` reference.

---

## 5. Failure modes

### 5.1 Error taxonomy

| Stage | Failure | Behavior |
|---|---|---|
| Validate question | Empty / too long | 400, error event |
| Embed query | Model crash | 500, error event |
| Rerank floor (Q9b) | Top-1 rerank score < `rerank_floor` | Hard refuse: send `answer` event with fixed string `"The retrieved evidence does not address this question."` — skip Claude entirely. Logged with `refusal: "hard"` in `query_traces` |
| Rerank floor per-chunk | Chunks above top-K but score < floor | Drop from `final_chunks` before prompt building (fewer garbage chunks → less hallucination risk) |
| Rerank model crash | Model error | 500, error event |
| Claude API | 429 / 5xx / network | Anthropic SDK auto-retries (3x exponential backoff). Surface error only after exhausted |
| Monthly cap reached | `query_traces.cost_usd` sum > `monthly_cap_usd` | Clean error event: `"monthly cap reached, raise in config.yaml"`. Predictable failure beats silent 429s (Q16) |
| 80% MTD warning | sum > 0.8 × `monthly_cap_usd` | NOT a fatal event. Log loud, set `/health.cost_warning=true`, frontend yellow banner. Soft signal before hard fail (Q21) |
| Per-query ceiling exceeded | `generator_cost_usd + judge_cost_usd > per_query_ceiling_usd` ($0.10) mid-pipeline | Abort current query: emit `error  data: {code: "per_query_ceiling", message: "single-query cost exceeded $0.10 ceiling — likely a bug, check logs", retryable: false}`. No retry — this is a code-bug signal, not a transient (Q21) |
| Anthropic console hard limit | API returns `quota_exceeded` from console-side cap | Same UX as 429 path. Distinguish in logs via response body. Console caps: $50 dev / $30 brother (Q21) |
| Mid-stream Claude disconnect | Network drop | Whatever tokens emitted, then `error` event, no verifier run |
| Hallucinated `[n]` | Cite resolves to no chunk_id (Q6a) | Sentence labeled `status='unsupported'`, `failure_kind='fabricated_citation'`. Pipeline continues |
| No-citation sentence | Sentence has no `[n]` (Q6b) | `status='unsupported'`, `failure_kind='no_citation'` |
| Sentence split / NLI / Judge | Crash | Verifier failure → emit `verified` event with `status: 'unknown'`, `failure_kind: 'verifier_crash'` for affected sentences. Answer still shown |
| SSE write to client | Browser closed | Cancel pipeline, log abort |

**Retry strategy:** rely on Anthropic SDK's built-in `max_retries=3` with exponential backoff. No custom retry code.

**Refuse strategy:** hard-refuse on empty retrieval (cheap, correct, fast). Soft-refuse phrasing (per Q12b in decisions.md) handled inside Claude generation when *some* chunks exist but partially cover the question.

---

## 6. Caching & state

**No answer cache.** Each query hits full pipeline. Reasons: rare exact-repeat queries, eval reproducibility, cache hides bugs.

**Embedding cache only.** `functools.lru_cache(maxsize=1000)` on the question-embedding function. Helps eval re-runs and any lucky repeats. ~5 lines of code.

**Stateless turns.** No multi-turn conversation memory. Each `/ask` independent. Brother re-states context if needed. (Multi-turn deferred — would interact awkwardly with verifier.)

---

## 7. Startup & readiness

### 7.1 Boot sequence

1. Read `.env` (secrets) and `config.yaml` (tunables) via Pydantic Settings.
2. Open SQLite connection.
3. **Parallel load** via `asyncio.gather` + `to_thread`:
   - FAISS index → RAM
   - BM25 serialized index → RAM
   - MedCPT-Query-Encoder weights (from `HF_HOME` cache; auto-downloaded on first boot, Q23f)
   - MedCPT-Cross-Encoder weights (same)
   - NLI deberta-v3-large weights (same)
4. Initialize Anthropic SDK client (no key validation ping).
5. Open HTTP port, mark `/health` as `ok`.

**Model weight distribution (Q23f locked):** weights are NOT baked into the Docker image. `transformers` library auto-pulls from HuggingFace Hub at first load. `HF_HOME` mounted as a docker volume → cache persists across container restarts. First boot ~5 min wait on broadband; subsequent boots instant. Image stays ~500 MB. Brother needs internet at first start (already required for Anthropic API).

### 7.2 Failure semantics

- **Strict on indexes/models.** Missing FAISS file, corrupt BM25 file, model weights fail to load → log error, exit 1, container restart loop. Brother sees logs and knows to fix.
- **Soft on API key.** Missing/invalid `ANTHROPIC_API_KEY` does NOT block boot. `/health` reports `status: "ok"` but `/ask` returns clean error event: `"set ANTHROPIC_API_KEY in .env"`. Avoids crash loop on first-run mistake.

### 7.3 Readiness signaling

Single endpoint: `/health`.
- Returns `503 {status: "loading"}` during boot.
- Returns `200 {status: "ok", ...}` once all models loaded.
- `docker compose` healthcheck polls every 5s.
- Frontend polls on page load, grays out input until `200`.

---

## 8. Indexing pipeline — Phase 1

Runs on dev machine. ~4–6 hours at full M5 corpus scale. Output uploaded to HuggingFace as a single bundle for brother to download.

**M1 toy ingest** (locked Q22): narrow-topic full-text-only sub-corpus to bring up the end-to-end skeleton. Topic = COPD (`"Pulmonary Disease, Chronic Obstructive"[MeSH]` + `"open access"[filter]` + 2020+ cutoff), 100 papers via E-utilities only. Day-1 prereq: NCBI API key + email registered (`NCBI_API_KEY`, `NCBI_EMAIL` in `.env`) → 10 req/s vs 3 req/s default. HTTP via `httpx` directly (no Biopython). XML parsing via `pubmed_parser` library (M5 escape hatch: hand-rolled `lxml`). Day-1 commit sequence + smoke-test pass criteria detailed in `decisions.md §Q22e`.

### 8.1 Staged shape

Five separate commands, each writes outputs to disk so re-running a stage is free:

```
python -m rag_med.indexing.pipeline fetch     # PubMed E-utilities + PMC OA → SQLite
python -m rag_med.indexing.pipeline chunk     # IMRaD chunking → SQLite chunks table
python -m rag_med.indexing.pipeline embed     # MedCPT-Article-Encoder (MPS) → FAISS IndexFlatIP file
python -m rag_med.indexing.pipeline bm25      # tokenize + build inverted index → bm25 file
python -m rag_med.indexing.pipeline manifest  # write index_manifest row, build bundle
```

**FAISS index type (Q23d locked):** `IndexFlatIP`. Exact inner-product search, no training, ~750 MB for ~250k × 768-dim vectors. Search <100 ms on M5 Pro / acceptable on i7-155H. Reasoning: at this scale + serialized single-user requests, approximate indexes (HNSW, IVF) optimise something we don't need. Eval reproducibility also cleaner without approx-recall variance.

### 8.2 Resumability

SQLite is the progress log. Each PMID's success writes a row to `papers`. Restart re-queries `SELECT pmid FROM papers WHERE pmid IN (...)` and skips already-done IDs. Single source of truth, no separate progress file.

`INSERT OR IGNORE` makes inserts idempotent — re-running fetch over already-ingested PMIDs is a no-op.

### 8.3 Per-paper failure handling

- **Network errors** (timeout, 5xx): bounded retry — 3 attempts, exponential backoff.
- **Parse errors** (malformed XML, missing required field): no retry, log to `failed_papers` table with `failure_reason` (`missing_title` | `no_content` | `xml_parse_error` | `encoding_error`), continue.

**Salvage rule (locked Q22d) — minimum viable record.** Keep paper iff:
- `pmid` present (defensive, always true from `efetch`)
- `title` present
- (`abstract` present OR ≥1 body section parsed)

**Per-chunk forgiveness.** If one section/table fails to parse, drop that chunk only, keep the rest of the paper. Don't let one bad table sink an entire paper.

Pipeline never crashes on a single bad paper. After completion, brother (or dev at M5) reviews `failed_papers` table to decide which need manual attention. If `failed_papers / total > 2%` at M5, write more aggressive salvage rules.

---

## 9. Eval architecture — Phase 3

### 9.1 How eval calls the system

**Direct import.** `from rag_med.serving.retrieve import retrieve` etc. Same process, no HTTP, no SSE parsing. Eval is testing retrieval/generation/verifier quality, not HTTP layer.

### 9.2 Results storage

Per-run Parquet file: `results/run_<git_sha>_<timestamp>.parquet`. Columns include question, top-k chunk IDs, expected chunk IDs, retrieval metrics, faithfulness, latency, tags. Git-friendly, immutable, columnar for fast pandas.

### 9.3 Comparison

`python -m eval compare <run1.parquet> <run2.parquet>` outputs:
- Per-metric delta table (Recall@10 +0.03, etc.).
- List of questions where verdict flipped (regression debug).
- Top 20 failing questions with full traces.

### 9.4 Cost control

Three modes:

| Mode | Flag | What runs | Cost (per full eval) |
|---|---|---|---|
| Retrieval-only | default | Steps 1–5 of pipeline | $0 |
| Full | `--full` | Full pipeline including Claude + judge, **via Anthropic batch API enforced** | ~$28 |
| Mock LLM | `--mock-llm` | Full pipeline with cached Claude responses | $0 |

Default = retrieval-only — most metrics (Recall@10, nDCG, MRR) don't need generation. Full eval gated behind flag for milestone runs. Mock-LLM mode replays cached Claude responses on a fixed 30-question subset — for testing eval *code* without paying.

**`--full` safety guards (Q21 amendment):**
- **Confirmation prompt.** Typing `python -m eval --full` prints expected cost + question count, requires `YES` typed back before running. Prevents fat-finger.
- **Batch API not optional.** `--full` flag enforces batch endpoint. If batch unavailable, eval fails loud rather than silently using non-batch endpoint (closes $60-vs-$28 footgun).
- **Run log.** Before starting, append `(timestamp, git_sha, expected_cost_usd, gold_set_size)` to `eval/runs.jsonl`; after completion, append `actual_cost_usd`. `tail eval/runs.jsonl` answers "did I already run this?" without grep-ing parquet files.

Total dev budget M1–M6: **~$140** (5 `--full` runs × $28 batch = $140 floor; ad-hoc dev queries push real budget closer to $150).

---

## 10. Frontend architecture

Plain HTML + vanilla JS + ES modules + SSE. No build step. Served by FastAPI via `StaticFiles`.

### 10.1 File layout

```
static/
├── index.html       # markup, single-page
├── app.js           # entry; wires up handlers, kicks off SSE
├── sse.js           # SSE client, event parsing
├── render.js        # DOM render functions (answer, dots, citation panel)
├── state.js         # single state object + setState pattern
├── api.js           # fetch wrappers for /chunks, /papers, /health
└── style.css
```

`<script type="module" src="app.js">` — native ES modules, no bundler.

### 10.2 State model

Single object, all updates through `setState`, render reads state, render writes DOM.

```js
let state = {
  question: '',
  chunks: [],            // top 10 retrieved
  answerSentences: [],   // [{text, citations: [1,3], status: 'pending'|'green'|'yellow'|'red'}]
  streaming: false,
  error: null
};
function setState(patch) { state = {...state, ...patch}; render(); }
```

Predictable: state changes always render. No DOM-as-source-of-truth bugs.

### 10.3 Streaming render strategy

- During `token` events: append to text node directly (`textNode.appendData(chunk)`). No `innerHTML` clobber → no flicker.
- On final `verified` event: re-render full answer block with sentence dots inline. One re-render acceptable.

### 10.4 Citation panel

User clicks `[3]` in answer. Frontend reads chunk_id 3 from `state.chunks`, then hits `/chunks/{id}` and `/papers/{pmid}` for full text + metadata. Side panel shows: paper title, authors, journal/year, section type, full chunk text, link to PubMed.

---

## 11. Configuration & secrets

### 11.1 Two files, two purposes

- **`.env`** (gitignored, copied from `.env.example`): secrets + per-machine identity.
  ```
  ANTHROPIC_API_KEY=sk-ant-...
  NCBI_API_KEY=...                      # Q22b — register day 1, free, 10 req/s vs 3
  NCBI_EMAIL=you@example.com            # Q22b — required by NCBI on every E-utilities request
  ```
- **`config.yaml`** (committed): tunable params, structured.
  ```yaml
  retrieval:
    top_k: 10
    rrf_k: 60
    rerank_candidates: 50
    rerank_floor: 0.0          # Q9b: hardcoded v1; calibrate empirically in M4
  verifier:
    nli_threshold_high: 0.9
    nli_threshold_low: 0.3
  llm:
    generator_model: claude-sonnet-4-6   # Q16
    judge_model: claude-haiku-4-5        # Q16
    max_tokens: 1024                     # Q21 — bounds single-completion length, ≈750 words
  cost:
    monthly_cap_usd: 15                  # Q16 app-level cap
    per_query_ceiling_usd: 0.10          # Q21 — abort single query if exceeds (5× typical)
    warn_threshold_pct: 0.80             # Q21 — log loud + /health.cost_warning when MTD > 80% of cap
  paths:
    faiss: /data/faiss.index
    faiss_chunk_ids: /data/faiss.chunk_ids.json   # sidecar: FAISS row idx → chunk_id
    sqlite: /data/sqlite.db
    bm25: /data/bm25.pkl
    bm25_chunk_ids: /data/bm25.chunk_ids.json     # sidecar: BM25 doc idx → chunk_id
  bundle:
    manifest_url: https://huggingface.co/datasets/<user>/rag-med-pneumology-bundle/raw/main/manifest.json
  ```

**Anthropic console hard limits (Q21).** Set on the Anthropic dashboard, NOT in `config.yaml` (per-account, not per-deploy). **Dev key: $50/mo** (one `--full` eval at $28 + headroom). **Brother's key: $30/mo** (2× app-cap as backstop). Defends against any code bug that bypasses `monthly_cap_usd` enforcement.

### 11.2 Code access

Pydantic Settings (`pydantic-settings` package) loads both files into a single typed `Settings` object. Validation runs at boot — bad config fails `/health` with a clean message.

Module-level singleton: `from rag_med.config import settings`. Tests override via cacheable getter (`@lru_cache get_settings()`).

---

## 12. Logging & observability

### 12.1 Two destinations + one CLI

1. **Structured JSON to stdout.** Every log line = JSON object via `structlog`. `docker logs rag-med | jq` for filtering. Real-time tailing during dev.
2. **SQLite `query_traces` table.** One row per query, persistent across restarts. SQL-queryable forensics.
3. **`python -m rag_med cost` CLI (Q21).** Prints MTD spend (generator vs judge split), days remaining in month, projected end-of-month at current rate. Read-only SQL over `query_traces`. Run it whenever curious, on either machine.

### 12.2 Granularity

- **Default (LOG_LEVEL=info):** summary-level rows. Question, chunk IDs (not text), sentence verdicts, latency, error, **per-query cost split** (`generator_cost_usd`, `judge_cost_usd`), **section_type histogram** of `final_chunks` (Q7 instrumentation), `refusal` (`null | "hard" | "soft"`). ~1–2 KB per row.
- **Debug (LOG_LEVEL=debug):** full traces. Top-50 + scores, top-10 + scores, full prompt, full answer, per-sentence per-cited-chunk NLI confidence + judge verdict + `failure_kind`. ~20–50 KB per row.

Chunk text never duplicated into traces — already in `chunks` table, just store IDs.

### 12.3 Frontend visibility

Collapsible "Retrieved chunks" section per answer. Shows top-10 chunks (paper, section, snippet) — no internal scores. Researcher-friendly: brother can spot a relevant paper the system passed over. Internal debug (RRF scores, NLI confidence) lives only in backend logs.

---

## 13. Versioning & artifact identity

### 13.1 Bundle versioning

External version: single semver string (`v1.2.0`). **Hosted on HuggingFace Datasets, public repo** (Q17). Brother's app boots, async-fetches `manifest.json` from the public URL, compares `bundle_version` to local. If mismatch + remote newer → set `/health.update_available=true` + log warning. Frontend banner on next page load. **No auto-update** — brother runs `./scripts/download_index.sh` when convenient. Full re-download (~10 GB, no diffs v1).

### 13.2 Internal manifest

Inside each bundle, `index_manifest` SQLite row records:
- `bundle_version`
- `built_at`
- `embed_model` (e.g., `ncbi/MedCPT-Article-Encoder`) + `embed_model_revision` (HF commit SHA)
- `chunker_git_sha` (the chunking code that produced these chunks)
- `paper_count`, `chunk_count`

Manifest is for debugging + eval reproducibility, never for runtime decisions.

### 13.3 Code versioning

Git SHA of HEAD commit baked in at startup. Stamped on every log line, every `query_traces` row, every eval Parquet file. Single line of code (`subprocess.check_output(['git', 'rev-parse', 'HEAD'])` cached at import). No manual semver bumps.

### 13.4 Cache & trace invalidation

**No auto-invalidation.** Code change → new git SHA stamped on new rows. Old rows preserved, queryable, comparable across versions. Eval comparisons across versions are a *feature* (regression detection), not a bug.

---

## 14. Testing

Minimum-viable, all mocked, fast suite.

### 14.1 Approach

- **Mock everything in tests.** Fake embedder returns canned vectors. Fake NLI returns canned labels. Mock Anthropic client. No model loading in tests.
- **Hand-crafted synthetic XML fixtures.** `tests/fixtures/*.xml` — ~5 fake papers you write. No licensing concern, no commit-size noise.
- **Strict separation: tests vs eval.** Tests = code correctness. Eval = quality metrics. No smoke-eval in CI.

### 14.2 What gets tested

~30 unit tests, all pure-function:

1. Tokenizer keeps `IL-4`, `FEV1/FVC`, `CD8+`; drops stopwords.
2. IMRaD chunker splits at `## Methods` / `## Results`; tags sections correctly.
3. Chunker respects 300–500 token target; doesn't break sentences.
4. RRF fusion math.
5. Sentence splitter (regex on `.!?`).
6. Citation parser extracts `[1][3]` correctly.
7. Verifier dispatches to NLI vs judge based on confidence band.
8. Prompt builder includes all top-k chunks with metadata header.

Real-model integration tests deferred until something breaks.

---

## 15. Dev loop & CI

- **Local:** `pytest` manual on save. No watcher, no pre-commit hook (would block WIP commits).
- **CI:** GitHub Actions on push — `ruff check` + `ruff format --check` + `pytest`. ~3 min runs. Free for public repos.
- **Lint/format:** `ruff` only. Replaces black + isort + flake8. Single dep, sensible defaults, ~1 line config.

---

## 16. Summary diagrams

### 16.1 Three-process view

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│  Phase 1        │         │  Phase 2        │         │  Phase 3        │
│  Indexing       │         │  Serving        │         │  Eval           │
│  (dev machine)  │         │  (brother's box)│         │  (dev machine)  │
└────────┬────────┘         └────────┬────────┘         └────────┬────────┘
         │                            │                            │
         ▼                            ▼                            ▼
   PubMed/PMC                   FastAPI process              Direct imports
   ↓                            (single process,             of Phase 2 code
   SQLite + FAISS               serial requests,             ↓
   + BM25 bundle                ~3-4GB RAM)                  Parquet results
   ↓                            ↑                            ↓
   Upload to S3                 Brother downloads             Comparison script
                                bundle on first boot
```

### 16.2 Phase 2 single-request flow

See Section 3.1.

---

## 17. Open architecture questions (deferred)

Most items closed in May 10–11 grills. **Closed by Q23 (2026-05-11):** FAISS index type → `IndexFlatIP`; sentence splitter → `pysbd`; model weight distribution → auto-download from HF Hub + `HF_HOME` docker volume; SQLite schema sane defaults (WAL on, FK on, indices on `chunks(pmid)` + `chunks(section_type)`, no `query_traces` rotation v1).

**Remaining open (not v1 concerns):**

- **Ingestion ops residual (Q19):** dedup ordering (PMID/PMCID/DOI) for multi-source future; incremental refresh triggers.

## 18. Milestones (locked, decisions.md §Q14 + Q23g reorder)

Vertical slice through all three phases on day 3, then thicken outward. Each M is a working system; if a milestone slips, the system at M-1 still runs. **Q23g reorder (2026-05-11):** scaling to 150k corpus moved from M5 to M3 — M4 brother-labels against retrievals, which requires the broad corpus.

**Calendar pace (Q23b):** ~15–20 hr/wk → M1–M6 over **~10–12 calendar weeks**. Milestone numbers below are work units, not weeks.

```
M1   end-to-end skeleton on toy 100-paper COPD corpus + mock verifier
M2   real verifier (NLI + Haiku judge) wired up
M3   eval harness skeleton (retrieval-only, synthetic gold set) + SCALE CORPUS TO 150k
M4   brother labels 50 dev-authored questions; first --full eval; calibrate rerank_floor
M5   polish + bundle to HuggingFace + version-mismatch banner
M6   brother deploys on his i7-155H; Q15 gates verified
buffer  slack
```

**Side commitments tied to this:**
- Schedule brother's labeling for ~calendar week 7–8 — arrange before commit 1, not at M4 boundary.
- 50-q gold set dev-authored from **25 guidelines (GOLD/GINA/ATS-ERS) + 25 Cochrane** (Q23c). Draft incrementally during M1–M3; do not leave to M4.
- Reserve `--full` eval runs to milestone calibration only: M2 baseline, M4 first calibration, M5 post-scale, M6 acceptance, +1 slack = **5 runs total budget**.

## 19. Acceptance gates (locked, decisions.md §Q15)

Five gates. Eval harness reports against them in M4 onward. M6 ships only when all five are met or honestly marked "below target" in writeup.

| Gate | Metric | Target |
|---|---|---|
| Retrieval | Recall@10, paper-level, strict | ≥ 0.65 on brother's 50-q set |
| Faithfulness | % sentences `supported` | ≥ 0.80 |
| Refusal honesty | % `hard_refusal` on 10-q adversarial slice | ≥ 0.80 |
| Latency | p95 end-to-end on Q13 hardware | ≤ 20 s |
| User accept | Brother says "I'd use this in actual research" | Yes |

---

## Decision index

Each architectural decision in this document corresponds to a grilling question — listed here for traceability. Question numbers below refer to architecture grilling sessions; numbers in `decisions.md` are independent.

| Section | Question | Decision |
|---|---|---|
| 2 | A-Q1 — Process model | Single process, eager parallel in-memory load |
| 2 | A-Q7 — Concurrency | Fully serialized requests |
| 3.2 | A-Q2 — Stream protocol | Token streaming + verify-at-end (v1; incremental verify deferred) |
| 3.1 | A-Q5 — Parallelism | Parallel FAISS+BM25, batched NLI (AND-of-singles per cited chunk), parallel judge |
| 4 | A-Q3 — API surface | `/ask`, `/chunks/{id}`, `/papers/{pmid}`, `/health` |
| 5 | A-Q4 — Errors | Single error event, rerank-floor refusal trigger, fabricated-citation handling, monthly-cap clean-fail, SDK auto-retry |
| 6 | A-Q8 — Caching | No answer cache, embedding LRU only, stateless turns |
| 7 | A-Q6 — Startup | Parallel load, strict-on-models / soft-on-key, `/health` 503→200 |
| 8 | A-Q9 — Indexing | Staged checkpoints, SQLite resumability, retry-then-skip per paper |
| 9 | A-Q13 — Eval | Direct imports, Parquet results, retrieval-only default + `--full` (batch API)/`--mock-llm` |
| 10 | A-Q14 — Frontend | ES modules, single state object, append-stream + render-on-verify |
| 11 | A-Q12 — Config | `.env` + `config.yaml`, Pydantic Settings, singleton getter, Sonnet/Haiku model split, monthly cap, rerank floor |
| 12 | A-Q10 — Logging | JSON stdout + SQLite traces, summary default + debug flag, per-query cost split, section_type histogram |
| 13 | A-Q11 — Versioning | Bundle semver + manifest, hosted on HuggingFace public dataset, git SHA stamp, no auto-invalidation |
| 14 | A-Q15 — Testing | Mock everything, synthetic fixtures, strict test/eval separation |
| 15 | A-Q16 — Dev loop | Manual pytest, GitHub Actions for lint+unit, ruff |
| 18 | (decisions Q14) — Milestones | Vertical slice M1–M6, week-4 brother labeling, 5 `--full` runs total |
| 19 | (decisions Q15) — Acceptance | Five gates: Recall ≥ 0.65, Faithfulness ≥ 0.80, Refusal honesty ≥ 0.80, p95 ≤ 20 s, brother accept |
| 5, 9.4, 11, 12 | (decisions Q21) — Cost defense | Five-layer stack: app cap + console limits + per-query ceiling + max_tokens + 80% warning; `--full` confirm + batch-only + run log; `cost` CLI |
| 8 | (decisions Q22) — M1 ingest | COPD toy via E-utilities + httpx; NCBI key day 1; pubmed_parser; min-viable-record salvage; day-1 commit sequence + smoke test |
