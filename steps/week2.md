# Week 2 — M1 retrieval + mocked-verifier `/ask`

**Goal of week:** land commits 8–14 of M1. End-state: brother hits `POST /ask` against the 100-paper COPD toy corpus, sees a real streamed Sonnet answer with **mocked verifier returning all-green dots** (M1 acceptance per Q14). End-to-end skeleton done — Phase 2 first light.

**Budget:** ~15–20 hr (Q23b half-time pace). Buffer day at end.

**Anchors:** `.claude/research/decisions.md` Q5 (chunking), Q6 (MedCPT embedder), Q7 (BM25 tokenizer), Q8 (cross-encoder + RRF), Q9b (rerank floor + hard refusal), Q11a/b/c (FastAPI + plain JS frontend + FAISS), Q12 (prompt design), Q14 (M1 acceptance), Q16 (two-client LLM split), Q23d (FAISS=FlatIP), Q23e (pysbd splitter). `.claude/research/architecture.md` §2 (process model), §3 (request lifecycle), §4 (API surface), §7 (startup), §8.1 (staged indexing CLI), §10 (frontend), §14 (testing). `.claude/research/glossary.md` (`chunk_id` format, stage names, two clients).

---

## Day 0 — Fold week-1 drift into research docs

**Goal:** clean slate before retrieval code. Three drift items surfaced during week 1 deserve `decisions.md` lines so future-me doesn't re-derive. **Not a code commit — research-doc edit only.**

| Drift | Location | Edit |
|---|---|---|
| Q22b: `"open access"[filter]` returned 0 hits → corrected to `"pubmed pmc open access"[filter]` | decisions.md §Q22b | Drift fix already noted 2026-05-24 — confirm wording. |
| Q22b: NCBI elink stream-closes at ~100 ids in one URL → batch GETs to 50 PMIDs | decisions.md §Q22b OR §Q22e | Add one-line "elink batched at 50 ids/GET" note. Code lives at `c72d669`. |
| Q22c: `pubmed_parser==0.5.1` raises `KeyError('year')` on epub-only papers → monkey-patch `parse_date` to seed `year=None` | decisions.md §Q22c | Document the patch; flag for upstream PR or pinned-fork plan at M5. |
| Q23j: `microsoft/deberta-v3-large-mnli` 404 on HF → smoke ran on `cross-encoder/nli-deberta-v3-large` | decisions.md §Q23j | Already noted; confirm M2 verifier model re-lock is on the agenda for week 3. |

**Verify:** `git diff .claude/research/decisions.md` shows the four bullet edits. No code touched.
**Commit:** `docs(decisions): fold week-1 drift (elink batch, year patch, deberta swap)`.

---

## Day 1 — Commit 8: IMRaD chunker

**LOC:** ~180 (impl ~110, tests ~70). **Time:** ~3 hr.

`src/rag_med/indexing/chunk.py` per Q5:

- Input: `dict` from `parse.py` (sections list + abstract + tables/captions).
- Output: `list[Chunk]` with fields `chunk_id, pmid, section_type, ordinal, text, n_deberta_tokens, n_medcpt_tokens`.
- **`chunk_id` format (glossary):** `{pmid}_{section_type}_{ordinal:02d}` — e.g. `12345678_methods_03`.
- **`section_type` enum (glossary):** `abstract | introduction | methods | results | discussion | table | caption | other`. Map `pubmed_parser` `section_name` strings → enum via lowercase substring match; unknown → `other`.
- **Splitter:** `pysbd` (Q23e) for sentence boundaries inside a section. Then greedy-pack sentences until `n_deberta_tokens` ≥ 300; flush; continue. Soft ceiling 400 — if a single sentence overflows, emit alone and log warning.
- **Token counting:** load `microsoft/deberta-v3-large` tokenizer once at module import (canonical), `ncbi/MedCPT-Article-Encoder` tokenizer for sanity-check column. Both via `transformers.AutoTokenizer`; cached in `HF_HOME`.
- **Special chunks (Q5):**
  - `abstract` → one chunk regardless of length (split if > 400 tokens — rare).
  - `table` → own chunk per table, includes caption inline.
  - `caption` (figure caption without table) → own chunk.
  - References list → stripped at parse time; chunker shouldn't see it.
- **`shared/db.py`** gets `chunks` table DDL: `chunk_id PK, pmid FK, section_type, ordinal, text, n_deberta_tokens, n_medcpt_tokens`. Indices on `chunks(pmid)` + `chunks(section_type)` (Q23h). Migration appended to existing DDL helper.

`tests/test_chunk.py` (mock-only, fixture-driven):
- Empty section → no chunks.
- Single 200-token section → one chunk.
- 1500-token methods section → ~4 chunks, all 300–400 tokens.
- `chunk_id` format regex.
- Table = own chunk, caption included.
- Sentence not split mid-sentence (pysbd doesn't break `2.5 mg`, `Fig. 1`, `et al.`, `p < 0.05`).
- `n_deberta_tokens` and `n_medcpt_tokens` both populated.

**Verify:** `pytest tests/test_chunk.py` green; `pytest tests/test_db_schema.py` still green (new table additive).
**Commit:** `feat(chunk): IMRaD chunker, 350 deberta-token target, section-aware`.

---

## Day 2 — Commit 9: chunk pipeline subcommand + MedCPT-Article embedder

**LOC:** ~150 (pipeline wiring ~50, embedder ~60, tests ~40). **Time:** ~3 hr.

Two thin pieces, one commit. Sets up the indexing CLI shape from architecture.md §8.1.

### Part A — `python -m rag_med.indexing.pipeline chunk`

Wires `parse → chunker → INSERT OR IGNORE into chunks`. Reads `paper_xml` rows that don't yet have chunks; processes each; writes. Idempotent.

CLI: `python -m rag_med.indexing.pipeline chunk` (no flags — runs to completion on whatever papers exist).

### Part B — `src/rag_med/indexing/embed.py` — MedCPT-Article-Encoder

- Loads `ncbi/MedCPT-Article-Encoder` on **MPS** (Q23a dev HW). CPU fallback if MPS unavailable, log loud.
- `embed_chunks(texts: list[str], batch_size: int = 32) -> np.ndarray` returns `float32 (N, 768)`.
- Truncation: 512 tokens (model limit); chunker keeps us under, defensive `truncation=True`.
- No FAISS yet — returns array. FAISS comes day 3.

`tests/test_embed.py` (mock the model load + forward):
- Shape `(N, 768)`.
- `dtype == float32`.
- Empty input → empty array, no model call.

**Smoke (manual):** load real model, embed 5 chunks, sanity-check norm + dim. Not in pytest.

**Verify:** `pytest -q` green; `python -m rag_med.indexing.pipeline chunk` populates `chunks` table for the 100-paper corpus (expected ~1500–4000 rows).
**Commit:** `feat(embed): chunk pipeline subcommand + MedCPT-Article embedder`.

---

## Day 3 — Commit 10: FAISS IndexFlatIP build

**LOC:** ~100 (impl ~50, tests ~50). **Time:** ~2 hr.

`python -m rag_med.indexing.pipeline embed` per architecture.md §8.1. Despite the name overlap with day-2 module, this is the **end-to-end stage** that reads chunks, embeds, builds FAISS.

- Reads all `chunks.text` rows ordered by `chunk_id`.
- Calls `embed.embed_chunks` in batches.
- L2-normalize vectors → inner-product on normalized vectors = cosine (Q23d `IndexFlatIP`).
- Builds `faiss.IndexFlatIP(768)`, `index.add(vecs)`.
- Writes to `data/faiss.index` via `faiss.write_index`.
- Sidecar: `data/faiss.chunk_ids.json` — ordered list mapping FAISS row idx → `chunk_id`. (FAISS itself doesn't store string IDs cheaply at this scale; sidecar JSON is ~50 KB for the toy corpus.)

`tests/test_faiss_build.py` (mock embedder):
- Build with 10 fake chunks → index size = 10, dim = 768.
- Round-trip: write → read → identical search results.
- Sidecar JSON length matches index size.
- Query top-1 of a vector against itself → that vector's chunk_id, score ≈ 1.0.

**Verify:** `pytest -q` green; full build on toy corpus writes `data/faiss.index` (~10 MB for ~2k chunks).
**Commit:** `feat(faiss): IndexFlatIP build over chunk embeddings`.

---

## Day 4 — Commit 11: biomedical tokenizer + BM25 build

**LOC:** ~140 (tokenizer ~50, build ~40, tests ~50). **Time:** ~2.5 hr.

`src/rag_med/shared/tokenize.py` per Q7 (lives in `shared/` so eval can reuse):

- Regex rules: keep hyphens / `+` / `/` inside tokens (`IL-4`, `CD8+`, `FEV1/FVC`); keep digit-letter combos (`FEV1`, `25mg`); split on whitespace + non-medical punctuation; lowercase; drop English stopwords (small hand-list, ~30 words).
- One function: `bm25_tokenize(text: str) -> list[str]`.
- **Glossary rule:** stage name is `lexical_search`; this is the **tokenizer for BM25**, not "keyword search".

`src/rag_med/indexing/bm25_build.py`:
- `python -m rag_med.indexing.pipeline bm25` subcommand.
- Reads `chunks.text` ordered by `chunk_id`.
- Tokenizes each, builds `rank_bm25.BM25Okapi(corpus_tokens)`.
- Serialize the built `BM25Okapi` object to `data/bm25.pkl` (project standard per architecture.md §11.1 paths) + `data/bm25.chunk_ids.json` sidecar (mirror FAISS pattern). **Trusted-input only — this file is built by our own pipeline and loaded by our own server; never accept a `bm25.pkl` from anywhere else.**

`tests/test_tokenize.py`:
- `bm25_tokenize("IL-4 induces CD8+ T-cell proliferation in FEV1/FVC patients")` → keeps `il-4`, `cd8+`, `fev1/fvc`.
- Stopwords `"the"`, `"is"`, `"and"` dropped.
- `25mg` and `FEV1` kept intact.
- Empty / whitespace input → `[]`.

`tests/test_bm25_build.py` (mock the chunks):
- Build over 10 fake chunks → searchable.
- Round-trip serialization.
- Known-query top-1 sanity (chunk with exact match wins).

**Verify:** `pytest -q` green; `data/bm25.pkl` written (~1–5 MB toy scale).
**Commit:** `feat(bm25): biomedical tokenizer + rank_bm25 inverted index`.

---

## Day 5 — Commit 12: retrieval skeleton (dense + lexical + RRF + rerank)

**LOC:** ~220 (impl ~140, tests ~80). **Time:** ~4 hr — **densest day**.

`src/rag_med/serving/retrieve.py` — pipeline stages 1–5 of architecture.md §3.1. **Mock everything in tests.**

Functions (glossary-compliant names — NO `semantic_search`, NO `keyword_search`):

```python
def embed(question: str) -> np.ndarray                              # MedCPT-Query
def dense_search(query_vector, top_k=50) -> list[(chunk_id, score)]
def lexical_search(bm25_tokens, top_k=50) -> list[(chunk_id, score)]
def fuse(dense_hits, lexical_hits, rrf_k=60) -> list[(chunk_id, rrf_score)]
def rerank(question, fused_candidates, top_k=10) -> list[(chunk_id, rerank_score)]
def retrieve(question: str, top_k: int = 10) -> RetrievalResult
```

- **MedCPT-Query-Encoder** loaded once (sibling to MedCPT-Article-Encoder; ~400 MB). L2-normalize query vector to match index (cosine via IP).
- **FAISS load:** at module import, read `data/faiss.index` + sidecar JSON into memory (architecture.md §7 eager-load model). For week-2 tests this is monkey-patched.
- **BM25 load:** deserialize `data/bm25.pkl` at module import. Same monkey-patch story in tests.
- **RRF (Q8):** `score(chunk) = Σ 1/(rrf_k + rank_in_list)` across the two lists. `rrf_k = 60` hardcoded.
- **Cross-encoder:** `ncbi/MedCPT-Cross-Encoder` over the top-50 fused candidates. One batch forward pass (architecture.md §3.1 parallelism note). Returns `top_k=10`.
- **`rerank_floor` (Q9b) — TWO effects from one knob, both wired now:**
  - Top-1 score < `rerank_floor` → return `RetrievalResult(final_chunks=[], hard_refusal=True)`. `/ask` will short-circuit to the fixed-string answer.
  - Per-chunk: drop chunks with `rerank_score < rerank_floor` from `final_chunks` before returning.
  - Value = `0.0` placeholder per Q9b (calibrated empirically at M4).
- **Parallelism (architecture.md §3.1):** `dense_search` + `lexical_search` run via `asyncio.gather(to_thread(...), to_thread(...))`. Already async-shaped now so day-7 SSE doesn't need a refactor.
- **Embedding cache:** `functools.lru_cache(maxsize=1000)` on `embed(question)` (architecture.md §6). Five-line concern.

`RetrievalResult` dataclass:
```python
@dataclass
class RetrievalResult:
    final_chunks: list[Chunk]         # full Chunk objects, ready for prompt
    rerank_scores: list[float]        # parallel to final_chunks
    hard_refusal: bool                # True if top-1 < rerank_floor
    section_type_histogram: dict      # Q7 instrumentation, logged later
```

`tests/test_retrieve.py` (mock everything: fake embedder vectors, fake FAISS, fake BM25, fake cross-encoder):
- `dense_search` returns top-K in score order.
- `lexical_search` returns top-K in score order.
- **RRF math:** chunk in rank 1 of both lists scores `2/(60+1)`; chunk only in rank 5 of dense scores `1/(60+5)`. Hand-computed expected vs actual.
- `fuse` dedupes (chunk in both lists appears once with summed RRF score).
- `rerank` returns top-K by rerank score.
- `rerank_floor` enforcement: with floor=0.5, mock cross-encoder returns [0.9, 0.6, 0.4, 0.3] → `final_chunks` has 2 items.
- `hard_refusal=True` when mock cross-encoder top-1 < floor.
- `section_type_histogram` populated correctly from final chunks.
- Embedding LRU cache: second call with same question doesn't re-invoke embedder.

**Verify:** `pytest tests/test_retrieve.py` green. Manual smoke later in day 7 once the real models load.
**Commit:** `feat(retrieve): dense+lexical+RRF+cross-encoder retrieval skeleton`.

---

## Day 6 — Commit 13: generator (Sonnet streaming) + prompt + mocked verifier

**LOC:** ~180 (impl ~110, tests ~70). **Time:** ~3 hr.

Two clients per hard rule + Q16: **`generator_client` only this commit; `judge_client` deferred to M2.**

### Part A — `src/rag_med/serving/prompt.py`

Builds the chunk-numbered prompt per Q12d:

```
[1] (Calverley et al. 2007, NEJM, methods section)
"Patients were randomized to..."

[2] (...)
...
```

- Author truncation: first author + `et al.` always (Q12d).
- Section tag from `section_type` enum.
- System prompt = the verbatim 6-rule block from decisions.md §Q12c (scope lock, citation discipline, verbatim numbers, no medical advice, soft-refusal gap statement, hard-refusal exact string).
- Pure function `build_prompt(question, final_chunks) -> (system: str, user: str)`. No I/O.

### Part B — `src/rag_med/serving/generate.py`

- `generator_client` = Anthropic SDK client. **Distinct module-level singleton** from `judge_client` (which lands at M2). Hard rule.
- Model = `claude-sonnet-4-6` (config-driven from `settings.generator_model`, default per decisions.md §Q16).
- `max_tokens = settings.max_tokens` (1024, Q16 layer 4).
- `async def generate(system, user) -> AsyncIterator[str]` yields token text deltas. Anthropic streaming → at the SDK boundary, rename `delta` to `token` (glossary rule).
- Cost tracking stub: record `generator_cost_usd` into a returned `GenerationStats` alongside the stream-complete signal. Wire to `query_traces` writes at M2 (this week: structured-log only, no DB write).

### Part C — Mocked verifier (M1 acceptance per Q14)

`src/rag_med/serving/verify.py`:
- **Mocked for M1.** Splits answer with `pysbd`, parses `[n]` citations, **returns every sentence as `status='supported'`** regardless of content. Real NLI + Haiku judge land in M2.
- TODO comment with explicit `# M2: replace with NLI + Haiku judge per decisions.md §Q9` so future-me doesn't ship the mock.
- `judge_client` NOT created this week. No Haiku 4.5 import. Keeps the M1/M2 boundary clean.

`tests/test_prompt.py`:
- All chunks numbered `[1]`, `[2]`, ... in order.
- Author truncation: 5-author paper renders as `Smith et al.`
- System prompt contains the 6 rules verbatim.
- Hard-refusal pre-empts the prompt builder (`final_chunks=[]` → caller should never call this; assert defensive raise).

`tests/test_generate.py` (mock Anthropic client):
- Stream yields renamed `token` strings, not `delta`.
- `max_tokens` from config flowed into the API call.
- `generator_client` and `judge_client` are distinct symbols (introspection: `id(generator_client) != id(judge_client)` once judge_client exists — placeholder assert for M2).
- Mock 429 → SDK auto-retries (rely on SDK; assert SDK called with `max_retries`).

`tests/test_verify_mock.py`:
- 3-sentence answer with `[1][2]` citations → all 3 sentences `status='supported'`.
- Failure-kind field is `None` on supported sentences (glossary).

**Verify:** `pytest -q` green.
**Commit:** `feat(generate): Sonnet streaming generator + prompt builder + M1 mock verifier`.

---

## Day 7 — Commit 14: `/ask` SSE + minimal frontend + M1 acceptance smoke

**LOC:** ~220 (FastAPI ~100, frontend ~80, smoke test ~40). **Time:** ~4 hr.

The integration moment. End-to-end thin path: question → retrieve → generate → mock-verify → SSE → browser.

### Part A — `src/rag_med/serving/api.py`

FastAPI app, routes per architecture.md §4. **Only `/ask` and `/health` this week**; `/chunks/{id}` + `/papers/{pmid}` deferred to M2 polish (frontend doesn't need clickable citations for M1 acceptance).

- `POST /ask` — SSE response. Event schema verbatim from architecture.md §3.2:
  ```
  event: retrieved   data: {chunks: [{chunk_id, paper_pmid, section_type, text_preview}, ...]}
  event: token       data: {text: "..."}
  ...
  event: verified    data: {sentences: [{idx, text, citations, status: "supported", ...}]}
  event: done        data: {}
  ```
- **Hard-refusal short-circuit (Q9b):** if `RetrievalResult.hard_refusal=True`, emit `event: answer  data: {text: "The retrieved evidence does not address this question."}`, skip Claude, skip verifier, emit `done`. Logged with `refusal: "hard"` to structlog.
- Single error event on any fatal stage failure per architecture.md §5.1.
- **Concurrency:** fully serialized per architecture.md §2 — `asyncio.Lock` around the pipeline. M1 single-user, no contention to worry about.
- `GET /health` — returns `{status, paper_count, chunk_count, models_loaded, git_sha}` per architecture.md §4. Used by frontend gating + future docker healthcheck.

### Part B — Static frontend (architecture.md §10)

`static/index.html` + `static/app.js` — barebones for M1:

- Single textarea + Ask button.
- On submit: open `EventSource` to `/ask` (or `fetch` with manual SSE parsing — `EventSource` doesn't support POST; use `fetch` + ReadableStream, ~30 lines).
- `token` events: `textNode.appendData(chunk)` — no innerHTML clobber (architecture.md §10.3).
- `verified` event: re-render with green dots (all green for M1 since verifier is mocked).
- Error event: red banner.
- **No citation side panel this week** — `[n]` markers render inline as plain text. Side panel = M2 polish.

### Part C — `tests/test_api_e2e.py`

Real FastAPI `TestClient`, **mocked retrieve + generate + verify** (no model loads in CI):
- Happy path: question → SSE stream emits `retrieved → token×N → verified (all supported) → done` in order.
- Hard refusal path: mock `retrieve` returns `hard_refusal=True` → SSE emits `answer` event with exact fixed string, no `token` events.
- Empty question → 400.
- Generator raises → single `error` event, then `done`.
- `/health` returns `models_loaded: True` after mocked-init.

### Part D — M1 acceptance smoke (manual, not pytest)

Boot the real thing locally:
```
uvicorn rag_med.serving.api:app --reload
```

Open `http://localhost:8000/static/index.html`. Ask a real COPD question against the 100-paper corpus. Expected:
- Retrieved chunks land (~0.5–1 s with cold cross-encoder on MPS).
- Sonnet streams an answer with `[n]` citations.
- Green dots paint on every sentence after stream-complete.
- p50 latency on M5 Pro: rough log line. No formal Q15 gate measurement yet — that's M4.

Screenshot for `PROGRESS.md` if it works.

**Commit:** `feat(api): /ask SSE + mocked-verifier all-green + minimal HTML frontend`.

---

## Week 2 done — what's true at end

- ✅ All week-1 drift folded into `decisions.md`.
- ✅ Chunker writes IMRaD-tagged 350-DeBERTa-token chunks for the 100 COPD papers (~1500–4000 rows).
- ✅ FAISS `IndexFlatIP` + BM25 inverted index both built locally; round-trip tested.
- ✅ Retrieval skeleton (dense + lexical + RRF + cross-encoder rerank + rerank_floor) green under mocks.
- ✅ Sonnet generator streams; `generator_client` distinct from (placeholder) `judge_client`.
- ✅ `/ask` SSE end-to-end with mocked all-green verifier.
- ✅ **M1 acceptance met** — brother could hit the local server today, ask a question, see the streamed answer with green dots (Q14).
- ✅ ~14 commits total on `main` (or merged from a week-2 branch); CI green.

## Anti-patterns to refuse

- **Vertical-slicing M2 verifier into week 2.** Mock stays mocked. Real NLI + Haiku judge = M2. The whole point of the milestone shape is that M1 is M1.
- **Skipping the day-0 drift fold.** Future-me will re-derive elink batching and burn a debug afternoon.
- **Naming the BM25 stage `keyword_search` or the FAISS stage `semantic_search`** — glossary-banned. Code review yourself before commit.
- **Merging `generator_client` and `judge_client` into one `claude_client`.** Hard rule. Two clients, two models, two cost columns at M2.
- **Loading real models inside `pytest`.** All tests mock embedders, FAISS, BM25, cross-encoder, Anthropic. Smoke happens via `uvicorn` + browser, not pytest.
- **Doing chunker + embedder + retrieval refactors in one big commit** because "they're related." Seven commits, each green, each pushable.
- **Building the citation side panel "while we're in there."** M2 polish. M1 acceptance doesn't need it.

## Risk flags carried in

- **MedCPT-Cross-Encoder on MPS quirks (Q23j sibling).** Day-5 smoke at week 1 covered DeBERTa-v3-large; cross-encoder is a separate model, separate ops. Run a one-shot MPS forward at day 5 before wiring it into `retrieve.py`. If anything falls back loudly, pin to CPU and log honestly.
- **`rerank_floor=0.0` is placeholder.** Hard-refusal path basically can't trigger this week — every query will have a top-1 score > 0. That's fine; M4 calibration gives it the real value. Tests still exercise the floor branch via mock scores.
- **Sonnet 4.6 streaming SDK quirks.** First contact with real Anthropic streaming on this project. Budget ~1 hr extra on day 6 for "the docs lied about the field name" surprises. Rename to `token` at the boundary so day 7 doesn't carry the mess.

## Next week preview

**M2 starts.** Real verifier wired up: NLI + Haiku judge replacing the M1 mock. Locks the M2 verifier model (cross-encoder/nli-deberta-v3-large is the day-5 fallback; needs a real M2 grill before becoming canonical). `judge_client` introduced as a sibling to `generator_client`. Per-cited-chunk AND-of-singles (Q1) implemented; `failure_kind` taxonomy wired through to SSE `verified` events; brother sees the first real green / yellow / red dots. First `--full` eval baseline ($28 batch) at the M2 close.
