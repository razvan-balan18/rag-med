# Pneumology RAG — Ubiquitous Language

Shared vocabulary for the project. Every term has ONE meaning. Code, logs, UI, docs, eval, writeup all use these words consistently. Banned synonyms listed where collisions are likely.

Companion to `decisions.md` (what we picked) and `architecture.md` (how it runs).

Last updated: 2026-05-11 (Q23 closures: gold-set authoring shift to dev; `pysbd` splitter named)

---

## Core entities

### `paper`
Source document from PubMed or PMC. The unit of provenance. One row in `papers` SQLite table.
- **Canonical PK:** `PMID` (always present, universal).
- **Secondary IDs:** `PMCID` (only ~30% of corpus, full-text papers), `DOI` (display + linkout).
- **Banned synonyms in this sense:** `article`, `document`.

### `chunk`
Retrievable text unit derived from a paper. The atom of the search index. **Target size: 350 ± 50 DeBERTa-v3 tokens** (DeBERTa is the strictest downstream tokenizer — verifier 512-token cap drives the budget). One chunk = one row in `chunks` SQLite table + one vector in FAISS + one entry in BM25, all keyed by `chunk_id`.
- **`chunk_id` format:** `{pmid}_{section_type}_{ordinal:02d}` — e.g. `12345678_methods_03`.
- Citations in answers point to chunks, not papers.
- One paper produces ~15–40 chunks.
- **Schema fields on `chunks` table** include `n_deberta_tokens INTEGER` (canonical size) and `n_medcpt_tokens INTEGER` (sanity-check at index time). Bare `n_tokens` is **banned** — always specify which tokenizer.

### `section_type`
Enum tagging which part of the paper a chunk came from.
```
abstract | introduction | methods | results | discussion | table | caption | other
```
Abstract is a chunk like any other (`section_type='abstract'`). Tables and figure captions are their own chunks. References list is stripped during indexing.

---

## Request-side vocabulary

### `question`
The natural-language string the user types. The user-facing input. One field on `POST /ask`.
- **Banned synonym in this sense:** `query` (reserved for derived pipeline artifacts).

### `query_vector`
The 768-dim MedCPT-Query-Encoder embedding of the question. Input to FAISS.

### `bm25_tokens`
The biomedical-tokenized question (list of strings). Input to BM25.

### `answer`
The full text Claude produces. The whole response.
- **Banned synonyms:** `response`, `completion`, `output`.

### `sentence` / `claim`
Same 1:1 thing, two registers:
- **`sentence`** = code-side. The mechanical split unit. Field name, variable name, schema field.
- **`claim`** = UI + writeup. The semantic role each sentence plays. Used in copy like "8/9 claims verified".

#### Sentence schema (code-side, locked)

```python
class Sentence:
    idx: int
    text: str
    citations: list[int]              # raw [n] numbers from regex parse
    cited_chunk_ids: list[str]        # resolved via final_chunks map; missing → ''
    status: Literal['supported', 'unclear', 'unsupported', 'unknown']
    failure_kind: str | None          # forensic tag, see below
    nli_entailments: list[float]      # one per cited_chunk_id (Q1: AND-of-singles)
    judge_confidences: list[float]    # one per cited_chunk_id, None where NLI was confident
```

**`failure_reason` enum** (indexing-side, on `failed_papers` rows from Q22d salvage rule):
| Value | Meaning |
|---|---|
| `missing_title` | paper has no parseable title — UI couldn't display |
| `no_content` | both abstract and body empty/missing |
| `xml_parse_error` | `pubmed_parser` (or M5 hand-rolled `lxml`) raised on this paper's XML |
| `encoding_error` | UTF-8 cleanup failed |

Distinct from `failure_kind` below — `failure_reason` is for whole papers dropped at indexing time; `failure_kind` is for individual answer sentences that didn't verify at query time.

**`failure_kind` enum (when `status != 'supported'`):**
| Value | Meaning |
|---|---|
| `fabricated_citation` | `[n]` doesn't exist in `final_chunks` (Claude invented the number) |
| `no_citation` | sentence has no `[n]` at all (Q1 sub: strict — counts as unsupported) |
| `nli_contradiction` | NLI confidently contradicted on at least one cited chunk |
| `nli_neutral` | NLI neutral after escalation, judge inconclusive |
| `judge_inconclusive` | judge LLM returned `unclear` on borderline pair |
| `verifier_crash` | exception in NLI / judge stage; `status='unknown'` |
| `truncation_assert_failed` | (chunk + sentence) > 512 DeBERTa tokens — should never fire if Q5 chunker holds |
| `null` | `status='supported'`; no failure |

### `token`
Smallest unit of the streaming generation. Used in:
- SSE event name: `event: token`
- Loop variable when iterating Anthropic stream: `for token in stream: token.text` (we rename Anthropic's `delta` to `token` at the boundary for consistency).

### `citation`
The `[n]` marker rendered in the answer text. The visible link from a sentence to a chunk.
- **`cited_chunk_ids: list[str]`** = the resolved chunks a sentence points to (after mapping `[n]` → chunk_id via the per-query `final_chunks` map).
- **Two fields on each Sentence:** `citations: list[int]` (raw `[n]` numbers from regex parse) + `cited_chunk_ids: list[str]` (resolved).
- **Banned synonyms:** `reference` (reserved for the bibliography list of a paper, which we strip), `source` (reserved for `source_type`: full-text vs abstract).

### `status` (verdict)
The verifier's per-sentence label. Four values:
| Value | Meaning | UI color |
|---|---|---|
| `supported` | NLI or judge confirms cited chunk backs the sentence | green |
| `unclear` | borderline / can't tell | yellow |
| `unsupported` | cited chunk does NOT back the sentence | red |
| `unknown` | verifier crashed | gray |

UI maps semantic label → color. Logs/traces/eval store the semantic label.

---

## Retrieval pipeline vocabulary

### Stage names

| Stage | Term | What it does |
|---|---|---|
| 1 | `embed` | question → query_vector |
| 2 | `dense_search` | FAISS nearest-neighbor over query_vector |
| 3 | `lexical_search` | BM25 over bm25_tokens |
| 4 | `fuse` | RRF combination of dense + lexical results |
| 5 | `rerank` | MedCPT-Cross-Encoder over top candidates |

- **Banned synonyms:** `semantic_search`, `vector_search`, `keyword_search`, `bm25_search`, `rrf` (when used as a stage name — `rrf_score` is fine as a field).

### Result-set names

| Stage produces | Name | Size |
|---|---|---|
| post-FAISS | `dense_hits: list[(chunk_id, score)]` | top-50 |
| post-BM25 | `lexical_hits: list[(chunk_id, score)]` | top-50 |
| post-RRF | `fused_candidates: list[(chunk_id, rrf_score)]` | top-50 deduped |
| post-rerank | `reranked_chunks: list[(chunk_id, rerank_score)]` | top-10 |
| sent to Claude | `final_chunks: list[chunk_id]` | top-K (default 10) |

---

## Verifier sub-stage vocabulary

| Sub-stage | What it does |
|---|---|
| `split` | `pysbd` sentence-splitter on the answer (Q23e) — handles medical abbreviations, decimals, `et al.`, `Fig.` refs |
| `parse_citations` | extract `[n]` numbers per sentence |
| `resolve_citations` | map `[n]` → chunk_id via `final_chunks` |
| `nli_check` | batched NLI forward pass over (sentence, cited_chunk) pairs |
| `triage` | decide per pair: confident NLI verdict OR escalate to judge (borderline 0.3–0.9) |
| `judge` | Claude judge call on borderline pairs |
| `finalize` | collapse NLI + judge into single per-sentence `status` |

`triage` is the medical-metaphor stage name — quick screen, escalate hard cases.

---

## LLM role vocabulary

Claude plays two distinct roles, **using two distinct models for cost reasons**. Two thin wrapper clients. Never one shared `claude_client`. Every Anthropic call tagged with `role: "generator" | "judge"` for cost attribution.

| Role | Where | Module | Client var | Model | Streaming |
|---|---|---|---|---|---|
| `generator` | Phase 2 step 7 — writes the answer | `serving/generate.py` | `generator_client` | **Sonnet 4.6** | yes (token SSE) |
| `judge` | Verifier sub-stage 6 — judges borderline sentences | `serving/verify.py` | `judge_client` | **Haiku 4.5** | no (one-shot) |

Cost attribution recorded per call into `query_traces.cost_usd` with role split: `generator_cost_usd`, `judge_cost_usd`. See `decisions.md §Q16 — Cost discipline`.

---

## Cost-defense vocabulary (Q21)

Five-layer cost-defense stack. Each layer has ONE name. Use these terms verbatim in code, logs, errors, docs.

| Term | What it is | Where it lives |
|---|---|---|
| `monthly_cap_usd` | App-level monthly spend cap. Sums `query_traces.cost_usd` MTD. | `config.yaml::cost.monthly_cap_usd` |
| `console_limit` | Anthropic dashboard hard limit per API key. Backstop layer. | Anthropic console (NOT `config.yaml`) |
| `per_query_ceiling_usd` | Single-query abort threshold. Catches runaway loops. | `config.yaml::cost.per_query_ceiling_usd` |
| `max_tokens` | Generator single-completion length cap. | `config.yaml::llm.max_tokens` |
| `warn_threshold_pct` | MTD-spend % at which 80% warning fires. | `config.yaml::cost.warn_threshold_pct` |
| `cost_warning` | Boolean field on `/health` set when MTD > `warn_threshold_pct × monthly_cap_usd`. | `/health` JSON |
| `cost CLI` | `python -m rag_med cost` — read-only SQL over `query_traces`. | command, not field |

**Threat-model vocabulary** (used in design discussions, ADRs, postmortems):
- `dev burn` — bug during M1–M6 development that hammers the dev API key.
- `brother burn` — bug in v1.x update that causes runaway in production on brother's key.
- `single-query blowup` — one `/ask` overshoots before the monthly cap notices.

**`--full` eval safety vocabulary:**
- `confirm prompt` — interactive `YES`-required confirmation before `--full` runs.
- `runs.jsonl` — append-only `eval/runs.jsonl` log of `--full` invocations.

**Banned synonyms:**
| Don't say | Say instead |
|---|---|
| `query budget`, `single-call cap` | `per_query_ceiling_usd` |
| `Sonnet output limit`, `completion length cap` | `max_tokens` |
| `dashboard cap`, `key cap` | `console_limit` |
| `spending alert` | `cost_warning` |

---

## Refusal vocabulary

Two refusal types, both tracked in logs + eval as failure / graceful-degradation modes.

### `hard_refusal`
Zero chunks retrieved (post-rerank top-10 empty OR all below threshold). Skip Claude entirely. Emit fixed string:
> "The retrieved evidence does not address this question."

Logged with `refusal: "hard"`. Eval metric: `% hard_refusal` = recall failure rate.

### `soft_refusal`
Chunks exist but partially cover the question. Claude generates an answer that explicitly states what is NOT covered. Behavior driven by system prompt rule 5 (decisions.md §Q12c). Detected post-hoc via gap-statement language.

Logged with `refusal: "soft"`. Eval metric: `% soft_refusal` = graceful degradation rate.

---

## Eval vocabulary

### `gold_set`
The curated 290-question collection. Path: `data/gold_set/`.
- **Composition (revised Q23c, 2026-05-11):** **50 dev-authored** (25 from clinical practice guidelines — GOLD, GINA, ATS/ERS — plus 25 from Cochrane respiratory reviews) + 150 synthetic + 80 BioASQ + **10 adversarial** (out-of-scope questions for the `% hard_refusal` honesty gate). Brother still does the **paper-level labeling pass** at M4 (~25 min) — preserves real-clinician relevance ground truth for Recall@10. See `decisions.md §Q23c`.

### `gold_item`
One labeled question + its labels. Schema:
```python
class GoldItem:
    question_id: str            # stable, e.g. "g042"
    question: str
    tags: dict                  # {section_focus, topic, difficulty}
    relevance: dict[str, str]   # PMID → "relevant" | "partial" | "not_relevant"
    author: str                 # "brother" | "synthetic" | "bioasq" | "adversarial"
```

**Paper-level relevance (locked Q3).** Brother labels at PMID granularity, not chunk granularity. Survives chunker/embedder changes. Recall@K computed by joining retrieved chunk_ids → papers → label lookup. Loses fine-grained "right paper, wrong section" signal — recoverable post-hoc by tagging retrieved chunks with `section_type` and slicing eval metrics.

### `run`
One full eval execution. Identified by `(git_sha, timestamp)`. Output: `results/run_<git_sha>_<timestamp>.parquet`.

### `result_row`
One row in the Parquet output. One row per question per run. Columns: `(run_id, question_id, retrieved_chunk_ids, status_per_sentence, latency_ms, tags...)`.

### Relevance labels
Brother's labeling categories, applied **per PMID**:
- `relevant` — paper directly answers the question
- `partial` — paper relates but doesn't fully answer
- `not_relevant` — paper irrelevant

### Retrieval metric modes
- `strict` — only `relevant` counts as a hit. **Default for headline numbers.**
- `lenient` — `relevant` + `partial` count. Secondary column.

---

## Phase + package vocabulary

Three phases = three top-level packages under `src/rag_med/`.

| Phase | Package | Purpose |
|---|---|---|
| 1 | `indexing` | Build SQLite + FAISS + BM25 from PubMed/PMC |
| 2 | `serving` | FastAPI runtime that answers questions |
| 3 | `eval` | Measure retrieval + faithfulness against gold set |

### `ingest` (sub-stage of `indexing`)
The fetch + parse half of indexing. Module path: `indexing/ingest/`. Distinguishes "data in" from "indexes out".

```
indexing/
├── ingest/          # fetch + parse
│   ├── pubmed.py
│   ├── pmc.py
│   └── parse.py
├── chunk.py
├── embed.py
├── bm25_build.py
└── pipeline.py      # orchestrator
```

---

### `M1 toy corpus` (locked Q22)

The 100-paper COPD-only sub-corpus used to bring up the end-to-end skeleton in week 1.
- **Topic = COPD.** Not "pneumology in general."
- **Date cutoff = 2020+.** Tighter than the full-corpus 2015+ cutoff.
- **Full-text only.** Enforced via `"pubmed pmc open access"[filter]` in the esearch query (every result has a PMCID). See `decisions.md §Q22b` drift fix — the originally locked `"open access"[filter]` returns 0 hits.
- Distinct from the M5 **full corpus** (~150k papers, MeSH Respiratory Tract Diseases tree + journal whitelist + 2015+).
- **Banned synonyms:** `toy dataset`, `mini corpus`, `dev corpus`, `seed corpus`. Always `M1 toy corpus`.

### `bundle` (locked Q9)

The deployable artifact brother downloads. SQLite + FAISS + BM25 in one tarball.
- Term **`bundle`** everywhere. Field `bundle_version`. File `bundle.tar.gz`. Banned alternates: `artifact`, `distribution`, `index_bundle`.
- **Hosting:** HuggingFace Datasets, public repo (corpus is public-domain PubMed/PMC OA). Single `bundle.tar.gz` + `manifest.json`.
- **Update flow:** dev runs indexing pipeline → `huggingface-cli upload`. Brother's app boots, fetches remote `manifest.json`, compares `bundle_version`. Mismatch + remote newer → `/health.update_available=true` → frontend banner. **No auto-update** — brother runs `./scripts/download_index.sh` when convenient.
- **No diff bundles v1.** Full ~10 GB re-download per refresh. Acceptable at the project's update cadence.

## Tentative (not yet locked — for future grilling)

These have recommended defaults but were not formally locked in this session. Confirm before relying on them in code.

### `log` vs `trace` vs `event`
Three persistence mechanisms, three audiences. Each word means ONE thing — never overlap.
- `log` = stdout JSON line (ops, real-time tail)
- `trace` = `query_traces` SQLite row (forensics, persistent, one per query)
- `event` = SSE message to frontend (UX, ephemeral)

### Score field naming
Tentative convention:
- Ranking stage scores get `_score` suffix: `dense_score`, `lexical_score`, `rrf_score`, `rerank_score`.
- Raw NLI probabilities unsuffixed: `nli_entailment`, `nli_neutral`, `nli_contradiction`.
- Judge confidence: `judge_confidence`.

---

## Banned-word index

Quick lookup for words NOT to use (and what to use instead):

| Don't say | Say instead | Why |
|---|---|---|
| `article` | `paper` | collides with HTML `<article>` |
| `document` | `paper` (entity) or `chunk` (retrievable unit) | too generic |
| `query` (for user input) | `question` | reserved for `query_vector` / `bm25_tokens` |
| `response` / `completion` / `output` | `answer` | overloaded |
| `delta` (in our code) | `token` | renamed at Anthropic SDK boundary |
| `reference` (for `[n]`) | `citation` | reserved for paper bibliography |
| `source` (for `[n]`) | `citation` | reserved for `source_type` |
| `semantic_search` / `vector_search` | `dense_search` | overclaims |
| `keyword_search` / `bm25_search` (as stage) | `lexical_search` | symmetric with `dense_search` |
| `claude_client` (shared) | `generator_client` or `judge_client` | distinct roles, distinct models (Sonnet vs Haiku) |
| `n_tokens` (bare) | `n_deberta_tokens` or `n_medcpt_tokens` | three tokenizers in pipeline; bare term ambiguous |
| `chunk-level relevance` (in eval) | `paper-level relevance` (PMID) | locked Q3; eval relevance is PMID-keyed |

---

## How to use this glossary

- **Code:** variable names, class names, schema fields, log keys MUST follow these terms.
- **UI:** user-facing strings use the UI-side term where one is specified (`claim` not `sentence`, color names not status names).
- **Docs / writeup / commits:** use these terms verbatim. If a new term emerges, add it here first.
- **PRs:** if a PR introduces a new entity or stage, update this file in the same PR.
