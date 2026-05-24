# Pneumology RAG — Design Decisions Log

Living document. Captures locked decisions from grilling sessions. Update as new decisions get made.

Last updated: 2026-05-11 (Q23 cluster added from final grill: M3/M5 reorder, dev-authored gold set, FAISS=FlatIP, pysbd splitter, auto-download weights, dev HW, pre-commit-1 chore list)

---

## Project identity

**One-liner:** A search engine for pneumology research papers that explains its answers and proves the explanations are honest.

**Type (Q1):** Real tool brother (pneumology MD + PhD student) uses + portfolio piece. Brother is the alpha user, his feedback drives features. Writeup is honest about tradeoffs.

**Three differentiators:**
1. **Honest verified citations** — every claim in answer is checked post-hoc against its cited passage; failures surfaced to user, not hidden.
2. **Domain-aware backend** — chunking, embeddings, tokenization tuned specifically for biomedical structure. Not generic ChatGPT-over-PDFs.
3. **Measured rigorously** — eval harness with retrieval and faithfulness metrics on a realistic gold set produced by an actual domain expert.

---

## Locked decisions

### Q2 — Corpus scope
- **A. Pneumology only for v1.** COPD, asthma, ILD, lung cancer, pulmonary HTN, sleep apnea, etc.
- Path to "pneumology + adjacent" (immunology, ID, critical care) post-v1 if v1 lands.

### Q3 — Text source
- **C. PMC Open Access full-text where available + PubMed abstract fallback.**
- UI shows per-result badge: "full-text indexed" vs "abstract only".
- Eval metrics split by source type.

### Q4 — Filter logic
- **D. MeSH (Respiratory Tract Diseases tree + descendants) + journal whitelist union + date cutoff 2015+.**
- Configurable via YAML so brother can tune over time.
- Journal whitelist catches MeSH-lagged recent papers in core journals (NEJM, Lancet, etc.).

### Q5 — Chunking
- **C. Section-aware (IMRaD: Introduction / Methods / Results / Discussion) + sentence-boundary split inside section, target 350 ± 50 DeBERTa-v3 tokens.**
- **Why DeBERTa as canonical tokenizer:** verifier (DeBERTa-v3-large-mnli) and cross-encoder both cap at 512 tokens. DeBERTa is the strictest of the three pipeline tokenizers (MedCPT, DeBERTa, Claude). Measuring chunk size in DeBERTa tokens guarantees `(chunk + sentence)` and `(question + chunk)` fit with comfortable headroom — eliminates silent-truncation false negatives in verifier.
- Each chunk tagged with `section_type` metadata.
- `chunks` table records both `n_deberta_tokens` (canonical) and `n_medcpt_tokens` (sanity check).
- Tables = own chunk + caption. Abstract = own chunk. References list stripped entirely. Figures = caption only.
- Naive 512-token chunking kept as eval baseline for ablation chart.
- Claim-aware chunking (D) mentioned as future work in writeup.
- Side effect: chunk count grows ~20–25% vs the looser 300–500 target. FAISS index ~750 MB instead of ~600 MB. Fits brother's 32 GB RAM trivially.

### Q6 — Embedding model
- **MedCPT (`ncbi/MedCPT-Query-Encoder` + `ncbi/MedCPT-Article-Encoder`)** as primary embedder.
- Trained by NCBI on real PubMed click logs — directly aligned with task.
- `BAAI/bge-large-en-v1.5` kept as eval baseline so writeup can show domain-tuning lift.

### Q7 — BM25 (keyword retrieval) details
- **Library:** `rank_bm25` v1 for simplicity. Switch to Tantivy only if it gets slow.
- **Tokenizer:** custom biomedical regex rules — keep hyphens/plus/slash inside tokens (`IL-4`, `CD8+`, `FEV1/FVC`), keep digit-letter combos (`FEV1`, `25mg`), drop English stopwords. ~50 lines of code.
- **Synonym expansion:** none v1. MedCPT handles semantic synonymy. MeSH synonyms = future work if eval shows specific failure modes.

### Q8 — Cross-encoder reranker
- **`ncbi/MedCPT-Cross-Encoder`** — matched pair to embedder. Coherent NCBI stack end-to-end.
- **Funnel:** top 50 from first pass → reranked → top 10 to LLM.
- **First-pass score combination:** Reciprocal Rank Fusion (RRF, k=60). No tuning, score-scale agnostic.
- Eval baselines: `cross-encoder/ms-marco-MiniLM-L-6-v2` (generic) + `BAAI/bge-reranker-large` (SOTA generic) for the rerank ablation chart.

### Q9 — Citation verification (the trust differentiator)
- **Granularity:** sentence-level. Each sentence in the answer = one claim, verified against its cited chunk(s).
- **Mechanism:** hybrid NLI + LLM-judge.
  - NLI fast pass: `microsoft/deberta-v3-large-mnli`. Confident entailment (>0.9) or contradiction (>0.9) → labeled directly.
  - Borderline middle band (0.3–0.9) → escalated to **Claude Haiku 4.5 judge** (Q16: cost-cut from Sonnet, quality identical for binary entailment task).
  - Roughly 30% of claims need LLM judge; cuts cost ~70% vs LLM-judge-only.
- **Multi-citation semantics (locked Q1): AND-of-singles.** Sentence with `[1][3]` runs NLI(chunk_1, sentence) **and** NLI(chunk_3, sentence). All cited chunks must entail → `supported`. Any contradicts → `unsupported`. Else → judge for joint reading. Catches Claude over-citing (slapping `[3]` on a claim only `[1]` supports).
- **Edge cases (locked Q6):**
  - **Hallucinated `[n]`** (number not in `final_chunks`): `status='unsupported'`, `failure_kind='fabricated_citation'`. Distinguishable in eval/logs from "cited real chunk that doesn't entail".
  - **No-citation sentence** (Claude broke citation discipline): strict — `status='unsupported'`, `failure_kind='no_citation'`. Brother sees red, learns Claude misbehaved. Self-correct via prompt iteration.
  - **Vacuous-but-cited** ("These results are interesting [3]"): NLI returns neutral → judge → `status='unclear'` (yellow). Honest signal that claim is too vague to verify.
- **Failure UX:** color-coded inline markers (green/yellow/red dots) per claim + footer summary ("8/9 claims verified"). No auto-edit, no strikethrough — preserve answer, surface failures honestly.
- Eval baselines: lexical overlap, NLI-only, LLM-judge-only — three-way comparison vs hybrid.

### Q9b — Hard refusal trigger (locked)
- **Trigger:** rerank-score floor on the cross-encoder output. Hardcoded `rerank_floor` constant (config knob, single value).
- **Two effects from one knob:**
  - If top-1 rerank score < floor → **skip Claude entirely**, emit fixed string: `"The retrieved evidence does not address this question."`
  - For chunks that pass top-K but score < floor → **drop from `final_chunks`** before prompt building. Fewer garbage chunks in prompt = less hallucination opportunity.
- **Initial value:** hardcoded constant during M1–M3, calibrate empirically against gold set in M4 (knee where Recall@10 collapses).
- **Logged** in `query_traces` with `refusal: "hard"`. Eval metric `% hard_refusal` reported on adversarial gold-set slice (Q15 acceptance gate).

### LLM integration
- **Two roles, two models** (locked Q16):
  - **Generator** = Claude **Sonnet 4.6**. Streams the answer. Quality matters for prose + citation discipline.
  - **Judge** = Claude **Haiku 4.5**. Verifies borderline NLI cases. Binary entailment task — Haiku is plenty, ~3x cheaper than Sonnet, faster.
- Configurable via `LLM_PROVIDER` env var (paths to OpenAI / Ollama for ablation; Ollama is the documented v2 escape hatch if costs balloon).
- **Hosting model:** BYOK — brother runs the app on his own machine, his own Anthropic API key in `.env`. **Pro/Max subscription cannot power API calls** (verified) — API key is required.
- **Per-query cost (re-priced):**
  - Generator (Sonnet): ~$0.015–0.020/query
  - Judge (Haiku, ~30% of sentences): ~$0.003–0.005/query
  - **Total: ~$0.02/query.**
- **Brother monthly:** ~30 queries/day × 30 days × $0.02 = **~$18/month** (was $30 with Sonnet judge). Capped at $15/month app-side by default; brother edits `config.yaml` to raise.
- **`--full` eval cost:** uses **Anthropic batch API** (50% discount, ~24h SLA, fine for offline eval). ~$28/run instead of $60. Batch API path is **not optional** for `--full` — flag enforces batch endpoint, fails loud if unavailable rather than silently falling back to non-batch ($60 footgun closed).
- **`--full` confirmation prompt (Q21 amendment):** typing `python -m eval --full` prints expected cost + question count, requires `YES` typed back before running. ~5 lines, prevents fat-finger ($28 mistake).
- **`--full` run log (Q21 amendment):** before starting, append `(timestamp, git_sha, expected_cost_usd, gold_set_size)` to `eval/runs.jsonl`; after completion, append `actual_cost_usd`. `tail eval/runs.jsonl` answers "did I already run this?" without grep-ing parquet files.
- **Dev cost over M1–M6 build:** ~$140 total budgeted (5 `--full` runs × $28 batch = $140 floor; ad-hoc dev queries on top of that bring real budget closer to $150). Earlier "$35" figure in this doc was stale and contradicted §13 of `walkthrough.md`; reconciled 2026-05-11.
- **What leaves his machine:** only the generation/judge prompts + retrieved chunks. Index, embeddings, NLI, retrieval, reranking, splitting all run locally.
- **Per-query cost tracking (locked):** every Anthropic call records `cost_usd` into `query_traces`, split into `generator_cost_usd` + `judge_cost_usd`. Enables monthly cap enforcement and "where did the money go" forensics.
- **Prompt caching:** skip v1. Negligible savings on single-user volume.

### Q10 — Eval harness
- **Metrics:**
  - Retrieval: Recall@10, Recall@50, nDCG@10, MRR — all **paper-level** (locked Q3).
  - End-to-end: Faithfulness (reuses production verifier), Citation accuracy, Answer relevance.
  - Honesty: `% hard_refusal` measured on adversarial slice (Q15 gate).
  - Engineering: Latency p50 / p95.
  - **Slices** (locked Q7): faithfulness conditional on `section_type` of cited chunks; section_type histogram per query in `query_traces`. Tells you if abstract chunks dominate retrieval inappropriately.
- **Stack:** custom impl + `pytrec_eval` for retrieval metrics. Faithfulness uses the same NLI + LLM-judge pipeline as production verifier (single source of truth, no train/eval distribution gap).
- **Gold set composition (locked Q11 / Q15; authoring revised Q23):**
  - **50 questions written by dev** sourced **25 from clinical practice guidelines** (GOLD, GINA, ATS/ERS statements) + **25 from Cochrane respiratory reviews**. Brother lacks authoring time; labels (paper-level, ~25 min) at week ~7–8 calendar instead. Trade-off: questions are guideline/review-shaped, not "from his actual workflow"; preserves real-clinician relevance signal where it matters most (Recall@10 ground truth).
  - **150 synthetic** generated by LLM from corpus, brother spot-checks ~20 for quality.
  - **80 from BioASQ pneumology slice** (optional, external validity).
  - **10 adversarial out-of-scope** ("What's the capital of France?", "Best dose of acetaminophen for migraine?") — drives the `% hard_refusal` honesty gate.
  - **Total: 290 items.**
- **Stratification:** each question tagged `{section_focus, topic, difficulty}`. Metrics reported per tag.
- **Relevance labels (locked Q3):** **paper-level (PMID), not chunk-level.** Brother labels "is PMID 12345 relevant for q042?" Recall@K = "did any top-K chunk come from a relevant paper?" Survives chunker/embedder changes — chunker iteration is preserved as the highest-leverage R&D variable.
- **Labeling UI:** small custom UI shows the union of top-50 retrieved papers across multiple retrieval configs for each brother-written question; he clicks relevant / partial / not at the paper level. ~30 sec/question × 50 = **~25 minutes of his time** (down from 2.5 hours under chunk-level labeling).
- **Orchestration:** script-based (`python -m eval`), local SQLite/Parquet results store, comparison-to-previous-run output, top-20-failures debug list. No CI v1.
- **Three modes (architecture §9.4):** `default` retrieval-only ($0), `--full` ($28/run via batch API), `--mock-llm` ($0 cached responses). Default = retrieval-only — most metrics don't need generation. `--full` budgeted to **5 runs across M2–M6** (M2 baseline, M4 first calibration, M5 post-scale, M6 acceptance, +1 slack).

### Deployment
- **Shape B: pre-built index distributed as artifact.** Built once on dev machine, uploaded to HuggingFace Datasets / S3, brother downloads pre-built (~10GB, ~15 min) instead of running 4-6 hour ingest himself.
- **Storage simplification:** SQLite instead of Postgres for v1 (single-user, ships in container, single file on disk).
- **Install ritual for brother:** `git clone` → edit `.env` (Anthropic key) → `./scripts/download_index.sh` → `docker compose up -d`. Total ~25 min start to finish.
- **Containerized:** single `docker compose up` brings backend + bundled SQLite + bundled FAISS + bundled BM25.
- **Ingestion code still exists** as Phase 1 — just not the default path for brother. Documents reproducibility.

### Q11 — Tech stack

#### Q11a — Backend framework
- **FastAPI.** Async-native (clean SSE streaming for token-by-token answer), Pydantic for request/response validation, auto-generated `/docs` Swagger UI.
- Single-user, single-machine deploy → DX wins over scale concerns.

#### Q11b — Frontend
- **v1: plain HTML + vanilla JS + SSE.** ~200 lines total, no build step, served by FastAPI via `StaticFiles`.
- UI surface: question input, streamed answer area, inline color-coded verification dots (green/yellow/red), citation hover/click → side panel.
- **Post-v1: Next.js** rebuild if portfolio needs full-stack signal.
- Reasoning: project differentiator is RAG quality + verification, not frontend craft. Don't burn days scaffolding.

#### Q11c — Vector store
- **FAISS** (in-process, single `.index` file).
- ~200k chunks × 768 dims × 4 bytes = ~600MB → fits RAM easily.
- No separate service in `docker compose` — FastAPI loads index at startup.
- Metadata filter (section_type, MeSH, year) handled in SQLite *before* vector search by passing allowed `chunk_ids`.
- Qdrant deferred to multi-user / cloud-host future.

#### Q11d — Repo layout
- **Domain-split by phase**, single `pyproject.toml`.

```
rag-med/
├── pyproject.toml
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── README.md
├── research/
│   └── decisions.md
├── scripts/
│   ├── download_index.sh       # brother's path
│   └── build_index.sh          # dev path
├── data/                       # gitignored
│   ├── sqlite.db
│   ├── faiss.index
│   └── bm25.pkl
├── src/
│   └── rag_med/
│       ├── __init__.py
│       ├── config.py
│       ├── indexing/           # Phase 1
│       │   ├── pubmed.py
│       │   ├── pmc.py
│       │   ├── chunk.py
│       │   ├── embed.py
│       │   ├── bm25_build.py
│       │   └── pipeline.py
│       ├── serving/            # Phase 2
│       │   ├── api.py
│       │   ├── retrieve.py
│       │   ├── rerank.py
│       │   ├── generate.py
│       │   ├── verify.py
│       │   └── schemas.py
│       ├── eval/               # Phase 3
│       │   ├── runner.py
│       │   ├── metrics.py
│       │   ├── goldset.py
│       │   └── compare.py
│       └── shared/
│           ├── tokenize.py
│           ├── models.py
│           └── db.py
├── static/
│   ├── index.html
│   ├── app.js
│   └── style.css
└── tests/
    ├── test_chunk.py
    ├── test_retrieve.py
    ├── test_verify.py
    └── test_eval.py
```

- Three top folders mirror three architecture phases (Indexing / Serving / Eval).
- `shared/` for cross-phase utils (tokenizer, schemas, DB helpers).
- One venv, one install. `data/` gitignored, mountable as docker volume.

### Q12 — Generation prompt design

#### Q12a — Citation format
- **Inline numeric brackets `[n]`**, multi-cite as `[1][2]` (not `[1, 2]`).
- Strongest Claude prior (academic training data), streaming-friendly, regex-parseable.
- JSON output mode rejected (kills streaming UX, bloats tokens). XML rejected (less natural for academic prose).

#### Q12b — Refusal behavior
- **Soft refusal with explicit gap statement.** Claude states what chunks support, explicitly identifies what is NOT covered.
- Hard refusal exact phrase when zero relevant chunks: `"The retrieved evidence does not address this question."`
- Reasoning: hallucination is THE failure mode this project exists to prevent. Refusing too aggressively → tool feels useless. Soft refusal preserves utility while matching "honest" differentiator.

#### Q12c — Anti-hallucination guardrails
Active in system prompt:
1. **Scope lock** — only chunks, no background knowledge, no inference beyond chunk text.
2. **Citation discipline** — every factual sentence ends with `[n]`.
3. **Verbatim-quote-numbers** — doses, p-values, hazard ratios, FEV1, sample sizes, durations quoted verbatim in `"..."` before citation.
4. **No medical advice framing** — "Study X found..." not "You should...".

Skipped:
- Rigid response template (Direct answer / Evidence / Gaps) — too restrictive for short factual queries. Add post-v1 if eval shows shape inconsistency hurting users.

#### Q12d — Chunk presentation
- **Numbered with metadata header** in prompt context:
  ```
  [1] (Calverley et al. 2007, NEJM, TORCH trial, Methods section)
  "Patients were randomized to..."
  ```
- Author truncation: first author + `et al.` always.
- Cost: ~+500 tokens per query, ~$0.0015 extra. Negligible.
- Win: brother gets "TORCH trial found X [1]" not "Study [1] found X". Section tag helps Claude weight claims (Results > Discussion).

#### System prompt (draft)

```
You answer questions about pneumology research using ONLY the provided
chunks from indexed papers.

Rules:
1. Use only information from the chunks. Do not use background knowledge
   beyond the chunks. Do not infer beyond what a chunk states.
2. Every factual sentence must end with citation marker(s) like [1] or
   [1][3]. Numbers refer to chunks listed below.
3. For numerical results (doses, p-values, hazard ratios, FEV1, sample
   sizes, durations), quote the chunk verbatim in quotation marks before
   citing. Do not paraphrase numbers.
4. Do not give medical advice. Describe what studies report. Use
   "Study X found..." not "You should...".
5. If chunks partially cover the question, state what is supported and
   explicitly identify what is NOT covered.
6. If no chunk addresses the question, respond exactly:
   "The retrieved evidence does not address this question."

Chunks:
[1] (Author et al. YEAR, JOURNAL, SECTION)
{chunk_1_text}

[2] (Author et al. YEAR, JOURNAL, SECTION)
{chunk_2_text}
...

Question: {user_question}
```

---

## Architecture summary

Three phases:

### Phase 1 — Indexing (one-time + occasional refresh)
PubMed E-utilities + PMC OA Bulk FTP → ingestion pipeline (parse XML, IMRaD chunk, tag sections) → three storage layers:
- SQLite (paper metadata, chunk text)
- FAISS (chunk vectors via MedCPT-Article-Encoder)
- BM25 index (custom biomedical-tokenized chunks)

### Phase 2 — Querying (every user question)
Question → MedCPT-Query-Encoder + biomedical-tokenized BM25 → RRF fuse → top 50 → MedCPT-Cross-Encoder rerank → top 10 → Claude Sonnet 4.6 with inline-citation prompt → streamed answer → sentence-split + NLI/LLM-judge verifier → enriched response with color-coded verification markers.

### Phase 3 — Eval (developer-facing)
`python -m eval` runs full pipeline against gold set → metrics table + per-tag breakdown + failure list + comparison-to-previous-run. Outputs Parquet for plotting.

---

### Q13 — Brother's hardware (locked)

- **Machine:** Intel Core Ultra 7 155H (Meteor Lake, 16 cores: 6P + 8E + 2LPE, AVX2 + AVX-VNNI, integrated NPU + Arc iGPU), 32 GB RAM, no discrete GPU.
- **Verdict: workstation tier — no fallback work needed.** All models fit in RAM with comfortable headroom. CPU sufficient for inference.
- **Latency budget on this hardware:**
  - FAISS + BM25 parallel: ~0.3 s
  - Cross-encoder rerank top-50: ~1 s
  - Claude generation (network + Sonnet): ~4–8 s
  - NLI batch over 10 sentences + Haiku judge calls (parallel): ~5–8 s
  - **End-to-end target: ~10–15 s typical, ≤20 s p95** (Q15 gate).
- **Documented future speedups (NOT v1):**
  - PyTorch → ONNX Runtime + INT8 dynamic quant on NLI: ~2× speedup, free, ~1% quality loss.
  - OpenVINO targeting Intel NPU + Arc iGPU on 155H: ~3–5× speedup, significantly more code complexity.
- README will note "designed for workstation-class CPU (≥16 GB RAM, AVX2+); not tested on consumer laptops."

### Q14 — Project shape & milestones (locked)

**Vertical-slice ordering** — end-to-end thin path on day 3, then thicken outward. Each milestone is a working system, just thinner than the next.

**Calendar note (Q23):** week numbers below are *milestone* numbers, not calendar weeks. At ~15–20 hr/wk (locked Q23), each milestone takes ~1.5–2 calendar weeks. Full M1→M6 lands at **~10–12 calendar weeks**.

| Milestone | Scope | Acceptance |
|---|---|---|
| **M1** | End-to-end skeleton on toy 100-paper corpus. Real fetch + chunker + MedCPT + FAISS + BM25 + RRF + cross-encoder + Sonnet generator. **Mock verifier returning all-green.** Real SSE streaming. Plain HTML frontend. | Brother hits `/ask` end-to-end, sees streamed answer with fake green dots. |
| **M2** | Real verifier wired up. NLI + Haiku judge. Sentence dots paint real verdicts. Q1/Q6 edge-case handling in `failure_kind`. | Brother sees first real green/yellow/red dots on toy corpus. |
| **M3** | Eval harness skeleton (retrieval-only mode against synthetic 50-q set; compare-runs script) **+ scale corpus to full 150k papers** (production indexing run on dev M5 Pro; bundle artifact built locally, not yet uploaded). Reorder from earlier plan: scale moved here from M5 because M4 labeling requires broad corpus, else cross-domain questions return 0 results. | Retrieval changes falsifiable via metrics; full 150k indexed locally. |
| **M4** | **Brother labels real gold set** (50 dev-authored questions, paper-level, ~25 min). First `--full` eval against full 150k corpus. Calibrate `rerank_floor` empirically. | Q15 gates measurable on real data. |
| **M5** | **Polish + bundle hosting on HuggingFace** (was: scale + bundle — scale done in M3). Manual `huggingface-cli upload`. Error UX, `/health`, citation panel, version-mismatch banner. | Bundle public; install ritual rehearsable end-to-end. |
| **M6** | Brother's install ritual on his actual i7-155H. Deploy. | Brother runs `./scripts/download_index.sh` + `docker compose up`, asks real questions, Q15 gates verified. |
| **Buffer** | Slack for whatever broke. | n/a |

**Side commitment (revised Q23):** schedule brother's labeling session for **~7–8 calendar weeks out** (was: week 4). Arrange via email/calendar invite *before commit 1*. ~25 min paper-level only.

### Q15 — v1 acceptance gates (locked)

Five gates. Eval harness reports against them in M4 onward. M6 ships only when **all five** are met or honestly marked "below target" in writeup.

| Gate | Metric | Target |
|---|---|---|
| Retrieval | Recall@10, paper-level, strict | ≥ **0.65** on brother's 50-q set |
| Faithfulness | % sentences `supported` | ≥ **0.80** |
| Refusal honesty | % `hard_refusal` on 10-q adversarial slice | ≥ **0.80** |
| Latency | p95 end-to-end on Q13 hardware | ≤ **20 s** |
| User accept | Brother says "I'd use this in actual research" | Yes |

Portfolio writeup includes the gates table with hit/miss column and honest commentary.

### Q16 — Cost discipline (locked)

- **Generator = Sonnet 4.6, judge = Haiku 4.5.** Two distinct models for cost; matches existing two-client architecture.
- **`--full` eval uses Anthropic batch API.** 50% discount, ~24h SLA. Fine offline.
- **Ollama as v2 escape hatch.** `LLM_PROVIDER=ollama` config path documented but generator stays Sonnet for v1. If costs balloon and Q15 gates still hit on local model, swap later.
- **Cost-defense stack (Q21 amendment, 2026-05-11):** five layers, defense-in-depth against three threat models (dev burn, brother burn, single-query blowup):
  1. **App-level monthly cap.** `monthly_cap_usd: 15` in `config.yaml`. `query_traces.cost_usd` summed month-to-date; if > cap, `/ask` returns clean error event. Brother edits config to raise.
  2. **Anthropic console hard limits (REVERSED from earlier "skip" stance).** Set on each API key via the Anthropic dashboard. **Dev key: $50/mo** (one `--full` eval at $28 + headroom). **Brother's key: $30/mo** (2× app-cap as backstop). Defends against any code bug bypassing app cap.
  3. **Per-query cost ceiling.** Abort query if `generator_cost_usd + judge_cost_usd > 0.10` mid-pipeline. Plan typical = $0.02; 5× headroom catches runaway loops without false-tripping legit queries. ~15 lines wrapping the Anthropic client.
  4. **`max_tokens: 1024` on generator.** Bound single-completion length. Without it, Sonnet can ramble to 64k tokens = $0.96 per call. ≈750 words, comfortable for verified pneumology answers.
  5. **80% MTD warning.** When MTD spend > 0.8 × cap, log loud + `/health.cost_warning=true`. Frontend yellow banner. Soft signal before binary fail.
- **Per-query cost tracking:** every Anthropic call stamps `cost_usd` into `query_traces`, split as `generator_cost_usd` + `judge_cost_usd`. Feeds layers 1, 3, 5 above.
- **Cost CLI:** `python -m rag_med cost` prints MTD spend (generator vs judge split), days remaining in month, projected end-of-month at current rate. Read-only SQL over `query_traces`.

### Q17 — Bundle hosting (locked)

- **HuggingFace Datasets**, public repo. Free, fast CDN, native ML tooling. Public is fine — corpus is public-domain PubMed/PMC OA.
- **Single `bundle.tar.gz`** containing `sqlite.db`, `faiss.index`, `bm25.pkl`, `manifest.json`. Plus standalone `manifest.json` at the repo root for cheap version polling.
- **Update trigger from dev:** manual `huggingface-cli upload` after indexing. CI is overkill at v1 update cadence.
- **No diff bundles v1.** Full ~10 GB re-download per refresh.
- **Brother's UX:** banner on version mismatch, no auto-update. Manual `./scripts/download_index.sh` + `docker compose restart`.
- **Eval reproducibility:** Parquet results store `bundle_version` as a column; comparing across versions is a feature.

### Q21 — Cost-defense depth (locked, 2026-05-11)

Three threat models, five-layer defense. Threat models:
1. **Dev burn** — bug in code during M1–M6 hammers Sonnet.
2. **Brother burn** — bug in v1.x update causes runaway in production.
3. **Single-query blowup** — one `/ask` somehow loops Claude calls (e.g. judge in tight loop), single query overshoots before any monthly counter notices.

App-level monthly cap alone defends #2 well, #1 and #3 weakly. Five layers fix that. See updated Q16 §LLM-integration block above for the full enumerated stack:

1. App-level monthly cap (`monthly_cap_usd: 15`, existing).
2. **Anthropic console hard limits** — $50 dev key / $30 brother's key. Reverses the earlier "skip console-side cap" stance, which only made sense for brother's UX, not for dev key.
3. **Per-query cost ceiling** $0.10 — abort mid-pipeline if single query exceeds.
4. **`max_tokens: 1024`** on generator — bound single-completion length.
5. **80% MTD warning** + `python -m rag_med cost` CLI — soft signal before hard fail.

Plus `--full` eval safety: confirmation prompt, batch-API enforced (not optional), `eval/runs.jsonl` run log.

**Cost of all this code: ~50 lines total.** Cheap insurance against worst-case scenarios that decisions.md previously left undefended.

### Q22 — M1 toy ingest mechanics (locked, 2026-05-11)

M1 acceptance ("brother sees streamed answer with fake green dots") needs a working 100-paper toy corpus. Plan §8 covered the indexing pipeline shape but left first-steps mechanics open. Locked here.

#### Q22a — Toy corpus selection
- **Narrow topic, full-text only.** NOT random sample from the full filter (would be sparse, abstract-heavy, hard to write toy questions).
- **Topic = COPD.** Highest density of PMC OA full-text papers post-2020; landmark trials (TORCH, UPLIFT, IMPACT) make toy questions write themselves; strong IMRaD discipline in COPD RCT papers (regulatory pressure on trial reporting) gives the chunker a clean signal in week 1.
- **Date cutoff = 2020+** for M1 (denser full-text), tighter than the full-corpus 2015+ cutoff. M5 broadens to 2015+.
- **Caveat:** if brother's actual research focus is ILD or pulmonary HTN specifically, swap topic. M4 gold-set will reflect his focus regardless; M1 toy queries are written by dev so familiarity matters.

#### Q22b — Fetch mechanics
- **NCBI API key — register day 1.** Free, ~5 min, requires NCBI account + email. Without key: 3 req/s. With key: 10 req/s. Goes in `.env` as `NCBI_API_KEY` + `NCBI_EMAIL` (NCBI requires email field on every request — politeness contract, not auth).
- **HTTP library: `httpx` directly, not Biopython Entrez.** Biopython is heavyweight dep for one module, and its Entrez wrapper hides exactly the URL params and rate-limit headers you'll want to debug at M5. ~50 lines of `httpx` calls.
- **E-utilities exclusively for M1.** Skip PMC OA Bulk FTP / OAI-PMH infra entirely until M5. 100 papers via `efetch` PMC = ~30 seconds even at 3 req/s.
- **M1 esearch query:** `("Pulmonary Disease, Chronic Obstructive"[MeSH] OR "COPD"[Title/Abstract]) AND ("2020"[Date - Publication] : "3000"[Date - Publication]) AND "pubmed pmc open access"[filter]` — PMC Open Access Subset filter guarantees every result has a PMCID (full-text path always exercised). **Drift fix 2026-05-24:** originally locked as `"open access"[filter]`; that string returns 0 hits against PubMed esearch. Correct PubMed syntax for the PMC OA subset is `"pubmed pmc open access"[filter]` — verified empirically during day-3 smoke (5 PMIDs, ~155 KB XML).

#### Q22c — XML parsing
- **`pubmed_parser` library for M1**, escape hatch to hand-rolled `lxml` at M5 if quality bad.
- Library returns paragraphs tagged by `section_name` — exact shape the IMRaD chunker wants. ~50 lines of integration vs ~300 lines of hand-rolled XPath parsing.
- Module boundary is `indexing/ingest/parse.py` — single-file replacement if M5 quality assessment forces a swap.
- **Failure mode is graceful:** if `pubmed_parser` chokes on a paper, log to `failed_papers` table, continue. Plan §8.3 already specifies this.
- **Drift fix 2026-05-24 (day-5 smoke):** real NCBI PMC XML uses `<article-id pub-id-type="pmcid">` (with `"PMC"` prefix in the value) and frequently omits a `pub-id-type="pmid"` element entirely; `pubmed_parser.parse_article_meta` looks for the older `"pmc"` / `"pmid"` tags and returns empty strings for both. Not a parse-side bug — `pipeline.py` already holds the PMID from `esearch` results and can backfill the PMCID from the same call (or `elink`) on `INSERT`. `parse.py` keeps the empty-string fallback; pipeline owns the canonical IDs.

#### Q22d — Malformed-XML salvage rule
Resolves the "salvage rules" sub-bullet from Q19.

**Minimum viable record — keep paper iff:**
- `pmid` present (always true from efetch, defensive only)
- `title` present
- (`abstract` present **OR** ≥1 body section parsed)

**Per-chunk forgiveness:** if one section/table fails to parse, drop that chunk, keep the rest of the paper. Don't let a single bad table sink an entire paper.

Anything failing the rule → row in `failed_papers` table with `failure_reason` column (`missing_title`, `no_content`, `xml_parse_error`, `encoding_error`). At M5, query `failed_papers` to see scale of loss; if > 2% of corpus, write more aggressive salvage.

#### Q22e — Day-1 → M1-ingest-done sequence
Concrete commit order:
```
1. scaffolding         pyproject.toml + ruff + pytest + .env.example + folder layout
2. config              src/rag_med/config.py — Pydantic Settings, NCBI_API_KEY, NCBI_EMAIL,
                       monthly_cap_usd, per_query_ceiling_usd, max_tokens
3. fetch               indexing/ingest/pubmed.py — esearch + efetch wrappers (httpx)
                       hardcoded M1 query (Q22b above); no SQLite yet — print/dump JSON
4. schema              shared/db.py — SQLite create-table for papers + paper_xml + failed_papers
5. parse               indexing/ingest/parse.py — pubmed_parser wrapper, return dict
6. ingest pipeline     indexing/pipeline.py fetch — wires fetch+parse+insert with INSERT OR IGNORE
                       includes salvage rule (Q22d) + retry-3x on network
7. smoke               run pipeline against COPD query, take 100 PMIDs, watch SQLite fill
```
Each commit ~50–150 LOC. ~1–2 days at a normal pace.

**Drift fix 2026-05-24 (day-6 impl):** the step-6 `for each PMID: efetch → save paper_xml row` line is under-specified because `parse.py` (step 5) only handles PMC JATS, not the MEDLINE `<PubmedArticleSet>` shape that `efetch_pubmed` returns. Resolved by inserting an `elink` step between esearch and efetch:
- `pubmed.elink_pubmed_to_pmc(pmids) -> {pmid: "PMC<n>"}` — one HTTP call, multiple `&id=` params yield per-PMID linksets. PMIDs with no PMC counterpart are omitted from the result.
- PMIDs without a PMC mapping land in `failed_papers` with `failure_reason="no_content"` (no full-text retrievable through this pipeline).
- `efetch_pmc` is called **one PMCID at a time** rather than batched. Avoids the multi-article concatenation quirk in `pubmed_parser.parse_article_meta` (`.find` returns only the first `<article-meta>`). Wall cost at the 10 req/s rate limit ≈ 10 s for limit=100 — still inside the spec's 30 s budget.
- Pipeline-level `failure_reason` classification: `efetch_pmc` exception → `xml_parse_error` (retry budget burned in `_get_with_retry`); `parse()` returns its own reason for `xml_parse_error` / `missing_title` / `no_content`. `encoding_error` reserved for an XML decode path not exercised by lxml's tolerant reader.
- `INSERT OR IGNORE` is used on `failed_papers` too, so idempotency holds across re-runs. **Edge case (deferred):** a paper that fails on attempt 1 then succeeds on attempt 2 keeps the stale `failed_papers` row alongside the new `papers` row. Triage during Day-7 smoke or clean up in M5.

**Smoke test for "ingest done"** (`tests/test_smoke_ingest.py`):
- `papers` table has ≥ 95 rows (allows up to 5% salvage loss)
- Every row has `title` non-null
- ≥ 80 rows have `pmcid` non-null (full-text path exercised)
- ≥ 80 rows have body XML in `paper_xml`
- `failed_papers` table has < 5 rows
- Re-running pipeline command twice = identical row count (idempotency via `INSERT OR IGNORE`)

Pass → move to chunking. Fail → triage `failed_papers`, decide if salvage rule needs tweak.

**Anti-pattern (deliberately NOT doing):** vertical-slicing chunking + embedding + FAISS in parallel during week 1. Debugging "why is retrieval bad" with a broken upstream is a nightmare. M1 builds linearly: ingest first, prove it's solid, then chunk, then embed, then retrieve.

---

### Q23 — Final-grill closures (locked, 2026-05-11)

Closes plan §17 open items + adds operational gaps surfaced by final pre-build grill.

**Q23a — Dev machine spec (was undocumented gap).** Dev = **Apple M5 Pro / 24 GB unified RAM / 15 cores (5P + 10E)**. PyTorch MPS path for MedCPT + DeBERTa. Indexing 150k chunks projected ~1–2 hr (vs 12–18 hr CPU-only). 24 GB is enough for serving (~4 GB working set) and for sequential indexing stages; tight for "indexing + serving simultaneously" — don't.

**Q23b — Calendar pace.** **~15–20 hr/wk half-time.** Locked. M1–M6 lands at ~10–12 calendar weeks. Implication: brother's labeling slot ~week 7–8 calendar, not week 4. Schedule with him before commit 1.

**Q23c — Gold-set authoring shift.** Brother no longer authors the 50-q set (zero time). Dev authors from **25 guidelines (GOLD, GINA, ATS/ERS) + 25 Cochrane respiratory reviews**. Brother retains the 25-min paper-level **labeling** slot — preserves clinician relevance signal for Recall@10 ground truth, which was the load-bearing part of his contribution. Q15 gate 5 ("I'd use this in actual research") unchanged.

**Q23d — FAISS index type (closes Q18 partial).** `IndexFlatIP`. Exact search, no training, ~750 MB, <100 ms search on M5 Pro. Approximate indexes (HNSW/IVF) win at million+ scale; at ~250k chunks with serialized single-user requests, exact is free. Eval reproducibility cleaner (no approx-recall variance).

**Q23e — Sentence splitter (closes Q20).** `pysbd`. Plan-default regex would split inside `2.5 mg`, `Fig. 1`, `et al.`, `e.g.`, `p < 0.05`, `vs.` — all common in pneumology answers. 5-line drop-in, ~2 ms per answer. Cheaper than the M2 grilling that "discover-then-fix" implies.

**Q23f — Model weights distribution (closes Q19 partial).** Auto-download from HuggingFace Hub on first boot + `HF_HOME` mounted as docker volume for cache persistence. MedCPT-Query (~400 MB) + MedCPT-Cross (~400 MB) + DeBERTa-v3-large-mnli (~1.5 GB) = ~2.3 GB. First-boot wait ~5 min on broadband; subsequent boots cached. Smallest image (~500 MB). Requires internet on brother's machine at first start; Anthropic API needs internet anyway, so no new failure surface.

**Q23g — M3/M5 reorder.** Plan Q14 said M5 = "scale corpus to 150k", M3 = "eval harness only". But M4 brother-labels against retrievals — needs broad corpus, else cross-domain questions return 0. Reorder: **M3 = eval skeleton + scale to 150k**; **M5 = polish + bundle hosting** (HF upload, version-mismatch banner). Total work same, ordering correct.

**Q23h — SQLite schema sane defaults (closes Q18 partial; locked without further grill).** `PRAGMA journal_mode=WAL` (concurrent reader during writer — eval reads while serving writes), `PRAGMA foreign_keys=ON` (SQLite default OFF is a bug magnet), indices on `chunks(pmid)` + `chunks(section_type)`. **No `query_traces` retention rotation v1** — linear growth ~250 MB/yr fine.

### Pre-commit-1 chore list (must clear before any code, Q23i)

Operational prereqs surfaced in final grill. Each is cheap (~5 min) but blocking.

- [ ] **NCBI account + API key + email** → `.env`: `NCBI_API_KEY`, `NCBI_EMAIL`. Blocker for commit 3 (fetch). Free, 5 min at ncbi.nlm.nih.gov.
- [ ] **Anthropic console hard limit $50/mo on dev key**. Q21 cost-defense layer 2. Console → API Keys → spending limit.
- [ ] **HuggingFace account**. Needed by M5 (bundle host) and now also M1 (model auto-download for cleanliness). 2 min.
- [ ] **Git init + GitHub repo (public)**. Folder currently has no `.git`. Plan §15 assumes public for free GitHub Actions CI.
- [ ] **Confirm 25+ GB free disk on dev**. ~5 GB raw XML + ~10 GB bundle + ~3 GB HF cache + ~2 GB Docker + venv.
- [ ] **Lock brother's labeling slot at ~7–8 calendar weeks out**. Calendar invite while it's a flexible ask. Confirms Q15 gate 5 pathway.
- [ ] **Brother generates own Anthropic API key + $30/mo console limit**. BYOK per LLM-integration. Needed by M6, ask now so he sets up billing.
- [ ] **Start drafting the 50-q gold set during M1–M3.** Don't leave to week 7. Source order: GOLD COPD report → GINA asthma → ATS/ERS ILD/PH statements → Cochrane respiratory reviews. Tag each `{section_focus, topic, difficulty}` as written.

### Risk flags (locked Q23j, not in original plan)

- **Sonnet 4.6 deprecation mid-project.** ~10-week window. Anthropic typically gives months of notice; budget headroom enough to swap to next Sonnet without cost re-plan. Watch model-release announcements.
- **MPS PyTorch quirks.** Some DeBERTa/MedCPT ops may not be MPS-implemented; fallback = CPU per-op. Smoke-test at commit 5 (parse, before pipeline): load one model on MPS, dummy forward. If any op falls back loudly, document in `decisions.md` and pin the model layer to CPU. **Smoke result 2026-05-24:** clean — no CPU-fallback warnings, ~21 ms/pair forward on M5 Pro MPS, sanity pair `("Patients had COPD.", "Patients had a respiratory condition.")` → `entailment` p=0.9986. **Model swap:** the canonical `microsoft/deberta-v3-large-mnli` (Q23f) is no longer on HuggingFace Hub (404 / repo removed). Day-5 smoke used `cross-encoder/nli-deberta-v3-large` — same DeBERTa-v3-large backbone + NLI head, same MPS-coverage answer. M2 verifier must re-lock the production model (candidates: `cross-encoder/nli-deberta-v3-large`, `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`); update Q23f size estimate after pick.
- **`rerank_floor` calibration has no knee.** Plan Q9b assumes a clear knee in the rerank-score-vs-recall curve. If flat, hand-pick value, flag honestly in writeup.

---

## Open questions (next grilling sessions)

Mostly closed by Q22 + Q23. Remaining:

- **Q19 residual:** dedup ordering (PMID/PMCID/DOI) for multi-source future, incremental refresh triggers. Not v1 concerns.

---

## Key reference numbers

- Corpus size estimate: ~150k papers (~40k full-text PMC OA, ~110k abstracts) — **re-measured during M5 indexing**, numbers updated post-run (Q7).
- Total chunks estimate: ~200–250k after IMRaD chunking with 350-DeBERTa-token target. Re-measure during M5.
- Total disk footprint: ~10 GB (papers + indexes + cached models).
- FAISS index size: ~750 MB (200k × 768 × 4 bytes plus overhead).
- **Per-query cost: ~$0.02** (Sonnet generator + Haiku judge; everything else free/local).
- **Brother monthly cost: ~$18** at 30 q/day; **app-level cap default $15**, brother edits to raise.
- **Per-query latency: 10–15 s typical, ≤20 s p95** on i7-155H + 32 GB CPU-only.
- Eval gold set size: **290 questions** (50 brother + 150 synthetic + 80 BioASQ + 10 adversarial).
- Brother's time investment: **~25 minutes** for paper-level gold set labeling (down from ~3 hours under chunk-level labeling).
- Dev cost over M1–M6: **~$140 budgeted** (5 `--full` runs at ~$28 batch-discounted = $140 floor; ad-hoc dev queries push real budget closer to $150). Earlier "$35" figure was stale and contradicted `walkthrough.md §13`; reconciled 2026-05-11.
- Cost-defense layers: 5 (app cap + console limits + per-query ceiling + max_tokens + 80% warning). See Q21.
- Dev machine: **Apple M5 Pro / 24 GB / 15 cores**, MPS path (Q23a).
- Calendar pace: **~15–20 hr/wk → M1–M6 in ~10–12 calendar weeks** (Q23b).
