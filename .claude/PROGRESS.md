# Progress

Living log. Update before `/clear` and when crossing a meaningful step.

## Current

- **Milestone:** week 2 Day 4 done — commit 11 (biomedical tokenizer + rank_bm25 inverted index), all green + real artifact built. Ready for Day 5 (retrieval skeleton: dense + lexical + RRF + cross-encoder).
- **Branch:** `main`. Day 3 (FAISS) committed (`a28506c` + `099911d`); Day 4 work **uncommitted** (4 modified, 4 new). Drift-auditor: zero blocking, 2 minor review items (both addressed — see below).
- **Last done:** Day 4 — (1) `shared/tokenize.py` `bm25_tokenize` per Q7: regex keeps `-`/`+`/`/` intra-token (`il-4`, `cd8+`, `fev1/fvc`) + digit-letter combos (`fev1`, `25mg`), lowercases, drops ~38 stopwords; in `shared/` so eval reuses it. (2) `indexing/bm25_build.py` mirrors `faiss_build`: `build_index` (tokenize → `BM25Okapi`) → `write_index`/`read_index` (pickle + sidecar JSON, trusted-input-only per arch §11.1) → `run_bm25` orchestrator (chunks ordered by `chunk_id`). (3) `pipeline bm25` subcommand + `config.bm25_path`/`bm25_chunk_ids_path`. (4) dep `rank-bm25>=0.2` (0.2.2). 8 new tests (`test_tokenize.py` 4, `test_bm25_build.py` 4), all mocked. Real build: **3045 chunks → `data/bm25.pkl` 3.6M + sidecar 68K**; round-trip + COPD-query sanity clean. Suite **102/102 unit green, 1 network deselected**.
- **Next:** Day 5 commit 12 — `serving/retrieve.py` (densest day, ~4h): `embed`(MedCPT-Query) → `dense_search`(FAISS) + `lexical_search`(BM25) via `asyncio.gather` → `fuse`(RRF k=60) → `rerank`(MedCPT-Cross-Encoder top-10) → `RetrievalResult`. Wires `rerank_floor` (hard_refusal + per-chunk drop), LRU embedding cache. Mock everything in tests. **This is where `bm25.pkl` + `faiss.index` get loaded.**
- **Blockers:** none. Open notes: (a) #3 untitled-body sections — figure-caption reclaim deferred to own task. (b) `TARGET_TOKENS=300` vs Q5 "350±50" — reconcile in decisions.md before M2. (c) drift-auditor day-4 nits: tightened `bm25` subcommand help string off the library name; `data/bm25.chunk_ids.json` sidecar should be added to `architecture.md §11.1` config.yaml paths block (same-PR doc rule).

## Milestones

Work units, not weeks. ~10–12 calendar weeks at 15–20 hr/wk. Full detail in `.claude/research/decisions.md §Q14` + `architecture.md §18`.

- [ ] **M1** — end-to-end skeleton on 100-paper COPD toy corpus, mock verifier (all-green dots).
- [ ] **M2** — real verifier wired (NLI + Haiku judge), real green/yellow/red.
- [ ] **M3** — eval harness skeleton (retrieval-only) + scale corpus to full 150k.
- [ ] **M4** — brother labels 50 dev-authored questions (~25 min, paper-level); first `--full` eval; calibrate `rerank_floor`.
- [ ] **M5** — polish + bundle to HuggingFace + version-mismatch banner.
- [ ] **M6** — brother deploys on i7-155H; Q15 gates verified.

## Q15 acceptance gates (M6 ship criteria)

- [ ] Recall@10 (paper-level, strict) ≥ 0.65 on brother's 50-q
- [ ] Faithfulness (% sentences `supported`) ≥ 0.80
- [ ] % `hard_refusal` on 10-q adversarial slice ≥ 0.80
- [ ] p95 latency on i7-155H ≤ 20 s
- [ ] Brother: "I'd use this in actual research"

## Budgets

- **Dev cost M1–M6:** ~$140 floor (5 `--full` batch runs × $28). Real ~$150 with ad-hoc.
- **`--full` runs reserved:** M2 baseline · M4 first calibration · M5 post-scale · M6 acceptance · +1 slack = 5 total.
- **Calendar:** brother's labeling slot at ~cal week 7–8.

## Pre-commit chores (Q23i)

- [ ] NCBI API key + email in `.env`
- [ ] Anthropic dev key + $50/mo console limit
- [ ] HuggingFace account
- [x] git init + GitHub repo
- [ ] ≥25 GB free disk verified
- [ ] Brother labeling slot booked (~cal week 7–8)
- [ ] Brother Anthropic key + $30/mo console limit
- [ ] Gold-set drafting (deferred to week 7, batch-style)

## Recent sessions

- 2026-05-23 — wrote CLAUDE.md + PROGRESS.md from `.claude/research/` analysis.
- 2026-05-22 — config commit `e87b4c3`.
- 2026-05-24 — Day 3 ingest commit `f49a3b5`: httpx esearch/efetch_pubmed/efetch_pmc with rate guard + retry.
- 2026-05-24 — Day 4 SQLite schema: papers + paper_xml + failed_papers, WAL + FK pragmas, CHECK enum on failure_reason. CLAUDE.md `## Hard rules` now mandates TDD.
- 2026-05-24 — Day 5 parse: `pubmed_parser` wrapper + Q22d salvage rule, 4 JATS fixtures + 5 tests. Smoke on real PMC13197932 → 73 grouped sections. MPS DeBERTa smoke clean at 21 ms/pair on `cross-encoder/nli-deberta-v3-large` (spec model `microsoft/deberta-v3-large-mnli` 404'd on HF — drift in `decisions.md` Q22c + Q23j).
- 2026-05-24 — Day 6 pipeline: `indexing/pipeline.py` async `run_fetch` + CLI `fetch` subcommand, structlog JSON to stdout. Added `pubmed.elink_pubmed_to_pmc` (PMID→PMCID, repeated `&id=` params), refactored `parse()` to return `(dict, reason)`. One-PMCID-per-efetch to dodge the multi-article concat quirk. 11 new tests over mocked I/O; full suite 32/32. Drift in `decisions.md` Q22e.
- 2026-05-24 — Day 7 smoke + week-1 close: real M1 fetch (100/100 parsed). Six-gate `test_smoke_ingest.py` green incl. idempotency. Unblockers: elink batching at 50 (`c72d669`), `pubmed_parser` `KeyError('year')` patch on epub-only papers (`12a303b`). Smoke `9dfd7b2`. Branch `feat/m1-smoke-day7` pushed.
- 2026-05-24 — Wrote `steps/week2.md` (commits 8–14, M1 acceptance). Executed Day 0: folded week-1 drift into `decisions.md §Q22c` + `§Q22e` + `glossary.md` M1-toy-corpus. Three files changed, no code. Ready to commit + PR.
- 2026-05-26 — Day 1 commit 8: IMRaD chunker (`indexing/chunk.py`) + `chunks` DDL in `shared/db.py` + 14 chunker tests + 4 schema tests. pysbd + transformers added. 50/50 unit green; smoke `test_idempotent_rerun` flaked on NCBI network. Branch `feat/m1-chunker-day1`.
- 2026-05-29 — Day 2 commit 9 + chunker fixes on `feat/week2-day2` (5 commits, unpushed): `pipeline chunk`/`run_chunk` + `indexing/embed.py` MedCPT-Article embedder (mock seams, no torch in tests); real run 100 papers → 3045 chunks. Fixes from real run: abstract >400 split, IMRaD subsection map (`other` 70%→53%, methods 40→360), file-marker noise drop, `@network` smoke marker. Deps: torch/numpy/sentencepiece/protobuf. 89/89 unit green, 1 network deselected. drift-auditor clean. `decisions.md §Q5` +2 drift notes.
- 2026-05-31 — Day 4 commit 11 (uncommitted on `main`): `shared/tokenize.py` `bm25_tokenize` (Q7 biomedical regex) + `indexing/bm25_build.py` (`build_index`/`write_index`/`read_index`/`run_bm25`, pickle + sidecar) + `pipeline bm25` subcommand + `config.bm25_path`/`bm25_chunk_ids_path` + dep `rank-bm25`. 8 new tests, all mocked. Real build 3045 chunks → `data/bm25.pkl` 3.6M; round-trip + COPD query sanity clean. 102/102 unit green. drift-auditor: 0 blocking, 2 nits addressed (help-string wording + arch §11.1 sidecar path note). Day 3 FAISS landed earlier in `a28506c`+`099911d`.
