# Progress

Living log. Update before `/clear` and when crossing a meaningful step.

## Current

- **Milestone:** week 1 done — M1 ingest scaffold landed. Ready for week 2: chunker → embedder → FAISS/BM25 → mocked-verifier `/ask`.
- **Branch:** `feat/m1-smoke-day7`
- **Last done:** Day 7 smoke green. 100/100 COPD papers ingested, 0 salvaged, 0 failed. All six Q22e gates pass (`tests/test_smoke_ingest.py`, ~99s incl. idempotent re-fetch). Two unblockers landed before the smoke: (a) `elink_pubmed_to_pmc` now batches GETs in chunks of 50 — NCBI stream-closes at ~100 ids in one URL (`fix(ingest): batch elink GETs to 50 PMIDs`, `c72d669`); (b) monkey-patch on `pubmed_parser.pubmed_oa_parser.parse_date` seeds `year=None` on the collection fallback, dodging the upstream `KeyError('year')` that was salvage-failing ~40% of epub-only papers (`fix(parse): patch pubmed_parser KeyError on missing pub-year`, `12a303b`). Smoke commit `9dfd7b2`. Full suite 38/38 green.
- **Next:** Week 2 (commits 8–14): IMRaD chunker (350 DeBERTa-token target) → MedCPT-Article embedder → FAISS `IndexFlatIP` build → BM25 build → retrieval skeleton → end-to-end `/ask` with mocked verifier returning all-green (M1 acceptance per Q14).
- **Blockers:** none. Drift worth folding into `decisions.md` next session: Q22b elink batch ceiling (≤50 ids/GET), Q22c pubmed_parser 0.5.1 `KeyError('year')` workaround.

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
