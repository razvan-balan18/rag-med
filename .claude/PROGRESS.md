# Progress

Living log. Update before `/clear` and when crossing a meaningful step.

## Current

- **Milestone:** week 1 closed + week 2 plan written + week-1 drift folded into research docs. Ready for commit 8 (IMRaD chunker).
- **Branch:** `main` after week-1 merge; week-2 Day-0 drift-fold edits uncommitted (user is committing + opening PR).
- **Last done:** Wrote `steps/week2.md` (commits 8–14: chunker → embedder → FAISS → BM25 → retrieval skeleton → Sonnet generator + mock verifier → `/ask` SSE; M1 acceptance per Q14). Then executed week-2 Day 0 — folded three drift items into research docs (no code): `decisions.md §Q22e` updated to note `elink_pubmed_to_pmc` batches GETs at 50 ids per call (`c72d669`); `decisions.md §Q22c` got a new drift-fix bullet documenting the `pubmed_parser==0.5.1` `parse_date` `KeyError('year')` monkey-patch (`12a303b`) plus the `==0.5.1` pin + M5 cleanup path; `glossary.md` M1-toy-corpus filter corrected to `"pubmed pmc open access"[filter]` with Q22b backref. Q22b filter fix and Q23j DeBERTa swap were already in decisions.md — confirmed, no edit. Suggested commit msg: `docs(decisions): fold week-1 drift (elink batch, year patch, glossary filter)`.
- **Next:** Commit 8 — IMRaD chunker per `steps/week2.md` Day 1 (Q5 + Q23e). ~180 LOC, TDD-first, fixture-driven tests, new `chunks` table DDL appended to `shared/db.py`.
- **Blockers:** none.

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
