# Progress

Living log. Update before `/clear` and when crossing a meaningful step.

## Current

- **Milestone:** week 2 Day 1 done — commit 8 (IMRaD chunker) implemented + green. Ready for Day 2 (chunk pipeline subcommand + MedCPT-Article embedder).
- **Branch:** about to push `feat/m1-chunker-day1` off `main`.
- **Last done:** Day 1 commit 8 — `src/rag_med/indexing/chunk.py` (~175 LOC) with `Chunk` dataclass, pysbd sentence splitter, greedy-pack to 300 DeBERTa tokens (ceiling 400), section-name → enum substring map, abstract/table/caption special-cased, `chunk_id={pmid}_{section_type}_{ordinal:02d}`. DeBERTa + MedCPT-Article tokenizers lazy-loaded via `transformers.AutoTokenizer`. `shared/db.py` gained `chunks` table DDL + `SECTION_TYPES` tuple + `chunks_pmid_idx`/`chunks_section_type_idx`. `tests/test_chunk.py` (14 tests, tokenizers monkeypatched to word-split, real pysbd for abbreviation test) + 4 new tests in `test_db_schema.py`. Deps added: `pysbd>=0.3`, `transformers>=4.40`. Suite: 50/50 unit green; `test_smoke_ingest.py::test_idempotent_rerun` flaked on NCBI peer-closed-connection (network, unrelated to chunker).
- **Next:** Day 2 commit 9 — `python -m rag_med.indexing.pipeline chunk` subcommand + `src/rag_med/indexing/embed.py` MedCPT-Article-Encoder on MPS. ~150 LOC.
- **Blockers:** none. Smoke flake to retry next session.

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
