# Progress

Living log. Update before `/clear` and when crossing a meaningful step.

## Current

- **Milestone:** week 2 Day 2 done — commit 9 (chunk pipeline subcommand + MedCPT-Article embedder) + chunker-quality fixes, all green. Ready for Day 3 (FAISS IndexFlatIP build).
- **Branch:** `feat/week2-day2` (5 commits, unpushed) off `main`. Drift-auditor: clean, no blocking.
- **Last done:** Day 2 — (1) `pipeline chunk` subcommand + `run_chunk` in `indexing/pipeline.py` (parse stored XML → backfill pmid → `chunk_paper` → `INSERT OR IGNORE`, idempotent via LEFT JOIN); (2) `indexing/embed.py` MedCPT-Article-Encoder, `embed_chunks → float32 (N,768)`, CLS pooling, MPS w/ CPU fallback, `_load_model`/`_forward` mock seams (no torch in tests). Real run: 100 papers → **3045 chunks**. Plus chunker fixes surfaced by the real run: abstract >400-tok split (Q5; was 39/99 truncated → 0), expanded IMRaD section_name→enum map (`other` 70%→53%, methods 40→360), drop supplementary-file-marker noise chunks (`(DOCX)`/`(TIF)`), `@pytest.mark.network` on the NCBI smoke test (skipped by default via `addopts -m 'not network'`). Deps added: `torch`, `numpy`, `sentencepiece`, `protobuf` (DeBERTa-v3 SentencePiece tokenizer needs the latter two on transformers 5.9). Commits `cb33e36`, `8f8364e`, `2e9735f`, `a4eb701`, `11b3ef8`. Suite **89/89 unit green, 1 network deselected**. `decisions.md §Q5` got two drift notes.
- **Next:** push `feat/week2-day2` + PR; then Day 3 commit 10 — `pipeline embed` end-to-end stage: read chunks → `embed_chunks` → L2-normalize → `faiss.IndexFlatIP(768)` → `data/faiss.index` + `data/faiss.chunk_ids.json` sidecar.
- **Blockers:** none. Two open notes: (a) #3 untitled-body sections — 87% of empty `section_name` paras are genuine body text (`other` is honest); reclaiming the 13% that are figure captions needs a `parse.py`/`parse_pubmed_caption()` rework, deferred to its own task. (b) `TARGET_TOKENS=300` vs Q5 "350±50" — pre-existing doc/code mismatch, reconcile in decisions.md before M2.

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
