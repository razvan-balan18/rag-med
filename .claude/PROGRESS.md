# Progress

Living log. Update before `/clear` and when crossing a meaningful step.

## Current

- **Milestone:** pre-M1 (ingest scaffold, Day 5/7 of `steps/week1.md` done)
- **Branch:** `feat/parse`
- **Last done:** parse landed. `src/rag_med/indexing/ingest/parse.py` wraps `pubmed_parser.parse_pubmed_xml` + `parse_pubmed_paragraph(all_paragraph=True)`, groups paragraphs by section title, applies Q22d salvage rule (None iff no title OR no abstract+body OR XML unparseable). 5 unit tests green on JATS fixtures + real-PMC smoke (73 sections on PMC13197932). MPS DeBERTa smoke clean (~21 ms/pair, no CPU fallback) — but spec model `microsoft/deberta-v3-large-mnli` gone from HF; used `cross-encoder/nli-deberta-v3-large`. Both drifts logged in `decisions.md` Q22c + Q23j.
- **Next:** Day 6 ingest pipeline (`indexing/pipeline.py fetch`) wiring esearch → efetch_pmc → parse → INSERT, backfilling PMID/PMCID from esearch (parser leaves them empty). Day 7 smoke gate on 100 COPD papers.
- **Blockers:** none. M2 verifier model needs re-lock (Q23f) — not Week 1.

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
