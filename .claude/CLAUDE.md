# rag-med

Pneumology RAG with verifier-gated answers. Single-user tool for brother (MD/PhD). Hybrid retrieval (FAISS + BM25 + RRF + MedCPT cross-encoder) → Claude Sonnet generator → NLI + Haiku judge verifies every sentence against cited chunks.

# Project Research

Read this everytime a new conversation starts.
@.claude/research/architecture.md
@.claude/research/decisions.md
@.claude/research/glossary.md
@.claude/PROGRESS.md


## Source of truth

Locked decisions live in `.claude/research/`. **Read before non-trivial changes and at the start of every conversation** — don't re-derive from code.

- `decisions.md` — what we picked, why (Q1–Q23)
- `architecture.md` — runtime behavior
- `glossary.md` — ubiquitous language, banned synonyms enforced
- `steps/week1.md` — current execution plan

If code conflicts with research docs, research wins. Flag the drift.

## Progress

Current state lives in `PROGRESS.md`. **Update it before any `/clear` and when crossing a meaningful step** (commit landed, milestone reached, blocker found). Append a one-line entry to `Recent sessions` + edit `Current`.

## Tooling

- `/check` — run all quality gates (ruff lint+format, pytest, phase-isolation, two-client split). Run before any commit.
- `/progress` — update `PROGRESS.md` (Current + Recent sessions). Run before `/clear` or at a meaningful step.
- `drift-auditor` agent — audits a diff against the research docs for banned synonyms, phase-isolation breaks, merged clients, and contradictions with locked Q-decisions. Use before committing non-trivial changes.

## Hard rules
- TDD always: failing test first, minimal impl, run green. No code without test
- Three phases (`indexing` / `serving` / `eval`) never import each other. Share via `shared/`.
- Two Anthropic clients: `generator_client` (Sonnet 4.6) + `judge_client` (Haiku 4.5). Never merge.
- Follow `glossary.md` names: `paper` not `article`, `question` not `query`, `dense_search`/`lexical_search` not `semantic`/`keyword`, `n_deberta_tokens` not bare `n_tokens`, `paper-level` relevance.
- Mock everything in unit tests. Real SQLite only in `test_smoke_ingest.py`.
- Don't commit or push without explicit ask.
