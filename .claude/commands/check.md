---
description: Run all pre-commit quality gates (ruff, pytest, phase-isolation, client split)
allowed-tools: Bash(uv run *), Bash(grep *), Bash(ruff *)
---

Run the project's quality gates and report a pass/fail punch list. Do NOT fix anything yet — just report. Run these in parallel where possible:

1. **Lint** — `uv run ruff check src tests`
2. **Format** — `uv run ruff format --check src tests`
3. **Tests** — `uv run pytest` (unit suite; `test_smoke_ingest.py` hits real NCBI and may flake on network — note separately, don't count a network flake as a real failure)
4. **Phase isolation** — `grep -rn "from rag_med" src/rag_med` then check the hard rule: `indexing`/`serving`/`eval` must not import each other; share via `shared/`. **One allowed exception:** `eval` may import `serving` directly (architecture.md §9.1). Flag any other cross-phase import.
5. **Two-client split** — `grep -rn "claude_client\|anthropic" src/rag_med/serving` — there must never be a shared `claude_client`; only `generator_client` (Sonnet 4.6) and `judge_client` (Haiku 4.5).

Report each gate as ✅ / ❌ with the failing detail inline. End with one line: ready to commit or not.

If $ARGUMENTS is non-empty, scope tests to that path/expression (e.g. `/check test_chunk`).
