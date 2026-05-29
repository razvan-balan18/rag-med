---
description: Update PROGRESS.md before /clear or at a meaningful step
allowed-tools: Bash(git *), Read, Edit
---

Update `.claude/PROGRESS.md` to capture current state. This is the ritual before any `/clear` and when crossing a meaningful step (commit landed, milestone reached, blocker found).

Steps:
1. Run `git log --oneline -5` and `git status` to ground the entry in real commits, not memory.
2. Edit the **`## Current`** block: refresh `Milestone`, `Branch`, `Last done`, `Next`, `Blockers`. Be concrete — name files, LOC, test counts, commit SHAs.
3. Append ONE dated line to **`## Recent sessions`** (format: `- YYYY-MM-DD — <what landed>`). Use today's date.
4. If a milestone or pre-commit chore completed, tick its checkbox in `## Milestones` / `## Pre-commit chores`.

Keep entries terse and factual — this file is the cold-start context for the next session. Don't invent progress; if unsure what changed, ask or inspect the diff.

If $ARGUMENTS is given, treat it as the summary of what was just done and fold it into the entry.
