# Project rules

anomaly-report: SEC filings → accounting tests (`src/ffi/detectors`) → scoring (`src/ffi/scoring`) → AI investigation
and audit (`src/ffi/agent`, `src/ffi/audit`) → client reports (`src/ffi/report`). Formulas: `Algorithms.md`. Usage: `README.md`.

## Comparisons

Any code that compares or combines two figures must match them on period (quarter / year to date / full year, never
guidance), point in time, scope (whole company vs part), concept (total vs component), basis (GAAP vs adjusted, diluted,
first reported), unit and scale, sign, and the same measure across years; and must count each amount once.

- Never guess. Period, scope, basis and unit come only from explicit evidence (label, column header, sentence, XBRL concept).
  Missing or ambiguous evidence means skip the comparison and say so. Review every fallback (`else`, `or`, defaults).
- Anchor figures parsed from text to the filed XBRL figures for the same and neighbouring periods before using them.
- Fix the root cause and every sibling use of the same pattern, not one case.
- Every fix: a regression test from the real wording or layout, and the change documented in `Algorithms.md`.

## Workflow

- Venv only: `.venv/Scripts/python`, `.venv/Scripts/ruff check src tests tools`, `.venv/Scripts/python -m pytest -q -p no:cacheprovider`.
  Lint and tests must pass before a task is done.
- After changing a detector, the fact mapping or text parsing, run `.venv/Scripts/python tools/review/sweep.py`
  (`--wide` for broad changes); it must report no errors, implausible results or cross-source disagreements.
- Bash heredocs corrupt `\b`, `\n`, `\s` in Python source: edit Python with Edit/Write, then confirm
  `grep -rlP "[\x00-\x08\x0b\x0c\x0e-\x1f]" src tests tools --include=*.py` prints nothing.
- Saved runs keep no filing text; rebuild documents from the cache with `EdgarClient(offline=True)` and `build_doc`.
- Never read or commit `.env`. Commit or push only when asked. Never delete reports or caches unless asked.
- AI-mode runs cost money: test with `--mode deterministic --out <temp dir>` unless the change only shows in AI output.
- Report text is client-facing: plain language, no internal codes, no `None`/`nan`, every figure with its period and basis.
