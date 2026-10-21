---
name: run-analysis
description: Run anomaly-report for a company (single year, --period, or a multi-year --span review) and present the result. Use when the user asks to run, analyse or review a company, or types an anomaly-report command.
---

# Run an analysis

1. **Run** from the project root, in the background (runs take minutes), with no terminal input:
   `.venv/Scripts/anomaly-report.exe "<company>" [--period FY2024] [--span N] < /dev/null`
   Use the user's options exactly. A run replaces the earlier report for the same company and period.
2. **Read the result**: the console summary (risk index, findings, timings, AI cost) and `reports/<run>/report.md`.
   For details load the saved run: `snapshot.load(Path("reports/<run>"))` (findings, signals, `agent_cases` with the
   investigation and auditor verdicts).
3. **Check before presenting**:
   - The key figures (revenue, net income) match the company's filings.
   - For each finding, both figures the test compared are like for like (CLAUDE.md, Comparisons). An investigation that
     did not finish, a "set aside" result, or an auditor objection that blames the test is a lead to a test error.
   - If a finding is a test error: say so, fix it under the CLAUDE.md rules, re-run, and report the corrected result.
4. **Present** to the user, briefly: the risk index and band (per year for a span); each finding in plain words with its
   period, size and investigation outcome; what was fixed, if anything; the report path; and that nothing is committed.
