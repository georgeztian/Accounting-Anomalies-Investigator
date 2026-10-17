# Accounting Anomaly Investigator

Give it a public company. It reads the company's SEC filings and writes an **accounting-anomaly report**. Every number in the report links back to the filing it came from.

```powershell
anomaly-report "Super Micro Computer (SMCI)"
```

The report contains:

- A 0–100 risk index.
- Ranked findings. Each finding shows:
  - its estimated impact and materiality, in the company's reporting currency;
  - the evidence: XBRL facts and verbatim filing quotes, each linked to EDGAR;
  - the innocent explanations that were tested;
  - questions to ask management.
- An **independent audit** of every finding. Its evidence is re-derived from the source filings, and any problem goes through a correction loop of at most 3 rounds.
- Results for every detector, with a reason for each one that was skipped or doesn't apply to this kind of company.
- A data-quality section and a methodology section.

> The app flags departures from expectation for further inquiry. It does not allege wrongdoing. It is decision support for qualified analysts, not investment, legal or audit advice.

---

## Requirements

- Python **3.11 or 3.12**, run inside a **virtual environment**. Outside one, the app exits with code 2.
- Internet access to `sec.gov`. After the first run, `--offline` works from the local cache.
- Optional, for the LLM text detectors and the investigation agent, either of:
  - **a Claude subscription** (Pro or Max) with [Claude Code](https://claude.com/claude-code) installed and logged in; or
  - an Anthropic API key.

## Setup

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv            # or: uv venv --python 3.12 .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.lock  # exact pinned versions
pip install -e . --no-deps
Copy-Item .env.example .env       # then edit it (see below)
```

macOS / Linux:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock && pip install -e . --no-deps
cp .env.example .env
```

Without the lock file: `pip install -e ".[llm,pdf]"`.

Edit `.env`:

```ini
SEC_USER_AGENT="Your Name your.email@example.com"   # required: SEC asks every client to identify itself
ANTHROPIC_API_KEY=                                  # optional: leave empty to use your Claude subscription instead
# FFI_LLM_BACKEND=auto                              # auto | claude-code | api
# FFI_CLAUDE_CLI=C:\path\to\claude.exe              # only if the claude CLI isn't found automatically
# FFI_CACHE_DIR=.cache
# FFI_REASONING_MODEL=claude-opus-5-5
# FFI_EXTRACTION_MODEL=claude-haiku-5-5
# FFI_WRITING_MODEL=claude-sonnet-5-5                 # writes the plain-language explanations
```

### Choosing how the LLM runs

| Backend | When it is used | How it works | Cost |
|---|---|---|---|
| `claude-code` | `auto` with no `ANTHROPIC_API_KEY`, or `FFI_LLM_BACKEND=claude-code` | Calls the Claude Code CLI headless (`claude -p`) with your Claude subscription login. Each call has no tools, MCP servers or skills; it only reads the prompt and returns schema-validated JSON. | Counts toward your subscription's usage limits. It is not billed per token. |
| `api` | `auto` when `ANTHROPIC_API_KEY` is set, or `FFI_LLM_BACKEND=api` | Anthropic Messages API with structured outputs | Billed per token to your API account |

The app finds the Claude CLI in this order:
1. `FFI_CLAUDE_CLI`;
2. `claude` on your PATH;
3. the binary bundled with the Claude Code extension for VS Code or Cursor;
4. `~/.local/bin` and `~/.claude/local`, where the Claude Code installer puts it.

Log in once by running `claude` in a terminal. The API key is removed from the CLI's environment, so subscription runs are never billed to an API account.

With either backend, `--budget-usd` caps the run. For subscription runs the cap applies to the API-equivalent cost, which keeps long agent investigations bounded.

To change tuning settings, create an optional `ffi.toml` in the project folder. Anything you leave out keeps its built-in default; the values below are the defaults:

```toml
[peers]
count = 15            # peers compared against
min_candidates = 30   # widen the SIC code from 4 to 3 to 2 digits until this many active filers

[materiality]
pretax_pct = 0.05     # auditor-style thresholds
revenue_pct = 0.005
assets_pct = 0.005
equity_pct = 0.01

[scoring]
fdr_q = 0.10          # false-discovery rate (Benjamini-Hochberg)
flag_severity = 0.5   # severity at which a test result is flagged

[agent]
max_cases = 3         # findings the investigation agent takes on
run_budget_usd = 6.0  # AI spend cap per run (API-equivalent on a subscription)
recursion_limit = 40
```

---

## Usage

**A run produces one report.** By default it covers one fiscal year; `--span` makes it cover several. Three options are easy to confuse:

| Option | What it controls | Default |
|---|---|---|
| `--period FY2022` | **Which fiscal year the report covers** (with `--span`, the last year of the span). | the latest annual report |
| `--span 10` | **How many fiscal years one report covers.** Each year is analysed on its own, then the findings are merged into anomalies tracked across the years (see *Multi-year review* below). | one year |
| `--years 10` | **How much history feeds the tests** for each year analysed (trends, the company's own baselines, multi-year ratios). It never adds years to the report. | 10 |

For a separate report on each of the last five years instead, run each year on its own:

```powershell
foreach ($y in 2021..2025) { anomaly-report AAPL --period FY$y }
```

```powershell
anomaly-report "Apple Inc. (AAPL)"                 # preferred input form: "Company name (TICKER)"
anomaly-report AAPL                                # a ticker works
anomaly-report "Apple Inc."             # a name works (asks you to choose if ambiguous)

anomaly-report AAPL --mode deterministic           # no LLM at all
anomaly-report AAPL --period FY2024                # the fiscal year the report covers (default: the latest)
anomaly-report AAPL --as-of 2020-06-30             # point in time: only filings public on that date
anomaly-report AAPL --years 7                      # years of history the tests use (default 10); still one report
anomaly-report AAPL --span 10                      # one report covering the last 10 fiscal years
anomaly-report AAPL --span 5 --period FY2022       # one report covering FY2018 to FY2022
anomaly-report AAPL --peers MSFT,GOOGL,DELL        # override automatic peer selection
anomaly-report AAPL --format html,md,json,pdf      # pdf needs the [pdf] extra
anomaly-report AAPL --mode auto --budget-usd 3     # unattended agent with a $3 LLM cap
anomaly-report AAPL --pause-on review_publish      # stop at the publish review and save state
anomaly-report AAPL --resume AAPL_FY2026_ab12cd    # continue a paused or interrupted run
anomaly-report AAPL --offline                      # use cached data only
anomaly-report AAPL --keep-artifacts               # keep the run's graph database and agent checkpoints

anomaly-report render                              # rebuild the latest report with the current templates (no re-run)
anomaly-report render AAPL_FY2025_33b3ff           # rebuild one saved run's report

anomaly-report batch companies.txt --workers 3     # many companies (one per line), one ranked summary
anomaly-report detectors                           # list the detector catalogue
anomaly-report cache-info                          # cache location and size
python -m ffi AAPL                                 # same as anomaly-report
```

Training the supervised misstatement model (optional; a trained model ships in `src/ffi/models/`):

```powershell
anomaly-report labels --start 2016 --end 2022                     # starter labels from 8-K Item 4.02 filings
anomaly-report train --labels data/labels_4_02.csv --backtest     # walk-forward evaluation
anomaly-report train --labels data/labels_4_02.csv                # fit and save the model
```

### Multi-year review

`--span N` reviews the last N fiscal years in one report:

1. The company's filings and figures load once, and each annual report, earnings release and peer company is read
   once for all the years that use it. Each year is then analysed exactly like a single-year run, with its
   own comparison group, its own annual-report text and earnings releases, and the history available up to that year. Each year's figures are
   taken **as first reported**, which is what investors saw at the time.
2. The findings are **merged across years**. The same scheme pattern in several years becomes one anomaly; a single test
   flagged in one year joins that pattern when the pattern appears in another year. Each anomaly records the years it was
   flagged in and its strength in every year, and recurrence raises its score.
3. The AI investigates the **top 5 merged anomalies** (`--max-cases`), each in the context of the most recent year it was
   flagged in, with every year's results and the earlier years' annual reports in its evidence.
4. The independent re-check, the plain-language explanations and the report follow, as in a single-year run.

The first run downloads every year's annual reports and takes several minutes; later runs use the cache. The default AI
cost cap doubles (`--budget-usd`, $12) because every year's filings are read. The earliest years have less history, so
some of their tests cannot run; the report lists them.

### Modes

| Mode | What happens |
|---|---|
| `interactive` (default in a terminal) | The agent stops at each review point and asks you to approve, edit or reject. |
| `auto` | Fixed policies answer the review points. Each answer is labelled *auto* in the report. |
| `deterministic` | No LLM. Narratives come from templates. |

With no API key and no logged-in Claude CLI, the app runs in deterministic mode and says so in the report.

### Which companies it covers

The app covers any company that has filed at least one annual report on EDGAR: a **10-K, 20-F or 40-F**. The company's industry selects a **profile**. Each profile switches off the detectors whose inputs mean something different for that business model, and the report lists every one of them with the reason.

| Profile | SIC codes | What changes |
|---|---|---|
| Industrial / commercial | all others | The full detector set |
| Bank or lender (including card issuers) | 6000–6199 | Off: models built on trade receivables, inventory, operating-cycle accruals, operating cash flow and corporate leverage (e.g. Beneish, Sloan, Altman, DSO, CFO/NI, implied interest rate). Peer-ratio models drop ratios that don't apply. Earnings-release checks use net income only, because banks headline taxable-equivalent revenue. |
| Broker-dealer or asset manager | 6200–6299 | As for banks |
| Insurer | 6300–6411 | As for banks, plus the receivable-allowance checks |
| Real-estate company or REIT | 6500–6553, 6798 | Off: inventory, Altman Z, capex intensity and soft-asset checks |
| Holding or investment company | 6700–6799 | As for banks |
| SPAC / shell company | 6770 | Only integrity, governance and disclosure checks |

For financial and real-estate profiles, the earnings-management test also drops its small-ROA and real-activity parts, because those companies earn a small return on assets as a matter of course.

**Foreign filers.**
- 20-F and 40-F filers are supported, including IFRS statements. Their concepts map to the same line items as US GAAP.
- Amounts stay in the company's **reporting currency** (for example €, ¥, C$), and the report shows them that way.
- Peers are US filers. When the company reports in another currency, peers are matched on margins and growth but not on size.
- Models fitted on USD amounts or on 10-K restatement history are switched off.
- Quarterly and 8-K-based detectors run only where equivalent data exists.

**Other cases.**
- If a company has filed only one annual report, the app warns you and uses that report's prior-year comparatives.
- If a company has filed no annual report yet, the app exits with code 4.
- If a ticker moved to a new holding company with no annual reports yet, the app finds the operating predecessor and says so in the report.

### Exit codes

| Code | Meaning |
|---|---|
| 2 | Not running in a virtual environment |
| 3 | Company could not be resolved |
| 4 | Company has not filed an annual report (10-K, 20-F or 40-F) yet |
| 5 | SEC EDGAR error |
| 6 | Bad period or other input |

### Output

Each run writes `reports/<TICKER>_<FY>_<id>/`, or `reports/<TICKER>_<FY first>-<FY last>_<id>/` for a multi-year review. There is only ever one final report per company and period: when a run finishes, earlier runs for the same company and the same year (or the same span of years), including failed ones, are deleted.

| File | Content |
|---|---|
| `report.html` | Self-contained report: inline CSS and SVG charts, light and dark themes, works offline, prints cleanly |
| `report.md` | The same content in Markdown |
| `report.json` | Findings, every signal, every detector status and every evidence reference, for other tools |
| `report.pdf` | With `--format pdf` |
| `manifest.json` | What's needed to reproduce the run. It records versions, options, the company profile, detector statuses, peers, the hash of the source data, source documents, LLM spend and timings. |
| `run.pkl.gz` | Everything the report is built from, so `anomaly-report render` can rebuild it without downloads, tests or AI calls. Open only files this tool wrote. |
| `kg.duckdb`, `checkpoints.sqlite` | Working files: the run's knowledge graph (one per year in a multi-year review) and agent checkpoints. Removed after a successful run unless you pass `--keep-artifacts`. Checkpoints are kept when an investigation paused or failed, so `--resume` can continue it. |

`batch` also writes `batch_<time>.csv` and `.md` to the output folder. They rank the companies by risk index and list:
- finding counts (verified and unverified);
- the top finding;
- the report path;
- the time taken;
- any error.

### Reading the report

The HTML and Markdown reports are written for clients, meaning readers with basic accounting and corporate-finance knowledge:
- Tests appear under plain names, with no internal test codes, run IDs, SEC registrant numbers, XBRL concept names or file names.
- Each abbreviation or technical term is spelled out and briefly explained the first time it appears.
- A glossary at the end defines the key concepts.

`report.json` keeps the internal identifiers for other tools.

1. **Summary.** It shows the risk index and band (Low, Elevated, High or Severe), the matters worth following up, and key figures as first reported. The header shows the company type and the currency.
2. **Overview.** It shows warning-sign strength by accounting area, net income vs operating cash flow, scoring models vs peers, and the pressures on management. A chart that doesn't fit the company type is replaced by a short note saying why.
3. **Suspicious Accounting Anomalies.** Every anomaly has the same layout. It opens with three plain-language paragraphs:
   *What we found*, *Why this is unusual* (the benchmark it departs from) and *What it could mean financially*. Then it gives:
   - the AI investigation's conclusion and its estimated effect on profit;
   - whether it is material;
   - what each underlying test found;
   - the supporting evidence, linked to the filings;
   - the independent re-check;
   - innocent explanations to rule out;
   - questions to ask.

   Every test result in a finding says what the test checks, why it matters, how to interpret it, and what this particular result means. That covers severity against the flag threshold and where the company sits among its peers.
4. **All tests performed.** These are grouped by type, and each result is marked:
   - *Flagged*: unusual enough to count.
   - *Normal*.
   - *Explained*: unusual, but a legitimate cause was found.
   - *Not run*: data was missing.
   - *Not applicable*: the test doesn't fit this type of company.

   "About this test" gives the same three explanations for every test.
5. **Disclosure changes, how the review works, glossary.**

The risk index is a weighted combination of flagged results and corroborated patterns. It is **not** a probability.

**A multi-year report** has the same sections, with year-by-year views added:
- **Summary:** the risk index for the period and for each year, and the key figures for every year as first reported.
- **Overview:** warning-sign strength by accounting area *and year*. A dark column is a bad year; a dark row is a recurring problem.
- **Each anomaly:** the years it was flagged in, a strip and table of its strength by year, and a chart over the whole period. The estimated effect, investigation and re-check use the most recent year it was flagged in.
- **All tests performed:** every year's results. The HTML report lists every test for each year; the Markdown report lists each year's flagged and explained results.

---

## How it works

Every formula, threshold, regression and model behind the tests, the scoring and the AI investigation is documented in
[Algorithms.md](Algorithms.md).

```
"Name (TICKER)" ─► resolve ─► company profile ─► fetch (EDGAR, cached) ─► normalize XBRL (point-in-time, reporting currency)
     ─► knowledge graph ─► detector DAG (A–F, profile-aware) ─► false-positive control ─► scheme patterns
     ─► materiality ─► risk index ─► investigation agent (optional) ─► independent audit + correction loop
     ─► HTML / Markdown / JSON report
```

| Step | Module | What it does |
|---|---|---|
| Resolve | [resolve/resolver.py](src/ffi/resolve/resolver.py) | Parses `Name (TICKER)`, looks up the ticker in SEC's list, and checks the name against the SEC title and former names. If they don't match it asks you to choose. It requires at least one 10-K, 20-F or 40-F and counts annual reports across the full filing history, because large banks' recent filings are mostly prospectus supplements. |
| Profile | [profiles.py](src/ffi/profiles.py) | Chooses the detector profile from the SIC code, then adjusts it for 20-F/40-F filing, IFRS and a non-USD currency. |
| Fetch | [edgar/](src/ffi/edgar/) | Sends your User-Agent with every request. One shared rate limiter (≤ 8 requests/s) keeps all threads and batch workers within SEC's limit. Errors are retried with backoff, and responses are kept in a disk cache. Each company's XBRL data is stored only in processed form (refreshed daily for the company, weekly for its peers), and filing text, releases and events download in the background while the industry data is built. |
| Normalize | [xbrl/](src/ffi/xbrl/) | Maps us-gaap, ifrs-full and dei concepts to canonical line items, in priority order. Keeps every reported value with its filing date, and builds annual and quarterly panels both *as first reported* and *latest*. It detects the reporting currency and never mixes currencies. It accepts annual statements filed on 6-K, as many 40-F filers do. |
| Text | [text/](src/ffi/text/) | Splits annual reports into Item sections and notes. 20-F items are mapped onto the 10-K sections. When MD&A and the statements are filed as exhibits, it reads those too: Exhibit 13 for some banks and insurers, and the 40-F exhibits. Each paragraph gets a stable ID, so any quote can be checked verbatim. |
| Peers | [peers/](src/ffi/peers/) | Starts from active filers with the same 4-digit SIC code and widens to 3 and then 2 digits until there are enough. It picks the 15 nearest on size, margins and growth. The industry cross-section is built from XBRL frames and cached per calendar year, as a processed table only. |
| Knowledge graph | [kg/](src/ffi/kg/) | DuckDB storage with NetworkX traversal. It holds companies, filings, facts and their restated versions, line items, sections, quotes, events, the auditor and signals. Every read is filtered by the `--as-of` date. |
| Detect | [pipeline/detect.py](src/ffi/pipeline/detect.py) | Runs the detectors in dependency order. Detectors that don't apply to the profile are marked *n/a*, and a missing input gives a *skipped* status with a reason. A bug in one detector is reported, not fatal. Tests run in parallel; the AI tests get their own pool and start first. |
| Score | [scoring/engine.py](src/ffi/scoring/engine.py) | Applies the explained-away discounts, controls false discoveries (Benjamini–Hochberg) and matches scheme-pattern templates. A pattern needs corroboration from at least 2 detector families. Then it computes the risk index and ranks the findings. |
| Materiality | [materiality/impact.py](src/ffi/materiality/impact.py) | Estimates each finding's impact range and its effect on pre-tax income, net income and EPS. Compares it with auditor-style materiality thresholds and checks the SAB 99 qualitative factors. |
| Agent | [agent/](src/ffi/agent/) | Investigates the top findings with a LangGraph state graph (described below) |
| Audit | [audit/](src/ffi/audit/) | Independent audit of every reported finding, with a correction loop (described below) |
| Explain | [report/explain.py](src/ffi/report/explain.py) | Writes the three plain-language paragraphs for each anomaly. Every number must come from the run's data; standard text is used when no AI is available. |
| Report | [report/](src/ffi/report/) | Jinja2 templates with a fixed layout, inline SVG charts and a neutral-language linter. The run is saved so `anomaly-report render` can rebuild the report. |

### Detector families

Run `anomaly-report detectors` for the full list.

| Family | Covers |
|---|---|
| **A** Integrity | Balance-sheet identity (including redeemable interests), subtotals, cash, equity, retained-earnings, PP&E, allowance and goodwill roll-forwards, quarterly-to-annual consistency, statement tie-outs, EPS, **silent restatements**, MD&A numbers vs XBRL, rounding |
| **B** Forensic scores | Beneish M, Dechow F, discretionary accruals, accrual quality, Sloan, Montier C, Piotroski F, Altman Z, threshold management, real earnings management, income smoothing |
| **C** Account level | Revenue, receivables, inventory, expenses and capitalization, reserves, cash flow, debt and off-balance-sheet items, M&A, non-GAAP measures, governance events (auditor changes, non-reliance filings and annual-report amendments, officer turnover, late filings, control weaknesses, going-concern doubt) |
| **D** Statistical / ML | Seasonal residuals and change points, peer-ratio outliers, a multivariate outlier ensemble, structural expectation models, Benford tests, a supervised misstatement model, disclosure-structure change |
| **E** Text | Accounting policy and estimate changes†, risk-factor changes, MD&A claims vs numbers†, tone, boilerplate, hedging, critical audit matters, SEC comment letters, earnings releases vs filings, missing disclosures, related parties† |
| **F** Graph | Counterparty asymmetry†, circular flows†, auditor profile, evidence-coverage gaps, contradiction density, counterparty stress†, unusual new relationships† |

† Needs the LLM.

### What the LLM does and doesn't do

Python computes every number, flag, score and ranking. The LLM does three things:
1. Extracts structured facts from filing text.
2. Plans and interprets the agent's investigations.
3. Writes finding narratives.

Every LLM output is schema-bound JSON with verbatim quotes. Before a narrative reaches the report, it must pass a **grounding gate**: every number must match a known value, and every quote must exist verbatim in a filing. It must also pass the independent LLM auditor and the language linter.

### Independent audit of key conclusions

Every reported finding is audited independently, with a correction loop of **at most 3 rounds**.

**Deterministic auditor** (every mode). It doesn't trust what detectors say about their evidence. It re-derives everything from the run's sources:

| Check | What is verified |
|---|---|
| `evidence_present` | At least one primary XBRL fact or filing quote supports the conclusion |
| `fact_values` | Every cited value is re-read from the raw XBRL of the same filing and concept |
| `quotes` | Every quote is found verbatim at its position in the filing text |
| `impact` | The impact range is ordered, and materiality recomputed from scratch matches |
| `corroboration` | A pattern's members are still flagged and come from at least 2 families |
| `narrative` | Every number in the narrative traces to a source value |

In each round the auditor checks the finding, applies corrections and checks again. Corrections can remove unverifiable evidence, re-anchor quotes, recompute materiality, downgrade an uncorroborated pattern, or replace an ungrounded narrative.

A finding ends as *verified*, *verified after corrections* or *unverified*. Unverified findings stay in the report at half weight.

**Independent LLM auditor** (inside the agent):
- It sees only the drafted conclusions and a numbered evidence pack built by Python, never the drafter's reasoning.
- It gives each claim a verdict.
- A claim marked "supported" must cite evidence that exists in the pack.
- A failing claim sends the draft back for revision. After 3 failed rounds the case is downgraded to a basic write-up.

### Investigation agent

```
intake → hypothesize → plan_evidence → [review_plan] → gather (parallel per hypothesis)
      → test_alternatives → [review_explained] → quantify → draft → verify (grounding) → audit (LLM auditor)
                                                                     ├→ [review_sensitive] → review_publish → done
          correction loop: verify or audit fails → draft again with the failures as feedback (at most 3 rounds)
                                                                     └→ after the 3rd failed round → degrade → review_publish
   budget exceeded → [review_budget] (raise / narrow / degrade)      retries exhausted → degrade
```

- **Review points** (`[...]` above):
  - In `interactive` mode you're prompted at each one.
  - `--pause-on <name>` stops the run there and saves its state.
  - `--resume <run_id>` continues from the last checkpoint without repeating LLM calls.
- **Budgets.** Each step has limits on input size, output tokens, tool calls, time and dollars. `--budget-usd` caps the whole run (default $6).
- **Retries.** Transient API errors are retried 3 times with backoff. Budget and refusal errors are not retried. When retries run out, the case is downgraded rather than failing the run.
- **Parallelism.** Unattended runs investigate up to 3 findings in parallel. The statistical tests run on 4 threads, and each LLM detector gets a thread of its own. Interactive runs go one finding at a time.
- **Agent tools** are read-only Python functions:
  - `get_fact`, `ratio`, `peer_stats`, `search_filing`;
  - `explaining_text`, `get_events`, `diff_risk_factors`.

---

## Limitations

- **Reporting anomalies only.** The app can't verify cash, assets or transactions.
- **Financial companies** run with the industrial detectors switched off. There are no dedicated bank or insurer models yet, such as loan-loss reserve adequacy, net interest margin outliers or reserve development, so their reports rest mainly on the integrity, governance, disclosure and peer checks.
- **Peer baselines** use the latest values SEC publishes. They can include peers' later restatements. The company's own numbers are always as first reported.
- **Segment-level checks** are skipped, because SEC's companyfacts API carries no dimensional facts.
- **Earnings-call checks and company-specific operating metrics** are skipped. They need licensed transcripts or company-specific extraction.
- **The supervised misstatement model is a ranking signal, not a probability.** It is trained on weak labels: 8-K Item 4.02 non-reliance filings.
- **Market value** is approximated by the public float on the 10-K cover page.

## Project layout

```
src/ffi/
  _env_guard.py  cli.py  config.py  context.py  profiles.py  llm.py  models_store.py  concurrency.py
  edgar/        client (rate limit, retries, cache), endpoints
  resolve/      ticker/name -> CIK, eligibility
  xbrl/         canonical items (us-gaap, ifrs-full, dei), point-in-time fact store
  text/         annual-report sectioning (10-K, 20-F, 40-F, Exhibit 13), search, quote verification
  peers/        SIC universe, frames cross-section, peer selection
  kg/           DuckDB + NetworkX store, graph builder
  detectors/    base, helpers, stats, scores, textnum, xsec_models, catalog (plain-language explanations),
                a_integrity, b_scores, c_accounts, d_stats, e_text, f_graph
  pipeline/     detect (DAG runner), run (orchestration), span (multi-year review)
  scoring/      false-positive control, scheme templates, risk index, span (merging findings across years)
  materiality/  impact and materiality
  agent/        schemas, tools, budget, grounding, graph, runner, ui
  audit/        deterministic auditor + correction loop, independent LLM auditor
  report/       model, span_model, plain (client wording), explain, charts, linter, render, snapshot (saved runs), templates/
  training/     labels (8-K 4.02), train / backtest
  models/       trained model for the supervised detector
```

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

Reports produced by this tool are based on public SEC filings and are decision support, not investment, legal or audit advice.
