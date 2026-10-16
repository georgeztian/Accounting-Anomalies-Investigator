# Forensic Filings Investigator (`ffi`)

Give it a company. It reads the company's SEC filings and writes an **accounting-anomaly report**. Every number in the report links back to the filing it came from.

```powershell
anomaly-report "Super Micro Computer (SMCI)"
```

The report contains:

- A 0–100 risk index.
- Ranked findings. Each one shows its dollar impact and materiality, the evidence (XBRL facts and verbatim filing quotes, each linked to EDGAR), the innocent explanations that were tested, and questions for management.
- An **independent audit** of every finding: its evidence is re-derived from the source filings, and any problems go through a correction loop of at most 3 rounds.
- Results for all 117 detectors, including the reason for each one that was skipped.
- A data-quality section and a methodology section.

The design is in [BLUEPRINT.md](BLUEPRINT.md) (rev. 3). This README records what is built, how to run it, and where the build differs from the blueprint.

> The app flags departures from expectation for further inquiry. It does not allege wrongdoing. It is decision support for qualified analysts, not investment, legal or audit advice.

---

## Status — 2026-10-15

| Blueprint phase | Status | Notes |
|---|---|---|
| 0 Skeleton: package, venv guard, CLI, EDGAR client, resolution, XBRL facts, Family A | **Done** | |
| 1 Deterministic report: knowledge graph, peers, cross-section, Families B and C, materiality, scheme patterns, HTML/MD/JSON report | **Done** | |
| 2 Text, ML, graph: Families D, E and F, offline training | **Done**, with gaps listed below | The LLM detectors in E/F have not been run against the live API (no API key was available while building). The supervised model (D.06) is trained and shipped, but its backtest ROC-AUC is 0.65, below the blueprint's 0.75 gate (see below). |
| 3 Agent: LangGraph state graph, SQLite checkpoints, interrupts, budgets, retries, resume/pause, independent auditor | **Done** | Tested end to end with a scripted fake LLM. Not yet run against the live Claude API. |
| 4 Extensions: banks/insurers, market-wide screener, Neo4j backend | Not started | Optional PDF output is done. |

Tests: 48 unit tests and 5 live-EDGAR integration tests, all passing. `ruff` is clean. `mypy` still reports 33 type-narrowing errors (Optional values that are guarded at runtime); none is a runtime bug.

The 2026-10-15 inspection is summarised under [Inspection log](#inspection-log-2026-10-15).

---

## Requirements

- Python **3.11 or 3.12**, run inside a **virtual environment**. The app exits with code 2 otherwise (see [src/ffi/_env_guard.py](src/ffi/_env_guard.py)).
- Internet access to `sec.gov`. After the first run, `--offline` works from the cache.
- Optional: an Anthropic API key for the LLM detectors and the investigation agent.

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

Without the lock file: `pip install -e ".[llm,pdf,dev]"`.

Edit `.env`:

```ini
SEC_USER_AGENT="Your Name your.email@example.com"   # required: SEC asks every client to identify itself
ANTHROPIC_API_KEY=sk-ant-...                        # optional: enables Family E/F LLM detectors and the agent
# FFI_CACHE_DIR=.cache
# FFI_REASONING_MODEL=claude-opus-5-5
# FFI_EXTRACTION_MODEL=claude-haiku-5-5
```

Optional tuning (peer count, materiality thresholds, false-discovery rate, agent budget): copy `ffi.toml.example` to `ffi.toml` and edit it.

## Usage

```powershell
anomaly-report "Apple Inc. (AAPL)"                 # preferred input form
anomaly-report AAPL                                # a ticker works
anomaly-report "Microsoft Corporation"             # a name works (asks you to choose if ambiguous)

anomaly-report SMCI --mode deterministic           # no LLM at all
anomaly-report SMCI --period FY2024                # a specific fiscal year
anomaly-report AAPL --as-of 2020-06-30             # point in time: only filings public on that date
anomaly-report AAPL --peers MSFT,GOOGL,DELL        # override automatic peer selection
anomaly-report SMCI --format html,md,json,pdf      # pdf needs the [pdf] extra
anomaly-report SMCI --mode auto --budget-usd 3     # unattended agent with a $3 cap
anomaly-report SMCI --pause-on review_publish      # stop at the publish review and save state
anomaly-report SMCI --resume SMCI_FY2026_ab12cd    # continue a paused or interrupted run

anomaly-report batch companies.txt --workers 3     # many companies, one ranked summary (CSV + Markdown)
anomaly-report detectors                           # list the detector catalogue
anomaly-report labels --start 2016 --end 2022      # starter labels from 8-K Item 4.02 filings
anomaly-report train --labels data/labels_4_02.csv --backtest   # walk-forward evaluation
anomaly-report train --labels data/labels_4_02.csv              # fit and save the D.06 model
anomaly-report cache-info
python -m ffi AAPL                                 # same as anomaly-report
```

**Modes**

- `interactive` (default in a terminal): you're prompted at every agent interrupt.
- `auto`: fixed policies answer the interrupts, and each answer is labelled *auto* in the report.
- `deterministic`: no LLM.

Without `ANTHROPIC_API_KEY` the app runs in deterministic mode and says so in the report.

**Exit codes**

| Code | Meaning |
|---|---|
| 2 | Not running in a virtual environment |
| 3 | Company could not be resolved |
| 4 | Company not supported (financial company, 20-F filer, or fewer than 2 10-Ks) |
| 5 | SEC EDGAR error |
| 6 | Bad period or other input |

### Output

Each run writes `reports/<TICKER>_<FY>_<id>/`:

| File | Content |
|---|---|
| `report.html` | Self-contained report: inline CSS and SVG charts, light/dark themes, works offline, prints cleanly |
| `report.md` | The same content in Markdown |
| `report.json` | Findings, every signal, every detector status and every evidence reference, for other tools |
| `report.pdf` | With `--format pdf` |
| `manifest.json` | What's needed to reproduce the run: versions, options, detector statuses, peers, the hash of the source companyfacts, source documents, LLM spend, timings |
| `kg.duckdb` | The run's knowledge graph |
| `batch_<time>.csv/.md` | Written by `batch`, in the output folder: companies ranked by risk index, with finding counts (verified / unverified), top finding, report path, time taken and any error |
| `checkpoints.sqlite` | Agent checkpoints (LLM runs only) |

### Measured performance (this machine)

| Run | Time |
|---|---|
| First run with an empty cache (Microsoft, deterministic) | 50 s (was 54 s before concurrent fetching; the industry cross-section and peers step went from 43 s to 32 s, and SEC's rate limit is the floor) |
| Cached re-run | about 11–20 s |
| `batch` of 8 companies, 3 workers, warm cache | 1 min 54 s, against about 5 min if run one after another |

`batch` on 8 companies (deterministic, 2026-10-15). Every finding passed the independent audit:

| Company | Period | Risk index | Findings | Top finding |
|---|---|---:|---:|---|
| Super Micro Computer (SMCI) | FY2026 | 79 Severe | 6 | Earnings are running well ahead of operating cash flow |
| Salesforce (CRM) | FY2026 | 49 Elevated | 3 | Headline metrics may present a better picture than GAAP results |
| Pfizer (PFE) | FY2025 | 46 Elevated | 2 | Margin up while inventory builds |
| Under Armour (UAA) | FY2026 | 42 Elevated | 2 | Reported figures are internally inconsistent or were revised |
| Costco (COST) | FY2026 | 33 Elevated | 2 | Working-capital-driven CFO |
| Walmart (WMT) | FY2026 | 5 Low | 0 | |
| Apple (AAPL) | FY2025 | 5 Low | 0 | |
| Exxon Mobil (XOM) | FY2025 | 0 Low | 0 | |

---

## How it works

```
"Name (TICKER)" ─► resolve ─► fetch (EDGAR, cached) ─► normalize XBRL (point-in-time) ─► knowledge graph
     ─► detector DAG (A–F) ─► false-positive control ─► scheme patterns ─► materiality ─► risk index
     ─► investigation agent (optional, with independent LLM auditor) ─► independent audit + correction loop
     ─► HTML / Markdown / JSON report
```

| Step | Module | What it does |
|---|---|---|
| Resolve | [resolve/resolver.py](src/ffi/resolve/resolver.py) | Parses `Name (TICKER)`. Looks up the ticker in SEC's ticker list and checks the name against the SEC title and former names (rapidfuzz). Below a score of 60 it asks you to choose. If the ticker now belongs to a registrant with no 10-K history (for example a new holding company, as with XOM), it searches SEC by company name for the 10-K filer and says so in the report. Rejects banks, insurers and REITs (SIC 6000–6799), 20-F filers, and companies with fewer than 2 10-Ks. |
| Fetch | [edgar/](src/ffi/edgar/) | Requires a User-Agent. One rate limiter (≤ 8 requests/s) is shared by every client and thread in the process, so concurrent fetching (frames, peers, 10-Ks, releases, 8-Ks) and `batch` workers stay within SEC's limit. Retries 429/5xx with backoff and keeps a content-addressed disk cache. Fetches companyfacts, submissions, 10-K HTML, 8-K Ex. 99 earnings releases, comment letters, XBRL frames, and the company list by SIC code. |
| Normalize | [xbrl/](src/ffi/xbrl/) | Maps 107 canonical line items from 195 us-gaap/dei concepts, in priority order. Keeps every reported value with its filing date. Builds annual and quarterly panels, both *as first reported* and *latest*. Derives Q4 as FY minus Q1–Q3. Vectorized: about 0.7 s per company. |
| Text | [text/](src/ffi/text/) | Converts 10-K HTML to normalized text. Splits it into Item sections and notes, skipping the table of contents; long headers such as Costco's are accepted when they carry the Item's title. Each paragraph gets an ID `accession#section#i#start-end`. A quote verifier tolerates differences in whitespace and quote characters. |
| Peers | [peers/](src/ffi/peers/) | Takes active filers in the same industry, widening from 4-digit SIC to 3 and 2 digits until at least 30 have XBRL data. Picks the 15 nearest on size, margins and growth. The industry cross-section (about 6,000 filers × 4 years) is built from XBRL frames and cached as Parquet. |
| Knowledge graph | [kg/](src/ffi/kg/) | DuckDB storage plus NetworkX traversal, bitemporal, and every read filtered by `as_of`. Holds Company, Filing, Fact (and restated versions linked by `RESTATES`), LineItem, Section, Span, Note, Event, Auditor, peer Company, Entity and Signal nodes. Rule-based `EXPLAINS` edges link MD&A and notes to line items. |
| Detect | [detectors/](src/ffi/detectors/), [pipeline/detect.py](src/ffi/pipeline/detect.py) | 117 detectors run in dependency order. With the LLM on, independent detectors run on 4 threads; deterministic runs stay sequential. A bug in one detector is reported, not fatal. Skips always carry a reason. Every signal that raises any concern carries evidence: its own, or by default the facts it was computed from or the filing section it analysed. |
| Score | [scoring/engine.py](src/ffi/scoring/engine.py) | Calibration maps (if trained), explained-away discounts (acquisition, standard adoption, split), Benjamini–Hochberg at q = 0.10, then 12 scheme-pattern templates. A pattern needs at least 2 signals from at least 2 families (or at least 3 signals). Then the risk index and finding priority. |
| Audit | [audit/](src/ffi/audit/) | Independent audit of every reported finding, with a correction loop of at most 3 rounds (below) |
| Materiality | [materiality/impact.py](src/ffi/materiality/impact.py) | Dollar impact range, effects on pre-tax income, net income and EPS, ratios to pre-tax income, revenue, assets and equity, multiples of auditor-style thresholds, and SAB 99 qualitative flags. |
| Agent | [agent/](src/ffi/agent/) | LangGraph state graph (below) |
| Report | [report/](src/ffi/report/) | Jinja2 templates, hand-built SVG charts (reference palette, light/dark, tooltips on every mark), and a neutral-language linter |

### Detector catalogue

117 detectors in total; run `anomaly-report detectors` for the live list.

| Family | Count | Covers |
|---|---|---|
| **A** Integrity | 16 | Balance-sheet identity, subtotals, cash / equity / retained-earnings / PP&E / allowance / goodwill roll-forwards, quarterly-to-annual consistency, statement tie-outs, EPS, current vs cash taxes, **silent restatements** (first-reported vs re-reported values), MD&A numbers vs XBRL, rounding |
| **B** Forensic scores | 10 | Beneish M (8-variable), Dechow F (model 1), Kothari discretionary accruals (industry-year regression on frames), Dechow–Dichev accrual quality (pooled peer regression), Sloan ratio, Montier C, Piotroski F, Altman Z/Z'', threshold management, Roychowdhury real earnings management, Leuz smoothing |
| **C** Account level | 62 | Revenue (11), receivables (5), inventory (5), expenses and capitalization (9), reserves (6), cash flow (7), debt and off-balance-sheet (4), M&A (5), non-GAAP (4), governance events (6) |
| **D** Statistical / ML | 9 | STL residuals and PELT change points, peer-relative ratio outliers, an MCD + Isolation Forest + ECOD ensemble, LightGBM structural models with split-conformal intervals, Benford/Nigrini, supervised misstatement model, restatement-delta model, disclosure-structure change |
| **E** Text | 13 | Policy changes†, critical estimates†, risk-factor diff, MD&A claims vs numbers†, tone (Loughran–McDonald-style lists), year-over-year similarity, hedging, critical audit matters, SEC comment letters, earnings release vs filing, missing expected disclosures, related parties and side arrangements† |
| **F** Graph | 7 | Counterparty asymmetry†, circular flows†, auditor profile, evidence-coverage gaps (material moves with no `EXPLAINS` edge), contradiction density, counterparty stress†, new related-party/VIE edges† |

† Needs the LLM. Every LLM output is schema-bound JSON with verbatim quotes. A quote that can't be found in the filing is discarded, and any numbers it contains are re-checked against XBRL. (The C-family count includes C.BS.03, C.NG.03 and C.NG.04, which also need the LLM.)

### What the LLM does and doesn't do

Python computes every number, every flag, every score and every ranking. The LLM does three things:

1. Extracts structured facts from text (E.01, E.02, E.04, E.09, C.BS.03, C.NG.03).
2. Plans and interprets the agent's investigations.
3. Writes finding narratives.

Before a narrative reaches the report it must pass a **grounding gate** ([agent/grounding.py](src/ffi/agent/grounding.py)): every number must match a known value, and every quote must exist verbatim in a filing. Dates, detector IDs and fiscal labels are ignored. It must also pass the **independent LLM auditor** and the language linter. In deterministic mode the narratives are built from templates.

### Independent audit of key conclusions

Every reported finding is audited independently, with a correction loop of **at most 3 rounds**.

**1. Deterministic auditor** ([audit/deterministic.py](src/ffi/audit/deterministic.py)). Runs in every mode after scoring and the agent. It doesn't trust what the detectors say about their evidence; it re-derives the evidence from the run's sources:

| Check | What is verified independently |
|---|---|
| `evidence_present` | At least one primary XBRL fact or filing quote supports the conclusion |
| `fact_values` | Every cited value is re-read from the raw companyfacts rows of the same filing and concept |
| `quotes` | Every quote is found verbatim at its character span in the filing text |
| `impact` | low ≤ mid ≤ high, and materiality recomputed from scratch matches |
| `corroboration` | A pattern's members are still flagged, unsuppressed, and from ≥ 2 families (or ≥ 3 signals) |
| `narrative` | Every number in the narrative traces to a source value |

Each round: audit, apply corrections, audit again. Corrections can remove unverifiable facts, re-anchor or remove quotes, attach member evidence, re-order the impact range and recompute materiality, downgrade an uncorroborated pattern to a single-source signal, or replace an ungrounded narrative with one built from verified outputs.

Each finding ends as *verified*, *verified after N corrections*, or *unverified after 3 rounds*. Unverified findings keep their place in the report but carry half weight in the risk index and the ranking. The report shows every round's checks and corrections, and `manifest.json` lists each finding's audit result.

**2. Independent LLM auditor** ([audit/llm_auditor.py](src/ffi/audit/llm_auditor.py)), inside the agent:
- It has its own instructions.
- It sees **only** the drafted conclusions and a numbered evidence pack built by Python (facts, quotes, detector outputs, materiality, tool results). It never sees the drafter's hypotheses, plan or earlier feedback.
- It splits the conclusions into claims and gives each one a verdict: supported, overstated, needs context, or unsupported.
- A claim marked supported must cite evidence IDs that exist in the pack, or it counts as unsupported.
- Any failing claim sends the draft back with the auditor's corrections. Three failed rounds degrade the case.

### Investigation agent

[agent/graph.py](src/ffi/agent/graph.py) defines the graph:

```
intake → hypothesize → plan_evidence → [review_plan] → gather (one parallel branch per hypothesis)
      → test_alternatives → [review_explained] → quantify → draft → verify (grounding) → audit (independent LLM auditor)
                                                                     ├→ [review_sensitive] → review_publish → done
          correction loop: verify or audit fails → draft with the failures as feedback (at most 3 rounds)
                                                                     └→ after the 3rd failed round → degrade → review_publish
   budget exceeded → [review_budget] (raise / narrow / degrade)      retries exhausted → degrade
```

- **Parallel cases.** Unattended runs (`--mode auto`) investigate up to 3 findings in parallel; interactive runs go one at a time so prompts never interleave.
- **Checkpoints.** `SqliteSaver` writes `checkpoints.sqlite`. The thread ID is `run_id:finding_id`. `--resume <run_id>` continues every thread from its last checkpoint without repeating LLM calls. LLM responses are also cached on disk, keyed by a request hash.
- **Interrupts.** `review_plan` fires for high severity, material impact or low confidence. `review_explained` fires when an innocent explanation is strongly supported. The others are `review_budget`, `review_publish` (always) and `review_sensitive` (linter or auditor flagged wording). In a terminal you're prompted, and edits open in `$EDITOR`. `--pause-on` stops the process at a chosen interrupt and saves state.
- **Retries.** LLM nodes get 3 attempts with backoff from 2 s to 30 s, and only for transient errors. Budget and refusal errors are never retried. When retries run out, an error handler routes the case to `degrade`.
- **Budgets.** Each node has limits on input characters, output tokens, tool calls, wall time (enforced as the API request timeout) and dollars. There is also a run cap (`--budget-usd`, default $6).
- **Refusals.** Requests use server-side refusal fallback (`fallbacks="default"`, beta `server-side-fallback-2026-07-01`). If you'd rather not, remove `fallbacks` and `betas` in [llm.py](src/ffi/llm.py).
- **Agent tools** are read-only and computed in Python: `get_fact`, `ratio`, `peer_stats`, `search_filing`, `explaining_text`, `get_events`, `diff_risk_factors`.

---

## Tests

```powershell
python -m pytest -m "not network"   # 48 unit tests, no network
python -m pytest                    # adds 5 integration tests against live EDGAR
python -m ruff check src tests
```

- **Detectors:** synthetic companies built as companyfacts JSON, where each detector fires on the case built to trigger it and stays quiet on a clean company. These cover the identity break, cash roll-forward, silent restatement, DSO jump, inventory build-up, CFO shortfall, useful-life extension and Beneish response.
- **Guards:** the venv guard exits with code 2; `as_of` hides later filings; the knowledge graph filters by `as_of`; the grounding gate rejects invented numbers and quotes; the linter keeps the disclaimer.
- **Scoring:** a pattern needs corroboration; false-discovery control suppresses weak signals.
- **Audit:** clean findings verify in round 1; a tampered fact is removed and re-verified; an invented narrative number is replaced; an uncorroborated pattern is downgraded; an uncorrectable finding ends unverified within 3 rounds and loses weight.
- **Parallelism:** a parallel detector run matches the sequential one exactly; unattended agent cases run in parallel and all complete.
- **Agent, with a scripted fake LLM:** the happy path publishes, and the auditor never sees the drafter's hypotheses; invented numbers fail grounding for 3 rounds and then degrade; an auditor that keeps rejecting causes 3 drafts, 3 audits and then degrade; budget exhaustion goes through `review_budget` to `degrade`; three failed attempts degrade via the error handler; a supported alternative closes the finding; pause and resume continue without repeating LLM calls.
- **Report:** rendering a synthetic company produces valid HTML, Markdown and JSON with no unrendered template tags or NaN values.
- **Integration:** a full Apple run produces all reports with 0 errors, is self-contained, uses neutral language, links every fact to an EDGAR document, and works point-in-time as of 2020.

CI runs lint and unit tests on Windows, macOS and Linux in a fresh venv from the lock file: [.github/workflows/ci.yml](.github/workflows/ci.yml).

---

## Inspection log (2026-10-15)

A sweep over 8 companies (different fiscal calendars, industries and sizes), plus `mypy` and a code review, found these problems. All are fixed and covered by the test suite or the batch rerun above.

| Area | Problem | Fix |
|---|---|---|
| Resolution | `XOM` resolved to a new holding company with no 10-Ks | Search SEC by company name for the operating predecessor and warn in the report; keep the input ticker |
| Text | Costco's Item 7 header (177 characters) was rejected, so MD&A was parsed from the table of contents | Accept long headers that carry the Item title; F.04 skips when MD&A isn't located |
| A.15 / E.12 | Segment, product and quarterly amounts were compared with company totals (Under Armour: 33 false mismatches) | Only company-wide totals; skip segment, guidance and non-GAAP sentences; a mismatch must be within 25% of a total |
| F.05 | Re-counted A.15 / E.12 items and corroborated the patterns it was derived from | Counts flagged detectors (0–3); tagged *derived*; excluded from pattern templates and the index |
| D.04 | Tree models can't extrapolate size (Salesforce receivables expected at $0.8B vs $14.3B) | Predicts scale-free ratios (AR/revenue, inventory/COGS, SG&A/revenue) from margin, turnover and size |
| C.AR.01 / C.INV.01 / C.CF.03 | Flagged trivial moves (Costco DSO 4 → 5 days) | Minimum absolute change (5–7 days) |
| C.CF.04 | Ratio computed against negative CFO | Skips unless CFO > 0 in both years |
| A.05 | Pfizer tags dividends only as *declared* | New `DIVIDENDS_DECLARED` item, used first |
| C.EXP.07 | Recurring restructuring reached full severity on its own | Capped at 0.6 |
| Risk index | Pattern members were counted again on their own | Each signal counts once, through its pattern; derived signals excluded |
| Evidence | About 20 detectors could raise concerns without citing anything | Default evidence: input facts or the analysed filing section; C.NG.02 quotes the release |
| Grounding | Ranges ("$0-$174.5M"), ISO dates and detector IDs ("A.14") were read as numbers | Ranges use absolute values; dates and IDs are stripped before tracing |
| Config | `ffi.toml` materiality thresholds, `flag_severity`, agent budget and `max_cases` were ignored | All wired in; CLI defaults come from `ffi.toml` |
| Terminal UI | Rich markup swallowed the `[a]pprove` keys in prompts | Prompts escaped |
| Batch output | Printing "✓" to a non-UTF-8 Windows console discarded results | ASCII markers; a console error can never lose a result |
| Code | Duplicate A.15 / E.12 / D.04 definitions after patching; `href=None` crash path; shadowed variables; dead code; manifest "version" held a module name; KG `ABOUT` edges matched by label prefix; `.env` read only from the project root | Removed or fixed; detector versions in the manifest; `ABOUT` via concept mapping; `.env` also read from the working directory |
| Efficiency | Sequential fetching; per-client rate limiters; sequential LLM detectors and agent cases; training features fetched one at a time (about 70 min) | Shared limiter, concurrent fetching, parallel LLM detectors and unattended agent cases, concurrent training features, `batch` command |

## Where the build differs from the blueprint

| Blueprint | Built | Why |
|---|---|---|
| Arelle parses calculation linkbases (A.02) | A.02 checks canonical subtotals (gross profit, net PP&E, current assets) | companyfacts has no linkbases; Arelle is a large dependency for one detector |
| About 250 canonical items | 107 items | Covers every detector's inputs; add more in [canonical.py](src/ffi/xbrl/canonical.py) |
| Segment detectors A.11, C.REV.06 | Skip with a reason | companyfacts carries only non-dimensional facts |
| C.REV.09 KPIs, E.11 earnings calls, D.09 ego-network | Skip with a reason | Need company-specific KPI extraction, licensed transcripts, or market-wide peer graphs (Phase 4) |
| D.06 supervised model, AUC ≥ 0.75 gate | Trained and shipped at mean AUC 0.65 | Weak 8-K 4.02 labels and ratio-only features (see Known limitations) |
| D.07 restatement-delta model | Skips until a model is trained | Training code for it is not written yet |
| Calibrated P(misstatement) | A 0–100 risk index, labelled as *not* a probability | A calibrated probability needs a trained meta-model and isotonic maps, which `train` doesn't produce yet |
| Market cap for C.MA.02 / valuation | Public float from the 10-K cover page as a proxy | No market-data dependency; it understates market value, and the report says so |
| Dagster | A small in-process runner with HTTP and Parquet caches | Too heavy for a single-company CLI (blueprint rev. 3 §3) |
| matplotlib charts | Hand-built inline SVG | Light/dark theming via CSS, tooltips, no image files |
| Wall-time limit per node via LangGraph `timeout` | API request timeout per node | LangGraph only supports node timeouts on async nodes |
| E.05 Loughran–McDonald dictionaries | Built-in subsets of the uncertainty, weak-modal and negative lists | The full lists are a separate download |

## Known limitations

- **The LLM paths haven't run against the live API.** The agent, the independent LLM auditor and every LLM detector were tested with a scripted fake LLM; set `ANTHROPIC_API_KEY` to exercise them for real. In live use, check the API spend line in the run summary and the `llm` block in `manifest.json`.
- **Model ID check.** On first live use, confirm that `claude-opus-5-5` / `claude-haiku-5-5` accept `fallbacks="default"`. If one rejects it, drop the parameter in [llm.py](src/ffi/llm.py).
- **D.06 is weak and misses the blueprint's gate.** It is trained on 8-K Item 4.02 filings (fiscal years 2015–2021), with the restated year approximated as the year before the filing, against random same-year controls: 961 firm-years, 242 positives. Walk-forward backtest, training on years at least 3 years before each test year:

  | Test year | ROC-AUC | PR-AUC | Recall in top 10% | Firm-years (positives) |
  |---|---:|---:|---:|---|
  | 2018 | 0.678 | 0.447 | 0.18 | 142 (38) |
  | 2019 | 0.629 | 0.485 | 0.19 | 121 (36) |
  | 2020 | 0.577 | 0.151 | 0.06 | 126 (16) |
  | 2021 | 0.720 | 0.490 | 0.23 | 151 (39) |
  | **Mean** | **0.651** | | **0.17** | |

  This is better than chance but short of the blueprint's gate (AUC 0.75). D.06 is therefore weighted like any other detector and treated as a ranking signal. Stronger labels (AAERs) and the full detector outputs as features are the next steps. Reproduce with `anomaly-report labels` then `anomaly-report train --labels data/labels_4_02.csv --backtest`; the raw log is in `data/backtest.log`.
- **Frames carry latest values.** Peer and cross-section baselines can include peers' later restatements; the company's own numbers are always as first reported.
- **XBRL concept switches** between years can make ratios jump. C.BS.02 detects and discounts this; other detectors may still be affected.
- **Not covered:** banks, insurers, REITs and foreign private issuers.
- **Scope of what it can find:** reporting anomalies only. It can't verify cash, assets or transactions.

## Project layout

```
src/ffi/
  _env_guard.py  cli.py  config.py  context.py  llm.py  models_store.py  concurrency.py
  edgar/        client (rate limit, retries, cache), endpoints
  resolve/      ticker/name -> CIK
  xbrl/         canonical items, point-in-time fact store
  text/         10-K sectioning, search, quote verification
  peers/        SIC universe, frames cross-section, peer selection
  kg/           DuckDB + NetworkX store, graph builder
  detectors/    base, helpers, stats, scores, xsec models, a_integrity, b_scores, c_accounts, d_stats, e_text, f_graph
  pipeline/     detect (DAG runner), run (orchestration)
  scoring/      calibration, false-positive control, scheme templates, risk index
  materiality/  impact and materiality
  agent/        schemas, tools, budget, grounding, graph, runner, ui
  audit/        deterministic auditor + correction loop, independent LLM auditor
  report/       model, charts, linter, render, templates/
  training/     labels (8-K 4.02), train / backtest
  models/       trained artifacts: misstatement.lgb.txt + misstatement.json (D.06)
tests/          unit/ (no network), integration/ (live EDGAR)
```
