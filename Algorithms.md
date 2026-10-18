# Algorithms

This document explains how the anomaly investigation works and why each method is expected to work. For every
algorithm it gives:

- **What it is**: where the method comes from and what it measures.
- **Why it works**: the accounting and economic reasoning that connects the measurement to a misstatement risk.
- **How it is computed**: the exact formulas, thresholds and severity rules, as implemented. The module for each
  part is named so it can be checked against the source code.

Formulas are written in plain text so the file reads the same in any editor, on GitHub and in a PDF.

---

## Notation

- `X_t` is line item X for the fiscal year under review (the *focus year*); `X_t-1` is the prior year, and so on.
- A `d` prefix is the change from the prior year: `dAR = AR_t - AR_t-1`.
- `avg(X) = (X_t + X_t-1) / 2` is a two-year average.
- `sum3(X)` is the sum of X over the last three years.
- `clip(x, a, b)` limits x to the interval from a to b. Written `clip(x)` it means `clip(x, 0, 1)`.
- `|x|` is the absolute value, `ln` the natural logarithm, `exp(x)` is e to the power x, `×` is multiplication.
- `b0, b1, ...` are regression coefficients; `e` is a regression residual (actual minus predicted).
- `y_hat` is a predicted value.
- All amounts are in the company's reporting currency, *as first reported* unless stated otherwise.

Abbreviations: AP payables; AR receivables; CA / CL current assets / current liabilities; CFO / CFI / CFF cash from
operating / investing / financing activities; COGS cost of sales; DA depreciation and amortization; GM gross margin;
GP gross profit; Inv inventory; LTD long-term debt; NI net income; OI operating income; PP&E property, plant and
equipment; PT pre-tax income; RE retained earnings; Rev revenue; SGA selling, general and administrative expenses;
STI short-term investments; TA / TL total assets / total liabilities.

---

## Contents

1. [The overall design](#1-the-overall-design)
2. [Data preparation](#2-data-preparation)
3. [Shared statistics](#3-shared-statistics)
4. [Family A: accounting integrity](#4-family-a-accounting-integrity)
5. [Family B: forensic scores and accrual models](#5-family-b-forensic-scores-and-accrual-models)
6. [Family C: account-by-account tests](#6-family-c-account-by-account-tests)
7. [Family D: statistical and machine-learning tests](#7-family-d-statistical-and-machine-learning-tests)
8. [Family E: text and disclosure tests](#8-family-e-text-and-disclosure-tests)
9. [Family F: relationship-graph tests](#9-family-f-relationship-graph-tests)
10. [From test results to findings and the risk index](#10-from-test-results-to-findings-and-the-risk-index)
11. [Materiality](#11-materiality)
12. [The AI investigation](#12-the-ai-investigation)
13. [The grounding gate](#13-the-grounding-gate)
14. [The independent audit](#14-the-independent-audit)
15. [Plain-language explanations](#15-plain-language-explanations)
16. [Multi-year review](#16-multi-year-review)
17. [Known approximations](#17-known-approximations)
18. [References](#references)

---

## 1. The overall design

### 1.1 The economic premise

Accounting runs on accruals: revenue is recorded when it is earned rather than when cash arrives, and costs when
they are incurred rather than when they are paid. Accruals make earnings a better measure of performance than cash
flow, but they rest on management's estimates, and estimates can be bent. Three facts make misstatement detectable
from public filings:

1. **Accruals must reverse.** Every accrual eventually turns into cash or is written off. Profit recorded ahead of
   cash leaves a trace on the balance sheet (receivables, inventory, capitalized costs, low reserves) that grows
   until it reverses. Measuring how far earnings run ahead of cash, and where the difference sits, is the core of
   Families B and C.
2. **Double entry links everything.** A number changed in one place must be balanced somewhere else. Statements that
   do not tie together (Family A), or narrative that contradicts the numbers (Families A and E), signal late or
   inconsistent changes.
3. **Companies resemble their peers and their own past.** Businesses in the same industry face similar economics, so
   their ratios are similar, and a company's ratios move slowly unless its business changes. A sharp departure from
   both baselines is either a business change that should be explained, or an accounting choice (Families C and D).

No single test is conclusive: each has innocent causes. The system therefore looks for **corroboration**, several
independent tests pointing to the same scheme, and discounts results that a known event explains (section 10).

### 1.2 The pipeline

```
resolve company -> company profile -> fetch SEC data -> normalize XBRL (point in time)
  -> industry cross-section + peer set -> filing text -> knowledge graph
  -> ~117 tests (families A-F) -> calibration, explained-away discounts, false-discovery control
  -> scheme patterns -> findings -> materiality -> risk index
  -> AI investigation (top findings) -> independent audit + correction loop -> plain-language explanations -> report
```

Every test is a pure function of the run's data and returns zero or more **signals**:

| Field | Meaning |
|---|---|
| `raw_value`, `expected` | The measured value and the benchmark it was compared with |
| `severity` (0 to 1) | How unusual the result is: 0 = nothing unusual, 1 = strongest |
| `flagged` | `severity ≥ flag_at` (default 0.5; a few context tests use 0.45) |
| `percentile_peer`, `percentile_self`, `p_value` | Position against peers and the company's own history, where computed |
| `impact` | An estimated dollar effect (low, mid, high), where the test can size one |
| `area`, `scheme`, `direction`, `tags` | Used to group signals into scheme patterns |
| `evidence` | References to the exact XBRL facts, filing quotes or events used |

A test that lacks inputs is **skipped** with a reason; one that does not fit the company's type is **n/a**
(section 2.6); one that crashes is reported as **error** and the run continues (`pipeline/detect.py`). Tests run in
dependency order (e.g. F.01 needs E.09) in two thread pools, one for CPU-bound statistical tests and one for AI-backed
tests.

### 1.3 Why severity rather than pass/fail

Every test maps its statistic to a severity between 0 and 1, usually with a linear ramp: 0 below the level where
concern starts, 1 at the level where the result is as strong as that test can be. A common scale lets results from
very different tests (a ratio, a regression residual, a word count) be combined, and keeps "slightly unusual"
distinct from "extreme". The ramps encode judgment about where concern begins; section 10.1 describes how they can be
replaced by calibrated mappings.

---

## 2. Data preparation

### 2.1 Canonical line items (`xbrl/canonical.py`)

**What it is.** Companies tag the same item with different XBRL concepts (for example `Revenues`,
`RevenueFromContractWithCustomerExcludingAssessedTax` or `SalesRevenueNet` for revenue). Each of about 110 canonical
items has a priority-ordered list of us-gaap concepts, followed by ifrs-full concepts for foreign filers.

**Why it matters.** Every test compares the same item across years and across companies. A ratio built from
different concepts in different years would show changes that are only tagging changes.

**How.** For each item and period the highest-priority concept that is reported wins. For depreciation and
amortization (`DA`) the largest candidate value in the same filing wins instead, because companies often tag only a
component (such as depreciation alone) with a concept that ranks higher. Totals rank before components: for revenue, total revenue (`Revenues`, then
`RevenuesNetOfInterestExpense`) comes before ASC 606 contract revenue, which at banks and card issuers leaves out net
interest income. Monetary items accept only facts in the
**reporting currency**, chosen as the unit used in the most annual-report filings, so amounts are never mixed across
currencies.

**Same basis.** Some fallback concepts measure a narrower or broader amount than the item's main concepts: general and
administrative expense for SG&A, net income including noncontrolling interests, cash including restricted cash, PP&E
including finance-lease assets, pre-tax income before the share of equity-method income, long-term debt including its
current maturities, current debt including short-term borrowings (`DIFFERENT_BASIS` in `canonical.py`). Two values
of an item are compared only when both are on the same basis:

- the standard assessment (section 3.5) leaves out a history year whose inputs were on another basis than the focus
  year's;
- **total debt** never counts a part twice: long-term debt tagged including current maturities is not added to a
  current-maturities tag, and a current-debt tag that includes short-term borrowings is not added to a
  short-term-borrowings tag (`V.debt`);
- **depreciation** uses one measure for every year compared: depreciation when it is tagged in all of them, else
  D&A less amortization of intangibles when that amortization is tagged in all of them, else D&A (`dep_measure`);
- **gross margin** compared across years uses the tagged gross profit only when it is tagged in all of them, else
  revenue less cost of sales in all of them (Costco tagged gross profit on net sales, leaving out membership fees, in
  one year only);
- a change in a balance (receivables, inventory, deferred revenue) needs the balance in both years; a balance tagged
  in one year only is an unknown change, never the whole balance.

### 2.2 Periods and point-in-time values (`xbrl/facts.py`)

**What it is.** The SEC's companyfacts data holds every value a company has ever reported, each with the filing that
reported it.

**Why it matters.** Restated figures replace the originals in most databases. If the analysis used restated figures
it would test numbers the market never saw, and would miss the restatement itself. Using values *as first reported*,
and only filings public by the `--as-of` date, makes the analysis reproduce what an investor could have known.

**How.**

- **Fiscal years**: clustered end dates of facts covering 340-390 days in annual reports (10-K, 20-F, 40-F, and 6-K
  for filers that file annual statements that way). End dates within 8 days are merged (52/53-week years).
- **Quarters**: facts covering 80-100 days. Each fact is assigned to the nearest period end within 8 days.
- **First vs latest**: for each item and period the highest-priority concept reported wins; the *first* panel then
  takes its value from the earliest filing that reported it, the *latest* panel from the most recent. Tests use the
  first panel; the restatement test (A.14) compares the versions.
- **Point in time for each year**: a fiscal year is analysed with what was public by the end of its filing season,
  120 days after the year end or the day its annual report was filed if later, and never after `--as-of`
  (`point_in_time` in `pipeline/run.py`). Facts, restated comparatives, events, annual reports and earnings releases
  filed later do not reach that year's tests or investigation, so a past year (`--period`, or the earlier years of a
  multi-year review) sees what investors saw then. Comment letters are the one exception (E.10): the SEC reviews an
  annual report after it is filed, so letters up to a year after the year end are about that year's report.
- **Fourth quarter**: companies rarely report Q4 separately, so for flow items it is derived:

  ```
  Q4 = FY - (Q1 + Q2 + Q3)
  ```

- **Share-count scale fix**: if `|NI / EPS|` equals the tagged share count × 1,000,000 or × 1,000 within 5%, the
  count was tagged in millions or thousands; it is rescaled and the correction is noted in the report.

The processed company data is cached and refreshed daily.

### 2.3 Industry cross-section (`peers/crosssection.py`)

**What it is.** SEC "frames" return one concept for every filer in a calendar period. Four calendar years of revenue,
net income, operating cash flow, cost of sales, SG&A, operating income, gross profit (annual frames) and total
assets, receivables, net PP&E and inventory (the quarterly instant frame matching each firm's fiscal year end) form
the cross-section.

**Why it matters.** Regression models of "normal" accruals (B.03, B.09) and expectation models (D.04) need hundreds
of firms to estimate what a typical company in the industry looks like.

A fiscal year belongs to calendar year y if it ends in July of y or later, otherwise to y-1, matching how the SEC
assigns frames. Each item's concepts keep the canonical priority order (total revenue net of interest expense before
contract revenue, so a bank's or card issuer's revenue is its total). Frames carry each firm's latest value, so peer baselines can include peers' later restatements; the
company under review always uses its own first-reported values.

### 2.4 Industry universe and peer selection (`peers/universe.py`, `peers/select.py`)

**What it is.** A nearest-neighbour search for the most similar companies in the same industry.

**Why it works.** Ratios such as receivable days or gross margin are only meaningful against companies with the
same economics: a software company and a retailer differ by design. Matching on industry first, then on size,
profitability and growth, removes most of the business-model differences, so what remains is more likely to reflect
accounting choices.

**How.**

1. **Universe**: companies with the same 4-digit SIC code that appear in the current cross-section. If there are
   fewer than 30 (configurable), the code is widened to 3 digits, then 2.
2. **Distance**: each feature is standardized across the universe, `z = (x - mean) / standard deviation`, and the
   weighted distance to the company is

   ```
   d(i, target) = sqrt( sum over features f of  w_f × (z(i, f) - z(target, f))^2 )
   ```

   | Feature | Definition | Weight |
   |---|---|---|
   | Revenue size | `ln(Revenue)` | 2.0 |
   | Asset size | `ln(Total assets)` | 1.5 |
   | Gross margin | `GP / Revenue`, clipped to -1..1 | 1.0 |
   | Operating margin | `OI / Revenue`, clipped to -1..1 | 1.0 |
   | Growth | `Rev_t / Rev_t-1 - 1`, clipped to -1..3 | 0.5 |

   Size weighs most because scale drives cost structure, customer mix and accounting complexity. Missing features take
   the universe median. For a company reporting in another currency the two size features are dropped.
3. The 15 nearest (configurable) are the peers. Each peer's own annual data is then loaded, and comparisons use the
   peer fiscal year closest to the company's (within 183 days).

### 2.5 Filing text (`text/document.py`, `text/verify.py`)

Annual reports are converted from HTML to text, split into Items (Items 1, 1A, 7, 7A, 8, 9A and so on; 20-F items are
mapped onto their 10-K equivalents; Exhibit 13 and 40-F exhibits are appended when the financial statements are filed
as exhibits) and into paragraphs with fixed character positions. Every quote used anywhere must be found verbatim at
its position, after normalizing whitespace, dashes and quote marks. Earnings releases are the EX-99 exhibits of 8-K
Item 2.02 filings: the six latest filed between 430 days before the fiscal year end and 120 days after it that announce
a quarter of that year or earlier (a first-quarter release filed within 120 days of the year end, as American Express
files it, announces the next year and is not read), so a past year is tested against its own releases rather than
later ones. This makes every piece of text evidence traceable to
the filing.

### 2.6 Company profiles (`profiles.py`)

**What it is.** A classification of the company by business model, from its SIC code:

| SIC range | Profile |
|---|---|
| 6000-6199 | Bank |
| 6200-6299 | Broker-dealer / asset manager |
| 6300-6411 | Insurer |
| 6500-6553, 6798 | Real estate / REIT |
| 6770 | Blank-check company |
| other 6700-6799 | Investment vehicle |
| everything else | Industrial / commercial |

**Why it matters.** Most forensic models were built for industrial companies. For a bank, receivables are loans
rather than trade credit and operating cash flow includes lending flows; for an insurer, revenue includes investment
gains. Applying industrial models to them produces confident nonsense. Each profile therefore switches off the tests
whose inputs mean something different for that business, with the reason shown in the report, and removes
meaningless ratios from the peer-outlier models. A non-USD reporting currency switches off Altman Z and the D.04
expectation model (both use USD amounts); foreign private issuers lose D.06 (trained on 10-K filers' restatements).

---

## 3. Shared statistics

### 3.1 Robust z-score (`detectors/stats.py`)

**What it is.** A measure of how many "typical deviations" a value is from the middle of a reference group, using the
median and the median absolute deviation (MAD) instead of the mean and standard deviation:

```
z = (x - median(R)) / (1.4826 × MAD(R))
MAD(R) = median( |r - median(R)| )
```

**Why it works.** Financial ratios have fat tails: one peer with an unusual year can double the standard deviation
and hide a real outlier, and the mean is pulled towards that peer. The median and MAD ignore extreme values, so the
baseline reflects typical companies. The factor 1.4826 scales the MAD so it equals the standard deviation for
normally distributed data, which keeps the familiar reading (|z| of 2 is unusual, 3 is rare).

If the MAD is close to zero the sample standard deviation is used, and if that is also close to zero,
`max(0.05 × |median|, 1e-9)`. At least 3 reference values are required.

### 3.2 Percentile

`scipy.stats.percentileofscore(R, x, kind="mean")`: the share of reference values below x, counting ties as half.
Percentiles are easier to read than z-scores ("higher than 92% of peers") and are reported alongside them.

### 3.3 Severity from a z-score

```
severity = clip( (z - z_start) / (z_full - z_start), 0, 1 )
```

with defaults `z_start = 1.5` and `z_full = 4.0`; tests that are noisier use higher starting points. The z-score is
signed so that the direction that would favour reported profit is positive: a test only reacts to departures that
could flatter the results.

### 3.4 p-value

One-sided normal tail: `p = 1 - Phi(z)`, where Phi is the standard normal distribution function. p-values feed the
false-discovery control in section 10.3.

### 3.5 The standard ratio assessment (`assess`, used by most of Family C)

**What it is.** A two-baseline comparison: the company's ratio this year against its own last five years H, and
against its peers P.

**Why it works.** Each baseline alone is misleading. Own history alone treats a long-standing difference as normal
even if the company has always been aggressive; peers alone treat a business-model difference as an anomaly. A ratio
that is unusual against **both** is the strongest evidence of a change in accounting, so agreement is rewarded.

```
z_s = s × z(r_t against H)
z_p = s × z(r_t against P)
s = +1 when a higher ratio is worse, -1 when a lower ratio is worse
```

Own history needs 3 or more years and peers 5 or more. A history year whose inputs were reported on a different basis
from the focus year's (section 2.1) is left out of H. With S_s and S_p the severities from 3.3:

```
severity = 0.55 × S_s + 0.45 × S_p + 0.15 × min(S_s, S_p)    both baselines available
severity = 0.8 × S_s                                       own history only
severity = 0.6 × S_p                                       peers only
```

clipped to 0..1. Own history weighs slightly more because a change in the company's own behaviour is more telling
than a level difference from peers; single-baseline results are scaled down because they are weaker evidence. A
`min_abs` floor sets severity to 0 when the ratio differs from both medians by less than an economically trivial
amount (for example 5 days of receivables): statistically unusual but too small to matter. The **baseline** used for
dollar impacts is the more conservative median (the higher one when higher is worse).

### 3.6 Dollar impact of a ratio gap (`gap_impact`)

**What it is.** A conversion of "this ratio is too high" into "this many dollars of profit could be affected".

```
mid  = (r_t - baseline) × driver
low  = 0.5 × mid
high = max( mid, (r_t - lower of the two medians) × driver )
```

**Why it works.** If a ratio is above its normal level because of an accounting choice, the excess is what would
reverse if the ratio returned to normal. For receivable days, for example, the driver is daily revenue, so the excess
days times daily revenue is the receivable balance that would not be there at a normal collection pace. The range
reflects uncertainty about which baseline is "normal". No impact is produced when the gap is not in the direction
that flatters profit.

---

## 4. Family A: accounting integrity

### What this family does and why it works

Double-entry bookkeeping imposes identities: the balance sheet balances, the change in cash equals the cash flows,
equity moves only through profit and transactions with owners. Financial statements produced by a consistent
process satisfy these identities by construction. When the reported figures break an identity, something was changed
in one place without its counterpart: a late adjustment, a tagging error, or a figure produced outside the normal
ledger. Integrity breaks are rare and hard to explain innocently, so this family carries the highest weight in the
risk index (0.55).

Most tests compute a residual e (what is left unexplained) and scale it by a size base B:

```
sev(e, B, tol, f) = clip( (|e / B| - tol) / (f - tol), 0, 1 )    when |e / B| > tol
sev(e, B, tol, f) = 0                                       otherwise
```

tol is the tolerance for rounding and untagged minor items, f the size at which severity is full.

### A.01 Balance-sheet identity

- **Computes**: `e = TA - (TL + temporary equity + equity including noncontrolling interests)`, and
  `TA - liabilities and equity`; the larger residual counts. Base TA, tol 0.5%, full at 3%.
- **Why it works**: assets must equal claims on them. A gap means the tagged balance sheet is internally
  inconsistent.
- **Innocent causes**: equity classes or redeemable (temporary) equity tagged under uncommon concepts.

### A.02 Subtotal integrity

- **Computes**: `Rev - COGS - GP`; `PP&E gross - accumulated depreciation - PP&E net`; and the excess of
  (cash + short-term investments + receivables + inventory) over total current assets. The worst ratio counts. tol
  0.5%, full at 3%.
- **Why it works**: statements that do not add up point to manual edits or late changes, which reduce confidence in
  every ratio built on them.
- **Innocent causes**: a subtotal that includes a line not tagged separately.

### A.03 Cash roll-forward

- **Computes**: `e = (Cash_t - Cash_t-1) - (CFO + CFI + CFF + FX effect)`, with cash including restricted cash. Base
  `max(|Cash_t|, |CFO|)`, tol 1%, full at 5%. Skipped when cash includes restricted cash in one year and not the other.
- **Why it works**: cash is the hardest figure to manipulate because it can be confirmed with banks. A cash-flow
  statement that does not reconcile to the balance sheet suggests the two were prepared inconsistently.
- **Innocent causes**: restricted-cash definitions, discontinued operations.

### A.04 Equity roll-forward

- **Computes**: `e = E_t - (E_t-1 + NI + OCI - dividends - buybacks + shares issued + stock compensation + convertible
  equity components)`. Dividends are those declared; when only cash dividends paid are tagged they are adjusted by the
  change in dividends payable (Progressive declares an annual dividend in December and pays it in January), and used
  alone only when that change is not tagged. Buybacks are the cash paid, else the treasury stock acquired as the
  statement of equity reports it. Base `max(|E_t|, |E_t-1|)`, tol 4%, full at 15%, then × 0.7; halved again when noncontrolling
  interests are 20% or more of total equity.
- **Why it works**: equity changes that bypass the income statement can hide losses charged directly to equity, or
  reflect transactions the statements do not explain.
- **Innocent causes**: untagged items such as tax withheld when stock awards vest, warrant exercises, and in
  partnership structures exchanges of noncontrolling units into common stock.

### A.05 Retained-earnings roll-forward

- **Computes**: `e = RE_t - (RE_t-1 + NI - dividends)`, with dividends measured as in A.04. If e is negative and no larger than 1.1 × (buybacks + taxes
  paid on net share settlement of stock awards), the residual is treated as explained; if dividends exceed net income
  and e is positive, the excess dividend is deducted. Base `max(|NI|, 0.1 × |RE_t|)`, tol 5%, full at 30%.
- **Why it works**: retained earnings should move only through profit and distributions. Other movements can be
  direct charges to equity or restatement adjustments.
- **Innocent causes**: companies often charge share repurchases and stock-award tax withholding partly to retained
  earnings, and dividends above earnings to paid-in capital; both are recognized automatically.

### A.06 Quarterly-to-annual consistency

- **Computes**: the derived Q4 revenue and cost of sales. Severity 1.0 if Q4 is negative while the Q1-Q3 average is
  positive; 0.6 if Q4 is more than 3 times or less than 0.2 times the Q1-Q3 average.
- **Why it works**: year-end adjustments that make the full year look better often leave an implausible implied
  fourth quarter, because quarterly figures were already published and only the annual total could be changed.
- **Innocent causes**: strongly seasonal businesses.

### A.07 Income-statement tie-out

- **Computes**: `e = (PT - income tax) - (closest of: income from continuing operations, NI including noncontrolling
  interests, NI + income attributable to them, NI)`, adding equity-method income to `PT - tax` when pre-tax income is
  tagged before it (Toyota). Base PT, tol 3%, full at 15%.
- **Why it works**: a gap means lines between pre-tax and net income are missing or inconsistent.
- **Innocent causes**: equity-method income and discontinued operations sit legitimately between these lines.

### A.08 PP&E roll-forward

- **Computes**: `unexplained = dPP&E gross - capex - 0.5 × acquisitions` (only positive gaps count). Base prior gross
  PP&E, tol 3%, full at 15%.
- **Why it works**: fixed assets that grow faster than the cash spent on them can mean operating costs were
  capitalized (moved from expense to assets, raising profit), or assets were acquired through non-cash financing that
  deserves disclosure. Disposals would make the gap larger, so the test is conservative.
- **Innocent causes**: finance leases, vendor financing, asset-retirement obligations, capitalized interest.

### A.09 Allowance roll-forward

- **Computes**: `e = allowance_t - (allowance_t-1 + provision - write-offs)`. Base the larger allowance, tol 10%, full
  at 50%.
- **Why it works**: the allowance for doubtful accounts is a judgment-based reserve. Movements its roll-forward cannot
  explain may be unexplained reserve adjustments that move profit.
- **Innocent causes**: recoveries, acquisitions and currency effects, which are often untagged.

### A.10 Goodwill roll-forward

- **Computes**: `e = G_t - (G_t-1 + acquired - impaired ± FX ± measurement adjustments ± other - disposed)`. Base G_t,
  tol 2%, full at 10%; capped at 0.3 when acquisition and disposal lines are untagged.
- **Why it works**: goodwill should change only through deals, impairments and currency. Unexplained changes can
  signal purchase-accounting adjustments or impairments that were delayed.
- **Innocent causes**: roll-forwards reported only by segment, which makes the tagged roll-forward incomplete rather
  than wrong (hence the cap).

### A.11 Segment reconciliation

Needs segment-level (dimensional) XBRL, which the SEC's companyfacts data does not carry, so it is always skipped.

### A.12 EPS consistency

- **Computes**: `d = NI / diluted shares - diluted EPS`, tolerance `max(0.02, 0.02 × |EPS|)`;
  `severity = clip((|d| - tolerance) / (0.15 × |EPS| + 0.05))`; 0.8 if diluted EPS exceeds basic EPS while the
  company is profitable (dilution can only lower EPS).
- **Why it works**: EPS is the most-watched number. Inconsistencies point to errors in share counts or in the income
  attributed to common shareholders.
- **Innocent causes**: preferred dividends and participating securities create small gaps; share counts tagged in the
  wrong scale are corrected first (section 2.2).

### A.13 Current tax vs cash tax

- **Computes**: `gap = (sum3(tax expense) - sum3(deferred tax) - sum3(taxes paid)) / max(|current tax|, |cash tax|)`;
  `severity = 0.7 × clip((|gap| - 0.25) / 0.5)`. Every sum covers the same years (at least 2 of the last 3) in which
  tax expense and taxes paid are both reported; deferred tax is subtracted in all of them or, when it is never
  reported, in none.
- **Why it works**: tax authorities see a version of profit that management cannot shape as freely. Book profits that
  never attract cash tax can mean earnings are overstated relative to taxable income. Three years smooth payment
  timing.
- **Innocent causes**: tax credits, refunds, losses carried forward, low-tax jurisdictions.

### A.14 Silent restatement

- **Computes**: for each significant item and each of the last three years, the first and the last reported value.
  If the change is at least 0.5% of the first value: `severity = 0.3 + 4 × |change| / base` for income items
  (revenue, cost of sales, pre-tax, operating and net income, retained earnings, equity), or
  `0.15 + 2 × |change| / TA` for balance-sheet items, capped at 1. The base is NI for net, pre-tax and operating income
  and retained earnings, and TA for the others (a revenue revision measured against NI would look ten times larger
  than it is). The result is labelled when the later filing is an amendment or comes within a year after a
  non-reliance notice. Only filings public by the focus year's point in time (section 2.2) count, so a past year is
  never charged with a revision made in a later year's report.
- **Why it works**: a prior year re-reported differently changes the history investors relied on. Without a formal
  restatement it may go unnoticed. Income items weigh more because they directly change reported performance.
- **Innocent causes**: discontinued operations, segment recasts, new standards adopted retrospectively.

### A.15 MD&A text vs reported figures

- **Computes**: company-wide revenue and net income totals stated in MD&A, matched against the reported totals with
  the rules in section 8.4; `severity = min(1, 0.45 × mismatches)`. Earlier years are taken as this annual report
  presents them (its comparatives may be recast). A sentence that names a fiscal year before the figure is compared
  with that year; one that names none can match any of the three years, but is a mismatch only against the focus
  year and only when it is nearer that year's total than any other.
- **Why it works**: narrative that does not match the statements suggests numbers changed after the text was written.
- **Innocent causes**: drafting errors; segment or non-GAAP figures (which are skipped).

### A.16 Rounding consistency

- **Computes**: the share of monetary facts in the annual report not divisible by the dominant rounding unit (the
  largest unit that covers at least 80% of facts); `severity = 0.6 × clip((share - 0.10) / 0.30)`.
- **Why it works**: when most amounts are in millions but some are precise to the dollar, those figures may have been
  added or adjusted by hand late in the process.
- **Innocent causes**: per-share data and some notes legitimately use other precision; hence low weight.

---

## 5. Family B: forensic scores and accrual models

### What this family does and why it works

These are published academic models, each estimated on companies that were later found to have misstated, or built
around the mechanics of accruals. They summarize many ratios into one score and are computed identically for the
company and each peer (`detectors/scores.py`), so the company's position among its peers is always reported. Their
coefficients come from historical samples, so they are best used to rank risk rather than as calibrated
probabilities.

### 5.1 Beneish M-score, B.01

**What it is.** Messod Beneish (1999) compared 74 companies subject to SEC enforcement for earnings manipulation with
matched controls (1982-1992) and estimated a model on eight indices, each comparing this year with last year:

```
M = -4.84 + 0.920 × DSRI + 0.528 × GMI + 0.404 × AQI + 0.892 × SGI
          + 0.115 × DEPI - 0.172 × SGAI + 4.679 × TATA - 0.327 × LVGI
```

**Why it works, index by index:**

| Index | Formula | Economic reasoning |
|---|---|---|
| DSRI, receivables index | `(AR_t/Rev_t) / (AR_t-1/Rev_t-1)` | Receivables growing faster than sales suggest revenue recorded before it is collected, or not collectible at all |
| GMI, gross margin index | `GM_t-1 / GM_t` | Deteriorating margins create pressure to manipulate; values above 1 mean margins fell |
| AQI, asset quality index | `(1 - (CA_t + PP&E_t + STI_t)/TA_t) / (1 - (CA_t-1 + PP&E_t-1 + STI_t-1)/TA_t-1)` | A rising share of "soft" assets (neither current assets nor PP&E) suggests costs are being deferred as assets |
| SGI, sales growth index | `Rev_t / Rev_t-1` | Growth is not manipulation, but fast growers face pressure to sustain growth and have weaker controls |
| DEPI, depreciation index | `(Dep_t-1/(Dep_t-1 + PP&E_t-1)) / (Dep_t/(Dep_t + PP&E_t))`, with one depreciation measure in both years (section 2.1) | A slowing depreciation rate raises profit, possibly through longer useful lives |
| SGAI, SG&A index | `(SGA_t/Rev_t) / (SGA_t-1/Rev_t-1)` | Rising overheads relative to sales signal weakening prospects (the coefficient is negative in the estimated model) |
| TATA, total accruals | `(income from continuing operations_t - CFO_t) / TA_t` | Earnings not backed by cash are the most direct footprint of aggressive accruals; it has by far the largest coefficient |
| LVGI, leverage index | `((CL_t + LTD_t)/TA_t) / ((CL_t-1 + LTD_t-1)/TA_t-1)`, LTD excluding current maturities (already in CL); neutral when LTD is tagged in one year only | Rising leverage tightens debt covenants and increases the incentive to manage results |

**How.** Missing components take their neutral value (1 for indices, 0 for TATA); SGI and TATA are required.
`severity = clip((M + 2.4) / 1.4)`, so 0 at M = -2.4 and 1 at M = -1.0 (the published flag is M > -1.78). The three
components that raise M most above neutral are reported as drivers; a receivables-driven score joins the
premature-revenue pattern.

### 5.2 Dechow F-score, B.02

**What it is.** Dechow, Ge, Larson and Sloan (2011) studied companies subject to SEC Accounting and Auditing
Enforcement Releases (1979-2002) and estimated a logistic regression of the probability of misstatement (model 1,
financial statement variables only):

```
logit = -7.893 + 0.790 × RSST + 2.518 × dREC + 1.191 × dINV + 1.979 × SOFT
               + 0.171 × dCS - 0.932 × dROA + 1.029 × ISSUE
P = 1 / (1 + exp(-logit))
F = P / 0.0037
```

0.0037 is the unconditional misstatement rate in their sample, so F = 1 is average risk and F = 2 twice average.

**Why it works, variable by variable:**

- **RSST accruals**: `(dWC + dNCO + dFIN) / avg(TA)` with `WC = (CA - cash - STI) - (CL - short-term debt)` (short-term
  debt = total debt less LTD excluding current maturities, section 2.1),
  `NCO = (TA - CA - equity-method investments) - (TL - CL - LTD)`, `FIN = (STI + equity-method investments) - (LTD +
  short-term debt)`. This balance-sheet measure of total accruals (Richardson, Sloan, Soliman and Tuna 2005) captures
  accruals in all accounts, including long-term ones that a working-capital measure misses.
- **dREC = dAR / avg(TA)** and **dINV = dInventory / avg(TA)**: the two accounts most often used to inflate revenue
  and margins.
- **SOFT = (TA - PP&E - cash) / TA**: soft assets need more judgment to value, so they leave more room to manage
  earnings.
- **dCS = CS_t / CS_t-1 - 1**, with cash sales `CS = Rev - dAR`: managers who see real sales slowing have a motive to
  manage reported results.
- **dROA**, the change in return on assets: a negative coefficient, because manipulation typically follows a period
  of strong results that management struggles to sustain, and props up a declining ROA.
- **ISSUE = 1** if debt or equity was issued: firms raising capital benefit most from looking strong.

**How.** Inputs are capped to limit extreme values (RSST and dROA at ±1, dREC and dINV at ±0.5, dCS at ±3).
`severity = clip((F - 0.8) / 1.8)`, full at F = 2.6 (the paper's "high risk" band starts at 2.45).

### 5.3 Discretionary accruals: the Kothari modified Jones model, B.03

**What it is.** A cross-sectional regression that separates accruals into a "normal" part, explained by the
company's growth, asset base and performance, and a "discretionary" part that is not. It descends from Jones (1991),
modified by Dechow, Sloan and Sweeney (1995) to exclude credit sales, and by Kothari, Leone and Wasley (2005) to
control for performance. It is estimated by ordinary least squares across companies in the industry and year
(`detectors/xsec_models.py`):

```
(NI_i - CFO_i) / A_i,t-1 = b0 + b1 × (1 / A_i,t-1)
                         + b2 × (dRev_i - dAR_i) / A_i,t-1
                         + b3 × PP&E_i / A_i,t-1
                         + b4 × ROA_i + e_i

(i = each company in the industry-year; A_i,t-1 = its total assets at the start of the year)
```

**Why it works.** Some accruals are a natural consequence of the business: a growing company accumulates working
capital, and a company with more fixed assets records more depreciation. Regressing accruals on these drivers across
the industry estimates how much accrual a normal company with the same growth and asset intensity would have. The
residual e is the part management's choices most likely explain. Revenue growth is measured net of receivables
growth because credit sales are themselves discretionary. ROA is included because high performers naturally have
higher accruals, which would otherwise look discretionary. The 1/A term absorbs size effects.

**How.** At least 15 usable firms (otherwise all filers are used); the company itself is excluded from the fit;
firms with lagged assets under $1M are dropped; all variables are winsorized at the 1st and 99th percentiles to
limit the influence of extreme firms. The company's residual is scored by robust z against the industry residuals
(1.0 to 3.0), plus 0.15 when last year's residual was also positive, because accruals that stay high are less likely
to be noise. Impact: mid = residual × lagged assets; low = (residual - 75th percentile of industry residuals) ×
lagged assets.

### 5.4 Accrual quality: Dechow-Dichev / McNichols, B.04

**What it is.** Dechow and Dichev (2002) measure accrual quality as how well this year's working-capital accruals
map into cash flows of the previous, current and next year; McNichols (2002) added the Jones model's revenue and PP&E
terms. A pooled regression over the company's and its peers' firm-years (25 or more), scaled by average assets:

```
dWC_t = b0 + b1 × CFO_t-1 + b2 × CFO_t + b3 × CFO_t+1 + b4 × dRev_t + b5 × PP&E_t + e_t
```

with `dWC = dAR + dInventory + d(other current assets) - dAP - d(accrued liabilities)`.

**Why it works.** A good accrual is an early recognition of cash that later arrives (a receivable collected next
year) or a deferral of cash already received. If accruals are reliable, they are well explained by surrounding cash
flows, and residuals are small. Large or erratic residuals mean accruals that never turn into cash: estimation errors
or deliberate misstatement. The standard deviation of a company's residuals over its last five years (3 or more
required) is its accrual quality; higher is worse.

**How.** Severity from the robust z against peers' values (1.0 to 3.0), × 0.8. Each peer contributes only its years up
to the one aligned with the focus year. Because next year's cash flow is
needed, the latest year scored is the year before the focus year.

### 5.5 Sloan accrual ratio, B.05

**What it is.** Richard Sloan (1996) showed that the accrual part of earnings is less persistent than the cash part,
and that investors overprice high-accrual companies. The ratio scales accruals by net operating assets:

```
Sloan = (NI - CFO - operating investment) / average NOA
operating investment = -(capex + acquisitions)
NOA = (TA - cash - STI) - (TL - total debt)
```

**Why it works.** Earnings management usually works through accruals, and accruals reverse. A company in the top
decile of accruals is more likely to report lower earnings later, whatever the cause. Purchases of marketable
securities are deliberately left out: they are investing cash flow but not accruals (those securities are excluded
from net operating assets too), and including them would make cash-rich companies look as if their profit were not
backed by cash.

**How.** Total investing cash flow is used when capex is not reported, and average total assets when NOA is not
positive. Severity from the robust z against peers (5 or more) or else own history (1.0 to 3.5), × 0.4 when the ratio
is negative (low accruals are not a warning sign).

### 5.6 Montier C-score, B.06

**What it is.** James Montier's (2008) checklist of six common footprints of aggressive accounting, one point each:

| Test | Why it is a footprint |
|---|---|
| NI - CFO gap widening | Profit increasingly running ahead of cash |
| Receivable days (`AR / Rev`) rising | Sales recorded but not collected |
| Inventory days (`Inv / COGS`) rising | Inventory overstated, or costs left in inventory |
| Other current assets / revenue rising | Costs parked as prepaid or deferred items |
| Depreciation / gross PP&E falling (one depreciation measure in both years) | Profit raised by slower depreciation |
| Total assets growing more than 10% | Acquisitive or fast growth, where accounting is most flexible |

**How.** At least 4 tests must be computable; `severity = clip((C - 2) / 3)`. The score is simple by design: each
point is a separate, interpretable symptom.

### 5.7 Context scores: Piotroski F-score (B.07) and Altman Z (B.08)

**What they are.** Piotroski (2000) scores fundamental strength with nine binary tests: ROA positive, CFO positive,
ROA rising, CFO above ROA, leverage falling, current ratio rising, no dilution (shares at most 1.005 × prior year),
gross margin rising, asset turnover rising. Altman (1968) predicts bankruptcy:

```
Z = 1.2 × x1 + 1.4 × x2 + 3.3 × x3 + 0.6 × (market value of equity / TL) + 1.0 × x5
x1 = (CA - CL) / TA,  x2 = RE / TA,  x3 = EBIT / TA,  x5 = Rev / TA
```

with market value approximated by the public float; without it the book-value version
`Z'' = 6.56 × x1 + 3.26 × x2 + 6.72 × x3 + 1.05 × (book equity / TL)`.

**Why they are used.** The fraud triangle (Cressey 1953) says misstatement needs pressure, opportunity and
rationalization. Weak fundamentals and financial distress are the clearest measurable pressures: covenants,
refinancing and survival all depend on reported results. These scores are not anomalies, so they are tagged
*pressure* and count only 0.15 towards the index (section 10.6).

**How.** Piotroski: a test whose inputs are not reported is not computable, not failed (at least 7 of 9 required);
`severity = 0.5 × clip((4 - F × 9 / computable) / 4)`, flagged at 0.45. Altman:
`0.7 × clip((2.99 - Z) / (2.99 - 1.81))`, or `0.7 × clip((2.6 - Z'') / (2.6 - 1.1))`.

### 5.8 Earnings management around thresholds, B.09

**What it is.** Two tests of whether results are steered to meet targets.

1. **Small-profit and small-increase frequency.** Burgstahler and Dichev (1997) showed that reported earnings
   cluster just above zero and just above last year's figure far more often than chance predicts, and just below them
   far less often. Over 12 quarters:
   `rate = (quarters with ROA between 0 and 0.5% + quarters with EPS up by 0 to 1 cent vs the same quarter a year
   earlier) / quarters`; `severity = 0.8 × clip((rate - 0.10) / 0.25)`. The ROA part is used for industrial
   companies only.
2. **Real earnings management** (Roychowdhury 2006). Managers can also hit targets with real decisions: discounting
   to pull sales forward, overproducing to spread fixed costs, or cutting R&D and advertising. Three industry-year
   regressions estimate normal levels, with A = lagged assets and S = revenue:

   ```
   CFO / A           = b0 + b1 × (1/A) + b2 × (S_t / A) + b3 × (dS_t / A)
   (COGS + dInv) / A = b0 + b1 × (1/A) + b2 × (S_t / A) + b3 × (dS_t / A) + b4 × (dS_t-1 / A)
   SGA / A           = b0 + b1 × (1/A) + b2 × (S_t-1 / A)
   ```

   Price discounts lower cash flow per sales dollar; overproduction raises production costs; cuts lower
   discretionary spending. The composite `(-z_CFO + z_PROD - z_DISX) / k` (robust z of each residual against the
   industry, averaged over the k models available) is high when all three point the same way. Severity from z, 1.0
   to 2.5.

**Why it works.** Targets (zero, last year, analysts' forecasts) carry large rewards: missing them costs share price
and bonuses. If results were not managed, they would be distributed smoothly around targets; a pile-up just above is
evidence of steering.

### 5.9 Income smoothing (Leuz), B.10

**What it is.** Leuz, Nanda and Wysocki (2003) measure smoothing as the volatility of operating income relative to
the volatility of operating cash flow, and the correlation between changes in accruals and changes in cash flow:

```
smoothing ratio = stdev(OI / TA) / stdev(CFO / TA)       over the last 5 to 8 years
correlation     = corr( d accruals, d CFO ),  accruals = (NI - CFO) / TA
```

**Why it works.** Cash flows are volatile; earnings should be somewhat smoother because accruals legitimately spread
cash over time. But when earnings are much smoother than cash, and accruals move strongly against cash flows, reserves
are likely being used to even out results (building them in good years, releasing them in bad ones).

**How.** Robust z of the ratio against peers, sign reversed because low is bad (1.0 to 3.0), × 0.7; without peers,
0.4 if the ratio is below 0.3 and the correlation below -0.8.

---

## 6. Family C: account-by-account tests

### What this family does and why it works

Each test looks at one account where accounting judgment is concentrated and measures it in a way that reveals the
most common manipulation of that account. Most compute a ratio and use the two-baseline assessment of section 3.5
(marked "assess"); others use fixed rules. The logic is the same throughout: manipulation moves an account away from
where the business economics would put it, and because accruals must reverse, the departure tends to grow until it
does.

### 6.1 Revenue

Revenue recognition is the most common area of accounting fraud in SEC enforcement cases, because revenue drives
growth, margins and valuation, and because the timing of recognition rests on judgment.

**C.REV.01 Revenue growth vs cash collected.** Cash revenue is `CR = Rev - dAR + d(deferred revenue)` (deferred revenue
from the components reported in both years), the revenue
that actually turned into cash, or cash received in advance. The test measures the growth gap
`(Rev_t / Rev_t-1 - 1) - (CR_t / CR_t-1 - 1)` (assess). *Why*: genuine sales bring in cash; revenue that grows faster
than the cash it produces is the classic signature of premature or fictitious revenue. *Impact*:
`(gap - max(0, baseline)) × Rev_t-1`. *Innocent causes*: longer customer terms, larger customers, acquisitions.

**C.REV.02 Deferred revenue decline.** Billings are `Rev + d(deferred revenue)`; the test measures revenue growth
minus billings growth (skipped when deferred revenue is under 2% of revenue). `severity = clip((gap - 0.02) / 0.10)`,
+ 0.2 if deferred revenue fell more than 15% relative to revenue. *Why*: when revenue grows faster than billings, the
company may be recognizing revenue faster than it signs new business, drawing down its cushion of prepaid revenue.
*Innocent causes*: a switch from annual to monthly billing.

**C.REV.03 Contract assets.** Contract assets (unbilled revenue) / revenue (assess). *Why*: revenue recognized before
the customer can be billed rests on estimates of progress, a common area for aggressive recognition.

**C.REV.04 Quarter-end revenue concentration.** The fourth quarter's share of annual revenue, scored by robust z
against the company's own last five years with four reported quarters (1.5 to 4.0; the result names them), + 0.15 if the fourth quarter's gross margin is also unusual (z above
2). *Why*: pulling sales into the last quarter (channel stuffing, side agreements) inflates the year at the expense
of the next, and concentrates in the quarter that closes the year.

**C.REV.05 Revenue vs asset base.** Asset turnover `Rev / avg(TA)` (assess, 2.0 to 5.0). *Why*: revenue that jumps
without the assets to produce it may not be backed by real capacity. *Innocent causes*: outsourcing, asset-light
models and price increases; the wider z range reflects how often these apply.

**C.REV.07 Gross vs net presentation.** New principal-versus-agent wording in the filing, together with a revenue
jump and lower gross margin: 0.7 with all three, 0.25 with the wording only. *Why*: reporting the full value of sales
the company only arranges (gross) instead of its commission (net) inflates revenue without adding profit; the margin
falls as a result.

**C.REV.08 Related-party revenue.** Related-party revenue / revenue; `severity = clip((share - 0.02) / 0.10)`, + 0.2
if rising. *Why*: sales to related parties may not be at arm's length and can be used to round-trip cash.

**C.REV.10 Revenue vs remaining performance obligations.** The number of consecutive years in which revenue growth
exceeded backlog growth by more than 3 points, each growth rate measured between adjacent years; `severity = clip(years / 3) × (0.9 if backlog exceeds 30% of revenue, else 0.5)`. *Why*:
revenue that outgrows backlog year after year may be borrowing from future periods.

**C.REV.11 Aggressive revenue-arrangement language.** New bill-and-hold, consignment, extended-payment-terms or
side-letter wording; `severity = min(1, 0.3 × count + 0.25 if receivable days rose more than 10%)`. *Why*: these
arrangements are where revenue is most often recognized too early, and rising receivables are the corroborating
footprint.

C.REV.06 (segment growth) and C.REV.09 (operating KPIs) are skipped: they need segment-level or company-specific
data.

### 6.2 Receivables

**C.AR.01 Receivable days (DSO).** `DSO = AR / Rev × 365` (assess, minimum 5-day change). *Why*: receivables growing
faster than sales mean revenue is being recorded but not collected, whether through early recognition, extended
terms or fictitious sales. *Impact*: `(DSO - baseline) × Rev / 365`, the receivable balance in excess of a normal
collection pace. The latest quarter's DSO is also shown against the same quarter a year earlier, found by date.

**C.AR.02 Allowance coverage decline.** Coverage `allowance / (AR + allowance)` against its prior four-year median;
`severity = clip((drop - 0.10) / 0.40) × (1 if DSO rose, else 0.6)`. *Why*: reducing the reserve for bad debts while
receivables age raises profit by under-reserving.

**C.AR.03 Provision below write-offs.** Consecutive years back from the focus year (a year not reported ends the run)
in which the provision for credit losses was below write-offs; `severity = 0.8 × clip((years - 1) / 2)`. *Why*: the reserve is being drawn down into earnings.

**C.AR.04 Receivables factoring.** New wording about selling or factoring receivables: 0.55 if new, 0.2 if existing.
*Why*: selling receivables lowers receivable days and lifts operating cash flow, which can mask collection problems.

**C.AR.05 Long-term receivables growth.** Growth in non-current receivables against revenue growth;
`severity = clip((growth - revenue growth - 0.2) / 0.6) × clip(share of revenue / 0.05)`. *Why*: extended payment
terms move receivables out of current assets and can disguise customer financing.

### 6.3 Inventory

**C.INV.01 Inventory days (DSI).** `DSI = Inv / COGS × 365` (assess, minimum 5-day change); the finished-goods share
is reported. *Why*: inventory that builds faster than sales may be overstated or obsolete. *Impact*:
`(DSI - baseline) × COGS / 365`.

**C.INV.02 Inventory growth vs COGS growth.** Inventory growth minus cost-of-sales growth for two consecutive years;
`severity = clip((spread - 0.05) / 0.25)`, + 0.2 if both years. *Why*: inventory that keeps outgrowing what is sold
tends to end in write-downs.

**C.INV.03 Margin up while inventory builds.** `severity = clip(dGM / 0.03) × clip(dDSI / 20)` when both rise. *Why*:
producing more than is sold spreads fixed factory costs over more units, so cost per unit sold falls and margins rise
without more sales. Under-reserving for obsolescence has the same effect.

**C.INV.04 Inventory reserve decline.** Reserve / (inventory + reserve) falling;
`severity = clip((drop - 0.1) / 0.4) × (1 if DSI rose, else 0.5)`. *Why*: a shrinking reserve on ageing stock
overstates inventory and profit.

**C.INV.05 LIFO liquidation.** Decline in the LIFO reserve relative to pre-tax income;
`severity = clip(release / PT / 0.05)`. *Why*: selling old, low-cost inventory layers gives a one-time margin boost
that will not recur.

### 6.4 Expenses and capitalization

Capitalizing a cost moves it from this year's expenses to the balance sheet, to be expensed later. Every test here
looks for costs that are being moved forward in time.

**C.EXP.01 Capitalization rate.** Capitalized software / (SG&A + R&D + capitalized software) (assess), with the
overhead lines the focus year reports required in every year compared.
*Why*: capitalizing more development cost moves expense off the income statement.

**C.EXP.02 Soft-asset growth.** (Other current + other non-current assets) / revenue (assess). *Why*: deferred costs
and prepaid items are where expenses can be parked.

**C.EXP.03 Useful-life extension.** Implied useful life `gross PP&E / depreciation` against its prior three-year
median; `severity = clip((change - 0.10) / 0.30)`. *Why*: longer assumed lives cut depreciation and raise profit
without any change in the business. *Impact*: `gross PP&E / prior life - reported depreciation`, the depreciation at
the old life less what was reported. All four years use the same depreciation measure (section 2.1), so a year that
tags depreciation on its own is never compared with years that only report D&A.

**C.EXP.04 Depreciation rate drop.** Depreciation / average gross PP&E (assess, lower is worse), with the measure the
company reports in all of its last six years used for every year and every peer. *Why*: the same
effect seen as a rate. *Innocent causes*: large construction in progress, not yet depreciated.

**C.EXP.05 Capex far above depreciation.** Years with capex / D&A above 2 while revenue is flat;
`severity = clip((years - 1) / 2) × (1 if two-year revenue growth is under 5% a year, else 0.3)`. *Why*: heavy
investment that produces no growth can mean operating costs are being capitalized.

**C.EXP.06 Discretionary expense cut.** (SG&A + R&D) / revenue against own history only (assess, lower is worse,
z from 2); a year that does not report a line the focus year reports is left out, so an untagged line is never read
as a cut. *Why*: cutting discretionary spending to hit a target lifts this year's profit at the expense of future
growth (real earnings management).

**C.EXP.07 Recurring "non-recurring" charges.** Restructuring or impairment charges in 3 or more of the last 5 years;
`severity = 0.6 × clip((years - 2) / 2) × clip(average charge / PT / 0.05, 0.3, 1)`, with the last factor 0.3 when
pre-tax income is not reported. *Why*: charges labelled one-time
that recur every year are operating costs; excluding them flatters adjusted earnings.

**C.EXP.08 Pension assumptions.** Discount rate and expected return on assets against peer medians;
`severity = 0.7 × clip((gap - 0.25%) / 1%)`. *Why*: higher assumptions lower reported pension cost.

**C.EXP.09 Stock-based compensation intensity.** Stock compensation / revenue (assess, z from 2). *Why*: a real cost
that is usually excluded from adjusted earnings.

### 6.5 Reserves and tax

Reserves are estimates of future costs. Over-reserving in good years and releasing the excess in weak years
("cookie-jar" accounting) smooths earnings and hides deterioration.

**C.RES.01 Reserve release into earnings.** For the allowance, inventory reserve and warranty accrual,
`release = (reserve_t-1 / base_t-1 - reserve_t / base_t) × base_t`, the amount by which a falling reserve ratio raised
profit; `share = max(release / |PT|, release / growth in PT)`; `severity = clip((share - 0.05) / 0.20)`. *Why*: a
release that supplies more than 5% of profit, or of profit growth, is material to the results.

**C.RES.02 Warranty accrual vs claims.** The accrual rate falling while claims stay at 95% or more of the prior rate;
`severity = clip((drop - 0.10) / 0.4)`. *Why*: accruing less while claims stay steady inflates profit and leaves a
future shortfall.

**C.RES.03 Valuation-allowance release.** Decrease in the deferred-tax valuation allowance / pre-tax income;
`severity = clip((share - 0.05) / 0.25)`. *Why*: the release adds a tax benefit that is a judgment about future
profits, not current performance.

**C.RES.04 Big bath.** Large charges last year relative to revenue followed by a margin rebound;
`severity = clip((charges - 0.03) / 0.07) × clip(rebound / 0.05)`, + 0.2 with a CEO, CFO or chief accounting
officer change in the two years (pay-plan and director filings do not count). *Why*: front-loading
costs (often by new management, who can attribute them to predecessors) makes later results look better, and
overstated charges can reverse into profit.

**C.RES.05 Effective tax rate drop.** Tax / pre-tax income (assess, lower is worse, z from 2). *Why*: tax is a large,
judgment-based line; a falling rate can supply much of reported earnings growth.

**C.RES.06 Litigation exposure vs accrual.** Growth in litigation wording while the litigation accrual is flat;
`severity = 0.7 × clip((growth - 0.25) / 0.75) × (1 if flat, else 0.4)`. *Why*: growing legal exposure with no
growth in the accrual may mean losses are under-recognized.

### 6.6 Cash flow

Operating cash flow is harder to manipulate than earnings, which is why so many tests compare earnings with cash.
But cash flow can be flattered too, by timing and by classification.

**C.CF.01 Cash flow vs net income.** `r = sum3(CFO) / sum3(NI)` (three years of positive net income);
`severity = clip((0.9 - r) / 0.5)`, + 0.15 if r fell more than 0.1 from the previous window. *Why*: earnings that do
not turn into cash over several years are the single most common footprint of aggressive accruals; three years
smooth timing. *Impact*: mid `(0.9 - r) × average NI × 0.5`.

**C.CF.02 Working-capital-driven cash flow.** The largest single working-capital line's share of growth in operating
cash flow; `severity = clip((share - 0.5) / 0.5) × (1 for payables, 0.7 otherwise)`. *Why*: cash flow growth that
comes from one line, especially paying suppliers later, is not sustainable.

**C.CF.03 Payables stretch.** `DPO = AP / COGS × 365` (assess, minimum 7-day change), + 0.2 × the supplier-finance
share of payables where disclosed. *Why*: paying suppliers later flatters operating cash flow; supplier-finance
programmes can hide debt-like obligations.

**C.CF.04 Costs moving from operating to investing.** A jump in (capex + capitalized software) / operating cash flow
while D&A is flat; `severity = clip((jump - 0.10) / 0.30) × (1 if D&A growth is under 10%, else 0.5)`. *Why*:
classifying operating costs as investment moves cash outflows out of operating cash flow.

**C.CF.05 Acquisition-boosted cash flow.** Acquisitions over 5% of assets with operating cash flow growth above
revenue growth; `severity = clip((acquisitions / TA - 0.05) / 0.15) × clip((CFO growth - revenue growth) / 0.3)`.
*Why*: receivables bought with an acquisition and collected afterwards boost operating cash flow once, without
recurring.

**C.CF.06 Free cash flow vs EBITDA.** `(CFO - capex) / (OI + D&A)` (assess, lower is worse); not computed for a year
without D&A or capex. *Why*: operating profit
before depreciation that does not turn into free cash flow suggests earnings-quality issues or heavy reinvestment.

**C.CF.07 Quarter-end debt window dressing.** `dip = (average debt at the three preceding quarter-ends - year-end
debt) / that average`; `severity = 0.8 × clip((dip - 0.10) / 0.30)`. Debt is the borrowings (section 2.1) reported
under the same tags at all four quarter-ends, named in the result (JPMorgan tags only short-term borrowings). *Why*: paying down debt just before year end and
re-borrowing after makes the year-end balance sheet look stronger than the business usually is.

### 6.7 Debt and off-balance-sheet obligations

**C.BS.01 Implied interest rate.** Interest expense / average total debt (assess, z from 2; total debt as in
section 2.1). *Why*: if the company pays
interest at a rate far above market on its reported debt, part of the debt it pays interest on may not be on the
balance sheet.

**C.BS.02 Off-balance-sheet commitments.** Purchase obligations and guarantees: growth against revenue growth, and
size relative to assets; `severity = clip((growth - revenue growth - 0.25) / 0.75) × clip(size / 0.10)`, × 0.4 if
the tagged concept changed between years (the series may not be comparable). *Why*: commitments that are not
liabilities still have to be paid; fast growth can hide leverage.

**C.BS.03 Covenant proximity.** Distance between actual leverage and coverage and the covenant thresholds extracted
by the AI from the debt notes. *Why*: companies close to a covenant breach have a strong motive to manage results;
context, not evidence.

**C.BS.04 Unconsolidated affiliates.** Equity-method investments / equity, with loss-making investees;
`severity = 0.7 × clip((size - 0.05) / 0.20) × (1 if loss-making, else 0.3)`. *Why*: losses or debt can be parked in
affiliates that are not consolidated.

### 6.8 Mergers and acquisitions

Acquisitions give management wide discretion: how the price is allocated, when goodwill is impaired, and how
earn-outs are revalued.

**C.MA.01 Allocation to goodwill.** Goodwill acquired / acquisition spend;
`severity = clip((share - 0.70) / 0.25) × clip(acquisitions / TA / 0.05, 0.3, 1)`. *Why*: goodwill is not amortized
while identifiable intangibles are, so allocating more to goodwill raises future earnings.

**C.MA.02 Delayed goodwill impairment.** Public float below book equity with material goodwill and no impairment;
`severity = clip(gap / 0.4) × clip(goodwill / TA / 0.2, 0.4, 1)`. *Why*: when the market values the company below
its book value, goodwill impairment is usually overdue.

**C.MA.03 Acquisition-reliant growth.** Three-year acquisition spend / revenue;
`severity = 0.6 × clip((share - 0.20) / 0.40)`. *Why*: growth bought through acquisitions makes organic performance
hard to see and can mask decline.

**C.MA.04 Contingent-consideration gains.** Earn-out reductions booked as gains / pre-tax income;
`severity = clip((share - 0.02) / 0.10)`. *Why*: lowering an earn-out produces a gain exactly when the acquired
business underperforms.

**C.MA.05 Measurement-period adjustments.** Post-closing goodwill adjustments / last year's goodwill acquired (total
goodwill when none was acquired; the result says which);
`severity = 0.7 × clip((share - 0.10) / 0.30)`. *Why*: large adjustments mean the original purchase accounting was
not reliable.

### 6.9 Governance

These tests measure public signals that the reporting process is under strain. They come from filings rather than
from the numbers.

| ID | Measures | Severity | Why it works |
|---|---|---|---|
| C.GOV.01 | Auditor change (8-K Item 4.01) within two years | 0.55 | Auditor changes, especially after disagreements, often precede restatements |
| C.GOV.02 | Non-reliance notices (8-K Item 4.02) and annual-report amendments within three years; same-day amendments count once | `min(1, 0.8 × notices + 0.35 × financial amendments + 0.1 × other amendments)` | A non-reliance notice is the strongest public sign of a past error; amendments that re-file financial statements weigh more than exhibit-only ones |
| C.GOV.03 | Officer changes (8-K Item 5.02) in the two years to the fiscal year end (plus the 120 days after it); the latest 8 are read to tell CFO, chief accounting officer and CEO changes apart (the item's own title, which mentions appointments, is ignored). Filings about the same role within nine months are one transition; filings about pay plans or directors (most Item 5.02 filings) do not count | `0.8 x min(1, 0.05 x filings that could not be read + 0.35 x CFO + 0.2 x CEO + 0.15 x CAO transitions)` | Turnover in finance leadership often accompanies accounting problems; an announced succession reported twice is still one change |
| C.GOV.04 | Late-filing notices (NT 10-K, NT 10-Q, NT 20-F) | `min(1, 0.45 × notices)` | A missed deadline often signals unresolved accounting or audit issues |
| C.GOV.05 | Material weakness or ineffective controls in Item 9A | 0.85 (0.25 for weaker wording) | A control weakness means misstatements could go undetected |
| C.GOV.06 | Going-concern doubt | 0.8 | Severe distress is the strongest pressure to manage results |

### 6.10 Adjusted (non-GAAP) measures

Companies headline "adjusted" figures that leave out costs they call unusual. These measures are not defined by
accounting standards, so their definition is a choice, and the choice can manufacture improvement.

| ID | Measures | Severity | Why it works |
|---|---|---|---|
| C.NG.01 | (Adjusted EPS - GAAP EPS) / GAAP EPS for the same period (quarter with quarter, full year with full year) in the year's latest release; GAAP EPS from the release's own table, else the filings for that period, else no comparison. The result states the period, the release date and whether adjusted EPS is above or below GAAP EPS | `clip((gap - 0.20) / 0.60)` | A large gap steers investors towards a flattering measure |
| C.NG.02 | Adjustment categories present in every recent release | `min(0.7, 0.3 × categories)` | Items labelled unusual that recur every quarter are operating costs |
| C.NG.03 | New add-back categories and renamed measures vs the release for the same quarter a year earlier (AI extraction; skipped when there is none) | `min(0.7, 0.2 × new + 0.1 × renamed + 0.3 × asymmetric)` | Changing what is excluded can manufacture growth |
| C.NG.04 | Losses excluded while gains are kept | from C.NG.03 | One-sided adjustments bias adjusted earnings upwards |

---

## 7. Family D: statistical and machine-learning tests

### What this family does and why it works

Rule-based tests only find patterns someone thought to write down. This family looks for departures no rule encodes:
breaks in a company's own time series, combinations of ratios unlike any peer, balances outside what an industry model
predicts, and digit patterns. The methods are robust (they tolerate outliers in the reference data) and each is
weighted moderately (0.30), because statistical unusualness has many innocent causes.

### 7.1 The ratio vector

D.02, D.03 and D.06 use 24 ratios computed by one function (`ratio_vector`), chosen to cover profitability, working
capital, investment, accruals, leverage and growth: gross, operating and net margin; SG&A, R&D and stock compensation
to revenue; receivable, inventory and payable days; asset turnover; `(NI - CFO) / avg(TA)`; CFO / NI; capex / D&A;
D&A / gross PP&E; soft assets / assets; goodwill / assets; liabilities / assets; effective tax rate; revenue growth;
receivables growth minus revenue growth; inventory growth minus cost-of-sales growth; other current assets / revenue;
deferred revenue / revenue; interest / total debt. Ratios the profile marks meaningless are dropped.

### 7.2 Seasonal decomposition and change points, D.01

**What it is.** Two time-series methods applied to quarterly gross margin, receivable days, SG&A ratio and inventory
days (up to 24 quarters, at least 12):

- **STL** (Cleveland et al. 1990, seasonal-trend decomposition using LOESS, robust version, period 4) splits each
  series into a trend, a repeating seasonal pattern and a residual. The latest quarter's residual is scored by robust
  z against earlier residuals; |z| of 2.5 or more is reported, severity from |z| 2.5 to 5.0.
- **PELT** (Killick, Fearnhead and Eckley 2012, pruned exact linear time, via the `ruptures` library, kernel cost,
  minimum segment 3 quarters, penalty `3 × ln(n)`) finds the points where the series' level shifts. A shift in the
  last four quarters is reported with the means before and after, severity 0.45.

**Why it works.** Many ratios are seasonal: receivables peak after the holiday quarter, margins vary with product
mix. Comparing a quarter with the previous one would flag ordinary seasonality. STL removes the seasonal pattern and
the trend first, so what remains is genuinely unexpected. A level shift in a ratio that was stable for years is easier
to see in quarterly data than in annual figures, and its timing can be compared with events such as auditor changes,
CEO, CFO or chief accounting officer changes and acquisitions (listed when they fall within 400 days).

### 7.3 Peer-relative outliers, D.02

**What it is.** The robust z of each ratio against at least 5 peers;
`severity = 0.7 × clip((number of ratios with |z| ≥ 3.5 - 1) / 4)`.

**Why it works.** Any company has one or two ratios far from its peers because of its business model. Several extreme
ratios at once are harder to explain by business differences, so only clusters score.

### 7.4 Multivariate outlier ensemble, D.03

**What it is.** Three unsupervised outlier detectors applied to the ratio vectors of up to six years of each peer
(its years up to the one aligned with the focus year),
five years of the company's history and the focus year (at least 30 rows and 6 ratios with 70% coverage), after
median imputation, clipping at the 2nd and 98th percentiles, and robust scaling `(x - median) / interquartile range`:

1. **Robust Mahalanobis distance** with the Minimum Covariance Determinant estimator (Rousseeuw 1984; support
   fraction 0.8). The distance `d^2 = (x - mu_hat)^T × Sigma_hat^-1 × (x - mu_hat)` measures how far a row is from the
   centre, taking into account how ratios normally move together; mu_hat and Sigma_hat are estimated from the most
   concentrated 80% of rows, so outliers cannot distort them.
2. **Isolation Forest** (Liu, Ting and Zhou 2008; 300 trees). Each tree splits the data at random; unusual rows are
   isolated after fewer splits, so a short average path means an outlier.
3. **ECOD** (Li et al. 2022, empirical-cumulative-distribution outlier detection). For each ratio, the tail
   probability of the row's value, `F-(x) = P(X ≤ x)` or `F+(x) = P(X ≥ x)` depending on the direction the ratio is
   skewed, gives a contribution `-ln(tail probability)`; the row score is the sum. The per-ratio contributions show
   which ratios drive the result.

The three scores are converted to ranks, scaled to 0..1 and averaged; the company's percentile p gives
`severity = 0.75 × clip((p - 85) / 14)`.

**Why it works.** Manipulation rarely moves one ratio alone; it shifts a combination (receivables up, margins up,
cash conversion down). Combinations can be unusual even when each ratio is individually within range. Mahalanobis
distance captures linear relationships between ratios, Isolation Forest non-linear ones, and ECOD extreme tails per
ratio. Averaging their ranks makes the result robust to the weaknesses of any one method.

### 7.5 Structural expectation models with conformal intervals, D.04

**What it is.** A gradient-boosted regression (LightGBM: 300 trees, learning rate 0.05, 15 leaves, at least 15 rows
per leaf) that predicts what receivables, inventory and SG&A *should* be for a company with this company's economic
profile:

```
y_i = ln( balance_i / driver_i )          driver: revenue for receivables and SG&A, COGS for inventory
x_i = ( gross margin_i, asset turnover_i, ln(total assets_i) )
```

A prediction interval comes from **split conformal prediction** (Vovk, Gammerman and Shafer 2005): a seeded random
75% of the industry rows train the model and the other 25% calibrate it. The interval half-width is the 95th
percentile of absolute calibration residuals, `q = 95th percentile of |y - y_hat|`, which gives about 95% coverage
without assuming any distribution.

**Why it works.** How much receivables or inventory a company needs depends on its business model, and the model
learns that relationship from hundreds of companies: high-margin, asset-light companies carry different balances from
low-margin, asset-heavy ones. A balance outside the 95% interval for a company like this one is a quantified anomaly,
and the gap between actual and expected is a dollar impact. Conformal intervals are used because financial residuals
are not normally distributed.

**How.** Industry rows if there are 40 or more firms, else all filers; the company is excluded; at least 80 rows. For
the company: `e = ln(balance / driver) - y_hat`, expected balance `= exp(y_hat) × driver`, `z = e / (q / 1.96)`;
outside the interval means `|e| > q`. `severity = clip((|z| - 1.96) / 2)` when outside and above the prediction
(× 0.8 for SG&A); impact high = actual - expected, mid = half of that, low = actual - the upper end of the interval.
Trained models are cached by a hash of the training data.

### 7.6 Benford's law, D.05

**What it is.** Benford's law (Newcomb 1881, Benford 1938) says that in many naturally occurring sets of numbers the
leading digit d appears with probability `log10(1 + 1/d)`: 1 about 30% of the time, 9 under 5%. The test compares
the first-digit frequencies f_d of all monetary facts in the annual report (100 or more, each at least 10) with the
expected b_d:

```
MAD1 = (1/9) × sum over d = 1...9 of |f_d - b_d|
```

**Why it works.** Numbers that come from multiplying and adding many independent quantities (prices × volumes over
time) follow Benford's law; invented or adjusted numbers tend not to, because people choose digits more evenly.
Nigrini's (2012) thresholds: under 0.006 close conformity, 0.006 to 0.012 acceptable, 0.012 to 0.015 marginal, above
that nonconformity. The first-two-digit version is reported too.

**How.** `severity = 0.4 × clip((MAD1 - 0.012) / 0.012)`: deliberately low weight, because subtotals, small samples
and amounts with natural limits legitimately deviate.

### 7.7 Supervised misstatement model, D.06 (`training/`)

**What it is.** A classifier trained to recognize the ratio profile of companies in years that were later restated.

- **Labels**: companies that filed 8-K Item 4.02 non-reliance notices, with the restated fiscal year approximated
  from the filing.
- **Controls**: for each positive firm-year, 3 random companies from the same year's cross-section (case-control
  sampling).
- **Features**: the 24-ratio vector, as first reported and known by April 30 of the following year, so the model
  never sees information published later.
- **Model**: LightGBM binary classifier (300 rounds, learning rate 0.05, 15 leaves, at least 20 rows per leaf, 80%
  feature and row sampling, L2 regularization 1, positive class weighted by the ratio of negatives to positives).
- **Validation**: a walk-forward backtest (`anomaly-report train --backtest`) trains on years up to T-3 and tests on
  year T, so restatements announced later cannot leak into training; it reports ROC-AUC, precision-recall AUC and
  recall in the top 10%.

**Why it works.** Restatements leave patterns in the ratios before they are announced, and a model trained on
hundreds of cases can weigh many weak indicators together. Gradient boosting captures interactions (for example,
receivable growth matters more when margins are also rising) that linear scores miss.

**How.** Probability p against the training base rate; `severity = clip((p / base_rate - 1.5) / 3.5)`. Each feature's
contribution comes from LightGBM's built-in SHAP values. Because of case-control sampling, p ranks companies but is
not a population probability.

D.07 (which line item would be restated) needs a separate trained model and is skipped until one exists.

### 7.8 Disclosure structure change, D.08

**What it is.** The change in the number of distinct XBRL concepts tagged between the last two annual reports,
`c = (n_t - n_t-1) / n_t-1`, and canonical line items that disappeared;
`severity = 0.5 × clip((|c| - 0.2) / 0.4) + min(0.3, 0.05 × items gone)`.

**Why it works.** Dropping, merging or renaming line items can obscure trends investors were tracking.

D.09 (network features against peers' graphs) needs market-wide graphs and is skipped.

---

## 8. Family E: text and disclosure tests

### What this family does and why it works

Management writes the narrative (MD&A, risk factors, policy notes) and chooses what to emphasize. Research finds that
changes in disclosure often precede bad news: new risk factors, a shift in tone, more hedging around key accounts.
Text is also where numbers can be checked against claims. Text tests carry a moderate weight (0.30) because wording
changes for many reasons.

### 8.1 Text similarity

**What it is.** TF-IDF (term frequency-inverse document frequency) turns each passage into a vector of word and
two-word weights, `w(t, d) = tf(t, d) × ( ln((1 + N) / (1 + df(t))) + 1 )`, where tf is how often term t appears in
passage d, N is the number of passages and df(t) the number containing t (scikit-learn's smoothed form, common
English words removed, each vector scaled to length 1). Two passages are compared by the cosine of the angle between
their vectors, `cos(angle) = (u · v) / (|u| × |v|)`: 1 for identical wording, 0 for no words in common.

**Why it works.** Words that are common everywhere ("company", "revenue") get little weight and distinctive words a
lot, so similarity reflects substance rather than boilerplate.

| ID | Test | Method and severity | Why it works |
|---|---|---|---|
| E.03 | Risk-factor changes | Each Item 1A paragraph (over 80 characters) is matched with its most similar paragraph last year; similarity under 0.35 means added (or, the other way, removed). `severity = min(0.8, 0.2 × added paragraphs)` that mention material weakness, restatement, investigation, subpoena, going concern, covenants, liquidity, customer concentration, impairment, the SEC, whistleblowers or internal control | New risk factors on these topics are often the first public hint of a problem, because companies add them when they become aware of a risk |
| E.05 | MD&A tone | Uncertainty, weak-modal ("may", "could") and negative words per 1,000 words, this year against last (lists in the style of Loughran and McDonald 2011); `severity = 0.5 × clip((largest relative change - 0.30) / 0.70)` | A sharp change in tone often precedes bad news or reflects uncertainty management does not quantify |
| E.06 | Boilerplate and rewrites | Year-over-year cosine similarity of MD&A, notes and risk factors: 0.4 if MD&A similarity is under 0.6; 0.35 if over 0.97 while revenue moved more than 20% | A sudden rewrite can frame a change management wants to present; unchanged text despite a changed business is stale disclosure |
| E.07 | Hedging near key accounts | Density of hedge words in paragraphs about revenue, receivables, reserves, inventory and impairment; `severity = 0.4 × clip((change - 0.25) / 0.75)` | Vague language around the accounts that drive earnings can signal uncertainty about the numbers |
| E.08 | Critical audit matters | Count and topics in the auditor's report; `min(0.4, 0.15 × new matters)` | They show where the auditor found the most judgment |
| E.10 | SEC comment letters | Correspondence filings in the three years to the fiscal year end and the year after it (when its annual report is reviewed; the one exception to the point-in-time rule of section 2.2), the latest 6 company responses read for topics; `0.5 × clip((letter threads - 1) / 4) + 0.15` if revenue recognition is a topic | Repeated SEC questions on revenue or non-GAAP measures can precede accounting changes |
| E.13 | Missing companion disclosures | A checklist, e.g. receivables over 10% of assets but no allowance reported; `min(0.6, 0.2 × gaps)` | Material balances without their standard disclosures are hard to assess |

E.11 (earnings-call evasiveness) needs licensed transcripts and is skipped.

### 8.2 AI extraction with deterministic checking

**What it is.** Some disclosures are too varied for word lists: policy changes, estimates, management's claims,
relationships with customers and related parties. An AI model (the extraction model, by default Claude Haiku 5.5)
reads the passage and returns a fixed structure. It extracts; it never computes a number used in a decision. Every
extraction must quote the filing, and a quote that cannot be found verbatim is discarded, so the AI cannot invent
evidence.

| ID | What the AI extracts | What the code does with it, and why |
|---|---|---|
| E.01 | Changes between this year's and last year's accounting-policy notes, each with direction and whether it comes from a new standard | 0.6 for an income-increasing voluntary change, 0.25 for one required by a new standard, 0.1 otherwise. Voluntary changes that raise income are discretion; required ones are not |
| E.02 | Critical estimates and their values (useful lives, rates, assumptions) in both years | Changes of 5% or more are reported, 0.5 if income-increasing else 0.15. Changing an estimate is the quietest way to move earnings |
| E.04 | Full-year, company-wide claims in MD&A (metric, direction, stated %) | Duplicate claims are merged; each claim is checked against the reported figures, measured as the claim's own words state them: "operating expenses" as revenue less cost of sales less operating income, "costs and expenses" as revenue less operating income, "gross margin" as a percentage only when the claim speaks of a percentage or basis points and as gross profit in dollars when it names gross profit or a dollar amount (a claim that says neither is skipped). A contradiction is a wrong direction, or a stated size off by more than `2% + 10% × |actual|`; `severity = min(1, 0.45 × contradictions)`. Management's description should match the results |
| E.09 | Customers, suppliers, related parties, guarantees, factoring and variable-interest entities, with amounts or revenue shares and whose revenue a share refers to (the whole company or one segment or subsidiary) | Verified relationships become graph edges (section 9). New related-party, guarantee, VIE or side-agreement edges score `min(0.8, 0.25 × new)` |
| C.BS.03 | Numeric covenant thresholds | Compared with actual leverage and coverage |
| C.NG.03 | Non-GAAP reconciliation lines in the latest release and one a year earlier | New categories, renamed measures and asymmetric treatment are counted; a measure counts as renamed only if its core name (without qualifiers such as TTM, Y/Y growth or excluding F/X) appears nowhere in the earlier release |

AI answers are cached by a hash of the model, prompt, schema and settings, so a re-run with the same inputs gives the
same answer at no cost.

### 8.3 Earnings release vs filing, E.12

**What it is.** The revenue and net income totals in the focus year's (up to four) earnings releases, matched against
the totals later filed: the quarter each release announces and, in the fourth-quarter release, the fiscal year.
`severity = min(1, 0.4 × mismatches)`.

**Periods are never mixed.** This rule holds for every test that reads earnings releases (E.12, C.NG.01-03):

- The quarter a release announces is the latest quarter end on the company's fiscal calendar at least a week before
  the release date. A fiscal year's analysis reads only the releases announcing its own four quarters, so the next
  year's first-quarter release, filed weeks after the year end, is never read as this year's.
- Each stated figure is tagged with its period: first by its row label ("Full Year Adjusted EPS", "Q4 EPS"), then
  by the column header of its table (a header line such as "Years Ended December 31" or "Quarters Ended"; the same
  words inside a sentence are not a header). A year-only table holds full-year figures and a quarter-only table
  quarterly ones; under a quarter-and-year header a row of four values reads quarter, prior quarter, year, prior
  year. In prose, the words of the sentence decide ("fourth quarter", "full year"). Nine- and six-month figures and
  guidance are never used.
- A figure is compared only with a figure for the same period: from the release's own table, else from the filed
  statements for that period. A figure whose period cannot be told where both are possible (two values under a
  quarter-and-year header, an unlabelled row in a fourth-quarter release) is skipped rather than guessed, and when
  no same-period figure exists nothing is compared. A quarterly figure is never checked against a full-year figure,
  or the reverse.

**Why it works.** The market reacts to the press release. Numbers that change between the release and the filing
mean adjustments were made after investors were told, usually because the closing process was not finished or a
problem was found late.

### 8.4 Matching stated amounts to reported totals (`detectors/textnum.py`)

A pattern finds statements such as "total revenue was $X billion" or "net income increased ... to $X million". A
statement is **skipped** when:

- its sentence mentions segments, products, regions, guidance, non-GAAP or adjusted figures, per-share amounts,
  excluded items or (in MD&A) a quarter;
- a name stands right before the metric, as in "Data Center revenue" (a segment or product line), unless it is a
  word that can precede a company total (Record, Total, Quarterly, Fiscal and similar), or a noun that names a part
  of the company does ("GPU business revenue", "segment revenue", "services revenue");
- it is far from every reported total.

The tolerance is half a unit at the stated precision or 0.5%, whichever is larger. A statement **matches** within the
tolerance, and is a **mismatch** if it differs by more than that but by no more than 5% of the nearest total.
Revisions between a release and a filing are small, so a larger gap means a different measure, not a revision.

---

## 9. Family F: relationship-graph tests

### What this family does and why it works

Some schemes only appear when the company is seen together with its counterparties: revenue from a customer that
could not have bought that much, money that flows in a circle, customers in distress. The run builds a knowledge
graph (DuckDB tables with NetworkX views) of the company, its filings, facts and their restated versions, filing
sections and quotes, events, auditor, peers and the relationships verified in E.09. Every read is filtered by the
year's point in time (section 2.2).

| ID | Test | Method and severity | Why it works |
|---|---|---|---|
| F.01 | Sales to a major customer vs that customer's spending | For customers that file with the SEC: sales = the disclosed amount, or the revenue share × company revenue only when the share is of the whole company's revenue. The customer's spending = (its revenue - its operating income) + its capex, from its own filings for the fiscal year closest to the company's (within 183 days) and in the same currency. Sales over 50% of that spending are implausible; `severity = min(1, 0.6 × cases)` | A customer cannot buy more than it spends. Sales a customer could not plausibly have made are a red flag for fictitious revenue |
| F.02 | Circular flows | Directed cycles of up to 4 steps over sales, related-party, guarantee and side-agreement edges; `min(1, 0.5 × cycles)` | Money that flows from the company back to itself through counterparties can create revenue or cash that does not exist (round-tripping) |
| F.03 | Auditor profile | 0.45 for a large company audited by a small firm, + 0.2 per auditor change (at most 2) | A small firm may lack the resources for a complex audit; context only |
| F.04 | Unexplained movements | Material line-item movements with no passage of this year's MD&A or notes linked to them (earlier years' reports discussed earlier changes); `0.7 × clip(score / 10)` | Large changes management does not explain are disclosure gaps worth asking about |
| F.05 | Contradiction density | The number of text-vs-number tests (A.15, E.04, E.12) that found a contradiction; `min(1, 0.35 × tests)`; derived, so it does not add to the index on its own | Several independent contradictions are stronger evidence than one |
| F.06 | Customers under stress | Major customers with late-filing (NT 10-K / NT 10-Q) or non-reliance (8-K Item 4.02) filings; `min(0.8, 0.4 × customers)` | Revenue from distressed customers may not be collectible |
| F.07 | New relationships | New related-party, guarantee, VIE or side-agreement edges this year; `min(0.8, 0.3 × new)` | New arrangements with related or off-balance-sheet entities are where risks are often moved |

---

## 10. From test results to findings and the risk index

Implemented in `scoring/engine.py`. Over 100 tests run on every company, so some will fire by chance and some for
reasons the company can explain. This stage turns raw signals into a short list of findings that are corroborated,
explained away where appropriate, and controlled for multiple testing.

### 10.1 Calibration

If a calibration map has been trained for a test, its severity is remapped by piecewise-linear interpolation
through the map's points, and the remapped severity decides whether the result is flagged. This replaces the judgment-based ramps (section 1.3) with mappings learned from outcomes.
Without a map, severities are used as computed.

### 10.2 Explained-away discounts

**What it is.** Known events that legitimately move many ratios at once discount the tests they affect:

| Condition | Detected as | Tests discounted | Factor |
|---|---|---|---|
| Large acquisition | Acquisitions over 10% of prior-year assets | C.AR.01, C.INV.01, C.EXP.02, C.REV.05, D.04, B.01, A.08, C.REV.01, C.CF.02, D.02, D.03 | × 0.5 |
| New accounting standard | E.01 finds an adoption | C.REV.01, C.REV.02, C.REV.03, E.01, A.14 | × 0.6 |
| Stock split | Diluted shares up 1.8 times or more | A.12 | × 0.2 |

A discounted signal that falls below 0.5 is no longer flagged and is shown as *explained*.

**Why it works.** An acquisition brings in the target's receivables, inventory and assets at once, so working-capital
ratios jump without any change in accounting; a new standard changes revenue timing for every company; a split changes
per-share figures. These are the most common innocent causes of false alarms, and they can be detected from the data.

### 10.3 False-discovery control (Benjamini-Hochberg)

**What it is.** A correction for running many tests. Across all signals with a p-value, sorted
`p(1) ≤ p(2) ≤ ... ≤ p(m)`, the Benjamini-Hochberg (1995) adjusted value of the i-th is

```
q(i) = min over j ≥ i of  ( m × p(j) / j )
```

A flagged signal with q above 0.10 (configurable) is no longer flagged.

**Why it works.** With 50 independent tests at the 5% level, two or three would fire on a perfectly clean company.
The procedure keeps the expected share of false alarms among the flagged signals at or below 10%, while being much
less strict than correcting each test individually, so genuine signals are not lost.

### 10.4 Scheme patterns

**What it is.** Twelve templates describe common ways results get misstated: premature revenue, accrual-heavy
earnings, expense capitalization, asset-life stretching, cookie-jar reserves, cash-flow inflation, inventory
overstatement, hidden liabilities, round-tripping, acquisition masking, reporting integrity and misleading metrics.
Each lists *required* tests (at least one must be flagged) and *optional* tests that add weight; a token such as
`B.01:DSRI` matches only when the signal carries that tag (here, a Beneish score driven by receivables).

**Why it works.** A real scheme leaves several footprints. Premature revenue, for example, raises receivable days,
widens the gap between revenue and cash, and lowers cash conversion. Independent tests from different families
pointing to the same scheme are much stronger evidence than any one alone, and naming the scheme tells the reader
what to ask.

A pattern forms only when it is **corroborated**: at least 2 flagged members from at least 2 families, or at least 3
members. The one exception is reporting integrity, which may stand on a single integrity break of severity 0.8 or
more, because such breaks are rarely innocent. Pressure signals never join patterns.

**Pattern score.** A "noisy-OR": the probability that at least one member reflects a real problem, if each member is
independent evidence with 85% credibility:

```
score = 1 - product over member signals s of (1 - 0.85 × severity_s)
```

Adding members always raises the score, but with diminishing returns.

### 10.5 Standalone findings and priority

- Two or more leftover Family B signals with severity 0.6 or more fold into one "forensic scores are elevated"
  finding, scored `1 - product of (1 - 0.6 × severity)`.
- Any other flagged signal with severity 0.7 or more that is not in a pattern becomes a single-source finding, scored
  `0.85 × severity`.

**Priority** orders the findings, so that a large potential effect on profit ranks above a small one of similar
strength:

```
priority = score × (0.5 + 0.5 × min(materiality multiple, 3) / 3)    when the impact is quantified
priority = 0.5 × score                                              otherwise
```

### 10.6 The risk index

**What it is.** One number from 0 to 100 that summarizes how many warning signs were found and how strong and
corroborated they are. Contributions are summed without double counting:

| Source | Contribution |
|---|---|
| Each flagged signal not in a pattern (and not derived from other signals) | family weight × severity, the largest per test |
| Each pressure signal | 0.15 × severity |
| Each pattern | 0.8 × pattern score |

Family weights: A 0.55, C 0.45, B 0.35, F 0.35, D 0.30, E 0.30. With T the total contribution:

```
risk index = 100 × (1 - exp(-T / 2.5))
```

**Why it is built this way.** Integrity breaks weigh most because they have the fewest innocent explanations;
statistical and text signals least because they have the most. The exponential curve keeps the index below 100 and
gives each additional warning sign less weight than the one before, so many weak signals cannot add up to the same
alarm as a few strong ones. One strong, fully corroborated pattern (T about 0.8) gives about 27. Bands: Low below 25,
Elevated below 50, High below 75, Severe 75 and above, applied to the index rounded to a whole number as the report
shows it (so "25" is never labelled Low). The index is a screening score, not a probability of fraud.

After the independent audit (section 14), an unverified finding's priority and pattern contribution are halved, a
pattern the audit downgraded to a single source keeps 60% of its contribution, and the index is recomputed.

---

## 11. Materiality

Implemented in `materiality/impact.py`.

**What it is.** A measure of whether a finding's estimated dollar effect is large enough to matter, using the
benchmarks auditors use to plan an audit. The mid estimate is treated as an effect on pre-tax income (an upper bound
for balance-sheet gaps):

| Ratio | Formula | Threshold |
|---|---|---|
| Pre-tax | mid / pre-tax income | 5% |
| Revenue | mid / revenue | 0.5% |
| Assets | mid / total assets | 0.5% |
| Equity | mid / equity | 1% |

The **materiality multiple** is the largest ratio divided by its threshold (1.0 = at the threshold); the pre-tax
benchmark is preferred whenever its multiple is at least half the largest, because it is the auditor's primary
benchmark. The effect on net income is `mid × (1 - tax rate)`, on EPS that divided by diluted shares, and on market
value about the net-income effect × min(P/E ratio, 60) / public float.

The qualitative factors of SEC Staff Accounting Bulletin 99 are checked: whether the effect would turn a pre-tax
profit into a loss, turn growth in pre-tax income into a decline, or reverse an increase in EPS.

**Why it works.** A statistically striking anomaly worth $1M at a company earning $10B is not decision-relevant; a
modest one that turns a profit into a loss is. SAB 99 makes the same point: an amount below the numeric threshold can
still be material if it changes a trend or meets a target.

---

## 12. The AI investigation

**What it is.** For the top findings (default 3: the highest-ranked corroborated patterns, or the highest-ranked
findings when no pattern formed), a state machine built with LangGraph
(`agent/graph.py`) runs a structured investigation. The reasoning model (by default Claude Opus 5.5) forms hypotheses,
plans what evidence would confirm or refute them, weighs innocent explanations and writes the conclusion.

**Why it is designed this way.** A language model is good at proposing explanations and reading text, and unreliable
at arithmetic and at remembering exact figures. The design therefore gives the AI judgment and none of the numbers:
every number comes from deterministic tools, every claim is checked against sources, and a separate auditor reviews
the result. Every step is saved, so a run can be paused, reviewed by a person and resumed.

```
intake -> hypothesize -> plan evidence -> [plan review] -> gather (parallel, per hypothesis)
   -> test innocent explanations -> [closed as explained] or quantify -> draft -> verify -> audit
   -> [sensitive-wording review] -> [publish review] -> done
           correction loop: a failed check returns to "draft" (at most 3 rounds), then "degrade"
```

1. **Hypothesize**: 1-3 hypotheses about which reporting choices could produce the signals, each with the innocent
   explanations that must be ruled out.
2. **Plan evidence**: up to 12 tool calls. The tools are read-only and deterministic: `get_fact(item)`,
   `ratio(numerator, denominator)`, `peer_stats(ratio)`, `search_filing(query, section)`, `explaining_text(item)`,
   `get_events(types)` and `diff_risk_factors()`. In interactive mode a person approves the plan when the maximum
   severity is 0.8 or more, the materiality multiple is 1 or more, or a hypothesis has confidence below 0.5.
3. **Gather**: the tools run, grouped by hypothesis, in parallel; each result enters an evidence ledger with an ID.
4. **Test innocent explanations**: each is judged strictly from the ledger. A "supported" verdict without a ledger
   citation is overruled. If any explanation is supported and strong, the case closes as *explained* and the
   finding's score and its ranking weight are halved. This step exists because the most valuable output of an investigation is often a
   documented innocent explanation.
5. **Draft**: a summary and explanation using only numbers from the facts table, the test results or the ledger.
   The facts table is computed by code so the writer never does arithmetic: each test's result, its reference value
   and the gap between them; the cited line items with their year-on-year changes (in dollars and percent); the
   company's key figures and ratios; the materiality figures. Cash-flow change lines carry their sign convention
   ("positive = receivables rose, which used cash"), and the fiscal-year end dates and lengths (52 or 53 weeks) are
   given, as is each test's definition (how its result and reference value are computed), so neither the writer
   nor the auditor has to guess how a figure was built. The writer is told to claim only what the evidence shows: a search that found nothing is "not found in the
   text searched", an alternative is "refuted" or "supported" only when evidence says so (otherwise "plausible but
   untested"), and a peer comparison covers one year.
6. **Verify**: the grounding gate (section 13) and the neutral-language linter.
7. **Audit**: an independent AI auditor sees only the conclusions and an evidence pack built by code, and rules on
   each claim (supported, overstated, needs context, unsupported). A claim that is unsupported or overstated sends the
   draft back with the auditor's corrections. A "needs context" note asks for a finer qualification of a claim the
   evidence does support; it is recorded with the audit and passed on if the draft goes back anyway, but does not block
   publication on its own (otherwise the loop chases ever-finer caveats until its rounds run out). The auditor's
   evidence pack contains the same figures table and fiscal-year dates the writer was allowed to use, so sourced
   statements can be checked.

Each failed check is one round; after 3 rounds the case is *degraded* and the report says the investigation did not
finish. Budgets cap each step's prompt size, output, time and cost (`agent/budget.py`); the whole run has a cost cap
(`--budget-usd`, default $6, measured at API prices even on a subscription), and calls running at once reserve their
projected cost before they start, so together they cannot pass it. A step that hits a temporary AI error is attempted
up to 3 times with increasing delays (the API client also retries a failed request up to 3 times); budget, refusal and
validation errors are not retried. The grounding gate checks everything of a draft that reaches the report: the
summary, the explanation, the innocent explanations and the questions; when a case closes as explained, the reasoning
of the supporting explanation passes the same gate. All steps of a case send their prompts in the same
order (finding, hypotheses, evidence, alternatives and figures, then the step's instruction), so the API backend can
reuse the shared beginning from its prompt cache.

---

## 13. The grounding gate

Implemented in `agent/grounding.py`; used for investigation drafts, the audit and the plain-language explanations.

**What it is.** A deterministic check that every number and quote in AI-written text comes from a source.

**Why it works.** The most damaging error an AI can make in a report like this is a plausible but invented figure or
quote. Checking every number against an allow-list, with a tolerance for rounding, catches these mechanically; the
draft is sent back with the exact problems listed.

**Numbers.** Every number in the text (with $, %, billion, million, thousand, B, M, K, bps, x or days) is extracted.
Years, dates (including written dates such as "September 28, 2024"), calendar lengths ("52-week year"), single-digit counts (but not "$5"), evidence and test IDs, fiscal labels and references to standards or document parts
("ASC 842", "Topic 606", "Note 5", "Rule 10b5-1") are ignored. A number c is supported if, for some allowed value a and
scale k of 1, 100 or 0.01 (percent vs fraction):

```
|c - k·a| ≤ max( 1.5% × |k·a|,  eps,  ½ × unit × 10^(-decimals shown) )
eps = 0.006 when |k·a| < 10, otherwise 0.5
```

so "$1.9B" matches 1,940,000,000 at the precision shown. Allowed values: the facts table (test inputs, the company's
key figures and ratios, materiality figures), numbers returned by the tools, numbers inside quoted filing text in the
ledger, and numbers in the tests' own result sentences.

**Quotes.** Quotation marks are paired in order (re-synchronizing when a closing mark is followed by punctuation),
evidence references are removed, and each quote of four words or 25 characters or more must be found verbatim in a filing (after
normalizing whitespace, dashes and quote styles), or in the tests' own result sentences or labels. Long quotes are
checked on their first 300 characters.

**Language.** A linter rewrites accusatory wording into neutral terms and reports what it changed: "fraud" becomes
"inconsistent with expectations", "manipulation" becomes "management of reported results", "cooking the books"
becomes "aggressive accounting", and "embezzlement", "scam", "lie" and "criminal" become "irregularity",
"inconsistent statement" or "irregular". The disclaimer sentences are exempt. An anomaly is a reason to ask
questions, not proof of wrongdoing, and the report's language must say so.

---

## 14. The independent audit

**What it is.** A deterministic audit, with a correction loop of at most 3 rounds, of every reported finding (up to
12), whether or not an AI investigation ran (`audit/deterministic.py`). An unverified finding drops in the ranking,
which can bring the next finding into the 12 reported; that finding is audited too.

| Check | Passes when |
|---|---|
| Evidence present | At least one primary XBRL fact or filing quote supports the finding |
| Fact values | Every cited fact is re-read from the filing's XBRL for the same concept and period, and matches within the larger of 0.05% and 0.5 |
| Quotes | Every quote is found verbatim at its position in the filing |
| Impact | low ≤ mid ≤ high, mid is positive, and materiality recomputes to the same values |
| Corroboration | Patterns still have 2 or more live signals from 2 or more families (or 3 or more) |
| Narrative | The narrative passes the grounding gate |
| Investigation text | The innocent explanations and questions the investigation wrote pass the grounding gate (those that do not are removed) |

Failures are corrected automatically: unverifiable facts are removed, quotes re-anchored or removed, primary evidence
attached from member signals, the impact range reordered, a pattern that lost corroboration downgraded to a single
source (score × 0.6), and an ungrounded narrative replaced by one generated from the test outputs. The outcome is
*verified*, *verified after corrections* or *unverified*; unverified findings count for half (section 10.6).

**Why it works.** It re-derives every conclusion from the original sources independently of how the finding was
produced, so an error anywhere upstream (a mis-tagged fact, a stale quote, a pattern whose members were later
explained away) is caught before the report is written.

---

## 15. Plain-language explanations

Implemented in `report/explain.py`. For each reported finding, the writing model (by default Claude Sonnet 5.5)
writes three short paragraphs for a reader with basic accounting knowledge: what was found, why it is unusual (the
benchmark it departs from), and what it could mean financially. It writes from a fixed bundle: the test results with
each test's catalogue description, the company's key figures, the estimated impact, and the investigation's
conclusion and the innocent explanations it considered. The text must pass the grounding gate (every number from
that bundle) and the linter; one retry with the gate's feedback is allowed, after which standard text written for
each accounting area is used. The same safeguards as the investigation apply because this is the text a client reads
first.

---

## 16. Multi-year review

Implemented in `pipeline/span.py` and `scoring/span.py`; run with `--span N`.

**What it is.** One report for the last N fiscal years. Each year is analysed exactly as in a single-year run (its own
peer group, its own annual-report text, the history available up to that year, figures as first reported, and only
what was public by the end of that year's filing season, section 2.2), and the
findings of all years are then merged into anomalies tracked across the period.

**Why it works.** A single year's warning sign is often noise or a one-off event; the same sign returning year after
year is much harder to explain innocently, and a sign that appears only after a change (a new CFO, an acquisition) is
informative by its timing. Using figures as first reported keeps each year's test on what investors saw at the time:
a later restatement would otherwise erase the very anomaly it corrected.

**Merging.** Findings group by a key:

```
key = the pattern id, for a scheme pattern
key = a pattern the test is required for, when that pattern is reported in any year of the period
key = the test id, otherwise
```

so a test that stood alone in one year (not yet corroborated) joins the pattern it later became part of. Within a
merged anomaly every year's test results and evidence are kept (newest first); the most recent year it was flagged in
*represents* it: its impact, materiality, questions and context are used, and its investigation and re-check run there.

**Scores.** With `peak` the strongest yearly score, `k` the years flagged and `N` the years reviewed (a year counts in
`k` when it adds a result of its own: a result about an earlier year that a later year's tests see again, such as the
same revised comparative in A.14, is one event):

```
score    = peak + (1 - peak) x 0.5 x (k - 1) / (N - 1)
priority = score x (0.5 + 0.5 x min(largest materiality multiple, 3) / 3) x recency
recency  = 1 if flagged in one of the last 3 years reviewed, else 0.85
```

A sign flagged once keeps its single-year score; one flagged every year closes half the distance to 1. Recency gives
weight to what is still happening without dropping older episodes. An anomaly the independent re-check cannot confirm
counts for half, and one the investigation explains has its score halved, as in a single-year run.

**Period risk index.**

```
period index = mean(yearly index) + 0.5 x (max(yearly index) - mean(yearly index))
```

It sits halfway between the average year and the worst year, so one bad year is neither averaged away nor allowed to
define the whole period. The yearly indices are the screening scores of each year before any investigation.

**Investigation.** The AI investigates the top 5 merged anomalies (`--max-cases`). Each case runs in the representative
year's context with three additions: the finding lists every flagged year's results with their years and a strength
timeline; the tools return figures for every year up to the representative year; and the annual reports and earnings
releases of the earlier years are included, so quotes from those years can be found and checked. The facts table and the auditor's evidence pack both carry the lead test's result and severity in every
year reviewed, flagged or not, and every year's test results with their year and severity, so statements about the
anomaly's history can be checked. The grounding gate,
the independent auditor and the plain-language writer work unchanged; the writer is asked to say in which years the
anomaly appeared and whether it persists.

**Limits.** The earliest years have less history, so tests that need several prior years cannot run in them. Peer groups
are re-selected each year and thin out in early years. A change in what a company discloses (rather than in its
accounts) can recur across years too; the investigation is the step that separates the two.

---

## 17. Known approximations

- **Peer baselines are approximately point in time.** SEC frames carry peers' latest values, so a peer's later
  restatement can leak into a baseline. The company under review always uses its own first-reported values.
- **Segment data is unavailable.** The companyfacts data has no dimensional facts, so segment reconciliation (A.11)
  and segment growth (C.REV.06) are skipped.
- **Weak labels for D.06.** An 8-K Item 4.02 notice shows that a restatement happened, but the restated year is
  approximated, and case-control sampling inflates the base rate.
- **Published coefficients are reused.** The Beneish and Dechow coefficients come from their original samples
  (1982-1992 and 1979-2002). They rank risk well but are not recalibrated to today's companies.
- **Market value is a proxy.** The public float on the annual report's cover page stands in for market value
  (Altman Z, C.MA.02, materiality).
- **Severity ramps are judgment-based.** The linear maps from a statistic to severity encode judgment about where
  concern starts and where it saturates. A calibration map (section 10.1) can replace them once labelled outcomes
  exist.
- **The risk index is not a probability.** It summarizes the number, strength and corroboration of warning signs.

---

## References

- Altman, E. (1968). Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. *Journal of
  Finance*.
- Beneish, M. (1999). The detection of earnings manipulation. *Financial Analysts Journal*.
- Benford, F. (1938). The law of anomalous numbers. *Proceedings of the American Philosophical Society*.
- Benjamini, Y. and Hochberg, Y. (1995). Controlling the false discovery rate. *Journal of the Royal Statistical
  Society, Series B*.
- Burgstahler, D. and Dichev, I. (1997). Earnings management to avoid earnings decreases and losses. *Journal of
  Accounting and Economics*.
- Cleveland, R., Cleveland, W., McRae, J. and Terpenning, I. (1990). STL: a seasonal-trend decomposition procedure
  based on loess. *Journal of Official Statistics*.
- Cressey, D. (1953). *Other People's Money*.
- Dechow, P. and Dichev, I. (2002). The quality of accruals and earnings. *The Accounting Review*.
- Dechow, P., Ge, W., Larson, C. and Sloan, R. (2011). Predicting material accounting misstatements. *Contemporary
  Accounting Research*.
- Dechow, P., Sloan, R. and Sweeney, A. (1995). Detecting earnings management. *The Accounting Review*.
- Jones, J. (1991). Earnings management during import relief investigations. *Journal of Accounting Research*.
- Killick, R., Fearnhead, P. and Eckley, I. (2012). Optimal detection of changepoints with a linear computational
  cost. *Journal of the American Statistical Association*.
- Kothari, S., Leone, A. and Wasley, C. (2005). Performance matched discretionary accrual measures. *Journal of
  Accounting and Economics*.
- Leuz, C., Nanda, D. and Wysocki, P. (2003). Earnings management and investor protection. *Journal of Financial
  Economics*.
- Li, Z., Zhao, Y., Hu, X., Botta, N., Ionescu, C. and Chen, G. (2022). ECOD: unsupervised outlier detection using
  empirical cumulative distribution functions. *IEEE Transactions on Knowledge and Data Engineering*.
- Liu, F., Ting, K. and Zhou, Z. (2008). Isolation forest. *IEEE International Conference on Data Mining*.
- Loughran, T. and McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and
  10-Ks. *Journal of Finance*.
- McNichols, M. (2002). Discussion of the quality of accruals and earnings. *The Accounting Review*.
- Montier, J. (2008). Cooking the books, or, more sailing under the black flag. *Mind Matters*, Société Générale.
- Nigrini, M. (2012). *Benford's Law: Applications for Forensic Accounting, Auditing, and Fraud Detection*.
- Piotroski, J. (2000). Value investing: the use of historical financial statement information to separate winners
  from losers. *Journal of Accounting Research*.
- Richardson, S., Sloan, R., Soliman, M. and Tuna, I. (2005). Accrual reliability, earnings persistence and stock
  prices. *Journal of Accounting and Economics*.
- Rousseeuw, P. (1984). Least median of squares regression. *Journal of the American Statistical Association*.
- Roychowdhury, S. (2006). Earnings management through real activities manipulation. *Journal of Accounting and
  Economics*.
- Sloan, R. (1996). Do stock prices fully reflect information in accruals and cash flows about future earnings?
  *The Accounting Review*.
- Vovk, V., Gammerman, A. and Shafer, G. (2005). *Algorithmic Learning in a Random World*.
