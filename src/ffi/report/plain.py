"""Client-facing language for the report.

The rendered report is read by people with basic accounting and corporate-finance knowledge. This module:
- gives every test a plain name and removes internal test codes, file names and identifiers;
- spells out abbreviations and technical terms, with a short explanation the first time each appears in a passage;
- rewrites technical skip reasons and run notes.
The machine-readable report.json keeps the internal identifiers for other tools.
"""

from __future__ import annotations

import re
from typing import Any

from ffi.detectors.base import REGISTRY

ID_RE = re.compile(r"\b[A-F]\.(?:[A-Z]{2,4}\.)?\d{2}\b")

PLAIN_NAMES: dict[str, str] = {
    "A.01": "Balance sheet balances", "A.02": "Subtotals add up", "A.03": "Cash reconciles", "A.04": "Equity reconciles",
    "A.05": "Retained earnings reconcile", "A.06": "Quarters add up to the year", "A.07": "Income statement ties out",
    "A.08": "Growth in property and equipment explained by spending", "A.09": "Bad-debt reserve reconciles",
    "A.10": "Goodwill reconciles", "A.11": "Segments add up to the total", "A.12": "Earnings per share are consistent",
    "A.13": "Tax expense vs taxes actually paid", "A.14": "Prior-year figures changed later",
    "A.15": "Management's commentary matches the numbers", "A.16": "Consistent rounding",
    "B.01": "Beneish M-score (earnings-manipulation profile)", "B.02": "Dechow F-score (misstatement profile)",
    "B.03": "Unexplained accruals compared with the industry", "B.04": "How well accruals turn into cash",
    "B.05": "Sloan ratio (earnings not backed by cash)", "B.06": "Montier C-score (warning-sign checklist)",
    "B.07": "Piotroski F-score (financial strength)", "B.08": "Altman Z-score (financial distress)",
    "B.09": "Results landing just above targets", "B.10": "Unusually smooth earnings",
    "C.REV.01": "Revenue growth vs cash collected", "C.REV.02": "Billings vs revenue", "C.REV.03": "Revenue not yet billed",
    "C.REV.04": "Revenue concentrated at year-end", "C.REV.05": "Revenue vs asset base", "C.REV.06": "Segment growth consistency",
    "C.REV.07": "Gross vs net revenue presentation", "C.REV.08": "Revenue from related parties",
    "C.REV.09": "Revenue vs operating metrics", "C.REV.10": "Revenue vs contracted backlog", "C.REV.11": "Unusual sales terms",
    "C.AR.01": "Time taken to collect from customers", "C.AR.02": "Bad-debt reserve coverage",
    "C.AR.03": "Bad-debt charge below write-offs", "C.AR.04": "Selling receivables to raise cash",
    "C.AR.05": "Long-term amounts owed by customers",
    "C.INV.01": "Time inventory is held", "C.INV.02": "Inventory growing faster than sales",
    "C.INV.03": "Margins up while inventory builds", "C.INV.04": "Reserve for obsolete inventory",
    "C.INV.05": "Release of old low-cost inventory (LIFO)",
    "C.EXP.01": "Development costs recorded as assets", "C.EXP.02": "Growth in other ('soft') assets",
    "C.EXP.03": "Longer assumed asset lives", "C.EXP.04": "Depreciation rate", "C.EXP.05": "Capital spending vs depreciation",
    "C.EXP.06": "Cuts to overhead and research spending", "C.EXP.07": "Recurring 'one-time' charges",
    "C.EXP.08": "Pension assumptions", "C.EXP.09": "Stock-based pay",
    "C.RES.01": "Reserves released into profit", "C.RES.02": "Warranty reserve vs claims",
    "C.RES.03": "Release of deferred-tax allowance", "C.RES.04": "Large write-offs before a rebound ('big bath')",
    "C.RES.05": "Falling tax rate", "C.RES.06": "Lawsuits vs legal reserve",
    "C.CF.01": "Operating cash flow vs net income", "C.CF.02": "Cash flow driven by one working-capital item",
    "C.CF.03": "Paying suppliers later", "C.CF.04": "Costs moved into investing cash flow",
    "C.CF.05": "Cash flow boosted by acquisitions", "C.CF.06": "Free cash flow vs EBITDA",
    "C.CF.07": "Debt lowered just before year-end",
    "C.BS.01": "Interest cost vs reported debt", "C.BS.02": "Off-balance-sheet commitments",
    "C.BS.03": "Closeness to loan covenants", "C.BS.04": "Loss-making affiliates",
    "C.MA.01": "Acquisition price assigned to goodwill", "C.MA.02": "Goodwill not written down despite low market value",
    "C.MA.03": "Growth bought through acquisitions", "C.MA.04": "Gains from reduced earn-out payments",
    "C.MA.05": "Later changes to acquisition accounting",
    "C.GOV.01": "Change of auditor", "C.GOV.02": "Restatements and amended reports", "C.GOV.03": "Senior executive changes",
    "C.GOV.04": "Late filings", "C.GOV.05": "Internal-control weaknesses", "C.GOV.06": "Doubt about the company's survival",
    "D.01": "Breaks in quarterly trends", "D.02": "Ratios far from peers", "D.03": "Overall financial profile unusual vs peers",
    "D.04": "Balances vs what the industry would predict", "D.05": "Digit-pattern test (Benford's law)",
    "D.06": "Machine-learning restatement score", "D.07": "Line item most likely to be restated",
    "D.08": "Changes in reported line items", "D.09": "Unusual network of relationships",
    "E.01": "Accounting policy changes", "E.02": "Changes in key estimates", "E.03": "Changes in risk disclosures",
    "E.04": "Management's claims vs reported numbers", "E.05": "Change in tone of management's commentary",
    "E.06": "Rewritten or copy-pasted disclosures", "E.07": "Vague language about key accounts",
    "E.08": "Auditor's critical audit matters", "E.09": "Related parties and side arrangements",
    "E.10": "SEC comment letters", "E.11": "Evasive answers on earnings calls",
    "E.12": "Earnings announcement vs filed figures", "E.13": "Missing expected disclosures",
    "C.NG.01": "Adjusted vs reported earnings per share", "C.NG.02": "Recurring 'one-time' adjustments",
    "C.NG.03": "Changes in how adjusted earnings are defined", "C.NG.04": "One-sided adjustments to earnings",
    "F.01": "Sales vs a major customer's purchases", "F.02": "Circular transactions", "F.03": "Auditor size and changes",
    "F.04": "Large changes left unexplained", "F.05": "Contradictions between text and numbers",
    "F.06": "Customers in financial difficulty", "F.07": "New related-party arrangements",
}

FAMILY_NAMES = {"A": "Accounting integrity checks", "B": "Forensic scoring models", "C": "Account-by-account checks",
                "D": "Statistical and machine-learning checks", "E": "Disclosure and text review", "F": "Relationship analysis"}

AREA_NAMES = {"m&a": "acquisitions", "non_gaap": "adjusted (non-GAAP) measures", "related_party": "related parties",
              "earnings_quality": "earnings quality", "cash_flow": "cash flow", "integrity": "reporting integrity"}

STATUS_LABELS = {"flagged": "Flagged", "normal": "Normal", "suppressed": "Explained", "skipped": "Not run",
                 "n/a": "Not applicable", "error": "Error", "ok": "Normal"}

AUDIT_CHECKS = {"evidence_present": "Supporting evidence present", "fact_values": "Figures match the filings",
                "quotes": "Quotes found word for word", "impact": "Impact and materiality recalculated",
                "corroboration": "Supported by independent tests", "narrative": "Every number in the write-up traced to a source",
                "baseline": "Peer comparison"}

# (pattern, text for the first occurrence in a passage, text for later occurrences)
TERMS: list[tuple[str, Any, str | None]] = [
    (r"\s?\(CIK \d+\)", "", None),
    (r"\bCIK \d+\b", "", None),
    (r"\b\d{10}-\d{2}-\d{6}\b", "", None),  # SEC accession numbers
    (r"\s?\(NT 10-K/10-Q\)", "", None),
    (r"\bNT (?:10-K|10-Q|20-F)\b(?: / NT 10-Q)?", "late-filing notice", None),
    (r"\s?\(8-K Item 4\.01\)", " on Form 8-K", None),
    (r"\b8-K Item 4\.02\b", "non-reliance notice (Form 8-K)", None),
    (r"\bItem 5\.02 filing\(s\)", "executive-change filing(s) (Form 8-K)", None),
    (r"\bItem 9A\b", "The internal-control section of the annual report", None),
    (r"\bItem 1A\b", "the risk-factors section", None),
    (r"\bCFO and CEO\b", "chief financial officer and chief executive", None),
    (r"\bCFO (?=exits|departure|change)", "chief financial officer ", None),
    (r"\bCFO/CAO\b", "chief financial or chief accounting officer", None),
    (r"\bRSST accruals\b", "accruals (broad measure)", None),
    (r"\bunconditional misstatement rate\b", "the average misstatement rate", None),
    (r"\bReal earnings-management composite\b", "Combined score for managing results through real business decisions", None),
    (r"\babnormal PROD\b", "abnormal production costs", None),
    (r"\babnormal DISX\b", "abnormal discretionary spending", None),
    (r"\bDSRI\b", "receivables-to-sales index", None), (r"\bGMI\b", "gross-margin index", None),
    (r"\bAQI\b", "asset-quality index", None), (r"\bSGI\b", "sales-growth index", None),
    (r"\bDEPI\b", "depreciation index", None), (r"\bSGAI\b", "overhead index", None),
    (r"\bTATA\b", "accruals-to-assets", None), (r"\bLVGI\b", "leverage index", None),
    (r"\bFCF\b", "free cash flow", None),
    (r"\bCAGR\b", "average annual growth", None),
    (r"\bCapex\b", "Capital spending", None), (r"\bcapex\b", "capital spending", None),
    (r"\bcoef\b", "model weight", None),
    (r"\(n=(\d+)\)", r"(\1 companies)", None),
    (r"\b(rsst_acc|ch_rec|ch_inv|soft_assets|ch_cs|ch_roa|issuance)\b(?= =)",
     lambda m: {"rsst_acc": "accruals", "ch_rec": "change in receivables", "ch_inv": "change in inventory",
                "soft_assets": "soft assets", "ch_cs": "change in cash sales", "ch_roa": "change in return on assets",
                "issuance": "new securities issued"}[m.group(1)], None),
    (r"\bCEO\b", "chief executive", None),
    (r"\b(?:CFO|operating cash flow)\s*/\s*NI\b", "operating cash flow / net income", None),
    (r"\(NI - CFO\) / assets", "(net income - operating cash flow) / assets", None),
    (r"\bCFO\b", "operating cash flow", None),
    (r"\bCFI\b", "investing cash flow", None),
    (r"\bCFF\b", "financing cash flow", None),
    (r"\bNI\b", "net income", None),
    (r"\bDSO\b", "days sales outstanding (DSO, the average number of days taken to collect from customers)", "DSO"),
    (r"\bDSI\b", "days inventory outstanding (DSI, the average number of days inventory is held)", "DSI"),
    (r"\bDPO\b", "days payables outstanding (DPO, the average number of days taken to pay suppliers)", "DPO"),
    (r"\bD&A\b", "depreciation and amortization", None),
    (r"\bSG&A\b", "selling, general and administrative expenses (SG&A)", "SG&A"),
    (r"\bPP&E\b", "property, plant and equipment (PP&E)", "PP&E"),
    (r"\bCOGS\b", "cost of sales", None),
    (r"\bAR\b", "receivables", None),
    (r"\bSBC\b", "stock-based pay", None),
    (r"\bOCI\b", "other comprehensive income", None),
    (r"\bNCI\b", "noncontrolling (minority) interests", None),
    (r"\bFX\b", "currency", None),
    (r"\bUTP\b", "uncertain tax positions", None),
    (r"\bEBITDA\b", "EBITDA (earnings before interest, taxes, depreciation and amortization)", "EBITDA"),
    (r"\bROA\b", "return on assets (ROA)", "ROA"),
    (r"\bRPO\b", "contracted future revenue (remaining performance obligations, RPO)", "RPO"),
    (r"\bVIEs?\b", "variable-interest entities (outside entities the company may effectively control)", "variable-interest entities"),
    (r"\bMD&A\b", "Management's Discussion and Analysis (MD&A, management's commentary in the annual report)", "MD&A"),
    (r"\bXBRL\b", "XBRL (the machine-readable financial data companies file with the SEC)", "XBRL"),
    (r"\bLIFO\b", "LIFO (last-in, first-out inventory costing)", "LIFO"),
    (r"\bnon-GAAP\b", "non-GAAP (adjusted figures that leave out some costs and are not defined by accounting standards)", "non-GAAP"),
    (r"(?<!non-)\bGAAP\b", "GAAP (the official accounting rules)", "GAAP"),
    (r"\bCAMs?\b", "critical audit matters", None),
    (r"\bAIF\b", "Annual Information Form", None),
    (r"\b10-K/A\b", "amended annual report", None),
    (r"(?<!Form )\b10-Ks?\b", "annual report (Form 10-K)", "annual report"),
    (r"\b10-Qs?\b", "quarterly report", None),
    (r"\b8-K Ex\. 99\b", "", None),
    (r"(?<!Form )\b8-K\(s\)", "current report(s) on Form 8-K", None),
    (r"(?<!Form )\b8-K\b", "current report (Form 8-K)", "Form 8-K"),
    (r"\bSIC-(\d) filers", r"companies with the same \1-digit industry code", None),
    (r"\bSIC\b", "industry code", None),
    (r"\bnearest (\d+) of (\d+) companies with the same (\d)-digit industry code on\b",
     r"\1 most similar of \2 companies in the same industry (same \3-digit industry code), matched on", None),
    (r"\bCY(\d{4})\b", r"calendar \1", None),
    (r"\blagged assets\b", "prior-year assets", None),
    (r"\bindustry firms\b", "companies in the industry", None),
    (r"\bFY(\d{4})Q([1-4])\b", r"Q\2 of fiscal \1", None),
    (r"\bFY(\d{4})\b", r"fiscal \1", None),
    (r"\|z\|\s*>=\s*([\d.]+)", r"more than \1 standard deviations from the peer median", None),
    (r"\(z ([+-]?[\d.]+)\)", r"(\1 standard deviations from the median)", None),
    (r"\b(?:residual )?z\s?=\s?([+-]?[\d.]+)", r"\1 standard deviations from its usual level", None),
    (r"\brobust z\b", "statistical distance", None),
    (r"\bcomposite z\b", "combined score", None),
    (r"\bmahalanobis, isolation_forest, ecod\b", "three independent statistical methods", None),
    (r"\bMAD\b", "average deviation", None),
    (r"Nigrini's thresholds", "standard audit thresholds", None),
    (r"\bLLM\b", "AI model", None),
    (r"\bweak modal\b", "tentative words (may, might, could)", None),
    (r"Word lists are a subset of Loughran-McDonald categories\.", "Word lists come from standard finance-research dictionaries.", None),
    (r"\bextracted\b", "identified", None),
    (r"^required data not reported$", "The figures this test needs are not reported", None),
    (r"\bnot tagged\b", "not reported separately in the financial data", None),
    (r"\bUntagged\b", "Separately unreported", None),
    (r"\buntagged\b", "separately unreported", None),
    (r"\btagged\b", "reported", None),
    (r"\bmonetary facts\b", "amounts", None),
    (r"\bresidual\b", "difference", None),
    (r"\bcompanyfacts\b", "the SEC's standardized financial data", None),
    (r"\bknowledge graph\b", "relationship map", None),
    (r"\bKG\b", "relationship map", None),
]
_TERMS = [(re.compile(p), first, later) for p, first, later in TERMS]

# Technical reasons and notes -> client language (first match wins)
REWRITES: list[tuple[str, str]] = [
    (r"needs the LLM.*", "Needs AI-assisted reading of the filings, which was not enabled for this analysis."),
    (r"No LLM available.*", "AI-assisted reading of the filings was not enabled for this analysis, so the tests that need it were not run."),
    (r"segment facts are dimensional.*|.*dimensional; not available.*", "Segment-level figures are not available in the SEC's standardized financial data."),
    (r"no trained restatement-delta model.*", "This model has not been trained yet."),
    (r"peer knowledge graphs are only built.*", "Requires market-wide data that is not part of this analysis."),
    (r"operating KPIs are company-specific.*", "Requires company-specific operating metrics that are not collected yet."),
    (r"no earnings-call transcripts supplied.*", "Earnings-call transcripts were not provided."),
    (r"dependency failed: .*", "Not run because a test it depends on could not run."),
]
_REWRITES = [(re.compile(p, re.I), r) for p, r in REWRITES]


CITE_RE = re.compile(r"\s*\[L\d+(?:\.\d+)?(?:\s*[,;]\s*L\d+(?:\.\d+)?)*\]")  # [L0.1] and [L2.1, L2.2]
SECTION_RE = re.compile(r"\bitem(1a|1b|1c|7a|9a|9b|\d{1,2})\b", re.I)
SECTION_NAMES = {"1": "business description", "1a": "risk factors", "1b": "unresolved SEC comments", "1c": "cybersecurity",
                 "2": "properties", "3": "legal proceedings", "5": "market for the shares", "7": "management's discussion and analysis",
                 "7a": "market risk disclosures", "8": "financial statements and notes", "9": "changes in accountants",
                 "9a": "controls and procedures", "9b": "other information", "10": "directors and governance",
                 "11": "executive pay", "13": "related-party transactions", "14": "audit fees", "15": "exhibits"}


def name(detector_id: str) -> str:
    spec = REGISTRY.get(detector_id)
    return PLAIN_NAMES.get(detector_id, spec.name if spec else detector_id)


_NAME_TO_ID = {}


def _names_index() -> dict[str, str]:
    if not _NAME_TO_ID:
        _NAME_TO_ID.update({s.name: i for i, s in REGISTRY.items()})
    return _NAME_TO_ID


def title(text: str) -> str:
    """A finding title: single-test findings use the test's plain name."""
    base, suffix = (text[:-16], " (single source)") if text.endswith(" (single source)") else (text, "")
    did = _names_index().get(base)
    return plain(name(did) + suffix if did else text, expand=False)


def plain(text: str | None, expand: bool = True, terms: bool = True) -> str:
    """Rewrite one passage for a client reader. `terms=False` keeps the wording as written (text the AI already wrote
    for a client, which defines its own terms) and only removes internal codes and references."""
    if not text:
        return text or ""
    for pat, rep in _REWRITES:
        if pat.fullmatch(text.strip()):
            return rep
    out = CITE_RE.sub("", text)  # the investigation's evidence-ledger references, e.g. [L0.1]
    out = SECTION_RE.sub(lambda m: f"— {SECTION_NAMES.get(m.group(1).lower(), 'section')}", out)
    out = ID_RE.sub(lambda m: f"“{name(m.group(0))}”", out)
    for internal, did in sorted(_names_index().items(), key=lambda kv: -len(kv[0])):
        if internal in out and did in PLAIN_NAMES and len(internal) > 8:
            out = out.replace(f"{internal}:", f"{PLAIN_NAMES[did]}:")
    out = re.sub(r"`[^`]*`", "", out)  # command lines and file names
    for pat, first, later in _TERMS if terms else []:
        if later is None or not expand:
            out = pat.sub(later if (later and not expand) else first, out)
        else:
            # one pass, so the first-use explanation itself is never rewritten again
            seen = [False]

            def rep(m: re.Match, first=first, later=later, seen=seen) -> str:
                if seen[0]:
                    return m.expand(later)
                seen[0] = True
                return m.expand(first)

            out = pat.sub(rep, out)
    out = re.sub(r"(?<!\.)\.\.(?!\.)", ".", out)
    out = re.sub(r"\b(\w+(?: \w+)?) or \1(?=[.,;:!?)]|$)", r"\1", out)  # rewrites can collide: "reported or tagged." -> "reported." (a repeat followed by more words, as in "cash and cash equivalents", is real text)
    out = re.sub(r"(?<![A-Za-z])([Aa])n (?=Management's)", lambda m: m.group(1) + " ", out)  # "an MD&A" once expanded
    if expand and out[:1].islower():
        out = out[:1].upper() + out[1:]
    return re.sub(r"\s{2,}", " ", out).replace(" .", ".").strip()


def area(key: str) -> str:
    return AREA_NAMES.get(key, key.replace("_", " "))


def source_label(form: str | None, filed: str | None, kind: str) -> str:
    forms = {"10-K": "Annual report", "10-K/A": "Amended annual report", "10-Q": "Quarterly report", "8-K": "Current report",
             "20-F": "Annual report (foreign issuer)", "40-F": "Annual report (Canadian issuer)", "10-KT": "Transition annual report",
             "20-F/A": "Amended annual report (foreign issuer)", "40-F/A": "Amended annual report (Canadian issuer)", "6-K": "Foreign issuer report",
             "EX-99": "Earnings release"}
    label = forms.get(form or "", "SEC filing" if kind != "derived" else "Calculation")
    return f"{label}{', filed ' + filed if filed else ''}"


GLOSSARY: list[tuple[str, str]] = [
    ("Risk index", "A 0-100 summary of how many warning signs were found and how strong and well-corroborated they are. It is a screening score, not a probability that anything is wrong."),
    ("Severity", "How unusual a single test result is, from 0 (nothing unusual) to 1 (strongest). Results at or above the flag threshold (normally 0.5) are flagged."),
    ("Flagged / Explained / Not run / Not applicable", "Flagged: unusual enough to count. Explained: unusual, but a known legitimate cause was found (for example an acquisition), so it does not count. Not run: data was missing. Not applicable: the test does not fit this type of company."),
    ("Peers and percentile", "Peers are similar companies in the same industry. The 90th percentile means the company's value is higher than 90% of its peers'."),
    ("Standard deviation", "A measure of how far a value sits from the typical value. Three or more standard deviations is rare."),
    ("Materiality", "Whether an amount is large enough to matter to an investor. Auditors commonly use about 5% of pre-tax income as a reference point."),
    ("Accruals", "Income and expenses recorded before the cash moves, such as sales on credit or bills not yet paid. They rely on management's estimates, which is why unusual accruals deserve attention."),
    ("Operating cash flow", "Cash generated by the business's day-to-day operations in the year. Over time it should track net income."),
    ("Working capital", "Short-term operating balances: amounts owed by customers, inventory, and amounts owed to suppliers."),
    ("Goodwill", "The part of an acquisition price above the value of the identifiable assets bought. It is not amortized but must be written down if its value falls."),
    ("Allowance for credit losses", "A reserve for amounts customers are not expected to pay."),
    ("Restatement", "A correction of previously published financial statements. A 'non-reliance' notice tells investors not to rely on them."),
    ("Non-GAAP measures", "Adjusted figures, such as 'adjusted EPS', that leave out some costs. They are not defined by accounting standards."),
    ("Annual and current reports", "Form 10-K is the US annual report, and 20-F and 40-F are its foreign equivalents. Form 10-Q is a quarterly report, and Form 8-K reports significant events."),
    ("XBRL", "The machine-readable version of the financial statements that companies file with the SEC; this analysis reads the figures from it."),
]
