"""Compare company-wide totals stated in text (MD&A, earnings releases) with reported XBRL totals.

Only total revenue / net sales / net income statements are checked, and sentences about segments,
products, regions, guidance or non-GAAP measures are skipped. A stated figure counts as a mismatch only
when it is close to a known total (within 5%) but not equal to it within rounding: revisions between a release and
the filing are small, so a figure further away is a different metric (a segment, a product line), not a contradiction.
Figures named by a qualifier ("Data Center revenue", "Gaming revenue") are segment figures and are skipped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

AMOUNT_RE = re.compile(
    r"(?P<what>total net sales|total net revenues?|total revenues?|net sales|net revenues?|revenues?|net income)\s+"
    r"(?:was|were|of|totaled|totaling|amounted to|(?:increased|decreased|grew|declined|rose|fell)[^$€£¥.]{0,60}?to)\s+"
    r"(?:[A-Z]{0,3}\$|€|£|¥|₹|₩|(?:EUR|GBP|JPY|CHF|CAD|AUD|SEK|DKK|NOK|CNY|RMB|HKD|INR|KRW|BRL)\b)\s?"
    r"(?P<num>[\d,]+(?:\.\d+)?)\s*(?P<scale>billion|million)",
    re.IGNORECASE,
)
SKIP_CONTEXT = re.compile(
    r"(?i)segment|product|service[s]? revenue|subscription|region|americas|europe|asia|international|domestic|"
    r"guidance|outlook|expect|forecast|non-gaap|adjusted|organic|constant currency|backlog|per share|"
    r"managed|taxable.equivalent|\bFTE\b|segment revenues|fee.related|distributable|"
    r"excluding|significant items?|notable items?|"
    r"\bq[1-4]\b|first quarter|second quarter|third quarter|fourth quarter|quarter ended|three months|six months|nine months"
)


# Words that may stand right before "revenue" in a company-wide total; any other capitalised word names a segment or
# product ("Record Data Center revenue of $75.2 billion" sits next to "Record revenue of $81.6 billion").
TOTAL_LEAD = {"record", "total", "quarterly", "fiscal", "full", "full-year", "annual", "year", "gaap", "consolidated", "net",
              "the", "our", "its", "company", "group", "and", "with", "of", "a", "an"}


# Nouns that name a part of a company ("GPU business revenue", "Compute segment revenue"): a total named by one of them
# is that part's total, whatever word precedes it
PART_NOUNS = {"business", "businesses", "segment", "segments", "division", "divisions", "unit", "units", "platform",
              "platforms", "product", "products", "service", "services", "subscription", "subscriptions", "line", "lines",
              "category", "categories", "brand", "brands", "region", "regions", "market", "markets", "channel", "channels"}


def qualified(text: str, start: int) -> bool:
    """True when the matched total belongs to a part of the company: a name stands right before it (a segment, region
    or product, "Greater China", "iPhone"), or a noun that names a part ("GPU business revenue", NVIDIA FY2019)."""
    words = re.findall(r"[A-Za-z][\w&'-]*", text[max(0, start - 40): start])
    if not words:
        return False
    w = words[-1]  # the word right before "revenue" / "net income"
    if w.lower() in PART_NOUNS:
        return True
    allowed = w.lower() in TOTAL_LEAD or re.search(r"quarter|year|month", w.lower())
    return any(ch.isupper() for ch in w) and not allowed


@dataclass
class AmountCheck:
    text: str
    item: str
    stated: float
    nearest: float
    start: int


QUARTER_WORDS = re.compile(r"(?i)\bq[1-4]\b|first quarter|second quarter|third quarter|fourth quarter|quarter(?:ly)?\b|three months")
YEAR_WORDS = re.compile(r"(?i)full[- ]year|fiscal (?:year|20\d\d)|twelve months|year ended|for the year|annual|\bfy\s?\d")


def statement_period(sentence: str) -> str | None:
    """"Q" for a quarterly figure, "FY" for a full-year one, None when the sentence does not say (or says both)."""
    q, y = bool(QUARTER_WORDS.search(sentence)), bool(YEAR_WORDS.search(sentence))
    return "Q" if q and not y else "FY" if y and not q else None


YEAR_IN_TEXT_RE = re.compile(r"\b(?:fiscal(?: year)?\s*|FY\s?)?((?:19|20)\d{2})\b", re.IGNORECASE)


def check_amounts(text: str, known: dict, quarter_ok: bool = False) -> tuple[int, list[AmountCheck]]:
    """Return (number checked, mismatches). `known` maps REVENUE / NET_INCOME to candidate reported totals, in one of
    three forms:
    - {"Q": [...], "FY": [...]} when the text can state a quarter or a year. A stated figure is only ever compared with
      totals for its own period; when both are possible and the sentence does not say which, the figure is skipped,
      so a quarterly figure is never checked against a full-year total or the reverse;
    - {"years": {2024: total, 2023: total, ...}, "focus": 2024} for an annual report. A sentence that names one of
      those fiscal years is compared with that year's total only. A sentence that names none is checked against every
      year, but can only be a mismatch against the focus year, and only when it is nearer the focus year's total than
      any other year's: being near an earlier year's total says nothing about which year the figure describes. A
      sentence naming several years is skipped;
    - a plain list of totals (any of them may match)."""
    checked, bad = 0, []
    for m in AMOUNT_RE.finditer(text):
        lo = max(0, text.rfind(".", 0, m.start()) + 1)
        sentence = text[lo: m.end() + 40]
        ctx = SKIP_CONTEXT.search(sentence)
        if ctx and not (quarter_ok and re.search(r"(?i)quarter|three months", ctx.group(0))):
            continue
        if qualified(text, m.start()):
            continue
        item = "NET_INCOME" if "income" in m.group("what").lower() else "REVENUE"
        pool = known.get(item, [])
        mismatch_pool = None
        if isinstance(pool, dict) and "years" in pool:
            # the years named before the figure ("Revenue for fiscal year 2019 ... was $X"); a year after it is usually the
            # comparison ("up 21% from fiscal 2018")
            named = {int(y) for y in YEAR_IN_TEXT_RE.findall(text[lo: m.end()])} & set(pool["years"])
            if len(named) > 1:
                continue
            if named:
                pool = [pool["years"][named.pop()]]
            else:
                mismatch_pool = [pool["years"].get(pool["focus"])]
                pool = list(pool["years"].values())
        elif isinstance(pool, dict):
            period = statement_period(sentence)
            if period is None:
                present = [k for k, v in pool.items() if v]
                if len(present) != 1:
                    continue  # either period is possible and the sentence does not say which
                period = present[0]
            pool = pool.get(period, [])
        cands = [k for k in pool if k and k > 0]
        if not cands:
            continue
        unit = 1e9 if m.group("scale").lower() == "billion" else 1e6
        num = float(m.group("num").replace(",", "")) * unit
        decimals = len(m.group("num").split(".")[1]) if "." in m.group("num") else 0
        tol = max(0.5 * unit / 10**decimals, 0.005 * num)
        nearest = min(cands, key=lambda k: abs(num - k))
        if abs(num - nearest) <= tol:
            checked += 1
            continue
        if mismatch_pool is not None:
            focus_total = [k for k in mismatch_pool if k and k > 0]
            if not focus_total or nearest != focus_total[0]:
                continue  # nearer another year's total than the focus year's: which year it describes is not known
        if abs(num - nearest) / nearest <= 0.05:  # same metric, revised number
            checked += 1
            bad.append(AmountCheck(m.group(0), item, num, nearest, m.start()))
    return checked, bad
