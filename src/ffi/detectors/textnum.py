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


def qualified(text: str, start: int) -> bool:
    """True when a name (a segment, region or product, "Greater China", "iPhone") stands right before the matched total."""
    words = re.findall(r"[A-Za-z][\w&'-]*", text[max(0, start - 40): start])
    if not words:
        return False
    w = words[-1]  # the word right before "revenue" / "net income"
    allowed = w.lower() in TOTAL_LEAD or re.search(r"quarter|year|month", w.lower())
    return any(ch.isupper() for ch in w) and not allowed


@dataclass
class AmountCheck:
    text: str
    item: str
    stated: float
    nearest: float
    start: int


def check_amounts(text: str, known: dict[str, list[float]], quarter_ok: bool = False) -> tuple[int, list[AmountCheck]]:
    """Return (number checked, mismatches). `known` maps REVENUE / NET_INCOME to candidate reported totals."""
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
        cands = [k for k in known.get(item, []) if k and k > 0]
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
        if abs(num - nearest) / nearest <= 0.05:  # same metric, revised number
            checked += 1
            bad.append(AmountCheck(m.group(0), item, num, nearest, m.start()))
    return checked, bad
