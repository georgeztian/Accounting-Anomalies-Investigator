"""Grounding gate (BLUEPRINT §11): every number in generated text must match a known value; every quote must exist."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ffi.text.verify import locate

NUM_RE = re.compile(
    r"(?P<neg>-)?\$?\s?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s?(?P<suf>%|percent|billion|million|thousand|[BMK]\b|x\b|days?\b|bps\b)?",
    re.IGNORECASE,
)
YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
DATE_RE = re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/(?:19|20)\d{2}\b")
QUOTE_RE = re.compile(r"[\"“]([^\"”]{25,400})[\"”]")


ID_RE = re.compile(r"\b[A-F](?:\.[A-Z]{2,4})?\.\d{2}\b|\[L\d+\.\d+\]|\bFY\d{4}(?:Q[1-4])?\b|\bCY\d{4}\b")


def strip_dates(text: str) -> str:
    """Dates, detector ids (A.14, C.REV.01), ledger ids and fiscal labels are not financial figures."""
    return ID_RE.sub(" ", DATE_RE.sub(" ", text))


@dataclass
class GroundingReport:
    ok: bool
    numbers_checked: int = 0
    unsupported_numbers: list[str] = field(default_factory=list)
    unverified_quotes: list[str] = field(default_factory=list)


def _candidates(m: re.Match) -> list[float]:
    raw = float(m.group("num").replace(",", ""))
    suf = (m.group("suf") or "").lower()
    sign = -1 if m.group("neg") else 1
    if suf in ("billion", "b"):
        return [sign * raw * 1e9]
    if suf in ("million", "m"):
        return [sign * raw * 1e6]
    if suf in ("thousand", "k"):
        return [sign * raw * 1e3]
    if suf in ("%", "percent"):
        return [sign * raw / 100, sign * raw]
    if suf == "bps":
        return [sign * raw / 10000]
    return [sign * raw]


def check(text: str, allowed: list[float], docs: list, rel_tol: float = 0.015) -> GroundingReport:
    rep = GroundingReport(ok=True)
    quotes_src = text
    text = strip_dates(text)
    for m in NUM_RE.finditer(text):
        token = m.group(0).strip()
        num = m.group("num")
        if YEAR_RE.match(num) and not m.group("suf"):
            continue
        if re.fullmatch(r"\d", num) and not m.group("suf"):
            continue  # small counts ("two years", "3 signals") are not financial figures
        before = text[max(0, m.start() - 3): m.start()]
        if re.search(r"(?:FY|Q|\[L)$", before, re.I) or text[m.end(): m.end() + 1] == "]":
            continue  # fiscal labels and ledger ids
        rep.numbers_checked += 1
        cands = _candidates(m)
        cands += [abs(c) for c in cands if c < 0]  # "$0-$174.5M" is a range, not a negative amount
        found = False
        for c in cands:
            for a in allowed:
                for scale in (1, 100, 0.01):
                    v = a * scale
                    tol = max(abs(v) * rel_tol, 0.006 if abs(v) < 10 else 0.5)
                    # displayed values are rounded: allow rounding at the shown precision
                    shown = num.split(".")[1] if "." in num else ""
                    unit = {"billion": 1e9, "b": 1e9, "million": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3}.get((m.group("suf") or "").lower(), 1)
                    tol = max(tol, 0.5 * unit / (10 ** len(shown)))
                    if abs(c - v) <= tol:
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            rep.unsupported_numbers.append(token)
    for q in QUOTE_RE.findall(quotes_src):
        if not any(locate(d, q) for d in docs):
            rep.unverified_quotes.append(q[:80])
    rep.ok = not rep.unsupported_numbers and not rep.unverified_quotes
    return rep
