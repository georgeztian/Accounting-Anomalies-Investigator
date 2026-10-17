"""Grounding gate: every number in generated text must match a known value; every quote must exist."""

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


ID_RE = re.compile(r"\b[A-F](?:\.[A-Z]{2,4})?\.\d{2}\b|\[L\d+\.\d+\]|\bFY\d{4}(?:Q[1-4])?\b|\bCY\d{4}\b")
# references to standards, rules and document parts ("ASC 842", "Topic 606", "IFRS 16", "Rule 10b5-1", "Note 5")
REF_RE = re.compile(r"\b(?:ASC|ASU|Topic|IFRS|IAS|SFAS|FAS|SAB|Rule|Section|Item|Items|Note|Notes|Exhibit|Form|Regulation|"
                    r"Level|Tier|Class|Series|Phase)\s+[0-9][0-9A-Za-z\-–.()]*", re.IGNORECASE)
LEDGER_RE = re.compile(r"\s*\[L\d+(?:\.\d+)?\]")
# calendar descriptions: "52/53-week fiscal year", "53-week year", "13-week quarter"
CALENDAR_RE = re.compile(r"\b\d{1,2}(?:\s*/\s*\d{1,2})?[- ]weeks?\b", re.IGNORECASE)


def strip_dates(text: str) -> str:
    """Dates, detector ids (A.14, C.REV.01), ledger ids, fiscal labels, calendar descriptions and standard references are
    not financial figures."""
    return REF_RE.sub(" ", CALENDAR_RE.sub(" ", ID_RE.sub(" ", DATE_RE.sub(" ", text))))


def numbers_in(text: str) -> list[float]:
    """Every amount written in a passage (detector rationales, quoted filing text), for the allow-list."""
    out: list[float] = []
    for m in NUM_RE.finditer(strip_dates(text or "")):
        out += [abs(c) for c in _candidates(m)]
    return out


def quotes_in(text: str) -> list[str]:
    """Quoted passages, pairing quote marks in order so text between two quotes is never taken for a quote."""
    text = LEDGER_RE.sub("", text)
    out, open_at = [], None
    for m in re.finditer(r"[\"“”]", text):
        ch = m.group(0)
        if open_at is None:
            if ch != "”":
                open_at = m.end()
        elif ch != "“":
            q = text[open_at:m.start()].strip()
            if q[:1] in ".,;:)":
                open_at = m.end()  # out of step (an unmatched mark earlier): this mark opens a quote, it does not close one
                continue
            if len(q) >= 25:
                out.append(q[:300])  # a long quote is checked on its opening words
            open_at = None
    return out


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


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[\"'“”‘’]", "'", s)).strip().lower()


def check(text: str, allowed: list[float], docs: list, rel_tol: float = 0.015, known_text: list[str] | None = None) -> GroundingReport:
    """`known_text`: other verified sources a draft may quote, such as the detectors' own result sentences and labels."""
    rep = GroundingReport(ok=True)
    # sizes are compared without sign: prose carries direction in words ("3.5 points slower" for a -3.5 point gap,
    # "$0-$174.5M" for a range), so a figure is supported when its magnitude matches a source value
    allowed_abs = [abs(a) for a in allowed]
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
        found = False
        for c in (abs(x) for x in cands):
            for a in allowed_abs:
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
    known = [_norm(k) for k in known_text or []]
    for q in quotes_in(quotes_src):
        if any(_norm(q).strip(" .,;:") in k for k in known):
            continue
        if not any(locate(d, q) for d in docs):
            rep.unverified_quotes.append(q[:80])
    rep.ok = not rep.unsupported_numbers and not rep.unverified_quotes
    return rep
