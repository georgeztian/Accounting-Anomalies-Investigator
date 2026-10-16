"""Resolve 'Company Name (TICKER)' to exactly one SEC registrant."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd
from rapidfuzz import fuzz, process

from ffi.edgar import endpoints
from ffi.edgar.client import EdgarClient

INPUT_RE = re.compile(r"^\s*(?P<name>.*?)\s*\((?P<ticker>[A-Za-z0-9.\-]{1,10})\)\s*$")
TICKER_RE = re.compile(r"^[A-Z]{1,5}([.\-][A-Z]{1,2})?$")
LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "plc", "llc",
    "lp", "holdings", "holding", "group", "the", "sa", "nv", "ag", "se", "de", "del",
}
ANNUAL_REPORT_FORMS = {"10-K": ("10-K", "10-K405", "10-KT"), "20-F": ("20-F",), "40-F": ("40-F",)}


class ResolutionError(RuntimeError):
    """Raised when the input cannot be mapped to one eligible registrant."""

    def __init__(self, message: str, candidates: list[dict] | None = None, exit_code: int = 3):
        super().__init__(message)
        self.candidates = candidates or []
        self.exit_code = exit_code


@dataclass
class CompanyRef:
    cik: int
    ticker: str
    name: str
    sic: int | None
    sic_description: str
    fiscal_year_end: str  # MMDD
    former_names: list[str] = field(default_factory=list)
    match_note: str = ""
    warnings: list[str] = field(default_factory=list)
    annual_form: str = "10-K"  # the annual report form the company files: 10-K | 20-F | 40-F
    display_name: str = ""  # name for the client report: the user's spelling when it matches, else the SEC name in title case


@dataclass
class ParsedInput:
    name: str | None
    ticker: str | None


def parse_input(text: str) -> ParsedInput:
    m = INPUT_RE.match(text)
    if m:
        name = m.group("name").strip() or None
        return ParsedInput(name=name, ticker=normalize_ticker(m.group("ticker")))
    raw = text.strip()
    if TICKER_RE.match(raw):
        return ParsedInput(name=None, ticker=normalize_ticker(raw))
    return ParsedInput(name=raw, ticker=None)


def normalize_ticker(t: str) -> str:
    return t.strip().upper().replace(".", "-")


def normalize_name(name: str) -> str:
    s = re.sub(r"[^a-z0-9 ]+", " ", name.lower().replace("&", " and "))
    return " ".join(w for w in s.split() if w not in LEGAL_SUFFIXES)


def name_score(a: str, b: str) -> float:
    return float(fuzz.token_set_ratio(normalize_name(a), normalize_name(b)))


def resolve(text: str, client: EdgarClient, choose=None) -> CompanyRef:
    """`choose(candidates) -> index | None` is called on ambiguity in interactive mode."""
    parsed = parse_input(text)
    tickers = company_tickers_frame(client)
    warnings: list[str] = []

    cik: int | None = None
    note = ""
    if parsed.ticker:
        hit = tickers[tickers["ticker"] == parsed.ticker]
        if hit.empty and "-" in parsed.ticker:
            hit = tickers[tickers["ticker"] == parsed.ticker.replace("-", "")]
        if not hit.empty:
            row = hit.iloc[0]
            cik = int(row["cik"])
            if parsed.name:
                sub = endpoints.submissions(client, cik)
                names = [row["title"]] + [f.get("name", "") for f in sub.get("formerNames", [])]
                score = max(name_score(parsed.name, n) for n in names if n)
                note = f"ticker {parsed.ticker} -> '{row['title']}' (name match {score:.0f}/100)"
                if score < 60:
                    cands = _name_candidates(parsed.name, tickers)
                    cands.insert(0, {"cik": cik, "ticker": parsed.ticker, "title": row["title"], "score": score})
                    idx = choose(cands) if choose else None
                    if idx is None:
                        raise ResolutionError(
                            f"Ticker {parsed.ticker} belongs to '{row['title']}', which does not match "
                            f"'{parsed.name}' (score {score:.0f}).",
                            cands,
                        )
                    cik = int(cands[idx]["cik"])
                    note = f"chosen by user: {cands[idx]['title']}"
                elif score < 85:
                    warnings.append(
                        f"Name '{parsed.name}' only partly matches SEC title '{row['title']}' "
                        f"(score {score:.0f}); proceeding on ticker {parsed.ticker}."
                    )
            else:
                note = f"ticker {parsed.ticker} -> '{row['title']}'"
        elif not parsed.name:
            raise ResolutionError(f"Ticker '{parsed.ticker}' was not found in the SEC ticker list.")

    if cik is None:
        assert parsed.name
        cands = _name_candidates(parsed.name, tickers)
        if not cands:
            raise ResolutionError(f"No SEC registrant resembles '{parsed.name}'.")
        top = cands[0]
        gap = top["score"] - (cands[1]["score"] if len(cands) > 1 else 0)
        if top["score"] >= 92 and gap >= 5:
            cik = int(top["cik"])
            note = f"name '{parsed.name}' -> '{top['title']}' ({top['ticker']}, score {top['score']:.0f})"
        else:
            idx = choose(cands) if choose else None
            if idx is None:
                raise ResolutionError(f"'{parsed.name}' is ambiguous.", cands)
            cik = int(cands[idx]["cik"])
            note = f"chosen by user: {cands[idx]['title']}"

    try:
        ref = _company_ref(client, cik, tickers, note, warnings)
        ref.display_name = display_name(ref.name, parsed.name)
        return ref
    except ResolutionError as err:
        if "has not filed an annual report" not in str(err):
            raise
        # A ticker can move to a new registrant (e.g. a holding-company reorganisation) that has no annual report yet.
        # Look for the operating predecessor by name among EDGAR registrants that file annual reports.
        new_name = endpoints.submissions(client, cik).get("name", "")
        for query in [q for q in (parsed.name, new_name) if q]:
            for alt_cik, alt_name in endpoints.company_search(client, normalize_name(query) or query):
                if alt_cik == cik or name_score(query, alt_name) < 80:
                    continue
                try:
                    ref = _company_ref(client, alt_cik, tickers, f"{note}; ticker moved to '{new_name}', using '{alt_name}'", warnings)
                except ResolutionError:
                    continue
                ref.ticker = ref.ticker or parsed.ticker or ""
                ref.display_name = display_name(ref.name, parsed.name)
                ref.warnings.append(f"{parsed.ticker or query} now belongs to '{new_name}' (CIK {cik}), which has no annual report yet; "
                                    f"analysing the predecessor '{ref.name}' (CIK {alt_cik}).")
                return ref
        raise


KEEP_UPPER = {"LLC", "LP", "LLP", "PLC", "NV", "SA", "AG", "SE", "AB", "ASA", "USA", "US", "UK", "II", "III", "IV", "REIT", "ETF"}
SUFFIX_DOT = {"INC": "Inc.", "CORP": "Corp.", "CO": "Co.", "LTD": "Ltd.", "HLDGS": "Holdings", "INTL": "International"}


def display_name(sec_name: str, typed: str | None) -> str:
    """The user's spelling when it matches the SEC name; otherwise the SEC's all-caps name in title case."""
    if typed and name_score(typed, sec_name) >= 85:
        return typed.strip()
    if not sec_name.isupper():
        return sec_name
    words = []
    for w in sec_name.replace("/", " ").split():
        bare = w.rstrip(".,")
        words.append(SUFFIX_DOT.get(bare, bare if bare in KEEP_UPPER else bare.capitalize()))
    return " ".join(words)


def company_tickers_frame(client: EdgarClient) -> pd.DataFrame:
    return endpoints.company_tickers(client)


def _name_candidates(name: str, tickers: pd.DataFrame, limit: int = 5) -> list[dict]:
    titles = tickers.drop_duplicates("cik")
    choices = {i: normalize_name(t) for i, t in zip(titles.index, titles["title"], strict=True)}
    hits = process.extract(normalize_name(name), choices, scorer=fuzz.token_set_ratio, limit=limit * 3)
    out = []
    seen = set()
    for _, score, idx in hits:
        row = titles.loc[idx]
        if row["cik"] in seen:
            continue
        seen.add(row["cik"])
        # token_set_ratio rewards subsets; break ties toward the closer full-string match
        adj = 0.7 * score + 0.3 * fuzz.ratio(normalize_name(name), choices[idx])
        out.append({"cik": int(row["cik"]), "ticker": row["ticker"], "title": row["title"], "score": round(adj, 1)})
    out.sort(key=lambda c: -c["score"])
    return out[:limit]


def _company_ref(client: EdgarClient, cik: int, tickers: pd.DataFrame, note: str, warnings: list[str]) -> CompanyRef:
    sub = endpoints.submissions(client, cik)
    sic = int(sub["sic"]) if str(sub.get("sic", "")).isdigit() else None
    tick = sub.get("tickers") or tickers.loc[tickers["cik"] == cik, "ticker"].tolist() or [""]
    ref = CompanyRef(
        cik=cik,
        ticker=tick[0],
        name=sub.get("name", ""),
        sic=sic,
        sic_description=sub.get("sicDescription", ""),
        fiscal_year_end=sub.get("fiscalYearEnd") or "1231",
        former_names=[f.get("name", "") for f in sub.get("formerNames", [])],
        match_note=note,
        warnings=warnings,
    )
    check_eligibility(ref, sub, client)
    return ref


def annual_report_counts(forms: list[str]) -> dict[str, int]:
    return {k: sum(1 for f in forms if f in v) for k, v in ANNUAL_REPORT_FORMS.items()}


def check_eligibility(ref: CompanyRef, sub: dict, client: EdgarClient | None = None) -> None:
    """Any company with at least one annual report (10-K, 20-F or 40-F) on EDGAR can be analysed.

    The company type (bank, insurer, REIT, ...) selects a detector profile later; it never blocks a run.
    """
    counts = annual_report_counts(sub.get("filings", {}).get("recent", {}).get("form", []))
    if sum(counts.values()) < 2 and client is not None and sub.get("filings", {}).get("files"):
        # Prolific filers (large banks issue thousands of prospectus supplements) push older annual reports
        # out of the 'recent' block; count across the full filing history.
        counts = annual_report_counts([f.form for f in endpoints.filings_table(client, ref.cik)])
    total = sum(counts.values())
    if total == 0:
        raise ResolutionError(
            f"{ref.name} has not filed an annual report (10-K, 20-F or 40-F) on EDGAR yet, so there are no audited "
            "annual financial statements to analyse.", exit_code=4)
    ref.annual_form = max(counts, key=lambda k: counts[k])
    if total == 1:
        ref.warnings.append(f"{ref.name} has filed only one annual report; trend detectors use the prior-year comparatives "
                            "in that report and many history-based checks will be skipped.")
