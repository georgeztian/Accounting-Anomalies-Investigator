"""Typed accessors for the SEC endpoints the app uses."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

import pandas as pd
from lxml import etree
from selectolax.lexbor import LexborHTMLParser as HTMLParser

from ffi.edgar.client import EdgarClient, EdgarError

DAY = 86_400
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
FRAMES_URL = "https://data.sec.gov/api/xbrl/frames/{taxonomy}/{concept}/{unit}/{period}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accn_nodash}/{name}"
FILING_INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accn_nodash}/{accn}-index.htm"
SIC_BROWSE_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&SIC={sic}&owner=include"
    "&count=100&start={start}&output=atom"
)
SIC_LIST_URL = "https://www.sec.gov/corpfin/division-of-corporation-finance-standard-industrial-classification-sic-code-list"


@dataclass(frozen=True)
class FilingRef:
    accession: str
    form: str
    filing_date: date
    report_date: date | None
    primary_document: str
    items: str = ""

    @property
    def accn_nodash(self) -> str:
        return self.accession.replace("-", "")


def company_tickers(client: EdgarClient) -> pd.DataFrame:
    data = client.get_json(TICKERS_URL, max_age_s=DAY)
    df = pd.DataFrame(list(data.values()))
    df = df.rename(columns={"cik_str": "cik", "title": "title", "ticker": "ticker"})
    df["cik"] = df["cik"].astype(int)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    return df[["cik", "ticker", "title"]]


def submissions(client: EdgarClient, cik: int) -> dict:
    return client.get_json(SUBMISSIONS_URL.format(cik=cik), max_age_s=DAY)


def filings_table(client: EdgarClient, cik: int) -> list[FilingRef]:
    sub = submissions(client, cik)
    blocks = [sub.get("filings", {}).get("recent", {})]
    for extra in sub.get("filings", {}).get("files", []):
        try:
            blocks.append(client.get_json(f"https://data.sec.gov/submissions/{extra['name']}", max_age_s=7 * DAY))
        except EdgarError:
            continue
    out: list[FilingRef] = []
    for b in blocks:
        n = len(b.get("accessionNumber", []))
        for i in range(n):
            rd = b["reportDate"][i]
            out.append(
                FilingRef(
                    accession=b["accessionNumber"][i],
                    form=b["form"][i],
                    filing_date=date.fromisoformat(b["filingDate"][i]),
                    report_date=date.fromisoformat(rd) if rd else None,
                    primary_document=b["primaryDocument"][i],
                    items=b.get("items", [""] * n)[i] or "",
                )
            )
    out.sort(key=lambda f: f.filing_date)
    return out


def frame(client: EdgarClient, concept: str, period: str, unit: str = "USD", taxonomy: str = "us-gaap") -> pd.DataFrame:
    """One concept across all filers for a calendar period, e.g. CY2023 (duration) or CY2023Q4I (instant)."""
    url = FRAMES_URL.format(taxonomy=taxonomy, concept=concept, unit=unit, period=period)
    try:
        data = client.get_json(url, max_age_s=30 * DAY)
    except EdgarError:
        return pd.DataFrame(columns=["cik", "entityName", "end", "val", "accn"])
    df = pd.DataFrame(data.get("data", []))
    if df.empty:
        return pd.DataFrame(columns=["cik", "entityName", "end", "val", "accn"])
    df["cik"] = df["cik"].astype(int)
    return df


def document_url(cik: int, filing: FilingRef, name: str | None = None) -> str:
    return ARCHIVE_URL.format(cik=cik, accn_nodash=filing.accn_nodash, name=name or filing.primary_document)


def filing_index(client: EdgarClient, cik: int, filing: FilingRef) -> list[dict]:
    """Rows of the filing index page: seq, description, document, type."""
    url = FILING_INDEX_URL.format(cik=cik, accn_nodash=filing.accn_nodash, accn=filing.accession)
    html = HTMLParser(client.get_text(url))
    rows: list[dict] = []
    for table in html.css("table.tableFile"):
        for tr in table.css("tr")[1:]:
            tds = tr.css("td")
            if len(tds) < 4:
                continue
            link = tds[2].css_first("a")
            href = (link.attributes.get("href") or "") if link else ""
            rows.append(
                {
                    "description": tds[1].text(strip=True),
                    "document": href.rsplit("/", 1)[-1].replace("ix?doc=", ""),
                    "type": tds[3].text(strip=True),
                }
            )
    return rows


def companies_by_sic(client: EdgarClient, sic: int, max_pages: int = 10) -> list[int]:
    ciks: list[int] = []
    for page in range(max_pages):
        url = SIC_BROWSE_URL.format(sic=sic, start=page * 100)
        try:
            raw = client.get_bytes(url, max_age_s=30 * DAY)
        except EdgarError:
            break
        try:
            root = etree.fromstring(raw)
        except etree.XMLSyntaxError:
            break
        found = [int(x) for x in root.xpath("//*[local-name()='cik']/text()")]
        ciks.extend(found)
        if len(found) < 100:
            break
    return sorted(set(ciks))


COMPANY_SEARCH_URL = ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={q}&type=10-K"
                      "&dateb=&owner=include&count=40&output=atom")


def company_search(client: EdgarClient, name: str) -> list[tuple[int, str]]:
    """SEC company-name search (includes registrants without a current ticker). Returns (cik, name) pairs."""
    from urllib.parse import quote_plus

    try:
        raw = client.get_bytes(COMPANY_SEARCH_URL.format(q=quote_plus(name)), max_age_s=30 * DAY)
        root = etree.fromstring(raw)
    except (EdgarError, etree.XMLSyntaxError):
        return []
    out: list[tuple[int, str]] = []
    for info in root.xpath("//*[local-name()='company-info']"):
        cik = info.xpath("./*[local-name()='cik']/text()")
        nm = info.xpath("./*[local-name()='conformed-name']/text()") or info.xpath("./*[local-name()='name']/text()")
        if cik:
            out.append((int(cik[0]), nm[0] if nm else ""))
    return list(dict.fromkeys(out))


def sic_codes(client: EdgarClient) -> list[int]:
    try:
        html = client.get_text(SIC_LIST_URL, max_age_s=180 * DAY)
    except EdgarError:
        return []
    return sorted({int(m) for m in re.findall(r"<td[^>]*>\s*(\d{3,4})\s*</td>", html)})
