"""Filing text: HTML -> normalized text -> Items, notes and paragraph spans (BLUEPRINT §5 step 5).

Every paragraph has a stable id `accession#section#index#start-end` whose offsets index into
`TextDoc.text`, so any quote can be verified verbatim at its span.
"""

from __future__ import annotations

import html as html_lib
import re
from dataclasses import dataclass, field

from selectolax.lexbor import LexborHTMLParser

ITEM_RE = re.compile(
    r"^\s*(?:part\s+i+\s*[,.\-–—]?\s*)?item\s*(1a|1b|1c|1|2|3|4|5|6|7a|7|8|9a|9b|9c|9|10|11|12|13|14|15|16)\b\s*[\.:\-–—]?\s*(.*)$",
    re.IGNORECASE,
)
NOTE_RE = re.compile(r"^\s*(?:(?i:note)\s+(\d{1,2})|(\d{1,2})\.)\s*[\.:\-–—]?\s*([A-Z][^.]{2,90})$")
ITEM_NAMES = {
    "1": "Business", "1a": "Risk Factors", "1b": "Unresolved Staff Comments", "1c": "Cybersecurity",
    "2": "Properties", "3": "Legal Proceedings", "4": "Mine Safety", "5": "Market for Equity",
    "6": "Reserved", "7": "Management Discussion", "7a": "Market Risk", "8": "Financial Statements",
    "9": "Accountant Changes", "9a": "Controls and Procedures", "9b": "Other Information",
    "9c": "Foreign Jurisdictions", "10": "Directors and Officers", "11": "Executive Compensation",
    "12": "Security Ownership", "13": "Relationships", "14": "Accountant Fees", "15": "Exhibits", "16": "Summary",
}
# 20-F (foreign private issuer annual report): its own item numbers, mapped onto the 10-K section keys detectors use.
ITEM_RE_20F = re.compile(
    r"^\s*(?:part\s+i+\s*[,.\-–—]?\s*)?item\s*(16[a-j]|1[0-9]|[1-9])\b\s*[\.:\-–—]?\s*(.*)$", re.IGNORECASE)
ITEM_NAMES_20F = {
    "1": "Identity", "2": "Offer", "3": "Key Information", "4": "Information on the Company", "4a": "Unresolved",
    "5": "Operating and Financial Review", "6": "Directors", "7": "Major Shareholders", "8": "Financial Information",
    "9": "Offer", "10": "Additional Information", "11": "Quantitative", "12": "Description", "13": "Defaults",
    "14": "Material Modifications", "15": "Controls", "16": "Reserved", "16a": "Audit", "16b": "Code", "16c": "Principal",
    "16d": "Exemptions", "16e": "Purchases", "16f": "Change", "16g": "Corporate", "16h": "Mine", "16i": "Disclosure",
    "16j": "Insider", "17": "Financial Statements", "18": "Financial Statements", "19": "Exhibits",
}
SECTION_MAP_20F = {"3": "item1a", "4": "item1", "5": "item7", "8": "item3", "11": "item7a", "15": "item9a", "16f": "item9"}
BLOCK_CLOSE = re.compile(r"(?i)</(p|div|tr|li|h[1-6]|table|section|center)>")
CELL_CLOSE = re.compile(r"(?i)</t[dh]>")
BR = re.compile(r"(?i)<br\s*/?>")
HIDDEN = re.compile(r"(?is)<ix:header>.*?</ix:header>")
WS = re.compile(r"[ \t  ​ ]+")


@dataclass
class Paragraph:
    id: str
    section: str  # e.g. "item7"
    note: str | None
    start: int
    end: int
    text: str
    is_table_row: bool = False


@dataclass
class TextDoc:
    accession: str
    form: str
    period_end: str
    url: str
    text: str
    paragraphs: list[Paragraph] = field(default_factory=list)
    sections: dict[str, tuple[int, int]] = field(default_factory=dict)  # section -> (para_start, para_end)

    def section_paragraphs(self, section: str, include_tables: bool = False) -> list[Paragraph]:
        rng = self.sections.get(section)
        if not rng:
            return []
        ps = self.paragraphs[rng[0] : rng[1]]
        return ps if include_tables else [p for p in ps if not p.is_table_row]

    def section_text(self, section: str) -> str:
        return "\n".join(p.text for p in self.section_paragraphs(section))

    def find(self, quote: str) -> int:
        return self.text.find(quote)

    def verify(self, quote: str, start: int | None = None, end: int | None = None) -> bool:
        if start is not None and end is not None:
            return self.text[start:end] == quote
        return quote in self.text


def html_to_lines(raw_html: str) -> list[tuple[str, bool]]:
    """Return (line, is_table_row) for each visible block of the document."""
    s = HIDDEN.sub(" ", raw_html)
    s = CELL_CLOSE.sub(" | ", s)
    s = BLOCK_CLOSE.sub(lambda m: f" </{m.group(1)}>", s)
    s = BR.sub(" ", s)
    tree = LexborHTMLParser(s)
    for node in tree.css("script, style, head"):
        node.decompose()
    for node in tree.css('[style*="display:none"], [style*="display: none"]'):
        node.decompose()
    body = tree.body or tree.root
    text = html_lib.unescape(body.text(separator="") if body else "")
    lines: list[tuple[str, bool]] = []
    for raw in text.split(" "):
        line = WS.sub(" ", raw.replace("\n", " ")).strip()
        is_row = " | " in line
        if is_row:
            cells = [c.strip() for c in line.split("|")]
            cells = [c for c in cells if c and c not in ("$", "%", ")")]
            line = " | ".join(cells)
        if len(line) < 2 or re.fullmatch(r"[\d\s|]+", line) and len(line) < 4:
            continue
        lines.append((line, is_row))
    return lines


AUDIT_REPORT_RE = re.compile(r"(?i)^report of independent (?:registered public accounting firm|auditors?)\b")


def build_doc(accession: str, form: str, period_end: str, url: str, raw_html: str,
              appendix: list[tuple[str, str]] | None = None) -> TextDoc:
    """`appendix`: (synthetic Item header, exhibit HTML) pairs for annual-report content filed as exhibits
    (10-K Exhibit 13, 40-F Exhibits 99.x). They are appended after the main document; their headers come
    last, so they win over the main document's short 'incorporated by reference' Items."""
    lines = html_to_lines(raw_html)
    for header, html in appendix or []:
        part = html_to_lines(html)
        if header.startswith("Item 7"):
            # an annual-report exhibit holds MD&A and then the audited statements: start Item 8 at the audit report
            cut = next((j for j, (ln, _) in enumerate(part) if j > len(part) // 4 and len(ln) < 120 and AUDIT_REPORT_RE.match(ln)), None)
            if cut is not None:
                part = part[:cut] + [("Item 8. Financial Statements", False)] + part[cut:]
        lines += [(header, False)] + part
    parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    pos = 0
    for line, _ in lines:
        offsets.append((pos, pos + len(line)))
        parts.append(line)
        pos += len(line) + 1
    text = "\n".join(parts)

    # ---- locate Item headers: keep the last short header per item (the TOC comes first)
    is_20f = form.upper().startswith("20-F")
    item_re, names = (ITEM_RE_20F, ITEM_NAMES_20F) if is_20f else (ITEM_RE, ITEM_NAMES)
    hits: dict[str, int] = {}
    for i, (line, _is_row) in enumerate(lines):
        if len(line) > 260:
            continue
        m = item_re.match(line.replace(" | ", " "))
        if m:
            key = m.group(1).lower()
            title_words = (names.get(key) or "item").split()[0].lower()[:6]
            if len(line) > 140 and title_words not in line.lower():
                continue  # a long sentence that merely starts with "Item N" is a cross-reference, not a header
            hits[key] = i
    starts = sorted((idx, key) for key, idx in hits.items())
    sections: dict[str, tuple[int, int]] = {}
    for n, (idx, key) in enumerate(starts):
        end_idx = starts[n + 1][0] if n + 1 < len(starts) else len(lines)
        sections[f"item{key}"] = (idx, end_idx)
    if is_20f:
        sections = _map_20f_sections(sections, len(lines))

    def section_of(i: int) -> str:
        for name, (a, b) in sections.items():
            if a <= i < b:
                return name
        return "front" if not starts or i < starts[0][0] else "other"

    paragraphs: list[Paragraph] = []
    note: str | None = None
    for i, (line, is_row) in enumerate(lines):
        sec = section_of(i)
        if sec == "item8" and not is_row and len(line) <= 100:
            nm = NOTE_RE.match(line)
            if nm:
                note = f"Note {nm.group(1) or nm.group(2)} {nm.group(3).strip()}"
        elif sec != "item8":
            note = None
        a, b = offsets[i]
        paragraphs.append(Paragraph(f"{accession}#{sec}#{i}#{a}-{b}", sec, note, a, b, line, is_row))
    return TextDoc(accession, form, period_end, url, text, paragraphs, sections)


def _map_20f_sections(found: dict[str, tuple[int, int]], n_lines: int) -> dict[str, tuple[int, int]]:
    """Rename 20-F items to the 10-K keys detectors read. The audited statements (F-pages) follow Items 18-19
    at the end of the document, so 'item8' runs from Item 18 (or 17) to the end."""
    out: dict[str, tuple[int, int]] = {}
    for k20, k10 in SECTION_MAP_20F.items():
        if f"item{k20}" in found:
            out[k10] = found[f"item{k20}"]
    fs = found.get("item18") or found.get("item17")
    if fs:
        out["item8"] = (fs[0], n_lines)
    return out


AUDITOR_RE = re.compile(r"/s/\s*([A-Z][A-Za-z&,.' ]{2,60}?(?:LLP|LLC|P\.C\.|PC|Ltd\.?))")


def find_auditor(doc: TextDoc) -> str | None:
    """Audit firm name from the signed report in Item 8 (e.g. '/s/ Ernst & Young LLP')."""
    sec = doc.section_text("item8") or doc.text
    m = AUDITOR_RE.search(sec)
    return m.group(1).strip() if m else None
