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


def build_doc(accession: str, form: str, period_end: str, url: str, raw_html: str) -> TextDoc:
    lines = html_to_lines(raw_html)
    parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    pos = 0
    for line, _ in lines:
        offsets.append((pos, pos + len(line)))
        parts.append(line)
        pos += len(line) + 1
    text = "\n".join(parts)

    # ---- locate Item headers: keep the last short header per item (the TOC comes first)
    hits: dict[str, int] = {}
    for i, (line, _is_row) in enumerate(lines):
        if len(line) > 260:
            continue
        m = ITEM_RE.match(line.replace(" | ", " "))
        if m:
            key = m.group(1).lower()
            title_words = ITEM_NAMES.get(key, "").split()[0].lower()[:6]
            if len(line) > 140 and title_words not in line.lower():
                continue  # a long sentence that merely starts with "Item N" is a cross-reference, not a header
            hits[key] = i
    starts = sorted((idx, key) for key, idx in hits.items())
    sections: dict[str, tuple[int, int]] = {}
    for n, (idx, key) in enumerate(starts):
        end_idx = starts[n + 1][0] if n + 1 < len(starts) else len(lines)
        sections[f"item{key}"] = (idx, end_idx)

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


AUDITOR_RE = re.compile(r"/s/\s*([A-Z][A-Za-z&,.' ]{2,60}?(?:LLP|LLC|P\.C\.|PC|Ltd\.?))")


def find_auditor(doc: TextDoc) -> str | None:
    """Audit firm name from the signed report in Item 8 (e.g. '/s/ Ernst & Young LLP')."""
    sec = doc.section_text("item8") or doc.text
    m = AUDITOR_RE.search(sec)
    return m.group(1).strip() if m else None
