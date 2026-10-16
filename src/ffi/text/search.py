"""Keyword search over filing text with verifiable spans."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ffi.text.document import Paragraph, TextDoc


@dataclass
class Hit:
    doc: TextDoc
    paragraph: Paragraph
    start: int  # absolute offsets into doc.text
    end: int
    pattern: str

    @property
    def quote(self) -> str:
        return self.doc.text[self.start : self.end]


def sentence_bounds(text: str, pos: int, max_len: int = 380) -> tuple[int, int]:
    lo = max(text.rfind(". ", 0, pos), text.rfind("\n", 0, pos))
    lo = lo + 1 if lo >= 0 else 0
    while lo < len(text) and text[lo] == " ":
        lo += 1
    hi_dot = text.find(". ", pos)
    hi_nl = text.find("\n", pos)
    cands = [h for h in (hi_dot, hi_nl) if h != -1]
    hi = min(cands) + 1 if cands else len(text)
    if hi - lo > max_len:
        lo = max(lo, pos - max_len // 2)
        hi = min(hi, lo + max_len)
    return lo, hi


def search(doc: TextDoc, pattern: str, sections: tuple[str, ...] | None = None, flags: int = re.IGNORECASE) -> list[Hit]:
    rx = re.compile(pattern, flags)
    hits: list[Hit] = []
    for p in doc.paragraphs:
        if p.is_table_row or (sections and p.section not in sections):
            continue
        for m in rx.finditer(p.text):
            a, b = sentence_bounds(doc.text, p.start + m.start())
            hits.append(Hit(doc, p, a, min(b, p.end), pattern))
            break
    return hits


def count(doc: TextDoc, pattern: str, sections: tuple[str, ...] | None = None) -> int:
    rx = re.compile(pattern, re.IGNORECASE)
    return sum(len(rx.findall(p.text)) for p in doc.paragraphs if not sections or p.section in sections)


def new_mentions(cur: TextDoc, prior: TextDoc | None, pattern: str, sections: tuple[str, ...] | None = None) -> list[Hit]:
    """Paragraph hits in `cur` for a pattern that did not appear at all in `prior`."""
    hits = search(cur, pattern, sections)
    if prior is None:
        return hits
    if search(prior, pattern, sections):
        return []
    return hits
