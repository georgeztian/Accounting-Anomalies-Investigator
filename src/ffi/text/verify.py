"""Verify that a quote occurs verbatim in a document, tolerating whitespace and typographic quote/dash variants.

Returns exact offsets into the original document text, so the evidence span is the filing's own text.
"""

from __future__ import annotations

from functools import lru_cache

from ffi.text.document import TextDoc

# every quote mark becomes one character: a draft may write 'Venture' where the filing has “Venture”
_TRANS = str.maketrans({"‘": "'", "’": "'", "“": "'", "”": "'", '"': "'", "–": "-", "—": "-",
                        " ": " ", " ": " ", " ": " "})


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    out: list[str] = []
    idx: list[int] = []
    prev_space = False
    for i, ch in enumerate(text.translate(_TRANS)):
        if ch.isspace():
            if prev_space:
                continue
            out.append(" ")
            idx.append(i)
            prev_space = True
        else:
            out.append(ch.lower())
            idx.append(i)
            prev_space = False
    return "".join(out), idx


@lru_cache(maxsize=16)
def _doc_norm(doc_id: str, text: str) -> tuple[str, tuple[int, ...]]:
    norm, idx = _normalize_with_map(text)
    return norm, tuple(idx)


def locate(doc: TextDoc, quote: str) -> tuple[int, int] | None:
    if not quote or len(quote.strip()) < 12:
        return None
    pos = doc.text.find(quote)
    if pos >= 0:
        return pos, pos + len(quote)
    norm, idx = _doc_norm(doc.accession, doc.text)
    q, _ = _normalize_with_map(quote.strip().strip('"').strip())
    q = q.strip()
    p = norm.find(q)
    if p < 0:
        return None
    start = idx[p]
    end = idx[min(p + len(q) - 1, len(idx) - 1)] + 1
    return start, end
