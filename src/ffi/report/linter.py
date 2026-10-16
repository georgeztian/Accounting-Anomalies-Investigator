"""Neutral-language policy (BLUEPRINT §12.2): generated text must not accuse."""

from __future__ import annotations

import re

BANNED = [
    (r"\bfraud(?:ulent|ster)?s?\b", "inconsistent with expectations"),
    (r"\bcook(?:ing|ed)? the books\b", "aggressive accounting"),
    (r"\bmanipulat(?:e|ed|es|ing|ion)\b", "management of reported results"),
    (r"\bembezzl\w*\b", "irregularity"),
    (r"\bscam\w*\b", "irregularity"),
    (r"\blie[sd]?\b|\blying\b", "inconsistent statement"),
    (r"\bcriminal\w*\b", "irregular"),
]
ALLOW = re.compile(r"(?i)\bnot (?:investment|legal|audit) advice|does not (?:allege|imply)|is not evidence of fraud")


def lint(text: str) -> tuple[str, list[str]]:
    """Return (clean_text, violations). Disclaimer sentences are allowed to mention the word."""
    violations: list[str] = []
    if not text:
        return text, violations
    out = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if ALLOW.search(sentence):
            out.append(sentence)
            continue
        s = sentence
        for pat, repl in BANNED:
            if re.search(pat, s, re.I):
                violations.append(re.search(pat, s, re.I).group(0))
                s = re.sub(pat, repl, s, flags=re.I)
        out.append(s)
    return " ".join(out), violations
