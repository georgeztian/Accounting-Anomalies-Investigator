"""Neutral-language policy: generated text must not accuse."""

from __future__ import annotations

import re

BANNED = [
    (r"\bfraud(?:ulent|ster)?s?\b", "inconsistent with expectations"),
    (r"\bcook(?:ing|ed)? the books\b", "aggressive accounting"),
    (r"\bmanipulat(?:e|ed|es|ing|ion)\b", "management of reported results"),
    (r"\bembezzl\w*\b", "irregularity"),
    (r"\bscams?\b", "irregularity"),
    # an untruth, never "the difference lies in timing"
    (r"\blied\b|\bliars?\b|\b(?:lie|lies|lying) to\b", "inconsistent statement"),
    (r"\bcriminal\w*\b", "irregular"),
]
ALLOW = re.compile(r"(?i)\bnot (?:investment|legal|audit) advice|does not (?:allege|imply)|is not evidence of fraud")


def lint(text: str) -> tuple[str, list[str]]:
    """Return (clean_text, violations). Disclaimer sentences are allowed to mention the word. Line and paragraph breaks
    are kept."""
    violations: list[str] = []
    if not text:
        return text, violations
    parts = re.split(r"(?<=[.!?])(\s+)", text)  # sentences at even positions, the white space between them at odd ones
    for i in range(0, len(parts), 2):
        if ALLOW.search(parts[i]):
            continue
        for pat, repl in BANNED:
            if m := re.search(pat, parts[i], re.I):
                violations.append(m.group(0))
                parts[i] = re.sub(pat, repl, parts[i], flags=re.I)
    return "".join(parts), violations
