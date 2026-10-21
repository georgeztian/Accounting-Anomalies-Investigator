"""Multi-year review: one company's yearly findings merged into anomalies that are tracked across years.

Each fiscal year is first analysed on its own (detectors and scoring, as in a single-year run). Findings are then
grouped across years: a scheme pattern by its id, and a standalone test result by the pattern it belongs to when
that pattern appears in any year (otherwise by the test). A merged anomaly keeps every year's test results and
evidence, and is represented by its most recent occurrence (whose materiality, questions and context are used).

    score    = peak + (1 - peak) x 0.5 x (years flagged - 1) / (years reviewed - 1)
               (years flagged: those with a result of their own, so one event seen again by later years counts once)
    priority = score x (0.5 + 0.5 x min(largest materiality multiple, 3) / 3) x recency
    recency  = 1 if flagged in one of the last three years reviewed, else 0.85
    period risk index = mean(yearly index) + 0.5 x (max(yearly index) - mean(yearly index))

Recurrence raises the score because a pattern that repeats is less likely to be noise or a one-off event; the
period index sits between the average and the worst year, so one bad year is neither hidden nor allowed to define
the decade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ffi.scoring.engine import TEMPLATES, Finding, _matches, band_of, remove_findings

RECENT_YEARS = 3
OLD_FACTOR = 0.85
MAX_EVIDENCE = 16


@dataclass
class SpanResult:
    periods: list[str]  # fiscal-year labels reviewed, oldest first
    yearly: list[dict[str, Any]]  # one row per year: risk index, band, flagged tests, top finding
    risk_index: float
    band: str
    findings: list[Finding]  # merged anomalies, most important first
    rep_year: dict[str, int] = field(default_factory=dict)  # merged finding id -> index of the year that represents it
    area_by_year: dict[str, list[float]] = field(default_factory=dict)  # accounting area -> strength in each year
    set_aside: list[Finding] = field(default_factory=list)  # anomalies the investigation showed to be test errors


def _key(f: Finding, present_patterns: set[str]) -> str:
    """Patterns merge by id; a standalone test result joins a pattern it is required for, when that pattern
    appears in any year, so the same anomaly is not split by how strongly it was corroborated each year."""
    if f.kind == "pattern":
        return f.id
    s = f.signals[0]
    for t in TEMPLATES:
        if t.id in present_patterns and any(_matches(s, tok) for group in t.required_any for tok in group):
            return t.id
    return s.detector_id


def period_index(indices: list[float]) -> float:
    if not indices:
        return 0.0
    mean, top = float(np.mean(indices)), float(max(indices))
    return round(mean + 0.5 * (top - mean), 1)


def merged_score(peak: float, flagged: int, reviewed: int) -> float:
    return peak + (1 - peak) * 0.5 * (flagged - 1) / max(1, reviewed - 1)


def merge(years: list[Any]) -> SpanResult:
    """`years`: YearAnalysis objects, oldest first."""
    labels = [y.ctx.focus_label for y in years]
    n = len(years)
    present = {f.id for y in years for f in y.score.findings if f.kind == "pattern"}
    groups: dict[str, list[tuple[int, Finding]]] = {}
    for i, y in enumerate(years):
        for f in y.score.findings:
            groups.setdefault(_key(f, present), []).append((i, f))

    merged: list[Finding] = []
    rep_year: dict[str, int] = {}
    for key, occ in groups.items():
        by_year: dict[int, Finding] = {}
        extra: dict[int, list[Finding]] = {}
        for i, f in occ:  # a year can hold the same key twice (a pattern and a leftover test result): the stronger
            if i not in by_year or f.score > by_year[i].score:  # represents the year, the other's results are kept too
                if i in by_year:
                    extra.setdefault(i, []).append(by_year[i])
                by_year[i] = f
            else:
                extra.setdefault(i, []).append(f)
        order = sorted(by_year)
        rep_i = order[-1]
        rep = by_year[rep_i]
        pattern = next((by_year[i] for i in reversed(order) if by_year[i].kind == "pattern"), None)
        signals, seen = [], set()
        for i in reversed(order):  # newest first
            for s in [x for f in [by_year[i], *extra.get(i, [])] for x in f.signals]:
                if (s.detector_id, s.period) not in seen:
                    seen.add((s.detector_id, s.period))
                    signals.append(s)
        evidence, seen_ev = [], set()
        for i in reversed(order):
            for e in [x for f in [by_year[i], *extra.get(i, [])] for x in f.evidence]:
                if e.node_id not in seen_ev:
                    seen_ev.add(e.node_id)
                    evidence.append(e)
        # recurrence counts the years that add a result of their own: a result about an earlier year that a later year's
        # tests see again (the same revised comparative, A.14) is one event, not a pattern that repeats
        recurring, known = 0, set()
        for i in order:
            res = {(s.detector_id, s.period) for f in [by_year[i], *extra.get(i, [])] for s in f.signals}
            recurring += bool(res - known)
            known |= res
        peak = max(f.score for f in by_year.values())
        score = merged_score(peak, recurring, n)
        mult = max((f.materiality.multiple for f in by_year.values() if f.materiality), default=None)
        priority = score * (0.5 + 0.5 * min(mult, 3.0) / 3.0) if mult is not None else score * 0.5
        if rep_i < n - RECENT_YEARS:
            priority *= OLD_FACTOR
        lead = (pattern or rep).signals[0].detector_id
        timeline = []
        for i, y in enumerate(years):
            f = by_year.get(i)
            lead_sig = max(y.ctx.signals.get(lead, []), key=lambda s: s.severity, default=None)
            timeline.append({
                "period": labels[i], "flagged": f is not None, "score": round(f.score, 3) if f else 0.0,
                "lead_test": lead, "lead_severity": round(lead_sig.severity, 3) if lead_sig else None,
                "lead_value": lead_sig.raw_value if lead_sig else None, "lead_unit": lead_sig.unit if lead_sig else "",
                "impact_mid": f.materiality.impact_mid if f and f.materiality else None,
            })
        merged.append(Finding(
            id=key, title=(pattern or rep).title, kind="pattern" if pattern else "signal", score=score,
            signals=signals, impact=rep.impact, materiality=rep.materiality, evidence=evidence[:MAX_EVIDENCE],
            questions=list(rep.questions), alternatives=list(rep.alternatives), priority=priority,
            years=[labels[i] for i in order], timeline=timeline,
        ))
        rep_year[key] = rep_i
    merged.sort(key=lambda f: -f.priority)

    yearly = []
    for y in years:
        sc = y.score
        yearly.append({"period": y.ctx.focus_label, "period_end": str(y.ctx.focus.date()), "risk_index": sc.risk_index,
                       "band": sc.band, "findings": len(sc.findings),
                       "flagged": sum(1 for r in y.results for s in r.signals if s.flagged),
                       "top": sc.findings[0].title if sc.findings else ""})
    areas = sorted({a for y in years for a in y.score.area_scores})
    area_by_year = {a: [round(y.score.area_scores.get(a, 0.0), 2) for y in years] for a in areas}
    index = period_index([y.score.risk_index for y in years])
    return SpanResult(labels, yearly, index, band_of(index), merged, rep_year, area_by_year)


def apply_set_aside(span: SpanResult, years: list[Any]) -> list[Finding]:
    """Anomalies the investigation showed to be test errors (confirmed by the independent auditor) leave the list, and each
    year they were flagged in loses their contribution to its risk index; the period index is recomputed."""
    aside = [f for f in span.findings if f.set_aside]
    if not aside:
        return []
    for f in aside:
        # the year's occurrences: the pattern itself, or a test result standing alone in a year without the pattern
        detectors = {s.detector_id for s in f.signals}
        for y in (y for y in years if y.ctx.focus_label in f.years):
            remove_findings(y.score, lambda g, f=f, d=detectors: g.id != f.id and not (g.kind == "signal" and g.signals[0].detector_id in d),
                            [f"pattern:{f.id}", *detectors])
    span.findings = [f for f in span.findings if not f.set_aside]
    span.set_aside += aside
    for row, y in zip(span.yearly, years, strict=True):
        row.update(risk_index=y.score.risk_index, band=y.score.band, findings=len(y.score.findings),
                   top=y.score.findings[0].title if y.score.findings else "")
    span.risk_index = period_index([y.score.risk_index for y in years])
    span.band = band_of(span.risk_index)
    return aside


def apply_audit(span: SpanResult) -> None:
    """Unverified merged anomalies lose half their weight, as in a single-year run."""
    from ffi.scoring.engine import UNVERIFIED_FACTOR

    for f in span.findings:
        if f.audit is not None and f.audit.status == "unverified":
            f.priority *= UNVERIFIED_FACTOR
    span.findings.sort(key=lambda x: -x.priority)


def years_text(f: Finding, periods: list[str]) -> str:
    """'Flagged in 4 of 10 years (FY2019, FY2021-FY2024); still present in FY2025.'"""
    idx = sorted(periods.index(p) for p in f.years if p in periods)
    runs, start = [], None
    for k, i in enumerate(idx):
        if start is None:
            start = i
        if k == len(idx) - 1 or idx[k + 1] != i + 1:
            runs.append(periods[start] if start == i else f"{periods[start]}-{periods[i]}")
            start = None
    head = f"Flagged in {len(idx)} of {len(periods)} years ({', '.join(runs)})"
    if idx and idx[-1] == len(periods) - 1:
        return head + f"; still present in {periods[-1]}."
    return head + (f"; last seen in {periods[idx[-1]]}." if idx else ".")
