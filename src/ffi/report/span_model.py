"""Report model for a multi-year review (pipeline/span.py): the same finding blocks as a single-year report, plus
the year-by-year views (risk index, key figures, warning-sign strength by area, every year's test results)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from ffi.detectors.stats import fmt_money, fmt_pct
from ffi.report import charts
from ffi.report import plain as P
from ffi.report.model import (
    DISCLAIMER,
    KEY_ITEMS,
    NONE_TEXT,
    SET_ASIDE_NOTE,
    TEXT_TESTS,
    _ev_dict,
    audit_summary,
    company_block,
    finding_block,
    flag_level,
    key_value,
    peer_text,
    profile_block,
    run_notes,
    set_aside_blocks,
    test_families,
)
from ffi.scoring.span import years_text

MAX_SIGNALS = 15  # test results shown per merged anomaly (newest first); the JSON keeps all


def _level(v: float) -> int:
    return 0 if v < 0.01 else 1 if v < 0.3 else 2 if v < 0.5 else 3 if v < 0.7 else 4 if v < 0.85 else 5


def _strength_chart(f, flag_at: float) -> str:
    """The lead test's severity in every year reviewed, against the flag level."""
    labels = [t["period"] for t in f.timeline]
    vals = [t["lead_severity"] if t["lead_severity"] is not None else float("nan") for t in f.timeline]
    if sum(1 for v in vals if np.isfinite(v)) < 2:
        return ""
    return charts.line_chart(f"{P.name(f.timeline[0]['lead_test'])}: severity by year", labels,
                             [{"name": "Severity", "values": vals, "cls": "s1"},
                              {"name": "Flag level", "values": [flag_at] * len(labels), "cls": "s2", "dashed": True}], "num")


def build_span_model(out) -> dict[str, Any]:
    span, years, ctx = out.span, out.years, out.ctx
    periods = span.periods
    n = len(periods)
    flag_at = flag_level(ctx)

    findings = []
    for i, f in enumerate(span.findings[:12]):
        c = out.contexts.get(f.id, ctx)
        b = finding_block(f, c, i + 1, flag_at, chart_years=n)
        b["signals_total"] = len(b["signals"])
        b["signals"] = b["signals"][:MAX_SIGNALS]
        b["years_text"] = P.plain(years_text(f, periods), expand=False)
        b["timeline"] = [{"period": t["period"], "flagged": t["flagged"], "strength": t["score"],
                          "severity": t["lead_severity"]} for t in f.timeline]
        b["strength_chart"] = _strength_chart(f, flag_at)
        findings.append(b)

    yearly = [{**y, "top": P.title(y["top"]) if y["top"] else NONE_TEXT} for y in span.yearly]
    risk_chart = charts.line_chart("Risk index by year", periods,
                                   [{"name": "Risk index", "values": [y["risk_index"] for y in span.yearly], "cls": "s1"}], "num")
    earnings_cash = "" if (ctx.profile is not None and "C.CF.01" in ctx.profile.excluded) else charts.line_chart(
        "Net income vs cash from operations", periods,
        [{"name": "Net income", "values": [y.ctx.a("NET_INCOME") for y in years], "cls": "s1"},
         {"name": "Cash from operations", "values": [y.ctx.a("CFO") for y in years], "cls": "s2"}], "money")
    key_numbers = [{"label": lab, "values": [key_value(y.ctx, item) for y in years]} for lab, item in KEY_ITEMS]
    areas = sorted(span.area_by_year.items(), key=lambda kv: -max(kv[1]))
    area_rows = [{"area": P.area(a), "values": [{"v": v, "level": _level(v)} for v in vals]} for a, vals in areas]

    tests = []
    for y in years:
        families, statuses = test_families(y.results, flag_at)
        notable = [r for fam in families for r in fam["rows"] if r["status"] in ("flagged", "suppressed", "held")]
        tests.append({"period": y.ctx.focus_label, "families": families, "statuses": statuses, "notable": notable,
                      "flagged": sum(1 for r in notable if r["status"] == "flagged")})

    text_changes = []
    for did in TEXT_TESTS:
        for y in years:
            for r in y.results:
                if r.id != did:
                    continue
                for s in r.signals:
                    if s.flagged:
                        text_changes.append({"id": did, "name": P.name(did), "period": P.plain(s.period, expand=False),
                                             "rationale": P.plain(s.rationale),
                                             "quotes": [_ev_dict(e, y.ctx) for e in s.evidence if e.kind == "span"][:2]})
    pressure = [{"name": P.name(s.detector_id), "period": P.plain(s.period, expand=False), "rationale": P.plain(s.rationale)}
                for y in years for s in y.score.pressure]
    conditions = [f"{P.plain(y.ctx.focus_label, expand=False)}: {P.plain(v)}" for y in years for v in y.score.explain_conditions.values()]
    notes, seen = [], set()
    for y in years:
        for text in run_notes(y.ctx):
            if text not in seen:
                seen.add(text)
                notes.append(text)
    documents, seen_docs = [], set()
    for y in years:
        for d in y.ctx.docs + y.ctx.releases:
            if d.url in seen_docs:
                continue
            seen_docs.add(d.url)
            documents.append({"url": d.url, "period_end": d.period_end,
                              "label": (f"Earnings release, {d.period_end}" if d.form == "EX-99" else
                                        f"{P.source_label(d.form, None, 'span')} for the fiscal year ended {d.period_end}")})
    documents.sort(key=lambda d: d["period_end"])

    top_sentences = []
    for f in span.findings[:12]:
        m = f.materiality
        mt = (f" Latest estimate {fmt_money(m.impact_low)}-{fmt_money(m.impact_high)}, {fmt_pct(m.pct_pretax)} of pre-tax income."
              if m and m.pct_pretax is not None else "")
        top_sentences.append(f"{P.title(f.title)}. {P.plain(years_text(f, periods), expand=False)}{mt}")
    reported = span.findings[:12]
    llm = out.manifest.get("llm")
    return {
        "company": company_block(ctx),
        "periods": [P.plain(p, expand=False).capitalize() for p in periods],
        "period": f"{P.plain(periods[0], expand=False).capitalize()} to {P.plain(periods[-1], expand=False)}",
        "first_end": span.yearly[0]["period_end"], "last_end": span.yearly[-1]["period_end"], "years": n,
        "as_of": str(getattr(ctx, "cutoff", None) or ctx.as_of), "generated": datetime.now().strftime("%d %B %Y"),
        "analysis_type": ("Automated checks of the reported figures for each year, plus AI-assisted reading of the filings' text"
                          if llm else "Automated checks of the reported figures and filing text for each year"),
        "profile": profile_block(ctx),
        "currency": ctx.currency,
        "risk": {"index": span.risk_index, "band": span.band},
        "yearly": yearly,
        "top_sentences": top_sentences,
        "investigated": min(len(reported), int(out.manifest.get("options", {}).get("max_cases") or 0)) if llm else 0,
        "key_question": (findings[0]["questions"][0] if findings and findings[0]["questions"][0] != NONE_TEXT
                         else "None: no anomaly was reported." if not findings else NONE_TEXT),
        "key_numbers": key_numbers,
        "areas": area_rows,
        "findings": findings,
        "set_aside": set_aside_blocks(getattr(span, "set_aside", [])), "set_aside_note": SET_ASIDE_NOTE,
        "audit_summary": audit_summary(reported),
        "tests": tests,
        "flag_at": flag_at,
        "text_changes": text_changes,
        "pressure": pressure,
        "explain_conditions": conditions,
        "charts": {"risk": risk_chart, "earnings_cash": earnings_cash},
        "peer_method": peer_text(ctx),
        "peers": [{"name": p.name} for p in (ctx.peers.loaded if ctx.peers else [])],
        "documents": documents,
        "notes": notes or [NONE_TEXT],
        "none_text": NONE_TEXT,
        "glossary": P.GLOSSARY,
        "disclaimer": DISCLAIMER,
    }
