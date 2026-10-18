"""Assemble the report data model from a run."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from ffi.detectors import scores as S
from ffi.detectors.catalog import explain, interpret
from ffi.detectors.d_stats import RATIOS
from ffi.detectors.scores import V
from ffi.detectors.stats import fmt_money, fmt_num, fmt_pct, money_prefix
from ffi.report import charts
from ffi.report import plain as P
from ffi.report.explain import for_report
from ffi.report.linter import lint

FAMILY_NAMES = {"A": "Accounting integrity", "B": "Forensic scores & accrual models", "C": "Account-level detectors",
                "D": "Statistical & ML", "E": "Text & disclosure", "F": "Knowledge-graph"}
DETECTOR_RATIO = {"C.AR.01": ("dso", "days"), "C.INV.01": ("dsi", "days"), "C.CF.03": ("dpo", "days"),
                  "C.EXP.04": ("dep_rate", "pct"), "C.EXP.02": ("other_ca_rev", "pct"), "B.03": ("accruals_ta", "pct"),
                  "C.CF.01": ("cfo_ni", "num"), "C.REV.05": ("asset_turnover", "num"), "C.EXP.06": ("sga_ratio", "pct"),
                  "C.RES.05": ("etr", "pct"), "C.EXP.09": ("sbc_ratio", "pct"), "C.BS.01": ("interest_rate", "pct"),
                  "C.REV.02": ("deferred_rev_rev", "pct"), "B.05": ("accruals_ta", "pct"), "C.EXP.01": ("rnd_ratio", "pct"),
                  "C.REV.01": ("ar_growth_less_rev", "pct"), "C.INV.02": ("inv_growth_less_cogs", "pct")}
DISCLAIMER = ("This report flags departures from expectation in public filings for further inquiry. It does not allege "
              "wrongdoing and is not evidence of fraud. It is decision support for qualified analysts, not investment, "
              "legal or audit advice. Anomalies often have legitimate explanations.")


def _ratio_chart(ctx, detector_id: str, years: int = 6) -> str:
    if detector_id not in DETECTOR_RATIO:
        return ""
    key, kind = DETECTOR_RATIO[detector_id]
    name, fn = RATIOS[key]
    periods = ctx.periods[-years:]
    mine, med, q1, q3 = [], [], [], []
    for p in periods:
        mine.append(float(fn(V(ctx.A, p))))
        vals = []
        if ctx.peers:
            for peer in ctx.peers.loaded:
                ap = ctx.peers.aligned_period(peer, p)
                if ap is not None:
                    v = float(fn(V(peer.annual, ap)))
                    if np.isfinite(v):
                        vals.append(v)
        med.append(float(np.median(vals)) if len(vals) >= 3 else float("nan"))
        q1.append(float(np.quantile(vals, 0.25)) if len(vals) >= 3 else float("nan"))
        q3.append(float(np.quantile(vals, 0.75)) if len(vals) >= 3 else float("nan"))
    labels = [ctx.label(p) for p in periods]
    series = [{"name": ctx.company.ticker or "Company", "values": mine, "cls": "s1"}]
    band = None
    if any(np.isfinite(m) for m in med):
        series.append({"name": "Peer median", "values": med, "cls": "s2", "dashed": True})
        band = (q1, q3)
    return charts.line_chart(f"{P.plain(name, expand=False)}: company vs peers", labels, series, kind, band)


def _scores_chart(out) -> str:
    ctx = out.ctx
    rows = []
    for did, label, fn, ref, ref_label in (
        ("B.01", "Beneish M-score", lambda P, p: S.beneish(P, p)["M"], -1.78, "-1.78"),
        ("B.02", "Dechow F-score", lambda P, p: S.dechow_f(P, p)["F"], 1.0, "1.0"),
        ("B.05", "Sloan accrual ratio", S.sloan_accruals, None, ""),
        ("B.06", "Montier C-score", lambda P, p: S.montier_c(P, p)["C"], None, ""),
    ):
        if ctx.profile is not None and did in ctx.profile.excluded:
            continue
        try:
            v = float(fn(ctx.A, ctx.focus))
        except Exception:  # noqa: BLE001
            continue
        peers = ctx.peer_values(lambda P, p, k, fn=fn: fn(P, p))
        rows.append({"label": label, "value": v, "peers": peers, "ref": ref, "ref_label": ref_label,
                     "kind": "pct" if "Sloan" in label else "num"})
    return charts.dot_strip("Forensic scores vs peers", rows) if rows else ""


def _earnings_cash_chart(out) -> str:
    ctx = out.ctx
    if ctx.profile is not None and "C.CF.01" in ctx.profile.excluded:
        return ""  # operating cash flow is not the cash yield of earnings for this business model
    periods = ctx.periods[-7:]
    labels = [ctx.label(p) for p in periods]
    ni = [ctx.A.get("NET_INCOME", p) for p in periods]
    cfo = [ctx.A.get("CFO", p) for p in periods]
    return charts.line_chart("Net income vs cash from operations", labels,
                             [{"name": "Net income", "values": ni, "cls": "s1"}, {"name": "Cash from operations", "values": cfo, "cls": "s2"}],
                             "money")


BENCHMARKS = {"pretax": "pre-tax income", "revenue": "revenue", "assets": "total assets", "equity": "shareholders' equity"}


def _ev_dict(e, ctx=None) -> dict:
    d = e.to_dict()
    if e.value is not None:
        is_money = (e.unit is not None and len(e.unit) == 3 and e.unit.isalpha() and e.unit.isupper()) or (e.unit is None and e.kind == "fact")
        d["value_fmt"] = fmt_money(e.value) if is_money else fmt_num(e.value, 3)
    f = ctx.filing(e.accession) if (ctx is not None and e.accession) else None
    form = f.form if f is not None else ("EX-99" if e.kind == "span" and ctx is not None and any(r.accession == e.accession for r in ctx.releases) else None)
    d["source"] = P.source_label(form, e.filed or (str(f.filing_date) if f is not None else None), e.kind)
    d["label"] = P.plain(e.label)
    d["kind_label"] = {"fact": "Reported figure", "span": "Quote from the filing", "event": "Filing event",
                       "derived": "Calculation", "table_cell": "Table figure"}.get(e.kind, e.kind)
    return d


def _dedupe(evidence: list) -> list:
    seen, out = set(), []
    for e in evidence:
        if e.node_id in seen:
            continue
        seen.add(e.node_id)
        out.append(e)
    return out


def _about(detector_id: str) -> dict[str, str]:
    ex = explain(detector_id)
    if not ex:
        return {"what": "", "why": "", "how_to_read": ""}
    return {"what": P.plain(ex.what), "why": P.plain(ex.why), "how_to_read": P.plain(ex.read)}


def _strength(score: float) -> str:
    return "Strong" if score >= 0.75 else "Moderate" if score >= 0.5 else "Weak"


def _audit(a) -> dict | None:
    if a is None:
        return None
    d = a.to_dict()
    for r in d["rounds"]:
        for c in r["checks"]:
            c["label"] = P.AUDIT_CHECKS.get(c["name"], c["name"].replace("_", " "))
            c["detail"] = P.plain(c["detail"])
        r["corrections"] = [P.plain(x) for x in r["corrections"]]
    return d


# Every finding shows the same blocks in the same order; these fill a block that has nothing to show.
NONE_TEXT = "None identified."
MAX_FIGURES, MAX_QUOTES = 12, 6
INVESTIGATION = {  # narrative_source -> (status label, text when the AI investigation adds no explanation of its own)
    "agent": ("Completed", "The AI investigation reached no conclusion beyond the test results below."),
    "agent-degraded": ("Did not finish", "The AI investigation did not finish within its time and cost limits, so it adds no "
                                         "conclusion beyond the test results below. Evidence it gathered is listed under "
                                         "Supporting evidence."),
    "agent: closed as explained": ("Likely explained", "The AI investigation found a likely legitimate explanation, listed under "
                                                       "Innocent explanations below, so this finding counts for half."),
    "agent narrative dropped by reviewer": ("Withdrawn", "The AI-written explanation was withdrawn at review, so the conclusion "
                                                         "rests on the test results below."),
    "template": ("Not run", "No AI investigation was run for this finding, so the conclusion rests on the test results below."),
}


def _investigation(f) -> tuple[str, str, str]:
    """(summary, investigation status, investigation text) for one finding."""
    status, fallback = INVESTIGATION.get(f.narrative_source, INVESTIGATION["template"])
    if f.narrative and f.narrative_source in ("agent", "agent-degraded"):
        summary, _, expl = f.narrative.partition("\n\n")
        return summary.strip(), status, expl.strip() or fallback
    return f.signals[0].rationale, status, fallback


def _evidence_blocks(evidence: list[dict]) -> dict[str, Any]:
    figures = [e for e in evidence if not e.get("quote")]
    quotes = [e for e in evidence if e.get("quote")]
    return {"figures": figures[:MAX_FIGURES], "figures_total": len(figures),
            "quotes": quotes[:MAX_QUOTES], "quotes_total": len(quotes)}


def _resolution(ctx) -> str:
    c = ctx.company
    return (f"Identified as {c.display_name or c.name}{f' (ticker {c.ticker})' if c.ticker else ''} in the SEC's register of public companies"
            + (f"; industry: {c.sic_description}" if c.sic_description else "") + ".")


def peer_text(ctx) -> str:
    """How the comparison group was chosen, or why there is none (no industry code, too few filers with data)."""
    if ctx.peers and ctx.peers.loaded:
        return P.plain(f"The {ctx.peers.method}.")
    return "No comparable peer group could be formed."


INTERNAL_NOTES = ("C.NG.04 asymmetry candidates",)  # hand-offs between tests saved by runs of earlier versions


def run_notes(ctx) -> list[str]:
    return [P.plain(n) for n in ctx.notes if not n.startswith(INTERNAL_NOTES)]


def flag_level(ctx) -> float:
    return float(ctx.tuning.get("scoring", {}).get("flag_severity", 0.5)) if ctx.tuning else 0.5


def finding_block(f, ctx, rank: int, flag_at: float, chart_years: int = 6) -> dict[str, Any]:
    """One finding's fixed set of blocks, as the templates show it."""
    summary, inv_status, inv_text = _investigation(f)
    plain_text = for_report(f)  # what we found / why unusual / what it could mean
    summary, _ = lint(summary)
    inv_text, _ = lint(inv_text)
    evidence = [_ev_dict(e, ctx) for e in _dedupe(f.evidence)]
    m = f.materiality
    groups: dict[str, list[str]] = {"fact": [], "span": [], "derived": [], "event": []}
    for e in f.evidence:
        groups.setdefault(e.kind, []).append(P.plain(e.label, expand=False))
    return {
        "rank": rank, "title": P.title(f.title), "score": round(f.score, 2), "strength": _strength(f.score),
        "tests": [P.name(d) for d in f.detectors],
        "signals": [{"name": P.name(s.detector_id), "severity": round(s.severity, 2),
                     "rationale": P.plain(s.rationale), "period": P.plain(s.period, expand=False),
                     "reading": P.plain(interpret("ok", s.severity, flag_at, s.flagged, s.percentile_peer, s.suppressed_by)),
                     **_about(s.detector_id)} for s in f.signals],
        "summary": P.plain(summary), "investigation_status": inv_status, "investigation": P.plain(inv_text),
        **{k: P.plain(plain_text[src], terms=plain_text["source"] != "ai")
           for k, src in (("what_found", "what"), ("why_unusual", "why"), ("what_it_means", "means"))},
        "impact_basis": P.plain(f.impact.basis) if f.impact else "",
        "impact_text": (f"{fmt_money(m.impact_low)} to {fmt_money(m.impact_high)} (central estimate {fmt_money(m.impact_mid)})"
                        if m else "Not quantified: this test measures a pattern, not an amount of profit."),
        "materiality_text": (
            f"{fmt_pct(m.pct_pretax)} of pre-tax income and {fmt_pct(m.pct_revenue, 2)} of revenue; "
            f"{m.multiple:.1f} times the materiality threshold an auditor would typically use (based on {BENCHMARKS.get(m.benchmark, m.benchmark)})"
            + (f"; effect on earnings per share {money_prefix()}{m.eps_effect:.2f}" if m.eps_effect is not None else "")
            + (f"; about {fmt_pct(m.pct_market_value)} of market value" if m.pct_market_value is not None else "")
            if m else "Not assessed, because no effect on profit was estimated."),
        "sab99_text": ("; ".join(P.plain(x) for x in m.sab99) + " (qualitative factors from SEC guidance on materiality)"
                       if m and m.sab99 else NONE_TEXT),
        **_evidence_blocks(evidence),
        "alternatives": [P.plain(x) for x in f.alternatives] or [NONE_TEXT],
        "questions": [P.plain(x) for x in f.questions] or [NONE_TEXT],
        "audit": _audit(f.audit),
        "audit_label": f.audit.label if f.audit else "Not independently re-checked",
        "chart": _ratio_chart(ctx, f.signals[0].detector_id, chart_years),
        "graph": charts.evidence_graph(P.title(f.title), [(k, v) for k, v in groups.items() if v]),
    }


def test_families(results, flag_at: float) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Every test's result, grouped by family, and the count of tests by status."""
    families = []
    for fam, _name in FAMILY_NAMES.items():
        rows = []
        for r in results:
            if r.family != fam:
                continue
            if r.signals:
                for s in r.signals:
                    status = "flagged" if s.flagged else "suppressed" if s.suppressed_by else "normal"
                    rows.append({"id": r.id, "name": P.name(r.id), "status": status, "status_label": P.STATUS_LABELS[status],
                                 "severity": round(s.severity, 2), "value": s.raw_value, "unit": s.unit,
                                 "period": P.plain(s.period, expand=False),
                                 "pct_peer": None if s.percentile_peer is None else round(s.percentile_peer),
                                 "detail": P.plain(s.rationale + (f" Explained by: {s.suppressed_by}." if s.suppressed_by else "")),
                                 "reading": P.plain(interpret("ok", s.severity, flag_at, s.flagged, s.percentile_peer, s.suppressed_by)),
                                 **_about(r.id)})
            else:
                status = r.status if r.status != "ok" else "normal"
                detail = P.plain(r.reason) if r.reason else "The test ran and found nothing unusual for this period."
                rows.append({"id": r.id, "name": P.name(r.id), "status": status, "status_label": P.STATUS_LABELS.get(status, status),
                             "severity": None, "value": None, "unit": "", "period": "", "pct_peer": None,
                             "detail": detail, "reading": detail, **_about(r.id)})
        seen: set[str] = set()
        for row in rows:  # a test with several results (periods) is explained once
            row["first"] = row["id"] not in seen
            seen.add(row["id"])
        families.append({"code": fam, "name": P.FAMILY_NAMES[fam], "rows": rows,
                         "flagged": sum(1 for x in rows if x["status"] == "flagged"),
                         "ran": sum(1 for x in rows if x["status"] in ("flagged", "normal", "suppressed"))})
    statuses = {"ok": 0, "skipped": 0, "n/a": 0, "error": 0}
    for r in results:
        statuses[r.status] = statuses.get(r.status, 0) + 1
    return families, statuses


def build_model(out) -> dict[str, Any]:
    ctx, sc = out.ctx, out.score
    flag_at = flag_level(ctx)
    findings = [finding_block(f, ctx, i + 1, flag_at) for i, f in enumerate(sc.findings[:12])]
    families, statuses = test_families(out.results, flag_at)
    text_changes = []
    for did in ("E.01", "E.02", "E.03", "E.04", "C.REV.07", "C.REV.11", "C.AR.04", "E.09"):
        for r in out.results:
            if r.id == did:
                for s in r.signals:
                    if s.evidence or s.flagged:
                        text_changes.append({"id": did, "name": P.name(did), "rationale": P.plain(s.rationale),
                                             "quotes": [_ev_dict(e, ctx) for e in s.evidence if e.kind == "span"][:3]})
    area_rows = sorted(sc.area_scores.items(), key=lambda kv: -kv[1])
    top_sentences = []
    for f in sc.findings[:12]:  # one line per reported finding, so the list always matches the Findings section
        mt = f" ({fmt_money(f.materiality.impact_low)}-{fmt_money(f.materiality.impact_high)}, {fmt_pct(f.materiality.pct_pretax)} of pre-tax income)" if f.materiality and f.materiality.pct_pretax is not None else ""
        top_sentences.append(f"{P.title(f.title)}{mt}.")
    key_numbers = [{"label": lab, "value": fmt_money(ctx.a(item)), "prior": fmt_money(ctx.a(item, 1))}
                   for lab, item in (("Revenue", "REVENUE"), ("Net income", "NET_INCOME"), ("Cash from operations", "CFO"),
                                     ("Total assets", "TOTAL_ASSETS"), ("Shareholders' equity", "EQUITY"))]
    llm = out.manifest.get("llm")
    peers_txt = peer_text(ctx)
    return {
        "company": {"name": ctx.company.display_name or ctx.company.name, "ticker": ctx.company.ticker,
                    "sic_description": " ".join((ctx.company.sic_description or "").split()),
                    "match_note": _resolution(ctx), "warnings": [P.plain(w) for w in ctx.company.warnings], "auditor": ctx.auditor},
        "period": P.plain(ctx.focus_label, expand=False).capitalize(), "period_end": str(ctx.focus.date()), "as_of": str(getattr(ctx, "cutoff", None) or ctx.as_of),
        "generated": datetime.now().strftime("%d %B %Y"),
        "analysis_type": ("Automated checks of the reported figures, plus AI-assisted reading of the filings' text"
                          if llm else "Automated checks of the reported figures and filing text"),
        "profile": ({"key": ctx.profile.key, "label": ctx.profile.label, "description": P.plain(ctx.profile.description),
                     "notes": [P.plain(n) for n in ctx.profile.notes], "excluded": len(ctx.profile.excluded)} if ctx.profile else
                    {"key": "industrial", "label": "Industrial / commercial company", "description": "Operating company; the full test set applies.",
                     "notes": [], "excluded": 0}),
        "currency": ctx.currency, "annual_form": ctx.annual_form,
        "risk": {"index": sc.risk_index, "band": sc.band},
        "top_sentences": top_sentences,
        "key_question": (findings[0]["questions"][0] if findings and findings[0]["questions"][0] != NONE_TEXT
                         else "None: no finding was reported." if not findings else NONE_TEXT),
        "key_numbers": key_numbers,
        "areas": [{"area": P.area(k), "score": round(v, 2)} for k, v in area_rows],
        "findings": findings,
        "audit_summary": {
            "verified": sum(1 for f in sc.findings[:12] if f.audit and f.audit.status == "verified"),
            "corrected": sum(1 for f in sc.findings[:12] if f.audit and f.audit.status == "verified_after_corrections"),
            "unverified": sum(1 for f in sc.findings[:12] if f.audit and f.audit.status == "unverified"),
        },
        "families": families,
        "statuses": statuses,
        "flag_at": flag_at,
        "text_changes": text_changes,
        "pressure": [{"id": s.detector_id, "name": P.name(s.detector_id), "rationale": P.plain(s.rationale)} for s in sc.pressure],
        "explain_conditions": {k: P.plain(v) for k, v in sc.explain_conditions.items()},
        "charts": {"earnings_cash": _earnings_cash_chart(out), "scores": _scores_chart(out)},
        "peers": [{"name": p.name} for p in (ctx.peers.loaded if ctx.peers else [])],
        "peer_method": peers_txt,
        "documents": [{"url": d.url,
                       "label": (f"Earnings release, {d.period_end}" if d.form == "EX-99" else
                                 f"{P.source_label(d.form, None, 'span')} for the fiscal year ended {d.period_end}")}
                      for d in ctx.docs + ctx.releases],
        "notes": run_notes(ctx) or [NONE_TEXT],
        "none_text": NONE_TEXT,
        "glossary": P.GLOSSARY,
        "disclaimer": DISCLAIMER,
    }
