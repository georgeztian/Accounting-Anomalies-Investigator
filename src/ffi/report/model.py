"""Assemble the report data model from a run (BLUEPRINT §12.1)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from ffi import __version__
from ffi.detectors import scores as S
from ffi.detectors.d_stats import RATIOS
from ffi.detectors.scores import V
from ffi.detectors.stats import fmt_money, fmt_num, fmt_pct, money_prefix
from ffi.report import charts
from ffi.report.linter import lint
from ffi.scoring.engine import template_narrative

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


def _ratio_chart(out, detector_id: str) -> str:
    if detector_id not in DETECTOR_RATIO:
        return ""
    key, kind = DETECTOR_RATIO[detector_id]
    name, fn = RATIOS[key]
    ctx = out.ctx
    periods = ctx.periods[-6:]
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
    return charts.line_chart(f"{name}: company vs peers", labels, series, kind, band)


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


def _ev_dict(e) -> dict:
    d = e.to_dict()
    if e.value is not None:
        is_money = (e.unit is not None and len(e.unit) == 3 and e.unit.isalpha() and e.unit.isupper()) or (e.unit is None and e.kind == "fact")
        d["value_fmt"] = fmt_money(e.value) if is_money else fmt_num(e.value, 3)
    return d


def _dedupe(evidence: list) -> list:
    seen, out = set(), []
    for e in evidence:
        if e.node_id in seen:
            continue
        seen.add(e.node_id)
        out.append(e)
    return out


def build_model(out) -> dict[str, Any]:
    ctx, sc = out.ctx, out.score
    lint_hits: list[str] = []
    findings = []
    for i, f in enumerate(sc.findings[:12]):
        narrative = f.narrative or template_narrative(f)
        narrative, hits = lint(narrative)
        lint_hits += hits
        m = f.materiality
        groups = {"fact": [], "span": [], "derived": [], "event": []}
        for e in f.evidence:
            groups.setdefault(e.kind, []).append(e.label)
        findings.append({
            "rank": i + 1, "id": f.id, "title": f.title, "kind": f.kind, "score": round(f.score, 2),
            "priority": round(f.priority, 3), "detectors": f.detectors,
            "signals": [{"id": s.detector_id, "name": s.name, "family": s.family, "severity": round(s.severity, 2),
                         "rationale": s.rationale, "period": s.period} for s in f.signals],
            "narrative": narrative, "narrative_source": f.narrative_source,
            "impact": f.impact.to_dict() if f.impact else None,
            "materiality": m.to_dict() if m else None,
            "impact_text": (f"{fmt_money(m.impact_low)} to {fmt_money(m.impact_high)} (mid {fmt_money(m.impact_mid)})" if m else "Not quantified"),
            "materiality_text": (
                f"{fmt_pct(m.pct_pretax)} of pre-tax income, {fmt_pct(m.pct_revenue, 2)} of revenue; "
                f"{m.multiple:.1f}x the {m.benchmark} materiality threshold"
                + (f"; EPS effect {money_prefix()}{m.eps_effect:.2f}" if m.eps_effect is not None else "")
                + (f"; about {fmt_pct(m.pct_market_value)} of market value" if m.pct_market_value is not None else "")
                if m else ""),
            "sab99": m.sab99 if m else [],
            "evidence": [_ev_dict(e) for e in _dedupe(f.evidence)],
            "alternatives": f.alternatives, "questions": f.questions,
            "audit": f.audit.to_dict() if f.audit is not None else None,
            "chart": _ratio_chart(out, f.signals[0].detector_id),
            "graph": charts.evidence_graph(f.title, [(k, v) for k, v in groups.items() if v]) if i < 5 else "",
        })
    families = []
    for fam, name in FAMILY_NAMES.items():
        rows = []
        for r in out.results:
            if r.family != fam:
                continue
            if r.signals:
                for s in r.signals:
                    rows.append({"id": r.id, "name": r.name, "status": ("flagged" if s.flagged else "suppressed" if s.suppressed_by else "normal"),
                                 "severity": round(s.severity, 2), "value": s.raw_value, "unit": s.unit, "period": s.period,
                                 "pct_peer": None if s.percentile_peer is None else round(s.percentile_peer),
                                 "detail": s.rationale + (f" Suppressed: {s.suppressed_by}." if s.suppressed_by else "")})
            else:
                rows.append({"id": r.id, "name": r.name, "status": r.status if r.status != "ok" else "normal", "severity": None,
                             "value": None, "unit": "", "period": "", "pct_peer": None,
                             "detail": r.reason or "No signal produced."})
        families.append({"code": fam, "name": name, "rows": rows,
                         "flagged": sum(1 for x in rows if x["status"] == "flagged"),
                         "ran": sum(1 for x in rows if x["status"] in ("flagged", "normal", "suppressed"))})
    statuses = {"ok": 0, "skipped": 0, "n/a": 0, "error": 0}
    for r in out.results:
        statuses[r.status] = statuses.get(r.status, 0) + 1
    text_changes = []
    for did in ("E.01", "E.02", "E.03", "E.04", "C.REV.07", "C.REV.11", "C.AR.04", "E.09"):
        for r in out.results:
            if r.id == did:
                for s in r.signals:
                    if s.evidence or s.flagged:
                        text_changes.append({"id": did, "name": r.name, "rationale": s.rationale,
                                             "quotes": [_ev_dict(e) for e in s.evidence if e.kind == "span"][:3]})
    area_rows = sorted(sc.area_scores.items(), key=lambda kv: -kv[1])
    top_sentences = []
    for f in sc.findings[:5]:
        mt = f" ({fmt_money(f.materiality.impact_low)}-{fmt_money(f.materiality.impact_high)}, {fmt_pct(f.materiality.pct_pretax)} of pre-tax income)" if f.materiality and f.materiality.pct_pretax is not None else ""
        top_sentences.append(f"{f.title}{mt}.")
    key_numbers = [{"label": lab, "value": fmt_money(ctx.a(item)), "prior": fmt_money(ctx.a(item, 1))}
                   for lab, item in (("Revenue", "REVENUE"), ("Net income", "NET_INCOME"), ("Cash from operations", "CFO"),
                                     ("Total assets", "TOTAL_ASSETS"), ("Shareholders' equity", "EQUITY"))]
    llm = out.manifest.get("llm")
    integrity = [x for fam in families if fam["code"] == "A" for x in fam["rows"]]
    return {
        "title": f"{ctx.company.name} ({ctx.company.ticker}) - accounting anomaly analysis, {ctx.focus_label}",
        "company": {"name": ctx.company.name, "ticker": ctx.company.ticker, "cik": ctx.company.cik,
                    "sic": ctx.company.sic, "sic_description": ctx.company.sic_description,
                    "match_note": ctx.company.match_note, "warnings": ctx.company.warnings, "auditor": ctx.auditor},
        "period": ctx.focus_label, "period_end": str(ctx.focus.date()), "as_of": str(ctx.as_of),
        "run_id": out.manifest["run_id"], "generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "version": __version__,
        "mode": ctx.mode,
        "profile": ({"key": ctx.profile.key, "label": ctx.profile.label, "description": ctx.profile.description,
                     "notes": ctx.profile.notes, "excluded": len(ctx.profile.excluded)} if ctx.profile else None),
        "currency": ctx.currency, "annual_form": ctx.annual_form,
        "risk": {"index": sc.risk_index, "band": sc.band, "probability": sc.probability,
                 "peer_note": ctx.peers.method if ctx.peers else "no peer set"},
        "top_sentences": top_sentences,
        "key_numbers": key_numbers,
        "areas": [{"area": k, "score": round(v, 2)} for k, v in area_rows],
        "contributions": [{"name": k, "weight": round(v, 3)} for k, v in sc.contributions[:10]],
        "findings": findings,
        "audit_summary": {
            "verified": sum(1 for f in sc.findings[:12] if f.audit and f.audit.status == "verified"),
            "corrected": sum(1 for f in sc.findings[:12] if f.audit and f.audit.status == "verified_after_corrections"),
            "unverified": sum(1 for f in sc.findings[:12] if f.audit and f.audit.status == "unverified"),
        },
        "families": families,
        "statuses": statuses,
        "text_changes": text_changes,
        "pressure": [{"id": s.detector_id, "name": s.name, "rationale": s.rationale} for s in sc.pressure],
        "integrity": integrity,
        "explain_conditions": sc.explain_conditions,
        "charts": {"earnings_cash": _earnings_cash_chart(out), "scores": _scores_chart(out)},
        "peers": [{"cik": p.cik, "name": p.name, "distance": round(p.distance, 2), "loaded": p.annual is not None}
                  for p in (ctx.peers.peers if ctx.peers else [])],
        "peer_method": ctx.peers.method if ctx.peers else "no peer set",
        "industry": {"digits": ctx.industry_digits, "size": len(ctx.industry_ciks)},
        "documents": [{"accession": d.accession, "form": d.form, "period": d.period_end, "url": d.url} for d in ctx.docs + ctx.releases],
        "notes": ctx.notes,
        "agent": [c if isinstance(c, dict) else getattr(c, "__dict__", {}) for c in out.agent_cases],
        "llm": llm,
        "kg": out.manifest.get("kg", {}),
        "timings": out.timings,
        "lint_hits": lint_hits,
        "disclaimer": DISCLAIMER,
        "manifest": out.manifest,
    }
