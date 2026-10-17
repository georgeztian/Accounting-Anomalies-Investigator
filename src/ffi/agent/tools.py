"""Read-only, deterministic tools the agent can call. Python computes every number."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from ffi.agent.schemas import EvidenceTask
from ffi.context import RunContext
from ffi.detectors.d_stats import RATIOS, ratio_vector
from ffi.detectors.scores import V
from ffi.detectors.stats import fmt_money, fmt_num, fmt_pct, safe_div
from ffi.text.search import search
from ffi.xbrl.canonical import BY_ID

# XBRL "IncreaseDecrease" concepts report the change in the balance, not its effect on cash
SIGN_NOTES = {
    "CHANGE_AR": "positive = receivables rose, which used cash",
    "CHANGE_INVENTORY": "positive = inventory rose, which used cash",
    "CHANGE_AP": "positive = payables rose, which added cash",
    "CHANGE_DEFERRED_REV": "positive = deferred revenue rose, which added cash",
}


def _entry(lid: str, task: EvidenceTask, text: str, numbers: list[float], evidence: list[dict]) -> dict[str, Any]:
    return {"id": lid, "tool": task.tool, "purpose": task.purpose, "hypothesis": task.hypothesis_index,
            "text": text, "numbers": [float(n) for n in numbers if n is not None and np.isfinite(n)], "evidence": evidence}


def run_tool(ctx: RunContext, task: EvidenceTask, lid: str) -> dict[str, Any]:
    t = task.tool
    if t == "get_fact":
        item = (task.item or "").upper()
        if item not in BY_ID:
            return _entry(lid, task, f"Unknown line item '{task.item}'.", [], [])
        vals, ev = [], []
        for k in range(ctx.history):
            v = ctx.a(item, k)
            if np.isfinite(v):
                vals.append((ctx.label(ctx.period_at(k)), v))
                e = ctx.ev(item, k)
                if e:
                    ev.append(e.to_dict())
        fmt = fmt_money if BY_ID[item].unit == "USD" else (lambda x: fmt_num(x, 3))
        text = f"{BY_ID[item].label}: " + ", ".join(f"{lab} {fmt(v)}" for lab, v in vals) if vals else f"{BY_ID[item].label} not reported."
        if vals and item in SIGN_NOTES:
            text += f" (sign: {SIGN_NOTES[item]})"
        return _entry(lid, task, text, [v for _, v in vals], ev)
    if t == "ratio":
        n, d = (task.numerator or "").upper(), (task.denominator or "").upper()
        if n not in BY_ID or d not in BY_ID:
            return _entry(lid, task, "Unknown numerator/denominator.", [], [])
        vals = [(ctx.label(ctx.period_at(k)), safe_div(ctx.a(n, k), ctx.a(d, k))) for k in range(ctx.history)]
        vals = [(lab, v) for lab, v in vals if np.isfinite(v)]
        text = f"{BY_ID[n].label} / {BY_ID[d].label}: " + ", ".join(f"{lab} {fmt_pct(v)}" for lab, v in vals)
        ev = [e.to_dict() for e in ctx.evs([n, d], ks=(0, 1))]
        return _entry(lid, task, text, [v for _, v in vals] + [v * 100 for _, v in vals], ev)
    if t == "peer_stats":
        key = task.ratio_key or ""
        if key not in RATIOS:
            return _entry(lid, task, f"Unknown ratio '{key}'. Valid keys: {', '.join(RATIOS)}.", [], [])
        name, fn = RATIOS[key]
        mine = float(fn(V(ctx.A, ctx.focus)))
        peers = ctx.peer_values(lambda P, p, k: float(fn(V(P, p))))
        if len(peers) < 3:
            return _entry(lid, task, f"{name}: company {fmt_num(mine, 3)}; too few peers.", [mine], [])
        q1, med, q3 = np.quantile(peers, [0.25, 0.5, 0.75])
        text = (f"{name}, {ctx.focus_label}: company {fmt_num(mine, 3)} vs peer median {fmt_num(med, 3)} (IQR {fmt_num(q1, 3)} to "
                f"{fmt_num(q3, 3)}, n={len(peers)} same-industry peers, each at its fiscal year aligned to {ctx.focus_label}; one year only).")
        return _entry(lid, task, text, [mine, med, q1, q3], [])
    if t in ("search_filing", "explaining_text"):
        doc = ctx.doc
        if doc is None:
            return _entry(lid, task, "No 10-K text available.", [], [])
        if t == "explaining_text":
            item = (task.item or "").upper()
            from ffi.kg.build import item_pattern

            pat = item_pattern(item)
            if not pat:
                return _entry(lid, task, f"Unknown line item {item}: MD&A was not searched; use search_filing with words instead.", [], [])
            hits = search(doc, pat, ("item7",))
            if not hits:
                return _entry(lid, task, f"MD&A searched for {item}: no passage mentions it.", [], [])
        else:
            words = [re.escape(w) for w in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", task.query or "")][:6]
            if not words:
                return _entry(lid, task, "Empty query.", [], [])
            pat = r"\b(?:" + "|".join(words) + r")"
            secs = None if task.section in (None, "any") else (task.section,)
            hits = search(doc, pat, secs)
            hits.sort(key=lambda h: -sum(1 for w in words if re.search(w, h.quote, re.I)))
        hits = hits[:3]
        ev = [ctx.ev_span(doc, h.start, h.end, f"10-K {h.paragraph.section}").to_dict() for h in hits]
        text = " | ".join(f'"{h.quote}"' for h in hits) if hits else "No matching passages."
        nums = [float(x.replace(",", "")) for h in hits for x in re.findall(r"\d[\d,]*\.?\d*", h.quote)][:20]
        return _entry(lid, task, text, nums, ev)
    if t == "get_events":
        types = [x.upper() for x in (task.event_types or [])] or None
        evs = [e for e in ctx.events if types is None or e.type in types]
        text = "; ".join(f"{e.type} {e.date}: {e.description}" for e in evs[:8]) or "No matching events."
        return _entry(lid, task, text, [], [ctx.ev_event(e).to_dict() for e in evs[:4]])
    if t == "diff_risk_factors":
        sig = ctx.signals.get("E.03", [])
        if not sig:
            return _entry(lid, task, "Risk-factor comparison not available.", [], [])
        s = sig[0]
        return _entry(lid, task, s.rationale, [], [e.to_dict() for e in s.evidence])
    return _entry(lid, task, f"Unknown tool {t}.", [], [])


def period_context(ctx: RunContext) -> list[str]:
    """When each fiscal year ended and how long it was (52/53-week calendars), so neither the drafter nor the auditor
    has to assume it."""
    out = []
    for k in range(ctx.history):
        p, prev = ctx.period_at(k), ctx.period_at(k + 1)
        if p is None:
            break
        weeks = f", {round((p - prev).days / 7)} weeks" if prev is not None else ""
        out.append(f"{ctx.label(p)} ended {p.date().isoformat()}{weeks}")
    return out


def facts_table(ctx: RunContext, finding) -> list[dict[str, Any]]:
    """Every number the narrative may use, as formatted strings + raw values (the grounding gate's allow-list)."""
    rows: list[dict[str, Any]] = []
    for s in finding.signals:
        # the result and the test's reference value are labelled apart, so a baseline of 0 is never read as a second result
        for v, what in ((s.raw_value, "result"), (s.expected, "reference value (what would be normal)")):
            if v is not None and np.isfinite(v):
                rows.append({"source": f"{s.detector_id} {what}, {s.period}", "value": float(v), "unit": s.unit})
        if s.raw_value is not None and s.expected is not None and np.isfinite(s.raw_value - s.expected):
            rows.append({"source": f"{s.detector_id} result minus reference value, {s.period}", "value": float(s.raw_value - s.expected), "unit": s.unit})
        for e in s.evidence:
            if e.value is not None:
                note = next((n for i, n in SIGN_NOTES.items() if e.label.startswith(BY_ID[i].label)), None)
                rows.append({"source": e.label + (f" (sign: {note})" if note else ""), "value": float(e.value), "unit": e.unit or ""})
    # multi-year anomaly: the lead test's result and severity in every year reviewed, flagged or not
    for t in getattr(finding, "timeline", None) or []:
        state = "flagged" if t["flagged"] else "not flagged"
        if t["lead_value"] is not None and np.isfinite(t["lead_value"]):
            rows.append({"source": f"{t['lead_test']} result, {t['period']} ({state})", "value": float(t["lead_value"]), "unit": t["lead_unit"]})
        if t["lead_severity"] is not None:
            rows.append({"source": f"{t['lead_test']} severity, {t['period']} ({state})", "value": float(t["lead_severity"]), "unit": "0-1"})
    # year-on-year changes of the cited line items, so the narrative never has to do its own arithmetic
    # evidence labels read "<line item>, <period>[ (derived ...)]"; line-item labels can hold commas ("PP&E, net")
    labels = {it.label: i for i, it in BY_ID.items()}
    cited = {e.label.rsplit(", ", 1)[0] for s in finding.signals for e in s.evidence if e.kind == "fact"}
    items = {labels[c] for c in cited if c in labels}
    for item in sorted(items):
        for k in range(max(3, ctx.history - 1)):
            v1, v0 = ctx.a(item, k), ctx.a(item, k + 1)
            if np.isfinite(v1) and np.isfinite(v0):
                span = f"{ctx.label(ctx.period_at(k))} vs {ctx.label(ctx.period_at(k + 1))}"
                rows.append({"source": f"{BY_ID[item].label} change, {span}", "value": float(v1 - v0),
                             "unit": BY_ID[item].unit.replace("USD", ctx.currency)})
                if v0:
                    rows.append({"source": f"{BY_ID[item].label} % change, {span}", "value": float((v1 - v0) / abs(v0)), "unit": "ratio"})
    m = finding.materiality
    if m:
        for k in ("impact_low", "impact_mid", "impact_high", "net_income_effect", "pct_pretax", "pct_revenue", "pct_assets",
                  "pct_equity", "multiple", "eps_effect", "pct_market_value"):
            v = getattr(m, k)
            if v is not None:
                rows.append({"source": f"materiality.{k}", "value": float(v), "unit": ""})
    for item in ("REVENUE", "NET_INCOME", "PRETAX_INCOME", "CFO", "TOTAL_ASSETS", "EQUITY"):
        for k in (0, 1):
            v = ctx.a(item, k)
            if np.isfinite(v):
                rows.append({"source": f"{item} {ctx.label(ctx.period_at(k))}", "value": float(v), "unit": ctx.currency})
    rv = ratio_vector(ctx.A, ctx.focus)
    for key, v in rv.items():
        if np.isfinite(v):
            rows.append({"source": f"ratio.{key}", "value": float(v), "unit": ""})
    return rows
