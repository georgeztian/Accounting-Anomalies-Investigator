"""Read-only, deterministic tools the agent can call (BLUEPRINT §13.6). Python computes every number."""

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
        for k in range(4):
            v = ctx.a(item, k)
            if np.isfinite(v):
                vals.append((ctx.label(ctx.period_at(k)), v))
                e = ctx.ev(item, k)
                if e:
                    ev.append(e.to_dict())
        fmt = fmt_money if BY_ID[item].unit == "USD" else (lambda x: fmt_num(x, 3))
        text = f"{BY_ID[item].label}: " + ", ".join(f"{lab} {fmt(v)}" for lab, v in vals) if vals else f"{BY_ID[item].label} not reported."
        return _entry(lid, task, text, [v for _, v in vals], ev)
    if t == "ratio":
        n, d = (task.numerator or "").upper(), (task.denominator or "").upper()
        if n not in BY_ID or d not in BY_ID:
            return _entry(lid, task, "Unknown numerator/denominator.", [], [])
        vals = [(ctx.label(ctx.period_at(k)), safe_div(ctx.a(n, k), ctx.a(d, k))) for k in range(4)]
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
        text = f"{name}: company {fmt_num(mine, 3)} vs peer median {fmt_num(med, 3)} (IQR {fmt_num(q1, 3)} to {fmt_num(q3, 3)}, n={len(peers)})."
        return _entry(lid, task, text, [mine, med, q1, q3], [])
    if t in ("search_filing", "explaining_text"):
        doc = ctx.doc
        if doc is None:
            return _entry(lid, task, "No 10-K text available.", [], [])
        if t == "explaining_text":
            item = (task.item or "").upper()
            from ffi.kg.build import ITEM_KEYWORDS

            pat = ITEM_KEYWORDS.get(item)
            if not pat:
                return _entry(lid, task, f"No keyword pattern for {item}.", [], [])
            hits = search(doc, pat, ("item7",))
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


def facts_table(ctx: RunContext, finding) -> list[dict[str, Any]]:
    """Every number the narrative may use, as formatted strings + raw values (the grounding gate's allow-list)."""
    rows: list[dict[str, Any]] = []
    for s in finding.signals:
        for v, kind in ((s.raw_value, s.unit), (s.expected, s.unit)):
            if v is not None and np.isfinite(v):
                rows.append({"source": s.detector_id, "value": float(v), "unit": kind})
        for e in s.evidence:
            if e.value is not None:
                rows.append({"source": e.label, "value": float(e.value), "unit": e.unit or ""})
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
                rows.append({"source": f"{item} {ctx.label(ctx.period_at(k))}", "value": float(v), "unit": "USD"})
    rv = ratio_vector(ctx.A, ctx.focus)
    for key, v in rv.items():
        if np.isfinite(v):
            rows.append({"source": f"ratio.{key}", "value": float(v), "unit": ""})
    return rows
