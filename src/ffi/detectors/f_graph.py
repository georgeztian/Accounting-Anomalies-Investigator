"""Family F: graph-native detectors over the knowledge graph."""

from __future__ import annotations

import networkx as nx
import numpy as np

from ffi.context import RunContext
from ffi.detectors.base import Area, EvidenceRef, Scheme, SkipDetector, detector
from ffi.detectors.helpers import finite, mk
from ffi.detectors.stats import fmt_money, fmt_pct, safe_div
from ffi.xbrl.canonical import BY_ID

F = "F"
LARGE_AUDITORS = ("ernst & young", "deloitte", "pricewaterhousecoopers", "kpmg", "grant thornton", "bdo", "rsm", "crowe",
                  "baker tilly", "moss adams", "marcum", "cbiz", "forvis")


def _resolve_counterparty(ctx: RunContext, name: str) -> int | None:
    if ctx.client is None or name.lower() in ("unnamed", "customer a", "customer b"):
        return None
    from ffi.resolve.resolver import _name_candidates, company_tickers_frame

    cands = _name_candidates(name, company_tickers_frame(ctx.client), limit=2)
    if cands and cands[0]["score"] >= 92 and cands[0]["cik"] != ctx.company.cik:
        return int(cands[0]["cik"])
    return None


@detector("F.01", "Counterparty asymmetry", F, Area.RELATED_PARTY, depends_on=("E.09",), needs_llm=True)
def f01(ctx: RunContext):
    """Sales disclosed to a major customer that is an SEC filer vs that customer's own total cost base."""
    from ffi.xbrl.facts import FactStore

    cust = [(r, loc) for r, loc in ctx.relationships if r.kind == "major_customer"]
    if not cust:
        return [mk("F.01", ctx, raw_value=0, severity=0.0, rationale="No named major customers disclosed.")]
    rev = ctx.a("REVENUE")
    findings, checked = [], 0
    for r, loc in cust[:5]:
        # a percentage converts to dollars only when it is of the whole company's revenue: "13% of a subsidiary's
        # revenues" times consolidated revenue would overstate the sales many times over
        whole = r.share_basis == "whole company"
        amt = r.amount_usd or (r.share_of_revenue_pct / 100 * rev if r.share_of_revenue_pct and whole and finite(rev) else None)
        cik = _resolve_counterparty(ctx, r.counterparty)
        if amt is None or cik is None:
            continue
        try:
            fs = FactStore.load(ctx.client, cik, ctx.as_of)
        except Exception:  # noqa: BLE001
            continue
        if fs.currency != ctx.currency:
            continue  # sales in one currency against costs in another
        a = fs.annual()
        if a.values.empty:
            continue
        p = a.values.index[np.argmin(np.abs((a.values.index - ctx.focus).days))]
        if abs((p - ctx.focus).days) > 183:
            continue  # no fiscal year of the customer's that overlaps the focus year
        # what the customer spends in a year: all operating costs (revenue less operating income) plus capital spending;
        # summing cost lines misses businesses that report costs under other headings (e.g. restaurant expenses)
        opex = a.get("REVENUE", p) - a.get("OPERATING_INCOME", p)
        if not finite(opex) or opex <= 0:
            opex = sum(a.get(i, p) for i in ("COGS", "SGA", "RND") if finite(a.get(i, p)))
        costs = opex + (a.get("CAPEX", p) if finite(a.get("CAPEX", p)) else 0.0)
        checked += 1
        share = safe_div(amt, costs)
        ctx.notes.append(f"F.01 counterparty {r.counterparty} -> CIK {cik}")
        if finite(share) and share > 0.5:
            findings.append((r, loc, cik, amt, costs, share))
    if checked == 0:
        return [mk("F.01", ctx, raw_value=0, severity=0.0, rationale=f"{len(cust)} major customer(s) disclosed; none resolvable to an SEC filer with amounts.")]
    sev = min(1.0, 0.6 * len(findings))
    return [mk("F.01", ctx, raw_value=len(findings), severity=sev, unit="asymmetries", scheme=Scheme.BOGUS_REVENUE,
               direction="income_increasing",
               rationale=(f"{checked} customer(s) resolved to SEC filers; " + ("; ".join(
                   f"sales of {fmt_money(amt)} to {r.counterparty} equal {fmt_pct(share)} of everything that customer spends in a year "
                   f"({fmt_money(costs)} of operating costs and capital spending)"
                   for r, _, _, amt, costs, share in findings) if findings else "disclosed sales are plausible relative to their cost bases") + "."),
               evidence=[ctx.ev_span(ctx.doc, loc[0], loc[1], f"Major customer: {r.counterparty}") for r, loc, *_ in findings[:2]])]


@detector("F.02", "Circular flows", F, Area.RELATED_PARTY, depends_on=("E.09",), needs_llm=True)
def f02(ctx: RunContext):
    """Directed cycles (length <= 4) over sales, related-party, guarantee and side-agreement edges."""
    g = ctx.kg.graph(["SELLS_TO", "BUYS_FROM", "RELATED_PARTY", "GUARANTEES", "SIDE_AGREEMENT"])
    if g.number_of_edges() == 0:
        return [mk("F.02", ctx, raw_value=0, severity=0.0, rationale="No relationship edges to analyse.")]
    dg = nx.DiGraph()
    for u, v, d in g.edges(data=True):
        if d.get("type") == "BUYS_FROM":
            dg.add_edge(v, u)
        else:
            dg.add_edge(u, v)
        if d.get("type") == "RELATED_PARTY":
            dg.add_edge(v, u)  # related-party relationships are two-way
    cycles = [c for c in nx.simple_cycles(dg, length_bound=4) if len(c) >= 3]
    sev = min(1.0, 0.5 * len(cycles))
    return [mk("F.02", ctx, raw_value=len(cycles), severity=sev, unit="cycles", scheme=Scheme.BOGUS_REVENUE,
               rationale=(f"{len(cycles)} circular flow(s) through: " + "; ".join(" -> ".join(c) for c in cycles[:2]) + ".") if cycles
               else f"No circular flows among {dg.number_of_nodes()} related entities.")]


@detector("F.03", "Auditor profile and links", F, Area.GOVERNANCE)
def f03(ctx: RunContext):
    """Auditor size relative to the company, plus auditor-change events linked in the graph."""
    if not ctx.auditor:
        raise SkipDetector("auditor not identified in the 10-K")
    large = any(k in ctx.auditor.lower() for k in LARGE_AUDITORS)
    ta = ctx.a("TOTAL_ASSETS")
    big_company = finite(ta) and ta > 2e9
    changes = ctx.events_of("AUDITOR_CHANGE", within_days=1095)
    sev = (0.45 if (big_company and not large) else 0.0) + 0.2 * min(len(changes), 2)
    return [mk("F.03", ctx, raw_value=len(changes), severity=min(sev, 0.8), unit="", tags=["pressure"],
               rationale=(f"Auditor: {ctx.auditor} ({'large/national firm' if large else 'smaller firm'}) for a company with "
                          f"{fmt_money(ta)} of assets; {len(changes)} auditor change(s) in three years."),
               evidence=[ctx.ev_event(e) for e in changes[:2]])]


@detector("F.04", "Evidence-coverage gaps", F, Area.DISCLOSURE, needs_text=True)
def f04(ctx: RunContext):
    """Material line-item movements with no EXPLAINS edge from MD&A or notes: unexplained changes."""
    if ctx.doc is None:
        raise SkipDetector("10-K text not available")
    if len(ctx.doc.section_paragraphs("item7")) < 20:
        raise SkipDetector("MD&A section not located reliably in the 10-K text")
    edges = ctx.kg.edges("EXPLAINS")
    # only this year's annual report can explain this year's changes; earlier years' MD&A discussed earlier changes
    if not edges.empty:
        edges = edges[edges["src"].str.contains(ctx.doc.accession, regex=False)]
    explained = set(edges["dst"].str.replace("lineitem:", "", regex=False)) if not edges.empty else set()
    pt = abs(ctx.a("PRETAX_INCOME"))
    rev = abs(ctx.a("REVENUE"))
    mat = max(0.05 * pt if finite(pt) else 0, 0.005 * rev if finite(rev) else 0)
    if mat <= 0:
        raise SkipDetector("no materiality base")
    gaps, checked = [], 0
    for item in ("REVENUE", "COGS", "SGA", "RND", "OPERATING_INCOME", "AR", "INVENTORY", "GOODWILL", "DEFERRED_REV_CURRENT",
                 "LONG_TERM_DEBT", "RESTRUCTURING", "INCOME_TAX", "CAPEX"):
        a, b = ctx.a(item), ctx.a(item, 1)
        if not finite(a, b):
            continue
        d = abs(a - b)
        if d < mat or safe_div(d, abs(b)) < 0.10:
            continue
        checked += 1
        if item not in explained:
            gaps.append((item, d))
    score = sum(d for _, d in gaps) / mat if gaps else 0.0
    sev = float(np.clip(score / 10, 0, 1)) * 0.7
    return [mk("F.04", ctx, raw_value=len(gaps), severity=sev, unit="unexplained items",
               rationale=(f"{checked} line item{'' if checked == 1 else 's'} moved materially (> {fmt_money(mat)} and > 10%); "
                          + (f"{len(gaps)} {'has' if len(gaps) == 1 else 'have'} no discussion in this year's MD&A or notes: "
                             + ", ".join(f"{BY_ID[i].label} ({fmt_money(d)})" for i, d in gaps) + "." if gaps else "all are discussed.")),
               evidence=[e for i, _ in gaps[:3] for e in ctx.evs([i], ks=(0, 1))])]


@detector("F.05", "Contradiction density", F, Area.DISCLOSURE, depends_on=("A.15", "E.04", "E.12"))
def f05(ctx: RunContext):
    """Text-vs-number contradictions (A.15, E.04, E.12) weighted together."""
    n = 0
    ev: list[EvidenceRef] = []
    for did in ("A.15", "E.04", "E.12"):
        hits = [s for s in ctx.signals.get(did, []) if s.flagged]
        if hits:
            n += 1
            ev += hits[0].evidence[:1]
    if not any(ctx.signals.get(d) for d in ("A.15", "E.04", "E.12")):
        raise SkipDetector("no contradiction detectors ran")
    return [mk("F.05", ctx, raw_value=n, severity=min(1.0, 0.35 * n), unit="of 3 text-vs-number checks", tags=["derived"],
               rationale=f"{n} of the 3 text-vs-number checks (MD&A totals, MD&A claims, earnings releases) flagged an inconsistency.",
               evidence=ev[:3])]


@detector("F.06", "Counterparty stress", F, Area.RELATED_PARTY, depends_on=("F.01",), needs_llm=True)
def f06(ctx: RunContext):
    """Resolved major customers with going-concern, non-reliance or late-filing events."""
    from ffi.edgar import endpoints

    ciks = [int(n.split("CIK ")[-1]) for n in ctx.notes if n.startswith("F.01 counterparty")]
    if not ciks:
        return [mk("F.06", ctx, raw_value=0, severity=0.0, rationale="No counterparties resolved to SEC filers.")]
    stressed = []
    for cik in ciks:
        try:
            sub = endpoints.submissions(ctx.client, cik)
        except Exception:  # noqa: BLE001
            continue
        rec = sub.get("filings", {}).get("recent", {})
        forms, items = rec.get("form", []), rec.get("items", [])
        bad = [f for f, it in zip(forms[:200], items[:200], strict=False) if f in ("NT 10-K", "NT 10-Q") or "4.02" in (it or "")]
        if bad:
            stressed.append((sub.get("name", cik), len(bad)))
    sev = min(0.8, 0.4 * len(stressed))
    return [mk("F.06", ctx, raw_value=len(stressed), severity=sev, unit="stressed counterparties", tags=["pressure"],
               rationale=("Counterparties with late filings or non-reliance: " + ", ".join(f"{n} ({k})" for n, k in stressed) + ".") if stressed
               else f"{len(ciks)} resolved counterparties show no late-filing or non-reliance events.")]


@detector("F.07", "Unusual new relationships", F, Area.RELATED_PARTY, depends_on=("E.09",), needs_llm=True)
def f07(ctx: RunContext):
    """New related-party, guarantee, VIE or side-agreement edges this year."""
    new = [(r, loc) for r, loc in ctx.relationships if r.is_new_this_year and r.kind in ("related_party", "guarantee", "vie", "side_agreement")]
    sev = min(0.8, 0.3 * len(new))
    return [mk("F.07", ctx, raw_value=len(new), severity=sev, unit="new edges", scheme=Scheme.HIDE_LIABILITIES,
               rationale=("New this year: " + "; ".join(f"{r.kind.replace('_', ' ')} with {r.counterparty}" for r, _ in new[:4]) + ".") if new
               else "No new related-party, guarantee, VIE or side-agreement relationships.",
               evidence=[ctx.ev_span(ctx.doc, loc[0], loc[1], f"New {r.kind.replace('_', ' ')}") for r, loc in new[:3]])]
