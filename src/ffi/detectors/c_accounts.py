"""Family C: area-specific detectors.

Most detectors follow one pattern: compute a ratio at the focus year, compare it with the company's
own history and with peers (robust z-scores), and size the dollar gap
    Impact = (actual ratio - baseline ratio) x driver
where the baseline is the more conservative of the self median and the peer median.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ffi.context import RunContext
from ffi.detectors.base import Area, ImpactEstimate, Scheme, SkipDetector, detector
from ffi.detectors.helpers import finite, mk, need
from ffi.detectors.scores import V
from ffi.detectors.stats import (
    fmt_money,
    fmt_num,
    fmt_pct,
    percentile_of,
    robust_z,
    safe_div,
    severity_from_z,
    z_to_p,
)
from ffi.text.search import count, new_mentions, search

F = "C"
Ratio = Callable[[V], float]


@dataclass
class Assessment:
    cur: float
    self_hist: list[float]
    peers: list[float]
    z_self: float
    z_peer: float
    pct_peer: float | None
    pct_self: float | None
    baseline: float
    severity: float
    p_value: float | None

    def describe(self, fmt=fmt_pct) -> str:
        bits = [f"{fmt(self.cur)} this year"]
        if self.self_hist:
            bits.append(f"own median {fmt(float(np.median(self.self_hist)))}")
        if self.peers:
            bits.append(f"peer median {fmt(float(np.median(self.peers)))} (n={len(self.peers)})")
        return ", ".join(bits)


def ratio_hist(ctx: RunContext, fn: Ratio, n: int = 5) -> list[float]:
    vals = []
    for p in ctx.periods[:-1][-n:]:
        v = fn(V(ctx.A, p))
        if finite(v):
            vals.append(v)
    return vals


def assess(ctx: RunContext, fn: Ratio, higher_is_bad: bool = True, z_start: float = 1.5, z_full: float = 4.0,
           use_peers: bool = True, min_abs: float = 0.0) -> Assessment:
    cur = fn(V(ctx.A, ctx.focus))
    need(cur)
    hist = ratio_hist(ctx, fn)
    peers = ctx.peer_values(lambda P, p, k: fn(V(P, p))) if use_peers else []
    sign = 1 if higher_is_bad else -1
    zs = sign * robust_z(cur, hist) if len(hist) >= 3 else float("nan")
    zp = sign * robust_z(cur, peers) if len(peers) >= 5 else float("nan")
    s_self, s_peer = severity_from_z(zs, z_start, z_full), severity_from_z(zp, z_start, z_full)
    if np.isfinite(zs) and np.isfinite(zp):
        sev = 0.55 * s_self + 0.45 * s_peer + 0.15 * min(s_self, s_peer)
    elif np.isfinite(zs):
        sev = 0.8 * s_self
    elif np.isfinite(zp):
        sev = 0.6 * s_peer
    else:
        raise SkipDetector("not enough own history or peers for a baseline")
    meds = [float(np.median(x)) for x in (hist, peers) if len(x) >= 3]
    baseline = (max(meds) if higher_is_bad else min(meds)) if meds else float("nan")
    if min_abs and meds and all(abs(cur - m) < min_abs for m in meds):
        sev = 0.0  # statistically unusual but too small to matter (e.g. DSO 4 -> 5 days)
    pz = [z for z in (zs, zp) if np.isfinite(z)]
    return Assessment(
        cur, hist, peers, zs, zp,
        percentile_of(cur, peers) if len(peers) >= 3 else None,
        percentile_of(cur, hist) if len(hist) >= 3 else None,
        baseline, float(np.clip(sev, 0, 1)), z_to_p(max(pz)) if pz else None,
    )


def gap_impact(a: Assessment, driver: float, method: str, basis: str, higher_is_bad: bool = True) -> ImpactEstimate | None:
    if not finite(driver, a.baseline):
        return None
    gap = (a.cur - a.baseline) if higher_is_bad else (a.baseline - a.cur)
    if gap <= 0:
        return None
    mid = gap * driver
    alt = [float(np.median(x)) for x in (a.self_hist, a.peers) if len(x) >= 3]
    far = (a.cur - min(alt)) if higher_is_bad and alt else (max(alt) - a.cur) if alt else gap
    high = max(mid, far * driver)
    return ImpactEstimate(mid * 0.5, mid, high, method, basis)


# ============================================================================ revenue
def _total_deferred(v: V, b: int = 0) -> float:
    c, n = v.g("DEFERRED_REV_CURRENT", b), v.g("DEFERRED_REV_NONCURRENT", b)
    if not finite(c) and not finite(n):
        return float("nan")
    return (c if finite(c) else 0.0) + (n if finite(n) else 0.0)


def _cash_rev_gap(v: V) -> float:
    rev, rev0 = v.g("REVENUE"), v.g("REVENUE", 1)
    dar = v.g("AR") - v.g("AR", 1)
    d1 = _total_deferred(v)
    d0 = _total_deferred(v, 1)
    ddef = (d1 - d0) if finite(d1, d0) else 0.0
    cash = rev - dar + ddef
    dar0 = v.g("AR", 1) - v.g("AR", 2)
    d00 = _total_deferred(v, 2)
    ddef0 = (d0 - d00) if finite(d0, d00) else 0.0
    cash0 = rev0 - dar0 + ddef0
    return (safe_div(rev, rev0) - 1) - (safe_div(cash, cash0) - 1)


@detector("C.REV.01", "Revenue growth vs cash collected", F, Area.REVENUE, needs_peers=True)
def crev01(ctx: RunContext):
    """Cash revenue = revenue - dAR + d deferred revenue; flag when reported growth outruns cash growth."""
    a = assess(ctx, _cash_rev_gap, z_start=1.5)
    v = V(ctx.A, ctx.focus)
    rev = v.g("REVENUE")
    impact = None
    if a.cur > 0 and finite(rev):
        excess = a.cur - max(0.0, a.baseline if finite(a.baseline) else 0.0)
        if excess > 0:
            mid = excess * v.g("REVENUE", 1)
            impact = ImpactEstimate(mid * 0.5, mid, max(mid, a.cur * v.g("REVENUE", 1)), "ratio_reversion",
                                    "excess of revenue growth over cash-revenue growth x prior revenue")
    return [mk("C.REV.01", ctx, raw_value=a.cur, expected=a.baseline, severity=a.severity, unit="growth gap (pts)",
               percentile_peer=a.pct_peer, percentile_self=a.pct_self, p_value=a.p_value, impact=impact,
               scheme=Scheme.PREMATURE_REVENUE, direction="income_increasing",
               rationale=f"Reported revenue growth minus cash-collected revenue growth: {a.describe()}.",
               evidence=ctx.evs(["REVENUE", "AR", "DEFERRED_REV_CURRENT"], ks=(0, 1)))]


@detector("C.REV.02", "Deferred revenue decline", F, Area.REVENUE)
def crev02(ctx: RunContext):
    """Billings (revenue + d deferred revenue) growing slower than revenue: recognition outpacing new business."""
    v = V(ctx.A, ctx.focus)
    d1, d0 = _total_deferred(v), _total_deferred(v, 1)
    rev, rev0 = v.g("REVENUE"), v.g("REVENUE", 1)
    need(d1, d0, rev, rev0, reason="deferred revenue not reported")
    if safe_div(d0, rev0) < 0.02:
        raise SkipDetector("deferred revenue immaterial (< 2% of revenue)")
    d00 = _total_deferred(v, 2)
    bill1 = rev + (d1 - d0)
    bill0 = rev0 + ((d0 - d00) if finite(d00) else 0.0)
    rg, bg = safe_div(rev, rev0) - 1, safe_div(bill1, bill0) - 1
    gap = rg - bg
    ratio_now, ratio_prev = safe_div(d1, rev), safe_div(d0, rev0)
    sev = float(np.clip((gap - 0.02) / 0.10, 0, 1)) if finite(gap) else 0.0
    if ratio_now < ratio_prev * 0.85:
        sev = min(1.0, sev + 0.2)
    impact = ImpactEstimate(0.0, (ratio_prev - ratio_now) * rev * 0.5, (ratio_prev - ratio_now) * rev, "ratio_reversion",
                            "deferred revenue / revenue at prior-year ratio x revenue") if ratio_now < ratio_prev else None
    return [mk("C.REV.02", ctx, raw_value=gap, expected=0.0, severity=sev, unit="growth gap", impact=impact,
               scheme=Scheme.PREMATURE_REVENUE, direction="income_increasing",
               rationale=(f"Revenue grew {fmt_pct(rg)} while billings grew {fmt_pct(bg)}; deferred revenue fell from "
                          f"{fmt_pct(ratio_prev)} to {fmt_pct(ratio_now)} of revenue."),
               evidence=ctx.evs(["DEFERRED_REV_CURRENT", "DEFERRED_REV_NONCURRENT", "REVENUE"], ks=(0, 1)))]


@detector("C.REV.03", "Contract assets / unbilled receivables", F, Area.REVENUE, needs_peers=True)
def crev03(ctx: RunContext):
    """Contract assets relative to revenue vs own history and peers (percentage-of-completion aggressiveness)."""
    fn = lambda v: safe_div(v.g("CONTRACT_ASSETS"), v.g("REVENUE"))  # noqa: E731
    a = assess(ctx, fn)
    rev = ctx.a("REVENUE")
    return [mk("C.REV.03", ctx, raw_value=a.cur, expected=a.baseline, severity=a.severity, unit="share of revenue",
               percentile_peer=a.pct_peer, p_value=a.p_value, scheme=Scheme.PREMATURE_REVENUE, direction="income_increasing",
               impact=gap_impact(a, rev, "ratio_reversion", "(contract assets/revenue - baseline) x revenue"),
               rationale=f"Contract assets / revenue: {a.describe()}.",
               evidence=ctx.evs(["CONTRACT_ASSETS", "REVENUE"], ks=(0, 1)))]


@detector("C.REV.04", "Quarter-end revenue concentration", F, Area.REVENUE)
def crev04(ctx: RunContext):
    """Q4 share of fiscal-year revenue (and Q4 gross margin) vs prior years: channel stuffing / side deals."""
    rev = ctx.Q.series("REVENUE")
    cogs = ctx.Q.series("COGS")
    fyes = [p for p in ctx.A.periods if p <= ctx.focus]
    shares, margins = {}, {}
    for fye in fyes:
        qs = [q for q in rev.index if fye - pd.Timedelta(days=372) < q <= fye + pd.Timedelta(days=8)]
        if len(qs) < 4:
            continue
        qs = qs[-4:]
        r = rev[qs]
        if r.isna().any() or r.sum() <= 0:
            continue
        shares[fye] = float(r.iloc[-1] / r.sum())
        c = cogs.reindex(qs)
        if not c.isna().any():
            margins[fye] = float((r.iloc[-1] - c.iloc[-1]) / r.iloc[-1] - (r.iloc[:3].sum() - c.iloc[:3].sum()) / r.iloc[:3].sum())
    if ctx.focus not in shares:
        raise SkipDetector("four quarters of revenue not available for the focus year")
    cur = shares[ctx.focus]
    hist = [v for k, v in shares.items() if k < ctx.focus][-5:]
    z = robust_z(cur, hist) if len(hist) >= 3 else float("nan")
    sev = severity_from_z(z, 1.5, 4.0) if np.isfinite(z) else 0.0
    mz = robust_z(margins.get(ctx.focus, np.nan), [v for k, v in margins.items() if k < ctx.focus][-5:])
    if np.isfinite(mz) and mz > 2 and sev > 0:
        sev = min(1.0, sev + 0.15)
    fy_rev = ctx.a("REVENUE")
    impact = None
    if hist and cur > float(np.median(hist)):
        mid = (cur - float(np.median(hist))) * fy_rev
        impact = ImpactEstimate(mid * 0.3, mid * 0.5, mid, "ratio_reversion", "excess Q4 share x fiscal-year revenue")
    q4 = max(q for q in rev.index if q <= ctx.focus + pd.Timedelta(days=8))
    ev = [e for e in (ctx.ev("REVENUE", panel="Q", period=q4), ctx.ev("REVENUE")) if e]
    return [mk("C.REV.04", ctx, raw_value=cur, expected=float(np.median(hist)) if hist else None, severity=sev,
               unit="Q4 share", percentile_self=percentile_of(cur, hist) if len(hist) >= 3 else None,
               scheme=Scheme.PREMATURE_REVENUE, direction="income_increasing", impact=impact,
               rationale=(f"Q4 delivered {fmt_pct(cur)} of fiscal-year revenue vs a prior median of "
                          f"{fmt_pct(float(np.median(hist)) if hist else np.nan)}"
                          + (f"; Q4 gross margin exceeded Q1-Q3 by {fmt_pct(margins[ctx.focus])}" if ctx.focus in margins else "") + "."),
               evidence=ev)]


@detector("C.REV.05", "Revenue vs asset base", F, Area.REVENUE, needs_peers=True)
def crev05(ctx: RunContext):
    """Revenue per dollar of PP&E and of total assets jumps vs history and peers (capacity mismatch)."""
    fn = lambda v: safe_div(v.g("REVENUE"), v.avg("TOTAL_ASSETS"))  # noqa: E731
    a = assess(ctx, fn, z_start=2.0, z_full=5.0)
    return [mk("C.REV.05", ctx, raw_value=a.cur, expected=a.baseline, severity=a.severity * 0.8, unit="x",
               percentile_peer=a.pct_peer, scheme=Scheme.BOGUS_REVENUE, direction="income_increasing",
               rationale=f"Asset turnover (revenue / average assets): {a.describe(lambda x: fmt_num(x) + 'x')}.",
               evidence=ctx.evs(["REVENUE", "TOTAL_ASSETS"], ks=(0, 1)))]


@detector("C.REV.06", "Segment growth consistency", F, Area.REVENUE)
def crev06(ctx: RunContext):
    """Consolidated vs sum-of-segment growth. Needs dimensional XBRL."""
    raise SkipDetector("segment facts are dimensional; not available from companyfacts")


@detector("C.REV.07", "Gross-vs-net presentation change", F, Area.REVENUE, needs_text=True)
def crev07(ctx: RunContext):
    """New principal-vs-agent language plus a revenue jump with gross-margin compression."""
    if ctx.doc is None:
        raise SkipDetector("10-K text not available")
    hits = new_mentions(ctx.doc, ctx.prior_doc, r"principal versus agent|principal vs\.? agent|gross basis|on a net basis|acting as (?:a|an) (?:principal|agent)")
    v = V(ctx.A, ctx.focus)
    rg = safe_div(v.g("REVENUE"), v.g("REVENUE", 1)) - 1
    dgm = v.gross_margin(0) - v.gross_margin(1)
    sev = 0.0
    if hits and finite(rg, dgm) and rg > 0.15 and dgm < -0.02:
        sev = 0.7
    elif hits:
        sev = 0.25
    return [mk("C.REV.07", ctx, raw_value=rg, severity=sev, unit="revenue growth", scheme=Scheme.MISLEADING_METRICS,
               direction="income_increasing",
               rationale=(f"{len(hits)} new principal/agent passage(s); revenue growth {fmt_pct(rg)}, gross-margin change {fmt_pct(dgm)}."
                          if hits else "No new principal-versus-agent language this year."),
               evidence=[ctx.ev_span(h.doc, h.start, h.end, "New principal/agent language") for h in hits[:2]])]


@detector("C.REV.08", "Related-party revenue", F, Area.RELATED_PARTY)
def crev08(ctx: RunContext):
    """Revenue from related parties as a share of revenue and its growth (round-tripping risk)."""
    rp, rev = ctx.a("RELATED_PARTY_REVENUE"), ctx.a("REVENUE")
    need(rp, rev, reason="related-party revenue not tagged")
    share = safe_div(rp, rev)
    prev = safe_div(ctx.a("RELATED_PARTY_REVENUE", 1), ctx.a("REVENUE", 1))
    sev = float(np.clip((share - 0.02) / 0.10, 0, 1))
    if finite(prev) and share > prev * 1.5 and share > 0.02:
        sev = min(1.0, sev + 0.2)
    return [mk("C.REV.08", ctx, raw_value=share, expected=prev, severity=sev, unit="share of revenue",
               scheme=Scheme.BOGUS_REVENUE, direction="income_increasing",
               rationale=f"Related-party revenue {fmt_money(rp)} = {fmt_pct(share)} of revenue (prior year {fmt_pct(prev)}).",
               evidence=ctx.evs(["RELATED_PARTY_REVENUE", "REVENUE"], ks=(0, 1)))]


@detector("C.REV.09", "Revenue vs operating KPIs", F, Area.REVENUE)
def crev09(ctx: RunContext):
    """Revenue elasticity to users/units/ARR. KPIs are company-specific extensions."""
    raise SkipDetector("operating KPIs are company-specific; requires LLM extraction (planned)")


@detector("C.REV.10", "Revenue vs remaining performance obligations", F, Area.REVENUE)
def crev10(ctx: RunContext):
    """Revenue growth exceeding RPO (backlog) growth for consecutive years: borrowing from the future."""
    rpo = ctx.hist("RPO", 4).dropna()
    rev = ctx.hist("REVENUE", 4).dropna()
    if len(rpo) < 3:
        raise SkipDetector("RPO not reported for 3 years")
    rg = rev.pct_change().reindex(rpo.index)
    pg = rpo.pct_change()
    gaps = (rg - pg).dropna()
    streak = 0
    for g in reversed(gaps.tolist()):
        if g > 0.03:
            streak += 1
        else:
            break
    sev = float(np.clip(streak / 3, 0, 1)) * (0.9 if safe_div(rpo.iloc[-1], rev.iloc[-1]) > 0.3 else 0.5)
    return [mk("C.REV.10", ctx, raw_value=float(gaps.iloc[-1]) if len(gaps) else None, severity=sev, unit="growth gap",
               scheme=Scheme.PREMATURE_REVENUE, direction="income_increasing",
               rationale=f"Revenue growth exceeded RPO growth in {streak} consecutive year(s); RPO = {fmt_pct(safe_div(rpo.iloc[-1], rev.iloc[-1]))} of revenue.",
               evidence=ctx.evs(["RPO", "REVENUE"], ks=(0, 1)))]


@detector("C.REV.11", "Aggressive revenue-arrangement language", F, Area.REVENUE, needs_text=True)
def crev11(ctx: RunContext):
    """New bill-and-hold, consignment, extended-terms or side-letter language, weighted by a DSO increase."""
    if ctx.doc is None:
        raise SkipDetector("10-K text not available")
    pats = {
        "bill-and-hold": r"bill[- ]and[- ]hold",
        "consignment": r"\bconsignment\b",
        "extended payment terms": r"extended payment terms|extended credit terms",
        "side agreements": r"side (?:letter|agreement)s?",
        "channel inventory / sell-in": r"channel inventor(?:y|ies)|sell[- ]in",
    }
    found = []
    for label, pat in pats.items():
        for h in new_mentions(ctx.doc, ctx.prior_doc, pat)[:1]:
            found.append((label, h))
    dso_up = safe_div(ctx.a("AR"), ctx.a("REVENUE")) > 1.1 * safe_div(ctx.a("AR", 1), ctx.a("REVENUE", 1))
    sev = min(1.0, 0.3 * len(found) + (0.25 if found and dso_up else 0.0))
    return [mk("C.REV.11", ctx, raw_value=len(found), severity=sev, unit="new topics", scheme=Scheme.PREMATURE_REVENUE,
               direction="income_increasing",
               rationale=("New this year: " + ", ".join(lbl for lbl, _ in found) + ("; DSO also rose >10%." if dso_up else ".")) if found
               else "No new bill-and-hold, consignment, extended-terms or side-agreement language.",
               evidence=[ctx.ev_span(h.doc, h.start, h.end, f"New language: {lbl}") for lbl, h in found[:3]])]


# ============================================================================ receivables
def _dso(v: V, b: int = 0) -> float:
    return safe_div(v.g("AR", b), v.g("REVENUE", b)) * 365


@detector("C.AR.01", "DSO jump", F, Area.RECEIVABLES, needs_peers=True)
def car01(ctx: RunContext):
    """Days sales outstanding vs own history and peers; impact = excess DSO x revenue / 365."""
    a = assess(ctx, _dso, min_abs=5.0)
    rev = ctx.a("REVENUE")
    imp = gap_impact(a, rev / 365, "ratio_reversion", "(DSO - baseline DSO) x revenue / 365")
    # quarterly, same-quarter-last-year comparison for seasonality
    qnote = ""
    qa, qr = ctx.qhist("AR", 8), ctx.qhist("REVENUE", 8)
    if len(qa.dropna()) >= 5 and len(qr.dropna()) >= 5:
        qd = (qa / qr * 91).dropna()
        if len(qd) >= 5:
            yoy = qd.iloc[-1] - qd.iloc[-5]
            qnote = f" Latest quarter DSO {qd.iloc[-1]:.0f} days vs {qd.iloc[-5]:.0f} a year earlier ({yoy:+.0f})."
    return [mk("C.AR.01", ctx, raw_value=a.cur, expected=a.baseline, severity=a.severity, unit="days",
               percentile_peer=a.pct_peer, percentile_self=a.pct_self, p_value=a.p_value, impact=imp,
               scheme=Scheme.PREMATURE_REVENUE, direction="income_increasing",
               rationale=f"DSO: {a.describe(lambda x: f'{x:.0f} days')}.{qnote}",
               evidence=ctx.evs(["AR", "REVENUE"], ks=(0, 1)))]


@detector("C.AR.02", "Allowance coverage decline", F, Area.RECEIVABLES)
def car02(ctx: RunContext):
    """Allowance / gross AR falling while DSO rises; impact = coverage gap x gross AR."""
    al, ar = ctx.a("ALLOWANCE"), ctx.a("AR")
    need(al, ar, reason="allowance not tagged")
    cov = safe_div(al, ar + al)
    hist = [safe_div(ctx.a("ALLOWANCE", k), ctx.a("AR", k) + ctx.a("ALLOWANCE", k)) for k in range(1, 5)]
    hist = [h for h in hist if finite(h)]
    if not hist:
        raise SkipDetector("no allowance history")
    base = float(np.median(hist))
    if base < 0.002 and cov < 0.002:
        return [mk("C.AR.02", ctx, raw_value=cov, expected=base, severity=0.0, unit="coverage",
                   rationale=f"Allowance is immaterial ({fmt_pct(cov, 2)} of gross receivables, {fmt_pct(base, 2)} historically).",
                   evidence=ctx.evs(["ALLOWANCE", "AR"], ks=(0,)))]
    drop = safe_div(base - cov, base)
    dso_up = _dso(V(ctx.A, ctx.focus)) > _dso(V(ctx.A, ctx.focus), 1)
    sev = float(np.clip((drop - 0.10) / 0.40, 0, 1)) * (1.0 if dso_up else 0.6)
    imp = ImpactEstimate((base - cov) * (ar + al) * 0.5, (base - cov) * (ar + al), (max(hist) - cov) * (ar + al),
                         "ratio_reversion", "(baseline coverage - coverage) x gross AR") if cov < base else None
    return [mk("C.AR.02", ctx, raw_value=cov, expected=base, severity=sev, unit="coverage", impact=imp,
               scheme=Scheme.COOKIE_JAR, direction="income_increasing",
               rationale=f"Allowance covers {fmt_pct(cov)} of gross receivables vs {fmt_pct(base)} historically; DSO {'rose' if dso_up else 'did not rise'}.",
               evidence=ctx.evs(["ALLOWANCE", "AR"], ks=(0, 1)))]


@detector("C.AR.03", "Provision below write-offs", F, Area.RECEIVABLES)
def car03(ctx: RunContext):
    """Provision for credit losses below write-offs for 2+ years: the reserve is being drawn down into income."""
    rows = [(ctx.a("PROVISION_DOUBTFUL", k), ctx.a("WRITE_OFFS", k)) for k in range(3)]
    rows = [(p, w) for p, w in rows if finite(p, w)]
    if len(rows) < 2:
        raise SkipDetector("provision and write-offs not tagged for 2 years")
    short = [w - p for p, w in rows if p < w]
    streak = 0
    for p, w in rows:
        if p < w:
            streak += 1
        else:
            break
    sev = float(np.clip((streak - 1) / 2, 0, 1)) * 0.8
    return [mk("C.AR.03", ctx, raw_value=streak, severity=sev, unit="years", scheme=Scheme.COOKIE_JAR,
               direction="income_increasing",
               impact=ImpactEstimate(0, sum(short[:streak]) * 0.5, sum(short[:streak]), "disclosed_amount",
                                     "write-offs in excess of provisions") if streak >= 2 else None,
               rationale=f"Provision was below write-offs in {streak} consecutive year(s).",
               evidence=ctx.evs(["PROVISION_DOUBTFUL", "WRITE_OFFS"], ks=(0, 1)))]


@detector("C.AR.04", "Receivables factoring / sales", F, Area.RECEIVABLES, needs_text=True)
def car04(ctx: RunContext):
    """New factoring or receivable-sale language (boosts CFO, lowers DSO)."""
    if ctx.doc is None:
        raise SkipDetector("10-K text not available")
    hits = new_mentions(ctx.doc, ctx.prior_doc, r"factoring|sales? of (?:trade |accounts )?receivables|receivables? (?:purchase|securitization) (?:facility|program|agreement)")
    existing = search(ctx.doc, r"factoring|sales? of (?:trade |accounts )?receivables|securitization facility")
    sev = 0.55 if hits else (0.2 if existing else 0.0)
    shown = hits or existing
    return [mk("C.AR.04", ctx, raw_value=len(hits), severity=sev, unit="new mentions", scheme=Scheme.CFO_INFLATION,
               area=Area.CASH_FLOW, direction="neutral",
               rationale=("New factoring / receivable-sale language this year; DSO and CFO should be read net of sold receivables." if hits
                          else "Factoring / receivable sales mentioned (not new)." if existing else "No factoring or receivable-sale language."),
               evidence=[ctx.ev_span(h.doc, h.start, h.end, "Receivable sale / factoring") for h in shown[:2]])]


@detector("C.AR.05", "Long-term and other receivables growth", F, Area.RECEIVABLES)
def car05(ctx: RunContext):
    """Non-current receivables growing faster than revenue (extended terms, financing customers)."""
    lt, lt0 = ctx.a("LT_RECEIVABLES"), ctx.a("LT_RECEIVABLES", 1)
    need(lt, lt0, reason="long-term receivables not tagged")
    rev, rev0 = ctx.a("REVENUE"), ctx.a("REVENUE", 1)
    g, rg = safe_div(lt, lt0) - 1, safe_div(rev, rev0) - 1
    share = safe_div(lt, rev)
    sev = float(np.clip((g - rg - 0.2) / 0.6, 0, 1)) * float(np.clip(share / 0.05, 0, 1))
    return [mk("C.AR.05", ctx, raw_value=g - rg, severity=sev, unit="growth gap", scheme=Scheme.PREMATURE_REVENUE,
               direction="income_increasing",
               rationale=f"Long-term receivables grew {fmt_pct(g)} vs revenue {fmt_pct(rg)}; they equal {fmt_pct(share)} of revenue.",
               evidence=ctx.evs(["LT_RECEIVABLES", "REVENUE"], ks=(0, 1)))]


# ============================================================================ inventory
def _dsi(v: V, b: int = 0) -> float:
    return safe_div(v.g("INVENTORY", b), v.g("COGS", b)) * 365


def _need_inventory(ctx: RunContext) -> None:
    inv, rev = ctx.a("INVENTORY"), ctx.a("REVENUE")
    need(inv, reason="no inventory reported")
    if safe_div(inv, rev) < 0.01:
        raise SkipDetector("inventory immaterial")


@detector("C.INV.01", "DSI jump", F, Area.INVENTORY, needs_peers=True)
def cinv01(ctx: RunContext):
    """Days inventory outstanding vs history and peers; finished-goods mix shift noted."""
    _need_inventory(ctx)
    a = assess(ctx, _dsi, min_abs=5.0)
    cogs = ctx.a("COGS")
    mix = ""
    fg = ctx.a("INV_FINISHED")
    fg0, inv0 = ctx.a("INV_FINISHED", 1), ctx.a("INVENTORY", 1)
    if finite(fg, fg0, inv0):
        mix = f" Finished goods are {fmt_pct(safe_div(fg, ctx.a('INVENTORY')))} of inventory (prior {fmt_pct(safe_div(fg0, inv0))})."
    return [mk("C.INV.01", ctx, raw_value=a.cur, expected=a.baseline, severity=a.severity, unit="days",
               percentile_peer=a.pct_peer, percentile_self=a.pct_self, p_value=a.p_value,
               impact=gap_impact(a, cogs / 365, "ratio_reversion", "(DSI - baseline) x COGS / 365"),
               scheme=Scheme.SHIFT_EXPENSES_LATER, direction="income_increasing",
               rationale=f"DSI: {a.describe(lambda x: f'{x:.0f} days')}.{mix}",
               evidence=ctx.evs(["INVENTORY", "COGS"], ks=(0, 1)))]


@detector("C.INV.02", "Inventory growth vs COGS growth", F, Area.INVENTORY)
def cinv02(ctx: RunContext):
    """Inventory growing faster than cost of sales for two consecutive years."""
    _need_inventory(ctx)
    spreads = []
    for k in (0, 1):
        ig = safe_div(ctx.a("INVENTORY", k), ctx.a("INVENTORY", k + 1)) - 1
        cg = safe_div(ctx.a("COGS", k), ctx.a("COGS", k + 1)) - 1
        spreads.append(ig - cg)
    need(spreads[0])
    s0, s1 = spreads
    sev = float(np.clip((s0 - 0.05) / 0.25, 0, 1))
    if finite(s1) and s1 > 0.05 and s0 > 0.05:
        sev = min(1.0, sev + 0.2)
    inv0 = ctx.a("INVENTORY", 1)
    imp = (ImpactEstimate(0.0, 0.25 * s0 * inv0, 0.5 * s0 * inv0, "ratio_reversion",
                          "part of excess inventory growth (inventory growth - COGS growth) x prior inventory that may need reserving")
           if s0 > 0.05 and finite(inv0) else None)
    return [mk("C.INV.02", ctx, raw_value=s0, expected=0.0, severity=sev, unit="growth spread", impact=imp,
               scheme=Scheme.SHIFT_EXPENSES_LATER, direction="income_increasing",
               rationale=f"Inventory growth minus COGS growth: {fmt_pct(s0)} this year, {fmt_pct(s1)} last year.",
               evidence=ctx.evs(["INVENTORY", "COGS"], ks=(0, 1, 2)))]  # three years: last year's spread needs the year before


@detector("C.INV.03", "Margin up while inventory builds", F, Area.INVENTORY)
def cinv03(ctx: RunContext):
    """Gross margin rising while DSI rises: overhead absorption or under-reserved obsolescence."""
    _need_inventory(ctx)
    v = V(ctx.A, ctx.focus)
    dgm = v.gross_margin(0) - v.gross_margin(1)
    ddsi = _dsi(v) - _dsi(v, 1)
    need(dgm, ddsi)
    sev = float(np.clip(dgm / 0.03, 0, 1) * np.clip(ddsi / 20, 0, 1)) if dgm > 0 and ddsi > 0 else 0.0
    return [mk("C.INV.03", ctx, raw_value=dgm, severity=sev, unit="margin change", scheme=Scheme.SHIFT_EXPENSES_LATER,
               area=Area.MARGINS, direction="income_increasing",
               rationale=f"Gross margin changed {fmt_pct(dgm)} while DSI changed {ddsi:+.0f} days.",
               evidence=ctx.evs(["REVENUE", "COGS", "INVENTORY"], ks=(0, 1)))]


@detector("C.INV.04", "Inventory reserve ratio decline", F, Area.INVENTORY)
def cinv04(ctx: RunContext):
    """Obsolescence reserve / inventory falling while DSI rises."""
    r, inv = ctx.a("INV_RESERVE"), ctx.a("INVENTORY")
    r0, inv0 = ctx.a("INV_RESERVE", 1), ctx.a("INVENTORY", 1)
    need(r, inv, r0, inv0, reason="inventory reserve not tagged")
    c, c0 = safe_div(r, inv + r), safe_div(r0, inv0 + r0)
    v = V(ctx.A, ctx.focus)
    dsi_up = _dsi(v) > _dsi(v, 1)
    drop = safe_div(c0 - c, c0)
    sev = float(np.clip((drop - 0.1) / 0.4, 0, 1)) * (1.0 if dsi_up else 0.5)
    return [mk("C.INV.04", ctx, raw_value=c, expected=c0, severity=sev, unit="coverage", scheme=Scheme.COOKIE_JAR,
               direction="income_increasing",
               impact=ImpactEstimate(0, (c0 - c) * (inv + r) * 0.5, (c0 - c) * (inv + r), "ratio_reversion",
                                     "(prior coverage - coverage) x gross inventory") if c < c0 else None,
               rationale=f"Inventory reserve coverage {fmt_pct(c)} vs {fmt_pct(c0)} last year; DSI {'rose' if dsi_up else 'fell'}.",
               evidence=ctx.evs(["INV_RESERVE", "INVENTORY"], ks=(0, 1)))]


@detector("C.INV.05", "LIFO liquidation", F, Area.INVENTORY)
def cinv05(ctx: RunContext):
    """Decline in the LIFO reserve releases old low costs into COGS (one-time margin boost)."""
    l1, l0 = ctx.a("LIFO_RESERVE"), ctx.a("LIFO_RESERVE", 1)
    need(l1, l0, reason="no LIFO reserve")
    rel = l0 - l1
    pt = abs(ctx.a("PRETAX_INCOME"))
    sev = float(np.clip(safe_div(rel, pt) / 0.05, 0, 1)) if rel > 0 else 0.0
    return [mk("C.INV.05", ctx, raw_value=rel, severity=sev, unit="USD", scheme=Scheme.ONE_TIME_GAINS,
               direction="income_increasing",
               impact=ImpactEstimate(0, rel * 0.5, rel, "disclosed_amount", "decline in LIFO reserve") if rel > 0 else None,
               rationale=f"LIFO reserve changed by {fmt_money(-rel)} ({fmt_pct(safe_div(rel, pt))} of pre-tax income).",
               evidence=ctx.evs(["LIFO_RESERVE"], ks=(0, 1)))]


# ============================================================================ expenses & capitalization
@detector("C.EXP.01", "Capitalization rate", F, Area.CAPITALIZATION, needs_peers=True)
def cexp01(ctx: RunContext):
    """Capitalized software additions / (opex + capitalized) vs history and peers."""
    def fn(v: V) -> float:
        cap = v.g("CAP_SOFTWARE_ADDITIONS")
        opex = v.z("SGA") + v.z("RND")
        return safe_div(cap, opex + cap) if finite(cap) and opex > 0 else float("nan")

    a = assess(ctx, fn)
    base = ctx.a("CAP_SOFTWARE_ADDITIONS") + ctx.a("SGA") + (ctx.a("RND") if finite(ctx.a("RND")) else 0)
    return [mk("C.EXP.01", ctx, raw_value=a.cur, expected=a.baseline, severity=a.severity, unit="share of spend",
               percentile_peer=a.pct_peer, p_value=a.p_value,
               impact=gap_impact(a, base, "ratio_reversion", "(capitalization rate - baseline) x (opex + capitalized)"),
               scheme=Scheme.SHIFT_EXPENSES_LATER, direction="income_increasing",
               rationale=f"Capitalized software as a share of total spend: {a.describe()}.",
               evidence=ctx.evs(["CAP_SOFTWARE_ADDITIONS", "SGA", "RND"], ks=(0, 1)))]


@detector("C.EXP.02", "Soft-asset growth", F, Area.CAPITALIZATION, needs_peers=True)
def cexp02(ctx: RunContext):
    """Other current + other non-current assets relative to revenue (deferred costs, prepaid)."""
    fn = lambda v: safe_div(v.z("OTHER_CURRENT_ASSETS") + v.z("OTHER_NONCURRENT_ASSETS"), v.g("REVENUE"))  # noqa: E731
    a = assess(ctx, fn)
    return [mk("C.EXP.02", ctx, raw_value=a.cur, expected=a.baseline, severity=a.severity * 0.8, unit="share of revenue",
               percentile_peer=a.pct_peer, p_value=a.p_value,
               impact=gap_impact(a, ctx.a("REVENUE"), "ratio_reversion", "(other assets/revenue - baseline) x revenue"),
               scheme=Scheme.SHIFT_EXPENSES_LATER, direction="income_increasing",
               rationale=f"Other assets / revenue: {a.describe()}.",
               evidence=ctx.evs(["OTHER_CURRENT_ASSETS", "OTHER_NONCURRENT_ASSETS", "REVENUE"], ks=(0, 1)))]


def _dep(v: V, b: int = 0) -> float:
    d = v.g("DEPRECIATION", b)
    return d if finite(d) else v.g("DA", b) - v.z("AMORTIZATION_INTANGIBLES", b)


@detector("C.EXP.03", "Useful-life extension", F, Area.DEPRECIATION)
def cexp03(ctx: RunContext):
    """Implied average life = gross PP&E / depreciation rising > 10%; impact = depreciation at prior life - reported."""
    v = V(ctx.A, ctx.focus)
    life = safe_div(v.g("PPE_GROSS"), _dep(v))
    lives = [safe_div(v.g("PPE_GROSS", k), _dep(v, k)) for k in range(1, 4)]
    lives = [x for x in lives if finite(x) and x > 0]
    need(life, reason="gross PP&E or depreciation not reported")
    if not lives:
        raise SkipDetector("no prior implied life")
    prior = float(np.median(lives))
    chg = safe_div(life - prior, prior)
    sev = float(np.clip((chg - 0.10) / 0.30, 0, 1))
    dep_at_old = safe_div(v.g("PPE_GROSS"), prior)
    imp = dep_at_old - _dep(v)
    return [mk("C.EXP.03", ctx, raw_value=life, expected=prior, severity=sev, unit="years",
               scheme=Scheme.SHIFT_EXPENSES_LATER, direction="income_increasing",
               impact=ImpactEstimate(imp * 0.5, imp, imp * 1.2, "policy_counterfactual",
                                     "gross PP&E / prior implied life - reported depreciation") if imp > 0 and chg > 0.05 else None,
               rationale=f"Implied average useful life {life:.1f} years vs {prior:.1f} previously ({fmt_pct(chg)}). Mix shifts toward long-lived assets are a legitimate cause.",
               evidence=ctx.evs(["PPE_GROSS", "DEPRECIATION", "DA"], ks=(0, 1)))]


@detector("C.EXP.04", "Depreciation rate drop", F, Area.DEPRECIATION, needs_peers=True)
def cexp04(ctx: RunContext):
    """Depreciation / average gross PP&E vs history and peers."""
    fn = lambda v: safe_div(_dep(v), v.avg("PPE_GROSS"))  # noqa: E731
    a = assess(ctx, fn, higher_is_bad=False)
    return [mk("C.EXP.04", ctx, raw_value=a.cur, expected=a.baseline, severity=a.severity, unit="rate",
               percentile_peer=a.pct_peer, p_value=a.p_value,
               impact=gap_impact(a, ctx.avg("PPE_GROSS"), "ratio_reversion", "(baseline rate - rate) x avg gross PP&E", higher_is_bad=False),
               scheme=Scheme.SHIFT_EXPENSES_LATER, direction="income_increasing",
               rationale=f"Depreciation rate: {a.describe()}.",
               evidence=ctx.evs(["DEPRECIATION", "DA", "PPE_GROSS"], ks=(0, 1)))]


@detector("C.EXP.05", "Capex far above depreciation", F, Area.CAPITALIZATION)
def cexp05(ctx: RunContext):
    """Capex/D&A persistently > 2 while revenue is flat (possible capitalization of operating costs)."""
    ratios = [safe_div(ctx.a("CAPEX", k), ctx.a("DA", k)) for k in range(3)]
    rg = safe_div(ctx.a("REVENUE"), ctx.a("REVENUE", 2)) ** 0.5 - 1
    ratios = [r for r in ratios if finite(r)]
    need(*(ratios[:1] or [float("nan")]), rg)
    persistent = sum(1 for r in ratios if r > 2)
    sev = float(np.clip((persistent - 1) / 2, 0, 1)) * (1.0 if rg < 0.05 else 0.3)
    return [mk("C.EXP.05", ctx, raw_value=ratios[0], expected=1.5, severity=sev * 0.8, unit="x",
               scheme=Scheme.SHIFT_EXPENSES_LATER, direction="income_increasing",
               rationale=f"Capex / D&A {fmt_num(ratios[0])}x (> 2x in {persistent} of {len(ratios)} years); revenue CAGR {fmt_pct(rg)}.",
               evidence=ctx.evs(["CAPEX", "DA", "REVENUE"], ks=(0,)))]


@detector("C.EXP.06", "Discretionary expense cut", F, Area.MARGINS)
def cexp06(ctx: RunContext):
    """(SG&A + R&D) / revenue drops sharply vs own trend (real earnings management)."""
    fn = lambda v: safe_div(v.z("SGA") + v.z("RND"), v.g("REVENUE"))  # noqa: E731
    a = assess(ctx, fn, higher_is_bad=False, use_peers=False, z_start=2.0)
    return [mk("C.EXP.06", ctx, raw_value=a.cur, expected=a.baseline, severity=a.severity * 0.8, unit="share of revenue",
               percentile_self=a.pct_self,
               impact=gap_impact(a, ctx.a("REVENUE"), "ratio_reversion", "(baseline - ratio) x revenue", higher_is_bad=False),
               scheme=Scheme.SHIFT_EXPENSES_LATER, direction="income_increasing",
               rationale=f"SG&A + R&D as a share of revenue: {a.describe()}.",
               evidence=ctx.evs(["SGA", "RND", "REVENUE"], ks=(0, 1)))]


@detector("C.EXP.07", "Recurring 'non-recurring' charges", F, Area.MARGINS)
def cexp07(ctx: RunContext):
    """Restructuring/impairment charges in 3+ of the last 5 years."""
    yrs = 0
    total = 0.0
    for k in range(5):
        c = sum(ctx.a(i, k) for i in ("RESTRUCTURING", "ASSET_IMPAIRMENT") if finite(ctx.a(i, k)) and ctx.a(i, k) > 0)
        if c > 0:
            yrs += 1
            total += c
    pt = abs(ctx.a("PRETAX_INCOME"))
    sev = 0.6 * float(np.clip((yrs - 2) / 2, 0, 1)) * float(np.clip(safe_div(total / 5, pt) / 0.05, 0.3, 1))
    return [mk("C.EXP.07", ctx, raw_value=yrs, severity=sev, unit="years of 5", scheme=Scheme.MISLEADING_METRICS,
               rationale=f"Restructuring or impairment charges in {yrs} of the last 5 years, totalling {fmt_money(total)}.",
               evidence=ctx.evs(["RESTRUCTURING", "ASSET_IMPAIRMENT"], ks=(0, 1)))]


@detector("C.EXP.08", "Pension assumptions", F, Area.RESERVES, needs_peers=True)
def cexp08(ctx: RunContext):
    """Discount rate and expected return vs peer medians (higher = lower pension cost)."""
    dr, er = ctx.a("PENSION_DISCOUNT_RATE"), ctx.a("PENSION_EXPECTED_RETURN")
    if not finite(dr) and not finite(er):
        raise SkipDetector("no defined-benefit plan assumptions tagged")
    pdr = ctx.peer_values(lambda P, p, k: P.get("PENSION_DISCOUNT_RATE", p))
    per = ctx.peer_values(lambda P, p, k: P.get("PENSION_EXPECTED_RETURN", p))
    gaps = []
    if finite(dr) and len(pdr) >= 3:
        gaps.append(("discount rate", dr - float(np.median(pdr))))
    if finite(er) and len(per) >= 3:
        gaps.append(("expected return", er - float(np.median(per))))
    if not gaps:
        raise SkipDetector("too few peers with pension assumptions")
    worst = max(g for _, g in gaps)
    sev = float(np.clip((worst - 0.0025) / 0.01, 0, 1)) * 0.7
    return [mk("C.EXP.08", ctx, raw_value=worst, expected=0.0, severity=sev, unit="rate gap",
               scheme=Scheme.SHIFT_EXPENSES_LATER, direction="income_increasing",
               rationale="; ".join(f"{n} {fmt_pct(g, 2)} vs peer median" for n, g in gaps) + ".",
               evidence=ctx.evs(["PENSION_DISCOUNT_RATE", "PENSION_EXPECTED_RETURN"], ks=(0,)))]


@detector("C.EXP.09", "Share-based compensation intensity", F, Area.MARGINS, needs_peers=True)
def cexp09(ctx: RunContext):
    """SBC / revenue vs peers (relevant when SBC is excluded from non-GAAP metrics)."""
    fn = lambda v: safe_div(v.g("SBC"), v.g("REVENUE"))  # noqa: E731
    a = assess(ctx, fn, z_start=2.0)
    return [mk("C.EXP.09", ctx, raw_value=a.cur, expected=a.baseline, severity=a.severity * 0.5, unit="share of revenue",
               percentile_peer=a.pct_peer, scheme=Scheme.MISLEADING_METRICS,
               rationale=f"SBC / revenue: {a.describe()}.", evidence=ctx.evs(["SBC", "REVENUE"], ks=(0,)))]


# ============================================================================ reserves
RESERVES = (("ALLOWANCE", "AR", "allowance for credit losses"), ("INV_RESERVE", "INVENTORY", "inventory reserve"),
            ("WARRANTY_ACCRUAL", "REVENUE", "warranty reserve"))


@detector("C.RES.01", "Reserve release into earnings", F, Area.RESERVES)
def cres01(ctx: RunContext):
    """A falling reserve ratio that contributes > 5% of pre-tax income (or of its growth)."""
    pt, pt0 = ctx.a("PRETAX_INCOME"), ctx.a("PRETAX_INCOME", 1)
    need(pt)
    found, evid = [], []
    for res, base, label in RESERVES:
        r1, b1, r0, b0 = ctx.a(res), ctx.a(base), ctx.a(res, 1), ctx.a(base, 1)
        if not finite(r1, b1, r0, b0) or b0 == 0:
            continue
        rel = (safe_div(r0, b0) - safe_div(r1, b1)) * b1
        if rel > 0:
            found.append((label, rel))
            evid += ctx.evs([res, base], ks=(0, 1))
    if not found:
        if not any(finite(ctx.a(r)) for r, _, _ in RESERVES):
            raise SkipDetector("no reserve accounts tagged")
        return [mk("C.RES.01", ctx, raw_value=0.0, severity=0.0, rationale="No reserve ratio declined.")]
    total = sum(r for _, r in found)
    growth = pt - pt0 if finite(pt0) else float("nan")
    share = max(safe_div(total, abs(pt)), safe_div(total, abs(growth)) if finite(growth) and growth > 0 else 0)
    sev = float(np.clip((share - 0.05) / 0.20, 0, 1))
    return [mk("C.RES.01", ctx, raw_value=total, severity=sev, unit="USD", scheme=Scheme.COOKIE_JAR, direction="income_increasing",
               impact=ImpactEstimate(total * 0.5, total, total, "ratio_reversion", "reserve ratio decline x base"),
               rationale="Reserve ratio declines released " + ", ".join(f"{fmt_money(r)} ({lbl})" for lbl, r in found)
               + f"; equal to {fmt_pct(safe_div(total, abs(pt)))} of pre-tax income.",
               evidence=evid)]


@detector("C.RES.02", "Warranty accrual vs claims", F, Area.RESERVES)
def cres02(ctx: RunContext):
    """Warranty accrual rate falling while claims paid are steady or rising."""
    p1, p0 = ctx.a("WARRANTY_PROVISION"), ctx.a("WARRANTY_PROVISION", 1)
    c1, c0 = ctx.a("WARRANTY_PAYMENTS"), ctx.a("WARRANTY_PAYMENTS", 1)
    rev, rev0 = ctx.a("REVENUE"), ctx.a("REVENUE", 1)
    need(p1, p0, c1, c0, reason="warranty roll-forward not tagged")
    a1, a0 = safe_div(p1, rev), safe_div(p0, rev0)
    k1, k0 = safe_div(c1, rev), safe_div(c0, rev0)
    drop = safe_div(a0 - a1, a0)
    sev = float(np.clip((drop - 0.10) / 0.4, 0, 1)) if k1 >= k0 * 0.95 else 0.0
    return [mk("C.RES.02", ctx, raw_value=a1, expected=a0, severity=sev, unit="accrual rate", scheme=Scheme.COOKIE_JAR,
               direction="income_increasing",
               impact=ImpactEstimate(0, (a0 - a1) * rev * 0.5, (a0 - a1) * rev, "ratio_reversion", "(prior accrual rate - rate) x revenue") if a1 < a0 else None,
               rationale=f"Warranty accrual {fmt_pct(a1, 2)} of revenue (prior {fmt_pct(a0, 2)}); claims {fmt_pct(k1, 2)} (prior {fmt_pct(k0, 2)}).",
               evidence=ctx.evs(["WARRANTY_PROVISION", "WARRANTY_PAYMENTS"], ks=(0, 1)))]


@detector("C.RES.03", "Valuation-allowance release", F, Area.RESERVES)
def cres03(ctx: RunContext):
    """Decrease in the deferred-tax valuation allowance relative to pre-tax income."""
    v1, v0, pt = ctx.a("VALUATION_ALLOWANCE"), ctx.a("VALUATION_ALLOWANCE", 1), ctx.a("PRETAX_INCOME")
    need(v1, v0, pt, reason="valuation allowance not tagged")
    rel = v0 - v1
    share = safe_div(rel, abs(pt))
    sev = float(np.clip((share - 0.05) / 0.25, 0, 1)) if rel > 0 else 0.0
    return [mk("C.RES.03", ctx, raw_value=rel, severity=sev, unit="USD", scheme=Scheme.COOKIE_JAR, direction="income_increasing",
               impact=ImpactEstimate(0, rel * 0.5, rel, "disclosed_amount", "decrease in valuation allowance") if rel > 0 else None,
               rationale=f"Valuation allowance changed by {fmt_money(-rel)} ({fmt_pct(share)} of pre-tax income). Releases can be legitimate after sustained profitability.",
               evidence=ctx.evs(["VALUATION_ALLOWANCE"], ks=(0, 1)))]


@detector("C.RES.04", "Big bath", F, Area.RESERVES)
def cres04(ctx: RunContext):
    """Large charges last year (or with a management change) followed by a margin rebound."""
    def charges(k: int) -> float:
        return sum(ctx.a(i, k) for i in ("RESTRUCTURING", "ASSET_IMPAIRMENT", "GOODWILL_IMPAIRMENT") if finite(ctx.a(i, k)) and ctx.a(i, k) > 0)

    c1, rev1 = charges(1), ctx.a("REVENUE", 1)
    m0 = safe_div(ctx.a("OPERATING_INCOME"), ctx.a("REVENUE"))
    m1 = safe_div(ctx.a("OPERATING_INCOME", 1), rev1)
    need(m0, m1)
    big = safe_div(c1, rev1)
    rebound = m0 - m1
    mgmt = ctx.events_of("OFFICER_CHANGE", within_days=730)
    sev = float(np.clip((big - 0.03) / 0.07, 0, 1) * np.clip(rebound / 0.05, 0, 1))
    if mgmt and sev > 0:
        sev = min(1.0, sev + 0.2)
    return [mk("C.RES.04", ctx, raw_value=big, severity=sev, unit="charges / revenue (prior year)", scheme=Scheme.COOKIE_JAR,
               direction="income_decreasing",
               rationale=(f"Prior-year charges {fmt_money(c1)} ({fmt_pct(big)} of revenue); operating margin moved {fmt_pct(m1)} -> {fmt_pct(m0)}"
                          + (f"; {len(mgmt)} officer change(s) in the window." if mgmt else ".")),
               evidence=ctx.evs(["RESTRUCTURING", "ASSET_IMPAIRMENT", "GOODWILL_IMPAIRMENT"], ks=(1,)) + [ctx.ev_event(e) for e in mgmt[:2]])]


@detector("C.RES.05", "Effective tax rate drop / UTP release", F, Area.RESERVES, needs_peers=True)
def cres05(ctx: RunContext):
    """Effective tax rate falling vs own history and peers; decreases in unrecognized tax benefits."""
    fn = lambda v: safe_div(v.g("INCOME_TAX"), v.g("PRETAX_INCOME")) if v.g("PRETAX_INCOME") > 0 else float("nan")  # noqa: E731
    a = assess(ctx, fn, higher_is_bad=False, z_start=2.0)
    u1, u0 = ctx.a("UTP"), ctx.a("UTP", 1)
    utp = ""
    if finite(u1, u0) and u1 < u0:
        utp = f" Unrecognized tax benefits fell {fmt_money(u0 - u1)}."
    return [mk("C.RES.05", ctx, raw_value=a.cur, expected=a.baseline, severity=a.severity * 0.7, unit="ETR",
               percentile_peer=a.pct_peer, p_value=a.p_value, scheme=Scheme.ONE_TIME_GAINS, direction="income_increasing",
               impact=gap_impact(a, ctx.a("PRETAX_INCOME"), "ratio_reversion", "(baseline ETR - ETR) x pre-tax income", higher_is_bad=False),
               rationale=f"Effective tax rate: {a.describe()}.{utp}",
               evidence=ctx.evs(["INCOME_TAX", "PRETAX_INCOME", "UTP"], ks=(0, 1)))]


@detector("C.RES.06", "Litigation exposure vs accrual", F, Area.RESERVES, needs_text=True)
def cres06(ctx: RunContext):
    """Litigation language rising year over year while the loss-contingency accrual is flat."""
    if ctx.doc is None or ctx.prior_doc is None:
        raise SkipDetector("two years of 10-K text needed")
    pat = r"\b(?:lawsuit|litigation|class action|subpoena|investigation|alleg(?:e|es|ed|ation))\b"
    c1 = count(ctx.doc, pat, ("item3", "item8", "item1a"))
    c0 = count(ctx.prior_doc, pat, ("item3", "item8", "item1a"))
    acc1, acc0 = ctx.a("LITIGATION_ACCRUAL"), ctx.a("LITIGATION_ACCRUAL", 1)
    growth = safe_div(c1 - c0, max(c0, 10))
    flat = (not finite(acc1, acc0)) or acc1 <= acc0 * 1.05
    sev = float(np.clip((growth - 0.25) / 0.75, 0, 1)) * (1.0 if flat else 0.4) * 0.7
    return [mk("C.RES.06", ctx, raw_value=growth, severity=sev, unit="mention growth", area=Area.RESERVES,
               scheme=Scheme.HIDE_LIABILITIES,
               rationale=(f"Litigation terms: {c1} mentions vs {c0} last year ({fmt_pct(growth)}); accrual "
                          f"{fmt_money(acc1) if finite(acc1) else 'not tagged'}."))]


# ============================================================================ cash flow
@detector("C.CF.01", "CFO / net income divergence", F, Area.CASH_FLOW, needs_peers=True)
def ccf01(ctx: RunContext):
    """Rolling 3-year sum CFO / sum NI below 0.8 and falling."""
    def r3(v: V) -> float:
        ni = sum(v.g("NET_INCOME", b) for b in range(3))
        cfo = sum(v.g("CFO", b) for b in range(3))
        return safe_div(cfo, ni) if ni > 0 else float("nan")

    cur = r3(V(ctx.A, ctx.focus))
    need(cur, reason="3 years of positive net income needed")
    prev = r3(V(ctx.A, ctx.period_at(1))) if ctx.period_at(1) is not None else float("nan")
    peers = ctx.peer_values(lambda P, p, k: r3(V(P, p)))
    sev = float(np.clip((0.9 - cur) / 0.5, 0, 1))
    if finite(prev) and cur < prev - 0.1:
        sev = min(1.0, sev + 0.15)
    ni3 = sum(ctx.a("NET_INCOME", k) for k in range(3))
    imp = ImpactEstimate(0, (0.9 - cur) * ni3 / 3 * 0.5, (1.0 - cur) * ni3 / 3, "ratio_reversion",
                         "(0.9 - CFO/NI) x average NI") if cur < 0.9 else None
    return [mk("C.CF.01", ctx, raw_value=cur, expected=1.0, severity=sev, unit="x",
               percentile_peer=percentile_of(cur, peers) if len(peers) >= 3 else None, impact=imp,
               scheme=Scheme.PREMATURE_REVENUE, direction="income_increasing",
               rationale=f"3-year CFO / NI = {fmt_num(cur)}x (prior window {fmt_num(prev)}x; peer median {fmt_num(float(np.median(peers)) if peers else np.nan)}x).",
               evidence=ctx.evs(["CFO", "NET_INCOME"], ks=(0, 1, 2)))]


@detector("C.CF.02", "Working-capital-driven CFO", F, Area.CASH_FLOW)
def ccf02(ctx: RunContext):
    """One working-capital line supplies > 50% of CFO growth (payables stretch, receivables collection)."""
    cfo, cfo0 = ctx.a("CFO"), ctx.a("CFO", 1)
    need(cfo, cfo0)
    growth = cfo - cfo0
    if growth <= 0:
        return [mk("C.CF.02", ctx, raw_value=growth, severity=0.0, rationale="CFO did not grow.")]
    contrib = {}
    for item, label in (("CHANGE_AP", "payables"), ("CHANGE_AR", "receivables"), ("CHANGE_INVENTORY", "inventory"),
                        ("CHANGE_DEFERRED_REV", "deferred revenue")):
        a, b = ctx.a(item), ctx.a(item, 1)
        if finite(a, b):
            # CF sign: increase in AR/inventory reduces CFO (tagged as positive increase)
            delta = (a - b) if item in ("CHANGE_AP", "CHANGE_DEFERRED_REV") else -(a - b)
            contrib[label] = delta
    if not contrib:
        raise SkipDetector("working-capital lines not tagged")
    label, top = max(contrib.items(), key=lambda kv: kv[1])
    share = safe_div(top, growth)
    sev = float(np.clip((share - 0.5) / 0.5, 0, 1)) * (1.0 if label == "payables" else 0.7)
    return [mk("C.CF.02", ctx, raw_value=share, severity=sev, unit="share of CFO growth", scheme=Scheme.CFO_INFLATION,
               rationale=f"CFO grew {fmt_money(growth)}; the change in {label} supplied {fmt_money(top)} ({fmt_pct(share)}).",
               evidence=ctx.evs(["CFO"], ks=(0, 1)) + ctx.evs(["CHANGE_AP", "CHANGE_AR", "CHANGE_INVENTORY"], ks=(0,)))]


@detector("C.CF.03", "Supplier finance / payables stretch", F, Area.CASH_FLOW, needs_peers=True)
def ccf03(ctx: RunContext):
    """DPO jump vs history/peers, with supplier-finance obligations where disclosed (ASU 2022-04)."""
    fn = lambda v: safe_div(v.g("AP"), v.g("COGS")) * 365  # noqa: E731
    a = assess(ctx, fn, min_abs=7.0)
    sf = ctx.a("SUPPLIER_FINANCE")
    sev = a.severity
    note = ""
    if finite(sf) and sf > 0:
        sev = min(1.0, sev + 0.2 * float(np.clip(safe_div(sf, ctx.a("AP")) / 0.3, 0, 1)))
        note = f" Supplier-finance obligations {fmt_money(sf)} ({fmt_pct(safe_div(sf, ctx.a('AP')))} of payables)."
    return [mk("C.CF.03", ctx, raw_value=a.cur, expected=a.baseline, severity=sev, unit="days",
               percentile_peer=a.pct_peer, p_value=a.p_value, scheme=Scheme.CFO_INFLATION, area=Area.CASH_FLOW,
               impact=gap_impact(a, ctx.a("COGS") / 365, "ratio_reversion", "(DPO - baseline) x COGS / 365 (CFO effect)"),
               rationale=f"DPO: {a.describe(lambda x: f'{x:.0f} days')}.{note}",
               evidence=ctx.evs(["AP", "COGS", "SUPPLIER_FINANCE"], ks=(0, 1)))]


@detector("C.CF.04", "Costs moving from CFO to CFI", F, Area.CASH_FLOW)
def ccf04(ctx: RunContext):
    """(Capex + capitalized software) / CFO rising while D&A is flat: operating costs may be classified as investing."""
    def r(k: int) -> float:
        inv = ctx.a("CAPEX", k) + (ctx.a("CAP_SOFTWARE_ADDITIONS", k) if finite(ctx.a("CAP_SOFTWARE_ADDITIONS", k)) else 0)
        return safe_div(inv, ctx.a("CFO", k))

    if not (ctx.a("CFO") > 0 and ctx.a("CFO", 1) > 0):
        raise SkipDetector("operating cash flow not positive in both years; the ratio is not meaningful")
    cur, prev = r(0), r(1)
    need(cur, prev)
    da_g = safe_div(ctx.a("DA"), ctx.a("DA", 1)) - 1
    jump = cur - prev
    sev = float(np.clip((jump - 0.10) / 0.30, 0, 1)) * (1.0 if finite(da_g) and da_g < 0.10 else 0.5)
    return [mk("C.CF.04", ctx, raw_value=jump, severity=sev * 0.8, unit="pts", scheme=Scheme.CFO_INFLATION,
               rationale=f"Capitalized spending / CFO {fmt_pct(prev)} -> {fmt_pct(cur)}; D&A growth {fmt_pct(da_g)}.",
               evidence=ctx.evs(["CAPEX", "CAP_SOFTWARE_ADDITIONS", "CFO"], ks=(0, 1)))]


@detector("C.CF.05", "Acquisition-boosted CFO", F, Area.CASH_FLOW)
def ccf05(ctx: RunContext):
    """Large acquisitions with CFO growth outpacing revenue growth (acquired working capital unwinding)."""
    acq, ta = ctx.a("ACQUISITIONS"), ctx.a("TOTAL_ASSETS", 1)
    need(acq, ta, reason="no acquisitions this year")
    size = safe_div(acq, ta)
    cg = safe_div(ctx.a("CFO"), ctx.a("CFO", 1)) - 1
    rg = safe_div(ctx.a("REVENUE"), ctx.a("REVENUE", 1)) - 1
    sev = float(np.clip((size - 0.05) / 0.15, 0, 1) * np.clip((cg - rg) / 0.3, 0, 1))
    return [mk("C.CF.05", ctx, raw_value=cg - rg, severity=sev, unit="growth gap", scheme=Scheme.ACQUISITION_GAMES,
               area=Area.MA,
               rationale=f"Acquisitions {fmt_money(acq)} ({fmt_pct(size)} of assets); CFO growth {fmt_pct(cg)} vs revenue growth {fmt_pct(rg)}.",
               evidence=ctx.evs(["ACQUISITIONS", "CFO", "REVENUE"], ks=(0,)))]


@detector("C.CF.06", "Free cash flow vs EBITDA", F, Area.CASH_FLOW, needs_peers=True)
def ccf06(ctx: RunContext):
    """(CFO - capex) / EBITDA below peers and falling."""
    fn = lambda v: safe_div(v.g("CFO") - v.z("CAPEX"), v.g("OPERATING_INCOME") + v.z("DA")) if (v.g("OPERATING_INCOME") + v.z("DA")) > 0 else float("nan")  # noqa: E731
    a = assess(ctx, fn, higher_is_bad=False, z_start=1.5)
    return [mk("C.CF.06", ctx, raw_value=a.cur, expected=a.baseline, severity=a.severity * 0.8, unit="x",
               percentile_peer=a.pct_peer, p_value=a.p_value, scheme=Scheme.CFO_INFLATION,
               rationale=f"FCF / EBITDA: {a.describe(lambda x: fmt_num(x) + 'x')}.",
               evidence=ctx.evs(["CFO", "CAPEX", "OPERATING_INCOME", "DA"], ks=(0,)))]


def _debt(v: V, b: int = 0) -> float:
    return v.z("LONG_TERM_DEBT", b) + v.z("CURRENT_DEBT", b) + v.z("ST_BORROWINGS", b)


@detector("C.CF.07", "Quarter-end debt window dressing", F, Area.DEBT)
def ccf07(ctx: RunContext):
    """Year-end debt well below the average of the three preceding quarter-ends."""
    q = ctx.Q
    ps = [p for p in q.periods if p <= ctx.focus + pd.Timedelta(days=8)][-4:]
    if len(ps) < 4:
        raise SkipDetector("four quarter-ends of debt needed")
    debts = [sum(q.get(i, p) for i in ("LONG_TERM_DEBT", "CURRENT_DEBT", "ST_BORROWINGS") if finite(q.get(i, p))) for p in ps]
    if min(debts[:3]) <= 0:
        raise SkipDetector("quarterly debt not tagged")
    avg3 = float(np.mean(debts[:3]))
    dip = safe_div(avg3 - debts[3], avg3)
    sev = float(np.clip((dip - 0.10) / 0.30, 0, 1)) * 0.8
    return [mk("C.CF.07", ctx, raw_value=dip, severity=sev, unit="share", scheme=Scheme.HIDE_LIABILITIES,
               rationale=f"Year-end debt {fmt_money(debts[3])} vs {fmt_money(avg3)} average at the prior three quarter-ends ({fmt_pct(dip)} lower).")]


# ============================================================================ debt / off-balance-sheet
@detector("C.BS.01", "Implied interest rate", F, Area.DEBT, needs_peers=True)
def cbs01(ctx: RunContext):
    """Interest expense / average reported debt far above peers: possible unrecorded debt."""
    def fn(v: V) -> float:
        d = (_debt(v) + _debt(v, 1)) / 2
        return safe_div(v.g("INTEREST_EXPENSE"), d) if d > 0 else float("nan")

    a = assess(ctx, fn, z_start=2.0)
    if a.cur < 0.08:
        a.severity *= 0.3
    return [mk("C.BS.01", ctx, raw_value=a.cur, expected=a.baseline, severity=a.severity, unit="rate",
               percentile_peer=a.pct_peer, p_value=a.p_value, scheme=Scheme.HIDE_LIABILITIES,
               rationale=f"Interest expense / average debt: {a.describe()}.",
               evidence=ctx.evs(["INTEREST_EXPENSE", "LONG_TERM_DEBT", "CURRENT_DEBT"], ks=(0,)))]


@detector("C.BS.02", "Off-balance-sheet commitments growth", F, Area.DEBT)
def cbs02(ctx: RunContext):
    """Purchase obligations, guarantees and lease liabilities growing faster than revenue and debt."""
    def obl(k: int) -> float:
        vals = [ctx.a(i, k) for i in ("PURCHASE_OBLIGATIONS", "GUARANTEES")]
        return sum(v for v in vals if finite(v)) if any(finite(v) for v in vals) else float("nan")

    o1, o0 = obl(0), obl(1)
    need(o1, o0, reason="commitments not tagged")
    g = safe_div(o1, o0) - 1
    rg = safe_div(ctx.a("REVENUE"), ctx.a("REVENUE", 1)) - 1
    size = safe_div(o1, ctx.a("TOTAL_ASSETS"))
    sev = float(np.clip((g - rg - 0.25) / 0.75, 0, 1)) * float(np.clip(size / 0.10, 0, 1))
    switched = [n for n in (ctx.concept_switched("PURCHASE_OBLIGATIONS"), ctx.concept_switched("GUARANTEES")) if n]
    if switched:
        sev *= 0.4
    return [mk("C.BS.02", ctx, raw_value=g, severity=sev, unit="growth", scheme=Scheme.HIDE_LIABILITIES,
               rationale=(f"Purchase obligations and guarantees {fmt_money(o1)} ({fmt_pct(size)} of assets) grew {fmt_pct(g)} vs revenue {fmt_pct(rg)}."
                          + (f" Discounted: {'; '.join(switched)}, so the years may not be comparable." if switched else "")),
               evidence=ctx.evs(["PURCHASE_OBLIGATIONS", "GUARANTEES"], ks=(0, 1)))]


@detector("C.BS.03", "Covenant proximity", F, Area.DEBT, needs_llm=True)
def cbs03(ctx: RunContext):
    """Leverage / coverage vs covenant thresholds extracted from debt notes."""
    from ffi.detectors.e_text import covenant_signal

    return covenant_signal(ctx)


@detector("C.BS.04", "Unconsolidated affiliates", F, Area.DEBT)
def cbs04(ctx: RunContext):
    """Loss-making equity-method investees that are material relative to equity."""
    inc, inv = ctx.a("EQUITY_METHOD_INCOME"), ctx.a("EQUITY_METHOD_INVESTMENTS")
    need(inv, reason="no equity-method investments")
    size = safe_div(inv, ctx.a("EQUITY"))
    loss = finite(inc) and inc < 0
    sev = float(np.clip((size - 0.05) / 0.20, 0, 1)) * (1.0 if loss else 0.3) * 0.7
    return [mk("C.BS.04", ctx, raw_value=size, severity=sev, unit="share of equity", scheme=Scheme.HIDE_LIABILITIES,
               rationale=f"Equity-method investments {fmt_money(inv)} ({fmt_pct(size)} of equity); share of their results {fmt_money(inc)}.",
               evidence=ctx.evs(["EQUITY_METHOD_INVESTMENTS", "EQUITY_METHOD_INCOME"], ks=(0,)))]


# ============================================================================ M&A
@detector("C.MA.01", "Purchase-price allocation to goodwill", F, Area.MA)
def cma01(ctx: RunContext):
    """Goodwill acquired / acquisition spend; high shares avoid amortization of identifiable intangibles."""
    ga, acq = ctx.a("GOODWILL_ACQUIRED"), ctx.a("ACQUISITIONS")
    need(ga, acq, reason="no acquisitions with goodwill this year")
    if acq <= 0:
        raise SkipDetector("no acquisition spend")
    share = safe_div(ga, acq)
    sev = float(np.clip((share - 0.70) / 0.25, 0, 1)) * float(np.clip(safe_div(acq, ctx.a("TOTAL_ASSETS")) / 0.05, 0.3, 1))
    return [mk("C.MA.01", ctx, raw_value=share, expected=0.6, severity=sev, unit="share", scheme=Scheme.ACQUISITION_GAMES,
               direction="income_increasing",
               rationale=f"Goodwill recognized {fmt_money(ga)} = {fmt_pct(share)} of cash paid for acquisitions {fmt_money(acq)}.",
               evidence=ctx.evs(["GOODWILL_ACQUIRED", "ACQUISITIONS"], ks=(0,)))]


@detector("C.MA.02", "Delayed goodwill impairment", F, Area.MA)
def cma02(ctx: RunContext):
    """Market value (public float proxy) below book equity with material goodwill and no impairment."""
    pf, eq, gw = ctx.a("PUBLIC_FLOAT"), ctx.a("EQUITY"), ctx.a("GOODWILL")
    need(eq, gw, reason="equity or goodwill not reported")
    if not finite(pf):
        raise SkipDetector("public float (market value proxy) not reported")
    imp = ctx.a("GOODWILL_IMPAIRMENT")
    gw_share = safe_div(gw, ctx.a("TOTAL_ASSETS"))
    gap = safe_div(eq - pf, eq)
    sev = 0.0
    if pf < eq and (not finite(imp) or imp <= 0) and gw_share > 0.05:
        sev = float(np.clip(gap / 0.4, 0, 1)) * float(np.clip(gw_share / 0.2, 0.4, 1))
    return [mk("C.MA.02", ctx, raw_value=safe_div(pf, eq), expected=1.0, severity=sev, unit="market/book (float proxy)",
               scheme=Scheme.SHIFT_EXPENSES_LATER, direction="income_increasing",
               impact=ImpactEstimate(0, min(gw, eq - pf) * 0.5, min(gw, eq - pf), "policy_counterfactual",
                                     "book equity in excess of market value, capped at goodwill") if sev > 0 else None,
               rationale=(f"Public float {fmt_money(pf)} vs book equity {fmt_money(eq)}; goodwill {fmt_money(gw)} ({fmt_pct(gw_share)} of assets); "
                          f"impairment this year: {fmt_money(imp) if finite(imp) else 'none tagged'}. Public float excludes affiliates' shares, so it understates market value."),
               evidence=ctx.evs(["PUBLIC_FLOAT", "EQUITY", "GOODWILL"], ks=(0,)))]


@detector("C.MA.03", "Acquisition-reliant growth", F, Area.MA)
def cma03(ctx: RunContext):
    """3-year acquisition spend large relative to revenue (organic growth hard to verify)."""
    spend = sum(ctx.a("ACQUISITIONS", k) for k in range(3) if finite(ctx.a("ACQUISITIONS", k)))
    rev = ctx.a("REVENUE")
    need(rev)
    share = safe_div(spend, rev)
    sev = float(np.clip((share - 0.20) / 0.40, 0, 1)) * 0.6
    return [mk("C.MA.03", ctx, raw_value=share, severity=sev, unit="3y spend / revenue", scheme=Scheme.ACQUISITION_GAMES,
               rationale=f"Acquisitions over 3 years {fmt_money(spend)} = {fmt_pct(share)} of current revenue.",
               evidence=ctx.evs(["ACQUISITIONS"], ks=(0, 1, 2)))]


@detector("C.MA.04", "Contingent-consideration gains", F, Area.MA)
def cma04(ctx: RunContext):
    """Earn-out liabilities reduced and booked as gains, relative to pre-tax income."""
    ch, pt = ctx.a("CONTINGENT_CONSIDERATION_CHANGE"), ctx.a("PRETAX_INCOME")
    need(ch, pt, reason="contingent consideration not tagged")
    gain = -ch if ch < 0 else 0.0
    share = safe_div(gain, abs(pt))
    sev = float(np.clip((share - 0.02) / 0.10, 0, 1))
    return [mk("C.MA.04", ctx, raw_value=gain, severity=sev, unit="USD", scheme=Scheme.ONE_TIME_GAINS, direction="income_increasing",
               impact=ImpactEstimate(0, gain, gain, "disclosed_amount", "reduction in contingent consideration") if gain > 0 else None,
               rationale=f"Contingent-consideration change {fmt_money(ch)} ({fmt_pct(share)} of pre-tax income).",
               evidence=ctx.evs(["CONTINGENT_CONSIDERATION_CHANGE"], ks=(0,)))]


@detector("C.MA.05", "Measurement-period adjustments", F, Area.MA)
def cma05(ctx: RunContext):
    """Large post-close goodwill adjustments relative to goodwill acquired."""
    adj, ga = ctx.a("GOODWILL_MEASUREMENT_ADJ"), ctx.a("GOODWILL_ACQUIRED", 1)
    need(adj, reason="measurement-period adjustments not tagged")
    base = ga if finite(ga) and ga > 0 else ctx.a("GOODWILL")
    share = abs(safe_div(adj, base))
    sev = float(np.clip((share - 0.10) / 0.30, 0, 1)) * 0.7
    return [mk("C.MA.05", ctx, raw_value=adj, severity=sev, unit="USD", scheme=Scheme.ACQUISITION_GAMES,
               rationale=f"Measurement-period goodwill adjustments {fmt_money(adj)} ({fmt_pct(share)} of the base).",
               evidence=ctx.evs(["GOODWILL_MEASUREMENT_ADJ", "GOODWILL_ACQUIRED"], ks=(0, 1)))]


# ============================================================================ governance & events
@detector("C.GOV.01", "Auditor change", F, Area.GOVERNANCE)
def cgov01(ctx: RunContext):
    """8-K Item 4.01 (change in certifying accountant) within two years of the focus year end."""
    ev = ctx.events_of("AUDITOR_CHANGE", within_days=730)
    sev = 0.55 if ev else 0.0
    return [mk("C.GOV.01", ctx, raw_value=len(ev), severity=sev, unit="events", tags=["pressure"],
               rationale=(f"{len(ev)} auditor change filing(s) (8-K Item 4.01) in the two years to {ctx.focus_label}."
                          if ev else "No auditor change in the window.")
               + (f" Current auditor: {ctx.auditor}." if ctx.auditor else ""),
               evidence=[ctx.ev_event(e, "primary") for e in ev[:3]])]


@detector("C.GOV.02", "Non-reliance / restatement filings", F, Area.GOVERNANCE)
def cgov02(ctx: RunContext):
    """8-K Item 4.02 non-reliance and annual-report amendments (10-K/A, 20-F/A, 40-F/A) within three years.

    Amendments filed on the same day are one action (issuers sometimes amend several years at once), and an
    amendment that re-files financial statements carries XBRL financial data; exhibit-only and Part III
    amendments do not and are routine."""
    nr = ctx.events_of("NON_RELIANCE", within_days=1095)
    am = ctx.events_of("AMENDED_10K", within_days=1095)
    fin_accns = set(ctx.store.df["accn"]) if not ctx.store.df.empty else set()
    fin_days = {e.date for e in am if e.accession in fin_accns}
    other_days = {e.date for e in am} - fin_days
    sev = min(1.0, 0.8 * len(nr) + 0.35 * len(fin_days) + 0.1 * len(other_days))
    return [mk("C.GOV.02", ctx, raw_value=len(nr) + len(fin_days) + len(other_days), severity=sev, unit="events", tags=["pressure"],
               scheme=Scheme.REPORTING_INTEGRITY,
               rationale=(f"{len(nr)} non-reliance 8-K(s) and {len(am)} amended annual report(s) in the last three years, filed on "
                          f"{len(fin_days) + len(other_days)} day(s); {len(fin_days)} of those day(s) re-filed financial statements "
                          "(the others amended exhibits or Part III information only)."),
               evidence=[ctx.ev_event(e, "primary") for e in (nr + am)[:4]])]


@detector("C.GOV.03", "Officer turnover", F, Area.GOVERNANCE)
def cgov03(ctx: RunContext):
    """8-K Item 5.02 officer/director departures (CFO exits are the strongest signal)."""
    ev = ctx.events_of("OFFICER_CHANGE", within_days=730)
    cfo = [e for e in ev if "CFO/CAO change" in e.description]
    ceo = [e for e in ev if "CEO change" in e.description]
    sev = min(1.0, 0.05 * len(ev) + 0.35 * len(cfo) + 0.2 * len(ceo))
    return [mk("C.GOV.03", ctx, raw_value=len(ev), severity=sev * 0.8, unit="events", tags=["pressure"],
               rationale=(f"{len(ev)} Item 5.02 filing(s) in two years; {len(cfo)} describe a CFO/CAO departure or appointment"
                          f" and {len(ceo)} a CEO change. A planned succession is a common benign cause."),
               evidence=[ctx.ev_event(e) for e in (cfo or ev)[:3]])]


@detector("C.GOV.04", "Late filing", F, Area.GOVERNANCE)
def cgov04(ctx: RunContext):
    """NT 10-K / NT 10-Q notifications of late filing within three years."""
    ev = ctx.events_of("LATE_FILING", within_days=1095)
    sev = min(1.0, 0.45 * len(ev))
    return [mk("C.GOV.04", ctx, raw_value=len(ev), severity=sev, unit="events", tags=["pressure"],
               rationale=f"{len(ev)} late-filing notification(s) (NT 10-K/10-Q) in three years.",
               evidence=[ctx.ev_event(e, "primary") for e in ev[:3]])]


@detector("C.GOV.05", "Internal-control weakness", F, Area.GOVERNANCE, needs_text=True)
def cgov05(ctx: RunContext):
    """Material weakness or ineffective internal control language in Item 9A."""
    if ctx.doc is None:
        raise SkipDetector("10-K text not available")
    pat = r"material weakness(?:es)?|(?:were|was) not effective|did not maintain effective"
    hits = [h for h in search(ctx.doc, pat, ("item9a",)) if not re.search(r"\bno material weakness", h.quote, re.I)]
    neg = [h for h in hits if re.search(r"not effective|did not maintain|identified (?:a )?material weakness", h.quote, re.I)]
    sev = 0.85 if neg else (0.25 if hits else 0.0)
    return [mk("C.GOV.05", ctx, raw_value=len(neg), severity=sev, unit="passages", tags=["pressure"],
               scheme=Scheme.REPORTING_INTEGRITY,
               rationale="Item 9A reports a material weakness or ineffective controls." if neg else
               ("Item 9A discusses material weaknesses (no adverse conclusion found)." if hits else "Item 9A reports effective internal control."),
               evidence=[ctx.ev_span(h.doc, h.start, h.end, "Item 9A") for h in (neg or hits)[:2]])]


@detector("C.GOV.06", "Going-concern language", F, Area.GOVERNANCE, needs_text=True)
def cgov06(ctx: RunContext):
    """'Substantial doubt about ... ability to continue as a going concern' in the 10-K."""
    if ctx.doc is None:
        raise SkipDetector("10-K text not available")
    hits = search(ctx.doc, r"substantial doubt (?:about|regarding|as to) (?:the company's|our|its) ability to continue as a going concern")
    hits = [h for h in hits if not re.search(r"\b(?:no|not|alleviat)", h.quote[:120], re.I)]
    return [mk("C.GOV.06", ctx, raw_value=len(hits), severity=0.8 if hits else 0.0, unit="passages", tags=["pressure"],
               rationale="Going-concern doubt is disclosed." if hits else "No going-concern doubt disclosed.",
               evidence=[ctx.ev_span(h.doc, h.start, h.end, "Going concern") for h in hits[:2]])]


