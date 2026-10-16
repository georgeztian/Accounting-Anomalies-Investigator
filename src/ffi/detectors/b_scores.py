"""Family B: forensic scores and accrual models (BLUEPRINT §9, Family B). Scores are features, not verdicts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ffi.context import RunContext, pv
from ffi.detectors import scores as S
from ffi.detectors.base import Area, EvidenceRef, ImpactEstimate, Scheme, SkipDetector, detector
from ffi.detectors.helpers import finite, mk
from ffi.detectors.stats import (
    fmt_money,
    fmt_num,
    fmt_pct,
    percentile_of,
    robust_z,
    safe_div,
    severity_from_z,
)
from ffi.detectors.xsec_models import kothari, roychowdhury

F = "B"


def _derived(name: str, value: float, label: str) -> EvidenceRef:
    return EvidenceRef(node_id=f"derived:{name}", kind="derived", label=label, value=value if np.isfinite(value) else None,
                       role="supporting")


@detector("B.01", "Beneish M-score (8-variable)", F, Area.EARNINGS_QUALITY, needs_peers=True)
def b01(ctx: RunContext):
    """M = -4.84 + 0.920 DSRI + 0.528 GMI + 0.404 AQI + 0.892 SGI + 0.115 DEPI - 0.172 SGAI + 4.679 TATA - 0.327 LVGI."""
    r = S.beneish(ctx.A, ctx.focus)
    m = r["M"]
    if not finite(m):
        raise SkipDetector("sales growth or accruals unavailable")
    peers = ctx.peer_values(lambda P, p, k: S.beneish(P, p)["M"])
    pct = percentile_of(m, peers) if len(peers) >= 3 else None
    sev = float(np.clip((m + 2.4) / 1.4, 0, 1))
    contrib = {k: S.BENEISH_W[k] * (r[k] if finite(r[k]) else (0 if k == "TATA" else 1)) for k in S.BENEISH_W}
    neutral = {k: S.BENEISH_W[k] * (0 if k == "TATA" else 1) for k in S.BENEISH_W}
    drivers = sorted(((contrib[k] - neutral[k], k) for k in S.BENEISH_W), reverse=True)[:3]
    tags = [f"driver:{k}" for d, k in drivers if d > 0.05]
    ev = [_derived(f"beneish.{k}", r[k], f"{k} = {fmt_num(r[k], 3)} (weight {S.BENEISH_W[k]:+.3f})") for k in S.BENEISH_W]
    ev += ctx.evs(["AR", "REVENUE"], ks=(0, 1), role="supporting")
    miss = r["_missing"]
    return [mk("B.01", ctx, raw_value=m, expected=-2.22, severity=sev, unit="score", percentile_peer=pct, tags=tags,
               scheme=Scheme.PREMATURE_REVENUE if "driver:DSRI" in tags else None,
               rationale=(f"M-score {m:.2f} (reference flag > -1.78; peer median "
                          f"{fmt_num(float(np.median(peers)) if peers else np.nan)}). Largest upward contributions: "
                          + ", ".join(f"{k} {fmt_num(r[k], 2)}" for d, k in drivers if finite(r[k]))
                          + (f". Imputed neutral for missing: {', '.join(miss)}." if miss else ".")),
               evidence=ev)]


@detector("B.02", "Dechow F-score (model 1)", F, Area.EARNINGS_QUALITY, needs_peers=True)
def b02(ctx: RunContext):
    """F = P(misstatement) / 0.0037 from RSST accruals, receivables, inventory, soft assets, cash sales, ROA, issuance."""
    r = S.dechow_f(ctx.A, ctx.focus)
    f = r["F"]
    if not finite(f):
        raise SkipDetector("balance-sheet inputs for RSST accruals unavailable")
    peers = ctx.peer_values(lambda P, p, k: S.dechow_f(P, p)["F"])
    pct = percentile_of(f, peers) if len(peers) >= 3 else None
    sev = float(np.clip((f - 0.8) / 1.8, 0, 1))
    ev = [_derived(f"dechow.{k}", r[k], f"{k} = {fmt_num(r[k], 3)} (coef {S.F_COEF[k]:+.3f})") for k in S.F_COEF if k != "const"]
    return [mk("B.02", ctx, raw_value=f, expected=1.0, severity=sev, unit="score", percentile_peer=pct,
               rationale=(f"F-score {f:.2f} (1.0 = unconditional misstatement rate; > 2.45 high). "
                          f"RSST accruals {fmt_pct(r['rsst_acc'])} of assets, soft assets {fmt_pct(r['soft_assets'])}, "
                          f"change in ROA {fmt_pct(r['ch_roa'])}, securities issued: {'yes' if r['issue'] else 'no'}."),
               evidence=ev + ctx.evs(["TOTAL_ASSETS", "NET_INCOME"], ks=(0,), role="supporting"))]


@detector("B.03", "Discretionary accruals (Kothari modified Jones)", F, Area.ACCRUALS)
def b03(ctx: RunContext):
    """Residual of TACC/A on 1/A, (dREV-dREC)/A, PPE/A, ROA estimated across the industry-year."""
    res = kothari(ctx)
    if res is None:
        raise SkipDetector("industry cross-section too small or target inputs missing")
    prev = kothari(ctx, 1)
    pct = percentile_of(res.residual, res.peers)
    z = robust_z(res.residual, res.peers)
    two_year = res.residual + (prev.residual if prev else 0.0)
    sev = severity_from_z(z, 1.0, 3.0)
    if prev is not None and prev.residual > 0 and res.residual > 0:
        sev = min(1.0, sev + 0.15)
    da_usd = res.residual * res.lag_assets
    impact = None
    if res.residual > 0:
        lo = max(0.0, (res.residual - float(np.quantile(res.peers, 0.75))) * res.lag_assets)
        impact = ImpactEstimate(lo, (lo + da_usd) / 2, da_usd, "accrual_reversal",
                                f"discretionary accruals {fmt_pct(res.residual)} x lagged assets {fmt_money(res.lag_assets)}")
    return [mk("B.03", ctx, raw_value=res.residual, expected=0.0, severity=sev, unit="share of assets",
               percentile_peer=pct, scheme=Scheme.SHIFT_EXPENSES_LATER if res.residual > 0 else None,
               direction="income_increasing" if res.residual > 0 else "income_decreasing",
               impact=impact,
               rationale=(f"Discretionary accruals {fmt_pct(res.residual)} of lagged assets ({fmt_money(da_usd)}), "
                          f"{pct:.0f}th percentile of {res.n} {res.scope} firms (CY{res.year}); "
                          f"two-year sum {fmt_pct(two_year)}."),
               evidence=ctx.evs(["NET_INCOME", "CFO", "TOTAL_ASSETS"], ks=(0,)) + ctx.evs(["TOTAL_ASSETS"], ks=(1,)))]


def _dwc(P, p, back: int = 0) -> float:
    def d(item: str) -> float:
        a, b = pv(P, item, p, back), pv(P, item, p, back + 1)
        return (a - b) if (np.isfinite(a) and np.isfinite(b)) else 0.0
    return d("AR") + d("INVENTORY") + d("OTHER_CURRENT_ASSETS") - d("AP") - d("ACCRUED_LIABILITIES")


def _aq_rows(P, periods: list[pd.Timestamp]) -> list[tuple[pd.Timestamp, list[float], float]]:
    rows = []
    for p in periods:
        aA = (pv(P, "TOTAL_ASSETS", p) + pv(P, "TOTAL_ASSETS", p, 1)) / 2
        cfo_m1, cfo_0 = pv(P, "CFO", p, 1), pv(P, "CFO", p)
        idx = list(P.values.index)
        i = idx.index(p)
        if i + 1 >= len(idx):
            continue
        cfo_p1 = P.get("CFO", idx[i + 1])
        drev = pv(P, "REVENUE", p) - pv(P, "REVENUE", p, 1)
        ppe = pv(P, "PPE_NET", p)
        x = [cfo_m1, cfo_0, cfo_p1, drev, ppe]
        y = _dwc(P, p)
        if np.isfinite(aA) and aA > 0 and all(np.isfinite(x)):
            rows.append((p, [v / aA for v in x], y / aA))
    return rows


@detector("B.04", "Accrual quality (Dechow-Dichev / McNichols)", F, Area.ACCRUALS, needs_peers=True)
def b04(ctx: RunContext):
    """Std. dev. of residuals from dWC ~ CFO(t-1), CFO(t), CFO(t+1), dREV, PPE; pooled peer coefficients."""
    if not ctx.peers or len(ctx.peers.loaded) < 4:
        raise SkipDetector("needs at least 4 peers with history")
    pooled = []
    peer_rows = {}
    for peer in ctx.peers.loaded:
        r = _aq_rows(peer.annual, list(peer.annual.values.index)[1:])
        peer_rows[peer.cik] = r
        pooled.extend(r)
    own = _aq_rows(ctx.A, [p for p in ctx.A.periods if p < ctx.focus][1:])
    pooled.extend(own)
    if len(pooled) < 25 or len(own) < 3:
        raise SkipDetector("not enough firm-years for the accrual-quality regression")
    X = np.array([r[1] for r in pooled])
    y = np.array([r[2] for r in pooled])
    Xc = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(Xc, y, rcond=None)

    def aq(rows) -> float:
        rows = rows[-5:]
        if len(rows) < 3:
            return float("nan")
        res = [r[2] - float(np.r_[1.0, r[1]] @ coef) for r in rows]
        return float(np.std(res, ddof=1))

    mine = aq(own)
    peer_aq = [v for v in (aq(r) for r in peer_rows.values()) if np.isfinite(v)]
    pct = percentile_of(mine, peer_aq) if len(peer_aq) >= 3 else None
    z = robust_z(mine, peer_aq)
    sev = severity_from_z(z, 1.0, 3.0) * 0.8
    return [mk("B.04", ctx, raw_value=mine, expected=float(np.median(peer_aq)) if peer_aq else None, severity=sev,
               unit="std of residual accruals", percentile_peer=pct, period=ctx.label(own[-1][0]),
               rationale=(f"Accrual-quality std {fmt_num(mine, 4)} over the last {min(5, len(own))} years with t+1 cash flows "
                          f"(peer median {fmt_num(float(np.median(peer_aq)) if peer_aq else np.nan, 4)}). Higher = accruals map less well to cash."),
               evidence=ctx.evs(["CFO", "AR", "INVENTORY"], ks=(1,), role="supporting"))]


@detector("B.05", "Sloan accrual ratio", F, Area.ACCRUALS, needs_peers=True)
def b05(ctx: RunContext):
    """(NI - CFO - CFI) / average net operating assets; top decile = low earnings persistence."""
    v = S.sloan_accruals(ctx.A, ctx.focus)
    if not finite(v):
        raise SkipDetector("inputs unavailable")
    peers = ctx.peer_values(lambda P, p, k: S.sloan_accruals(P, p))
    hist = [S.sloan_accruals(ctx.A, p) for p in ctx.periods[-6:-1]]
    pct = percentile_of(v, peers) if len(peers) >= 3 else None
    z = robust_z(v, peers) if len(peers) >= 5 else robust_z(v, hist)
    sev = severity_from_z(z, 1.0, 3.5) * (1.0 if v > 0 else 0.4)
    return [mk("B.05", ctx, raw_value=v, expected=float(np.median(peers)) if peers else None, severity=sev,
               unit="ratio", percentile_peer=pct, direction="income_increasing" if v > 0 else "neutral",
               rationale=f"Sloan accrual ratio {fmt_pct(v)} (peer median {fmt_pct(float(np.median(peers)) if peers else np.nan)}).",
               evidence=ctx.evs(["NET_INCOME", "CFO", "CFI"], ks=(0,)))]


@detector("B.06", "Montier C-score", F, Area.EARNINGS_QUALITY)
def b06(ctx: RunContext):
    """Six binary tests: NI-CFO gap widening, DSO up, DSI up, other CA/revenue up, depreciation rate down, assets +10%."""
    r = S.montier_c(ctx.A, ctx.focus)
    if r["available"] < 4:
        raise SkipDetector("fewer than 4 of 6 tests computable")
    failed = [k.replace("_", " ") for k in ("ni_cfo_gap_widening", "dso_rising", "dsi_rising", "other_ca_rising", "dep_rate_falling", "asset_growth_gt_10pct") if r[k] == 1.0]
    sev = float(np.clip((r["C"] - 2) / 3, 0, 1))
    return [mk("B.06", ctx, raw_value=r["C"], expected=2.0, severity=sev, unit=f"of {int(r['available'])}",
               rationale=f"C-score {int(r['C'])}/{int(r['available'])}" + (f": {', '.join(failed)}." if failed else "."))]


@detector("B.07", "Piotroski F-score (context)", F, Area.EARNINGS_QUALITY)
def b07(ctx: RunContext):
    """Nine fundamental-strength tests; low scores add context (weakness), not anomaly."""
    r = S.piotroski_f(ctx.A, ctx.focus)
    sev = float(np.clip((4 - r["F"]) / 4, 0, 1)) * 0.5
    return [mk("B.07", ctx, raw_value=r["F"], expected=5.0, severity=sev, unit="of 9", flag_at=0.45,
               rationale=f"Piotroski F-score {int(r['F'])}/9 (low = weak fundamentals, a pressure factor).")]


@detector("B.08", "Distress pressure (Altman Z)", F, Area.GOVERNANCE)
def b08(ctx: RunContext):
    """Z (with public float as market value) or Z'' (book-based); distress motivates manipulation."""
    r = S.altman(ctx.A, ctx.focus)
    z, zpp = r["Z"], r["Zpp"]
    if finite(z):
        sev = float(np.clip((2.99 - z) / (2.99 - 1.81), 0, 1)) * 0.7
        txt = f"Altman Z {z:.2f} (distress < 1.81, safe > 2.99; public float used as market value)"
        val = z
    elif finite(zpp):
        sev = float(np.clip((2.6 - zpp) / (2.6 - 1.1), 0, 1)) * 0.7
        txt = f"Altman Z'' {zpp:.2f} (distress < 1.1, safe > 2.6)"
        val = zpp
    else:
        raise SkipDetector("inputs unavailable")
    return [mk("B.08", ctx, raw_value=val, severity=sev, unit="score", flag_at=0.45, tags=["pressure"],
               rationale=txt + ". Used as a pressure feature, not an anomaly by itself.")]


@detector("B.09", "Earnings management around thresholds", F, Area.EARNINGS_QUALITY)
def b09(ctx: RunContext):
    """Frequency of just-positive results over 12 quarters, plus real earnings management (Roychowdhury)."""
    out = []
    ni = ctx.qhist("NET_INCOME", 12)
    ta = ctx.qhist("TOTAL_ASSETS", 12).reindex(ni.index).ffill()
    eps = ctx.qhist("EPS_DILUTED", 16)
    # leveraged balance sheets (banks, insurers, REITs) earn a quarterly ROA of 0-0.5% as a matter of course, and the
    # Roychowdhury production / discretionary-spend models describe industrial operating cycles
    industrial = ctx.profile is None or ctx.profile.key == "industrial"
    if len(ni.dropna()) >= 8:
        roa = (ni / ta).dropna()
        small_profit = int(((roa >= 0) & (roa <= 0.005)).sum()) if industrial else 0
        yoy = (eps - eps.shift(4)).dropna().tail(12)
        small_inc = int(((yoy >= 0) & (yoy <= 0.01)).sum())
        n = len(roa) if industrial else 0
        rate = (small_profit + small_inc) / max(n + len(yoy), 1)
        sev = float(np.clip((rate - 0.10) / 0.25, 0, 1)) * 0.8
        roa_txt = f"{small_profit} of {n} quarters with ROA in [0, 0.5%] and " if industrial else ""
        out.append(mk("B.09", ctx, raw_value=rate, expected=0.05, severity=sev, unit="share of quarters",
                      tags=["threshold"], scheme=Scheme.COOKIE_JAR if sev > 0 else None,
                      rationale=(f"{roa_txt}{small_inc} of {len(yoy)} quarters with diluted EPS up by 0-1 cent year over year."
                                 + ("" if industrial else " (The small-ROA test does not apply to this company profile.)"))))
    rem = roychowdhury(ctx) if industrial else None
    if rem:
        parts, score, zs = [], 0.0, {}
        for k, sign in (("CFO", -1), ("PROD", 1), ("DISX", -1)):
            if k in rem:
                z = robust_z(rem[k].residual, rem[k].peers)
                zs[k] = z
                score += sign * (z if np.isfinite(z) else 0)
                parts.append(f"abnormal {k} {fmt_pct(rem[k].residual)} of assets (z {z:+.1f})")
        comp = score / max(len(zs), 1)
        sev = severity_from_z(comp, 1.0, 2.5)
        any_r = next(iter(rem.values()))
        out.append(mk("B.09", ctx, raw_value=comp, expected=0.0, severity=sev, unit="composite z", tags=["real_em"],
                      scheme=Scheme.SHIFT_EXPENSES_LATER if sev >= 0.5 else None,
                      direction="income_increasing" if comp > 0 else "neutral",
                      rationale=(f"Real earnings-management composite {comp:+.2f} vs {any_r.n} {any_r.scope} firms (CY{any_r.year}): "
                                 + "; ".join(parts) + ". Positive = low cash flow, overproduction or cut discretionary spend."),
                      evidence=ctx.evs(["CFO", "COGS", "SGA"], ks=(0,), role="supporting")))
    if not out:
        raise SkipDetector("insufficient quarterly history and cross-section")
    return out


@detector("B.10", "Income smoothing (Leuz)", F, Area.RESERVES, needs_peers=True)
def b10(ctx: RunContext):
    """sigma(operating income)/sigma(CFO) and corr(d accruals, d CFO) over 6-8 years vs peers."""
    def smooth(P, p, n=8):
        idx = [q for q in P.values.index if q <= p][-n:]
        oi = pd.Series([P.get("OPERATING_INCOME", q) for q in idx]) / pd.Series([P.get("TOTAL_ASSETS", q) for q in idx])
        cfo = pd.Series([P.get("CFO", q) for q in idx]) / pd.Series([P.get("TOTAL_ASSETS", q) for q in idx])
        ni = pd.Series([P.get("NET_INCOME", q) for q in idx]) / pd.Series([P.get("TOTAL_ASSETS", q) for q in idx])
        ok = oi.notna() & cfo.notna()
        if ok.sum() < 5:
            return float("nan"), float("nan")
        ratio = safe_div(float(oi[ok].std()), float(cfo[ok].std()))
        acc = (ni - cfo).diff()
        corr = float(acc.corr(cfo.diff())) if (acc.notna() & cfo.diff().notna()).sum() >= 4 else float("nan")
        return ratio, corr

    ratio, corr = smooth(ctx.A, ctx.focus)
    if not finite(ratio):
        raise SkipDetector("fewer than 5 years of operating income and CFO")
    peers = ctx.peer_values(lambda P, p, k: smooth(P, p)[0])
    pct = percentile_of(ratio, peers) if len(peers) >= 3 else None
    z = -robust_z(ratio, peers) if len(peers) >= 5 else float("nan")
    sev = severity_from_z(z, 1.0, 3.0) * 0.7 if np.isfinite(z) else (0.4 if ratio < 0.3 and finite(corr) and corr < -0.8 else 0.0)
    return [mk("B.10", ctx, raw_value=ratio, expected=float(np.median(peers)) if peers else None, severity=sev,
               unit="ratio", percentile_peer=pct, scheme=Scheme.COOKIE_JAR if sev >= 0.5 else None,
               rationale=(f"Volatility of operating income / volatility of CFO = {fmt_num(ratio)} "
                          f"(peer median {fmt_num(float(np.median(peers)) if peers else np.nan)}); "
                          f"corr(d accruals, d CFO) = {fmt_num(corr)}. Low ratio + strongly negative correlation suggests smoothing."))]
