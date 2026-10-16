"""Family D: statistical and ML detectors (BLUEPRINT §9, Family D): find patterns no rule encodes."""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats as sps

from ffi.context import RunContext
from ffi.detectors.base import Area, ImpactEstimate, SkipDetector, detector
from ffi.detectors.helpers import finite, mk
from ffi.detectors.scores import V
from ffi.detectors.stats import (
    fmt_money,
    fmt_num,
    fmt_pct,
    robust_z,
    safe_div,
    severity_from_z,
)
from ffi.models_store import load_model

F = "D"

# ---------------------------------------------------------------------------- ratio library
RATIOS: dict[str, tuple[str, Callable[[V], float]]] = {
    "gross_margin": ("Gross margin", lambda v: v.gross_margin()),
    "op_margin": ("Operating margin", lambda v: safe_div(v.g("OPERATING_INCOME"), v.g("REVENUE"))),
    "net_margin": ("Net margin", lambda v: safe_div(v.g("NET_INCOME"), v.g("REVENUE"))),
    "sga_ratio": ("SG&A / revenue", lambda v: safe_div(v.g("SGA"), v.g("REVENUE"))),
    "rnd_ratio": ("R&D / revenue", lambda v: safe_div(v.g("RND"), v.g("REVENUE"))),
    "dso": ("DSO", lambda v: safe_div(v.g("AR"), v.g("REVENUE")) * 365),
    "dsi": ("DSI", lambda v: safe_div(v.g("INVENTORY"), v.g("COGS")) * 365),
    "dpo": ("DPO", lambda v: safe_div(v.g("AP"), v.g("COGS")) * 365),
    "asset_turnover": ("Asset turnover", lambda v: safe_div(v.g("REVENUE"), v.avg("TOTAL_ASSETS"))),
    "accruals_ta": ("(NI - CFO) / assets", lambda v: safe_div(v.g("NET_INCOME") - v.g("CFO"), v.avg("TOTAL_ASSETS"))),
    "cfo_ni": ("CFO / NI", lambda v: safe_div(v.g("CFO"), v.g("NET_INCOME")) if v.g("NET_INCOME") > 0 else float("nan")),
    "capex_da": ("Capex / D&A", lambda v: safe_div(v.g("CAPEX"), v.g("DA"))),
    "dep_rate": ("Depreciation / gross PP&E", lambda v: safe_div(v.g("DA"), v.g("PPE_GROSS"))),
    "soft_assets": ("Soft assets / assets", lambda v: safe_div(v.g("TOTAL_ASSETS") - v.g("PPE_NET") - v.g("CASH"), v.g("TOTAL_ASSETS"))),
    "goodwill_ta": ("Goodwill / assets", lambda v: safe_div(v.z("GOODWILL"), v.g("TOTAL_ASSETS"))),
    "leverage": ("Liabilities / assets", lambda v: safe_div(v.total_liabilities(), v.g("TOTAL_ASSETS"))),
    "etr": ("Effective tax rate", lambda v: safe_div(v.g("INCOME_TAX"), v.g("PRETAX_INCOME")) if v.g("PRETAX_INCOME") > 0 else float("nan")),
    "sbc_ratio": ("SBC / revenue", lambda v: safe_div(v.g("SBC"), v.g("REVENUE"))),
    "rev_growth": ("Revenue growth", lambda v: safe_div(v.g("REVENUE"), v.g("REVENUE", 1)) - 1),
    "ar_growth_less_rev": ("AR growth - revenue growth", lambda v: (safe_div(v.g("AR"), v.g("AR", 1)) - 1) - (safe_div(v.g("REVENUE"), v.g("REVENUE", 1)) - 1)),
    "inv_growth_less_cogs": ("Inventory growth - COGS growth", lambda v: (safe_div(v.g("INVENTORY"), v.g("INVENTORY", 1)) - 1) - (safe_div(v.g("COGS"), v.g("COGS", 1)) - 1)),
    "other_ca_rev": ("Other current assets / revenue", lambda v: safe_div(v.g("OTHER_CURRENT_ASSETS"), v.g("REVENUE"))),
    "deferred_rev_rev": ("Deferred revenue / revenue", lambda v: safe_div(v.g("DEFERRED_REV_CURRENT"), v.g("REVENUE"))),
    "interest_rate": ("Interest / debt", lambda v: safe_div(v.g("INTEREST_EXPENSE"), v.z("LONG_TERM_DEBT") + v.z("CURRENT_DEBT"))),
}


def ratio_vector(P, p) -> dict[str, float]:
    v = V(P, p)
    out = {}
    for k, (_, fn) in RATIOS.items():
        try:
            out[k] = float(fn(v))
        except (ZeroDivisionError, TypeError, ValueError):
            out[k] = float("nan")
    return out


# ---------------------------------------------------------------------------- D.01
@detector("D.01", "Seasonal time-series residuals & change points", F, Area.EARNINGS_QUALITY)
def d01(ctx: RunContext):
    """STL residual of the latest quarter per key ratio, plus PELT change points in the last 4 quarters."""
    import ruptures as rpt
    from statsmodels.tsa.seasonal import STL

    series = {
        "gross margin": (ctx.qhist("REVENUE", 24), ctx.qhist("COGS", 24), lambda r, c: (r - c) / r),
        "DSO": (ctx.qhist("AR", 24), ctx.qhist("REVENUE", 24), lambda a, r: a / r * 91),
        "SG&A ratio": (ctx.qhist("SGA", 24), ctx.qhist("REVENUE", 24), lambda s, r: s / r),
        "DSI": (ctx.qhist("INVENTORY", 24), ctx.qhist("COGS", 24), lambda i, c: i / c * 91),
    }
    out_bits, worst, cps = [], 0.0, []
    for name, (a, b, f) in series.items():
        s = f(a, b.reindex(a.index)).replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) < 12:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = STL(s.to_numpy(), period=4, robust=True).fit()
        resid = pd.Series(res.resid, index=s.index)
        z = robust_z(float(resid.iloc[-1]), resid.iloc[:-1].tolist())
        if np.isfinite(z) and abs(z) >= 2.5:
            out_bits.append(f"{name} latest-quarter residual z={z:+.1f}")
            worst = max(worst, severity_from_z(abs(z), 2.5, 5.0))
        try:
            algo = rpt.Pelt(model="rbf", min_size=3).fit(s.to_numpy().reshape(-1, 1))
            bkps = algo.predict(pen=3 * np.log(len(s)))
        except Exception:  # noqa: BLE001 - ruptures raises bare errors on degenerate series
            bkps = []
        recent = [bp for bp in bkps[:-1] if bp >= len(s) - 4]
        if recent:
            when = s.index[recent[-1]]
            before, after = s.iloc[: recent[-1]].tail(6).mean(), s.iloc[recent[-1]:].mean()
            cps.append(f"{name} shifted {fmt_num(before, 3)} -> {fmt_num(after, 3)} at {ctx.Q.label(when)}")
            worst = max(worst, 0.45)
    if not out_bits and not cps and all(len(x[0].dropna()) < 12 for x in series.values()):
        raise SkipDetector("fewer than 12 quarters of history")
    near_events = ctx.events_of("AUDITOR_CHANGE", "OFFICER_CHANGE", "ACQUISITION", within_days=400)
    txt = "; ".join(out_bits + cps) if (out_bits or cps) else "No unusual seasonal residuals or recent level shifts."
    if near_events and (out_bits or cps):
        txt += f" Co-dated events: {', '.join(e.type.lower() for e in near_events[:3])}."
    return [mk("D.01", ctx, raw_value=len(out_bits) + len(cps), severity=worst, unit="findings", rationale=txt,
               evidence=[ctx.ev_event(e) for e in near_events[:2]])]


# ---------------------------------------------------------------------------- D.02
@detector("D.02", "Peer-relative ratio outliers", F, Area.EARNINGS_QUALITY, needs_peers=True)
def d02(ctx: RunContext):
    """Robust z of ~24 ratios vs the peer set; reports the most extreme ones."""
    if not ctx.peers or len(ctx.peers.loaded) < 5:
        raise SkipDetector("needs at least 5 peers")
    mine = ratio_vector(ctx.A, ctx.focus)
    peers = [ratio_vector(p.annual, ctx.peers.aligned_period(p, ctx.focus)) for p in ctx.peers.loaded
             if ctx.peers.aligned_period(p, ctx.focus) is not None]
    rows = []
    for k, val in mine.items():
        ref = [pr[k] for pr in peers if np.isfinite(pr.get(k, np.nan))]
        if len(ref) < 5 or not np.isfinite(val):
            continue
        z = robust_z(val, ref)
        if np.isfinite(z):
            rows.append((abs(z), z, k, val, float(np.median(ref))))
    if not rows:
        raise SkipDetector("no comparable ratios")
    rows.sort(reverse=True)
    extreme = [r for r in rows if r[0] >= 3.5]
    sev = float(np.clip((len(extreme) - 1) / 4, 0, 1)) * 0.7
    top = ", ".join(f"{RATIOS[k][0]} {fmt_num(v, 3)} vs {fmt_num(m, 3)} (z {z:+.1f})" for _, z, k, v, m in rows[:4])
    return [mk("D.02", ctx, raw_value=len(extreme), severity=sev, unit=f"of {len(rows)} ratios beyond |z|>=3.5",
               rationale=f"Most extreme vs peers: {top}. Business-model differences explain many single outliers; clusters matter more.")]


# ---------------------------------------------------------------------------- D.03
def _ecod(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ECOD (Li et al. 2022): empirical-CDF tail probabilities, per-feature contributions."""
    n, d = X.shape
    contrib = np.zeros_like(X)
    for j in range(d):
        col = X[:, j]
        left = np.array([(col <= x).mean() for x in col])
        right = np.array([(col >= x).mean() for x in col])
        skew = sps.skew(col) if np.std(col) > 0 else 0.0
        lt, rt = -np.log(np.clip(left, 1e-9, 1)), -np.log(np.clip(right, 1e-9, 1))
        contrib[:, j] = rt if skew > 0 else lt if skew < 0 else np.maximum(lt, rt)
    return contrib.sum(axis=1), contrib


@detector("D.03", "Multivariate outlier ensemble", F, Area.EARNINGS_QUALITY, needs_peers=True)
def d03(ctx: RunContext):
    """Robust Mahalanobis (MCD), Isolation Forest and ECOD over peer-years + own history; rank-averaged."""
    from sklearn.covariance import MinCovDet
    from sklearn.ensemble import IsolationForest

    if not ctx.peers or len(ctx.peers.loaded) < 5:
        raise SkipDetector("needs at least 5 peers")
    keys = list(RATIOS)
    pool, labels = [], []
    for p in ctx.peers.loaded:
        for per in list(p.annual.values.index)[-6:]:
            pool.append(ratio_vector(p.annual, per))
            labels.append(f"peer:{p.cik}")
    for per in ctx.periods[-6:-1]:
        pool.append(ratio_vector(ctx.A, per))
        labels.append("self-history")
    pool.append(ratio_vector(ctx.A, ctx.focus))
    labels.append("target")
    df = pd.DataFrame(pool)[keys]
    keep = [c for c in keys if df[c].notna().mean() >= 0.7 and np.isfinite(df[c].iloc[-1])]
    if len(keep) < 6 or len(df) < 30:
        raise SkipDetector("not enough complete peer-years")
    X = df[keep].copy()
    X = X.fillna(X.median())
    lo, hi = X.quantile(0.02), X.quantile(0.98)
    X = X.clip(lo, hi, axis=1)
    Z = ((X - X.median()) / (X.quantile(0.75) - X.quantile(0.25)).replace(0, 1)).to_numpy()
    scores = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            mcd = MinCovDet(random_state=0, support_fraction=0.8).fit(Z)
            scores["mahalanobis"] = mcd.mahalanobis(Z)
        except ValueError:
            pass
        iso = IsolationForest(n_estimators=300, random_state=0).fit(Z)
        scores["isolation_forest"] = -iso.score_samples(Z)
    ecod_total, ecod_contrib = _ecod(Z)
    scores["ecod"] = ecod_total
    ranks = np.mean([sps.rankdata(s) / len(s) for s in scores.values()], axis=0)
    pct = float(ranks[-1] * 100)
    top_feats = np.argsort(-ecod_contrib[-1])[:3]
    drivers = ", ".join(f"{RATIOS[keep[i]][0]} ({fmt_num(df[keep[i]].iloc[-1], 3)})" for i in top_feats)
    sev = float(np.clip((pct - 85) / 14, 0, 1)) * 0.75
    return [mk("D.03", ctx, raw_value=pct, expected=50.0, severity=sev, unit="ensemble percentile",
               percentile_peer=pct,
               rationale=(f"Outlier ensemble ({', '.join(scores)}) ranks the company at the {pct:.0f}th percentile of "
                          f"{len(df)} peer-years and own history on {len(keep)} ratios. Largest contributions: {drivers}."))]


# ---------------------------------------------------------------------------- D.04
STRUCTURAL = {
    # target balance, its driver, label. The model predicts log(target / driver) from scale-free features,
    # because tree models cannot extrapolate size to firms larger than any in the training data.
    "AR": ("REVENUE", "Receivables"),
    "INVENTORY": ("COGS", "Inventory"),
    "SGA": ("REVENUE", "SG&A"),
}


def _structural_features(gp_margin, turnover, log_assets) -> np.ndarray:
    return np.column_stack([gp_margin, turnover, log_assets])


@detector("D.04", "Structural expectation models (conformal)", F, Area.EARNINGS_QUALITY)
def d04(ctx: RunContext):
    """Gradient-boosted models predict each balance's ratio to its driver from gross margin, asset turnover and size
    across the industry; a dollar gap outside a split-conformal 95% interval is anomalous."""
    import lightgbm as lgb

    if ctx.xsec.empty:
        raise SkipDetector("no cross-section")
    xs = ctx.xsec
    ind = xs[xs["cik"].isin(ctx.industry_ciks)]
    data = ind if ind["cik"].nunique() >= 40 else xs
    data = data[data["cik"] != ctx.company.cik]
    gp = data["GROSS_PROFIT"].fillna(data["REVENUE"] - data["COGS"])
    base = data.assign(_gm=gp / data["REVENUE"], _to=data["REVENUE"] / data["TOTAL_ASSETS"], _la=np.log(data["TOTAL_ASSETS"].clip(lower=1)))
    rev, ta = ctx.a("REVENUE"), ctx.a("TOTAL_ASSETS")
    gp_t = ctx.a("GROSS_PROFIT") if finite(ctx.a("GROSS_PROFIT")) else rev - ctx.a("COGS")
    if not finite(rev, ta, gp_t) or rev <= 0 or ta <= 0:
        raise SkipDetector("target revenue / assets unavailable")
    x_t = _structural_features([gp_t / rev], [rev / ta], [np.log(ta)])
    out = []
    for target, (driver, label) in STRUCTURAL.items():
        d = base.dropna(subset=[target, driver, "_gm", "_to"])
        d = d[(d[target] > 0) & (d[driver] > 0) & d["_gm"].between(-1, 1) & (d["_to"] > 0)]
        mine, drv = ctx.a(target), ctx.a(driver)
        if len(d) < 80 or not finite(mine, drv) or mine <= 0 or drv <= 0:
            continue
        X = _structural_features(d["_gm"], d["_to"], d["_la"])
        y = np.log(d[target].to_numpy(float) / d[driver].to_numpy(float))
        idx = np.random.default_rng(0).permutation(len(d))
        cut = int(len(d) * 0.75)
        tr, cal = idx[:cut], idx[cut:]
        model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=15, verbose=-1)
        model.fit(X[tr], y[tr])
        q = float(np.quantile(np.abs(y[cal] - model.predict(X[cal])), 0.95))
        pred = float(model.predict(x_t)[0])
        resid = float(np.log(mine / drv)) - pred
        exp_usd = float(np.exp(pred) * drv)
        gap_usd = mine - exp_usd
        z = resid / (q / 1.96) if q > 0 else float("nan")
        outside = abs(resid) > q
        sev = float(np.clip((abs(z) - 1.96) / 2.0, 0, 1)) if outside and resid > 0 else 0.0
        lo_usd = mine - float(np.exp(pred + q) * drv)
        impact = (ImpactEstimate(max(lo_usd, 0.0), max(gap_usd, 0.0) * 0.5, max(gap_usd, 0.0), "model_residual",
                                 f"actual {label.lower()} - model expectation (low end = excess over the 95% upper bound)")
                  if outside and resid > 0 else None)
        out.append(mk("D.04", ctx, raw_value=mine, expected=exp_usd, severity=sev * (0.8 if target == "SGA" else 1.0),
                      unit="USD", impact=impact, tags=[f"target:{target}"],
                      area=Area.RECEIVABLES if target == "AR" else Area.INVENTORY if target == "INVENTORY" else Area.MARGINS,
                      rationale=(f"{label}: {fmt_money(mine)} vs {fmt_money(exp_usd)} expected ({label.lower()} / {driver.lower()} "
                                 f"{fmt_pct(mine / drv)} vs model {fmt_pct(np.exp(pred))}) given gross margin, asset turnover and size "
                                 f"across {len(d)} firm-years; {'outside' if outside else 'inside'} the 95% interval "
                                 f"(x{np.exp(-q):.2f}-x{np.exp(q):.2f})."),
                      evidence=ctx.evs([target, driver], ks=(0,))))
    if not out:
        raise SkipDetector("cross-section too small for structural models")
    return out



# ---------------------------------------------------------------------------- D.05
BENFORD1 = np.log10(1 + 1 / np.arange(1, 10))
BENFORD2 = np.log10(1 + 1 / np.arange(10, 100))


@detector("D.05", "Benford / digit tests", F, Area.INTEGRITY)
def d05(ctx: RunContext):
    """First- and first-two-digit MAD vs Benford for all monetary facts in the focus 10-K (Nigrini thresholds)."""
    rec = ctx.A.prov.get(("TOTAL_ASSETS", ctx.focus))
    if rec is None:
        raise SkipDetector("focus 10-K not identified")
    d = ctx.store.df
    vals = d[(d["accn"] == rec.accn) & (d["unit"] == "USD")].drop_duplicates(["concept", "start", "end"])["val"].abs()
    vals = vals[vals >= 10]
    if len(vals) < 100:
        raise SkipDetector(f"only {len(vals)} monetary facts; Benford needs 100+")
    s = vals.map(lambda x: f"{x:.15e}".replace(".", "")[:2])
    d1 = s.str[0].astype(int)
    d12 = s.astype(int)
    f1 = np.array([(d1 == k).mean() for k in range(1, 10)])
    f2 = np.array([(d12 == k).mean() for k in range(10, 100)])
    mad1 = float(np.mean(np.abs(f1 - BENFORD1)))
    mad2 = float(np.mean(np.abs(f2 - BENFORD2)))
    sev = float(np.clip((mad1 - 0.012) / 0.012, 0, 1)) * 0.4  # deliberately low weight
    verdict = "close conformity" if mad1 < 0.006 else "acceptable conformity" if mad1 < 0.012 else "marginal conformity" if mad1 < 0.015 else "nonconformity"
    return [mk("D.05", ctx, raw_value=mad1, expected=0.006, severity=sev, unit="MAD",
               rationale=(f"First-digit MAD {mad1:.4f} ({verdict} by Nigrini's thresholds) and first-two-digit MAD {mad2:.4f} "
                          f"over {len(vals)} monetary facts. Small samples and subtotals legitimately deviate; low weight."))]


# ---------------------------------------------------------------------------- D.06 / D.07
@detector("D.06", "Supervised misstatement model", F, Area.EARNINGS_QUALITY)
def d06(ctx: RunContext):
    """LightGBM trained offline on restatement / AAER labels (anomaly-report train); SHAP-style contributions."""
    bundle = load_model("misstatement")
    if bundle is None:
        raise SkipDetector("no trained model artifact; run `anomaly-report train` to build one")
    model, meta = bundle
    feats = ratio_vector(ctx.A, ctx.focus)
    X = pd.DataFrame([feats])[meta["features"]].replace([np.inf, -np.inf], np.nan).clip(-1e3, 1e3)
    prob = float(model.predict(X.to_numpy(float))[0])
    contrib = model.predict(X.to_numpy(float), pred_contrib=True)[0][:-1]
    top = np.argsort(-contrib)[:3]
    base = meta.get("base_rate", 0.03)
    sev = float(np.clip((prob / base - 1.5) / 3.5, 0, 1))
    return [mk("D.06", ctx, raw_value=prob, expected=base, severity=sev, unit="probability",
               rationale=(f"Model {meta.get('version')} (trained {meta.get('trained_on')}) scores {fmt_pct(prob)} against a training-sample "
                          f"rate of {fmt_pct(base)} ({prob / base:.1f}x). The sample is case-control, so this is a relative ranking, "
                          "not a population probability. Top contributions: "
                          + ", ".join(f"{RATIOS.get(meta['features'][i], (meta['features'][i],))[0]} {contrib[i]:+.2f}" for i in top) + "."))]


@detector("D.07", "Restatement-delta model", F, Area.EARNINGS_QUALITY)
def d07(ctx: RunContext):
    """Predicts which line item is most likely to be restated; needs a trained artifact."""
    if load_model("restatement_delta") is None:
        raise SkipDetector("no trained restatement-delta model; run `anomaly-report train --restatement-delta`")
    raise SkipDetector("restatement-delta model loaded but per-item inference not yet wired")


# ---------------------------------------------------------------------------- D.08
@detector("D.08", "Disclosure structure change", F, Area.DISCLOSURE)
def d08(ctx: RunContext):
    """Change in number of tagged standard concepts between the last two 10-Ks; canonical items that disappeared."""
    a1 = ctx.A.prov.get(("TOTAL_ASSETS", ctx.focus))
    p1 = ctx.period_at(1)
    a0 = ctx.A.prov.get(("TOTAL_ASSETS", p1)) if p1 is not None else None
    if a1 is None or a0 is None:
        raise SkipDetector("two 10-Ks needed")
    d = ctx.store.df
    n1 = d[d["accn"] == a1.accn]["concept"].nunique()
    n0 = d[d["accn"] == a0.accn]["concept"].nunique()
    chg = safe_div(n1 - n0, n0)
    prev_items = {i for (i, p) in ctx.A.prov if p == p1}
    cur_items = {i for (i, p) in ctx.A.prov if p == ctx.focus}
    gone = sorted(prev_items - cur_items)
    sev = float(np.clip((abs(chg) - 0.2) / 0.4, 0, 1)) * 0.5 + min(0.3, 0.05 * len(gone))
    return [mk("D.08", ctx, raw_value=chg, severity=sev, unit="change in tagged concepts",
               rationale=(f"Standard concepts tagged: {n1} vs {n0} last year ({fmt_pct(chg)})."
                          + (f" Line items no longer reported: {', '.join(gone[:8])}." if gone else "")))]


@detector("D.09", "Ego-network outliers", F, Area.GOVERNANCE)
def d09(ctx: RunContext):
    """Graph features of the company's neighbourhood vs peers' (needs peer graphs from market-wide mode)."""
    raise SkipDetector("peer knowledge graphs are only built in market-wide mode (Phase 4)")

