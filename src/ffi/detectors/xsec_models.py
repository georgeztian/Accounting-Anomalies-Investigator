"""Industry-year cross-sectional regressions on frames data (Kothari/Jones accruals, Roychowdhury REM)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ffi.peers.crosssection import calendar_year_of

MIN_FIRMS = 15


@dataclass
class XsecResult:
    residual: float  # target's residual (scaled by lagged assets)
    peers: np.ndarray  # industry residuals
    coef: np.ndarray
    n: int
    year: int
    scope: str
    predicted: float = float("nan")
    lag_assets: float = float("nan")


def _winsor(a: np.ndarray, q: float = 0.01) -> np.ndarray:
    lo, hi = np.nanquantile(a, [q, 1 - q])
    return np.clip(a, lo, hi)


def _ols(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xc = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    return coef


def panel_rows(xsec: pd.DataFrame, year: int) -> pd.DataFrame:
    """Join year t with t-1 and t-2 for each firm."""
    cur = xsec[xsec["year"] == year].set_index("cik")
    p1 = xsec[xsec["year"] == year - 1].set_index("cik").add_suffix("_1")
    p2 = xsec[xsec["year"] == year - 2].set_index("cik")[["REVENUE"]].add_suffix("_2")
    return cur.join(p1, how="inner").join(p2, how="left")


def industry_scope(xsec: pd.DataFrame, industry: set[int], year: int) -> tuple[pd.DataFrame, str]:
    rows = panel_rows(xsec, year)
    ind = rows[rows.index.isin(industry)]
    if len(ind.dropna(subset=["NET_INCOME", "CFO", "TOTAL_ASSETS_1"])) >= MIN_FIRMS:
        return ind, "industry"
    return rows, "all filers (industry too small)"


def target_row(ctx, year_offset: int = 0) -> dict[str, float]:
    """Target's own as-first-reported values in the same layout as panel_rows."""
    def g(item: str, k: int) -> float:
        return ctx.a(item, k + year_offset)

    return {
        "REVENUE": g("REVENUE", 0), "REVENUE_1": g("REVENUE", 1), "REVENUE_2": g("REVENUE", 2),
        "NET_INCOME": g("NET_INCOME", 0), "CFO": g("CFO", 0), "COGS": g("COGS", 0), "SGA": g("SGA", 0),
        "TOTAL_ASSETS_1": g("TOTAL_ASSETS", 1), "AR": g("AR", 0), "AR_1": g("AR", 1),
        "PPE_NET": g("PPE_NET", 0), "INVENTORY": g("INVENTORY", 0), "INVENTORY_1": g("INVENTORY", 1),
    }


def _kothari_xy(r: pd.DataFrame | dict) -> tuple[np.ndarray, np.ndarray]:
    A = np.asarray(r["TOTAL_ASSETS_1"], dtype=float)
    tacc = (np.asarray(r["NET_INCOME"], float) - np.asarray(r["CFO"], float)) / A
    drev = (np.asarray(r["REVENUE"], float) - np.asarray(r["REVENUE_1"], float)) / A
    drec = (np.nan_to_num(np.asarray(r["AR"], float)) - np.nan_to_num(np.asarray(r["AR_1"], float))) / A
    ppe = np.nan_to_num(np.asarray(r["PPE_NET"], float)) / A
    roa = np.asarray(r["NET_INCOME"], float) / A
    X = np.column_stack([np.atleast_1d(1 / A), np.atleast_1d(drev - drec), np.atleast_1d(ppe), np.atleast_1d(roa)])
    return np.atleast_1d(tacc), X


def kothari(ctx, year_offset: int = 0) -> XsecResult | None:
    year = calendar_year_of(ctx.period_at(year_offset)) if ctx.period_at(year_offset) is not None else None
    if year is None or ctx.xsec.empty:
        return None
    rows, scope = industry_scope(ctx.xsec, ctx.industry_ciks, year)
    rows = rows.drop(index=ctx.company.cik, errors="ignore")
    rows = rows.dropna(subset=["NET_INCOME", "CFO", "TOTAL_ASSETS_1", "REVENUE", "REVENUE_1"])
    rows = rows[rows["TOTAL_ASSETS_1"] > 1e6]
    if len(rows) < MIN_FIRMS:
        return None
    y, X = _kothari_xy(rows)
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    y, X = _winsor(y[ok]), np.apply_along_axis(_winsor, 0, X[ok])
    coef = _ols(y, X)
    resid = y - np.column_stack([np.ones(len(X)), X]) @ coef
    t = target_row(ctx, year_offset)
    if not all(np.isfinite([t["NET_INCOME"], t["CFO"], t["TOTAL_ASSETS_1"], t["REVENUE"], t["REVENUE_1"]])):
        return None
    ty, tX = _kothari_xy({k: np.array([v]) for k, v in t.items()})
    pred = float((np.r_[1.0, tX[0]]) @ coef)
    return XsecResult(float(ty[0] - pred), resid, coef, len(y), year, scope, pred, t["TOTAL_ASSETS_1"])


def roychowdhury(ctx) -> dict[str, XsecResult] | None:
    """Abnormal CFO, production costs and discretionary expenses (Roychowdhury 2006)."""
    p = ctx.period_at(0)
    if p is None or ctx.xsec.empty:
        return None
    year = calendar_year_of(p)
    rows, scope = industry_scope(ctx.xsec, ctx.industry_ciks, year)
    rows = rows.drop(index=ctx.company.cik, errors="ignore")
    rows = rows.dropna(subset=["REVENUE", "REVENUE_1", "TOTAL_ASSETS_1", "CFO"])
    rows = rows[rows["TOTAL_ASSETS_1"] > 1e6]
    t = target_row(ctx)
    out: dict[str, XsecResult] = {}

    def fit(name: str, y: np.ndarray, X: np.ndarray, ty: float, tX: np.ndarray) -> None:
        ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
        if ok.sum() < MIN_FIRMS or not (np.isfinite(ty) and np.all(np.isfinite(tX))):
            return
        yy, XX = _winsor(y[ok]), np.apply_along_axis(_winsor, 0, X[ok])
        coef = _ols(yy, XX)
        resid = yy - np.column_stack([np.ones(len(XX)), XX]) @ coef
        pred = float(np.r_[1.0, tX] @ coef)
        out[name] = XsecResult(ty - pred, resid, coef, int(ok.sum()), year, scope, pred, t["TOTAL_ASSETS_1"])

    A = rows["TOTAL_ASSETS_1"].to_numpy(float)
    S, S1 = rows["REVENUE"].to_numpy(float), rows["REVENUE_1"].to_numpy(float)
    dS = S - S1
    tA = t["TOTAL_ASSETS_1"]
    if not np.isfinite(tA) or tA <= 0:
        return None
    tS, tS1, tS2 = t["REVENUE"], t["REVENUE_1"], t["REVENUE_2"]
    # abnormal CFO: CFO/A = a + b1(1/A) + b2(S/A) + b3(dS/A)
    fit("CFO", rows["CFO"].to_numpy(float) / A, np.column_stack([1 / A, S / A, dS / A]),
        t["CFO"] / tA, np.array([1 / tA, tS / tA, (tS - tS1) / tA]))
    # abnormal production: (COGS + dINV)/A = a + b1(1/A) + b2(S/A) + b3(dS/A) + b4(dS_{t-1}/A)
    if "COGS" in rows and "INVENTORY" in rows:
        prod = rows["COGS"].to_numpy(float) + (rows["INVENTORY"].fillna(0) - rows["INVENTORY_1"].fillna(0)).to_numpy(float)
        dS1 = S1 - rows["REVENUE_2"].to_numpy(float)
        tprod = t["COGS"] + (np.nan_to_num(t["INVENTORY"]) - np.nan_to_num(t["INVENTORY_1"]))
        fit("PROD", prod / A, np.column_stack([1 / A, S / A, dS / A, dS1 / A]),
            tprod / tA, np.array([1 / tA, tS / tA, (tS - tS1) / tA, (tS1 - tS2) / tA]))
    # abnormal discretionary expenses: SGA/A = a + b1(1/A) + b2(S_{t-1}/A)
    if "SGA" in rows:
        fit("DISX", rows["SGA"].to_numpy(float) / A, np.column_stack([1 / A, S1 / A]),
            t["SGA"] / tA, np.array([1 / tA, tS1 / tA]))
    for r in out.values():
        r.scope = scope
    return out or None
