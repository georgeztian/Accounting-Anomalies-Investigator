"""Forensic score formulas as pure functions of (Panel, period) so they apply to target and peers alike."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ffi.context import pv
from ffi.detectors.stats import safe_div
from ffi.xbrl.facts import Panel

NAN = float("nan")


class V:
    """Value accessor: V(panel, period).g(item, back=0)."""

    def __init__(self, panel: Panel, period: pd.Timestamp) -> None:
        self.P, self.p = panel, period

    def g(self, item: str, back: int = 0) -> float:
        return pv(self.P, item, self.p, back)

    def z(self, item: str, back: int = 0) -> float:
        """Value, treating a missing optional line as zero."""
        v = self.g(item, back)
        return v if np.isfinite(v) else 0.0

    def avg(self, item: str, back: int = 0) -> float:
        a, b = self.g(item, back), self.g(item, back + 1)
        return (a + b) / 2 if np.isfinite(a) and np.isfinite(b) else a

    def total_liabilities(self, back: int = 0) -> float:
        tl = self.g("TOTAL_LIABILITIES", back)
        if np.isfinite(tl):
            return tl
        ta = self.g("TOTAL_ASSETS", back)
        eq = self.g("EQUITY_INCL_NCI", back)
        if not np.isfinite(eq):
            eq = self.g("EQUITY", back) + self.z("NCI", back)
        return ta - eq

    def gross_margin(self, back: int = 0) -> float:
        rev = self.g("REVENUE", back)
        gp = self.g("GROSS_PROFIT", back)
        if not np.isfinite(gp):
            gp = rev - self.g("COGS", back)
        return safe_div(gp, rev)


# ---------------------------------------------------------------------------- Beneish M (8-var)
BENEISH_W = {"DSRI": 0.920, "GMI": 0.528, "AQI": 0.404, "SGI": 0.892, "DEPI": 0.115, "SGAI": -0.172, "TATA": 4.679, "LVGI": -0.327}
BENEISH_C = -4.84


def beneish(P: Panel, p: pd.Timestamp) -> dict[str, float]:
    v = V(P, p)
    rev, rev0 = v.g("REVENUE"), v.g("REVENUE", 1)
    ar, ar0 = v.g("AR"), v.g("AR", 1)
    ta = v.g("TOTAL_ASSETS")

    def aq(b: int) -> float:
        return 1 - safe_div(v.g("CURRENT_ASSETS", b) + v.g("PPE_NET", b) + v.z("ST_INVESTMENTS", b), v.g("TOTAL_ASSETS", b))

    def deprate(b: int) -> float:
        d = v.g("DEPRECIATION", b)
        if not np.isfinite(d):
            d = v.g("DA", b)
        return safe_div(d, d + v.g("PPE_NET", b))

    def sga(b: int) -> float:
        return safe_div(v.g("SGA", b), v.g("REVENUE", b))

    def lev(b: int) -> float:
        return safe_div(v.g("CURRENT_LIABILITIES", b) + v.z("LONG_TERM_DEBT", b), v.g("TOTAL_ASSETS", b))

    ico = v.g("INCOME_CONT_OPS")
    if not np.isfinite(ico):
        ico = v.g("NET_INCOME")
    comp = {
        "DSRI": safe_div(safe_div(ar, rev), safe_div(ar0, rev0)),
        "GMI": safe_div(v.gross_margin(1), v.gross_margin(0)),
        "AQI": safe_div(aq(0), aq(1)),
        "SGI": safe_div(rev, rev0),
        "DEPI": safe_div(deprate(1), deprate(0)),
        "SGAI": safe_div(sga(0), sga(1)),
        "TATA": safe_div(ico - v.g("CFO"), ta),
        "LVGI": safe_div(lev(0), lev(1)),
    }
    # Neutral imputation (1.0 for indices, 0 for TATA) when a component can't be computed, as is common practice
    filled = {k: (val if np.isfinite(val) else (0.0 if k == "TATA" else 1.0)) for k, val in comp.items()}
    missing = [k for k, val in comp.items() if not np.isfinite(val)]
    core_ok = np.isfinite(comp["SGI"]) and np.isfinite(comp["TATA"])
    m = BENEISH_C + sum(BENEISH_W[k] * filled[k] for k in BENEISH_W) if core_ok else NAN
    return {"M": m, **comp, "missing": len(missing), "_missing": missing}  # type: ignore[dict-item]


# ---------------------------------------------------------------------------- Dechow et al. (2011) F-score model 1
F_COEF = {"const": -7.893, "rsst_acc": 0.790, "ch_rec": 2.518, "ch_inv": 1.191, "soft_assets": 1.979,
          "ch_cs": 0.171, "ch_roa": -0.932, "issue": 1.029}
F_BASE_RATE = 0.0037


def _rsst_parts(v: V, b: int) -> tuple[float, float, float]:
    ca, cash, sti = v.g("CURRENT_ASSETS", b), v.g("CASH", b), v.z("ST_INVESTMENTS", b)
    cl = v.g("CURRENT_LIABILITIES", b)
    debt_cl = v.z("CURRENT_DEBT", b) + v.z("ST_BORROWINGS", b)
    ta, tl, ltd = v.g("TOTAL_ASSETS", b), v.total_liabilities(b), v.z("LONG_TERM_DEBT", b)
    inv_adv = v.z("EQUITY_METHOD_INVESTMENTS", b)
    wc = (ca - cash - sti) - (cl - debt_cl)
    nco = (ta - ca - inv_adv) - (tl - cl - ltd)
    fin = (sti + inv_adv) - (ltd + debt_cl)
    return wc, nco, fin


def dechow_f(P: Panel, p: pd.Timestamp) -> dict[str, float]:
    v = V(P, p)
    ata = v.avg("TOTAL_ASSETS")
    w1, n1, f1 = _rsst_parts(v, 0)
    w0, n0, f0 = _rsst_parts(v, 1)
    rsst = safe_div((w1 - w0) + (n1 - n0) + (f1 - f0), ata)
    ch_rec = safe_div(v.g("AR") - v.g("AR", 1), ata)
    ch_inv = safe_div(v.z("INVENTORY") - v.z("INVENTORY", 1), ata)
    soft = safe_div(v.g("TOTAL_ASSETS") - v.g("PPE_NET") - v.g("CASH"), v.g("TOTAL_ASSETS"))
    cs1 = v.g("REVENUE") - (v.g("AR") - v.g("AR", 1))
    cs0 = v.g("REVENUE", 1) - (v.g("AR", 1) - v.g("AR", 2))
    ch_cs = safe_div(cs1, cs0) - 1
    roa1 = safe_div(v.g("NET_INCOME"), ata)
    roa0 = safe_div(v.g("NET_INCOME", 1), v.avg("TOTAL_ASSETS", 1))
    ch_roa = roa1 - roa0
    issue = 1.0 if (v.z("DEBT_ISSUED") > 0 or v.z("EQUITY_ISSUED") > 0) else 0.0
    comp = {"rsst_acc": rsst, "ch_rec": ch_rec, "ch_inv": ch_inv, "soft_assets": soft, "ch_cs": ch_cs,
            "ch_roa": ch_roa, "issue": issue}
    if not all(np.isfinite(comp[k]) for k in ("rsst_acc", "ch_rec", "soft_assets", "ch_roa")):
        return {"F": NAN, "P": NAN, **comp}
    ch_cs_f = comp["ch_cs"] if np.isfinite(comp["ch_cs"]) else 0.0
    # Winsorize like the paper (1/99%) with practical caps
    clip = {"rsst_acc": 1.0, "ch_rec": 0.5, "ch_inv": 0.5, "ch_cs": 3.0, "ch_roa": 1.0}
    x = F_COEF["const"] + sum(
        F_COEF[k] * float(np.clip(comp[k] if k != "ch_cs" else ch_cs_f, -clip.get(k, 9), clip.get(k, 9)))
        for k in ("rsst_acc", "ch_rec", "ch_inv", "soft_assets", "ch_cs", "ch_roa", "issue")
    )
    prob = 1 / (1 + math.exp(-x))
    return {"F": prob / F_BASE_RATE, "P": prob, **comp}


# ---------------------------------------------------------------------------- others
def sloan_accruals(P: Panel, p: pd.Timestamp) -> float:
    v = V(P, p)

    def noa(b: int) -> float:
        op_assets = v.g("TOTAL_ASSETS", b) - v.g("CASH", b) - v.z("ST_INVESTMENTS", b)
        op_liab = v.total_liabilities(b) - v.z("CURRENT_DEBT", b) - v.z("ST_BORROWINGS", b) - v.z("LONG_TERM_DEBT", b)
        return op_assets - op_liab

    avg_noa = (noa(0) + noa(1)) / 2
    if not np.isfinite(avg_noa) or avg_noa <= 0:
        avg_noa = v.avg("TOTAL_ASSETS")
    return safe_div(v.g("NET_INCOME") - v.g("CFO") - v.g("CFI"), avg_noa)


def montier_c(P: Panel, p: pd.Timestamp) -> dict[str, float]:
    v = V(P, p)
    gap1 = v.g("NET_INCOME") - v.g("CFO")
    gap0 = v.g("NET_INCOME", 1) - v.g("CFO", 1)
    dso1, dso0 = safe_div(v.g("AR"), v.g("REVENUE")), safe_div(v.g("AR", 1), v.g("REVENUE", 1))
    dsi1, dsi0 = safe_div(v.g("INVENTORY"), v.g("COGS")), safe_div(v.g("INVENTORY", 1), v.g("COGS", 1))
    oca1, oca0 = safe_div(v.g("OTHER_CURRENT_ASSETS"), v.g("REVENUE")), safe_div(v.g("OTHER_CURRENT_ASSETS", 1), v.g("REVENUE", 1))
    dep = "DEPRECIATION" if np.isfinite(v.g("DEPRECIATION")) else "DA"
    dr1, dr0 = safe_div(v.g(dep), v.g("PPE_GROSS")), safe_div(v.g(dep, 1), v.g("PPE_GROSS", 1))
    ag = safe_div(v.g("TOTAL_ASSETS"), v.g("TOTAL_ASSETS", 1)) - 1
    tests = {
        "ni_cfo_gap_widening": gap1 > gap0 if np.isfinite(gap1) and np.isfinite(gap0) else None,
        "dso_rising": dso1 > dso0 if np.isfinite(dso1) and np.isfinite(dso0) else None,
        "dsi_rising": dsi1 > dsi0 if np.isfinite(dsi1) and np.isfinite(dsi0) else None,
        "other_ca_rising": oca1 > oca0 if np.isfinite(oca1) and np.isfinite(oca0) else None,
        "dep_rate_falling": dr1 < dr0 if np.isfinite(dr1) and np.isfinite(dr0) else None,
        "asset_growth_gt_10pct": ag > 0.10 if np.isfinite(ag) else None,
    }
    score = sum(1 for t in tests.values() if t)
    avail = sum(1 for t in tests.values() if t is not None)
    return {"C": float(score), "available": float(avail), **{k: (float(t) if t is not None else NAN) for k, t in tests.items()}}


def piotroski_f(P: Panel, p: pd.Timestamp) -> dict[str, float]:
    v = V(P, p)
    roa = safe_div(v.g("NET_INCOME"), v.g("TOTAL_ASSETS", 1))
    roa0 = safe_div(v.g("NET_INCOME", 1), v.g("TOTAL_ASSETS", 2))
    cfo = safe_div(v.g("CFO"), v.g("TOTAL_ASSETS", 1))
    lev1, lev0 = safe_div(v.z("LONG_TERM_DEBT"), v.avg("TOTAL_ASSETS")), safe_div(v.z("LONG_TERM_DEBT", 1), v.avg("TOTAL_ASSETS", 1))
    cr1 = safe_div(v.g("CURRENT_ASSETS"), v.g("CURRENT_LIABILITIES"))
    cr0 = safe_div(v.g("CURRENT_ASSETS", 1), v.g("CURRENT_LIABILITIES", 1))
    sh1, sh0 = v.g("SHARES_DILUTED"), v.g("SHARES_DILUTED", 1)
    gm1, gm0 = v.gross_margin(0), v.gross_margin(1)
    at1 = safe_div(v.g("REVENUE"), v.g("TOTAL_ASSETS", 1))
    at0 = safe_div(v.g("REVENUE", 1), v.g("TOTAL_ASSETS", 2))
    tests = [roa > 0, cfo > 0, roa > roa0, cfo > roa, lev1 < lev0, cr1 > cr0, sh1 <= sh0 * 1.005, gm1 > gm0, at1 > at0]
    return {"F": float(sum(bool(t) for t in tests))}


def altman(P: Panel, p: pd.Timestamp) -> dict[str, float]:
    v = V(P, p)
    ta = v.g("TOTAL_ASSETS")
    wc = v.g("CURRENT_ASSETS") - v.g("CURRENT_LIABILITIES")
    re_ = v.g("RETAINED_EARNINGS")
    ebit = v.g("OPERATING_INCOME")
    if not np.isfinite(ebit):
        ebit = v.g("PRETAX_INCOME") + v.z("INTEREST_EXPENSE")
    tl = v.total_liabilities()
    eq = v.g("EQUITY")
    mve = v.g("PUBLIC_FLOAT")
    x1, x2, x3 = safe_div(wc, ta), safe_div(re_, ta), safe_div(ebit, ta)
    x5 = safe_div(v.g("REVENUE"), ta)
    z_pp = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * safe_div(eq, tl)
    z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * safe_div(mve, tl) + 1.0 * x5 if np.isfinite(mve) else NAN
    return {"Z": z, "Zpp": z_pp, "x1": x1, "x2": x2, "x3": x3, "x5": x5}
