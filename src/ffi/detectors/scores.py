"""Forensic score formulas as pure functions of (Panel, period) so they apply to target and peers alike."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd

from ffi.context import prev_period, pv
from ffi.detectors.stats import safe_div
from ffi.xbrl.canonical import basis
from ffi.xbrl.facts import Panel

NAN = float("nan")


class V:
    """Value accessor: V(panel, period).g(item, back=0).

    With `track` (a dict), every value read is recorded as (item, back) -> its basis key, or None when the value is
    missing, so a comparison can tell whether two years were measured on the same basis (see `same_basis`)."""

    def __init__(self, panel: Panel, period: pd.Timestamp, track: dict | None = None) -> None:
        self.P, self.p, self.track = panel, period, track

    def period(self, back: int = 0) -> pd.Timestamp | None:
        p: pd.Timestamp | None = self.p
        for _ in range(back):
            p = prev_period(self.P, p) if p is not None else None
        return p

    def concept(self, item: str, back: int = 0) -> str | None:
        """The XBRL concept a value was read from (None when it is missing or derived)."""
        p = self.period(back)
        rec = self.P.prov.get((item, p)) if p is not None else None
        return rec.concept if rec is not None else None

    def g(self, item: str, back: int = 0) -> float:
        v = pv(self.P, item, self.p, back)
        if self.track is not None:
            self.track[(item, back)] = basis(item, self.concept(item, back)) if np.isfinite(v) else None
        return v

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

    def gross_margin(self, back: int = 0, years: tuple[int, ...] | None = None) -> float:
        """Gross profit / revenue. When years are compared, every one of `years` uses the same measure: the tagged gross
        profit only if it is reported in all of them, else revenue less cost of sales in all of them. (Costco tagged
        gross profit on net sales, which leaves out membership fees, in one year only: mixing it with revenue less cost
        of sales from total revenue moved the margin 2 points with no change in the business.)"""
        yrs = years or (back,)
        rev = self.g("REVENUE", back)
        if all(np.isfinite(self.g("GROSS_PROFIT", b)) for b in yrs):
            return safe_div(self.g("GROSS_PROFIT", back), rev)
        if all(np.isfinite(self.g("COGS", b)) for b in yrs):
            return safe_div(rev - self.g("COGS", back), rev)
        return NAN

    def debt(self, back: int = 0) -> float:
        """Total borrowings without counting a part twice: long-term debt tagged including its current maturities
        already holds what a current-maturities tag reports, and a current-debt tag that includes short-term borrowings
        already holds what a short-term-borrowings tag reports. Missing parts count as zero; NaN when none is tagged."""
        parts = debt_parts(self, back)
        return float(sum(parts.values())) if parts else NAN


def depreciable_ppe(v: V, backs: tuple[int, ...]) -> Callable[[int], float]:
    """Gross PP&E that carries depreciation, one measure for every year in `backs`: gross PP&E less construction in progress
    and land when those are tagged in all of the years and the tagged gross figure includes them (gross less accumulated
    depreciation reaches net PP&E; a gross tag that is a subtotal leaving them out, as at Alphabet, is not reduced again),
    else gross PP&E as tagged. Construction in progress is not depreciated until it is put to use, so a build-out (Oracle
    2026: $40B of $123B) lengthens the implied life of the whole asset base and cuts the depreciation rate without any
    change in policy."""
    def includes(b: int) -> bool:
        g, ad, net, cip = v.g("PPE_GROSS", b), v.g("ACCUM_DEP", b), v.g("PPE_NET", b), v.g("PPE_CIP", b)
        return bool(np.isfinite(g) and np.isfinite(ad) and np.isfinite(net) and np.isfinite(cip) and g - ad >= 0.98 * net)

    if backs and all(includes(b) for b in backs):
        land = all(np.isfinite(v.g("PPE_LAND", b)) for b in backs)
        return lambda b: v.g("PPE_GROSS", b) - v.g("PPE_CIP", b) - (v.g("PPE_LAND", b) if land else 0.0)
    return lambda b: v.g("PPE_GROSS", b)


def debt_parts(v: V, back: int = 0) -> dict[str, float]:
    """The borrowings that make up total debt, by canonical item, with overlapping tags left out (see V.debt)."""
    lt, cur, st = v.g("LONG_TERM_DEBT", back), v.g("CURRENT_DEBT", back), v.g("ST_BORROWINGS", back)
    lt_incl_current = bool(basis("LONG_TERM_DEBT", v.concept("LONG_TERM_DEBT", back))) and np.isfinite(lt)
    cur_basis = basis("CURRENT_DEBT", v.concept("CURRENT_DEBT", back))
    out: dict[str, float] = {}
    if np.isfinite(lt):
        out["LONG_TERM_DEBT"] = lt
    keep_st = np.isfinite(st)
    if np.isfinite(cur):
        if cur_basis == "":  # current maturities of long-term debt
            if not lt_incl_current:
                out["CURRENT_DEBT"] = cur
        elif cur_basis == "including short-term borrowings":
            if not lt_incl_current:
                out["CURRENT_DEBT"] = cur
                keep_st = False  # already inside the current-debt total
            elif not keep_st:
                out["CURRENT_DEBT"] = cur  # the only record of short-term borrowings (its current maturities count twice)
        elif not keep_st:  # short-term borrowings only, and not tagged separately
            out["CURRENT_DEBT"] = cur
    if keep_st:
        out["ST_BORROWINGS"] = st
    return out


def noncurrent_ltd(v: V, back: int = 0) -> float:
    """Long-term debt excluding its current maturities (which current liabilities already hold)."""
    lt = v.g("LONG_TERM_DEBT", back)
    if not np.isfinite(lt) or not basis("LONG_TERM_DEBT", v.concept("LONG_TERM_DEBT", back)):
        return lt
    cur = v.g("CURRENT_DEBT", back)
    if np.isfinite(cur) and basis("CURRENT_DEBT", v.concept("CURRENT_DEBT", back)) == "":
        return lt - cur
    return lt  # tagged only including its current maturities, which are not reported apart


def same_basis(focus: dict, other: dict) -> bool:
    """Two tracked readings (see V) measured every item they share on the same basis."""
    return all(focus[k] == other[k] for k in focus.keys() & other.keys() if focus[k] is not None and other[k] is not None)


def dep_measure(v: V, backs: tuple[int, ...]) -> tuple[str, Callable[[int], float]]:
    """One depreciation measure for every year compared: depreciation when it is tagged in all of them, else D&A less
    amortization of intangibles when that amortization is tagged in all of them, else D&A. A series that mixes
    measures (depreciation in one year, D&A less amortization in another) moves with the tagging, not the business."""
    if all(np.isfinite(v.g("DEPRECIATION", b)) for b in backs):
        return "depreciation", lambda b: v.g("DEPRECIATION", b)
    if all(np.isfinite(v.g("AMORTIZATION_INTANGIBLES", b)) for b in backs):
        return ("depreciation and amortization less amortization of intangibles",
                lambda b: v.g("DA", b) - v.g("AMORTIZATION_INTANGIBLES", b))
    return "depreciation and amortization", lambda b: v.g("DA", b)


# ---------------------------------------------------------------------------- Beneish M (8-var)
BENEISH_W = {"DSRI": 0.920, "GMI": 0.528, "AQI": 0.404, "SGI": 0.892, "DEPI": 0.115, "SGAI": -0.172, "TATA": 4.679, "LVGI": -0.327}
BENEISH_C = -4.84


def beneish(P: Panel, p: pd.Timestamp) -> dict[str, float]:
    v = V(P, p)
    rev, rev0 = v.g("REVENUE"), v.g("REVENUE", 1)
    ar, ar0 = v.g("AR"), v.g("AR", 1)
    ta = v.g("TOTAL_ASSETS")

    def aq(b: int) -> float:
        # Short-term investments are already part of current assets: adding them again double-counts and, for a company
        # holding large marketable securities (Alphabet: the sum reaches 98% of assets), sends the ratio towards zero and
        # the index to meaningless values (-2.1 and 4.0 instead of about 1.0).
        return 1 - safe_div(v.g("CURRENT_ASSETS", b) + v.g("PPE_NET", b), v.g("TOTAL_ASSETS", b))

    _, dep = dep_measure(v, (0, 1))  # the same measure in both years

    def deprate(b: int) -> float:
        d = dep(b)
        return safe_div(d, d + v.g("PPE_NET", b))

    def sga(b: int) -> float:
        return safe_div(v.g("SGA", b), v.g("REVENUE", b))

    ltd_years = [np.isfinite(v.g("LONG_TERM_DEBT", b)) for b in (0, 1)]

    def lev(b: int) -> float:
        if any(ltd_years) and not all(ltd_years):
            return NAN  # long-term debt reported in one year only: the index would measure the tagging
        ltd = noncurrent_ltd(v, b)
        return safe_div(v.g("CURRENT_LIABILITIES", b) + (ltd if np.isfinite(ltd) else 0.0), v.g("TOTAL_ASSETS", b))

    ico = v.g("INCOME_CONT_OPS")
    if not np.isfinite(ico):
        ico = v.g("NET_INCOME")
    comp = {
        "DSRI": safe_div(safe_div(ar, rev), safe_div(ar0, rev0)),
        "GMI": safe_div(v.gross_margin(1, years=(0, 1)), v.gross_margin(0, years=(0, 1))),
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
    total, ltd = v.debt(b), noncurrent_ltd(v, b)
    total, ltd = (total if np.isfinite(total) else 0.0), (ltd if np.isfinite(ltd) else 0.0)
    debt_cl = total - ltd  # borrowings inside current liabilities
    ta, tl = v.g("TOTAL_ASSETS", b), v.total_liabilities(b)
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
    # securities sold for cash (the cash-flow statement); shares the statement of equity shows as issued include employee
    # stock plans, which raise no capital
    raised = v.z("EQUITY_ISSUED") if basis("EQUITY_ISSUED", v.concept("EQUITY_ISSUED")) != "shares issued, from the statement of equity" else 0.0
    issue = 1.0 if (v.z("DEBT_ISSUED") > 0 or raised > 0) else 0.0
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
        debt = v.debt(b)
        op_liab = v.total_liabilities(b) - (debt if np.isfinite(debt) else 0.0)
        return op_assets - op_liab

    avg_noa = (noa(0) + noa(1)) / 2
    # organic investment only: buying marketable securities with surplus cash is part of investing cash flow but not an
    # accrual (those securities are excluded from net operating assets above), and counting it made cash-rich companies
    # look as if most of their profit were unbacked by cash. Cash paid for acquired businesses is left out too: it
    # buys assets, it is not an accounting estimate (Johnson & Johnson 2025: $17.5B of acquisitions took the ratio to 25%).
    # Without capex, or with net operating assets that are not positive, the ratio is not defined: investing cash flow
    # (Toyota's earlier years) or total assets in its place made a different measure that was compared with this one
    capex = v.g("CAPEX")
    if not np.isfinite(capex) or not np.isfinite(avg_noa) or avg_noa <= 0:
        return NAN
    return safe_div(v.g("NET_INCOME") - v.g("CFO") + capex, avg_noa)


def montier_c(P: Panel, p: pd.Timestamp) -> dict[str, float]:
    v = V(P, p)
    gap1 = v.g("NET_INCOME") - v.g("CFO")
    gap0 = v.g("NET_INCOME", 1) - v.g("CFO", 1)
    dso1, dso0 = safe_div(v.g("AR"), v.g("REVENUE")), safe_div(v.g("AR", 1), v.g("REVENUE", 1))
    dsi1, dsi0 = safe_div(v.g("INVENTORY"), v.g("COGS")), safe_div(v.g("INVENTORY", 1), v.g("COGS", 1))
    oca1, oca0 = safe_div(v.g("OTHER_CURRENT_ASSETS"), v.g("REVENUE")), safe_div(v.g("OTHER_CURRENT_ASSETS", 1), v.g("REVENUE", 1))
    _, dep = dep_measure(v, (0, 1))
    base = depreciable_ppe(v, (0, 1))
    dr1, dr0 = safe_div(dep(0), base(0)), safe_div(dep(1), base(1))
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
    l1, l0 = noncurrent_ltd(v, 0), noncurrent_ltd(v, 1)
    if np.isfinite(l1) == np.isfinite(l0):  # no long-term debt tagged in either year counts as none in both
        l1, l0 = (l1, l0) if np.isfinite(l1) else (0.0, 0.0)
        lev1, lev0 = safe_div(l1, v.avg("TOTAL_ASSETS")), safe_div(l0, v.avg("TOTAL_ASSETS", 1))
    else:
        lev1 = lev0 = NAN  # tagged in one year only: the change in leverage is unknown
    cr1 = safe_div(v.g("CURRENT_ASSETS"), v.g("CURRENT_LIABILITIES"))
    cr0 = safe_div(v.g("CURRENT_ASSETS", 1), v.g("CURRENT_LIABILITIES", 1))
    sh1, sh0 = v.g("SHARES_DILUTED"), v.g("SHARES_DILUTED", 1)
    gm1, gm0 = v.gross_margin(0, years=(0, 1)), v.gross_margin(1, years=(0, 1))
    at1 = safe_div(v.g("REVENUE"), v.g("TOTAL_ASSETS", 1))
    at0 = safe_div(v.g("REVENUE", 1), v.g("TOTAL_ASSETS", 2))
    pairs = [(roa, 0.0), (cfo, 0.0), (roa, roa0), (cfo, roa), (lev0, lev1), (cr1, cr0), (sh0 * 1.005, sh1), (gm1, gm0), (at1, at0)]
    # a test whose inputs are missing is not computable, not failed: a missing tag must not read as weak fundamentals
    done = [a > b if i != 6 else a >= b for i, (a, b) in enumerate(pairs) if np.isfinite(a) and np.isfinite(b)]
    return {"F": float(sum(done)), "available": float(len(done))}


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
