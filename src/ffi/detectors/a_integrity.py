"""Family A: accounting integrity checks (BLUEPRINT §9, Family A). Deterministic; run first."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ffi.context import RunContext
from ffi.detectors.base import Area, EvidenceRef, ImpactEstimate, Scheme, SkipDetector, detector
from ffi.detectors.helpers import finite, mk, need
from ffi.detectors.stats import fmt_money, fmt_pct, safe_div
from ffi.xbrl.canonical import BY_ID

F = "A"


def _residual_severity(residual: float, base: float, tol: float, full: float) -> float:
    r = abs(safe_div(residual, base))
    if not np.isfinite(r):
        return 0.0
    return float(np.clip((r - tol) / (full - tol), 0, 1)) if r > tol else 0.0


@detector("A.01", "Balance-sheet identity", F, Area.INTEGRITY)
def a01(ctx: RunContext):
    """Assets = Liabilities + Equity (incl. NCI)."""
    ta, le = ctx.a("TOTAL_ASSETS"), ctx.a("LIABILITIES_AND_EQUITY")
    tl = ctx.a("TOTAL_LIABILITIES")
    eq = ctx.a("EQUITY_INCL_NCI")
    if not finite(eq):
        eq = ctx.a("EQUITY") + (ctx.a("NCI") if finite(ctx.a("NCI")) else 0.0)
    need(ta)
    checks = []
    if finite(le):
        checks.append(("Assets vs Liabilities & equity", ta - le, ["TOTAL_ASSETS", "LIABILITIES_AND_EQUITY"]))
    if finite(tl, eq):
        temp = ctx.a("TEMPORARY_EQUITY")  # redeemable interests sit between liabilities and equity
        temp = temp if finite(temp) else 0.0
        label = "Assets vs Liabilities + redeemable interests + Equity" if temp else "Assets vs Liabilities + Equity"
        checks.append((label, ta - (tl + temp + eq), ["TOTAL_ASSETS", "TOTAL_LIABILITIES", "TEMPORARY_EQUITY", "EQUITY"]))
    if not checks:
        raise SkipDetector("total liabilities and liabilities-and-equity not reported")
    worst = max(checks, key=lambda c: abs(c[1]))
    sev = _residual_severity(worst[1], ta, 0.005, 0.03)
    return [mk("A.01", ctx, raw_value=worst[1], expected=0.0, severity=sev, unit="USD",
               scheme=Scheme.REPORTING_INTEGRITY,
               rationale=f"{worst[0]}: residual {fmt_money(worst[1])} ({fmt_pct(safe_div(worst[1], ta), 2)} of assets).",
               evidence=ctx.evs(worst[2], ks=(0,)))]


@detector("A.02", "Subtotal integrity", F, Area.INTEGRITY)
def a02(ctx: RunContext):
    """Canonical subtotals add up: gross profit, net PP&E, current assets coverage."""
    rows = []
    rev, cogs, gp = ctx.a("REVENUE"), ctx.a("COGS"), ctx.a("GROSS_PROFIT")
    if finite(rev, cogs, gp):
        rows.append(("Revenue - COGS = Gross profit", rev - cogs - gp, rev, ["REVENUE", "COGS", "GROSS_PROFIT"]))
    g, ad, n = ctx.a("PPE_GROSS"), ctx.a("ACCUM_DEP"), ctx.a("PPE_NET")
    if finite(g, ad, n):
        rows.append(("Gross PP&E - accumulated depreciation = Net PP&E", g - ad - n, g, ["PPE_GROSS", "ACCUM_DEP", "PPE_NET"]))
    ca = ctx.a("CURRENT_ASSETS")
    parts = [ctx.a(i) for i in ("CASH", "ST_INVESTMENTS", "AR", "INVENTORY")]
    if finite(ca) and sum(1 for p in parts if finite(p)) >= 2:
        s = sum(p for p in parts if finite(p))
        excess = max(0.0, s - ca)
        rows.append(("Current-asset components <= total current assets", excess, ca, ["CURRENT_ASSETS", "CASH", "AR", "INVENTORY"]))
    if not rows:
        raise SkipDetector("no subtotal pairs reported")
    worst = max(rows, key=lambda r: abs(safe_div(r[1], r[2])) if finite(safe_div(r[1], r[2])) else 0)
    sev = _residual_severity(worst[1], worst[2], 0.005, 0.03)
    return [mk("A.02", ctx, raw_value=worst[1], expected=0.0, severity=sev, unit="USD", scheme=Scheme.REPORTING_INTEGRITY,
               rationale=f"{worst[0]}: residual {fmt_money(worst[1])}. Checked {len(rows)} subtotal relationship(s).",
               evidence=ctx.evs(worst[3], ks=(0,)))]


@detector("A.03", "Cash roll-forward", F, Area.INTEGRITY)
def a03(ctx: RunContext):
    """Begin cash + CFO + CFI + CFF + FX = end cash (incl. restricted)."""
    end, beg = ctx.a("CASH_RESTRICTED_TOTAL"), ctx.a("CASH_RESTRICTED_TOTAL", 1)
    cfo, cfi, cff = ctx.a("CFO"), ctx.a("CFI"), ctx.a("CFF")
    fx = ctx.a("FX_EFFECT_CASH")
    need(end, beg, cfo, cfi, cff)
    flows = cfo + cfi + cff + (fx if finite(fx) else 0.0)
    resid = (end - beg) - flows
    base = max(abs(end), abs(cfo), 1.0)
    sev = _residual_severity(resid, base, 0.01, 0.05)
    return [mk("A.03", ctx, raw_value=resid, expected=0.0, severity=sev, unit="USD", scheme=Scheme.REPORTING_INTEGRITY,
               rationale=(f"Change in cash {fmt_money(end - beg)} vs CFO+CFI+CFF+FX {fmt_money(flows)}; "
                          f"unexplained {fmt_money(resid)}."),
               evidence=ctx.evs(["CASH_RESTRICTED_TOTAL"], ks=(0, 1)) + ctx.evs(["CFO", "CFI", "CFF"], ks=(0,)))]


@detector("A.04", "Equity roll-forward", F, Area.INTEGRITY)
def a04(ctx: RunContext):
    """Begin equity + NI + OCI - dividends - buybacks + issuance + SBC = end equity."""
    e1, e0 = ctx.a("EQUITY"), ctx.a("EQUITY", 1)
    ni = ctx.a("NET_INCOME")
    need(e1, e0, ni)
    add = {i: ctx.a(i) for i in ("OCI", "DIVIDENDS_PAID", "BUYBACKS", "EQUITY_ISSUED", "SBC", "CONVERTIBLE_EQUITY_COMPONENT")}
    g = {k: (v if finite(v) else 0.0) for k, v in add.items()}
    expected_end = (e0 + ni + g["OCI"] - g["DIVIDENDS_PAID"] - g["BUYBACKS"] + g["EQUITY_ISSUED"] + g["SBC"]
                    + g["CONVERTIBLE_EQUITY_COMPONENT"])
    resid = e1 - expected_end
    base = max(abs(e1), abs(e0), 1.0)
    # employee tax withholding on vesting and other items routinely leave a few percent unexplained
    sev = _residual_severity(resid, base, 0.04, 0.15) * 0.7
    nci_note = ""
    nci, incl = ctx.a("NCI"), ctx.a("EQUITY_INCL_NCI")
    if not finite(nci) and finite(incl):
        nci = incl - e1  # many filers tag NCI only by class; total equity less the parent's share is the total
    if finite(nci, incl) and incl > 0 and nci / incl >= 0.2:
        # partnership (Up-C) structures: unit exchanges move equity between NCI and the parent without cash
        sev *= 0.5
        nci_note = (f" Noncontrolling interests are {fmt_pct(nci / incl)} of total equity; exchanges of their units "
                    "into common stock move equity to the parent without any cash flow.")
    return [mk("A.04", ctx, raw_value=resid, expected=0.0, severity=sev, unit="USD", scheme=Scheme.REPORTING_INTEGRITY,
               rationale=(f"Equity moved {fmt_money(e1 - e0)}; NI, OCI, dividends, buybacks, issuance and SBC explain "
                          f"{fmt_money(expected_end - e0)}. Unexplained {fmt_money(resid)} "
                          f"({fmt_pct(safe_div(resid, base))} of equity). Untagged equity offerings, convertible-note "
                          "components, warrant exercises and tax withholding on vesting are common causes." + nci_note),
               evidence=ctx.evs(["EQUITY"], ks=(0, 1)) + ctx.evs(["NET_INCOME", "BUYBACKS", "DIVIDENDS_PAID"], ks=(0,)))]


@detector("A.05", "Retained-earnings roll-forward", F, Area.INTEGRITY)
def a05(ctx: RunContext):
    """RE(t) = RE(t-1) + NI - dividends (repurchases charged to RE are recognised)."""
    r1, r0, ni = ctx.a("RETAINED_EARNINGS"), ctx.a("RETAINED_EARNINGS", 1), ctx.a("NET_INCOME")
    need(r1, r0, ni)
    div = ctx.a("DIVIDENDS_DECLARED")
    if not finite(div):
        div = ctx.a("DIVIDENDS_PAID")
    div = div if finite(div) else 0.0
    resid = r1 - (r0 + ni - div)
    bb = ctx.a("BUYBACKS")
    explained = ""
    if finite(bb) and bb > 0 and resid < 0 and abs(resid + bb) <= 0.35 * bb:
        explained = " The residual is consistent with share repurchases charged to retained earnings."
        resid_eff = resid + bb
    elif div > ni and resid > 0:
        # dividends above earnings are often charged to additional paid-in capital, so retained earnings fall less
        explained = (f" Dividends exceeded net income by {fmt_money(div - ni)}; companies often charge the excess to "
                     "additional paid-in capital, which explains up to that amount.")
        resid_eff = max(0.0, resid - (div - ni))
    else:
        resid_eff = resid
    base = max(abs(ni), abs(r1) * 0.1, 1.0)
    sev = _residual_severity(resid_eff, base, 0.05, 0.30)
    return [mk("A.05", ctx, raw_value=resid, expected=0.0, severity=sev, unit="USD", scheme=Scheme.REPORTING_INTEGRITY,
               rationale=f"Retained earnings moved {fmt_money(r1 - r0)} vs NI less dividends {fmt_money(ni - div)}; residual {fmt_money(resid)}.{explained}",
               evidence=ctx.evs(["RETAINED_EARNINGS"], ks=(0, 1)) + ctx.evs(["NET_INCOME", "DIVIDENDS_PAID"], ks=(0,)))]


@detector("A.06", "Quarterly-to-annual consistency", F, Area.INTEGRITY)
def a06(ctx: RunContext):
    """Q4 implied by FY less Q1-Q3 is plausible (sign and magnitude) for revenue and net income."""
    qp = ctx.q_periods()
    if not qp:
        raise SkipDetector("no quarterly data")
    q4 = [p for p in qp if (("REVENUE", p) in ctx.Q.derived)]
    q4 = [p for p in q4 if abs((p - ctx.focus).days) <= 10]
    if not q4:
        raise SkipDetector("Q4 for the focus year is not derivable")
    p = q4[0]
    worst_sev, msgs, evid = 0.0, [], []
    for item in ("REVENUE", "COGS"):
        s = ctx.Q.series(item)
        if p not in s.index or not np.isfinite(s[p]):
            continue
        prior3 = s[s.index < p].tail(3)
        if len(prior3) < 3:
            continue
        mean3 = float(prior3.mean())
        v = float(s[p])
        if v < 0 < mean3:
            worst_sev = max(worst_sev, 1.0)
            msgs.append(f"implied Q4 {BY_ID[item].label.lower()} is negative ({fmt_money(v)})")
        ratio = safe_div(v, mean3)
        if finite(ratio) and (ratio > 3 or ratio < 0.2):
            worst_sev = max(worst_sev, 0.6)
            msgs.append(f"implied Q4 {BY_ID[item].label.lower()} is {ratio:.1f}x the Q1-Q3 average")
        e = ctx.ev(item, panel="Q", period=p)
        if e:
            evid.append(e)
    return [mk("A.06", ctx, raw_value=None, severity=worst_sev, scheme=Scheme.REPORTING_INTEGRITY,
               rationale=("; ".join(msgs) + ".") if msgs else "Implied Q4 values are consistent with Q1-Q3.", evidence=evid)]


@detector("A.07", "Income-statement tie-outs", F, Area.INTEGRITY)
def a07(ctx: RunContext):
    """Pre-tax income - tax = income from continuing operations; EPS x shares = NI."""
    pt, tax = ctx.a("PRETAX_INCOME"), ctx.a("INCOME_TAX")
    ni, nci = ctx.a("NET_INCOME"), ctx.a("NCI_INCOME")
    # pre-tax income includes the noncontrolling holders' share, so any after-tax subtotal may tie: continuing
    # operations, net income including NCI, or the parent's net income plus the NCI share
    cands = [v for v in (ctx.a("INCOME_CONT_OPS"), ctx.a("NET_INCOME_INCL_NCI"),
                         ni + nci if finite(ni, nci) else float("nan"), ni) if finite(v)]
    need(pt, tax, *(cands[:1] or [float("nan")]))
    ico = min(cands, key=lambda v: abs((pt - tax) - v))
    resid = (pt - tax) - ico
    base = max(abs(pt), 1.0)
    # equity-method income and discontinued ops can legitimately sit between these lines
    sev = _residual_severity(resid, base, 0.03, 0.15)
    return [mk("A.07", ctx, raw_value=resid, expected=0.0, severity=sev, unit="USD", scheme=Scheme.REPORTING_INTEGRITY,
               rationale=f"Pre-tax income less tax {fmt_money(pt - tax)} vs income after tax {fmt_money(ico)}; gap {fmt_money(resid)}.",
               evidence=ctx.evs(["PRETAX_INCOME", "INCOME_TAX", "NET_INCOME"], ks=(0,)))]


@detector("A.08", "PP&E roll-forward", F, Area.INTEGRITY)
def a08(ctx: RunContext):
    """Growth in gross PP&E not explained by capex and acquisitions (possible capitalization outside capex)."""
    g1, g0, capex = ctx.a("PPE_GROSS"), ctx.a("PPE_GROSS", 1), ctx.a("CAPEX")
    need(g1, g0, capex)
    acq = ctx.a("ACQUISITIONS")
    acq = acq if finite(acq) else 0.0
    unexplained = (g1 - g0) - capex - 0.5 * acq
    sev = _residual_severity(max(unexplained, 0.0), g0, 0.03, 0.15)
    return [mk("A.08", ctx, raw_value=unexplained, expected=0.0, severity=sev, unit="USD",
               scheme=Scheme.SHIFT_EXPENSES_LATER, area=Area.CAPITALIZATION, direction="income_increasing",
               rationale=(f"Gross PP&E grew {fmt_money(g1 - g0)} with capex {fmt_money(capex)}"
                          f"{' and acquisitions ' + fmt_money(acq) if acq else ''}; "
                          f"{fmt_money(unexplained)} of additions are not explained by cash spending (disposals would make the gap larger)."),
               evidence=ctx.evs(["PPE_GROSS"], ks=(0, 1)) + ctx.evs(["CAPEX"], ks=(0,)))]


@detector("A.09", "Allowance roll-forward", F, Area.INTEGRITY)
def a09(ctx: RunContext):
    """Allowance(t) = Allowance(t-1) + provision - write-offs."""
    a1, a0, prov, wo = ctx.a("ALLOWANCE"), ctx.a("ALLOWANCE", 1), ctx.a("PROVISION_DOUBTFUL"), ctx.a("WRITE_OFFS")
    need(a1, a0, prov, wo, reason="allowance roll-forward components not tagged")
    resid = a1 - (a0 + prov - wo)
    sev = _residual_severity(resid, max(a1, a0, 1.0), 0.10, 0.50)
    return [mk("A.09", ctx, raw_value=resid, expected=0.0, severity=sev, unit="USD", area=Area.RECEIVABLES,
               scheme=Scheme.REPORTING_INTEGRITY,
               rationale=f"Allowance roll-forward residual {fmt_money(resid)} (recoveries and FX can explain small gaps).",
               evidence=ctx.evs(["ALLOWANCE"], ks=(0, 1)) + ctx.evs(["PROVISION_DOUBTFUL", "WRITE_OFFS"], ks=(0,)))]


@detector("A.10", "Goodwill roll-forward", F, Area.INTEGRITY)
def a10(ctx: RunContext):
    """Goodwill(t) = Goodwill(t-1) + acquired - impaired +/- FX +/- adjustments."""
    g1, g0 = ctx.a("GOODWILL"), ctx.a("GOODWILL", 1)
    need(g1, g0)
    comps = {i: ctx.a(i) for i in ("GOODWILL_ACQUIRED", "GOODWILL_IMPAIRMENT", "GOODWILL_FX", "GOODWILL_MEASUREMENT_ADJ", "GOODWILL_OTHER", "GOODWILL_DISPOSED")}
    if not any(finite(v) for v in comps.values()):
        if abs(g1 - g0) <= 0.01 * max(g0, 1):
            return [mk("A.10", ctx, raw_value=0.0, expected=0.0, severity=0.0, rationale="Goodwill unchanged.")]
        raise SkipDetector("goodwill roll-forward components not tagged")
    c = {k: (v if finite(v) else 0.0) for k, v in comps.items()}
    expected = g0 + c["GOODWILL_ACQUIRED"] - c["GOODWILL_IMPAIRMENT"] + c["GOODWILL_FX"] + c["GOODWILL_MEASUREMENT_ADJ"] + c["GOODWILL_OTHER"] - c["GOODWILL_DISPOSED"]
    resid = g1 - expected
    sev = _residual_severity(resid, max(g1, 1.0), 0.02, 0.10)
    note = ""
    if not finite(comps["GOODWILL_ACQUIRED"]) and not finite(comps["GOODWILL_DISPOSED"]):
        # only FX / other lines tagged: the roll-forward is incomplete rather than broken
        sev = min(sev, 0.3)
        note = (" Goodwill from acquisitions and disposals is not tagged (often reported only by segment), so the "
                "roll-forward is incomplete rather than inconsistent.")
    return [mk("A.10", ctx, raw_value=resid, expected=0.0, severity=sev, unit="USD", area=Area.MA,
               scheme=Scheme.REPORTING_INTEGRITY,
               rationale=f"Goodwill moved {fmt_money(g1 - g0)}; tagged roll-forward components explain {fmt_money(expected - g0)}; residual {fmt_money(resid)}.{note}",
               evidence=ctx.evs(["GOODWILL"], ks=(0, 1)))]


@detector("A.11", "Segment reconciliation", F, Area.INTEGRITY)
def a11(ctx: RunContext):
    """Sum of segments = consolidated. Needs dimensional XBRL (not in companyfacts)."""
    raise SkipDetector("segment facts are dimensional; the companyfacts API carries only non-dimensional facts")


@detector("A.12", "EPS consistency", F, Area.INTEGRITY)
def a12(ctx: RunContext):
    """Net income / diluted shares ~= diluted EPS; diluted <= basic when profitable."""
    ni, sh, eps, epsb = ctx.a("NET_INCOME"), ctx.a("SHARES_DILUTED"), ctx.a("EPS_DILUTED"), ctx.a("EPS_BASIC")
    need(ni, sh, eps)
    implied = safe_div(ni, sh)
    diff = implied - eps
    tol = max(0.02, abs(eps) * 0.02)  # preferred dividends / participating securities create small gaps
    sev = float(np.clip((abs(diff) - tol) / (abs(eps) * 0.15 + 0.05), 0, 1)) if abs(diff) > tol else 0.0
    msgs = [f"NI / diluted shares = {implied:.2f} vs reported diluted EPS {eps:.2f}"]
    factor = ctx.A.rescaled.get(("SHARES_DILUTED", ctx.focus))
    if factor:
        msgs.append(f"diluted shares are tagged in {'millions' if factor == 1e6 else 'thousands'} in the XBRL "
                    "(a tagging scale error, corrected here; the financial statements themselves are consistent)")
    if finite(epsb) and ni > 0 and eps > epsb + 0.005:
        sev = max(sev, 0.8)
        msgs.append(f"diluted EPS {eps:.2f} exceeds basic {epsb:.2f}")
    return [mk("A.12", ctx, raw_value=diff, expected=0.0, severity=sev, unit="USD/share", scheme=Scheme.REPORTING_INTEGRITY,
               rationale="; ".join(msgs) + ".", evidence=ctx.evs(["NET_INCOME", "SHARES_DILUTED", "EPS_DILUTED"], ks=(0,)))]


@detector("A.13", "Current tax vs cash taxes", F, Area.INTEGRITY)
def a13(ctx: RunContext):
    """Current tax expense (total - deferred) vs income taxes paid, 3-year cumulative."""
    tot = ctx.hist("INCOME_TAX", 3)
    dfd = ctx.hist("DEFERRED_TAX_EXPENSE", 3)
    paid = ctx.hist("INCOME_TAXES_PAID", 3)
    if len(tot.dropna()) < 2 or len(paid.dropna()) < 2:
        raise SkipDetector("tax expense or taxes paid not reported for 2+ years")
    cur = (tot - dfd.reindex(tot.index).fillna(0)).sum()
    cash = paid.sum()
    gap = safe_div(cur - cash, max(abs(cur), abs(cash), 1.0))
    sev = float(np.clip((abs(gap) - 0.25) / 0.5, 0, 1)) if finite(gap) else 0.0
    return [mk("A.13", ctx, raw_value=gap, expected=0.0, severity=sev * 0.7, unit="ratio", area=Area.RESERVES,
               rationale=f"3-year current tax expense {fmt_money(cur)} vs cash taxes paid {fmt_money(cash)} (gap {fmt_pct(gap)}). Timing of payments and uncertain positions explain moderate gaps.",
               evidence=ctx.evs(["INCOME_TAX", "INCOME_TAXES_PAID"], ks=(0,)))]


SIGNIFICANT_ITEMS = ("REVENUE", "NET_INCOME", "PRETAX_INCOME", "OPERATING_INCOME", "CFO", "TOTAL_ASSETS", "EQUITY",
                     "AR", "INVENTORY", "COGS", "TOTAL_LIABILITIES", "RETAINED_EARNINGS")


@detector("A.14", "Silent restatement (revised comparatives)", F, Area.INTEGRITY)
def a14(ctx: RunContext):
    """A prior-period value re-reported differently in a later 10-K, with or without a restatement label."""
    v = ctx.store.versions("A")
    if v.empty:
        raise SkipDetector("no versioned annual facts")
    v = v[v["item"].isin(SIGNIFICANT_ITEMS) & (v["end"] <= ctx.focus + pd.Timedelta(days=8))]
    v = v[v["end"] >= ctx.focus - pd.Timedelta(days=3 * 366)]
    labeled = {e.accession for e in ctx.events_of("NON_RELIANCE", "AMENDED_10K")}
    ni_base = max(abs(ctx.a("NET_INCOME")), 1.0)
    ta_base = max(abs(ctx.a("TOTAL_ASSETS")), 1.0)
    out = []
    for (item, end), g in v.groupby(["item", "end"]):
        vals = g.drop_duplicates("val")
        if len(vals) < 2:
            continue
        first, last = vals.iloc[0], vals.iloc[-1]
        delta = float(last["val"] - first["val"])
        rel = abs(safe_div(delta, first["val"]))
        if not finite(rel) or rel < 0.005:
            continue
        income_item = item in ("REVENUE", "NET_INCOME", "PRETAX_INCOME", "OPERATING_INCOME", "COGS", "RETAINED_EARNINGS", "EQUITY")
        base = ni_base if item in ("NET_INCOME", "PRETAX_INCOME", "OPERATING_INCOME", "RETAINED_EARNINGS") else ta_base
        mat = abs(delta) / base
        sev = float(np.clip(0.3 + 4 * mat if income_item else 0.15 + 2 * mat, 0, 1))
        is_labeled = last["accn"] in labeled
        period_label = f"FY{pd.Timestamp(end).year}"
        tags = ["labeled_restatement" if is_labeled else "unlabeled_revision"]
        direction = "income_increasing" if (income_item and first["val"] > last["val"] and item != "COGS") else "neutral"
        ev = [
            EvidenceRef(node_id=f"fact:{first['concept']}|{first['accn']}|{end}", kind="fact",
                        label=f"{BY_ID[item].label} {period_label} as first reported", accession=first["accn"],
                        doc_url=ctx.doc_url(first["accn"]), concept=first["concept"], period=period_label,
                        value=float(first["val"]), unit="USD", filed=str(pd.Timestamp(first["filed"]).date())),
            EvidenceRef(node_id=f"fact:{last['concept']}|{last['accn']}|{end}", kind="fact",
                        label=f"{BY_ID[item].label} {period_label} as re-reported", accession=last["accn"],
                        doc_url=ctx.doc_url(last["accn"]), concept=last["concept"], period=period_label,
                        value=float(last["val"]), unit="USD", filed=str(pd.Timestamp(last["filed"]).date())),
        ]
        out.append(mk("A.14", ctx, period=period_label, raw_value=delta, expected=0.0, severity=sev, unit="USD",
                      scheme=Scheme.REPORTING_INTEGRITY, direction=direction, tags=tags,
                      impact=ImpactEstimate(abs(delta), abs(delta), abs(delta), "disclosed_amount",
                                            f"{BY_ID[item].label}: re-reported {fmt_money(last['val'])} vs first {fmt_money(first['val'])}") if income_item else None,
                      rationale=(f"{BY_ID[item].label} for {period_label} was first reported as {fmt_money(first['val'])} "
                                 f"(filed {pd.Timestamp(first['filed']).date()}) and later as {fmt_money(last['val'])} "
                                 f"(filed {pd.Timestamp(last['filed']).date()}), a change of {fmt_pct(delta / first['val'] if first['val'] else np.nan)}"
                                 f"{'' if is_labeled else ', without a 10-K/A or non-reliance filing'}. Reclassifications and recasts are common legitimate causes."),
                      evidence=ev))
    if not out:
        return [mk("A.14", ctx, raw_value=0.0, expected=0.0, severity=0.0,
                   rationale="No significant prior-period values were re-reported differently in the last 3 years.")]
    out.sort(key=lambda s: -s.severity)
    return out[:6]


@detector("A.15", "Text-number consistency (MD&A vs XBRL)", F, Area.INTEGRITY, needs_text=True)
def a15(ctx: RunContext):
    """Company-wide revenue / net income totals stated in MD&A match the tagged financial statements."""
    from ffi.detectors.textnum import check_amounts

    doc = ctx.doc
    if doc is None:
        raise SkipDetector("10-K text not available")
    text = doc.section_text("item7")
    if len(text) < 2000:
        raise SkipDetector("MD&A section not found")
    known = {item: [ctx.a(item, k) for k in range(3)] for item in ("REVENUE", "NET_INCOME")}
    checked, bad = check_amounts(text, known)
    if checked == 0:
        raise SkipDetector("no company-wide revenue / net income totals stated in MD&A in a recognisable form")
    sev = min(1.0, 0.45 * len(bad))
    ev = []
    for b in bad[:3]:
        s = doc.text.find(b.text)
        if s >= 0:
            ev.append(ctx.ev_span(doc, s, s + len(b.text), f"MD&A statement: {b.text[:60]}"))
    return [mk("A.15", ctx, raw_value=len(bad), expected=0, severity=sev, unit="mismatches",
               scheme=Scheme.REPORTING_INTEGRITY,
               rationale=(f"{checked} company-wide revenue / net income totals in MD&A checked against XBRL; {len(bad)} differ from the "
                          "nearest reported total" + (f" (e.g. '{bad[0].text[:90]}' vs {fmt_money(bad[0].nearest)})." if bad else ".")),
               evidence=ev)]



@detector("A.16", "Rounding consistency", F, Area.INTEGRITY)
def a16(ctx: RunContext):
    """Monetary facts in the focus 10-K share one rounding unit; mixed precision suggests manual adjustments."""
    rec = ctx.A.prov.get(("TOTAL_ASSETS", ctx.focus))
    if rec is None:
        raise SkipDetector("focus 10-K not identified")
    d = ctx.store.df
    vals = d[(d["accn"] == rec.accn) & (d["unit"] == ctx.currency)]["val"].abs()
    vals = vals[vals >= 1e4]
    if len(vals) < 30:
        raise SkipDetector("too few monetary facts")
    units = [1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1.0]
    share = {u: float((np.mod(vals, u) == 0).mean()) for u in units}
    dominant = next((u for u in units if share[u] >= 0.8), 1.0)
    off = 1 - share[dominant]
    sev = float(np.clip((off - 0.10) / 0.3, 0, 1)) * 0.6
    return [mk("A.16", ctx, raw_value=off, expected=0.0, severity=sev, unit="share",
               rationale=f"{fmt_pct(share[dominant])} of {len(vals)} monetary facts are rounded to {fmt_money(dominant)}; {fmt_pct(off)} use finer precision.")]
