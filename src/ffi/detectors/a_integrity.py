"""Family A: accounting integrity checks. Deterministic; run first."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ffi.context import RunContext
from ffi.detectors.base import Area, EvidenceRef, ImpactEstimate, Scheme, SkipDetector, detector
from ffi.detectors.helpers import finite, mk, need, need_same_basis
from ffi.detectors.scores import V
from ffi.detectors.stats import fmt_money, fmt_pct, safe_div
from ffi.text.search import search, sentence_bounds
from ffi.xbrl.canonical import BY_ID, basis
from ffi.xbrl.facts import ANNUAL_DAYS, FactRecord, fiscal_year_of

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
        gap = ta - (tl + temp + eq)
        # When the filed total "liabilities and equity" ties to assets, parts that fall short of it are a line between
        # liabilities and equity that is not tagged (BlackRock 2025: $5.43B of redeemable interests, one of them a company
        # extension that companyfacts does not carry), not a statement that fails to balance
        ties = finite(le) and abs(ta - le) <= 0.001 * abs(ta)
        if not (ties and gap > 0):
            checks.append((label, gap, ["TOTAL_ASSETS", "TOTAL_LIABILITIES", "TEMPORARY_EQUITY", "EQUITY"]))
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
        # tagged gross profit below revenue less cost of sales is usually gross profit on product sales, with revenue from fees
        # or other sources outside it (Costco FY2019: $3.35B of membership fees on $152.7B of revenue); gross profit above
        # it is not explained that way and is tested
        short = rev - cogs - gp
        if not 0 < short <= 0.25 * abs(rev):
            rows.append(("Revenue - COGS = Gross profit", short, rev, ["REVENUE", "COGS", "GROSS_PROFIT"]))
    g, ad, n = ctx.a("PPE_GROSS"), ctx.a("ACCUM_DEP"), ctx.a("PPE_NET")
    if finite(g, ad, n) and g - ad - n >= 0:
        # Net PP&E above gross less depreciation is not a break: the tagged gross figure is often a subtotal that leaves out
        # land and construction in progress, which the balance sheet includes (Alphabet 2024: $199.8B gross excluding
        # $50.6B of construction in progress). Net below the implied figure cannot be explained that way and is tested.
        rows.append(("Gross PP&E - accumulated depreciation = Net PP&E", g - ad - n, g, ["PPE_GROSS", "ACCUM_DEP", "PPE_NET"]))
    ca = ctx.a("CURRENT_ASSETS")
    # a part tagged on a broader basis is not a part of current assets: cash including restricted cash (often non-current),
    # inventory before reserves
    parts = [float("nan") if i in ("CASH", "INVENTORY") and ctx.basis(i) else ctx.a(i) for i in ("CASH", "ST_INVESTMENTS", "AR", "INVENTORY")]
    if finite(ca) and sum(1 for p in parts if finite(p)) >= 2:
        s = sum(p for p in parts if finite(p))
        excess = max(0.0, s - ca)
        sti = ctx.a("ST_INVESTMENTS")
        if finite(sti) and excess > 0:
            # short-term investments are often a part of cash and equivalents, not a further line (Target: "cash and cash
            # equivalents included short-term investments of $4,611M"); the parts only break the total if they do so without them
            excess = min(excess, max(0.0, s - sti - ca))
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
    v = V(ctx.A, ctx.focus)
    if basis("CASH_RESTRICTED_TOTAL", v.concept("CASH_RESTRICTED_TOTAL")) != basis("CASH_RESTRICTED_TOTAL", v.concept("CASH_RESTRICTED_TOTAL", 1)):
        raise SkipDetector("cash is reported including restricted cash in one year and excluding it in the other")
    # the cash flows of discontinued operations are a line of their own beside continuing-operations totals (Disney FY2019:
    # 5,984 - 15,096 - 464 - 98 plus 10,974 = the 1,300 change) or already inside the totals, and neither the tags nor the
    # totals say which reliably (of 172 company-years that tag the line, 42 tied only without it and 52 only with it, whatever
    # the tags): the statements reconcile when either presentation does, so the line is counted at most once
    disc = ctx.a("CASH_DISCONTINUED")
    flows = cfo + cfi + cff + (fx if finite(fx) else 0.0)
    readings = [(flows, "")] + ([(flows + disc, " and the cash flows of discontinued operations")] if finite(disc) and disc else [])
    flows, with_disc = min(readings, key=lambda r: abs(end - beg - r[0]))
    resid = (end - beg) - flows
    base = max(abs(end), abs(cfo), 1.0)
    sev = _residual_severity(resid, base, 0.01, 0.05)
    return [mk("A.03", ctx, raw_value=resid, expected=0.0, severity=sev, unit="USD", scheme=Scheme.REPORTING_INTEGRITY,
               rationale=(f"Change in cash {fmt_money(end - beg)} vs operating, investing and financing cash flows plus the "
                          f"currency effect{with_disc} {fmt_money(flows)}; "
                          f"unexplained {fmt_money(resid)}."),
               evidence=ctx.evs(["CASH_RESTRICTED_TOTAL"], ks=(0, 1)) + ctx.evs(["CFO", "CFI", "CFF", "CASH_DISCONTINUED"], ks=(0,)))]


def dividends_declared(ctx: RunContext) -> tuple[float, str]:
    """Dividends charged to equity in the focus year and how the figure was obtained. Declared dividends are what reduce
    equity; cash dividends paid lag them when a dividend declared near the year end is paid after it (Progressive
    declares an annual dividend in December and pays it in January), so paid dividends are adjusted for the change in
    dividends payable, and used alone only when that change is not reported."""
    dec = ctx.a("DIVIDENDS_DECLARED")
    if finite(dec):
        return dec, "declared"
    paid = ctx.a("DIVIDENDS_PAID")
    if not finite(paid):
        return 0.0, "none reported"
    p1, p0 = ctx.a("DIVIDENDS_PAYABLE"), ctx.a("DIVIDENDS_PAYABLE", 1)
    if finite(p1, p0):
        return paid + (p1 - p0), "paid, adjusted for the change in dividends payable"
    return paid, "paid (declared not reported)"


STOCK_ISSUED_NONCASH_RE = (r"issued[^.]{0,120}shares[^.]{0,200}(?:as consideration|in exchange for)"
                           r"|(?:issued|issuance of)[^.]{0,80}shares[^.]{0,160}(?:in (?:connection|exchange) with|as consideration|to (?:acquire|fund the|pay for))"
                           r"[^.]{0,80}(?:acquisition|merger|business combination|acquire)"
                           r"|(?:acquisition|merger)[^.]{0,160}(?:consideration|purchase price)[^.]{0,120}(?:in|with|of) (?:shares|common stock|our stock)"
                           r"|(?:convertible )?preferred stock[^.]{0,120}(?:converted|conversion)[^.]{0,120}(?:common stock|shares)"
                           r"|(?:converted|conversion)[^.]{0,120}preferred[^.]{0,120}(?:initial public offering|IPO)")


@detector("A.04", "Equity roll-forward", F, Area.INTEGRITY)
def a04(ctx: RunContext):
    """Begin equity + NI + OCI - dividends - buybacks + issuance + SBC = end equity."""
    e1, e0 = ctx.a("EQUITY"), ctx.a("EQUITY", 1)
    ni = ctx.ni_parent()  # the parent's equity moves with the parent's share of income
    need(e1, e0, ni, reason="equity or net income attributable to the company not reported")
    add = {i: ctx.a(i) for i in ("OCI", "BUYBACKS", "EQUITY_ISSUED", "EQUITY_ISSUED_STMT", "EQUITY_ISSUED_ESPP", "SBC",
                                 "CONVERTIBLE_EQUITY_COMPONENT", "SHARE_TAX_WITHHELD", "EQUITY_ISSUED_ACQ", "EQUITY_CONVERTED",
                                 "EQUITY_ISSUED_OTHER")}
    g = {k: (v if finite(v) else 0.0) for k, v in add.items()}
    div, div_how = dividends_declared(ctx)
    # shares issued to pay for acquisitions (Boeing 2025: Spirit AeroSystems, $4.7B; Disney 2019: 21st Century Fox), converted
    # securities (Uber's IPO: $14B of preferred stock) and taxes paid on net share settlement (Snowflake: $0.7B) are
    # tagged lines of the statement of equity and the cash flow statement, not unexplained movements
    # shares issued as the statement of equity shows them include what cash proceeds miss (dividend reinvestment, forward sales
    # settled later: Realty Income 2019, $2.1B against $0.8B of proceeds); the cash employees pay for plan shares can be in
    # both the proceeds and the statement's employee-plan line, so the larger of the two readings counts, once
    issued = max(g["EQUITY_ISSUED"], g["EQUITY_ISSUED_STMT"] + g["EQUITY_ISSUED_ESPP"])
    expected_end = (e0 + ni + g["OCI"] - div - g["BUYBACKS"] + issued + g["SBC"]
                    + g["CONVERTIBLE_EQUITY_COMPONENT"] - abs(g["SHARE_TAX_WITHHELD"]) + g["EQUITY_ISSUED_ACQ"]
                    + g["EQUITY_CONVERTED"] + g["EQUITY_ISSUED_OTHER"])
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
    if sev > 0 and resid > 0 and ctx.doc is not None:
        # equity that rose without cash or income: shares issued as the price of an acquisition (Disney 2019, 21st Century Fox:
        # $35.7B) or preferred stock converted at an IPO (Uber 2019) are in the statement of equity, whose lines the SEC's data
        # does not carry when the company breaks them out by class of equity
        hit = next(iter(search(ctx.doc, STOCK_ISSUED_NONCASH_RE, ("item7", "item8"))), None)
        if hit is not None:
            sev *= 0.4
            nci_note += f" The annual report describes shares issued without cash: \"{hit.quote[:200]}\"."
    return [mk("A.04", ctx, raw_value=resid, expected=0.0, severity=sev, unit="USD", scheme=Scheme.REPORTING_INTEGRITY,
               rationale=(f"Equity moved {fmt_money(e1 - e0)}; NI, OCI, dividends ({div_how}), buybacks, issuance (including shares issued for "
                          f"acquisitions and conversions), tax withheld on stock awards and SBC explain "
                          f"{fmt_money(expected_end - e0)}. Unexplained {fmt_money(resid)} "
                          f"({fmt_pct(safe_div(resid, base))} of equity). Untagged equity offerings, convertible-note "
                          "components, warrant exercises and tax withholding on vesting are common causes." + nci_note),
               evidence=ctx.evs(["EQUITY", "DIVIDENDS_PAYABLE"], ks=(0, 1))
               + ctx.evs(["NET_INCOME", "BUYBACKS", "DIVIDENDS_DECLARED", "DIVIDENDS_PAID"], ks=(0,)))]


RE_RECLASS_RE = (r"cumulative[- ]effect (?:adjustment|of (?:the )?(?:adoption|change in accounting|initial application))"
                 r"|cumulative adjustment to (?:the )?(?:opening|beginning)? ?retained earnings"
                 r"|(?:retire|retired|retirement|cancel|cancell?ed|cancellation)\w*\s+(?:of\s+)?(?:its\s+|our\s+|all\s+)?.{0,40}?treasury (?:stock|shares)"
                 r"|treasury (?:stock|shares)[^.]{0,60}(?:were |was )?(?:retired|cancell?ed)")


@detector("A.05", "Retained-earnings roll-forward", F, Area.INTEGRITY)
def a05(ctx: RunContext):
    """RE(t) = RE(t-1) + NI - dividends (repurchases charged to RE are recognised)."""
    r1, r0, ni = ctx.a("RETAINED_EARNINGS"), ctx.a("RETAINED_EARNINGS", 1), ctx.ni_parent()
    need(r1, r0, ni, reason="retained earnings or net income attributable to the company not reported")
    div, div_how = dividends_declared(ctx)
    resid = r1 - (r0 + ni - div)
    bb = ctx.a("BUYBACKS")
    tw = ctx.a("SHARE_TAX_WITHHELD")
    # equity outflows that companies may charge (wholly or partly) to retained earnings instead of paid-in capital
    outflows = (bb if finite(bb) and bb > 0 else 0.0) + (tw if finite(tw) and tw > 0 else 0.0)
    explained = ""
    if outflows > 0 and resid < 0 and -resid <= 1.10 * outflows:
        parts = "share repurchases" + (f" ({fmt_money(bb)})" if finite(bb) and bb > 0 else "")
        if finite(tw) and tw > 0:
            parts += f" and taxes paid on net share settlement of stock awards ({fmt_money(tw)})"
        explained = (f" The difference is within the {parts}, which companies commonly charge in part to retained earnings "
                     "rather than paid-in capital.")
        resid_eff = 0.0
    elif div > ni and resid > 0:
        # dividends above earnings are often charged to additional paid-in capital, so retained earnings fall less
        explained = (f" Dividends exceeded net income by {fmt_money(div - ni)}; companies often charge the excess to "
                     "additional paid-in capital, which explains up to that amount.")
        resid_eff = max(0.0, resid - (div - ni))
    else:
        resid_eff = resid
    base = max(abs(ni), abs(r1) * 0.1, 1.0)
    sev = _residual_severity(resid_eff, base, 0.05, 0.30)
    if sev > 0 and ctx.doc is not None:
        # equity moves between its parts without touching income or dividends: a cumulative-effect adjustment on adopting a
        # standard (Toyota 2019: ASU 2016-01, 1.27T yen; Salesforce: ASC 606) and treasury shares retired against retained
        # earnings (Disney 2019: $40B; ASML) show only in the statement of equity, whose lines companyfacts leaves out
        hit = next(iter(search(ctx.doc, RE_RECLASS_RE, ("item7", "item8"))), None)
        if hit is not None:
            sev *= 0.4
            explained += f" The annual report describes a change in retained earnings that is not income or dividends: \"{hit.quote[:200]}\"."
    return [mk("A.05", ctx, raw_value=resid, expected=0.0, severity=sev, unit="USD", scheme=Scheme.REPORTING_INTEGRITY,
               rationale=(f"Retained earnings moved {fmt_money(r1 - r0)} vs NI less dividends ({div_how}) {fmt_money(ni - div)}; "
                          f"residual {fmt_money(resid)}.{explained}"),
               evidence=ctx.evs(["RETAINED_EARNINGS", "DIVIDENDS_PAYABLE"], ks=(0, 1))
               + ctx.evs(["NET_INCOME", "DIVIDENDS_DECLARED", "DIVIDENDS_PAID", "BUYBACKS", "SHARE_TAX_WITHHELD"], ks=(0,)))]


@detector("A.06", "Quarterly-to-annual consistency", F, Area.INTEGRITY)
def a06(ctx: RunContext):
    """Q4 implied by FY less Q1-Q3 is plausible (sign and magnitude) for revenue and cost of sales."""
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
    # pre-tax income tagged before the share of equity-method investees' results (Toyota): that share is added below tax
    eqm = ctx.a("EQUITY_METHOD_INCOME")
    before_eqm = bool(basis("PRETAX_INCOME", V(ctx.A, ctx.focus).concept("PRETAX_INCOME"))) and finite(eqm)
    # pre-tax income includes the noncontrolling holders' share, so any after-tax subtotal may tie: continuing
    # operations, net income including NCI, or the parent's net income plus the NCI share
    cands = [v for v in (ctx.a("INCOME_CONT_OPS"), ctx.a("NET_INCOME_INCL_NCI"),
                         ni + nci if finite(ni, nci) else float("nan"), ni) if finite(v)]
    need(pt, tax, *(cands[:1] or [float("nan")]))
    after_tax = pt - tax + (eqm if before_eqm else 0.0)
    ico = min(cands, key=lambda v: abs(after_tax - v))
    resid = after_tax - ico
    base = max(abs(pt), 1.0)
    # equity-method income and discontinued ops can legitimately sit between these lines
    sev = _residual_severity(resid, base, 0.03, 0.15)
    return [mk("A.07", ctx, raw_value=resid, expected=0.0, severity=sev, unit="USD", scheme=Scheme.REPORTING_INTEGRITY,
               rationale=(f"Pre-tax income less tax{' plus equity-method income' if before_eqm else ''} {fmt_money(after_tax)} "
                          f"vs income after tax {fmt_money(ico)}; gap {fmt_money(resid)}."),
               evidence=ctx.evs(["PRETAX_INCOME", "INCOME_TAX", "NET_INCOME"] + (["EQUITY_METHOD_INCOME"] if before_eqm else []), ks=(0,)))]


@detector("A.08", "PP&E roll-forward", F, Area.INTEGRITY)
def a08(ctx: RunContext):
    """Growth in gross PP&E not explained by capex and acquisitions (possible capitalization outside capex)."""
    g1, g0, capex = ctx.a("PPE_GROSS"), ctx.a("PPE_GROSS", 1), ctx.a("CAPEX")
    need(g1, g0, capex)
    need_same_basis(ctx, "PPE_GROSS")  # finance-lease assets in one year only would read as unexplained additions
    acq = ctx.a("ACQUISITIONS")
    acq = acq if finite(acq) else 0.0
    # additions that are not cash capex of the year: assets acquired under finance leases, and capital expenditure accrued
    # but unpaid at the year end net of what was accrued a year earlier (Oracle 2026: 4.9B + 5.3B - 1.4B of the 7.4B gap)
    lease, up1, up0 = ctx.a("FINANCE_LEASE_ASSETS_ADDED"), ctx.a("CAPEX_UNPAID"), ctx.a("CAPEX_UNPAID", 1)
    noncash = (lease if finite(lease) else 0.0) + (up1 - up0 if finite(up1, up0) else 0.0)
    unexplained = (g1 - g0) - capex - 0.5 * acq - noncash
    sev = _residual_severity(max(unexplained, 0.0), g0, 0.03, 0.15)
    return [mk("A.08", ctx, raw_value=unexplained, expected=0.0, severity=sev, unit="USD",
               scheme=Scheme.SHIFT_EXPENSES_LATER, area=Area.CAPITALIZATION, direction="income_increasing",
               rationale=(f"Gross PP&E grew {fmt_money(g1 - g0)} with capex {fmt_money(capex)}"
                          f"{' and acquisitions ' + fmt_money(acq) if acq else ''}; "
                          f"{fmt_money(unexplained)} of additions are not explained by cash spending"
                          f"{' and by ' + fmt_money(noncash) + ' of finance-lease additions and unpaid capex' if noncash else ''} "
                          "(disposals would make the gap larger; asset-retirement additions and capitalized interest also add to "
                          "PP&E without cash capex)."),
               evidence=ctx.evs(["PPE_GROSS"], ks=(0, 1)) + ctx.evs(["CAPEX", "FINANCE_LEASE_ASSETS_ADDED", "CAPEX_UNPAID"], ks=(0,)))]


@detector("A.09", "Allowance roll-forward", F, Area.INTEGRITY)
def a09(ctx: RunContext):
    """Allowance(t) = Allowance(t-1) + provision - write-offs."""
    a1, a0, prov, wo = abs(ctx.a("ALLOWANCE")), abs(ctx.a("ALLOWANCE", 1)), ctx.a("PROVISION_DOUBTFUL"), ctx.a("WRITE_OFFS")
    need(a1, a0, prov, wo, reason="allowance roll-forward components not tagged")
    need_same_basis(ctx, "ALLOWANCE")
    resid = a1 - (a0 + prov - wo)
    sev = _residual_severity(resid, max(a1, a0, 1.0), 0.10, 0.50)
    note = ""
    acq, ta0 = ctx.a("ACQUISITIONS"), ctx.a("TOTAL_ASSETS", 1)
    if resid > 0 and finite(acq, ta0) and ta0 > 0 and acq / ta0 > 0.10:
        # allowance taken on with acquired receivables is a roll-forward item the provision and write-off tags do not carry
        sev *= 0.5
        note = (f" Discounted: {fmt_money(acq)} was paid for acquisitions ({fmt_pct(acq / ta0)} of prior-year assets), and "
                f"allowance acquired with the receivables is not part of the tagged provision or write-offs.")
    return [mk("A.09", ctx, raw_value=resid, expected=0.0, severity=sev, unit="USD", area=Area.RECEIVABLES,
               scheme=Scheme.REPORTING_INTEGRITY,
               rationale=f"Allowance roll-forward residual {fmt_money(resid)} (recoveries and FX can explain small gaps)." + note,
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
    acq = sum(x for x in (ctx.a("ACQUISITIONS"), ctx.a("EQUITY_ISSUED_ACQ")) if finite(x))
    div = ctx.a("DIVESTITURES")
    if resid > 0 and acq > 0 and sev > 0:
        # goodwill of acquired businesses is usually tagged by segment only (Coca-Cola 2019: Costa), where companyfacts has none
        sev *= 0.5
        note += f" Discounted: {fmt_money(acq)} was paid for acquisitions in the year, and the goodwill acquired is often tagged by segment only."
    elif resid < 0 and finite(div) and div > 0 and sev > 0:
        # goodwill of a business sold or moved to held for sale leaves the balance sheet without a roll-forward line (GE 2019)
        sev *= 0.5
        note += f" Discounted: {fmt_money(div)} was received from divestitures in the year, whose goodwill leaves the balance."
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
    ni, sh, eps, epsb = ctx.ni_parent(), ctx.a("SHARES_DILUTED"), ctx.a("EPS_DILUTED"), ctx.a("EPS_BASIC")
    need(ni, sh, eps)
    # EPS divides the income left for common shareholders (after preferred dividends) by the shares: Boeing 2025 net income
    # 2,235 / 762.3 = 2.93, but (2,235 - 345 of preferred dividends) / 762.3 = 2.48, the reported figure
    ni_eps = ctx.a("NI_COMMON_DILUTED")
    num = ni_eps if finite(ni_eps) and abs(ni_eps) > 0 else ni
    implied = safe_div(num, sh)
    diff = implied - eps
    tol = max(0.02, abs(eps) * 0.02)  # preferred dividends / participating securities create small gaps
    sev = float(np.clip((abs(diff) - tol) / (abs(eps) * 0.15 + 0.05), 0, 1)) if abs(diff) > tol else 0.0
    msgs = [f"{'Net income for diluted EPS' if num is not ni else 'NI'} / diluted shares = {implied:.2f} vs reported diluted EPS {eps:.2f}"]
    factor = ctx.A.rescaled.get(("SHARES_DILUTED", ctx.focus))
    if factor:
        msgs.append(f"diluted shares are tagged in {'millions' if factor == 1e6 else 'thousands'} in the XBRL "
                    "(a tagging scale error, corrected here; the financial statements themselves are consistent)")
    if finite(epsb) and ni > 0 and eps > epsb + 0.005:
        sev = max(sev, 0.8)
        msgs.append(f"diluted EPS {eps:.2f} exceeds basic {epsb:.2f}")
    return [mk("A.12", ctx, raw_value=diff, expected=0.0, severity=sev, unit="USD/share", scheme=Scheme.REPORTING_INTEGRITY,
               rationale="; ".join(msgs) + ".", evidence=ctx.evs(["NET_INCOME", "NI_COMMON_DILUTED", "SHARES_DILUTED", "EPS_DILUTED"], ks=(0,)))]


@detector("A.13", "Current tax vs cash taxes", F, Area.INTEGRITY)
def a13(ctx: RunContext):
    """Current tax expense (total - deferred) vs income taxes paid, 3-year cumulative."""
    tot = ctx.hist("INCOME_TAX", 3)
    dfd = ctx.hist("DEFERRED_TAX_EXPENSE", 3).reindex(tot.index)
    paid = ctx.hist("INCOME_TAXES_PAID", 3).reindex(tot.index)
    # both sums cover the same years; deferred tax is subtracted in every year or (never reported) in none
    years = tot.index[tot.notna() & paid.notna()]
    if dfd[years].notna().any():
        years = years[dfd[years].notna().to_numpy()]
    if len(years) < 2:
        raise SkipDetector("tax expense and taxes paid not both reported for 2+ years")
    cur = float((tot[years] - dfd[years].fillna(0)).sum())
    cash = float(paid[years].sum())
    gap = safe_div(cur - cash, max(abs(cur), abs(cash), 1.0))
    sev = float(np.clip((abs(gap) - 0.25) / 0.5, 0, 1)) if finite(gap) else 0.0
    return [mk("A.13", ctx, raw_value=gap, expected=0.0, severity=sev * 0.7, unit="ratio", area=Area.RESERVES,
               rationale=(f"{len(years)}-year current tax expense {fmt_money(cur)} vs cash taxes paid {fmt_money(cash)} over the same years "
                          f"({ctx.label(years[0])}-{ctx.label(years[-1])}; gap {fmt_pct(gap)})"
                          + ("" if dfd[years].notna().any() else "; deferred tax is not reported, so total tax expense is used")
                          + ". Timing of payments and uncertain positions explain moderate gaps."),
               evidence=ctx.evs(["INCOME_TAX", "INCOME_TAXES_PAID"], ks=(0,)))]


def _fact_id(row) -> str:
    """The fact id of a versions row, in the form every other cited fact uses (taxonomy:concept|unit|start|end|accession)."""
    start = row["start"] if "start" in row and not pd.isna(row["start"]) else None
    return FactRecord(row["item"], row["concept"], row["taxonomy"], row["unit"], start, pd.Timestamp(row["end"]),
                      float(row["val"]), row["accn"], row["form"], pd.Timestamp(row["filed"])).fact_id


SIGNIFICANT_ITEMS = ("REVENUE", "NET_INCOME", "PRETAX_INCOME", "OPERATING_INCOME", "CFO", "TOTAL_ASSETS", "EQUITY",
                     "AR", "INVENTORY", "COGS", "TOTAL_LIABILITIES", "RETAINED_EARNINGS")


RECAST_CAUSE_RE = re.compile(
    r"(?i)\b(?:adopt\w*|ASU|accounting standards update|ASC \d{3}|new (?:accounting )?(?:standard|guidance)|discontinued operations?"
    r"|spin-?off|separation|divestitures?|long-duration targeted improvements"
    r"|chang\w+ (?:its |our |the |in )?(?:accounting )?(?:methods?|principles?|polic\w+)|accounting changes?)\b")
RECAST_WORD_RE = re.compile(
    r"(?i)\b(?:retrospective(?:ly)?|recast|re-cast|as adjusted|have been (?:revised|reclassified|adjusted)"
    r"|(?:revised|reclassified|adjusted|recast)[^.]{0,80}\bto (?:conform|reflect))\b")
NOT_A_RECAST_RE = re.compile(r"(?i)\brestate\w*|material weakness|misstatement|\berrors?\b")


def disclosed_recast(doc) -> str | None:
    """The sentence in which an annual report says it recast earlier periods on adopting a standard or for discontinued
    operations ("the amounts for prior periods have been recast"), or None. A paragraph about errors or a restatement is not
    one. The comparatives of such a report differ from the first-reported figures by the company's own, disclosed choice."""
    if doc is None:
        return None
    cache = doc.__dict__.setdefault("_recast_sentence", [])
    if cache:
        return cache[0]
    found = None
    for p in doc.paragraphs:
        if p.is_table_row or not 60 <= len(p.text) <= 2500:
            continue
        w = RECAST_WORD_RE.search(p.text)
        if w and RECAST_CAUSE_RE.search(p.text) and not NOT_A_RECAST_RE.search(p.text):
            lo, hi = sentence_bounds(doc.text, p.start + w.start())
            found = doc.text[lo:hi].strip()
            break
    cache.append(found)
    return found


@detector("A.14", "Silent restatement (revised comparatives)", F, Area.INTEGRITY)
def a14(ctx: RunContext):
    """A prior-period value re-reported differently in a later 10-K, with or without a restatement label."""
    v = ctx.store.versions("A")
    if v.empty:
        raise SkipDetector("no versioned annual facts")
    all_versions = v
    docs = {d.accession: d for d in ctx.docs}
    v = v[v["item"].isin(SIGNIFICANT_ITEMS) & (v["end"] <= ctx.focus + pd.Timedelta(days=8))]
    v = v[v["end"] >= ctx.focus - pd.Timedelta(days=3 * 366)]
    v = v[v.groupby(["item", "end"])["val"].transform("nunique") > 1]  # only values that were re-reported differently
    amended = {e.accession for e in ctx.events_of("AMENDED_10K", "NON_RELIANCE")}
    non_reliance = [pd.Timestamp(e.date) for e in ctx.events_of("NON_RELIANCE")]
    ni_base = max(abs(ctx.a("NET_INCOME")), 1.0)
    ta_base = max(abs(ctx.a("TOTAL_ASSETS")), 1.0)
    out = []
    for (item, end), g in v.groupby(["item", "end"]):
        first, last = g.iloc[0], g.iloc[-1]  # the first and the latest report (a revision later reversed is no revision)
        delta = float(last["val"] - first["val"])
        rel = abs(safe_div(delta, first["val"]))
        if not finite(rel) or rel < 0.005:
            continue
        income_item = item in ("REVENUE", "NET_INCOME", "PRETAX_INCOME", "OPERATING_INCOME", "COGS", "RETAINED_EARNINGS", "EQUITY")
        base = ni_base if item in ("NET_INCOME", "PRETAX_INCOME", "OPERATING_INCOME", "RETAINED_EARNINGS") else ta_base
        mat = abs(delta) / base
        sev = float(np.clip(0.3 + 4 * mat if income_item else 0.15 + 2 * mat, 0, 1))
        # labelled: re-reported in an amendment, or within a year after a non-reliance notice
        is_labeled = last["accn"] in amended or any(0 <= (pd.Timestamp(last["filed"]) - d).days <= 366 for d in non_reliance)
        period_label = ctx.A.label(pd.Timestamp(end)) if pd.Timestamp(end) in ctx.A.labels else f"FY{fiscal_year_of(pd.Timestamp(end))}"
        tags = ["labeled_restatement" if is_labeled else "unlabeled_revision"]
        recast_note = ""
        if not is_labeled:
            # a recast leaves a trace: net income of the period is the same in both reports while revenue, operating income or
            # pre-tax income moved (discontinued operations, a reclassification), or the later report says it recast prior
            # periods on adopting a standard. Neither is a correction of earnings, so it cannot reach the flag line.
            ni = all_versions[(all_versions["item"] == "NET_INCOME") & (all_versions["end"] == end)]
            ni_first = ni[ni["accn"] == first["accn"]]["val"]
            ni_last = ni[ni["accn"] == last["accn"]]["val"]
            if (item in ("REVENUE", "COGS", "OPERATING_INCOME", "PRETAX_INCOME") and not ni_first.empty and not ni_last.empty
                    and abs(float(ni_last.iloc[0]) - float(ni_first.iloc[0])) <= 0.005 * max(abs(float(ni_first.iloc[0])), 1.0)):
                tags[0] = "presentation_recast"
                recast_note = (" Net income for the period is the same in both reports, so this is a reclassification or a recast "
                               "(discontinued operations, a change in presentation), not a change in earnings.")
            elif (quote := disclosed_recast(docs.get(last["accn"]))):
                tags[0] = "disclosed_recast"
                recast_note = f" The later report discloses a recast: \"{quote[:200]}\"."
            if recast_note:
                sev = min(sev, 0.35)
        direction = "income_increasing" if (income_item and first["val"] > last["val"] and item != "COGS") else "neutral"
        ev = [
            EvidenceRef(node_id=f"fact:{_fact_id(first)}", kind="fact",
                        label=f"{BY_ID[item].label} {period_label} as first reported", accession=first["accn"],
                        doc_url=ctx.doc_url(first["accn"]), concept=first["concept"], period=period_label,
                        value=float(first["val"]), unit=ctx.currency, filed=str(pd.Timestamp(first["filed"]).date())),
            EvidenceRef(node_id=f"fact:{_fact_id(last)}", kind="fact",
                        label=f"{BY_ID[item].label} {period_label} as re-reported", accession=last["accn"],
                        doc_url=ctx.doc_url(last["accn"]), concept=last["concept"], period=period_label,
                        value=float(last["val"]), unit=ctx.currency, filed=str(pd.Timestamp(last["filed"]).date())),
        ]
        out.append(mk("A.14", ctx, period=period_label, raw_value=delta, expected=0.0, severity=sev, unit=ctx.currency,
                      scheme=Scheme.REPORTING_INTEGRITY, direction=direction, tags=tags,
                      impact=ImpactEstimate(abs(delta), abs(delta), abs(delta), "disclosed_amount",
                                            f"{BY_ID[item].label}: re-reported {fmt_money(last['val'])} vs first {fmt_money(first['val'])}") if income_item else None,
                      rationale=(f"{BY_ID[item].label} for {period_label} was reported as {fmt_money(first['val'])} "
                                 f"(filed {pd.Timestamp(first['filed']).date()}) and later as {fmt_money(last['val'])} "
                                 f"(filed {pd.Timestamp(last['filed']).date()}), a change of {fmt_pct(delta / first['val'] if first['val'] else np.nan)}"
                                 f"{'' if is_labeled else ', without a 10-K/A or non-reliance filing'}. Reclassifications and recasts are common legitimate causes."
                                 + recast_note),
                      evidence=ev))
    if not out:
        return [mk("A.14", ctx, raw_value=0.0, expected=0.0, severity=0.0,
                   rationale="No significant prior-period values were re-reported differently in the last 3 years.")]
    out.sort(key=lambda s: -s.severity)
    return out[:6]


def _as_presented(ctx: RunContext, item: str, period: pd.Timestamp) -> float:
    """`item` for `period` as the focus year's annual report states it: its comparatives may be recast, and its text
    describes the recast figures. The first-reported value when that report does not carry the period."""
    rec = ctx.A.prov.get((item, ctx.focus))
    d = ctx.store.df
    if rec is not None and not d.empty:
        rows = d[(d["accn"] == rec.accn) & (d["item"] == item) & d["days"].between(*ANNUAL_DAYS)
                 & ((d["end"] - period).abs() <= pd.Timedelta(days=8))]
        if not rows.empty:
            return float(rows.sort_values("rank").iloc[0]["val"])
    return ctx.A.get(item, period)


@detector("A.15", "Text-number consistency (MD&A vs XBRL)", F, Area.INTEGRITY, needs_text=True)
def a15(ctx: RunContext):
    """Company-wide revenue / net income totals stated in MD&A match the tagged financial statements."""
    from ffi.detectors.textnum import check_amounts, totals_by_year

    doc = ctx.doc
    if doc is None:
        raise SkipDetector("10-K text not available")
    text = doc.section_text("item7")
    if len(text) < 2000:
        raise SkipDetector("MD&A section not found")
    # each stated total is compared with the year the sentence names; an unnamed year is the focus year (a match with
    # an earlier year's total still counts as checked, but only the focus year's total can make it a mismatch)
    # earlier years as this annual report presents them (its comparatives), which is what its MD&A describes
    years = [ctx.period_at(k) for k in range(3)]

    def totals(item: str, p) -> list[float]:
        # MD&A states the parent's net income; a company that tags only the total including noncontrolling interests has
        # the parent's share (total less theirs) as a second candidate
        incl, nci = _as_presented(ctx, "NET_INCOME_INCL_NCI", p), _as_presented(ctx, "NCI_INCOME", p)
        return [_as_presented(ctx, item, p)] + ([incl - nci] if item == "NET_INCOME" and finite(incl, nci) else [])

    known = {item: {"years": totals_by_year([(p, t) for p in years if p is not None for t in totals(item, p)]),
                    "focus": ctx.focus.year} for item in ("REVENUE", "NET_INCOME")}
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
