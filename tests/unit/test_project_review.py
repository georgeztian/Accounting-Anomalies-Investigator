"""Regression tests for the project-wide review: comparisons across a switch of concept, periods that were not the
year's own, amounts counted twice, figures known only after the year's cutoff, and the report wording."""

from __future__ import annotations

import copy
from datetime import date
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from ffi.context import RunContext
from ffi.detectors.base import SkipDetector
from ffi.kg.store import DuckNXGraphStore
from ffi.resolve.resolver import CompanyRef
from ffi.xbrl.canonical import BY_ID
from ffi.xbrl.facts import FactStore
from tests.conftest import companyfacts, run_one, series


def ctx_of(cf: dict, as_of: date = date(2025, 12, 31)) -> RunContext:
    store = FactStore.from_companyfacts(1, cf, as_of)
    A, AL, Q = store.annual("first"), store.annual("latest"), store.quarterly("first")
    ref = CompanyRef(cik=1, ticker="SYN", name="Synthetic Co", sic=3571, sic_description="Test", fiscal_year_end="1231")
    return RunContext(company=ref, store=store, A=A, A_latest=AL, Q=Q, filings=[], focus=A.periods[-1], as_of=as_of,
                      kg=DuckNXGraphStore(":memory:", as_of=as_of), tuning={"scoring": {"fdr_q": 0.10}})


def switch(cf: dict, item: str, years, concept: str, taxonomy: str = "us-gaap") -> dict:
    """The company's facts with `item` for `years` reported under `concept` instead of the item's main concept."""
    cf = copy.deepcopy(cf)
    main = BY_ID[item].concepts[0]
    for unit, rows in cf["facts"]["us-gaap"][main]["units"].items():
        moved = [r for r in rows if int(r["end"][:4]) in years]
        cf["facts"]["us-gaap"][main]["units"][unit] = [r for r in rows if int(r["end"][:4]) not in years]
        cf["facts"].setdefault(taxonomy, {}).setdefault(concept, {"units": {}})["units"].setdefault(unit, []).extend(moved)
    return cf


def flagged(sigs) -> bool:
    return any(s.flagged for s in sigs)


# ------------------------------------------------------------------ periods
def _quarterly_cf(q1_days: int) -> dict:
    """Revenue for 2023-2024: the fourth quarter of 2023 and the first three quarters of 2024 as filed, the first of them
    lasting `q1_days` (112 = Jack in the Box's 16 weeks)."""
    cf = companyfacts({"REVENUE": {2023: 400.0, 2024: 440.0}, "NET_INCOME": {2023: 40.0, 2024: 44.0}})
    q1_end = (pd.Timestamp("2024-01-01") + pd.Timedelta(days=q1_days - 1)).date().isoformat()
    for start, end, v in (("2023-10-01", "2023-12-31", 100.0), ("2024-01-01", q1_end, 120.0), ("2024-04-01", "2024-06-30", 90.0),
                          ("2024-07-01", "2024-09-30", 95.0)):
        cf["facts"]["us-gaap"]["Revenues"]["units"]["USD"].append(
            {"start": start, "end": end, "val": v * 1e6, "accn": f"q-{end}", "fy": 2024, "fp": "Q", "form": "10-Q",
             "filed": (pd.Timestamp(end) + pd.Timedelta(days=40)).date().isoformat()})
    return cf


def test_fourth_quarter_is_never_derived_from_the_previous_years_fourth_quarter():
    # Jack in the Box: a 16-week first quarter is no quarter (80-100 days), so the three quarter ends before the year end
    # were the previous year's Q4, Q2 and Q3, and "Q4" was the year less those three
    ctx = ctx_of(_quarterly_cf(q1_days=112))
    assert ("REVENUE", ctx.focus) not in ctx.Q.derived
    with pytest.raises(SkipDetector):
        run_one(ctx, "C.REV.04")
    # with a 13-week first quarter the fourth quarter is the year less the year's own three quarters
    ctx = ctx_of(_quarterly_cf(q1_days=90))
    assert ("REVENUE", ctx.focus) in ctx.Q.derived and ctx.Q.get("REVENUE", ctx.focus) == pytest.approx((440 - 120 - 90 - 95) * 1e6)


def test_peer_years_filed_after_the_cutoff_are_unknown():
    cf = companyfacts(series(years=range(2020, 2025)))
    panel = FactStore.from_companyfacts(2, cf).annual("first")
    # each year's annual report is filed on Feb 20 of the next year: on 2024-06-30 the peer's FY2024 is not known yet
    seen = panel.known_by(date(2024, 6, 30))
    assert max(seen.periods) == pd.Timestamp("2023-12-31") and pd.Timestamp("2024-12-31") in panel.periods
    assert panel.known_by(None) is panel


def test_depreciation_including_impairment_never_beats_plain_depreciation():
    # an IFRS filer tags both D&A and "depreciation, amortisation and impairment": the larger one used to win
    cf = companyfacts(series())
    da = cf["facts"]["us-gaap"]["DepreciationDepletionAndAmortization"]["units"]["USD"]
    cf["facts"]["ifrs-full"] = {"DepreciationAmortisationAndImpairmentLossReversalOfImpairmentLossRecognisedInProfitOrLoss":
                                {"units": {"USD": [{**r, "val": r["val"] * 3} for r in da]}}}
    ctx = ctx_of(cf)
    assert ctx.a("DA") == pytest.approx(ctx.a("DEPRECIATION")) and ctx.basis("DA") == ""
    # reported alone it is used, and says so
    del cf["facts"]["us-gaap"]["DepreciationDepletionAndAmortization"]
    assert ctx_of(cf).basis("DA") == "including impairment"


# ------------------------------------------------------------------ integrity tests
def _cash_ctx(disc: float, inside: bool) -> RunContext:
    """Cash rose by the three totals plus `disc`, or by the totals alone when they already hold it (`inside`)."""
    data = series(overrides={"CASH_DISCONTINUED": {2024: disc}})
    if not inside:
        data["CASH_RESTRICTED_TOTAL"][2024] += disc
        data["CASH"][2024] += disc
    return ctx_of(companyfacts(data))


@pytest.mark.parametrize("inside", [True, False])
def test_cash_of_discontinued_operations_is_counted_at_most_once(inside):
    # 42 company-years tie only without the discontinued line and 52 only with it, whatever their tags say
    (s,) = run_one(_cash_ctx(50.0, inside), "A.03")
    assert abs(s.raw_value) < 1 and not s.flagged
    assert ("discontinued" in s.rationale) is not inside


def test_cash_including_restricted_cash_is_not_a_part_of_current_assets():
    data = series(overrides={"CURRENT_ASSETS": {2024: 600.0}, "CASH": {2024: 900.0}, "AR": {2024: 50.0}, "INVENTORY": {2024: 50.0}})
    del data["CASH_RESTRICTED_TOTAL"]
    assert run_one(ctx_of(companyfacts(data)), "A.02")[0].raw_value > 0  # cash and equivalents above current assets is a break
    # cash including restricted cash, much of it non-current, is no part of current assets
    restricted = ctx_of(switch(companyfacts(data), "CASH", [2024], "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"))
    assert restricted.basis("CASH") == "including restricted cash"
    assert run_one(restricted, "A.02")[0].raw_value == 0


def test_employee_plan_shares_bought_for_cash_are_counted_once():
    data = series(overrides={"EQUITY_ISSUED": {2024: 30.0}, "EQUITY_ISSUED_ESPP": {2024: 30.0}})
    data["EQUITY"][2024] += 30.0
    (s,) = run_one(ctx_of(companyfacts(data)), "A.04")
    assert abs(s.raw_value) < 1e-3  # the proceeds and the statement's plan line are the same 30


def test_equity_and_eps_use_the_parents_share_of_income():
    # net income tagged including noncontrolling interests (ProfitLoss): equity and EPS belong to the parent
    data = series(overrides={"NCI_INCOME": {y: 15.0 for y in range(2017, 2025)}})
    cf = switch(companyfacts(data), "NET_INCOME", range(2017, 2025), "ProfitLoss")
    for r in cf["facts"]["us-gaap"]["ProfitLoss"]["units"]["USD"]:
        r["val"] += 15e6
    ctx = ctx_of(cf)
    assert ctx.basis("NET_INCOME") == "including noncontrolling interests"
    assert ctx.ni_parent() == pytest.approx(ctx.a("NET_INCOME") - 15e6)
    assert abs(run_one(ctx, "A.05")[0].raw_value) < 1e-3
    assert abs(run_one(ctx, "A.12")[0].raw_value) < 0.01
    del cf["facts"]["us-gaap"]["NetIncomeLossAttributableToNoncontrollingInterest"]
    with pytest.raises(SkipDetector):
        run_one(ctx_of(cf), "A.04")


def _ni_with_small_nci() -> dict:
    """Net income tagged only including noncontrolling interests (ProfitLoss), whose share is 5 a year (about 2%)."""
    cf = switch(companyfacts(series(overrides={"NCI_INCOME": {y: 5.0 for y in range(2017, 2025)}})), "NET_INCOME", range(2017, 2025), "ProfitLoss")
    for r in cf["facts"]["us-gaap"]["ProfitLoss"]["units"]["USD"]:
        r["val"] += 5e6
    return cf


def test_stated_net_income_is_compared_with_the_parents_share_too():
    from ffi.detectors.e_text import _release_period_totals
    from ffi.text.document import build_doc

    ctx = ctx_of(_ni_with_small_nci())
    parent = ctx.ni_parent()
    # MD&A states the parent's net income, 2% below the total the company tagged: no revised figure, no mismatch
    body = ["Item 7. Management's Discussion and Analysis", f"Net income was ${parent / 1e6:.1f} million for the year."]
    body += [f"Discussion paragraph {n} about the results of operations for the year and the business." for n in range(40)]
    ctx.docs = [build_doc("a", "10-K", "2024-12-31", "", "".join(f"<p>{t}</p>" for t in body))]
    (s,) = run_one(ctx, "A.15")
    assert s.raw_value == 0 and "1 company-wide" in s.rationale  # the heading on the line above names no segment
    from ffi.detectors.textnum import qualified

    for text in ("Results of Operations\nNet income was", "Sales fell in Europe. Net income was"):
        assert not qualified(text, text.index("Net income"))
    assert qualified("Data Center revenue was", 12) and qualified("U.S. revenue was", 5)  # "U.S." ends no sentence
    # the fourth-quarter release's full-year figures are compared with the parent's share as well
    assert parent in _release_period_totals(ctx, pd.Timestamp("2025-02-05"))["NET_INCOME"]["FY"]


def test_a_switch_of_gross_ppe_concept_is_not_unexplained_additions():
    cf = switch(companyfacts(series()), "PPE_GROSS", [2024],
                "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetBeforeAccumulatedDepreciationAndAmortization")
    with pytest.raises(SkipDetector, match="different bases"):
        run_one(ctx_of(cf), "A.08")


def test_a_useful_life_year_on_another_ppe_basis_is_left_out_not_the_test():
    # Intel: gross PP&E includes finance-lease assets from FY2023, not in FY2022; the three later years are still compared
    cf = switch(companyfacts(series()), "PPE_GROSS", [2022, 2023, 2024],
                "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetBeforeAccumulatedDepreciationAndAmortization")
    (s,) = run_one(ctx_of(cf), "C.EXP.03")
    assert "prior 2 years" in s.rationale


def test_structural_model_names_a_balance_on_another_basis():
    from ffi.detectors.d_stats import d04

    # Shopify: receivables tagged as all current receivables, SG&A as general and administrative expense only
    cf = switch(switch(companyfacts(series()), "AR", range(2017, 2025), "ReceivablesNetCurrent"),
                "SGA", range(2017, 2025), "GeneralAndAdministrativeExpense")
    ctx = ctx_of(cf)
    ctx.xsec = pd.DataFrame({"cik": [9], "REVENUE": [1.0], "COGS": [0.5], "GROSS_PROFIT": [0.5], "TOTAL_ASSETS": [1.0],
                             "AR": [0.1], "INVENTORY": [0.1], "SGA": [0.1]})
    with pytest.raises(SkipDetector, match="receivables and sg&a reported on another basis"):
        d04(ctx)


def test_a_revision_later_reversed_is_compared_with_the_latest_report():
    # revenue for 2022: first reported, re-reported 4% higher a year later, then back to the first figure in the third report
    cf = companyfacts(series(), restated={("REVENUE", 2022): 1150.0})
    assert [s for s in run_one(ctx_of(cf), "A.14") if "Revenue for FY2022" in s.rationale]
    rows = cf["facts"]["us-gaap"]["Revenues"]["units"]["USD"]
    first = next(r for r in rows if r["end"] == "2022-12-31" and r["filed"] == "2023-02-20")
    rows.append({**first, "accn": "0000000000-25-000001", "filed": "2025-02-20"})
    assert not [s for s in run_one(ctx_of(cf), "A.14") if "Revenue for FY2022" in s.rationale]


# ------------------------------------------------------------------ ratios across years and peers
def test_a_ratio_across_a_switch_of_receivables_concept_is_not_compared():
    # receivables with notes and loans in 2024, trade receivables in 2023: the change in receivables is the tagging
    cf = switch(companyfacts(series()), "AR", [2024], "AccountsNotesAndLoansReceivableNetCurrent")
    with pytest.raises(SkipDetector, match="different bases"):
        run_one(ctx_of(cf), "C.REV.01")


def test_two_year_averages_need_both_years():
    from ffi.detectors.scores import V

    ctx = ctx_of(companyfacts(series(years=range(2023, 2025))))
    v = V(ctx.A, ctx.focus)
    assert np.isfinite(v.avg("TOTAL_ASSETS")) and np.isnan(v.avg("TOTAL_ASSETS", 1)) and np.isnan(ctx.avg("TOTAL_ASSETS", 1))


def test_working_capital_change_with_a_line_in_one_year_only_is_unknown():
    from ffi.detectors.b_scores import _dwc

    data = series(overrides={"ACCRUED_LIABILITIES": {2024: 80.0}})
    ctx = ctx_of(companyfacts(data))
    assert np.isnan(_dwc(ctx.A, ctx.focus)) and np.isfinite(_dwc(ctx.A, ctx.periods[-2]))


def test_peers_with_payables_on_another_basis_are_left_out():
    from ffi.detectors.d_stats import ratio_with_peers
    from ffi.peers.select import Peer, PeerSet

    ctx = ctx_of(companyfacts(series()))
    trade = FactStore.from_companyfacts(2, companyfacts(series())).annual("first")
    broad = FactStore.from_companyfacts(3, switch(companyfacts(series()), "AP", range(2017, 2025),
                                                  "AccountsPayableAndAccruedLiabilitiesCurrent")).annual("first")
    ctx.peers = PeerSet([Peer(2, "trade", 0.0, trade), Peer(3, "broad", 0.0, broad)], 4, 2, "test")
    mine, peers = ratio_with_peers(ctx, "dpo")  # Coca-Cola: payables with accrued liabilities, DPO 294 days against 62
    assert len(peers) == 1 and peers[0] == pytest.approx(mine)
    assert len(ratio_with_peers(ctx, "dso")[1]) == 2  # receivables are on one basis at all three companies


def test_allowance_on_all_receivables_is_not_coverage_or_a_roll_forward_of_current_ones():
    data = series(overrides={"ALLOWANCE": {y: 5.0 for y in range(2017, 2025)}, "PROVISION_DOUBTFUL": {2024: 1.0},
                             "WRITE_OFFS": {2024: 1.0}})
    ctx = ctx_of(switch(companyfacts(data), "ALLOWANCE", [2024], "AccountsReceivableAllowanceForCreditLoss"))
    with pytest.raises(SkipDetector, match="non-current"):
        run_one(ctx, "C.AR.02")
    with pytest.raises(SkipDetector, match="different bases"):
        run_one(ctx, "A.09")


# ------------------------------------------------------------------ text read by the AI
def test_an_estimate_is_compared_only_with_the_same_estimate():
    from ffi.detectors.e_text import Estimate, Estimates, e02
    from ffi.text.document import build_doc

    def est(desc, low):
        return Estimate(name="useful life", description=desc, value_low=low, value_high=low, unit="years",
                        quote="Buildings are depreciated over their estimated useful lives of 40 years.")
    body = "".join(f"<p>{t}</p>" for t in ["Item 7. Management's Discussion and Analysis",
                                            "Buildings are depreciated over their estimated useful lives of 40 years."]
                   + [f"Paragraph {n} of the discussion of results." for n in range(30)])
    ctx = ctx_of(companyfacts(series()))
    ctx.docs = [build_doc("p", "10-K", "2023-12-31", "", body), build_doc("c", "10-K", "2024-12-31", "", body)]
    answers = iter([Estimates(estimates=[est("Buildings", 40), est("Machinery and equipment", 7)]),
                    Estimates(estimates=[est("Equipment and machinery", 7), est("Structures", 40)])])
    ctx.llm = NS(structured=lambda *a, **k: next(answers))
    (s,) = e02(ctx)  # buildings (40 years) were compared with equipment (7 years) when the descriptions differed
    assert s.severity == 0 and "none changed" in s.rationale


def test_mdna_claims_about_a_quarter_or_an_expectation_are_not_checked():
    from ffi.detectors.e_text import Claim, Claims, e04
    from ffi.text.document import build_doc

    quotes = ["Fourth quarter net sales decreased compared to the fourth quarter of 2023.",
              "We expect total net sales to decrease in 2025 compared to 2024."]
    body = ["Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations", *quotes]
    body += [f"Discussion paragraph {n} about the results of operations for the year." for n in range(60)]
    ctx = ctx_of(companyfacts(series()))  # revenue rose 5% in 2024
    ctx.docs = [build_doc("a", "10-K", "2024-12-31", "", "".join(f"<p>{t}</p>" for t in body))]
    ctx.llm = NS(structured=lambda *a, **k: Claims(claims=[Claim(metric="revenue", direction="decrease", driver="", quote=q)
                                                           for q in quotes]))
    (s,) = e04(ctx)
    assert s.raw_value == 0 and "0 verifiable MD&A claims checked" in s.rationale


def test_amounts_read_by_the_ai_must_be_stated_at_their_scale():
    from ffi.detectors.textnum import states_amount, states_percent

    quote = "Sales to Walmart were $5.2 billion, or 13% of net sales, in fiscal 2024."
    assert states_amount(quote, 5.2e9) and not states_amount(quote, 5.2e6) and not states_amount(quote, 5.2)
    assert states_percent(quote, 13) and not states_percent(quote, 31)
    assert states_amount("a minimum consolidated net worth of $500,000,000", 5e8)


def test_covenant_leverage_needs_ebitda_and_net_worth_its_stated_scale():
    from ffi.detectors.e_text import Covenant, Covenants, covenant_signal
    from ffi.text.document import build_doc

    lev = "A financial covenant requires a maximum leverage ratio of 3.50 to 1.00 under the revolving facility."
    nw = "Another covenant requires a minimum consolidated net worth of $500 million under the revolving facility."
    ctx = ctx_of(companyfacts({k: v for k, v in series().items() if k not in ("DA",)}))
    ctx.docs = [build_doc("a", "10-K", "2024-12-31", "", f"<p>{lev}</p><p>{nw}</p>")]
    ctx.llm = NS(structured=lambda *a, **k: Covenants(covenants=[Covenant(metric="max_leverage_debt_to_ebitda", threshold=3.5, quote=lev),
                                                                 Covenant(metric="min_net_worth", threshold=500.0, quote=nw)]))
    (s,) = covenant_signal(ctx)  # no D&A: no EBITDA; a $500 million floor read as 500 is no floor
    assert "no numeric thresholds" in s.rationale


def test_counterparty_stress_counts_only_filings_up_to_the_years_cutoff(monkeypatch):
    from ffi.detectors import f_graph
    from ffi.edgar import endpoints

    ctx = ctx_of(companyfacts(series()))
    ctx.cutoff, ctx.client = date(2025, 4, 30), object()
    ctx.notes.append("F.01 counterparty Retailer -> CIK 7")
    sub = {"filings": {"recent": {"form": ["NT 10-K", "8-K"], "items": ["", "4.02"], "filingDate": ["2026-03-31", "2019-01-10"]}}}
    monkeypatch.setattr(endpoints, "submissions", lambda client, cik: sub)
    (s,) = f_graph.f06(ctx)  # a late filing two years later and a notice from 2019 are outside the window
    assert s.raw_value == 0


# ------------------------------------------------------------------ AI text checks and report wording
def test_a_money_figure_is_never_supported_at_a_hundred_times_another_value():
    from ffi.agent import grounding

    assert not grounding.check("The gap is $1.9B.", [19e6], []).ok
    assert grounding.check("The gap is $1.9B.", [1.94e9], []).ok
    assert grounding.check("The ratio fell to 13.9.", [0.139], []).ok  # a bare number may be a percentage


def test_linter_keeps_ordinary_lies_and_paragraph_breaks():
    from ffi.report.linter import lint

    text, hits = lint("The difference lies in the timing of shipments.\n\nNo wording here is accusatory.")
    assert hits == [] and text == "The difference lies in the timing of shipments.\n\nNo wording here is accusatory."
    assert lint("Management lied to auditors.")[1] == ["lied"]


def test_a_signal_shared_by_two_patterns_is_counted_once_in_the_index():
    from ffi.detectors.base import Area, Signal
    from ffi.scoring.engine import score

    def sig(did, fam, sev=0.8):
        return Signal(did, did, fam, Area.REVENUE, "FY2024", 1.0, 0.0, sev, True, "")

    # operating cash flow far below net income is required by premature revenue (with a customer-capacity result) and by
    # accrual-heavy earnings (with discretionary accruals): both patterns form, and their scores show it
    ctx = ctx_of(companyfacts(series()))
    sc = score(ctx, [NS(signals=[sig("C.CF.01", "C"), sig("F.01", "F"), sig("B.03", "B")])])
    assert {f.id for f in sc.findings if f.kind == "pattern"} == {"premature_revenue", "accrual_heavy_earnings"}
    # in the index the shared result enters each pattern with half its weight: held once between them, not twice
    contrib, own = dict(sc.contributions), 1 - 0.85 * 0.8
    assert contrib["pattern:premature_revenue"] == pytest.approx(0.8 * (1 - own ** 0.5 * own))
    assert all(f.score == pytest.approx(1 - own * own) for f in sc.findings)


def test_an_explained_finding_counts_for_half_in_the_index():
    from ffi.scoring.engine import EXPLAINED, ScoreResult, apply_explained

    f = NS(id="premature_revenue", kind="pattern", narrative_source=EXPLAINED, signals=[])
    sc = ScoreResult(0, "Low", None, [f], {}, {}, [], [], {}, [("pattern:premature_revenue", 0.8), ("C.GOV.03", 0.2)])
    apply_explained(sc)
    assert dict(sc.contributions)["pattern:premature_revenue"] == pytest.approx(0.4) and dict(sc.contributions)["C.GOV.03"] == 0.2


def test_a_discount_is_judged_against_the_tests_own_flag_level():
    from ffi.detectors.base import Area, Signal
    from ffi.scoring.engine import score

    ctx = ctx_of(companyfacts(series(overrides={"ACQUISITIONS": {2024: 500.0}})))
    s = Signal("D.02", "D.02", "D", Area.EARNINGS_QUALITY, "FY2024", 1.0, 0.0, 0.9, True, "", flag_at=0.4)
    score(ctx, [NS(signals=[s])])  # 0.9 x 0.5 = 0.45 is still above this test's flag level of 0.4
    assert s.flagged and s.severity == pytest.approx(0.45)


def test_a_saved_run_keeps_the_year_the_company_changed_shape():
    from ffi.report.snapshot import _lighten

    ctx = ctx_of(companyfacts(series(), restated={("REVENUE", 2022): 500.0}))
    assert ctx.perimeter_break == pd.Timestamp("2022-12-31")
    light = _lighten({})(ctx)
    assert light.store is None and light.perimeter_break == pd.Timestamp("2022-12-31")


def test_key_figures_state_their_basis():
    from ffi.report.model import DETECTOR_RATIO, key_value

    cf = switch(companyfacts(series()), "NET_INCOME", [2024], "ProfitLoss")
    ctx = ctx_of(cf)
    assert key_value(ctx, "NET_INCOME").endswith("(including noncontrolling interests)") and "(" not in key_value(ctx, "NET_INCOME", 1)
    assert "C.REV.01" not in DETECTOR_RATIO  # its chart showed receivables growth, another comparison than the test's


def test_a_year_with_no_first_reported_figures_is_not_analysed():
    from ffi.pipeline.run import MissingYear, require_first_reported

    # S&P Global's SEC data holds FY2024 only in the 2026 report (restated): as first reported by 2025-04-30 it has none, and
    # the year used to be read as the one before it (FY2024 showed FY2023's revenue)
    cf = companyfacts(series(years=range(2022, 2025)))
    for unit_rows in cf["facts"]["us-gaap"].values():
        for rows in unit_rows["units"].values():
            rows[:] = [r for r in rows if not (r["end"].startswith("2024") and r["filed"].startswith("2025-02"))]
    store = FactStore.from_companyfacts(1, cf, date(2025, 4, 30))
    A = store.annual("first")
    with pytest.raises(MissingYear, match="FY2024"):
        require_first_reported(A, pd.Timestamp("2024-12-31"), date(2025, 4, 30))
    require_first_reported(A, pd.Timestamp("2023-12-31"), date(2025, 4, 30))


def test_chart_amounts_are_in_the_reporting_currency():
    from ffi.detectors.stats import CURRENCY
    from ffi.report.charts import _fmt

    token = CURRENCY.set("EUR")  # a euro reporter's (ASML's) charts put a dollar sign on euro amounts
    try:
        assert _fmt(9.6e9, "money") == "€9.6B" and _fmt(-250e6, "money") == "-€250M"
    finally:
        CURRENCY.reset(token)


def test_the_trained_model_loads_from_a_checkout_with_windows_line_endings(tmp_path, monkeypatch):
    # git with autocrlf converted the shipped model to CRLF in a fresh Windows checkout, and LightGBM's parser aborted the run
    from ffi import models_store

    for name in ("misstatement.lgb.txt", "misstatement.json"):
        data = (models_store.MODELS_DIR / name).read_bytes().replace(b"\r\n", b"\n")
        (tmp_path / name).write_bytes(data.replace(b"\n", b"\r\n") if name.endswith(".txt") else data)
    monkeypatch.setattr(models_store, "MODELS_DIR", tmp_path)
    model, meta = models_store.load_model("misstatement")
    assert model.num_trees() > 0 and len(meta["features"]) == model.num_feature()


def test_a_typed_budget_that_is_not_a_number_does_not_fail_the_case():
    from ffi.agent.ui import TerminalUI

    answers = iter(["r", "two dollars"])
    ui = TerminalUI(NS(input=lambda *a: next(answers), print=lambda *a, **k: None), mode="interactive")
    assert ui._ask("review_budget", {"summary": ""}) == {"choice": "raise", "auto": False, "amount": 2.0}
