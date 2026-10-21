"""Regression tests for false findings found by the wide sweep (42 companies): each uses the layout or wording of the
filing that exposed the bug."""

from __future__ import annotations

import pandas as pd
import pytest

from ffi.detectors.base import SkipDetector
from tests.conftest import make_ctx, run_one, series


def flagged(sigs):
    return any(s.flagged for s in sigs)


def test_a01_untagged_redeemable_interests_are_not_a_broken_balance_sheet():
    # BlackRock 2025: assets 169,998 = liabilities 108,456 + temporary equity 5,427 (2,636 + 2,791, the second a company
    # extension) + permanent equity 56,115. Only 'Liabilities and equity' (169,998) and the parts are tagged in companyfacts.
    ctx = make_ctx(series(overrides={"EQUITY": {2024: 450.0}, "TOTAL_LIABILITIES": {2024: 700.0}}))  # parts fall 50 short of the filed total, which ties to assets
    assert not flagged(run_one(ctx, "A.01"))
    # parts that exceed the filed total are still a break
    ctx = make_ctx(series(overrides={"EQUITY": {2024: 1000.0}, "TOTAL_LIABILITIES": {2024: 700.0}}))
    assert flagged(run_one(ctx, "A.01"))


def test_sloan_accrual_ratio_ignores_cash_paid_for_acquisitions():
    from ffi.detectors import scores as S

    plain = make_ctx(series())
    bought = make_ctx(series(overrides={"ACQUISITIONS": {y: 500.0 for y in range(2017, 2025)}}))
    assert bought.a("ACQUISITIONS") > 0
    # Johnson & Johnson 2025: net income 26.8B, operating cash flow 24.5B, capex 4.8B, acquisitions 17.5B on 98B of net
    # operating assets read as a 25% accrual ratio; the acquisitions are no accrual
    assert S.sloan_accruals(bought.A, bought.focus) == S.sloan_accruals(plain.A, plain.focus)


def test_sloan_accrual_ratio_has_no_substitute_measure():
    import math

    from ffi.detectors import scores as S

    base = series()
    assert math.isfinite(S.sloan_accruals(make_ctx(base).A, make_ctx(base).focus))
    # Toyota's earlier years tag no capex: total investing cash flow (securities purchases included) was used instead,
    # and the history the focus year was compared with mixed the two measures
    no_capex = {k: v for k, v in base.items() if k != "CAPEX"}
    c = make_ctx(no_capex)
    assert math.isnan(S.sloan_accruals(c.A, c.focus))
    # net operating assets not positive: total assets in their place was another denominator
    c = make_ctx(series(overrides={"TOTAL_LIABILITIES": {y: 10_000_000.0 for y in range(2017, 2025)}}))
    assert math.isnan(S.sloan_accruals(c.A, c.focus))


def test_a02_short_term_investments_inside_cash_are_not_double_counted():
    # Target FY2025 (year ended 2026-01-31): current assets 20,005; cash 5,488 (which includes short-term investments of 4,611);
    # inventory 12,304. Adding the 4,611 again gave a 2.4B "excess".
    data = series(overrides={"CURRENT_ASSETS": {2024: 20_005.0}, "CASH": {2024: 5_488.0}, "ST_INVESTMENTS": {2024: 4_611.0},
                             "INVENTORY": {2024: 12_304.0}, "AR": {2024: 0.0}})
    ctx = make_ctx(data)
    assert run_one(ctx, "A.02")[0].raw_value == 0
    # parts that exceed current assets even without the investments are still a break
    ctx = make_ctx(series(overrides={"CURRENT_ASSETS": {2024: 10_000.0}, "CASH": {2024: 5_488.0}, "ST_INVESTMENTS": {2024: 4_611.0},
                                     "INVENTORY": {2024: 12_304.0}, "AR": {2024: 0.0}}))
    assert run_one(ctx, "A.02")[0].raw_value > 0


def test_a15_year_ended_in_february_is_checked_against_both_fiscal_year_names():
    import pandas as pd

    from ffi.detectors.textnum import check_amounts, totals_by_year

    # Home Depot 10-K for the year ended 2026-02-01 (its fiscal 2025): "net sales of $164.7 billion", up from $159.5 billion
    years = totals_by_year([(pd.Timestamp("2026-02-01"), 164.7e9), (pd.Timestamp("2025-02-02"), 159.51e9),
                            (pd.Timestamp("2024-01-28"), 152.67e9)])
    known = {"REVENUE": {"years": years, "focus": 2026}}
    text = "Fiscal 2025 total net sales of $164.7 billion increased 3.3% from fiscal 2024 net sales of $159.5 billion."
    assert check_amounts(text, known)[1] == []  # read as the 2025 year-end's total ($159.5B) it looked like a revised number
    assert check_amounts("Net sales of $164.7 billion rose.", known) == (1, [])
    # a stated figure that is near the year but wrong is still a mismatch
    assert len(check_amounts("In fiscal 2025, net sales of $166.0 billion were reported.", known)[1]) == 1


def _equity_jump(**extra):
    """The clean synthetic company with equity 500m above what its income and cash flows explain in the last year."""
    base = series()
    eq = {2024: base["EQUITY"][2024] + 500.0}
    return make_ctx(series(overrides={"EQUITY": eq, **{k: {2024: v} for k, v in extra.items()}}))


def test_a04_shares_issued_for_acquisitions_explain_the_equity_movement():
    # Boeing 2025: Spirit AeroSystems paid for with $4.7B of stock (StockIssuedDuringPeriodValueAcquisitions); Disney 2019: 21st
    # Century Fox; Uber 2019: preferred stock converted at the IPO
    assert flagged(run_one(_equity_jump(), "A.04"))
    for item in ("EQUITY_ISSUED_ACQ", "EQUITY_CONVERTED", "EQUITY_ISSUED_OTHER"):
        s = run_one(_equity_jump(**{item: 500.0}), "A.04")[0]
        assert abs(s.raw_value) < 1.0, item


def test_a04_tax_withheld_on_net_share_settlement_is_an_equity_outflow():
    # Snowflake FY2026: equity fell $0.56B more than net income, buybacks, SBC and option proceeds explain; the missing
    # outflow was $672.9M of taxes paid on net share settlement
    base = series()
    ctx = make_ctx(series(overrides={"EQUITY": {2024: base["EQUITY"][2024] - 500.0}, "SHARE_TAX_WITHHELD": {2024: 500.0}}))
    assert abs(run_one(ctx, "A.04")[0].raw_value) < 1.0


def test_a08_finance_leases_and_unpaid_capex_are_not_unexplained_additions():
    # Oracle FY2026: gross PP&E +63.1B, cash capex 55.7B; finance-lease additions 4.9B and capex accrued but unpaid 5.3B (1.4B a year earlier)
    base = series()
    grow = {2024: base["PPE_GROSS"][2023] + base["CAPEX"][2024] + 300.0}
    ctx = make_ctx(series(overrides={"PPE_GROSS": grow}))
    assert run_one(ctx, "A.08")[0].raw_value > 250.0
    ctx = make_ctx(series(overrides={"PPE_GROSS": grow, "FINANCE_LEASE_ASSETS_ADDED": {2024: 200.0},
                                     "CAPEX_UNPAID": {2023: 50.0, 2024: 150.0}}))
    assert abs(run_one(ctx, "A.08")[0].raw_value) < 1.0


def test_a12_eps_uses_income_after_preferred_dividends():
    # Boeing 2025: net income 2,235, preferred dividends 345, net income for diluted EPS 1,890, diluted shares 762.3, EPS 2.48
    ctx = make_ctx(series(overrides={"NET_INCOME": {2024: 2235.0}, "NI_COMMON_DILUTED": {2024: 1890.0},
                                     "SHARES_DILUTED": {2024: 762.3}, "EPS_DILUTED": {2024: 2.48}, "EPS_BASIC": {2024: 2.49}}))
    s = run_one(ctx, "A.12")[0]
    assert not s.flagged and abs(s.raw_value) < 0.01
    ctx = make_ctx(series(overrides={"NET_INCOME": {2024: 2235.0}, "SHARES_DILUTED": {2024: 762.3}, "EPS_DILUTED": {2024: 2.48},
                                     "EPS_BASIC": {2024: 2.49}}))
    assert flagged(run_one(ctx, "A.12"))  # without the preferred dividends tagged the old reading remains


def _build_out(with_cip: bool):
    """A build-out in the last year: 400m of construction in progress, which carries no depreciation yet."""
    base = series()
    gross, net = base["PPE_GROSS"][2024] + 400.0, base["PPE_NET"][2024] + 400.0
    over = {"PPE_GROSS": {2024: gross}, "PPE_NET": {2024: net}}
    if with_cip:
        over["PPE_CIP"] = {y: (400.0 if y == 2024 else 0.0) for y in range(2017, 2025)}
    return make_ctx(series(overrides=over))


def test_c_exp03_construction_in_progress_does_not_lengthen_the_useful_life():
    # Oracle FY2026: gross PP&E 122.7B of which 40.0B construction in progress and 1.3B land; the implied life read 16.1
    # years against 11.4 although the depreciable assets lasted 10.7 years against 9.3
    assert flagged(run_one(_build_out(False), "C.EXP.03"))
    s = run_one(_build_out(True), "C.EXP.03")[0]
    assert not s.flagged and "excluding construction in progress" in s.rationale


def test_c_exp06_sga_is_selling_plus_general_when_they_are_reported_apart():
    from types import SimpleNamespace as NS

    from ffi.detectors.c_accounts import _spend

    # Microsoft FY2026: the SG&A tag found is GeneralAndAdministrativeExpense (7,956); sales and marketing is its own line (26,000)
    def v(general_only: bool, sm: float):
        vals = {"SGA": 7_956.0, "RND": 35_562.0, "SELLING_MARKETING": sm}
        concept = "GeneralAndAdministrativeExpense" if general_only else "SellingGeneralAndAdministrativeExpense"
        return NS(g=lambda item, b=0: vals.get(item, float("nan")), concept=lambda item, b=0: concept if item == "SGA" else None)

    assert _spend(v(True, 26_000.0), ("SGA", "RND")) == 7_956.0 + 26_000.0 + 35_562.0
    assert _spend(v(False, 26_000.0), ("SGA", "RND")) == 7_956.0 + 35_562.0  # a full SG&A tag already includes selling
    assert _spend(v(True, float("nan")), ("SGA", "RND")) != _spend(v(True, float("nan")), ("SGA", "RND"))  # partial: NaN, not compared


def test_a14_recast_with_unchanged_net_income_is_a_presentation_change():
    # General Electric FY2023 revenue: $67.95B first reported, $35.35B after the spin-offs; Johnson & Johnson (Kenvue) the same
    # way: net income of the period is unchanged in both reports
    data = series()
    ctx = make_ctx(data, restated={("REVENUE", 2023): data["REVENUE"][2023] * 0.5})
    s = next(x for x in run_one(ctx, "A.14") if "REVENUE" in x.rationale.upper() or "revenue" in x.rationale)
    assert not s.flagged and "presentation_recast" in s.tags and "same in both reports" in s.rationale
    # net income restated as well: remains a flagged revision
    ctx = make_ctx(data, restated={("REVENUE", 2023): data["REVENUE"][2023] * 0.5, ("NET_INCOME", 2023): data["NET_INCOME"][2023] * 0.5})
    assert any(x.flagged and "unlabeled_revision" in x.tags for x in run_one(ctx, "A.14"))


def test_a14_disclosed_recast_on_adopting_a_standard_is_not_flagged():
    from ffi.text.document import build_doc

    data = series()
    ctx = make_ctx(data, restated={("NET_INCOME", 2023): data["NET_INCOME"][2023] * 0.8})
    quote = ("We adopted ASU 2018-12 on January 1, 2025 using the full retrospective method; the amounts for 2023 and 2022 "
             "have been recast to reflect the new accounting standard.")
    ctx.docs = [build_doc("0000000000-25-000001", "10-K", "2024-12-31", "u", f"<p>Item 7. Management's Discussion</p><p>{quote}</p>")]
    s = next(x for x in run_one(ctx, "A.14") if "Net income" in x.rationale)
    assert not s.flagged and "disclosed_recast" in s.tags and "retrospective" in s.rationale
    ctx.docs = []
    assert any(x.flagged for x in run_one(ctx, "A.14"))
    # Bank of America FY2025: "elected to change its accounting methods ... applied retrospectively" for tax-related equity
    # investments moved pre-tax income of FY2023 and FY2024 by 14% without a standard to name
    bac = ("Effective in the fourth quarter of 2025, the Corporation elected to change its accounting methods for certain tax-related "
           "equity investments, which were applied on a retrospective basis. The primary impact is a reclassification.")
    ctx.docs = [build_doc("0000000000-25-000001", "10-K", "2024-12-31", "u", f"<p>Item 7. Management's Discussion</p><p>{bac}</p>")]
    s = next(x for x in run_one(ctx, "A.14") if "Net income" in x.rationale)
    assert "disclosed_recast" in s.tags


def test_a05_retained_earnings_change_the_annual_report_describes_is_discounted():
    from ffi.text.document import build_doc

    base = series()
    # Disney FY2019: retained earnings fell $40.2B against $8.2B of income less dividends because treasury shares were retired
    ctx = make_ctx(series(overrides={"RETAINED_EARNINGS": {2024: base["RETAINED_EARNINGS"][2024] - 1000.0}}))
    plain = run_one(ctx, "A.05")[0]
    assert plain.flagged
    text = "During fiscal 2019, the Company retired 1.1 billion treasury shares, which reduced retained earnings."
    ctx.docs = [build_doc("a", "10-K", "2024-12-31", "u", f"<p>Item 8. Financial Statements and Supplementary Data</p><p>{text}</p>")]
    s = run_one(ctx, "A.05")[0]
    assert s.severity < plain.severity * 0.5 and "retired 1.1 billion treasury shares" in s.rationale


def test_c_ar02_negative_allowance_tag_is_read_as_its_size():
    # Boeing FY2019 tagged the allowance as -73 on receivables of 3,266 (57 and 62 in earlier years): coverage read -2.3%
    base = series()
    ctx = make_ctx(series(overrides={"ALLOWANCE": {y: -2.0 if y == 2024 else 2.0 for y in range(2017, 2025)}}))
    s = run_one(ctx, "C.AR.02")[0]
    assert s.raw_value > 0 and not s.flagged
    assert base["AR"][2024] > 0


def test_a02_gross_profit_below_revenue_less_cogs_is_fees_outside_gross_profit():
    # Costco FY2019: revenue 152.7B, cost of sales 123.2B, tagged gross profit 26.1B (net sales less merchandise costs);
    # the 3.35B gap is membership fees
    ctx = make_ctx(series(overrides={"GROSS_PROFIT": {2024: series()["REVENUE"][2024] - series()["COGS"][2024] - 20.0}}))
    assert run_one(ctx, "A.02")[0].raw_value == 0
    ctx = make_ctx(series(overrides={"GROSS_PROFIT": {2024: series()["REVENUE"][2024] - series()["COGS"][2024] + 200.0}}))
    assert run_one(ctx, "A.02")[0].raw_value != 0  # gross profit above revenue less cost of sales is still a break


def test_ticker_moved_to_a_holding_company_finds_the_predecessor_despite_camel_case_name(monkeypatch):
    import pandas as pd

    from ffi.edgar import endpoints
    from ffi.resolve import resolver

    # XOM now belongs to "ExxonMobil Holdings Corp", which has filed no annual report; EDGAR's name search finds nothing
    # for "exxonmobil" but finds EXXON MOBIL CORP for "exxon mobil"
    tickers = pd.DataFrame({"cik": [9999], "ticker": ["XOM"], "title": ["ExxonMobil Holdings Corp"]})
    monkeypatch.setattr(resolver, "company_tickers_frame", lambda client: tickers)
    subs = {9999: {"name": "ExxonMobil Holdings Corp", "tickers": ["XOM"], "filings": {"recent": {"form": ["8-K"]}}},
            34088: {"name": "EXXON MOBIL CORP", "tickers": [], "sic": "2911",
                    "filings": {"recent": {"form": ["10-K", "10-K", "8-K"]}}}}
    monkeypatch.setattr(endpoints, "submissions", lambda client, cik: subs[cik])
    monkeypatch.setattr(endpoints, "company_search", lambda client, q: [(34088, "EXXON MOBIL CORP")] if q == "exxon mobil" else [])
    ref = resolver.resolve("XOM", object())
    assert ref.cik == 34088 and ref.ticker == "XOM" and any("predecessor" in w for w in ref.warnings)


def test_c_rev02_rationale_says_rose_when_deferred_revenue_rose():
    # Rivian FY2025: "deferred revenue fell from 37.0% to 43.5% of revenue" although it rose
    base = series()
    over = {"DEFERRED_REV_CURRENT": {2023: 100.0, 2024: 600.0}}
    ctx = make_ctx(series(overrides={"DEFERRED_REV_CURRENT": {y: 20.0 + 5 * (y - 2017) for y in range(2017, 2023)} | over["DEFERRED_REV_CURRENT"],
                                     "REVENUE": {2024: base["REVENUE"][2024]}}))
    s = run_one(ctx, "C.REV.02")[0]
    assert " rose from " in s.rationale and " fell from " not in s.rationale


def test_assess_ignores_a_ratio_within_a_tenth_of_its_baseline():
    # Caterpillar FY2025: contract assets 0.439% of revenue, own median 0.4%, peer median 0.445%, flagged at 0.55 because the
    # history is so tight that a 1% wobble is several standard deviations
    base = series()
    wobble = [1.000, 1.004, 0.996, 1.003, 0.998, 1.002, 0.999]
    ca = {y: 0.04 * base["REVENUE"][y] * wobble[y - 2017] for y in range(2017, 2024)}
    flat = make_ctx(series(overrides={"CONTRACT_ASSETS": {**ca, 2024: 0.04 * base["REVENUE"][2024] * 1.07}}))
    assert not flagged(run_one(flat, "C.REV.03"))
    jump = make_ctx(series(overrides={"CONTRACT_ASSETS": {**ca, 2024: 0.04 * base["REVENUE"][2024] * 1.5}}))
    assert flagged(run_one(jump, "C.REV.03"))


def test_c_exp06_steady_decline_in_overhead_share_is_scale_not_a_cut():
    # Snowflake FY2026: SG&A + R&D 97.8% of revenue against an own median of 107.0%, after falling every year with growth
    base = series()
    share = {2017: 0.70, 2018: 0.64, 2019: 0.58, 2020: 0.52, 2021: 0.46, 2022: 0.40, 2023: 0.34, 2024: 0.28}
    ctx = make_ctx(series(overrides={"SGA": {y: share[y] * base["REVENUE"][y] for y in share}}))
    s = run_one(ctx, "C.EXP.06")[0]
    assert not s.flagged and "continues that trend" in s.rationale
    steady = {**share, 2017: 0.40, 2018: 0.40, 2019: 0.40, 2020: 0.40, 2021: 0.40, 2022: 0.40, 2023: 0.40, 2024: 0.20}
    ctx = make_ctx(series(overrides={"SGA": {y: steady[y] * base["REVENUE"][y] for y in steady}}))
    assert flagged(run_one(ctx, "C.EXP.06"))  # a sudden halving after years of a flat ratio is still a cut


def test_f04_line_item_keywords_know_the_common_captions():
    import re

    from ffi.kg.build import ITEM_KEYWORDS

    cases = {  # captions that the evidence-coverage test reported as "no discussion" although the MD&A discusses the item
        "OPERATING_INCOME": ["Operating earnings increased 11%", "Income from operations was $2.8 billion"],  # AbbVie, Uber
        "SGA": ["Marketing, general and administrative expenses decreased", "Sales and marketing expenses increased 12%"],  # Intel, Microsoft
        "CAPEX": ["Cash used for additions to property and equipment was $64.6 billion"],  # Microsoft
        "DEFERRED_REV_CURRENT": ["Unearned revenue increased due to billings"],  # Microsoft
        "INCOME_TAX": ["Our provision for income taxes was lower", "Income tax benefit was $4.9 billion"],
    }
    for item, texts in cases.items():
        for t in texts:
            assert re.search(ITEM_KEYWORDS[item], t, re.I), (item, t)


def test_c_gov02_part_iii_amendment_with_cover_page_data_did_not_refile_statements():
    from datetime import date

    import pandas as pd

    from ffi.context import Event

    # Disney's 10-K/A of 2023-01-24 and 2024-01-24 added Part III only ("filed solely to amend Part III"); they carry cover-page
    # XBRL (public float), which made them look like re-filed financial statements
    ctx = make_ctx(series())
    df = ctx.store.df
    cover = df[df["item"] == "REVENUE"].iloc[:1].assign(item="PUBLIC_FLOAT", accn="0000000000-99-000009", form="10-K/A")
    ctx.store.df = pd.concat([df, cover], ignore_index=True)
    ctx.events = [Event("e1", "AMENDED_10K", date(2024, 6, 1), "0000000000-99-000009", "Amended annual report (10-K/A)", "u")]
    assert "0 of those day(s) re-filed financial statements" in run_one(ctx, "C.GOV.02")[0].rationale
    full = df[df["item"] == "REVENUE"].iloc[:1].assign(accn="0000000000-99-000010", form="10-K/A")
    ctx.store.df = pd.concat([df, cover, full], ignore_index=True)
    ctx.events = [Event("e2", "AMENDED_10K", date(2024, 6, 1), "0000000000-99-000010", "Amended annual report (10-K/A)", "u")]
    assert "1 of those day(s) re-filed financial statements" in run_one(ctx, "C.GOV.02")[0].rationale


def test_officer_change_kind_ignores_a_new_directors_other_post_and_a_leavers_next_job():
    from ffi.pipeline.run import _officer_change_kind

    # NextEra Energy 8-K of 2024-07-10: a director appointed; "Mr. Martha is chairman and chief executive officer of Medtronic plc"
    director = ("<p>Item 5.02 (d) On July 9, 2024, NextEra Energy, Inc.'s Board approved an increase in the size of the Board and "
                "appointed Geoffrey S. Martha to fill the newly created directorship. Mr. Martha is chairman and chief executive "
                "officer of Medtronic plc, a global healthcare technology company.</p>")
    assert "no senior finance" in _officer_change_kind(director)
    # NextEra Energy 8-K of 2026-03-10: the Chief Risk Officer resigns "to accept a chief financial officer position with another company"
    leaver = ("<p>Item 5.02 (b) On March 5, 2026, Terrell Kirk Crews II, Executive Vice President and Chief Risk Officer of NextEra, Inc. "
              "notified NEE of his decision to resign effective March 20, 2026 in order to accept a chief financial officer position "
              "with another company.</p>")
    assert "no senior finance" in _officer_change_kind(leaver)
    # a real succession is still seen, also when the biography says the new officer served the company
    real = ("<p>Item 5.02 (b) Effective May 6, 2024, Brian Bolster was appointed Chief Financial Officer to succeed Mr. Crews. "
            "Mr. Bolster has served as chief financial officer of the Company's subsidiary.</p>")
    assert "CFO change" in _officer_change_kind(real)


def test_d01_skips_a_fourth_quarter_that_is_the_year_less_recast_quarters():
    from types import SimpleNamespace as NS

    import pandas as pd

    from ffi.detectors.d_stats import implausible_fourth_quarters
    from ffi.xbrl.facts import Panel

    # General Electric: FY2015 revenue (recast for discontinued operations) less the first three quarters as first reported
    # left a "fourth quarter" of 1/20th of their level, a DSO of 951 days
    idx = pd.to_datetime(["2015-03-31", "2015-06-30", "2015-09-30", "2015-12-31", "2016-03-31"])
    vals = pd.DataFrame({"REVENUE": [29.0, 30.0, 31.0, 1.5, 28.0]}, index=idx)
    Q = Panel("Q", vals, {}, derived={("REVENUE", idx[3])})
    assert implausible_fourth_quarters(NS(Q=Q)) == {idx[3]}
    vals.loc[idx[3], "REVENUE"] = 33.0
    assert implausible_fourth_quarters(NS(Q=Q)) == set()


def test_a03_discontinued_operations_cash_flows_complete_the_roll_forward():
    # Disney FY2019: continuing operations 5,984 - 15,096 - 464 and FX -98 = -9,674 against a change in cash of +1,300; the
    # 10,974 from discontinued operations (the sale of the Fox businesses) is its own line
    base = series()
    cash = base["CASH_RESTRICTED_TOTAL"]
    flows = base["CFO"][2024] + base["CFI"][2024] + base["CFF"][2024]
    gap = 800.0
    over = {"CASH_RESTRICTED_TOTAL": {2024: cash[2023] + flows + gap}}
    assert flagged(run_one(make_ctx(series(overrides=over)), "A.03"))
    s = run_one(make_ctx(series(overrides={**over, "CASH_DISCONTINUED": {2024: gap}})), "A.03")[0]
    assert abs(s.raw_value) < 1.0


def test_a04_statement_of_equity_issuance_replaces_cash_proceeds_when_larger():
    # Realty Income 2019: $845M of cash proceeds from stock issued, $2,118M of shares issued in the statement of equity
    base = series()
    jump = {"EQUITY": {2024: base["EQUITY"][2024] + 800.0}}
    ctx = make_ctx(series(overrides={**jump, "EQUITY_ISSUED": {2024: 100.0}, "EQUITY_ISSUED_STMT": {2024: 800.0}}))
    ref = make_ctx(series(overrides={**jump, "EQUITY_ISSUED": {2024: 100.0}}))
    assert abs(run_one(ctx, "A.04")[0].raw_value) < 1.0 and run_one(ref, "A.04")[0].raw_value > 500.0


def test_a10_goodwill_gap_beside_acquisitions_or_divestitures_is_discounted():
    base = series()
    g = {y: 300.0 for y in range(2017, 2025)}
    g[2024] = 700.0
    over = {"GOODWILL": g, "GOODWILL_ACQUIRED": {2024: 100.0}}
    plain = run_one(make_ctx(series(overrides=over)), "A.10")[0]
    bought = run_one(make_ctx(series(overrides={**over, "ACQUISITIONS": {2024: 900.0}})), "A.10")[0]
    assert plain.severity > 0 and bought.severity == plain.severity * 0.5 and "paid for acquisitions" in bought.rationale
    g[2024] = 100.0
    over = {"GOODWILL": g, "GOODWILL_ACQUIRED": {2024: 0.0}}
    plain = run_one(make_ctx(series(overrides=over)), "A.10")[0]
    sold = run_one(make_ctx(series(overrides={**over, "DIVESTITURES": {2024: 400.0}})), "A.10")[0]
    assert sold.severity == plain.severity * 0.5 and base["REVENUE"][2024] > 0


def test_e10_letters_about_a_registration_statement_or_proxy_are_not_annual_report_comments():
    from datetime import date
    from types import SimpleNamespace as NS

    import numpy as np
    import pandas as pd

    from ffi.detectors.base import REGISTRY, load_all
    from ffi.detectors.e_text import letter_subject

    load_all()
    # Boeing 2024: CORRESP letters about the S-4 for Spirit AeroSystems; Disney 2024: the preliminary proxy statement
    s4 = "<p>The Boeing Company</p><p>Registration Statement on Form S-4</p><p>Filed October 2, 2024 File No. 333-282000</p>"
    proxy = "<div>Walt Disney Co</div><div>Preliminary Proxy Statement filed January 16, 2024</div><div>File No. 001-38842</div>"
    tenk = "<p>Re: Acme Corp</p><p>Form 10-K for the Fiscal Year Ended December 31, 2023</p><p>File No. 001-1</p>"
    assert (letter_subject(s4), letter_subject(proxy), letter_subject(tenk)) == ("other", "other", "periodic")

    def fil(form, d, n):
        return NS(form=form, filing_date=d, primary_document="filename1.htm", accession=f"a{n}")

    docs = {"a1": s4, "a2": proxy}
    client = NS(get_text=lambda url: next(v for k, v in docs.items() if url.endswith(k)))
    letters = [fil("UPLOAD", date(2024, 10, 2), 3), fil("UPLOAD", date(2024, 11, 5), 4), fil("UPLOAD", date(2024, 12, 9), 5),
               fil("CORRESP", date(2024, 10, 18), 1), fil("CORRESP", date(2024, 11, 14), 2)]
    ctx = NS(focus=pd.Timestamp("2024-12-31"), as_of=date(2025, 6, 1), filings=letters, client=client, focus_label="FY2024",
             tuning={}, currency="USD", evs=lambda *a, **k: [], doc=None, company=NS(cik=1),
             ev_span=lambda *a, **k: None)
    import ffi.edgar.endpoints as ep

    orig = ep.document_url
    ep.document_url = lambda cik, f: f"https://x/{f.accession}"
    try:
        s = REGISTRY["E.10"].fn(ctx)[0]
    finally:
        ep.document_url = orig
    assert "more concern registration or proxy statements" in s.rationale and np.isfinite(s.severity)
    assert s.severity < 0.5 * 0.75 + 0.01


def test_assess_leaves_out_peers_reporting_an_item_on_another_basis():
    import dataclasses
    from types import SimpleNamespace as NS

    from ffi.detectors.c_accounts import assess
    from ffi.detectors.stats import safe_div

    # Coca-Cola's DPO (payables with accrued liabilities, 294 days) against peers' trade payables (median 62 days)
    base = series()
    peers = []
    for i in range(6):
        c = make_ctx(series(overrides={"AP": {y: base["AP"][y] * (0.8 + 0.1 * i) for y in range(2017, 2025)}}))
        peers.append(NS(annual=c.A, per=c.focus))
    ctx = make_ctx(series())
    ctx.peers = NS(loaded=peers, aligned_period=lambda p, target: p.per)
    fn = lambda v: safe_div(v.g("AP"), v.g("COGS")) * 365  # noqa: E731
    assert len(assess(ctx, fn).peers) == 6
    for p in peers:  # the peers' payables are payables with accrued liabilities, the company's are not
        key = ("AP", p.per)
        p.annual.prov[key] = dataclasses.replace(p.annual.prov[key], concept="AccountsPayableAndAccruedLiabilitiesCurrent")
    assert assess(ctx, fn).peers == []


def test_years_before_a_spin_off_are_not_the_same_company():
    # General Electric FY2025: FY2023 revenue was reported as $67.95B and re-reported as $35.35B (Vernova and HealthCare spun
    # off); allowance coverage 0.8% against 4.3% "historically" compared GE Aerospace with the group it came from
    data = series()
    plain = make_ctx(data)
    assert plain.perimeter_break is None and plain.period_at(2) is not None
    ctx = make_ctx(data, restated={("REVENUE", 2022): data["REVENUE"][2022] * 0.5})
    assert ctx.perimeter_break == pd.Timestamp("2022-12-31")
    assert ctx.period_at(1) == pd.Timestamp("2023-12-31") and ctx.period_at(2) is None
    assert len(ctx.hist("REVENUE", 6)) == 2
    assert pd.Timestamp("2022-12-31") in ctx.periods  # the key-figure tables of the report still show those years
    small = make_ctx(data, restated={("REVENUE", 2022): data["REVENUE"][2022] * 0.9})
    assert small.perimeter_break is None  # a 10% revision is a restatement or a reclassification, not a new perimeter


def test_c_cf06_skips_free_cash_flow_over_a_negligible_ebitda():
    # Boeing FY2019: operating loss 0.4B plus D&A 2.0B = EBITDA 0.3B on revenue of 76.6B; free cash flow -4.2B
    base = series()
    over = {"OPERATING_INCOME": {2024: -base["DA"][2024] + 0.005 * base["REVENUE"][2024]}, "CFO": {2024: -100.0}}
    with pytest.raises(SkipDetector):
        run_one(make_ctx(series(overrides=over)), "C.CF.06")


def test_a04_shares_issued_for_an_acquisition_described_in_the_report_discount_an_unexplained_rise():
    from ffi.text.document import build_doc

    base = series()
    ctx = make_ctx(series(overrides={"EQUITY": {2024: base["EQUITY"][2024] + 500.0}}))
    plain = run_one(ctx, "A.04")[0]
    assert plain.flagged
    # Disney FY2019 note on the acquisition of 21st Century Fox; the SEC's data carries no tag for the shares issued
    text = ("In connection with the acquisition, the Company issued approximately 307 million shares of its common stock "
            "as consideration to the former shareholders of TFCF.")
    ctx.docs = [build_doc("a", "10-K", "2024-12-31", "u", f"<p>Item 8. Financial Statements and Supplementary Data</p><p>{text}</p>")]
    s = run_one(ctx, "A.04")[0]
    assert s.severity < plain.severity * 0.5 and "307 million shares" in s.rationale
    upo = "Upon the closing of the IPO, all outstanding convertible preferred stock converted into shares of common stock."
    ctx.docs = [build_doc("a", "10-K", "2024-12-31", "u", f"<p>Item 8. Financial Statements and Supplementary Data</p><p>{upo}</p>")]
    assert run_one(ctx, "A.04")[0].severity < plain.severity * 0.5


def test_unexplained_movements_are_discounted_in_a_large_acquisition_year():
    from ffi.scoring.engine import EXPLAIN_RULES

    # Waste Management 2024: Stericycle (23% of prior-year assets) moved receivables, goodwill and debt; MD&A discusses the
    # acquisition, not each line, and the investigation closed F.04 as explained by the deal
    rule = next(ids for key, ids, _ in EXPLAIN_RULES if key == "large_acquisition")
    assert "F.04" in rule
