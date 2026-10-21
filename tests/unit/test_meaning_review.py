"""Regression tests for the meaning review: every comparison matches on period, point in time, scope, concept, basis,
unit, sign and denominator, and AI-written text cannot bypass the checks. Each test uses the layout or figures of the
filing that exposed the problem where possible."""

from __future__ import annotations

import threading
from datetime import date
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from ffi.detectors.base import REGISTRY, load_all
from ffi.detectors.scores import V, debt_parts, dep_measure
from ffi.edgar.endpoints import FilingRef
from ffi.xbrl.facts import FactStore
from tests.conftest import make_ctx, run_one, series

load_all()


def _cf(rows: list[tuple]) -> dict:
    """companyfacts from (concept, fiscal year, value, filed year, unit) rows; durations run Jan-Dec, instants Dec 31."""
    from ffi.xbrl.canonical import CONCEPT_TO_ITEMS

    facts: dict = {"us-gaap": {}}
    for concept, y, val, filed_y, *unit in rows:
        u = unit[0] if unit else "USD"
        items = CONCEPT_TO_ITEMS[("us-gaap", concept)]
        from ffi.xbrl.canonical import BY_ID

        row = {"end": f"{y}-12-31", "val": val, "accn": f"0000000000-{filed_y % 100:02d}-000001", "fy": filed_y - 1,
               "fp": "FY", "form": "10-K", "filed": f"{filed_y}-02-20"}
        if BY_ID[items[0][0]].kind == "D":
            row["start"] = f"{y}-01-01"
        facts["us-gaap"].setdefault(concept, {"units": {}})["units"].setdefault(u, []).append(row)
    return {"cik": 1, "entityName": "Test", "facts": facts}


def _panel(rows: list[tuple]):
    years = sorted({r[1] for r in rows})
    rows = rows + [("Revenues", y, 1000.0, y + 1) for y in years if not any(r[0] == "Revenues" and r[1] == y for r in rows)]
    return FactStore.from_companyfacts(1, _cf(rows), date(2030, 1, 1)).annual("first")


# ------------------------------------------------------------------ point in time
def test_a_past_year_never_sees_a_later_restatement():
    """FY2022 revenue re-reported in the FY2023 annual report (filed Feb 2024) is later-period information for a FY2022
    analysis: the store cut at the end of FY2022's filing season does not hold it, so A.14 has nothing to report."""
    from ffi.pipeline.run import point_in_time

    ctx = make_ctx(series(), restated={("REVENUE", 2022): 1200.0})
    focus = pd.Timestamp("2022-12-31")
    filings = [FilingRef("0000000000-23-000001", "10-K", date(2023, 2, 20), date(2022, 12, 31), "d.htm", "")]
    cutoff = point_in_time(focus, filings, date(2025, 12, 31))
    assert cutoff == date(2023, 4, 30)  # 120 days after the year end (the annual report came earlier)
    past = ctx.store.restricted(cutoff)
    v = past.versions("A")
    assert len(v[(v["item"] == "REVENUE") & (v["end"] == focus)]["val"].unique()) == 1
    assert past.annual("latest").get("REVENUE", focus) == ctx.store.annual("first").get("REVENUE", focus)
    assert ctx.A_latest.get("REVENUE", focus) == pytest.approx(1200e6)  # the full store still holds the revision
    assert point_in_time(focus, filings, date(2023, 3, 1)) == date(2023, 3, 1)  # never after --as-of


def test_a_release_announcing_the_next_years_first_quarter_is_not_read(monkeypatch):
    """American Express files its first-quarter release about 107 days after its year end, inside the 120-day window
    for the fourth-quarter release: it announces fiscal 2025 and must not reach a FY2024 analysis."""
    from ffi.pipeline import run as R

    def f(accn, filed):
        return FilingRef(accn, "8-K", date.fromisoformat(filed), date.fromisoformat(filed), "d.htm", "2.02,9.01")

    filings = [f("q3", "2024-10-18"), f("q4", "2025-01-24"), f("q1next", "2025-04-17")]
    monkeypatch.setattr(R.endpoints, "filing_index", lambda c, cik, fl: [{"type": "EX-99.1", "document": "ex.htm", "description": ""}])
    docs = R.load_releases(NS(get_text=lambda url: "<p>Revenue was $1 billion.</p>"), 1, filings, pd.Timestamp("2024-12-31"),
                           date(2026, 1, 1))
    assert [d.accession for d in docs] == ["q3", "q4"]


def test_peer_years_after_the_focus_year_are_not_pooled():
    """B.04 and D.03 used each peer's latest years whatever the focus year, so a FY2019 review pooled peers' 2025 data."""
    import inspect

    from ffi.detectors import b_scores, d_stats

    assert "aligned_period" in inspect.getsource(b_scores.b04) and "aligned_period" in inspect.getsource(d_stats.d03)


# ------------------------------------------------------------------ concept and scope
def test_debt_is_never_counted_twice():
    """LongTermDebt includes current maturities; DebtCurrent includes short-term borrowings."""
    P = _panel([("LongTermDebt", 2024, 100.0, 2025), ("LongTermDebtCurrent", 2024, 10.0, 2025), ("ShortTermBorrowings", 2024, 5.0, 2025)])
    v = V(P, P.periods[-1])
    assert v.debt() == 105.0 and set(debt_parts(v)) == {"LONG_TERM_DEBT", "ST_BORROWINGS"}
    P = _panel([("LongTermDebtNoncurrent", 2024, 90.0, 2025), ("DebtCurrent", 2024, 15.0, 2025), ("ShortTermBorrowings", 2024, 5.0, 2025)])
    assert V(P, P.periods[-1]).debt() == 105.0
    P = _panel([("LongTermDebtNoncurrent", 2024, 90.0, 2025), ("LongTermDebtCurrent", 2024, 10.0, 2025), ("CommercialPaper", 2024, 5.0, 2025)])
    assert V(P, P.periods[-1]).debt() == 105.0


def test_one_depreciation_measure_for_every_year():
    """Depreciation tagged in one year only: both years use D&A (less amortization when that is tagged in both)."""
    P = _panel([("Depreciation", 2024, 50.0, 2025), ("DepreciationDepletionAndAmortization", 2024, 80.0, 2025),
                ("DepreciationDepletionAndAmortization", 2023, 70.0, 2024), ("AmortizationOfIntangibleAssets", 2024, 20.0, 2025)])
    label, dep = dep_measure(V(P, P.periods[-1]), (0, 1))
    assert label == "depreciation and amortization" and (dep(0), dep(1)) == (80.0, 70.0)


def test_gross_margin_uses_one_measure_for_the_years_compared():
    """Costco FY2019 tagged gross profit on net sales ($16,465M), leaving out $3,352M of membership fees in revenue;
    FY2020 tagged none. Compared, both years use revenue less cost of sales."""
    P = _panel([("Revenues", 2019, 152703.0, 2020), ("CostOfGoodsAndServicesSold", 2019, 132886.0, 2020), ("GrossProfit", 2019, 16465.0, 2020),
                ("Revenues", 2020, 166761.0, 2021), ("CostOfGoodsAndServicesSold", 2020, 144939.0, 2021)])
    v = V(P, P.periods[-1])
    assert v.gross_margin(1, years=(0, 1)) == pytest.approx((152703 - 132886) / 152703)
    assert V(P, P.periods[0]).gross_margin() == pytest.approx(16465 / 152703)  # one year alone: the company's own figure


def test_history_years_on_another_basis_are_left_out():
    """A year of general and administrative expense is not compared with years of SG&A."""
    from ffi.detectors.c_accounts import ratio_hist

    ctx = make_ctx(series())
    first = ctx.A.periods[-3]
    rec = ctx.A.prov[("SGA", first)]
    ctx.A.prov[("SGA", first)] = type(rec)(**{**rec.__dict__, "concept": "GeneralAndAdministrativeExpense"})
    fn = lambda v: v.g("SGA") / v.g("REVENUE")  # noqa: E731
    focus: dict = {}
    fn(V(ctx.A, ctx.focus, focus))
    assert len(ratio_hist(ctx, fn, basis_of=focus)) == len(ratio_hist(ctx, fn)) - 1


def test_buybacks_tagged_only_in_the_equity_statement_explain_the_equity_roll_forward():
    """American Express FY2019: $4,585M of treasury stock acquired, no cash-flow repurchase tag."""
    from ffi.xbrl.canonical import CONCEPT_TO_ITEMS

    assert ("BUYBACKS", 2) in CONCEPT_TO_ITEMS[("us-gaap", "TreasuryStockValueAcquiredCostMethod")]


def test_dividends_declared_come_from_paid_and_the_change_in_payable():
    """Progressive FY2025: $7,972M of dividends payable at year end (the annual dividend declared in December), so the
    cash paid during the year (last year's declaration) understates what reduced retained earnings."""
    ctx = make_ctx(series(overrides={"DIVIDENDS_PAID": {2023: 2600.0, 2024: 2870.0}, "DIVIDENDS_PAYABLE": {2023: 2640.0, 2024: 7972.0},
                                     "RETAINED_EARNINGS": {2024: 300 + sum(150 * 1.05 ** i for i in range(8)) - 8202.0}}))
    from ffi.detectors.a_integrity import dividends_declared

    div, how = dividends_declared(ctx)
    assert div == pytest.approx((2870 + 7972 - 2640) * 1e6) and "payable" in how
    assert run_one(ctx, "A.05")[0].severity < 0.5


def test_pre_tax_income_before_equity_method_income_ties_after_adding_it():
    """Toyota FY2019 (US GAAP): pre-tax income before equity-method income ¥2,285.5B, tax ¥660.0B, equity-method income
    ¥360.1B, income after tax (incl. NCI) ¥1,985.6B."""
    rows = [("IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments", 2019, 2285.5, 2020),
            ("IncomeTaxExpenseBenefit", 2019, 660.0, 2020), ("IncomeLossFromEquityMethodInvestments", 2019, 360.1, 2020),
            ("ProfitLoss", 2019, 1985.6, 2020), ("NetIncomeLoss", 2019, 1882.9, 2020)]
    ctx = make_ctx(series())
    store = FactStore.from_companyfacts(1, _cf(rows), date(2030, 1, 1))
    import dataclasses

    ctx = dataclasses.replace(ctx, store=store, A=store.annual("first"), focus=store.annual("first").periods[-1])
    s = run_one(ctx, "A.07")[0]
    assert s.severity == 0.0 and "plus equity-method income" in s.rationale


def test_tax_sums_cover_the_same_years():
    """Taxes paid reported for two of three years: the current-tax sum must not include the third year."""
    ctx = make_ctx(series(overrides={"INCOME_TAXES_PAID": {2023: 40.0 * 1.05 ** 6, 2024: 40.0 * 1.05 ** 7}}))
    s = run_one(ctx, "A.13")[0]
    assert abs(s.raw_value) < 0.01 and "2-year" in s.rationale


def test_a_business_unit_total_is_not_read_as_the_company_total():
    """NVIDIA FY2019 MD&A: "GPU business revenue was $10.17 billion" (a segment) sat within 5% of FY2018 total revenue
    ($9.71B) and was reported as a mismatch."""
    from ffi.detectors.textnum import check_amounts

    text = ("Revenue for fiscal year 2019 increased 21% year over year. GPU business revenue was $10.17 billion, up 25% from a "
            "year earlier. Tegra Processor business revenue was $1.54 billion.")
    known = {"REVENUE": {"years": {2019: 11.716e9, 2018: 9.714e9, 2017: 6.91e9}, "focus": 2019}}
    assert check_amounts(text, known) == (0, [])
    from ffi.detectors.textnum import qualified

    assert qualified(text, text.index("revenue was $10.17"))  # the business unit's total, whatever its size
    assert check_amounts(text, {"REVENUE": [9.714e9]}) == (0, [])  # even against the year it happens to be near


def test_a_stated_total_is_compared_with_the_year_it_describes():
    from ffi.detectors.textnum import check_amounts

    known = {"REVENUE": {"years": {2024: 100e9, 2023: 97e9}, "focus": 2024}}
    assert check_amounts("In fiscal 2023, total revenue was $97.0 billion.", known)[1] == []
    # no year named: near last year's total is not a contradiction of this year's
    assert check_amounts("Total revenue was $96.9 billion.", known) == (1, [])  # last year's total, checked
    assert check_amounts("Total revenue was $97.3 billion.", known)[1] == []  # nearer last year's total: which year is unclear
    _, bad = check_amounts("Total revenue was $98.9 billion.", known)
    assert bad and bad[0].nearest == 100e9


def test_growth_is_measured_between_adjacent_years_only():
    """RPO missing for one year: no two-year change may be compared with a one-year revenue change."""
    ctx = make_ctx(series(overrides={"RPO": {2020: 500.0, 2022: 600.0, 2023: 660.0, 2024: 700.0}}))
    s = run_one(ctx, "C.REV.10")[0]
    assert "consecutive" in s.rationale


def test_backlog_that_is_a_sliver_of_revenue_is_not_compared():
    from ffi.detectors.base import SkipDetector

    # NVIDIA FY2026: RPO $2.3B against revenue of $215.9B (1.1%) was flagged for "revenue outgrowing backlog"
    rev = series()["REVENUE"]
    ctx = make_ctx(series(overrides={"RPO": {y: 0.011 * rev[y] / (1.1 + 0.2 * (y - 2021)) for y in range(2021, 2025)}}))
    with pytest.raises(SkipDetector, match="immaterial"):
        run_one(ctx, "C.REV.10")


def test_a_year_missing_breaks_a_streak():
    ctx = make_ctx(series(overrides={"PROVISION_DOUBTFUL": {2024: 5.0, 2022: 5.0}, "WRITE_OFFS": {2024: 9.0, 2022: 9.0}}))
    with pytest.raises(Exception, match="2 years"):
        run_one(ctx, "C.AR.03")


def test_window_dressing_compares_the_same_borrowings_and_names_them():
    """JPMorgan FY2019 tags only short-term borrowings ($40.9B at year end): the result is about short-term borrowings."""
    import dataclasses

    from ffi.xbrl.facts import Panel

    ps = [pd.Timestamp(d) for d in ("2019-03-31", "2019-06-30", "2019-09-30", "2019-12-31")]
    vals = pd.DataFrame({"ST_BORROWINGS": [71305e6, 59890e6, 48893e6, 40920e6]}, index=ps)
    Q = Panel("Q", vals, {p: str(p.date()) for p in ps})
    ctx = dataclasses.replace(make_ctx(series()), Q=Q, focus=pd.Timestamp("2019-12-31"))
    s = run_one(ctx, "C.CF.07")[0]
    assert "short-term borrowings" in s.rationale and "Year-end debt" not in s.rationale


# ------------------------------------------------------------------ governance and events
def test_pay_plan_filings_are_not_officer_turnover():
    from ffi.context import Event

    evs = [Event(str(i), "OFFICER_CHANGE", date(2024, 3, i + 1), str(i), "8-K Item 5.02 - no senior finance/CEO departure or appointment language")
           for i in range(6)] + [Event("c", "OFFICER_CHANGE", date(2024, 9, 1), "c", "8-K Item 5.02 - CFO change")]
    ctx = NS(events_of=lambda t, within_days: evs, ev_event=lambda e: None, focus_label="FY2024", currency="USD", tuning={})
    s = REGISTRY["C.GOV.03"].fn(ctx)[0]
    assert s.severity == pytest.approx(0.35 * 0.8) and not s.flagged  # one planned CFO succession alone


def test_officer_changes_helper_ignores_pay_plan_filings():
    from ffi.context import Event, RunContext

    ctx = make_ctx(series())
    ctx.events = [Event("a", "OFFICER_CHANGE", date(2024, 6, 1), "a", "8-K Item 5.02 - no senior finance/CEO departure or appointment language"),
                  Event("b", "OFFICER_CHANGE", date(2024, 6, 2), "b", "8-K Item 5.02 - CEO change")]
    assert [e.id for e in RunContext.officer_changes(ctx, 730)] == ["b"]


# ------------------------------------------------------------------ releases and non-GAAP
def test_non_gaap_gap_states_its_direction_once():
    from ffi.detectors import e_text

    ctx = make_ctx(series())
    rel = NS(period_end="2025-01-24", text="Adjusted diluted EPS\n$3.04\nDiluted EPS\n$3.49\n", accession="r", url="u", paragraphs=[])
    ctx.releases = [rel]
    filed = ctx.a("EPS_DILUTED")  # the release's GAAP figure must match the filed one, or the release is not used at all
    adj = [e_text.EpsFigure(round(filed * 0.9, 2), "FY", 0, 10)]
    gaap = [e_text.EpsFigure(filed, "FY", 20, 30)]
    orig = e_text.release_eps
    e_text.release_eps = lambda c, r: (adj, gaap)
    try:
        s = run_one(ctx, "C.NG.01")[0]
    finally:
        e_text.release_eps = orig
    assert "below GAAP EPS" in s.rationale and "-" not in s.rationale.split("adjusted EPS is ")[1][:3]
    assert s.period == ctx.focus_label


def test_definition_drift_never_falls_back_to_another_quarter(monkeypatch):
    from ffi.detectors.base import SkipDetector
    from ffi.detectors.e_text import cng03

    ctx = make_ctx(series())
    ctx.llm = object()
    ctx.releases = [NS(period_end="2024-07-25", text="x"), NS(period_end="2025-01-24", text="y")]  # Q2 2024 and Q4 2024
    with pytest.raises(SkipDetector, match="same quarter a year earlier"):
        cng03(ctx)


def test_mdna_claims_are_measured_as_stated():
    from ffi.detectors.e_text import Claim, claim_metric

    def c(metric, quote, pct=None):
        return Claim(metric=metric, direction="increase", magnitude_pct=pct, driver="", quote=quote)

    assert claim_metric(c("gross margin", "Total gross margin increased $5.2 billion")) == "gross profit"
    assert claim_metric(c("gross margin", "Gross margin percentage increased 120 basis points")) == "gross margin"
    assert claim_metric(c("gross margin", "Gross margin increased 5%", 5.0)) is None  # a percent of what is not said
    assert claim_metric(c("operating expenses", "Total costs and expenses increased 9%", 9.0)) == "costs and expenses"


def test_only_this_years_mdna_explains_this_years_changes():
    import inspect

    from ffi.detectors import f_graph

    assert "ctx.doc.accession" in inspect.getsource(f_graph.f04)


# ------------------------------------------------------------------ scores
def test_missing_inputs_are_not_failed_piotroski_tests():
    from ffi.detectors.scores import piotroski_f

    ctx = make_ctx(series())
    r = piotroski_f(ctx.A, ctx.focus)
    assert r["available"] == 9
    ctx2 = make_ctx({k: v for k, v in series().items() if k not in ("CURRENT_ASSETS", "CURRENT_LIABILITIES")})
    assert piotroski_f(ctx2.A, ctx2.focus)["available"] == r["available"] - 1


def test_cross_section_concepts_keep_the_canonical_priority():
    from ffi.peers.crosssection import DURATION_CONCEPTS, INSTANT_CONCEPTS
    from ffi.xbrl.canonical import BY_ID

    for item, concepts in {**DURATION_CONCEPTS, **INSTANT_CONCEPTS}.items():
        order = [BY_ID[item].concepts.index(c) for c in concepts]
        assert order == sorted(order), item


def test_a_recalibrated_severity_decides_the_flag(monkeypatch):
    from ffi.pipeline.detect import run_detectors
    from ffi.scoring import engine

    ctx = make_ctx(series(overrides={"AR": {2024: 450.0}}))
    results = run_detectors(ctx, only={"C.AR.01"})
    assert results[0].signals[0].flagged
    monkeypatch.setattr(engine, "load_calibration", lambda: {"C.AR.01": [(0.0, 0.0), (1.0, 0.2)]})
    engine.score(ctx, results)
    assert not results[0].signals[0].flagged


# ------------------------------------------------------------------ AI text and the audit
def test_single_digit_dollar_amounts_and_short_quotes_are_checked():
    from ffi.agent import grounding

    assert grounding.check("A charge of $5 per share.", [1.0], []).unsupported_numbers == ["$5"]
    assert grounding.check("Three items rose.", [], []).ok
    assert grounding.check('Management called it "a temporary pause in demand".', [], []).unverified_quotes


def test_a_cited_fact_must_be_the_one_for_its_period():
    """The audit re-reads a cited value for its own period: a filing's prior-year value under the same concept is not a
    match for the current year."""
    from ffi.audit.deterministic import _fact_matches, _facts_index

    ctx = make_ctx(series())
    e = ctx.ev("REVENUE")
    assert _fact_matches(_facts_index(ctx), e)
    prior = ctx.a("REVENUE", 1)
    e.value = prior  # the FY2024 10-K also reports FY2023 revenue under the same concept
    assert not _fact_matches(_facts_index(ctx), e)


def test_untraceable_questions_written_by_the_investigation_are_removed():
    from ffi.audit.deterministic import run_audit
    from ffi.pipeline.detect import run_detectors
    from ffi.scoring.engine import score, template_narrative

    ctx = make_ctx(series(overrides={"AR": {2024: 450.0}, "CFO": {2022: 40.0, 2023: 30.0, 2024: 10.0}}))
    sc = score(ctx, run_detectors(ctx, only={"C.AR.01", "C.CF.01", "B.01", "B.03", "B.05"}))
    f = sc.findings[0]
    f.narrative = template_narrative(f)
    f.questions = ["Why did receivables reach $987.6B?", "What drove collections?"]
    f.agent_text = list(f.questions)
    res = run_audit(ctx, f, template_narrative)
    assert f.questions == ["What drove collections?"] and res.status == "verified_after_corrections"


def test_concurrent_calls_cannot_overrun_the_budget(tmp_path, monkeypatch):
    from pydantic import BaseModel

    from ffi import llm as llm_mod

    class Out(BaseModel):
        x: int

    obj = llm_mod.LLM.__new__(llm_mod.LLM)
    obj.backend, obj.cli, obj.client = "claude-code", "claude", None
    obj.reasoning_model, obj.extraction_model = "claude-opus-5-5", "claude-haiku-5-5"
    obj.cache_dir, obj.offline, obj.spend, obj._lock, obj._reserved = tmp_path, False, llm_mod.Spend(), threading.Lock(), 0.0
    started = threading.Barrier(2, timeout=5)
    gate = threading.Event()

    def slow(*a, **k):
        gate.wait(5)
        return Out(x=1)

    monkeypatch.setattr(obj, "_call", slow)
    one_call = 16000 * 0.25 * 20 / 1e6 + 100 / 3.5 * 4 / 1e6
    obj.budget_usd = one_call * 1.5  # room for one call, not two
    errors: list = []

    def call(i):
        try:
            started.wait()
            obj.structured(Out, "s", f"prompt {i}" + " " * 90, purpose="t", model="claude-opus-5-5")
        except llm_mod.BudgetExceeded as e:
            errors.append(e)

    threads = [threading.Thread(target=call, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    import time

    time.sleep(0.3)
    gate.set()
    for t in threads:
        t.join()
    assert len(errors) == 1 and obj._reserved == pytest.approx(0.0)
    assert np.isfinite(obj.remaining())


def test_a_cache_file_held_open_elsewhere_does_not_stop_the_run(tmp_path, monkeypatch):
    """The project lives in a Dropbox folder: Dropbox holds cache files open while it syncs them, and removing a raw
    frame after the cross-section was built raised PermissionError and ended a FY2019 run."""
    from pathlib import Path

    from ffi.edgar.client import EdgarClient

    client = EdgarClient.__new__(EdgarClient)
    client.cache_dir = tmp_path
    url = "https://data.sec.gov/api/xbrl/frames/us-gaap/Revenues/USD/CY2019.json"
    f = client._path(url)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"{}")

    def locked(self, missing_ok=False):
        raise PermissionError(32, "The process cannot access the file because it is being used by another process")

    monkeypatch.setattr(Path, "unlink", locked)
    client.evict(url)  # no exception
    assert f.exists()


def test_client_wording_has_no_doubled_or_internal_terms():
    from ffi.detectors.catalog import explain
    from ffi.report.plain import plain

    for did in ("A.02", "A.11"):
        ex = explain(did)
        text = plain(" ".join((ex.what, ex.why, ex.read)))
        assert "separately in the financial data separately" not in text and "data data" not in text
        assert "companyfacts" not in text and "SEC's the SEC's" not in text


def test_mdna_after_the_statements_in_an_annual_report_exhibit_is_read_as_mdna():
    """Progressive's Exhibit 13 holds the statements and notes first, then the auditor's report, then MD&A. MD&A was
    labelled as financial statements, so no test read it and F.04 reported revenue as never discussed."""
    from ffi.text.document import build_doc

    main = "<p>Item 7. Management's Discussion and Analysis</p><p>Incorporated by reference from the Annual Report.</p>" \
           "<p>Item 8. Financial Statements and Supplementary Data</p><p>Incorporated by reference.</p>"
    notes = "".join(f"<p>Note {i} – POLICY {i}</p><p>Text of note {i} about accounting policies and estimates.</p>" for i in range(1, 9))
    ex13 = ("<p>Consolidated Statements of Comprehensive Income</p>" + notes
            + "<p>Report of Independent Registered Public Accounting Firm</p><p>We have audited the statements.</p>"
            + "<p>Management's Discussion and Analysis of Financial Condition and Results of Operations</p>"
            + "<p>Net premiums written increased 9% compared with last year, driven by growth in policies in force.</p>")
    doc = build_doc("a", "10-K", "2025-12-31", "u", main, [("Item 7. Management Discussion and Analysis (Exhibit 13 annual report)", ex13)])
    assert "Net premiums written increased" in doc.section_text("item7")
    assert "Net premiums written" not in doc.section_text("item8") and "Text of note 3" in doc.section_text("item8")


def test_net_ppe_above_gross_less_depreciation_is_not_a_subtotal_break():
    from types import SimpleNamespace as NS

    from ffi.detectors.base import REGISTRY, SkipDetector, load_all

    load_all()
    vals = {"PPE_GROSS": 199.829e9, "ACCUM_DEP": 79.39e9, "PPE_NET": 171.036e9}  # Alphabet 2024: gross leaves out construction in progress
    ctx = NS(a=lambda item, k=0: vals.get(item, float("nan")), focus_label="FY2024", evs=lambda *a, **k: [])
    try:
        out = REGISTRY["A.02"].fn(ctx)
    except SkipDetector:
        return  # no other subtotal pair: skipped, not flagged
    assert out[0].severity == 0.0


def test_beneish_aqi_does_not_double_count_short_term_investments():
    import pandas as pd

    from ffi.detectors import scores as S
    from ffi.xbrl.facts import Panel

    p0, p1 = pd.Timestamp("2023-12-31"), pd.Timestamp("2024-12-31")
    # Alphabet: marketable securities sit inside current assets
    values = pd.DataFrame({"CURRENT_ASSETS": [171.53e9, 163.711e9], "PPE_NET": [134.345e9, 171.036e9],
                           "TOTAL_ASSETS": [402.392e9, 450.256e9], "ST_INVESTMENTS": [87e9, 72e9]}, index=[p0, p1])
    r = S.beneish(Panel("A", values, {p0: "FY2023", p1: "FY2024"}), p1)
    assert 0.9 < r["AQI"] < 1.2


def test_adjustment_rows_ignore_boilerplate_and_keep_table_rows():
    from ffi.detectors.e_text import NG_CATEGORIES, adjustment_rows

    text = ("Forward-looking statements involve risks, including litigation, impairment of assets and restructuring charges that "
            "could cause results to differ.\n"
            "Reconciliation of Adjusted Operating EBITDA\n| 2025 | 2024\n"
            "Restructuring | $51 | $4\n"
            "Impairments and other\n| $275\n| $50\n"
            "Litigation matters, net\n")
    labels = [lab for lab, _, _ in adjustment_rows(text)]
    assert "Restructuring" in labels and "Impairments and other" in labels
    assert not any("could cause results" in lab for lab in labels)  # the boilerplate sentence has no figures and is no row
    import re

    assert [n for n, p in NG_CATEGORIES.items() if any(re.search(p, lab, re.I) for lab in labels)] == ["restructuring", "impairment"]


def test_allowance_gap_in_a_year_of_large_acquisition_is_discounted():
    from types import SimpleNamespace as NS

    from ffi.detectors.base import REGISTRY, load_all

    load_all()
    vals = {"ALLOWANCE": 165e6, "PROVISION_DOUBTFUL": 65e6, "WRITE_OFFS": 55e6, "ACQUISITIONS": 7.49e9, "TOTAL_ASSETS": 31.2e9}
    ctx = NS(a=lambda item, k=0: {"ALLOWANCE": 165e6 if k == 0 else 30e6}.get(item) if item == "ALLOWANCE"
             else (31.2e9 if (item == "TOTAL_ASSETS" and k == 1) else vals.get(item, float("nan"))),
             focus_label="FY2024", evs=lambda *a, **k: [])
    s = REGISTRY["A.09"].fn(ctx)[0]
    assert s.severity <= 0.55 and "Discounted" in s.rationale  # Waste Management bought Stericycle in 2024


def test_year_on_year_quarters_are_matched_by_date_not_position():
    from types import SimpleNamespace as NS

    import pandas as pd

    from ffi.detectors.b_scores import b09

    # seasonal EPS, the same quarter equal every year; one quarter (2023-06-30) was never filed
    ends = [e for e in pd.date_range("2021-03-31", "2024-12-31", freq="QE") if e != pd.Timestamp("2023-06-30")]
    eps = pd.Series([(0.50, 0.90, 1.30, 2.00)[(e.month - 1) // 3] for e in ends], index=ends)
    ctx = NS(qhist=lambda item, n=20: eps.tail(n) if item == "EPS_DILUTED" else pd.Series(1.0, index=ends).tail(n),
             profile=NS(key="bank"), focus_label="FY2024", tuning={}, currency="USD", a=lambda *a, **k: float("nan"))
    (s,) = b09(ctx)
    # every quarter with a filed year-earlier quarter is flat year on year; a shift by four rows paired quarters a
    # year and a quarter apart after the gap
    assert "10 of 10 quarters with diluted EPS up by 0-1 cent" in s.rationale


def test_seasonal_model_reads_only_unbroken_quarters():
    import pandas as pd

    from ffi.detectors.d_stats import unbroken_quarters

    ends = pd.date_range("2019-03-31", "2024-12-31", freq="QE")
    s = pd.Series(range(len(ends)), index=ends, dtype=float).drop(pd.Timestamp("2021-12-31"))  # a fourth quarter not filed
    tail = unbroken_quarters(s)
    assert tail.index[0] == pd.Timestamp("2022-03-31") and len(tail) == 12
    assert len(unbroken_quarters(s.loc["2022":])) == 12  # nothing missing: unchanged


def test_customer_without_a_total_cost_base_is_not_compared(monkeypatch):
    from ffi.detectors import f_graph
    from ffi.detectors.e_text import Relationship
    from ffi.xbrl.facts import FactStore

    data = series()
    del data["OPERATING_INCOME"]  # the customer reports no operating income
    customer = make_ctx(data).store
    a = customer.annual()
    p = a.values.index[-1]
    lines = sum(a.get(i, p) for i in ("COGS", "SGA", "RND", "CAPEX") if np.isfinite(a.get(i, p)))
    ctx = make_ctx(series())
    ctx.client = object()
    ctx.relationships = [(Relationship(counterparty="Big Customer Inc", kind="major_customer", amount_usd=0.8 * lines,
                                       is_new_this_year=False, quote="q"), (0, 1))]
    monkeypatch.setattr(f_graph, "_resolve_counterparty", lambda ctx, name: 99)
    monkeypatch.setattr(FactStore, "load", classmethod(lambda cls, client, cik, as_of: customer))
    (s,) = f_graph.f01(ctx)
    # sales of 80% of the customer's cost lines used to read as an impossible 80% of everything it spends
    assert s.severity == 0.0 and "none resolvable" in s.rationale
