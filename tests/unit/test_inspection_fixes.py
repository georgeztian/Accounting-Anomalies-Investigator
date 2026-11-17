"""Regression tests for the full-project inspection: evidence citations read as figures, a finding halved for an explanation
the audit removed, a stated fall compared with its sign, an earlier year's ratio shown as this year's, a run of years counted
across a gap, facts counted twice, Benford thresholds on a small sample, and command-line inputs."""

from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np
import pytest

from ffi.detectors.base import SkipDetector, load_all
from tests.conftest import make_ctx, run_one, series

load_all()

# the Costco FY2026 run: the investigation's supported explanation, as the agent wrote it into the report
COSTCO_EXPLANATION = ("Higher sales and faster inventory turnover naturally raise payables, because members pay at the register "
                      "before vendors are paid.: supported by the evidence gathered (L0.1, L2.2, L2.3)")


# ------------------------------------------------------------------ evidence citations
def test_ledger_citations_are_not_financial_figures():
    from ffi.agent import grounding

    # "(L0.1, L2.2, L2.3)" was read as the figures 0.1, 2.2 and 2.3, which no source states
    rep = grounding.check(COSTCO_EXPLANATION, [], [])
    assert rep.ok and rep.numbers_checked == 0
    assert grounding.check("See L1.2 and [L0.1, L3.4]: revenue rose.", [], []).ok
    assert grounding.numbers_in("Cited in L2.1 and (L0.3).") == []
    # a figure beside a citation is still checked
    assert grounding.check("Payables rose 2.2% (L0.1).", [], []).unsupported_numbers == ["2.2%"]


def test_ledger_citations_never_reach_the_client_report():
    from ffi.report.plain import plain

    for cited in (COSTCO_EXPLANATION, "Payables rose with sales [L0.1, L2.2].", "Payables rose with sales (evidence L0.1 and L2.2)."):
        out = plain(cited, terms=False)
        assert "L0.1" not in out and "L2.2" not in out and "()" not in out, out
    assert plain("Level 3 assets (see Note 4) rose.", terms=False) == "Level 3 assets (see Note 4) rose."


def _explained_finding():
    from ffi.pipeline.detect import run_detectors
    from ffi.scoring.engine import EXPLAINED, EXPLAINED_FACTOR, score, template_narrative

    ctx = make_ctx(series(overrides={"AR": {2024: 450.0}, "CFO": {2022: 40.0, 2023: 30.0, 2024: 10.0}}))
    sc = score(ctx, run_detectors(ctx, only={"C.AR.01", "C.CF.01", "B.01", "B.03", "B.05"}))
    f = sc.findings[0]
    f.narrative = template_narrative(f)
    full = (f.score, f.priority)
    f.narrative_source = EXPLAINED  # as agent.runner._apply leaves a case closed as explained
    f.score *= EXPLAINED_FACTOR
    f.priority *= EXPLAINED_FACTOR
    return ctx, sc, f, full


def test_a_finding_is_not_halved_for_an_explanation_the_audit_removed():
    from ffi.audit.deterministic import run_audit
    from ffi.report.model import INVESTIGATION
    from ffi.scoring.engine import (
        EXPLANATION_WITHDRAWN,
        apply_audit,
        apply_explained,
        index_of,
        template_narrative,
    )

    ctx, sc, f, full = _explained_finding()
    before = dict(sc.contributions)
    f.alternatives = ["Receivables of $987.6B were collected after the year end: supported by the evidence gathered"] + f.alternatives
    f.agent_text = f.alternatives[:1]
    apply_explained(sc)
    assert sc.risk_index < index_of(before)
    f.audit = run_audit(ctx, f, template_narrative)
    apply_audit(sc)
    # the only explanation it was closed on cites a figure no source states: removed, and the half weight with it
    assert f.audit.status == "verified_after_corrections" and f.agent_text[0] not in f.alternatives
    assert f.narrative_source == EXPLANATION_WITHDRAWN and (f.score, f.priority) == pytest.approx(full)
    assert dict(sc.contributions) == pytest.approx(before) and sc.risk_index == index_of(before)
    assert "counts in full" in INVESTIGATION[EXPLANATION_WITHDRAWN][1]


def test_an_explanation_that_cites_its_evidence_keeps_the_finding_explained():
    from ffi.audit.deterministic import run_audit
    from ffi.scoring.engine import EXPLAINED, template_narrative

    ctx, _, f, full = _explained_finding()
    f.alternatives = [COSTCO_EXPLANATION] + f.alternatives
    f.agent_text = [COSTCO_EXPLANATION]
    res = run_audit(ctx, f, template_narrative)
    assert res.status == "verified" and COSTCO_EXPLANATION in f.alternatives
    assert f.narrative_source == EXPLAINED and f.score == pytest.approx(full[0] / 2)


# ------------------------------------------------------------------ MD&A claims
def _mdna_ctx(quote: str, claims):
    from ffi.detectors.e_text import Claims
    from ffi.text.document import build_doc

    body = ["Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations", quote]
    body += [f"Discussion paragraph {n} about the results of operations for the year." for n in range(60)]
    ctx = make_ctx(series(overrides={"REVENUE": {2024: 1000.0 * 1.05 ** 6 * 0.95}}))  # revenue fell 5% in 2024
    ctx.docs = [build_doc("a", "10-K", "2024-12-31", "", "".join(f"<p>{t}</p>" for t in body))]
    ctx.llm = NS(structured=lambda *a, **k: Claims(claims=claims))
    return ctx


@pytest.mark.parametrize("stated", [5.0, -5.0])
def test_a_stated_fall_is_compared_by_size_not_sign(stated):
    from ffi.detectors.e_text import Claim, e04

    # "decreased 5%" is extracted as 5 or as -5; against an actual -5% the unsigned 5 read as a gap of 10 points
    quote = "Total net sales decreased 5% during 2024 compared to 2023."
    ctx = _mdna_ctx(quote, [Claim(metric="revenue", direction="decrease", magnitude_pct=stated, driver="", quote=quote)])
    (s,) = e04(ctx)
    assert s.raw_value == 0 and "1 verifiable MD&A claim checked" in s.rationale


def test_a_stated_fall_of_the_wrong_size_is_still_a_contradiction():
    from ffi.detectors.e_text import Claim, e04

    quote = "Total net sales decreased 15% during 2024 compared to 2023."
    ctx = _mdna_ctx(quote, [Claim(metric="revenue", direction="decrease", magnitude_pct=15.0, driver="", quote=quote)])
    (s,) = e04(ctx)
    assert s.raw_value == 1 and "fell 15.0%" in s.rationale


# ------------------------------------------------------------------ the focus year's own figures
def test_capex_ratio_of_an_earlier_year_is_never_shown_as_this_years():
    # D&A tagged as zero for the focus year: there is no ratio, and the result was last year's capex / D&A, labelled as this year's
    with pytest.raises(SkipDetector):
        run_one(make_ctx(series(overrides={"DA": {2024: 0.0}})), "C.EXP.05")
    (s,) = run_one(make_ctx(series()), "C.EXP.05")
    assert s.raw_value == pytest.approx(60 / 50)


def test_backlog_growth_needs_the_focus_years_backlog():
    rev = series()["REVENUE"]
    ctx = make_ctx(series(overrides={"RPO": {y: 0.5 * rev[y] for y in (2021, 2022, 2023)}}))
    with pytest.raises(SkipDetector, match="focus year"):
        run_one(ctx, "C.REV.10")


def test_a_year_without_backlog_ends_the_run_of_years():
    # history read by the test: 2021-2024. Backlog falls in 2022 and 2024 while revenue grows; 2023 is not reported, so the
    # focus year's gap stands alone (it was counted as the second of two consecutive years)
    rev = series()["REVENUE"]
    rpo = {2021: 0.50 * rev[2021], 2022: 0.45 * rev[2022], 2024: 0.40 * rev[2024]}
    base = {2020: 0.52 * rev[2020]}
    ctx = make_ctx(series(overrides={"RPO": {**base, **rpo}}))
    (s,) = run_one(ctx, "C.REV.10")
    assert "0 consecutive year(s)" in s.rationale and s.raw_value is None
    ctx = make_ctx(series(overrides={"RPO": {**base, **rpo, 2023: 0.42 * rev[2023]}}))
    (s,) = run_one(ctx, "C.REV.10")
    assert "3 consecutive year(s)" in s.rationale and s.raw_value > 0.03


# ------------------------------------------------------------------ each fact once, and small samples
def test_rounding_test_counts_each_reported_fact_once():
    ctx = make_ctx(series())
    d = ctx.store.df
    accn = ctx.A.prov[("TOTAL_ASSETS", ctx.focus)].accn
    rows = d[(d["accn"] == accn) & (d["unit"] == "USD") & (d["val"].abs() >= 1e4)]
    assert len(rows) > rows.drop_duplicates(["concept", "start", "end"]).shape[0] >= 30  # one concept feeds several line items
    (s,) = run_one(ctx, "A.16")
    assert f"of {rows.drop_duplicates(['concept', 'start', 'end']).shape[0]} monetary facts" in s.rationale


def test_benford_threshold_allows_for_the_sample_size():
    from ffi.detectors.d_stats import BENFORD1, benford_sampling_mad

    # simulated: 150 amounts that follow Benford's law exactly show a first-digit MAD of about 0.019 on average, above
    # Nigrini's "nonconformity" line of 0.015
    rng = np.random.default_rng(0)
    mads = [np.mean(np.abs(np.bincount(rng.choice(9, size=150, p=BENFORD1 / BENFORD1.sum()), minlength=9) / 150 - BENFORD1))
            for _ in range(400)]
    assert benford_sampling_mad(150) == pytest.approx(float(np.mean(mads)), rel=0.1)
    assert benford_sampling_mad(150) > 0.015 > benford_sampling_mad(5000)


def test_benford_conforming_sample_is_not_scored(monkeypatch):
    import pandas as pd

    from ffi.detectors import d_stats

    ctx = make_ctx(series())
    rec = ctx.A.prov[("TOTAL_ASSETS", ctx.focus)]
    scored_before = 0
    for seed in range(20):
        vals = 10 ** np.random.default_rng(seed).uniform(4, 9, size=150)  # log-uniform amounts follow Benford's law
        df = pd.DataFrame({"accn": rec.accn, "unit": "USD", "concept": [f"c{i}" for i in range(150)], "start": pd.NaT,
                           "end": ctx.focus, "val": vals})
        monkeypatch.setattr(ctx.store, "df", df)
        (s,) = d_stats.d05(ctx)
        scored_before += s.raw_value > 0.012  # the large-sample line gave these a severity
        assert s.severity == 0 and "by chance" in s.rationale, seed
    assert scored_before >= 15


# ------------------------------------------------------------------ command line and tools
def test_period_accepts_a_bare_year():
    from ffi.pipeline.run import pick_focus

    store = make_ctx(series()).store
    assert pick_focus(store, "2023") == pick_focus(store, "FY2023") == pick_focus(store, "fy 2023")
    with pytest.raises(ValueError, match="not found"):
        pick_focus(store, "2031")


def test_a_malformed_as_of_date_is_a_usage_error():
    from typer.testing import CliRunner

    from ffi.cli import app

    res = CliRunner().invoke(app, ["run", "AAPL", "--as-of", "30/06/2020", "--mode", "deterministic"])
    assert res.exit_code == 2 and "YYYY-MM-DD" in res.output


def test_ratio_tool_names_a_year_reported_on_another_basis():
    from ffi.agent.schemas import EvidenceTask
    from ffi.agent.tools import run_tool
    from ffi.xbrl.canonical import DIFFERENT_BASIS
    from tests.conftest import companyfacts
    from tests.unit.test_project_review import ctx_of, switch

    concept, label = next(iter(DIFFERENT_BASIS["AR"].items()))
    ctx = ctx_of(switch(companyfacts(series()), "AR", {2023}, concept))
    task = EvidenceTask(tool="ratio", purpose="receivables against revenue", hypothesis_index=0, numerator="AR", denominator="REVENUE")
    text = run_tool(ctx, task, "L0.1")["text"]
    assert f"FY2023 (accounts receivable, net: {label})" in text and "FY2024 12.0%" in text


def test_search_prefers_sentences_to_fragments():
    from ffi.agent.schemas import EvidenceTask
    from ffi.agent.tools import run_tool
    from ffi.text.document import build_doc

    # Moody's FY2025: the first hit was a heading broken off mid-sentence, 'Current market,'
    body = ["Item 1A. Risk Factors", "Current market,", "Adverse market conditions could reduce debt issuance and our revenue."]
    body += [f"Risk paragraph {n} about the business." for n in range(30)]
    ctx = make_ctx(series())
    ctx.docs = [build_doc("a", "10-K", "2024-12-31", "", "".join(f"<p>{t}</p>" for t in body))]
    task = EvidenceTask(tool="search_filing", purpose="market conditions", hypothesis_index=0, query="market conditions", section="any")
    text = run_tool(ctx, task, "L0.1")["text"]
    assert text.startswith('"Adverse market conditions') and "Current market," not in text
