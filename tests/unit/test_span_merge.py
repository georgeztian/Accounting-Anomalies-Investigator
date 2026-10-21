"""Multi-year review: findings merge across years, recurrence raises the score, and the period index."""

from types import SimpleNamespace as NS

import pandas as pd

from ffi.scoring.engine import Finding
from ffi.scoring.span import merge, merged_score, period_index, years_text


def sig(did: str, period: str, sev: float = 0.8):
    return NS(detector_id=did, period=period, severity=sev, raw_value=sev, unit="", flagged=True, tags=[])


def finding(fid: str, kind: str, period: str, did: str, score: float = 0.6) -> Finding:
    return Finding(fid, fid.replace("_", " "), kind, score, [sig(did, period, score)], None, None, [], ["Why?"], ["Checked"])


def year(label: str, findings: list[Finding], index: float = 20.0):
    signals = {}
    for f in findings:
        for s in f.signals:
            signals.setdefault(s.detector_id, []).append(s)
    ctx = NS(focus_label=label, focus=pd.Timestamp(f"{label[2:]}-12-31"), signals=signals)
    return NS(ctx=ctx, score=NS(findings=findings, risk_index=index, band="Low", area_scores={"cash_flow": 0.5}),
              results=[NS(signals=[s for f in findings for s in f.signals])])


def test_standalone_result_joins_its_pattern_from_another_year():
    years = [year("FY2022", [finding("C.CF.03@FY2022", "signal", "FY2022", "C.CF.03")]),
             year("FY2023", []),
             year("FY2024", [finding("cfo_inflation", "pattern", "FY2024", "C.CF.03")])]
    res = merge(years)
    assert [f.id for f in res.findings] == ["cfo_inflation"]
    f = res.findings[0]
    assert f.years == ["FY2022", "FY2024"] and f.kind == "pattern"
    assert [t["flagged"] for t in f.timeline] == [True, False, True]
    assert res.rep_year["cfo_inflation"] == 2  # the most recent occurrence represents it
    assert {s.period for s in f.signals} == {"FY2022", "FY2024"}


def test_unrelated_results_stay_separate():
    years = [year("FY2023", [finding("C.INV.02@FY2023", "signal", "FY2023", "C.INV.02")]),
             year("FY2024", [finding("C.CF.03@FY2024", "signal", "FY2024", "C.CF.03")])]
    assert sorted(f.id for f in merge(years).findings) == ["C.CF.03", "C.INV.02"]


def test_recurrence_raises_the_score():
    assert merged_score(0.6, 1, 10) == 0.6
    assert merged_score(0.6, 10, 10) == 0.8
    assert 0.6 < merged_score(0.6, 4, 10) < 0.8


def test_period_index_between_mean_and_worst_year():
    assert period_index([10, 10, 40]) == 30.0
    assert period_index([]) == 0.0


def test_years_text_groups_consecutive_years():
    periods = [f"FY{y}" for y in range(2016, 2026)]
    f = NS(years=["FY2019", "FY2021", "FY2022", "FY2023", "FY2024", "FY2025"])
    assert years_text(f, periods) == "Flagged in 6 of 10 years (FY2019, FY2021-FY2025); still present in FY2025."
    assert years_text(NS(years=["FY2017"]), periods) == "Flagged in 1 of 10 years (FY2017); last seen in FY2017."
