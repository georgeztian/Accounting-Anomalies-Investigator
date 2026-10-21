"""A result the investigation shows to be a test error (and the auditor confirms) leaves the list and the risk index."""

from types import SimpleNamespace as NS

import pandas as pd

from ffi.agent import runner
from ffi.scoring.engine import Finding, ScoreResult, apply_set_aside, index_of
from ffi.scoring.span import SpanResult
from ffi.scoring.span import apply_set_aside as span_set_aside


def sig(did, period="FY2024", sev=0.8):
    return NS(detector_id=did, period=period, severity=sev, name=did, raw_value=sev, unit="", flagged=True, tags=[])


def finding(fid, kind="pattern", did="C.NG.01", **kw):
    return Finding(fid, fid, kind, 0.7, [sig(did)], None, None, [], [], [], **kw)


def score(findings, contrib):
    return ScoreResult(index_of(contrib), "x", None, findings, {}, {}, [], [], {}, sorted(contrib.items(), key=lambda kv: -kv[1]))


def test_set_aside_leaves_list_and_index():
    keep, bad = finding("cfo_inflation"), finding("misleading_metrics", set_aside="a quarter was compared with a full year")
    contrib = {"pattern:cfo_inflation": 0.4, "pattern:misleading_metrics": 0.5, "B.06": 0.1}
    sc = score([bad, keep], contrib)
    before = sc.risk_index
    assert apply_set_aside(sc) == [bad]
    assert sc.findings == [keep] and sc.set_aside == [bad]
    assert dict(sc.contributions) == {"pattern:cfo_inflation": 0.4, "B.06": 0.1}
    assert sc.risk_index == index_of({"pattern:cfo_inflation": 0.4, "B.06": 0.1}) < before


def test_standalone_result_removes_its_test_contribution():
    bad = finding("A.02@FY2024", kind="signal", did="A.02", set_aside="gross PP&E leaves out construction in progress")
    sc = score([bad], {"A.02": 0.3, "B.06": 0.1})
    apply_set_aside(sc)
    assert dict(sc.contributions) == {"B.06": 0.1}


def draft(**kw):
    return {"summary": "S", "explanation": "E", "alternatives": [], "questions": [], "cited_ledger_ids": [],
            "test_assessment": "likely_test_error", "test_assessment_reason": "quarter vs full year [L0.1]", **kw}


def apply(f, d, check):
    case = {}
    runner._apply(f, {"status": "published", "draft": d, "verification": {"test_error_check": check}, "ledger": []}, case)
    return case


def test_test_error_needs_the_auditors_confirmation():
    f = finding("misleading_metrics")
    apply(f, draft(), "unsupported")
    assert f.set_aside == "" and f.narrative_source == "agent"  # claimed but not confirmed: the finding stands
    apply(f, draft(), "supported")
    assert f.set_aside and f.narrative_source == "agent: set aside as probable test error"


def test_a_real_anomaly_is_never_set_aside():
    f = finding("misleading_metrics")
    apply(f, draft(test_assessment="real_anomaly_possible", test_assessment_reason=""), "supported")
    assert f.set_aside == ""


def test_a_claim_without_a_reason_is_not_applied():
    f = finding("misleading_metrics")
    apply(f, draft(test_assessment_reason=""), "supported")
    assert f.set_aside == ""


def test_backfill_replaces_each_set_aside_with_the_next_candidate(monkeypatch):
    seen = []

    def fake_cases(pairs, run_dir, run_id, ui=None):
        seen.append([f.id for _, f in pairs])
        for _, f in pairs:
            if f.id in ("a", "c"):
                f.set_aside = "test error"
        return [{"finding_id": f.id, "status": "published"} for _, f in pairs]

    monkeypatch.setattr(runner, "investigate_cases", fake_cases)
    fs = [finding(i) for i in "abcdef"]
    runner.investigate_with_backfill([(None, f) for f in fs], 3, None, "run")
    assert seen == [["a", "b", "c"], ["d", "e"]]  # two set aside -> two more investigated


def test_multi_year_set_aside_removes_it_from_each_year_and_the_period_index():
    bad = finding("misleading_metrics", set_aside="figures not like for like")
    bad.years = ["FY2023", "FY2024"]
    keep = finding("cfo_inflation", did="C.CF.03")
    y0 = NS(ctx=NS(focus_label="FY2023", focus=pd.Timestamp("2023-12-31")),
            score=score([bad], {"pattern:misleading_metrics": 0.6}))
    y1 = NS(ctx=NS(focus_label="FY2024", focus=pd.Timestamp("2024-12-31")),
            score=score([bad, keep], {"pattern:misleading_metrics": 0.6, "pattern:cfo_inflation": 0.3}))
    span = SpanResult(["FY2023", "FY2024"], [{"period": "FY2023", "risk_index": y0.score.risk_index, "band": "x", "findings": 1, "top": "t"},
                                             {"period": "FY2024", "risk_index": y1.score.risk_index, "band": "x", "findings": 2, "top": "t"}],
                      50.0, "Elevated", [bad, keep])
    span_set_aside(span, [y0, y1])
    assert span.findings == [keep] and span.set_aside == [bad]
    assert y0.score.risk_index == 0.0 and y1.score.risk_index == index_of({"pattern:cfo_inflation": 0.3})
    assert span.yearly[0]["risk_index"] == 0.0 and span.risk_index < 50.0
