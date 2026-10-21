"""Each detector fires on a synthetic company built to trigger it, and stays quiet on a clean one."""

from __future__ import annotations

import pytest

from ffi.detectors.base import SkipDetector
from tests.conftest import make_ctx, run_one, series


def flagged(sigs):
    return any(s.flagged for s in sigs)


def test_clean_company_integrity_passes(clean_ctx):
    for did in ("A.01", "A.03", "A.05", "A.07", "A.12"):
        assert not flagged(run_one(clean_ctx, did)), did


def test_a01_balance_sheet_break():
    ctx = make_ctx(series(overrides={"LIABILITIES_AND_EQUITY": {2024: 1300.0}}))
    sigs = run_one(ctx, "A.01")
    assert flagged(sigs)
    assert sigs[0].evidence and sigs[0].evidence[0].accession


def test_a03_cash_rollforward_break():
    ctx = make_ctx(series(overrides={"CFO": {2024: 400.0}}))
    assert flagged(run_one(ctx, "A.03"))


def test_a14_silent_restatement_detected():
    data = series()
    ctx = make_ctx(data, restated={("NET_INCOME", 2023): data["NET_INCOME"][2023] * 0.8})
    sigs = run_one(ctx, "A.14")
    s = sigs[0]
    assert s.flagged and "unlabeled_revision" in s.tags
    assert s.impact is not None and s.impact.mid > 0
    assert {e.label.split(" as ")[-1] for e in s.evidence} == {"first reported", "re-reported"}


def test_a14_quiet_without_revisions(clean_ctx):
    assert not flagged(run_one(clean_ctx, "A.14"))


def test_c_ar01_dso_jump_flags_with_impact():
    ctx = make_ctx(series(overrides={"AR": {2024: 450.0}}))
    sigs = run_one(ctx, "C.AR.01")
    assert flagged(sigs)
    assert sigs[0].impact is not None and sigs[0].impact.mid > 0


def test_c_ar01_quiet(clean_ctx):
    assert not flagged(run_one(clean_ctx, "C.AR.01"))


def test_c_inv02_inventory_build():
    ctx = make_ctx(series(overrides={"INVENTORY": {2023: 130.0, 2024: 260.0}}))
    sigs = run_one(ctx, "C.INV.02")
    assert flagged(sigs) and sigs[0].impact is not None


def test_c_cf01_cfo_far_below_income():
    ctx = make_ctx(series(overrides={"CFO": {2022: 40.0, 2023: 30.0, 2024: 10.0}}))
    assert flagged(run_one(ctx, "C.CF.01"))


def test_c_exp03_useful_life_extension():
    ctx = make_ctx(series(overrides={"DEPRECIATION": {2024: 30.0}, "DA": {2024: 30.0}}))
    sigs = run_one(ctx, "C.EXP.03")
    assert sigs[0].raw_value > sigs[0].expected
    assert sigs[0].impact is not None


def test_b01_beneish_reacts_to_receivables_and_accruals():
    ctx = make_ctx(series(overrides={"AR": {2024: 450.0}, "CFO": {2024: -50.0}}))
    hot = run_one(ctx, "B.01")[0]
    cold = run_one(make_ctx(series()), "B.01")[0]
    assert hot.raw_value > cold.raw_value
    assert hot.raw_value > -1.78


def test_b02_dechow_f_score_is_finite(clean_ctx):
    s = run_one(clean_ctx, "B.02")[0]
    assert s.raw_value is not None and s.raw_value > 0


def test_needs_text_detector_skips_without_doc(clean_ctx):
    with pytest.raises(SkipDetector):
        run_one(clean_ctx, "C.GOV.05")


def test_registry_has_all_families():
    from ffi.detectors.base import load_all

    fams = {s.family for s in load_all().values()}
    assert fams == {"A", "B", "C", "D", "E", "F"}
    assert len(load_all()) >= 100


def test_parallel_detector_run_matches_sequential():
    from ffi.pipeline.detect import run_detectors

    data = series(overrides={"AR": {2024: 450.0}})
    seq = run_detectors(make_ctx(data), workers=1)
    par = run_detectors(make_ctx(data), workers=4)
    key = lambda rs: [(r.id, r.status, [round(s.severity, 6) for s in r.signals]) for r in rs]  # noqa: E731
    assert key(seq) == key(par)
