"""Deterministic independent audit and its correction loop (max 3 rounds)."""

from __future__ import annotations

from ffi.audit.deterministic import MAX_ROUNDS, run_audit
from ffi.pipeline.detect import run_detectors
from ffi.scoring.engine import apply_audit, score, template_narrative
from tests.conftest import make_ctx, series


def _findings():
    ctx = make_ctx(series(overrides={"AR": {2024: 450.0}, "CFO": {2022: 40.0, 2023: 30.0, 2024: 10.0}}))
    sc = score(ctx, run_detectors(ctx, only={"C.AR.01", "C.CF.01", "B.01", "B.03", "B.05"}))
    for f in sc.findings:
        f.narrative = template_narrative(f)
    return ctx, sc


def test_clean_findings_verify_in_round_one():
    ctx, sc = _findings()
    res = run_audit(ctx, sc.findings[0], template_narrative)
    assert res.status == "verified" and len(res.rounds) == 1
    assert {c.name for c in res.rounds[0].checks} >= {"evidence_present", "fact_values", "narrative"}


def test_tampered_fact_value_is_removed_and_reverified():
    ctx, sc = _findings()
    f = sc.findings[0]
    fact = next(e for e in f.evidence if e.kind == "fact")
    fact.value = fact.value * 1.37  # value no longer matches the filing
    res = run_audit(ctx, f, template_narrative)
    assert res.status == "verified_after_corrections"
    assert any("removed unverifiable fact" in c for c in res.rounds[0].corrections)
    assert fact not in f.evidence


def test_invented_narrative_number_is_replaced():
    ctx, sc = _findings()
    f = sc.findings[0]
    f.narrative = "Receivables are overstated by $987.6B."
    f.narrative_source = "agent"
    res = run_audit(ctx, f, template_narrative)
    assert res.status == "verified_after_corrections"
    assert "987.6" not in f.narrative and "failed audit" in f.narrative_source


def test_uncorroborated_pattern_is_downgraded():
    ctx, sc = _findings()
    f = next(f for f in sc.findings if f.kind == "pattern")
    for s in f.signals[1:]:
        s.flagged, s.suppressed_by = False, "test"
    res = run_audit(ctx, f, template_narrative)
    assert f.kind == "signal" and "(single source)" in f.title
    assert res.status in ("verified_after_corrections", "unverified")


def test_uncorrectable_finding_is_unverified_within_max_rounds_and_loses_weight():
    ctx, sc = _findings()
    f = sc.findings[0]
    f.evidence = []
    for s in f.signals:
        s.evidence = []  # nothing left to attach
    before = sc.risk_index
    f.audit = run_audit(ctx, f, template_narrative)
    assert f.audit.status == "unverified" and len(f.audit.rounds) <= MAX_ROUNDS
    apply_audit(sc)
    assert sc.risk_index <= before


def test_auditor_sees_each_tool_result_whole():
    from types import SimpleNamespace as NS

    from ffi.audit.llm_auditor import evidence_pack

    # Moody's 2025: a search result of three quotes; the third reached the auditor cut to "Current market,"
    quotes = ['"In July 2025, the FASB issued ASU 2025-05 ... ' + "x" * 380 + '"',
              '"In order to determine an estimate of expected credit losses, receivables are segmented ..."',
              '"Current market, economic and government factors could negatively impact the volume of debt securities issued."']
    entry = {"id": "L2.1", "tool": "search_filing", "text": " | ".join(quotes)}
    finding = NS(evidence=[], signals=[], detectors=[], materiality=None)
    pack, ids = evidence_pack(finding, [entry])
    assert "L2.1" in ids and quotes[2] in pack


def test_cited_per_share_value_must_match_without_a_half_unit_floor():
    import numpy as np

    from ffi.audit.deterministic import _fact_matches
    from ffi.detectors.base import EvidenceRef

    node = "fact:us-gaap:EarningsPerShareDiluted|USD/shares|2025-01-01|2025-12-31|0001-25-000001"
    idx = {("0001-25-000001", "EarningsPerShareDiluted", "2025-12-31"): np.array([3.20])}

    def ev(value, unit="USD/shares", node_id=node):
        return EvidenceRef(node_id=node_id, kind="fact", label="EPS (diluted), FY2025", accession="0001-25-000001",
                           concept="us-gaap:EarningsPerShareDiluted", value=value, unit=unit)

    assert _fact_matches(idx, ev(3.20)) and not _fact_matches(idx, ev(3.64))  # 3.64 used to pass within 0.5
    usd = node.replace("EarningsPerShareDiluted|USD/shares", "Revenues|USD")
    idx[("0001-25-000001", "Revenues", "2025-12-31")] = np.array([1_000_000.0])
    revenue = EvidenceRef(node_id=usd, kind="fact", label="Revenue, FY2025", accession="0001-25-000001",
                          concept="us-gaap:Revenues", value=1_000_000.4, unit="USD")
    assert _fact_matches(idx, revenue)  # an amount still matches within rounding
