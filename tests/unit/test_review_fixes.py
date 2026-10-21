"""Regression tests for the workflow review: NaN handling, valid JSON, audit coverage, band rounding, span recurrence,
facts-table matching, release window, peer and document caches, snapshot size."""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from ffi.detectors.base import load_all
from ffi.detectors.helpers import mk
from ffi.edgar.endpoints import FilingRef
from ffi.scoring.engine import Finding, audit_reported, band_of
from ffi.scoring.span import merge
from tests.conftest import make_ctx, series

load_all()


def test_nan_severity_and_percentiles_are_sanitised():
    ctx = NS(focus_label="FY2024", tuning={}, currency="EUR", evs=lambda *a, **k: [], doc=None)
    s = mk("C.EXP.07", ctx, raw_value=3, severity=float("nan"), percentile_peer=float("nan"), p_value=float("nan"), unit="USD")
    assert s.severity == 0.0 and not s.flagged
    assert s.percentile_peer is None and s.p_value is None
    assert s.unit == "EUR"  # monetary results carry the reporting currency


def test_report_json_is_strict():
    from ffi.report.render import to_json

    text = to_json({"a": float("nan"), "b": [np.float64("inf"), 1.5], "c": np.float32(2.0), "d": date(2024, 1, 1)})
    data = json.loads(text, parse_constant=lambda c: pytest.fail(f"non-standard JSON constant {c}"))
    assert data == {"a": None, "b": [None, 1.5], "c": 2.0, "d": "2024-01-01"}


def test_band_follows_the_index_as_shown():
    assert band_of(24.4) == "Low" and band_of(24.6) == "Elevated" and band_of(74.6) == "Severe"


def _f(fid: str, priority: float) -> Finding:
    f = Finding(fid, fid, "signal", priority, [], None, None, [], [], [])
    f.priority = priority
    return f


def test_a_finding_lifted_into_the_report_by_the_audit_is_audited_too():
    findings = [_f(f"f{i}", 1.0 - i * 0.01) for i in range(14)]
    audited = []

    def audit(f):
        audited.append(f.id)
        f.audit = NS(status="unverified" if f.id == "f0" else "verified")

    audit_reported(findings, audit, limit=12)
    assert "f12" in audited  # f0 dropped to half weight, so f12 is now reported
    assert "f13" not in audited


def _sig(did, period, sev=0.8):
    return NS(detector_id=did, period=period, severity=sev, raw_value=sev, unit="", flagged=True, tags=[])


def _year(label, findings):
    signals = {}
    for f in findings:
        for s in f.signals:
            signals.setdefault(s.detector_id, []).append(s)
    ctx = NS(focus_label=label, focus=pd.Timestamp(f"{label[2:]}-12-31"), signals=signals)
    return NS(ctx=ctx, score=NS(findings=findings, risk_index=10.0, band="Low", area_scores={}),
              results=[NS(signals=[s for f in findings for s in f.signals])])


def test_one_revision_seen_by_several_years_does_not_count_as_recurrence():
    # the same FY2021 revision is detected in the FY2022 and FY2023 analyses (A.14 looks back three years)
    rev = lambda: Finding("A.14@FY2021", "Silent restatement", "signal", 0.6, [_sig("A.14", "FY2021", 0.6)],  # noqa: E731
                          None, None, [], [], [])
    res = merge([_year("FY2022", [rev()]), _year("FY2023", [rev()]), _year("FY2024", [])])
    f = res.findings[0]
    assert f.years == ["FY2022", "FY2023"]
    assert f.score == pytest.approx(0.6)  # one event: no recurrence bonus


def test_facts_table_includes_changes_of_items_whose_label_has_a_comma():
    from ffi.agent.tools import facts_table

    ctx = make_ctx(series())
    ev = ctx.ev("AR")
    assert ev is not None and ev.label.startswith("Accounts receivable, net")
    f = Finding("C.AR.01@FY2024", "x", "signal", 0.8, [NS(detector_id="C.AR.01", period="FY2024", raw_value=None, expected=None,
                                                          unit="days", evidence=[ev])], None, None, [], [], [])
    sources = [r["source"] for r in facts_table(ctx, f)]
    assert any(s.startswith("Accounts receivable, net change") for s in sources)


def _filing(accn, form, filed, items="", report=None):
    return FilingRef(accn, form, date.fromisoformat(filed), date.fromisoformat(report) if report else None, "doc.htm", items)


def test_releases_stop_after_the_focus_years_results(monkeypatch):
    from ffi.pipeline import run as R

    filings = [_filing(f"r{i}", "8-K", d, "2.02,9.01") for i, d in enumerate(
        ["2023-02-01", "2023-05-01", "2023-08-01", "2023-11-01", "2024-02-01", "2024-05-01", "2024-08-01", "2024-11-01"])]
    monkeypatch.setattr(R.endpoints, "filing_index", lambda c, cik, f: [{"type": "EX-99.1", "document": "ex99.htm", "description": ""}])
    client = NS(get_text=lambda url: "<p>Revenue was $100 million.</p>")
    cache: dict = {}
    docs = R.load_releases(client, 1, filings, pd.Timestamp("2023-12-31"), date(2026, 1, 1), cache=cache)
    assert [d.accession for d in docs] == ["r0", "r1", "r2", "r3", "r4"]  # nothing about fiscal 2024
    again = R.load_releases(NS(get_text=lambda url: pytest.fail("parsed again")), 1, filings, pd.Timestamp("2023-12-31"),
                            date(2026, 1, 1), cache=cache)
    assert [d.accession for d in again] == ["r0", "r1", "r2", "r3", "r4"]


def test_officer_changes_are_read_for_the_filings_the_test_counts():
    from ffi.pipeline import run as R

    old = [_filing(f"o{i}", "8-K", f"2021-0{i + 1}-01", "5.02") for i in range(8)]  # before C.GOV.03's two-year window
    recent = [_filing("n1", "8-K", "2024-03-01", "5.02")]
    read = []

    def get_text(url):
        read.append(url)
        return "<p>Our Chief Financial Officer resigned.</p>"

    events = R.load_events(NS(get_text=get_text), 1, old + recent, date(2026, 1, 1), pd.Timestamp("2024-12-31"))
    assert len(read) == 1  # only the filing inside the window is downloaded
    desc = {e.accession: e.description for e in events}
    assert "CFO change" in desc["n1"] and desc["o0"] == "8-K Item 5.02"


def test_peer_panels_are_loaded_once_across_years(monkeypatch):
    from ffi.peers import select as S

    calls = []

    def fake_load(client, cik, max_age_s=None):
        calls.append(cik)
        raise S.EdgarError("offline")

    monkeypatch.setattr(S, "load_companyfacts", fake_load)
    cache = {7: ("Peer Co", None)}
    ps = S.select_peers(None, 1, [7, 8], 4, pd.DataFrame(), pd.Timestamp("2024-12-31"), 2, None, override=[7, 8], cache=cache)
    assert calls == [8] and {p.cik: p.name for p in ps.peers}[7] == "Peer Co"
    assert 8 not in cache  # a failed load is retried by the next year


def test_snapshot_drops_bulk_data_and_keeps_shared_objects_shared():
    import dataclasses

    from ffi.report import snapshot
    from ffi.text.document import TextDoc

    ctx = dataclasses.replace(make_ctx(series()), docs=[TextDoc("a", "10-K", "2024-12-31", "u", "x" * 100)])
    light = snapshot._lighten({})
    a, b = light(ctx), light(ctx)
    assert a is b and a.store is None and a.docs[0].text == "" and ctx.docs[0].text  # the live context is untouched


def test_kg_restatement_edges_link_consecutive_versions():
    from ffi.kg import build as kg_build

    ctx = make_ctx(series(), restated={("REVENUE", 2022): 1200.0})
    kg_build.build(ctx)
    e = ctx.kg.edges("RESTATES")
    rev = e[e["src"].str.startswith("version:REVENUE|2022-12-31")]
    assert len(rev) == 1 and rev.iloc[0]["props"]["delta"] == pytest.approx(1200e6 - 1000 * 1.05 ** 5 * 1e6)


def test_item_title_alone_is_not_an_officer_change():
    from ffi.pipeline.run import _officer_change_kind

    title = ("<p>Item 5.02 Departure of Directors or Certain Officers; Election of Directors; Appointment of Certain Officers; "
             "Compensatory Arrangements of Certain Officers.</p><p>The Board approved the 2025 bonus plan.</p>"
             "<p>/s/ Jane Doe, Chief Financial Officer</p>")
    assert "no senior finance" in _officer_change_kind(title)
    assert "CAO change" in _officer_change_kind("<p>We appointed Ben Borders as Principal Accounting Officer.</p>")


def test_one_transition_reported_twice_counts_once():
    from ffi.context import Event
    from ffi.detectors.base import REGISTRY, load_all

    evs = [Event("a", "OFFICER_CHANGE", date(2024, 3, 13), "a", "8-K Item 5.02 - CAO change"),
           Event("b", "OFFICER_CHANGE", date(2024, 7, 26), "b", "8-K Item 5.02 - CAO change")]
    ctx = NS(events_of=lambda t, within_days: evs, ev_event=lambda e: None, focus_label="FY2024", currency="USD")
    load_all()
    s = REGISTRY["C.GOV.03"].fn(ctx)[0]
    assert "0 chief financial officer, 0 chief executive and 1 chief accounting officer" in s.rationale
