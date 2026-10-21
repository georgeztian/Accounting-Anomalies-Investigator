"""Guards and core mechanics: venv guard, resolver parsing, point-in-time store, KG as_of, scoring, grounding, linter."""

from __future__ import annotations

import sys
from datetime import date, datetime

import pandas as pd
import pytest

from ffi._env_guard import require_virtualenv
from ffi.agent import grounding
from ffi.kg.store import DuckNXGraphStore
from ffi.report.linter import lint
from ffi.resolve.resolver import normalize_name, parse_input
from ffi.scoring.engine import benjamini_hochberg, score
from ffi.text.document import build_doc
from ffi.text.verify import locate
from tests.conftest import make_ctx, run_one, series


# ---------------------------------------------------------------- environment
def test_venv_guard_exits_outside_virtualenv(monkeypatch):
    monkeypatch.setattr(sys, "base_prefix", sys.prefix)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("FFI_ALLOW_SYSTEM_PYTHON", raising=False)
    with pytest.raises(SystemExit) as e:
        require_virtualenv()
    assert e.value.code == 2


def test_venv_guard_passes_inside_virtualenv():
    require_virtualenv()  # the test suite itself runs in the project venv


# ---------------------------------------------------------------- resolution
@pytest.mark.parametrize("text,name,ticker", [
    ("Apple Inc. (AAPL)", "Apple Inc.", "AAPL"),
    ("AAPL", None, "AAPL"),
    ("Berkshire Hathaway (BRK.B)", "Berkshire Hathaway", "BRK-B"),
    ("Microsoft Corporation", "Microsoft Corporation", None),
])
def test_parse_input(text, name, ticker):
    p = parse_input(text)
    assert (p.name, p.ticker) == (name, ticker)


def test_normalize_name_strips_legal_suffixes():
    assert normalize_name("The Coca-Cola Company, Inc.") == "coca cola"


# ---------------------------------------------------------------- point in time
def test_as_of_hides_later_filings():
    data = series()
    late = make_ctx(data, as_of=date(2025, 12, 31))
    early = make_ctx(data, as_of=date(2022, 6, 30))
    assert late.focus.year == 2024
    assert early.focus.year == 2021  # the FY2022 10-K was filed in Feb 2023


def test_first_reported_vs_latest():
    data = series()
    ctx = make_ctx(data, restated={("REVENUE", 2023): 999.0})
    p = [x for x in ctx.A.periods if x.year == 2023][0]
    assert ctx.A.get("REVENUE", p) == pytest.approx(data["REVENUE"][2023] * 1e6)
    assert ctx.A_latest.get("REVENUE", p) == pytest.approx(999.0 * 1e6)


def test_kg_as_of_filter():
    kg = DuckNXGraphStore(":memory:", as_of=date(2024, 1, 1))
    kg.upsert_nodes([{"id": "a", "type": "Fact", "recorded_at": datetime(2023, 5, 1)},
                     {"id": "b", "type": "Fact", "recorded_at": datetime(2024, 5, 1)}])
    assert set(kg.nodes()["id"]) == {"a"}


# ---------------------------------------------------------------- text
HTML = """<html><body><div style="display:none"><ix:header>hidden</ix:header></div>
<p>Table of Contents</p><p>Item 7. Management's Discussion</p><p>Item 8. Financial Statements</p>
<p>Item 7. Management&#8217;s Discussion and Analysis</p>
<p>Net sales increased 10% to $5.0 billion, driven primarily by higher volumes.</p>
<p>Item 8. Financial Statements and Supplementary Data</p><p>Note 1 &#8211; Summary of Significant Accounting Policies</p>
<p>Revenue is recognized when control transfers to the customer.</p></body></html>"""


def test_sections_skip_table_of_contents_and_quotes_verify():
    doc = build_doc("0001", "10-K", "2024-12-31", "u", HTML)
    assert "item7" in doc.sections and "item8" in doc.sections
    mdna = doc.section_text("item7")
    assert "Net sales increased 10%" in mdna
    p = doc.section_paragraphs("item7")[-1]
    assert doc.verify(p.text, p.start, p.end)
    assert any(x.note and x.note.startswith("Note 1") for x in doc.paragraphs)
    loc = locate(doc, "Net sales increased 10% to $5.0 billion, driven  primarily by higher volumes")
    assert loc is not None


# ---------------------------------------------------------------- scoring
def test_benjamini_hochberg_monotone():
    q = benjamini_hochberg([0.01, 0.04, 0.03, 0.5])
    assert q[0] <= q[2] <= q[1] <= q[3]
    assert all(0 <= x <= 1 for x in q)


def test_pattern_requires_corroboration():
    from ffi.pipeline.detect import run_detectors

    hot = make_ctx(series(overrides={"AR": {2024: 450.0}, "CFO": {2022: 40.0, 2023: 30.0, 2024: 10.0}}))
    sc = score(hot, run_detectors(hot, only={"C.AR.01", "C.CF.01", "B.01", "B.03", "B.05"}))
    ids = [f.id for f in sc.findings]
    assert "premature_revenue" in ids
    f = next(f for f in sc.findings if f.id == "premature_revenue")
    assert f.materiality is not None and f.materiality.multiple > 0
    clean = make_ctx(series())
    sc2 = score(clean, run_detectors(clean, only={"C.AR.01", "C.CF.01", "B.01"}))
    assert sc2.risk_index < sc.risk_index


def test_fdr_suppresses_weak_pvalue_flags():
    ctx = make_ctx(series())
    from ffi.detectors.helpers import mk
    from ffi.pipeline.detect import DetectorResult

    run_one(ctx, "C.AR.01")
    s = mk("C.AR.01", ctx, raw_value=1.0, severity=0.6, p_value=0.4)
    sc = score(ctx, [DetectorResult("C.AR.01", "DSO", "C", "ok", signals=[s])])
    assert not s.flagged and "false-discovery" in (s.suppressed_by or "")
    assert sc.findings == []


# ---------------------------------------------------------------- grounding & language
def test_grounding_accepts_known_numbers_and_rejects_invented():
    allowed = [1.94e9, 0.139, 35.0]
    ok = grounding.check("The gap is $1.94B, or 13.9% of pre-tax income, with DSO of 35 days in FY2024 [L0.1].", allowed, [])
    assert ok.ok, ok.unsupported_numbers
    bad = grounding.check("The gap is $7.2B.", allowed, [])
    assert not bad.ok and bad.unsupported_numbers


def test_grounding_rejects_fabricated_quote():
    doc = build_doc("0001", "10-K", "2024-12-31", "u", HTML)
    rep = grounding.check('Management said "we recognised all revenue at shipment without exception".', [], [doc])
    assert rep.unverified_quotes


def test_linter_rewrites_accusations_but_keeps_disclaimer():
    text, hits = lint("This looks like fraud. This report is not evidence of fraud.")
    assert hits == ["fraud"]
    assert "not evidence of fraud" in text
    assert text.startswith("This looks like inconsistent with expectations")


def test_pd_available():
    assert pd.__version__
