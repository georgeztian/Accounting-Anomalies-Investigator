"""Report rendering from a synthetic run (no network)."""

from __future__ import annotations

import json
import re

import pandas as pd

from ffi.pipeline.detect import run_detectors
from ffi.pipeline.run import RunOutput
from ffi.report.render import render
from ffi.scoring.engine import score
from tests.conftest import make_ctx, series


def test_render_all_formats(tmp_path):
    ctx = make_ctx(series(overrides={"AR": {2024: 450.0}, "CFO": {2022: 40.0, 2023: 30.0, 2024: 10.0}}))
    results = run_detectors(ctx)
    sc = score(ctx, results)
    out = RunOutput(ctx, results, sc, tmp_path, {"run_id": "SYN_TEST", "kg": {}, "llm": None})
    files = render(out, ("html", "md", "json"))
    html = files["html"].read_text(encoding="utf-8")
    assert "Synthetic Co" in html and "Risk index" in html
    assert "{{" not in html and not re.search(r"\bnan\b|NaN", html)
    assert html.count('class="card finding"') == len(sc.findings[:12]) >= 1
    md = files["md"].read_text(encoding="utf-8")
    assert md.startswith("# Synthetic Co")
    # client-facing: no internal test codes, run ids, concept names, file names or identifiers
    body = re.sub(r"<[^>]+>", " ", html.split("<body>", 1)[1])
    for text in (body, md):
        assert not re.search(r"\b[A-F]\.(?:[A-Z]{2,4}\.)?\d{2}\b", text), re.findall(r"\b[A-F]\.(?:[A-Z]{2,4}\.)?\d{2}\b", text)[:5]
        assert "SYN_TEST" not in text and "us-gaap:" not in text and "CIK" not in text
        assert not re.search(r"manifest|\.json|\.duckdb|ANTHROPIC|\bLLM\b", text)
    # every test result is explained: what it checks, why it matters, how to read it
    assert md.count("*What it checks:*") >= 60 and "## Glossary" in md
    # nothing was set aside (no AI here): the section, which says "These results were set aside", is left out
    assert "set aside" not in md.lower() and 'id="setaside"' not in html and "#setaside" not in html
    data = json.loads(files["json"].read_text(encoding="utf-8"))
    assert data["risk_index"] == sc.risk_index
    assert {d["status"] for d in data["detectors"]} <= {"ok", "skipped", "error"}
    assert all(d.get("what") and d.get("why") and d.get("how_to_read") for d in data["detectors"])
    assert not [d for d in data["detectors"] if d["status"] == "error"], [d for d in data["detectors"] if d["status"] == "error"]


def test_explanation_material_states_the_impact_is_before_tax():
    from ffi.detectors.base import ImpactEstimate
    from ffi.materiality.impact import assess
    from ffi.report.explain import _material
    from ffi.scoring.engine import Finding

    ctx = make_ctx(series())
    imp = ImpactEstimate(50e6, 100e6, 150e6, "ratio_reversion", "test")
    sig = run_detectors(ctx)[0].signals[0]
    f = Finding("x", "x", "signal", 0.8, [sig], imp, assess(ctx, imp), [], [], [])
    material = json.loads(_material(ctx, f))
    # the range is a pre-tax effect; labelled "effect on profit" it was read as an after-tax figure
    assert "estimated_effect_on_profit" not in material and material["estimated_pre_tax_effect"].endswith("before tax")


def test_officer_changes_are_never_read_as_cash_flow():
    from types import SimpleNamespace as NS

    from ffi.report import plain as P
    from tests.conftest import run_one

    ctx = make_ctx(series())
    ctx.events = [NS(type="OFFICER_CHANGE", date=pd.Timestamp("2024-06-01").date(), accession=None, url=None,
                     id="e1", description="8-K Item 5.02; CFO change")]
    (s,) = run_one(ctx, "C.GOV.03")
    # Moody's 2025: "1 CFO, 1 CEO" was shown to clients as "1 operating cash flow, 1 chief executive"
    assert "operating cash flow" not in P.plain(s.rationale) and "1 chief financial officer" in P.plain(s.rationale)


def test_a_sentence_joined_to_its_assessment_keeps_one_mark():
    from ffi.report import plain as P

    text = "Write-offs used up reserves built earlier, as intended.: Plausible but untested."
    assert P.plain(text) == "Write-offs used up reserves built earlier, as intended: Plausible but untested."


def test_montier_label_is_true_when_net_income_is_below_cash_flow():
    from ffi.detectors.b_scores import MONTIER_LABELS

    # Alphabet 2023: net income $73.8B, operating cash flow $101.8B; the test fired as NI - CFO rose from -$31.5B
    assert "ahead" not in MONTIER_LABELS["ni_cfo_gap_widening"] and "rising" in MONTIER_LABELS["ni_cfo_gap_widening"]


def test_no_event_placeholder_is_dropped_once_the_investigation_explains():
    from types import SimpleNamespace as NS

    from ffi.report.model import _alternatives
    from ffi.scoring.engine import NO_EVENT

    written = "A documented change in estimate: server lives extended from 4 to 6 years"
    assert _alternatives(NS(alternatives=[written, NO_EVENT])) == [written]
    assert _alternatives(NS(alternatives=[NO_EVENT])) == [NO_EVENT]


def test_saved_officer_counts_and_parenthesised_labels_read_correctly():
    from ffi.report import plain as P

    # the wording saved runs hold (Moody's, Waste Management 2025)
    assert P.plain("9 Item 5.02 filing(s) in two years: 1 CFO, 1 CEO and 1 chief accounting officer transition(s).").startswith(
        "9 executive-change filing(s) (Form 8-K) in two years: 1 chief financial officer, 1 chief executive and 1 chief")
    assert P.plain("Adoption of a new standard (ASU 2025-05).: Not supported.") == "Adoption of a new standard (ASU 2025-05): Not supported."
