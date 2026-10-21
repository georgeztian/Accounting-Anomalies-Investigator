"""End-to-end run against live SEC EDGAR (deselect with -m 'not network')."""

from __future__ import annotations

import json
import re

import pytest

from ffi.pipeline.run import RunOptions, run

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def aapl(tmp_path_factory):
    out = tmp_path_factory.mktemp("reports")
    return run(RunOptions(company="Apple Inc. (AAPL)", mode="deterministic", out_dir=out, run_id="AAPL_TEST"))


def test_run_produces_all_reports(aapl):
    assert {"html", "md", "json"} <= set(aapl.files)
    assert all(p.exists() and p.stat().st_size > 1000 for p in aapl.files.values())
    assert (aapl.run_dir / "manifest.json").exists()


def test_no_detector_errors(aapl):
    errors = [(r.id, r.reason) for r in aapl.results if r.status == "error"]
    assert errors == []


def test_report_is_self_contained_and_neutral(aapl):
    html = aapl.files["html"].read_text(encoding="utf-8")
    assert not re.search(r"<script[^>]+src=", html)
    assert not re.search(r"<link[^>]+stylesheet", html)
    body = re.sub(r"<[^>]+>", " ", html)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", body) if re.search(r"\bfraud", s, re.I)]
    assert all("not evidence of fraud" in s for s in sentences)


def test_every_fact_evidence_links_to_a_filing(aapl):
    data = json.loads(aapl.files["json"].read_text(encoding="utf-8"))
    facts = [e for s in data["signals"] for e in s["evidence"] if e["kind"] == "fact"]
    assert facts
    assert all(e["doc_url"] and e["doc_url"].startswith("https://www.sec.gov/Archives/") for e in facts)


def test_point_in_time_run(tmp_path):
    res = run(RunOptions(company="AAPL", mode="deterministic", out_dir=tmp_path, as_of=__import__("datetime").date(2020, 6, 30),
                         formats=("json",), run_id="AAPL_PIT"))
    assert res.ctx.focus_label == "FY2019"
    assert all(f.filing_date <= res.ctx.as_of for f in res.ctx.filings if f.form == "10-K" and f.accession in
               {d.accession for d in res.ctx.docs})
