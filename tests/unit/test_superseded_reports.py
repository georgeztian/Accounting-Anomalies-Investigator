"""One final report per company and fiscal year: a finished run deletes earlier runs for the same company and year."""

from __future__ import annotations

import json

from ffi.pipeline.run import _remove_superseded_reports


def _run(folder, cik, report=True):
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps({"company": {"cik": cik}}), encoding="utf-8")
    if report:
        (folder / "report.html").write_text("<html></html>", encoding="utf-8")
    return folder


def test_old_runs_for_same_company_and_year_are_removed(tmp_path):
    new = _run(tmp_path / "BRK-B_FY2025_5e2fac", 1067983)
    old = _run(tmp_path / "BRK-B_FY2025_51381d", 1067983)
    failed = tmp_path / "BRK-B_FY2025_aaaaaa"  # crashed before writing a manifest or report
    failed.mkdir()
    (failed / "kg.duckdb").write_bytes(b"x")
    other_year = _run(tmp_path / "BRK-B_FY2024_bbbbbb", 1067983)
    other_company = _run(tmp_path / "META_FY2025_cccccc", 1326801)
    same_ticker_other_cik = _run(tmp_path / "BRK-B_FY2025_dddddd", 999)  # never delete another registrant's report

    _remove_superseded_reports(new, 1067983)

    assert new.exists() and other_year.exists() and other_company.exists() and same_ticker_other_cik.exists()
    assert not old.exists() and not failed.exists()
