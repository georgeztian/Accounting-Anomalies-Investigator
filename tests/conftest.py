"""Shared fixtures: synthetic companies built as companyfacts JSON, so detectors run on known inputs."""

from __future__ import annotations

import os
from datetime import date

import pandas as pd
import pytest

os.environ.setdefault("SEC_USER_AGENT", "ffi-tests tests@example.com")

from ffi.context import RunContext  # noqa: E402
from ffi.kg.store import DuckNXGraphStore  # noqa: E402
from ffi.resolve.resolver import CompanyRef  # noqa: E402
from ffi.xbrl.canonical import BY_ID  # noqa: E402
from ffi.xbrl.facts import FactStore  # noqa: E402

BASE = {
    "REVENUE": 1000.0, "COGS": 600.0, "SGA": 200.0, "OPERATING_INCOME": 200.0, "PRETAX_INCOME": 190.0,
    "INCOME_TAX": 40.0, "NET_INCOME": 150.0, "CFO": 170.0, "CFI": -60.0, "CFF": -90.0, "CAPEX": 60.0,
    "DA": 50.0, "DEPRECIATION": 50.0, "AR": 120.0, "INVENTORY": 100.0, "CASH": 200.0, "CASH_RESTRICTED_TOTAL": 200.0,
    "CURRENT_ASSETS": 450.0, "PPE_NET": 400.0, "PPE_GROSS": 800.0, "ACCUM_DEP": 400.0, "TOTAL_ASSETS": 1000.0,
    "CURRENT_LIABILITIES": 250.0, "TOTAL_LIABILITIES": 500.0, "EQUITY": 500.0, "LIABILITIES_AND_EQUITY": 1000.0,
    "RETAINED_EARNINGS": 300.0, "LONG_TERM_DEBT": 200.0, "AP": 90.0, "SHARES_DILUTED": 100.0, "EPS_DILUTED": 1.5,
    "EPS_BASIC": 1.52, "DIVIDENDS_PAID": 0.0, "INTEREST_EXPENSE": 10.0,
}
SCALE = 1e6


def series(years=range(2017, 2025), growth=0.05, overrides: dict | None = None) -> dict[str, dict[int, float]]:
    """A steady company growing `growth` a year; overrides: {item: {year: value_in_millions}}."""
    out: dict[str, dict[int, float]] = {}
    for i, y in enumerate(years):
        g = (1 + growth) ** i
        for item, v in BASE.items():
            out.setdefault(item, {})[y] = v * g if item not in ("SHARES_DILUTED", "EPS_DILUTED", "EPS_BASIC") else v
    # keep cash roll-forward and retained earnings consistent year to year
    ys = list(years)
    for a, b in zip(ys[:-1], ys[1:], strict=True):
        out["CFF"][b] = out["CASH_RESTRICTED_TOTAL"][b] - out["CASH_RESTRICTED_TOTAL"][a] - out["CFO"][b] - out["CFI"][b]
        out["RETAINED_EARNINGS"][b] = out["RETAINED_EARNINGS"][a] + out["NET_INCOME"][b]
        out["EQUITY"][b] = out["EQUITY"][a] + out["NET_INCOME"][b]
        out["TOTAL_LIABILITIES"][b] = out["TOTAL_ASSETS"][b] - out["EQUITY"][b]
    for y in ys:
        out["EPS_DILUTED"][y] = out["NET_INCOME"][y] / out["SHARES_DILUTED"][y]  # millions / millions of shares
        out["EPS_BASIC"][y] = out["EPS_DILUTED"][y] * 1.01
    for item, vals in (overrides or {}).items():
        out.setdefault(item, {}).update(vals)
    return out


def companyfacts(data: dict[str, dict[int, float]], restated: dict[tuple[str, int], float] | None = None) -> dict:
    """Build a companyfacts-shaped dict. FY ends Dec 31; each 10-K is filed Feb 20 of the next year and also
    carries the prior year's comparative (re-reported value from `restated` if given)."""
    facts: dict[str, dict] = {"us-gaap": {}, "dei": {}}
    for item, vals in data.items():
        it = BY_ID[item]
        tax, concept = (it.concepts[0].split(":") if ":" in it.concepts[0] else ("us-gaap", it.concepts[0]))
        unit = it.unit
        rows = []
        for y, v in vals.items():
            scale = 1 if unit in ("USD/shares",) else SCALE
            for filing_year, val in ((y + 1, v), (y + 2, (restated or {}).get((item, y), v))):
                accn = f"0000000000-{filing_year % 100:02d}-000001"
                row = {"end": f"{y}-12-31", "val": val * scale, "accn": accn, "fy": filing_year - 1, "fp": "FY",
                       "form": "10-K", "filed": f"{filing_year}-02-20"}
                if it.kind == "D":
                    row["start"] = f"{y}-01-01"
                rows.append(row)
        facts.setdefault(tax, {}).setdefault(concept, {"units": {}})["units"].setdefault(unit, []).extend(rows)
    return {"cik": 1, "entityName": "Synthetic Co", "facts": facts}


def make_ctx(data, restated=None, as_of=date(2025, 12, 31)) -> RunContext:
    cf = companyfacts(data, restated)
    store = FactStore.from_companyfacts(1, cf, as_of)
    A, AL, Q = store.annual("first"), store.annual("latest"), store.quarterly("first")
    focus = A.periods[-1]
    ref = CompanyRef(cik=1, ticker="SYN", name="Synthetic Co", sic=3571, sic_description="Test", fiscal_year_end="1231")
    kg = DuckNXGraphStore(":memory:", as_of=as_of)
    return RunContext(company=ref, store=store, A=A, A_latest=AL, Q=Q, filings=[], focus=focus, as_of=as_of, kg=kg,
                      tuning={"scoring": {"fdr_q": 0.10}, "agent": {"recursion_limit": 40}})


@pytest.fixture
def clean_ctx():
    return make_ctx(series())


def run_one(ctx, detector_id):
    from ffi.detectors.base import load_all

    spec = load_all()[detector_id]
    sigs = spec.fn(ctx)
    ctx.signals[detector_id] = sigs
    return sigs


@pytest.fixture
def ts():
    return pd.Timestamp
