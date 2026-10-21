"""Company profiles, foreign filers and reporting currency."""

import pandas as pd

from ffi.detectors.base import load_all
from ffi.detectors.stats import CURRENCY, fmt_money
from ffi.profiles import PROFILES, classify_sic, for_company
from ffi.resolve.resolver import CompanyRef, ResolutionError, check_eligibility
from ffi.text.document import build_doc
from ffi.xbrl.facts import FactStore


def test_classify_sic():
    assert classify_sic(6199) == "bank" and classify_sic(6021) == "bank"
    assert classify_sic(6211) == "broker" and classify_sic(6311) == "insurance"
    assert classify_sic(6798) == "real_estate" and classify_sic(6770) == "shell"
    assert classify_sic(3571) == "industrial" and classify_sic(None) == "industrial"


def test_profile_exclusions_reference_real_detectors():
    ids = set(load_all())
    for p in PROFILES.values():
        assert set(p.excluded) <= ids, p.key
    assert not PROFILES["industrial"].excluded
    bank = for_company(6021)
    assert "B.01" in bank.excluded and "C.INV.01" in bank.excluded and "A.01" not in bank.excluded
    assert bank.skip_reason("B.01").startswith("Not applicable to bank or lender")


def test_foreign_and_currency_adjustments():
    p = for_company(3711, annual_form="20-F", currency="JPY", taxonomy="ifrs-full")
    assert {"D.06", "D.04", "B.08"} <= set(p.excluded)
    assert any("IFRS" in n for n in p.notes) and any("JPY" in n for n in p.notes)


def _ref():
    return CompanyRef(cik=1, ticker="X", name="X Co", sic=6021, sic_description="", fiscal_year_end="1231")


def test_eligibility_accepts_financial_and_foreign_filers():
    ref = _ref()
    check_eligibility(ref, {"filings": {"recent": {"form": ["10-K", "10-K", "8-K"]}}})
    assert ref.annual_form == "10-K"
    ref = _ref()
    check_eligibility(ref, {"filings": {"recent": {"form": ["20-F", "6-K"]}}})
    assert ref.annual_form == "20-F" and ref.warnings  # one annual report: warned, not rejected


def test_eligibility_rejects_no_annual_report():
    try:
        check_eligibility(_ref(), {"filings": {"recent": {"form": ["S-1", "10-Q"]}}})
    except ResolutionError as e:
        assert e.exit_code == 4
    else:
        raise AssertionError("expected ResolutionError")


def _raw(rows):
    df = pd.DataFrame(rows, columns=["taxonomy", "concept", "unit", "start", "end", "val", "accn", "fy", "fp", "form", "filed", "frame"])
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c])
    df["days"] = (df["end"] - df["start"]).dt.days
    return df


def test_ifrs_currency_and_6k_annual_facts():
    rows = []
    for y, rev in ((2023, 100.0), (2024, 120.0)):
        s, e = f"{y}-01-01", f"{y}-12-31"
        rows += [("ifrs-full", "Revenue", "EUR", s, e, rev * 1e9, f"a{y}", y, "FY", "6-K", f"{y + 1}-02-20", None),
                 ("ifrs-full", "Revenue", "USD", s, e, rev * 1.1e9, f"a{y}", y, "FY", "6-K", f"{y + 1}-02-20", None),
                 ("ifrs-full", "Assets", "EUR", None, e, 900e9, f"a{y}", y, "FY", "6-K", f"{y + 1}-02-20", None)]
    st = FactStore(1, _raw(rows))
    assert st.currency == "EUR" and st.taxonomy == "ifrs-full"
    A = st.annual()
    assert A.get("REVENUE", pd.Timestamp("2024-12-31")) == 120e9  # EUR values only, from 6-K annual facts


def test_money_uses_run_currency():
    tok = CURRENCY.set("EUR")
    try:
        assert fmt_money(2.5e9) == "€2.50B"
    finally:
        CURRENCY.reset(tok)
    assert fmt_money(2.5e9) == "$2.50B"


def test_20f_sections_map_to_10k_keys():
    html = "".join(f"<p>{t}</p>" for t in [
        "Item 3. Key Information", "D. Risk Factors", "Our business faces risks.",
        "Item 5. Operating and Financial Review and Prospects", "Revenue grew strongly this year.",
        "Item 18. Financial Statements", "See pages F-1.", "Consolidated Statements of Income"])
    doc = build_doc("a", "20-F", "2024-12-31", "", html)
    assert "Revenue grew" in doc.section_text("item7")
    assert "risks" in doc.section_text("item1a")
    assert "Consolidated Statements" in doc.section_text("item8")
