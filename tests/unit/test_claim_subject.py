"""E.04 checks only company-wide MD&A claims; claims about a named segment, region or product are skipped."""

import pytest

from ffi.detectors.e_text import Claim, _measure_core, _part_of_business


def claim(metric: str, quote: str) -> Claim:
    return Claim(metric=metric, direction="decrease", magnitude_pct=None, driver="", quote=quote)


@pytest.mark.parametrize("metric,quote", [
    ("revenue", "Wearables, Home and Accessories net sales decreased during 2025 compared to 2024"),
    ("gross margin", "Products gross margin percentage decreased during 2025 compared to 2024"),
    ("revenue", "Greater China net sales decreased during 2025"),
    ("revenue", "iPhone net sales increased during 2025"),
])
def test_part_of_business_skipped(metric, quote):
    assert _part_of_business(claim(metric, quote))


@pytest.mark.parametrize("metric,quote", [
    ("revenue", "Total net sales increased 6% during 2025 compared to 2024"),
    ("revenue", "Net sales for 2025 increased"),
    ("revenue", "The Company's total net sales increased"),
    ("operating income", "Operating income increased to $5.2 billion"),
])
def test_company_wide_checked(metric, quote):
    assert not _part_of_business(claim(metric, quote))


def test_measure_core_ignores_presentation_qualifiers():
    assert _measure_core("Free cash flow -- TTM") == "free cash flow"
    assert _measure_core("WW net sales -- Y/Y growth, excluding F/X") == "net sales"
    assert _measure_core("Adjusted EBITDA") == "ebitda"


@pytest.mark.parametrize("quote", [
    "Cost of revenue in 2025 increased $6.01 billion, or 20%, compared to 2024",
    "Research and development expenses increased 14% in 2024 compared to 2023",
    "Marketing and sales expenses decreased 8% in 2024",
])
def test_other_expense_lines_are_not_total_operating_expenses(quote):
    assert _part_of_business(claim("operating expenses", quote))


def test_total_operating_expenses_claim_is_checked():
    assert not _part_of_business(claim("operating expenses", "Total costs and expenses increased 12% in 2024"))
    assert not _part_of_business(claim("operating expenses", "Operating expenses increased 9% compared to 2023"))


def test_skipped_segment_claim_does_not_hide_the_company_wide_claim():
    from types import SimpleNamespace as NS

    from ffi.detectors.e_text import Claims, e04
    from ffi.text.document import build_doc
    from tests.conftest import make_ctx, series

    segment = "Wearables, Home and Accessories net sales decreased during 2024 compared to 2023."
    total = "Total net sales decreased during 2024 compared to 2023."
    body = ["Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations", segment, total]
    body += [f"Discussion paragraph {n} about the results of operations for the year." for n in range(60)]
    body += ["Item 8. Financial Statements and Supplementary Data", "Statements."]
    ctx = make_ctx(series())  # revenue rose 5% in 2024
    ctx.docs = [build_doc("a", "10-K", "2024-12-31", "", "".join(f"<p>{t}</p>" for t in body))]
    # the extractor lists the segment claim first, with the same metric, direction and magnitude as the total one
    ctx.llm = NS(structured=lambda *a, **k: Claims(claims=[claim("revenue", segment), claim("revenue", total)]))
    (sig,) = e04(ctx)
    assert sig.raw_value == 1 and "1 verifiable MD&A claim checked" in sig.rationale
