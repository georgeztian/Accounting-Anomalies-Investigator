"""Line-item search patterns for the investigation, and sentence capitals after plain-language rewrites."""

import re

from ffi.kg.build import item_pattern
from ffi.report.plain import plain


def test_payables_and_commitments_have_patterns():
    assert re.search(item_pattern("AP"), "Accounts payable increased due to timing of payments", re.I)
    assert re.search(item_pattern("PURCHASE_OBLIGATIONS"), "unconditional purchase obligations", re.I)


def test_items_without_curated_pattern_fall_back_to_label():
    pat = item_pattern("VALUATION_ALLOWANCE")
    assert pat and re.search(pat, "the deferred tax valuation allowance was released", re.I)
    assert item_pattern("NOT_AN_ITEM") is None


def test_sentence_starts_are_capitalised():
    assert plain("A low base. operating cash flow rose.", terms=False) == "A low base. Operating cash flow rose."


def test_abbreviations_and_lowercase_names_are_kept():
    assert plain("Apple vs. peers is high (e.g. inventory). iPhone sales rose in the U.S. market.", terms=False) == \
        "Apple vs. peers is high (e.g. inventory). iPhone sales rose in the U.S. market."


def test_written_dates_are_not_amounts():
    from ffi.agent.grounding import check

    assert check("The year ended September 28, 2024, a 52-week year.", [], []).ok
    assert not check("Sales of $28 billion in May.", [], []).ok


def test_gaap_definition_never_lands_inside_non_gaap():
    out = plain("Non-GAAP diluted EPS $14.94 vs GAAP diluted EPS $13.67 for the full year FY2025.")
    assert "Non-GAAP (the official accounting rules)" not in out and "non-gaap (adjusted figures" in out.lower()
    assert "GAAP (the official accounting rules) diluted EPS $13.67" in out and "full year fiscal 2025" in out
