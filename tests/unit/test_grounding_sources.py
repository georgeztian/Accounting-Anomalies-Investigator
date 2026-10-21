"""Grounding gate: the false rejections seen on META FY2025 drafts must pass; real fabrications must still fail."""

from __future__ import annotations

from ffi.agent import grounding


def test_numbers_from_rationale_and_quoted_text_are_allowed():
    rationale = "Retained earnings moved $18.67B vs net income less dividends $55.04B; difference -$36.36B."
    ledger_text = '"$18.40 billion of taxes paid related to net share settlement" and "$72.22 billion of capital expenditures"'
    allowed = grounding.numbers_in(rationale) + grounding.numbers_in(ledger_text)
    text = "The $36.36B gap compares with $18.40B of share-settlement taxes; capex was $72.22 billion."
    rep = grounding.check(text, allowed, docs=[])
    assert rep.unsupported_numbers == [], rep.unsupported_numbers


def test_standard_references_are_not_amounts():
    rep = grounding.check("Leases follow ASC 842 and guarantees ASC 460; see Note 5 and Rule 10b5-1.", [], docs=[])
    assert rep.numbers_checked == 0


def test_invented_numbers_still_fail():
    rep = grounding.check("The gap is $12.34B.", grounding.numbers_in("Gap $36.36B."), docs=[])
    assert rep.unsupported_numbers == ["$12.34B"]


def test_quote_pairing_ignores_text_between_quotes():
    text = 'It said "the Company entered into an arrangement to co-develop" [L1.1], so the guarantee part has "no data at all in the ledger".'
    qs = grounding.quotes_in(text)
    assert all("[L1.1]" not in q and "so the guarantee" not in q for q in qs)
    assert any(q.startswith("the Company entered") for q in qs)


def test_quoting_the_detector_label_is_allowed():
    rationale = "Purchase obligations and guarantees $103.77B (28.4% of assets) grew 204.1% vs revenue 22.2%."
    rep = grounding.check('The detector\'s "purchase obligations and guarantees" label covers one fact.', [], docs=[],
                          known_text=["Off-balance-sheet commitments growth", rationale])
    assert rep.unverified_quotes == []


def test_quote_pairing_recovers_from_an_unmatched_mark():
    text = 'It notes the “Venture”). As a result, the Venture\'s assets are not consolidated" and "the Company entered into an arrangement to co-develop".'
    assert not any(q.startswith(".") for q in grounding.quotes_in(text))


def test_calendar_weeks_are_not_amounts():
    rep = grounding.check("The one-day difference fits an ordinary 52/53-week fiscal calendar and a 13-week quarter.", [], docs=[])
    assert rep.numbers_checked == 0


def test_magnitude_matches_regardless_of_sign():
    rep = grounding.check("Inventory grew about 3.5 points slower than cost of revenue.", [-0.035], docs=[])
    assert rep.unsupported_numbers == []
    assert grounding.check("A gap of $1.6B.", [2.75e9, 1.15e9], docs=[]).unsupported_numbers == ["$1.6B"]  # arithmetic stays out
