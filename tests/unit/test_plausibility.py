"""No result counts when its value or its inputs are physically impossible."""

from types import SimpleNamespace as NS

from ffi.detectors.base import Signal
from ffi.detectors.plausibility import hold_back, incoherent_inputs


def ctx(**vals):
    base = {"TOTAL_ASSETS": 450e9, "REVENUE": 350e9}
    base.update(vals)
    return NS(a=lambda item, k=0: base.get(item, float("nan")), currency="USD")


def sig(unit, value, family="C", did="C.AR.01", sev=0.9, evidence=()):
    return Signal(did, "t", family, "area", "FY2024", value, None, sev, sev > 0, "rationale", unit=unit, evidence=list(evidence), tags=[])


def test_impossible_values_are_held_back():
    for unit, v in (("days", 5000.0), ("years", 400.0), ("relative gap", 250.0), ("USD", 9e12)):
        s = hold_back(sig(unit, v), ctx())
        assert not s.flagged and s.severity == 0.0 and "implausible" in s.tags and "Held back" in s.rationale, unit


def test_plausible_values_pass_untouched():
    for unit, v in (("days", 120.0), ("years", 12.0), ("relative gap", 0.4), ("USD", 8e9)):
        s = hold_back(sig(unit, v), ctx())
        assert s.flagged and s.severity == 0.9 and not s.suppressed_by, unit


def test_a_part_larger_than_its_whole_holds_back_the_tests_that_read_it():
    c = ctx(PPE_NET=500e9)  # net PP&E above total assets: a tagging or scale error
    # the reason reaches the client report: the line item as the statements name it, never the internal id ("ppe net", "ar")
    assert incoherent_inputs(c)["PPE_NET"] == "PP&E, net ($500.00B) is larger than total assets ($450.00B)"
    ev = NS(kind="fact", label="PP&E, net, FY2024")
    s = hold_back(sig("years", 12.0, evidence=[ev]), c)
    assert not s.flagged and "larger than total assets" in s.suppressed_by
    other = hold_back(sig("years", 12.0, evidence=[NS(kind="fact", label="Revenue, FY2024")]), ctx(PPE_NET=500e9))
    assert other.flagged  # a test that does not read the bad input is unaffected


def test_integrity_tests_are_exempt():
    s = hold_back(sig("USD", 1e9, family="A", did="A.02"), ctx(PPE_NET=500e9))
    assert s.flagged  # finding relationships that do not hold is what they are for


def test_zero_results_are_left_alone():
    s = hold_back(sig("days", 5000.0, sev=0.0), ctx())
    assert not s.suppressed_by


def test_held_back_row_is_not_described_as_explained():
    from ffi.report.model import test_families

    s = hold_back(sig("days", 5000.0), ctx())
    (fam,) = [f for f in test_families([NS(family="C", id="C.AR.01", signals=[s], status="ok", reason=None)], 0.5)[0] if f["rows"]]
    row = fam["rows"][0]
    # the status says "Held back"; the detail used to add "Explained by: implausible result held back: ..." as well
    assert row["status_label"] == "Held back" and "Explained by" not in row["detail"] and row["detail"].count("Held back") == 1
