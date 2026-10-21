"""Hard rule: a quarterly figure is never compared with a full-year figure, or the reverse."""

from types import SimpleNamespace as NS

from ffi.detectors.e_text import ADJUSTED_RE, _eps_rows, _header_kinds, _row_figures
from ffi.detectors.textnum import check_amounts, statement_period

HEADER = "| Quarters Ended\nDecember 31,\n| YoY%\nInc/(Dec)\n| Years Ended\nDecember 31,\n| YoY%\nInc/(Dec)\n2025 | 2024\n2025 | 2024\n"
# American Express, fourth-quarter 2024 release (no column header captured): quarter, prior quarter, change, year, prior year
AXP_FY2024 = ("Diluted Earnings Per Common Share (EPS)2\n$3.04 | $2.62 | 16% | $14.01 | $11.21 | 25%\n"
              "Full Year Adjusted EPS Excluding Transaction Gain3\n$13.35 | $11.21\n| 19% |\n")
# fourth-quarter 2025 release: headline, guidance, a two-period table with one cell per line, a year-only appendix
AXP_FY2025 = ("FY 2025 EARNINGS PER SHARE ROSE TO $15.38\nCOMPANY EXPECTS FY 2026 REVENUE GROWTH OF 9% TO 10% AND EPS OF $17.30 TO $17.90\n"
              + HEADER + "Diluted Earnings Per Common Share (EPS)2\n| $3.53\n| $3.04\n| 16%\n| $15.38\n| $14.01\n| 10%\n"
              "Adjusted EPS Excluding Transaction Gain3\n| $15.38\n| $13.35\n| 15%\n"
              "Appendix I\nReconciliation of Adjusted EPS\n| Years Ended\nDecember 31,\n| YoY%\n2025 | 2024\n"
              "GAAP Diluted EPS\n| $15.38 |\n$14.01 | 10%\n")


def figures(text: str):
    adjusted, gaap = [], []
    for label, vals, s, e in _eps_rows(text):
        figs = _row_figures(label, vals, s, e, _header_kinds(text, s))
        (adjusted if ADJUSTED_RE.search(label) else gaap).extend((f.value, f.period) for f in figs)
    return adjusted, gaap


def test_full_year_adjusted_eps_pairs_with_full_year_gaap():
    adjusted, gaap = figures(AXP_FY2024)
    assert adjusted == [(13.35, "FY")]
    # a 4-value row without a column header in a fourth-quarter release: which column is which is not stated, so it is
    # skipped and the full-year figure is compared with the full-year EPS in the filed statements instead
    assert gaap == []


def test_two_period_table_and_year_only_appendix():
    adjusted, gaap = figures(AXP_FY2025)
    assert (15.38, "Q") not in adjusted  # two values under a quarter-and-year header: period unclear, skipped
    assert (3.53, "Q") in gaap and (15.38, "FY") in gaap
    assert all(v not in (17.30, 17.90) for v, _ in gaap)  # guidance is not a reported figure


def test_quarterly_release_without_header():
    adjusted, gaap = figures("Diluted EPS\n$3.49 | $3.30 | 6% | $10.97 | $8.59 | 28%\nAdjusted EPS\n$3.49 | $3.30\n")
    # HARD RULE: no label, no header, no sentence saying the period: nothing is assumed (it used to be read as the quarter)
    assert gaap == [] and adjusted == []


def test_statement_period():
    assert statement_period("Fourth-quarter revenue was $17.2 billion") == "Q"
    assert statement_period("Full-year revenue was $65.9 billion") == "FY"
    assert statement_period("Revenue was $17.2 billion") is None
    assert statement_period("Revenue for the fourth quarter and full year") is None


def test_amounts_compared_only_with_their_own_period():
    known = {"REVENUE": {"Q": [17.2e9], "FY": [65.9e9]}}
    checked, bad = check_amounts("Full-year total revenues were $65.9 billion.", known, quarter_ok=True)
    assert (checked, bad) == (1, [])
    # a quarterly total within 5% of a full-year figure is not read as a revised full-year figure
    checked, bad = check_amounts("Fourth quarter revenues were $63.0 billion.", {"REVENUE": {"Q": [], "FY": [65.9e9]}}, quarter_ok=True)
    assert (checked, bad) == (0, [])
    # no period stated and both periods possible: skipped
    checked, _ = check_amounts("Total revenues were $65.9 billion.", known, quarter_ok=True)
    assert checked == 0


def test_header_words_inside_sentences_are_not_headers():
    legal = ("Annual Report on Form 10-K for the year ended December 31, 2024, Quarterly Reports on Form 10-Q for the "
             "quarters ended March 31, June 30 and September 30, 2025.\nAppendix I\n| Years Ended\nDecember 31,\n2025 | 2024\n")
    text = legal + "Adjusted Diluted EPS Excluding the Impact of Accertify Gain\n| $15.38\n| $13.35\n"
    adjusted, _ = figures(text)
    assert adjusted == [(15.38, "FY")]


def test_nine_month_columns_are_never_a_year_or_a_quarter():
    text = "| Quarters Ended\nSeptember 30,\n| Nine Months Ended\nSeptember 30,\n2025 | 2024\n2025 | 2024\nAdjusted EPS\n| $11.85\n| $10.31\n"
    adjusted, _ = figures(text)
    assert adjusted == []  # two values under a quarter and nine-month header: unclear


def test_total_revenue_concept_before_contract_revenue():
    from ffi.xbrl.canonical import BY_ID

    c = BY_ID["REVENUE"].concepts
    # banks and card issuers tag contract revenue (ASC 606) as a part of total revenue net of interest expense
    assert c.index("RevenuesNetOfInterestExpense") < c.index("RevenueFromContractWithCustomerExcludingAssessedTax")


def test_basis_from_table_title_and_mixed_rows_skipped():
    from ffi.detectors.e_text import _table_basis

    text = ("For the quarter, GAAP and non-GAAP earnings per diluted share were $0.76 and $0.81, respectively.\n"
            "GAAP |\n($ in millions) | Q1 FY26 | Q4 FY25\nNet income | $18,775 | $22,091\nDiluted earnings per share* | $0.76 | $0.89\n"
            "Non-GAAP |\n($ in millions) | Q1 FY26 | Q4 FY25\nNet income | $19,894 | $22,066\nDiluted earnings per share* | $0.81 | $0.89\n")
    rows = _eps_rows(text)
    assert [_table_basis(text, s) for _, _, s, _ in rows[1:]] == ["gaap", "adjusted"]
    from ffi.detectors.e_text import MIXED_BASIS_RE

    assert MIXED_BASIS_RE.search(rows[0][0])  # "GAAP and non-GAAP ... were" states both bases: skipped
    assert not MIXED_BASIS_RE.search("Adjusted Diluted EPS") and not MIXED_BASIS_RE.search("GAAP Diluted EPS")


# Intel, fourth-quarter 2025 release (2026-01-22), news summary: one bullet states the quarter and the year
INTEL_Q4_2025 = ("News Summary\n▪Fourth-quarter revenue was $13.7 billion, down 4% year-over-year (YoY). Full-year revenue was $52.9 billion, flat YoY.\n"
                 "▪Fourth-quarter earnings (loss) per share (EPS) attributable to Intel was $(0.12); non-GAAP EPS attributable to Intel was $0.15. "
                 "Full-year EPS attributable to Intel was $(0.06); non-GAAP EPS attributable to Intel was $0.42.\n")


def test_decimal_points_do_not_end_a_sentence():
    from ffi.detectors.e_text import NG_EPS_RE
    from ffi.detectors.textnum import sentence_end, sentence_start

    periods = []
    for m in NG_EPS_RE.finditer(INTEL_Q4_2025):
        sentence = INTEL_Q4_2025[sentence_start(INTEL_Q4_2025, m.start()): sentence_end(INTEL_Q4_2025, m.end(), 80)]
        periods.append((m.group(1), statement_period(sentence)))
    # $0.15 is the quarter's non-GAAP EPS and $0.42 the year's; the "$(0.12)" decimal point used to cut the sentence
    # before "Fourth-quarter", so $0.15 was read as a full-year figure and compared with the full-year GAAP loss
    assert periods == [("0.15", "Q"), ("0.42", "FY")]


def test_prose_bullet_is_not_an_eps_table_row_and_losses_keep_their_column():
    from ffi.detectors.e_text import _amounts, _is_prose

    rows = _eps_rows(INTEL_Q4_2025)
    assert rows and all(_is_prose(INTEL_Q4_2025[s:e]) for _, _, s, e in rows)
    assert _amounts("| $(0.12) | $ (0.06) | ($0.03) | $0.15") == [-0.12, -0.06, -0.03, 0.15]
    assert _amounts("| (0.10) | 3.64 | 1,014.94", bare=True) == [-0.10, 3.64, 1014.94]
    table = "Diluted EPS\n| $ (0.12)\n| $ 0.15\n"
    (label, vals, s, e), = _eps_rows(table)
    assert vals == [-0.12, 0.15] and not _is_prose(table[s:e])


def test_four_columns_under_a_lone_year_header_do_not_read_the_first_as_the_year():
    # Johnson & Johnson Q4 2025 release: EPS (diluted) $2.10 | $1.41 | $11.03 | $5.79 (quarter, prior quarter, year, prior year)
    # with only a "Fiscal year" header found above the row: $2.10 was taken as the full-year EPS
    text = "Fiscal Year Ended\nDecember 28, 2025\nEPS (diluted)\n| $2.10\n| $1.41\n| $11.03\n| $5.79\n"
    adjusted, gaap = figures(text)
    assert gaap == [] and adjusted == []


def test_guidance_tables_and_long_sentences_are_not_reported_eps_rows():
    from ffi.detectors.e_text import _in_guidance_table

    # UnitedHealth Q2 2025: the outlook table gave net earnings per share of $14.65 to $16.00 for the year
    text = "2025 Outlook\nNet Earnings per Share\n| $14.65\n| $16.00\n"
    (_, _, s, _), = _eps_rows(text)
    assert _in_guidance_table(text, s)
    results = "2025 Outlook\n| Quarters Ended\nJune 30,\nDiluted EPS\n| $4.08\n| $4.54\n"
    (_, _, s, _), = _eps_rows(results)
    assert not _in_guidance_table(results, s)


def test_effect_sentences_and_prior_year_reconciliations_are_not_reported_eps():
    from ffi.detectors.e_text import IMPACT_RE, LABEL_TAIL_RE, PERIOD_HEAD_RE, RANGE_RE

    # Salesforce Q4 FY26 release: "gains (losses) on strategic investments impacted GAAP diluted EPS by $0.81" is an effect
    assert IMPACT_RE.search("gains on strategic investments impacted GAAP diluted EPS by $0.81") and LABEL_TAIL_RE.search("EPS by")
    # UnitedHealth: outlook amounts are "At least $14.65"; Walmart's release repeats "Three Months Ended April 30, 2024"
    assert RANGE_RE.search("Net Earnings per Share | At least $14.65 | At least $16.00")
    assert PERIOD_HEAD_RE.search("Three Months Ended April 30, 20241\nDiluted earnings per share:\nReported EPS | $0.63")


# Moody's 3Q 2025 release opens with a summary box: labels and figures in a grid, read line by line. "$10.26" is the
# nine-month GAAP EPS, sitting under the column label "Adjusted Diluted EPS"; the quarter's adjusted EPS was $3.92.
MCO_BOX = ("MCO Diluted EPS | MCO Adjusted Diluted EPS2\nMCO FY 2025 Projected1\n3Q 2025\n3Q 2025\nDiluted EPS\n"
           "$3.60 ⇑ 23%\n$3.92 ⇑ 22%\n$13.15 to $13.40\nYTD 2025\nYTD 2025\nAdjusted Diluted EPS2\n$10.26 ⇑ 13%\n"
           "$11.30 ⇑ 15%\n$14.50 to $14.75\n")
# its reconciliation tables print per-share amounts without a currency sign
MCO_TABLE = ("Below is a reconciliation of these measures to their most directly comparable U.S. GAAP measures:\n"
             "| Three Months Ended December 31,\nYear Ended December 31,\nAmounts in millions | 2025 | 2024 | 2025 | 2024\n"
             "Diluted EPS attributable to Moody's common shareholders | 3.41 | 2.17 | 13.67 | 11.26\n"
             "Net gain on divestiture of business and related costs\n(0.10) | — | (0.10) | —\n"
             "Adjusted Diluted EPS | 3.64 | 2.62 | 14.94 | 12.47\n")


def test_summary_box_without_readable_periods_gives_no_figures():
    adjusted, gaap = figures(MCO_BOX)
    assert (10.26, "Q") not in adjusted  # the old default read the nine-month GAAP EPS as the quarter's adjusted EPS
    assert all(p != "Q" or v in (3.60, 3.92) for v, p in adjusted + gaap)


def test_reconciliation_rows_without_currency_signs_pair_by_period():
    adjusted, gaap = figures(MCO_TABLE)
    assert (3.64, "Q") in adjusted and (14.94, "FY") in adjusted
    assert (3.41, "Q") in gaap and (13.67, "FY") in gaap


def _ctx_with_filed_eps():
    import pandas as pd

    from ffi.xbrl.facts import Panel

    qs = [pd.Timestamp(d) for d in ("2025-03-31", "2025-06-30", "2025-09-30")]
    Q = Panel("Q", pd.DataFrame({"EPS_DILUTED": [3.46, 3.21, 3.60]}, index=qs), {p: str(p.date()) for p in qs})
    ys = [pd.Timestamp("2024-12-31"), pd.Timestamp("2025-12-31")]
    A = Panel("A", pd.DataFrame({"EPS_DILUTED": [11.26, 13.67]}, index=ys), {ys[0]: "FY2024", ys[1]: "FY2025"})

    class Ctx:
        focus, focus_label = ys[1], "FY2025"

        def a(self, item, k=0):
            return A.get(item, ys[1 - k])

    c = Ctx()
    c.A, c.Q = A, Q
    return c


def test_hard_check_refuses_a_figure_from_another_period_whatever_the_layout():
    from types import SimpleNamespace as NS

    from ffi.detectors.e_text import EpsFigure, period_mismatch

    ctx, rel = _ctx_with_filed_eps(), NS(period_end="2025-10-22")
    # the original Moody's error: $10.26 equals the filed nine-month GAAP EPS ($3.46 + $3.21 + $3.60)
    assert "fiscal year to date" in period_mismatch(ctx, rel, EpsFigure(10.26, "Q", 0, 0), 3.60)
    # the original American Express error: a full-year adjusted figure against a quarter is on the year's scale
    assert period_mismatch(ctx, rel, EpsFigure(13.35, "Q", 0, 0), 3.60)
    # a genuine same-period gap is never refused
    assert period_mismatch(ctx, rel, EpsFigure(3.92, "Q", 0, 0), 3.60) is None
    assert period_mismatch(ctx, rel, EpsFigure(5.40, "Q", 0, 0), 3.60) is None  # +50%, a real anomaly, still compared


def test_hard_check_includes_the_same_quarter_a_year_earlier():
    import pandas as pd

    from ffi.detectors.e_text import EpsFigure, period_mismatch

    ctx = _ctx_with_filed_eps()
    q_prior = pd.Timestamp("2024-09-30")  # last year's third quarter, as a repeated prior-year table would show it
    ctx.Q.values.loc[q_prior, "EPS_DILUTED"] = 3.20
    ctx.Q.values = ctx.Q.values.sort_index()
    from types import SimpleNamespace as NS

    why = period_mismatch(ctx, NS(period_end="2025-10-22"), EpsFigure(3.20, "Q", 0, 0), 3.60)
    assert why and "a year earlier" in why


def test_release_whose_gaap_figure_disagrees_with_the_filing_is_not_used():
    from types import SimpleNamespace as NS

    from ffi.detectors.e_text import EpsFigure, release_tagging_problem

    ctx, rel = _ctx_with_filed_eps(), NS(period_end="2025-10-22")
    # the parser tagged the nine-month GAAP EPS as the quarter: the filed Q3 EPS is $3.60, so the release is not trusted
    assert release_tagging_problem(ctx, rel, [EpsFigure(10.27, "Q", 0, 0)])
    # a correctly tagged figure passes, and so does a period whose EPS is not filed
    assert release_tagging_problem(ctx, rel, [EpsFigure(3.60, "Q", 0, 0)]) is None
    assert release_tagging_problem(ctx, rel, []) is None


# NVIDIA, fourth-quarter fiscal 2026 release (2026-02-25): a sentence stating both bases, then GAAP and non-GAAP tables whose
# column headers name the periods in cells ("Q4 FY26", "FY26")
NVDA_Q4_FY26 = (
    "For the quarter, GAAP and non-GAAP earnings per diluted share were $1.76 and $1.62, respectively.\n"
    "For fiscal 2026, GAAP and non-GAAP earnings per diluted share were $4.90 and $4.77, respectively.\n"
    "GAAP |\n($ in millions, except earnings per share) | Q4 FY26 | Q3 FY26 | Q4 FY25 | Q/Q | Y/Y\n"
    "Net income | $42,960 | $31,910 | $22,091 | 35 | 94\nDiluted earnings per share\n$1.76 | $1.30 | $0.89 | 35 | 98\n"
    "Non-GAAP |\n($ in millions, except earnings per share) | Q4 FY26 | Q3 FY26 | Q4 FY25 | Q/Q | Y/Y\n"
    "Revenue | $68,127 | $57,006 | $39,331 | 20 | 73\nGross margin | 75.2 | 73.6 | 73.5 | 1.6 pts | 1.7 pts\n"
    "Operating expenses | $5,102 | $4,215 | $3,378 | 21 | 51\nOperating income | $46,107 | $37,752 | $25,516 | 22 | 81\n"
    "Net income | $39,552 | $31,767 | $22,066 | 25 | 79\nDiluted earnings per share\n$1.62 | $1.30 | $0.89 | 25 | 82\n"
    "Fiscal 2026 Summary\nGAAP |\n($ in millions, except earnings per share) | FY26 | FY25 | Y/Y\n"
    "Revenue | $215,938 | $130,497 | 65\nGross margin | 71.1 | 75.0 | (3.9) pts\nOperating expenses | $23,076 | $16,405 | 41\n"
    "Operating income | $130,387 | $81,453 | 60\n"
    "Net income | $120,067 | $72,880 | 65\nDiluted earnings per share\n$4.90 | $2.94 | 67\n"
    "Non-GAAP |\n($ in millions, except earnings per share) | FY26 | FY25 | Y/Y\n"
    "Net income | $116,997 | $74,265 | 58\nDiluted earnings per share\n$4.77 | $2.99 | 60\n")


def _eps(text: str, released="2026-02-25", focus="2026-01-25"):
    import pandas as pd

    from ffi.detectors.e_text import release_eps

    adjusted, gaap = release_eps(NS(focus=pd.Timestamp(focus)), NS(period_end=released, text=text))
    return [(f.value, f.period) for f in adjusted], [(f.value, f.period) for f in gaap]


def test_sentence_naming_both_bases_never_gives_the_gaap_figure_as_adjusted():
    # the prose fallback used to read "non-GAAP earnings per diluted share were $4.90" (the GAAP figure) as adjusted EPS,
    # and C.NG.01 reported adjusted EPS equal to GAAP EPS
    adjusted, _ = _eps(NVDA_Q4_FY26.split("GAAP |")[0])
    assert adjusted == []
    adjusted, _ = _eps("GAAP EPS was $1.20, while adjusted EPS for the fourth quarter was $1.35.\n")
    assert adjusted == [(1.35, "Q")]  # a clause of its own keeps its figure


def test_period_named_in_column_header_cells():
    adjusted, gaap = _eps(NVDA_Q4_FY26)
    assert sorted(adjusted) == [(1.62, "Q"), (4.77, "FY")] and sorted(gaap) == [(1.76, "Q"), (4.90, "FY")]


def test_prior_year_first_columns_are_not_read():
    from ffi.detectors.e_text import _header_kinds

    # Uber, fourth-quarter 2025 release: "| 2024 | 2025": the first value is last year's
    uber = ("Year Ended December 31,\n(In millions, except percentages) | 2024 | 2025 | % Change\n"
            "GAAP Diluted EPS | 4.56 | 4.73 | 4\n")
    assert "?" in _header_kinds(uber, uber.index("GAAP Diluted"))
    assert _eps(uber, released="2026-01-30", focus="2025-12-31") == ([], [])
    # newest first in each group, the usual layout, is read
    usual = "| Years Ended\nDecember 31,\n(In millions) | 2025 | 2024 | 2025 | 2024\n"
    assert _header_kinds(usual + "Diluted EPS", len(usual)) == {"FY"}


def test_continuing_operations_eps_is_anchored_to_the_filed_continuing_figure():
    import pandas as pd

    from ffi.detectors.e_text import filed_eps

    ctx = _ctx_with_filed_eps()
    q2 = pd.Timestamp("2025-06-30")
    ctx.Q.values["EPS_DILUTED_CONT"] = float("nan")
    ctx.Q.values.loc[q2, ["EPS_DILUTED", "EPS_DILUTED_CONT"]] = [1.89, 1.87]  # General Electric Q2 2025
    rel = NS(period_end="2025-07-17")
    assert filed_eps(ctx, rel, "Q", continuing=True) == 1.87
    assert filed_eps(ctx, rel, "Q") == 1.89
    q1_rel = NS(period_end="2025-04-22")  # no continuing figure filed: the one EPS a company files is the continuing one
    assert filed_eps(ctx, q1_rel, "Q", continuing=True) == 3.46


def test_fourth_quarter_year_to_date_is_the_full_year_and_not_another_period():
    import pandas as pd

    from ffi.detectors.e_text import filed_eps_other_periods

    ctx = _ctx_with_filed_eps()
    ctx.Q.values.loc[pd.Timestamp("2025-12-31"), "EPS_DILUTED"] = 3.40  # a company that tags its fourth quarter
    ctx.Q.values = ctx.Q.values.sort_index()
    other = filed_eps_other_periods(ctx, NS(period_end="2026-01-28"), "FY")
    # the sum of four quarters is the full year itself: a full-year adjusted EPS near GAAP EPS was refused as "the fiscal
    # year to date"
    assert "the fiscal year to date" not in other
    assert "the fiscal year to date" in filed_eps_other_periods(ctx, NS(period_end="2025-10-22"), "Q")


def test_every_context_word_counts_in_a_release_sentence():
    known = {"REVENUE": {"Q": [10.0e9], "FY": []}}
    # the first context word ("Fourth quarter") is allowed in a release; the adjusted basis after it is not
    assert check_amounts("Fourth quarter adjusted revenue was $10.4 billion.", known, quarter_ok=True) == (0, [])
    assert check_amounts("Fourth quarter revenue was $10.0 billion.", known, quarter_ok=True) == (1, [])
    assert check_amounts("Q4 revenue was $10.0 billion.", known, quarter_ok=True) == (1, [])
