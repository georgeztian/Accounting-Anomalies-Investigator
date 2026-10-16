"""Company profiles: which detectors apply to which kind of company (BLUEPRINT §4.3).

Most forensic models were built for industrial companies: they read receivables as trade credit, need
inventory and cost of sales, and treat operating cash flow as the cash yield of the business. For a
bank, a card issuer, an insurer or a REIT those inputs mean something else, and the models produce
confident nonsense. Every company is analysed, but detectors whose inputs do not mean the same thing
for its profile are switched off with a stated reason (status "n/a" in the report), and peer ratios
that do not apply are excluded from the peer-outlier models.

The profile comes from the SEC SIC code. Filing-regime adjustments (20-F / 40-F, IFRS, non-USD
reporting currency) are layered on top by `for_company`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------- reasons
_LOANS = "receivables are customer loans and card balances funded by the company, not trade credit from sales"
_NO_INV = "the business carries no inventory or cost of sales"
_CFO = "operating cash flow includes lending, trading and policyholder flows, so it is not the cash yield of earnings"
_IND = "the model was estimated on industrial companies; its coefficients do not transfer to financial balance sheets"
_DEBT = "interest expense and borrowings are the cost of funding the balance sheet (deposits, policy and repo funding), not corporate leverage"
_SOFT = "nearly all assets are financial instruments, so the soft-asset and asset-turnover ratios have no industrial meaning"
_ALTMAN = "Altman Z was built for manufacturers; it is not valid for financial companies or real estate"
_RE_CAPEX = "property acquisition and development is the business, so capex far above depreciation is expected"
_SHELL = "a blank-check company has no operating revenue, receivables or cost base"

# Detectors that rest on industrial operating-cycle accounting.
_INDUSTRIAL_ONLY: dict[str, str] = {
    "B.01": _IND, "B.02": _IND, "B.03": _IND, "B.04": _CFO, "B.05": _CFO, "B.06": _IND, "B.07": _IND,
    "B.08": _ALTMAN, "B.10": _CFO,
    "C.REV.01": _LOANS, "C.REV.05": _SOFT,
    "C.REV.04": "revenue includes investment and trading gains and lumpy premiums, so its quarterly pattern reflects markets, not sales timing",
    "C.AR.01": _LOANS, "C.AR.04": "securitizing and selling loans is a normal funding activity for lenders", "C.AR.05": _LOANS,
    "C.INV.01": _NO_INV, "C.INV.02": _NO_INV, "C.INV.03": _NO_INV, "C.INV.04": _NO_INV, "C.INV.05": _NO_INV,
    "C.EXP.02": _SOFT,
    "C.CF.01": _CFO, "C.CF.02": _CFO, "C.CF.03": "payables are not trade payables to suppliers", "C.CF.04": _CFO,
    "C.CF.05": _CFO, "C.CF.06": "EBITDA and free cash flow are not meaningful for financial companies",
    "C.BS.01": _DEBT, "C.BS.03": _DEBT,
    "D.04": _IND, "D.06": "the misstatement model was trained on industrial and service companies",
}
# Ratios (d_stats.RATIOS keys) that are meaningless for financial companies.
_FIN_RATIOS = {"gross_margin", "dso", "dsi", "dpo", "asset_turnover", "accruals_ta", "cfo_ni", "capex_da", "soft_assets",
               "ar_growth_less_rev", "inv_growth_less_cogs", "other_ca_rev", "interest_rate", "leverage"}


@dataclass
class Profile:
    key: str
    label: str
    description: str
    excluded: dict[str, str] = field(default_factory=dict)  # detector id -> reason it does not apply
    excluded_ratios: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)

    @property
    def financial(self) -> bool:
        return self.key not in ("industrial", "real_estate")

    def skip_reason(self, detector_id: str) -> str | None:
        r = self.excluded.get(detector_id)
        return f"Not applicable to {self.label.lower()}: {r}." if r else None


def _fin(key: str, label: str, description: str, extra: dict[str, str] | None = None, keep: tuple[str, ...] = ()) -> Profile:
    excluded = {k: v for k, v in _INDUSTRIAL_ONLY.items() if k not in keep}
    excluded.update(extra or {})
    return Profile(key, label, description, excluded, set(_FIN_RATIOS),
                   [f"{label}: {len(excluded)} industrial-company detectors were switched off because their inputs mean something "
                    "else for this business model; integrity, governance, disclosure, reserve and peer checks still apply. "
                    "Peers are drawn from the same industry code."])


PROFILES: dict[str, Profile] = {
    "industrial": Profile("industrial", "Industrial / commercial company",
                          "Operating company; the full detector set applies."),
    "bank": _fin("bank", "Bank or lender",
                 "Deposit-taking bank, card issuer or other credit company (SIC 6000-6199)."),
    "broker": _fin("broker", "Broker-dealer or asset manager",
                   "Securities broker-dealer, exchange or asset manager (SIC 6200-6299).",
                   keep=("C.REV.05",)),
    "insurance": _fin("insurance", "Insurer",
                      "Insurance carrier, agent or broker (SIC 6300-6411).",
                      extra={"C.AR.02": "premiums and reinsurance recoverables are not trade receivables",
                             "C.AR.03": "premiums and reinsurance recoverables are not trade receivables"}),
    "real_estate": Profile("real_estate", "Real-estate company or REIT",
                           "Real-estate operator, developer or REIT (SIC 6500-6553, 6798).",
                           {k: _INDUSTRIAL_ONLY[k] for k in ("C.INV.01", "C.INV.02", "C.INV.03", "C.INV.04", "C.INV.05")}
                           | {"B.08": _ALTMAN, "C.EXP.05": _RE_CAPEX, "C.EXP.02": _SOFT, "D.06": _INDUSTRIAL_ONLY["D.06"]},
                           {"dsi", "inv_growth_less_cogs", "gross_margin", "capex_da", "soft_assets"},
                           ["Real-estate profile: inventory, Altman Z and capex-intensity detectors were switched off; property "
                            "companies are judged against real-estate peers. FFO and NOI are non-GAAP measures and are checked "
                            "by the non-GAAP detectors."]),
    "investment": _fin("investment", "Holding or investment company",
                       "Holding company, trust or investment office (SIC 6700-6799)."),
    "shell": _fin("shell", "Blank-check (SPAC) or shell company",
                  "Special-purpose acquisition company or shell (SIC 6770) with no operating business.",
                  extra={k: _SHELL for k in ("B.09", "C.REV.02", "C.REV.03", "C.REV.04", "C.REV.06", "C.REV.07", "C.REV.09",
                                             "C.REV.10", "C.REV.11", "C.EXP.01", "C.EXP.03", "C.EXP.04", "C.EXP.05",
                                             "C.EXP.06", "C.EXP.09", "D.01", "D.02", "D.03", "C.NG.01", "C.NG.02", "C.NG.03",
                                             "C.NG.04")}),
}


def classify_sic(sic: int | None) -> str:
    if sic is None:
        return "industrial"
    if 6000 <= sic <= 6199:
        return "bank"
    if 6200 <= sic <= 6299:
        return "broker"
    if 6300 <= sic <= 6411:
        return "insurance"
    if 6500 <= sic <= 6553 or sic == 6798:
        return "real_estate"
    if sic == 6770:
        return "shell"
    if 6700 <= sic <= 6799:
        return "investment"
    return "industrial"


# Filing-regime adjustments ------------------------------------------------------
_NON_USD = {"B.08": "Altman Z mixes the USD market value of equity with balances in another reporting currency",
            "D.04": "the industry expectation model is fitted on USD amounts and cannot score another reporting currency"}
_FOREIGN = {"D.06": "the misstatement model was trained on 10-K filers' restatement history (8-K Item 4.02), "
                    "which foreign private issuers do not file"}


def for_company(sic: int | None, annual_form: str = "10-K", currency: str = "USD", taxonomy: str = "us-gaap") -> Profile:
    """The SIC profile plus adjustments for 20-F / 40-F filers, IFRS and non-USD reporting."""
    base = PROFILES[classify_sic(sic)]
    p = Profile(base.key, base.label, base.description, dict(base.excluded), set(base.excluded_ratios), list(base.notes))
    if annual_form in ("20-F", "40-F"):
        for k, v in _FOREIGN.items():
            p.excluded.setdefault(k, v)
        p.notes.append(f"Foreign private issuer filing {annual_form}: there are no 10-Q or 8-K filings, so quarterly, "
                       "earnings-release and 8-K event detectors run only where equivalent data exists (6-K filings are not parsed).")
    if taxonomy == "ifrs-full":
        p.notes.append("Financial statements are prepared under IFRS and mapped to the same canonical line items as US GAAP; "
                       "some US-GAAP-only disclosures (for example LIFO reserves) have no IFRS equivalent.")
    if currency != "USD":
        for k, v in _NON_USD.items():
            p.excluded.setdefault(k, v)
        p.notes.append(f"Amounts are in the reporting currency ({currency}). Peers are matched on ratios only, "
                       "not on size, because their amounts are in USD.")
    return p
