"""Canonical line items and the us-gaap concepts that map to them, in priority order.

Each item: (kind, unit, [concepts...]). kind = "D" (duration / flow) or "I" (instant / balance).
Concepts may be prefixed "dei:" for the dei taxonomy; otherwise us-gaap is assumed.
For each period the first concept with a value wins, so put the most specific concept first.
"""

from __future__ import annotations

from dataclasses import dataclass

USD = "USD"
SHARES = "shares"
EPS = "USD/shares"


@dataclass(frozen=True)
class CanonicalItem:
    id: str
    label: str
    kind: str  # "D" | "I"
    unit: str
    concepts: tuple[str, ...]
    statement: str  # IS, BS, CF, NOTE, DEI


def _i(id_, label, kind, unit, statement, *concepts) -> CanonicalItem:
    return CanonicalItem(id_, label, kind, unit, tuple(concepts), statement)


ITEMS: tuple[CanonicalItem, ...] = (
    # ---------------- Income statement
    # total revenue before contract revenue: at banks and card issuers ASC 606 contract revenue leaves out net interest
    # income (American Express: $38.8B of $65.9B in 2024), so it must not stand in for the total
    _i("REVENUE", "Revenue", "D", USD, "IS",
       "Revenues", "RevenuesNetOfInterestExpense", "RevenueFromContractWithCustomerExcludingAssessedTax",
       "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet", "SalesRevenueGoodsNet",
       "SalesRevenueServicesNet"),
    _i("COGS", "Cost of revenue", "D", USD, "IS",
       "CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold", "CostOfServices",
       "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization"),
    _i("GROSS_PROFIT", "Gross profit", "D", USD, "IS", "GrossProfit"),
    _i("SGA", "SG&A expense", "D", USD, "IS",
       "SellingGeneralAndAdministrativeExpense", "GeneralAndAdministrativeExpense"),
    # selling expense reported apart from general and administrative expense (Microsoft, Salesforce): SG&A is the sum
    _i("SELLING_MARKETING", "Selling and marketing expense", "D", USD, "IS", "SellingAndMarketingExpense"),
    _i("RND", "R&D expense", "D", USD, "IS",
       "ResearchAndDevelopmentExpense", "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"),
    _i("OPERATING_INCOME", "Operating income", "D", USD, "IS", "OperatingIncomeLoss"),
    _i("INTEREST_EXPENSE", "Interest expense", "D", USD, "IS",
       "InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt", "InterestAndDebtExpense"),
    _i("PRETAX_INCOME", "Pre-tax income", "D", USD, "IS",
       "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
       "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"),
    _i("INCOME_TAX", "Income tax expense", "D", USD, "IS", "IncomeTaxExpenseBenefit"),
    _i("INCOME_CONT_OPS", "Income from continuing operations", "D", USD, "IS",
       "IncomeLossFromContinuingOperations",
       "IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest"),
    _i("NET_INCOME", "Net income", "D", USD, "IS", "NetIncomeLoss", "ProfitLoss",
       "NetIncomeLossAvailableToCommonStockholdersBasic"),
    _i("NET_INCOME_TO_COMMON", "Net income available to common shareholders", "D", USD, "IS",
       "NetIncomeLossAvailableToCommonStockholdersBasic"),
    # the numerator of diluted EPS: net income less preferred dividends (Boeing 2025: 2,235 - 345 = 1,890)
    _i("NI_COMMON_DILUTED", "Net income for diluted EPS", "D", USD, "IS", "NetIncomeLossAvailableToCommonStockholdersDiluted",
       "NetIncomeLossAvailableToCommonStockholdersBasic"),
    _i("NET_INCOME_INCL_NCI", "Net income incl. noncontrolling interests", "D", USD, "IS", "ProfitLoss",
       "IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest"),
    _i("NCI_INCOME", "Net income attributable to noncontrolling interests", "D", USD, "IS",
       "NetIncomeLossAttributableToNoncontrollingInterest", "NetIncomeLossAttributableToNonredeemableNoncontrollingInterest"),
    _i("EPS_BASIC", "EPS (basic)", "D", EPS, "IS", "EarningsPerShareBasic", "EarningsPerShareBasicAndDiluted"),
    _i("EPS_DILUTED", "EPS (diluted)", "D", EPS, "IS", "EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"),
    # what releases headline when there are discontinued operations (General Electric: "Continuing EPS"), and what adjusted
    # EPS is reconciled from
    _i("EPS_DILUTED_CONT", "EPS from continuing operations (diluted)", "D", EPS, "IS",
       "IncomeLossFromContinuingOperationsPerDilutedShare"),
    _i("SHARES_DILUTED", "Weighted diluted shares", "D", SHARES, "IS",
       "WeightedAverageNumberOfDilutedSharesOutstanding"),
    _i("SHARES_BASIC", "Weighted basic shares", "D", SHARES, "IS", "WeightedAverageNumberOfSharesOutstandingBasic"),
    _i("GOODWILL_IMPAIRMENT", "Goodwill impairment", "D", USD, "IS", "GoodwillImpairmentLoss"),
    _i("ASSET_IMPAIRMENT", "Asset impairment", "D", USD, "IS",
       "AssetImpairmentCharges", "ImpairmentOfLongLivedAssetsHeldForUse"),
    _i("RESTRUCTURING", "Restructuring charges", "D", USD, "IS",
       "RestructuringCharges", "RestructuringSettlementAndImpairmentProvisions"),
    _i("OCI", "Other comprehensive income", "D", USD, "IS",
       "OtherComprehensiveIncomeLossNetOfTax",
       "OtherComprehensiveIncomeLossNetOfTaxPortionAttributableToParent"),
    _i("DEFERRED_TAX_EXPENSE", "Deferred tax expense", "D", USD, "IS", "DeferredIncomeTaxExpenseBenefit"),
    # ---------------- Cash flow
    _i("DA", "Depreciation & amortization", "D", USD, "CF",
       "DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet",
       "DepreciationAndAmortization"),
    _i("DEPRECIATION", "Depreciation", "D", USD, "CF", "Depreciation", "DepreciationNonproduction"),
    _i("AMORTIZATION_INTANGIBLES", "Amortization of intangibles", "D", USD, "CF", "AmortizationOfIntangibleAssets"),
    _i("SBC", "Share-based compensation", "D", USD, "CF",
       "ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"),
    _i("CFO", "Cash from operations", "D", USD, "CF",
       "NetCashProvidedByUsedInOperatingActivities",
       "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
    _i("CFI", "Cash from investing", "D", USD, "CF",
       "NetCashProvidedByUsedInInvestingActivities",
       "NetCashProvidedByUsedInInvestingActivitiesContinuingOperations"),
    _i("CFF", "Cash from financing", "D", USD, "CF",
       "NetCashProvidedByUsedInFinancingActivities",
       "NetCashProvidedByUsedInFinancingActivitiesContinuingOperations"),
    _i("CAPEX", "Capital expenditures", "D", USD, "CF",
       "PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"),
    _i("ACQUISITIONS", "Cash paid for acquisitions", "D", USD, "CF", "PaymentsToAcquireBusinessesNetOfCashAcquired"),
    _i("DIVESTITURES", "Proceeds from divestitures", "D", USD, "CF",
       "ProceedsFromDivestitureOfBusinesses", "ProceedsFromDivestitureOfBusinessesNetOfCashDivested"),
    _i("DIVIDENDS_PAID", "Dividends paid", "D", USD, "CF", "PaymentsOfDividends", "PaymentsOfDividendsCommonStock"),
    _i("DIVIDENDS_DECLARED", "Dividends declared", "D", USD, "NOTE",
       "DividendsCommonStockCash", "DividendsCommonStock", "Dividends", "DividendsCash"),
    # cash paid first; filers that tag repurchases only in the statement of equity (American Express: $4.6B of treasury
    # stock acquired in 2019 and no cash-flow tag) would otherwise show an unexplained fall in equity
    _i("BUYBACKS", "Share repurchases", "D", USD, "CF", "PaymentsForRepurchaseOfCommonStock", "PaymentsForRepurchaseOfEquity",
       "TreasuryStockValueAcquiredCostMethod", "StockRepurchasedDuringPeriodValue", "StockRepurchasedAndRetiredDuringPeriodValue"),
    _i("SHARE_TAX_WITHHELD", "Taxes paid on net share settlement of stock awards", "D", USD, "CF",
       "PaymentsRelatedToTaxWithholdingForShareBasedCompensation"),
    _i("DEBT_ISSUED", "Debt issued", "D", USD, "CF",
       "ProceedsFromIssuanceOfLongTermDebt", "ProceedsFromIssuanceOfDebt", "ProceedsFromIssuanceOfSeniorLongTermDebt"),
    # option exercises are only part of the shares issued, so they rank last
    _i("EQUITY_ISSUED", "Equity issued", "D", USD, "CF", "ProceedsFromIssuanceOfCommonStock",
       "ProceedsFromIssuanceOrSaleOfEquity", "StockIssuedDuringPeriodValueNewIssues", "ProceedsFromStockOptionsExercised"),
    # other lines of the statement of equity that move equity without cash in the same line (totals, no dimensions)
    _i("EQUITY_ISSUED_ACQ", "Stock issued for acquisitions", "D", USD, "NOTE", "StockIssuedDuringPeriodValueAcquisitions"),
    _i("EQUITY_ISSUED_STMT", "Stock issued (statement of equity)", "D", USD, "NOTE", "StockIssuedDuringPeriodValueNewIssues"),
    _i("EQUITY_CONVERTED", "Convertible securities converted to stock", "D", USD, "NOTE",
       "StockIssuedDuringPeriodValueConversionOfConvertibleSecurities", "ConversionOfStockAmountConverted1"),
    _i("EQUITY_ISSUED_OTHER", "Other stock issued", "D", USD, "NOTE", "StockIssuedDuringPeriodValueOther"),
    # bought by employees for cash: the cash may also be inside the proceeds from issuing stock (EQUITY_ISSUED)
    _i("EQUITY_ISSUED_ESPP", "Stock issued under employee purchase plans", "D", USD, "NOTE",
       "StockIssuedDuringPeriodValueEmployeeStockPurchasePlan"),
    _i("CONVERTIBLE_EQUITY_COMPONENT", "Equity component of convertible debt", "D", USD, "NOTE",
       "AdjustmentsToAdditionalPaidInCapitalEquityComponentOfConvertibleDebt"),
    _i("CASH_CHANGE", "Net change in cash", "D", USD, "CF",
       "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect",
       "CashAndCashEquivalentsPeriodIncreaseDecrease"),
    _i("CASH_DISCONTINUED", "Net cash of discontinued operations", "D", USD, "CF", "NetCashProvidedByUsedInDiscontinuedOperations"),
    _i("FX_EFFECT_CASH", "FX effect on cash", "D", USD, "CF",
       "EffectOfExchangeRateOnCashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
       "EffectOfExchangeRateOnCashAndCashEquivalents",
       # Goldman Sachs 2025 tags its $5.4B effect of exchange rates with the disposal-group variant
       "EffectOfExchangeRateOnCashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsIncludingDisposalGroupAndDiscontinuedOperations",
       "EffectOfExchangeRateOnCashAndCashEquivalentsIncludingDisposalGroupAndDiscontinuedOperations"),
    # additions to PP&E that are not cash capex: capital expenditures accrued but unpaid, assets acquired under finance leases
    _i("CAPEX_UNPAID", "Capital expenditures incurred but not yet paid", "D", USD, "NOTE", "CapitalExpendituresIncurredButNotYetPaid"),
    _i("FINANCE_LEASE_ASSETS_ADDED", "Assets obtained under finance leases", "D", USD, "NOTE",
       "RightOfUseAssetObtainedInExchangeForFinanceLeaseLiability"),
    _i("CHANGE_AR", "Change in receivables (CF)", "D", USD, "CF", "IncreaseDecreaseInAccountsReceivable"),
    _i("CHANGE_INVENTORY", "Change in inventory (CF)", "D", USD, "CF", "IncreaseDecreaseInInventories"),
    _i("CHANGE_AP", "Change in payables (CF)", "D", USD, "CF", "IncreaseDecreaseInAccountsPayable"),
    _i("CHANGE_DEFERRED_REV", "Change in deferred revenue (CF)", "D", USD, "CF",
       "IncreaseDecreaseInContractWithCustomerLiability", "IncreaseDecreaseInDeferredRevenue"),
    _i("PROVISION_DOUBTFUL", "Provision for credit losses", "D", USD, "CF",
       "ProvisionForDoubtfulAccounts", "AccountsReceivableCreditLossExpenseReversal"),
    _i("WRITE_OFFS", "Receivable write-offs", "D", USD, "NOTE",
       "AllowanceForDoubtfulAccountsReceivableWriteOffs", "AccountsReceivableAllowanceForCreditLossWriteoff"),
    _i("CAP_SOFTWARE_ADDITIONS", "Capitalized software additions", "D", USD, "CF",
       "PaymentsToDevelopSoftware", "CapitalizedComputerSoftwareAdditions"),
    _i("GAIN_ON_SALE", "Gain on asset sales", "D", USD, "CF",
       "GainLossOnSaleOfPropertyPlantEquipment", "GainLossOnDispositionOfAssets"),
    _i("INCOME_TAXES_PAID", "Income taxes paid", "D", USD, "CF", "IncomeTaxesPaidNet", "IncomeTaxesPaid"),
    _i("INTEREST_PAID", "Interest paid", "D", USD, "CF", "InterestPaidNet", "InterestPaid"),
    _i("WARRANTY_PROVISION", "Warranty accruals issued", "D", USD, "NOTE",
       "StandardProductWarrantyAccrualWarrantiesIssued", "ProductWarrantyExpense"),
    _i("WARRANTY_PAYMENTS", "Warranty payments", "D", USD, "NOTE", "StandardProductWarrantyAccrualDecreaseForPayments"),
    _i("GOODWILL_ACQUIRED", "Goodwill acquired", "D", USD, "NOTE", "GoodwillAcquiredDuringPeriod"),
    _i("GOODWILL_FX", "Goodwill FX translation", "D", USD, "NOTE", "GoodwillForeignCurrencyTranslationGainLoss"),
    _i("GOODWILL_MEASUREMENT_ADJ", "Goodwill measurement-period adjustments", "D", USD, "NOTE",
       "GoodwillPurchaseAccountingAdjustments"),
    _i("GOODWILL_OTHER", "Goodwill other changes", "D", USD, "NOTE",
       "GoodwillOtherIncreaseDecrease", "GoodwillTransfers"),
    _i("GOODWILL_DISPOSED", "Goodwill written off on disposal", "D", USD, "NOTE",
       "GoodwillWrittenOffRelatedToSaleOfBusinessUnit"),
    _i("CONTINGENT_CONSIDERATION_CHANGE", "Change in contingent consideration", "D", USD, "NOTE",
       "BusinessCombinationContingentConsiderationArrangementsChangeInAmountOfContingentConsiderationLiability1"),
    # ---------------- Balance sheet
    _i("CASH", "Cash & equivalents", "I", USD, "BS",
       "CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
    _i("CASH_RESTRICTED_TOTAL", "Cash incl. restricted", "I", USD, "BS",
       "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", "CashAndCashEquivalentsAtCarryingValue"),
    _i("ST_INVESTMENTS", "Short-term investments", "I", USD, "BS",
       "ShortTermInvestments", "MarketableSecuritiesCurrent", "AvailableForSaleSecuritiesDebtSecuritiesCurrent"),
    _i("AR", "Accounts receivable, net", "I", USD, "BS",
       "AccountsReceivableNetCurrent", "ReceivablesNetCurrent", "AccountsNotesAndLoansReceivableNetCurrent"),
    _i("ALLOWANCE", "Allowance for credit losses", "I", USD, "BS",
       "AllowanceForDoubtfulAccountsReceivableCurrent", "AccountsReceivableAllowanceForCreditLossCurrent",
       "AccountsReceivableAllowanceForCreditLoss"),
    _i("CONTRACT_ASSETS", "Contract assets", "I", USD, "BS",
       "ContractWithCustomerAssetNetCurrent", "ContractWithCustomerAssetNet"),
    _i("INVENTORY", "Inventory", "I", USD, "BS", "InventoryNet", "InventoryGross"),
    _i("INV_FINISHED", "Finished goods", "I", USD, "NOTE", "InventoryFinishedGoods", "InventoryFinishedGoodsNetOfReserves"),
    _i("INV_RAW", "Raw materials", "I", USD, "NOTE", "InventoryRawMaterials", "InventoryRawMaterialsNetOfReserves"),
    _i("INV_RESERVE", "Inventory reserve", "I", USD, "NOTE", "InventoryValuationReserves"),
    _i("LIFO_RESERVE", "LIFO reserve", "I", USD, "NOTE", "InventoryLIFOReserve"),
    _i("OTHER_CURRENT_ASSETS", "Other current assets", "I", USD, "BS",
       "OtherAssetsCurrent", "PrepaidExpenseAndOtherAssetsCurrent"),
    _i("CURRENT_ASSETS", "Total current assets", "I", USD, "BS", "AssetsCurrent"),
    _i("PPE_NET", "PP&E, net", "I", USD, "BS",
       "PropertyPlantAndEquipmentNet",
       "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization"),
    _i("PPE_GROSS", "PP&E, gross", "I", USD, "NOTE",
       "PropertyPlantAndEquipmentGross",
       "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetBeforeAccumulatedDepreciationAndAmortization"),
    # parts of gross PP&E that carry no depreciation yet
    _i("PPE_CIP", "Construction in progress", "I", USD, "NOTE", "ConstructionInProgressGross"),
    _i("PPE_LAND", "Land", "I", USD, "NOTE", "Land"),
    _i("ACCUM_DEP", "Accumulated depreciation", "I", USD, "NOTE",
       "AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment"),
    _i("GOODWILL", "Goodwill", "I", USD, "BS", "Goodwill"),
    _i("INTANGIBLES", "Intangibles (ex goodwill)", "I", USD, "BS",
       "IntangibleAssetsNetExcludingGoodwill", "FiniteLivedIntangibleAssetsNet"),
    _i("CAP_SOFTWARE", "Capitalized software, net", "I", USD, "NOTE", "CapitalizedComputerSoftwareNet"),
    _i("OTHER_NONCURRENT_ASSETS", "Other non-current assets", "I", USD, "BS", "OtherAssetsNoncurrent"),
    _i("EQUITY_METHOD_INVESTMENTS", "Equity-method investments", "I", USD, "BS", "EquityMethodInvestments"),
    _i("TOTAL_ASSETS", "Total assets", "I", USD, "BS", "Assets"),
    _i("AP", "Accounts payable", "I", USD, "BS", "AccountsPayableCurrent", "AccountsPayableAndAccruedLiabilitiesCurrent"),
    _i("ACCRUED_LIABILITIES", "Accrued liabilities", "I", USD, "BS", "AccruedLiabilitiesCurrent"),
    _i("DEFERRED_REV_CURRENT", "Deferred revenue, current", "I", USD, "BS",
       "ContractWithCustomerLiabilityCurrent", "DeferredRevenueCurrent"),
    _i("DEFERRED_REV_NONCURRENT", "Deferred revenue, non-current", "I", USD, "BS",
       "ContractWithCustomerLiabilityNoncurrent", "DeferredRevenueNoncurrent"),
    _i("CURRENT_DEBT", "Current debt", "I", USD, "BS", "LongTermDebtCurrent", "DebtCurrent",
       "LongTermDebtAndCapitalLeaseObligationsCurrent", "ConvertibleNotesPayableCurrent", "LinesOfCreditCurrent",
       "ShortTermBankLoansAndNotesPayable"),
    _i("ST_BORROWINGS", "Short-term borrowings", "I", USD, "BS", "ShortTermBorrowings", "CommercialPaper"),
    _i("LONG_TERM_DEBT", "Long-term debt", "I", USD, "BS",
       "LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations", "LongTermDebt",
       "ConvertibleLongTermNotesPayable", "ConvertibleNotesPayable", "LongTermNotesPayable", "SeniorLongTermNotes",
       "LongTermLineOfCredit", "LongTermLoansPayable", "SecuredLongTermDebt"),
    _i("OPERATING_LEASE_LIABILITY", "Operating lease liability", "I", USD, "BS", "OperatingLeaseLiability"),
    _i("CURRENT_LIABILITIES", "Total current liabilities", "I", USD, "BS", "LiabilitiesCurrent"),
    _i("TOTAL_LIABILITIES", "Total liabilities", "I", USD, "BS", "Liabilities"),
    _i("EQUITY", "Shareholders' equity", "I", USD, "BS", "StockholdersEquity"),
    _i("EQUITY_INCL_NCI", "Equity incl. NCI", "I", USD, "BS",
       "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    _i("NCI", "Non-controlling interest", "I", USD, "BS", "MinorityInterest"),
    _i("TEMPORARY_EQUITY", "Redeemable (temporary) equity", "I", USD, "BS",
       "TemporaryEquityCarryingAmountIncludingPortionAttributableToNoncontrollingInterests",
       "RedeemableNoncontrollingInterestEquityCarryingAmount", "TemporaryEquityCarryingAmountAttributableToParent"),
    _i("LIABILITIES_AND_EQUITY", "Liabilities & equity", "I", USD, "BS", "LiabilitiesAndStockholdersEquity"),
    _i("RETAINED_EARNINGS", "Retained earnings", "I", USD, "BS", "RetainedEarningsAccumulatedDeficit"),
    # dividends declared = paid + the change in dividends payable (Progressive declares an annual dividend in December
    # and pays it in January, so the year's cash dividends belong to the previous year's declaration)
    _i("DIVIDENDS_PAYABLE", "Dividends payable", "I", USD, "BS", "DividendsPayableCurrentAndNoncurrent", "DividendsPayableCurrent"),
    _i("WARRANTY_ACCRUAL", "Warranty reserve", "I", USD, "NOTE", "StandardProductWarrantyAccrual"),
    _i("VALUATION_ALLOWANCE", "Deferred tax valuation allowance", "I", USD, "NOTE", "DeferredTaxAssetsValuationAllowance"),
    _i("UTP", "Unrecognized tax benefits", "I", USD, "NOTE", "UnrecognizedTaxBenefits"),
    _i("RPO", "Remaining performance obligations", "I", USD, "NOTE", "RevenueRemainingPerformanceObligation"),
    _i("PURCHASE_OBLIGATIONS", "Purchase obligations", "I", USD, "NOTE",
       "PurchaseObligation", "UnrecordedUnconditionalPurchaseObligationBalanceSheetAmount"),
    _i("GUARANTEES", "Guarantee maximum exposure", "I", USD, "NOTE", "GuaranteeObligationsMaximumExposure"),
    _i("SUPPLIER_FINANCE", "Supplier finance obligations", "I", USD, "NOTE", "SupplierFinanceProgramObligation"),
    _i("RELATED_PARTY_REVENUE", "Revenue from related parties", "D", USD, "NOTE", "RevenueFromRelatedParties"),
    _i("EQUITY_METHOD_INCOME", "Equity-method income (loss)", "D", USD, "IS", "IncomeLossFromEquityMethodInvestments"),
    _i("LT_RECEIVABLES", "Long-term receivables", "I", USD, "BS",
       "AccountsReceivableNoncurrent", "NotesAndLoansReceivableNetNoncurrent", "NotesReceivableNet"),
    _i("LITIGATION_ACCRUAL", "Loss-contingency accrual", "I", USD, "NOTE", "LossContingencyAccrualAtCarryingValue"),
    _i("PENSION_DISCOUNT_RATE", "Pension discount rate", "I", "pure", "NOTE",
       "DefinedBenefitPlanAssumptionsUsedCalculatingBenefitObligationDiscountRate"),
    _i("PENSION_EXPECTED_RETURN", "Pension expected return on assets", "D", "pure", "NOTE",
       "DefinedBenefitPlanAssumptionsUsedCalculatingNetPeriodicBenefitCostExpectedLongTermReturnOnAssets"),
    # ---------------- Cover page (dei)
    _i("PUBLIC_FLOAT", "Public float", "I", USD, "DEI", "dei:EntityPublicFloat"),
    _i("SHARES_OUTSTANDING", "Shares outstanding (cover)", "I", SHARES, "DEI", "dei:EntityCommonStockSharesOutstanding"),
)

# IFRS (ifrs-full taxonomy) equivalents for 20-F / 40-F filers. They rank after the us-gaap concepts; an IFRS
# filer has no us-gaap facts, so its values come from these. Cash-flow working-capital lines are omitted on
# purpose: IFRS tags them as "decrease (increase)", the opposite sign of the us-gaap concepts.
IFRS: dict[str, tuple[str, ...]] = {
    "REVENUE": ("Revenue", "RevenueFromContractsWithCustomers"),
    "COGS": ("CostOfSales",),
    "GROSS_PROFIT": ("GrossProfit",),
    "SGA": ("SellingGeneralAndAdministrativeExpense", "AdministrativeExpense"),
    "RND": ("ResearchAndDevelopmentExpense",),
    "OPERATING_INCOME": ("ProfitLossFromOperatingActivities",),
    "INTEREST_EXPENSE": ("InterestExpense", "FinanceCosts"),
    "PRETAX_INCOME": ("ProfitLossBeforeTax",),
    "INCOME_TAX": ("IncomeTaxExpenseContinuingOperations",),
    "INCOME_CONT_OPS": ("ProfitLossFromContinuingOperations",),
    "NET_INCOME": ("ProfitLossAttributableToOwnersOfParent", "ProfitLoss"),
    "NET_INCOME_INCL_NCI": ("ProfitLoss",),
    "NCI_INCOME": ("ProfitLossAttributableToNoncontrollingInterests",),
    "EPS_BASIC": ("BasicEarningsLossPerShare",),
    "EPS_DILUTED": ("DilutedEarningsLossPerShare",),
    "EPS_DILUTED_CONT": ("DilutedEarningsLossPerShareFromContinuingOperations",),
    "SHARES_DILUTED": ("AdjustedWeightedAverageShares",),
    "SHARES_BASIC": ("WeightedAverageShares",),
    "GOODWILL_IMPAIRMENT": ("ImpairmentLossRecognisedInProfitOrLossGoodwill",),
    "OCI": ("OtherComprehensiveIncome",),
    "DEFERRED_TAX_EXPENSE": ("DeferredTaxExpenseIncome",),
    "DA": ("DepreciationAndAmortisationExpense",
           "DepreciationAmortisationAndImpairmentLossReversalOfImpairmentLossRecognisedInProfitOrLoss"),
    "DEPRECIATION": ("DepreciationPropertyPlantAndEquipment", "DepreciationExpense"),
    "AMORTIZATION_INTANGIBLES": ("AmortisationIntangibleAssetsOtherThanGoodwill",),
    "SBC": ("AdjustmentsForSharebasedPayments", "ExpenseFromSharebasedPaymentTransactionsWithEmployees"),
    "CFO": ("CashFlowsFromUsedInOperatingActivities",),
    "CFI": ("CashFlowsFromUsedInInvestingActivities",),
    "CFF": ("CashFlowsFromUsedInFinancingActivities",),
    "CAPEX": ("PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities", "PurchaseOfPropertyPlantAndEquipment"),
    "ACQUISITIONS": ("CashFlowsUsedInObtainingControlOfSubsidiariesOrOtherBusinessesClassifiedAsInvestingActivities",),
    "DIVIDENDS_PAID": ("DividendsPaidClassifiedAsFinancingActivities", "DividendsPaid"),
    "DIVIDENDS_DECLARED": ("DividendsRecognisedAsDistributionsToOwnersOfParent",),
    "BUYBACKS": ("PaymentsToAcquireOrRedeemEntitysShares",),
    "DEBT_ISSUED": ("ProceedsFromNoncurrentBorrowings", "ProceedsFromBorrowingsClassifiedAsFinancingActivities"),
    "EQUITY_ISSUED": ("ProceedsFromIssuingShares",),
    "CASH_CHANGE": ("IncreaseDecreaseInCashAndCashEquivalents",),
    "FX_EFFECT_CASH": ("EffectOfExchangeRateChangesOnCashAndCashEquivalents",),
    "INCOME_TAXES_PAID": ("IncomeTaxesPaidRefundClassifiedAsOperatingActivities", "IncomeTaxesPaidRefund"),
    "INTEREST_PAID": ("InterestPaidClassifiedAsOperatingActivities", "InterestPaidClassifiedAsFinancingActivities"),
    "CASH": ("CashAndCashEquivalents",),
    "CASH_RESTRICTED_TOTAL": ("CashAndCashEquivalents",),
    "AR": ("CurrentTradeReceivables", "TradeAndOtherCurrentReceivables"),
    "CONTRACT_ASSETS": ("CurrentContractAssets", "ContractAssets"),
    "INVENTORY": ("Inventories",),
    "INV_FINISHED": ("CurrentFinishedGoods",),
    "INV_RAW": ("CurrentRawMaterials",),
    "OTHER_CURRENT_ASSETS": ("OtherCurrentAssets",),
    "CURRENT_ASSETS": ("CurrentAssets",),
    "PPE_NET": ("PropertyPlantAndEquipment",),
    "GOODWILL": ("Goodwill",),
    "INTANGIBLES": ("IntangibleAssetsOtherThanGoodwill",),
    "OTHER_NONCURRENT_ASSETS": ("OtherNoncurrentAssets",),
    "EQUITY_METHOD_INVESTMENTS": ("InvestmentsAccountedForUsingEquityMethod",),
    "TOTAL_ASSETS": ("Assets",),
    "AP": ("TradeAndOtherCurrentPayablesToTradeSuppliers", "TradeAndOtherCurrentPayables"),
    "DEFERRED_REV_CURRENT": ("CurrentContractLiabilities",),
    "DEFERRED_REV_NONCURRENT": ("NoncurrentContractLiabilities",),
    "CURRENT_DEBT": ("CurrentPortionOfLongtermBorrowings", "CurrentBorrowingsAndCurrentPortionOfNoncurrentBorrowings"),
    "ST_BORROWINGS": ("ShorttermBorrowings",),
    "LONG_TERM_DEBT": ("NoncurrentPortionOfNoncurrentBorrowings", "LongtermBorrowings"),
    "OPERATING_LEASE_LIABILITY": ("LeaseLiabilities",),
    "CURRENT_LIABILITIES": ("CurrentLiabilities",),
    "TOTAL_LIABILITIES": ("Liabilities",),
    "EQUITY": ("EquityAttributableToOwnersOfParent",),
    "EQUITY_INCL_NCI": ("Equity",),
    "NCI": ("NoncontrollingInterests",),
    "LIABILITIES_AND_EQUITY": ("EquityAndLiabilities",),
    "RETAINED_EARNINGS": ("RetainedEarnings",),
}

BY_ID: dict[str, CanonicalItem] = {it.id: it for it in ITEMS}

# Fallback concepts that measure a narrower or broader amount than the item's main concepts. Two values of an item are
# on the same basis only when their concepts have the same basis key ("" = the main basis): a year of general and
# administrative expense is not comparable with a year of selling, general and administrative expense.
DIFFERENT_BASIS: dict[str, dict[str, str]] = {
    "SGA": {"GeneralAndAdministrativeExpense": "general and administrative only", "AdministrativeExpense": "administrative only"},
    "NET_INCOME": {"ProfitLoss": "including noncontrolling interests",
                   "NetIncomeLossAvailableToCommonStockholdersBasic": "available to common shareholders"},
    "CASH": {"CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents": "including restricted cash"},
    "CASH_RESTRICTED_TOTAL": {"CashAndCashEquivalentsAtCarryingValue": "excluding restricted cash"},
    "PPE_NET": {"PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization":
                "including finance-lease assets"},
    "PPE_GROSS": {"PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetBeforeAccumulatedDepreciationAndAmortization":
                  "including finance-lease assets"},
    "AR": {"ReceivablesNetCurrent": "all current receivables", "AccountsNotesAndLoansReceivableNetCurrent": "including notes and loans",
           "TradeAndOtherCurrentReceivables": "trade and other receivables"},
    # payables together with accrued liabilities (Coca-Cola: DPO 294 days against a peer median of 62): not comparable with trade payables
    "AP": {"AccountsPayableAndAccruedLiabilitiesCurrent": "payables and accrued liabilities",
           "TradeAndOtherCurrentPayables": "trade and other payables"},
    "ALLOWANCE": {"AccountsReceivableAllowanceForCreditLoss": "current and non-current receivables"},
    "LT_RECEIVABLES": {"NotesReceivableNet": "including current notes"},
    # IFRS depreciation, amortisation and impairment in one line: a year with an impairment is no depreciation charge
    "DA": {"DepreciationAmortisationAndImpairmentLossReversalOfImpairmentLossRecognisedInProfitOrLoss": "including impairment"},
    "COGS": {"CostOfServices": "services only",
             "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization": "excluding depreciation"},
    # pre-tax income before the share of equity-method investees' results (Toyota), which sits below income tax
    "PRETAX_INCOME": {"IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments":
                      "before equity-method income"},
    "CAPEX": {"PaymentsToAcquireProductiveAssets": "all productive assets"},
    "INVENTORY": {"InventoryGross": "before reserves"},
    "EQUITY_ISSUED": {"ProceedsFromStockOptionsExercised": "option exercises only",
                      "StockIssuedDuringPeriodValueNewIssues": "shares issued, from the statement of equity"},
    "INTEREST_EXPENSE": {"InterestExpenseDebt": "on debt only"},
    "RND": {"ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost": "excluding acquired research"},
    # long-term debt including its current maturities, which CURRENT_DEBT may also hold
    "LONG_TERM_DEBT": {c: "including current maturities" for c in
                       ("LongTermDebt", "SeniorLongTermNotes", "SecuredLongTermDebt", "ConvertibleNotesPayable", "LongtermBorrowings")},
    # current debt including short-term borrowings (DebtCurrent), or short-term borrowings alone, which ST_BORROWINGS may hold
    "CURRENT_DEBT": {"DebtCurrent": "including short-term borrowings",
                     "CurrentBorrowingsAndCurrentPortionOfNoncurrentBorrowings": "including short-term borrowings",
                     "ShortTermBankLoansAndNotesPayable": "short-term borrowings only", "LinesOfCreditCurrent": "short-term borrowings only"},
}


def basis(item: str, concept: str | None) -> str:
    """The basis key of a value of `item` reported under `concept` ("" = the item's main basis)."""
    return DIFFERENT_BASIS.get(item, {}).get(concept or "", "")


CONCEPT_TO_ITEMS: dict[tuple[str, str], list[tuple[str, int]]] = {}
for _item in ITEMS:
    for _rank, _c in enumerate(_item.concepts):
        _tax, _name = (_c.split(":", 1) if ":" in _c else ("us-gaap", _c))
        CONCEPT_TO_ITEMS.setdefault((_tax, _name), []).append((_item.id, _rank))
    for _rank, _c in enumerate(IFRS.get(_item.id, ()), start=len(_item.concepts)):
        CONCEPT_TO_ITEMS.setdefault(("ifrs-full", _c), []).append((_item.id, _rank))
