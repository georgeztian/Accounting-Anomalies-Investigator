"""Canonical line items and the us-gaap concepts that map to them, in priority order (BLUEPRINT §5).

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
    _i("REVENUE", "Revenue", "D", USD, "IS",
       "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
       "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet", "SalesRevenueGoodsNet",
       "SalesRevenueServicesNet", "RevenuesNetOfInterestExpense"),
    _i("COGS", "Cost of revenue", "D", USD, "IS",
       "CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold", "CostOfServices",
       "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization"),
    _i("GROSS_PROFIT", "Gross profit", "D", USD, "IS", "GrossProfit"),
    _i("SGA", "SG&A expense", "D", USD, "IS",
       "SellingGeneralAndAdministrativeExpense", "GeneralAndAdministrativeExpense"),
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
    _i("EPS_BASIC", "EPS (basic)", "D", EPS, "IS", "EarningsPerShareBasic", "EarningsPerShareBasicAndDiluted"),
    _i("EPS_DILUTED", "EPS (diluted)", "D", EPS, "IS", "EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"),
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
    _i("DIVIDENDS_PAID", "Dividends paid", "D", USD, "CF", "PaymentsOfDividends", "PaymentsOfDividendsCommonStock"),
    _i("DIVIDENDS_DECLARED", "Dividends declared", "D", USD, "NOTE",
       "DividendsCommonStockCash", "DividendsCommonStock", "Dividends", "DividendsCash"),
    _i("BUYBACKS", "Share repurchases", "D", USD, "CF", "PaymentsForRepurchaseOfCommonStock"),
    _i("DEBT_ISSUED", "Debt issued", "D", USD, "CF",
       "ProceedsFromIssuanceOfLongTermDebt", "ProceedsFromIssuanceOfDebt", "ProceedsFromIssuanceOfSeniorLongTermDebt"),
    _i("EQUITY_ISSUED", "Equity issued", "D", USD, "CF", "ProceedsFromIssuanceOfCommonStock",
       "ProceedsFromIssuanceOrSaleOfEquity", "ProceedsFromStockOptionsExercised"),
    _i("CONVERTIBLE_EQUITY_COMPONENT", "Equity component of convertible debt", "D", USD, "NOTE",
       "AdjustmentsToAdditionalPaidInCapitalEquityComponentOfConvertibleDebt"),
    _i("CASH_CHANGE", "Net change in cash", "D", USD, "CF",
       "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect",
       "CashAndCashEquivalentsPeriodIncreaseDecrease"),
    _i("FX_EFFECT_CASH", "FX effect on cash", "D", USD, "CF",
       "EffectOfExchangeRateOnCashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
       "EffectOfExchangeRateOnCashAndCashEquivalents"),
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
    _i("LIABILITIES_AND_EQUITY", "Liabilities & equity", "I", USD, "BS", "LiabilitiesAndStockholdersEquity"),
    _i("RETAINED_EARNINGS", "Retained earnings", "I", USD, "BS", "RetainedEarningsAccumulatedDeficit"),
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

BY_ID: dict[str, CanonicalItem] = {it.id: it for it in ITEMS}

CONCEPT_TO_ITEMS: dict[tuple[str, str], list[tuple[str, int]]] = {}
for _item in ITEMS:
    for _rank, _c in enumerate(_item.concepts):
        tax, name = (_c.split(":", 1) if ":" in _c else ("us-gaap", _c))
        CONCEPT_TO_ITEMS.setdefault((tax, name), []).append((_item.id, _rank))
