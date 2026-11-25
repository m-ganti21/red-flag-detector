"""Mapping from the financial concepts we need to the XBRL tags companies actually use.

Why this file exists
--------------------
There is no single tag for "revenue" in SEC filings. A software company may report
``RevenueFromContractWithCustomerExcludingAssessedTax``, an industrial may use
``Revenues``, and an older filing may use ``SalesRevenueNet``. All three mean the same
thing to an analyst. The same is true for cost of goods sold, depreciation, and most
other lines we need.

Each concept below lists its candidate tags in priority order. The parser walks the list
and takes the first tag the company actually reports. Getting this mapping right is most
of the work in any EDGAR pipeline -- the ratios downstream are trivial by comparison.

Reference: https://www.sec.gov/cgi-bin/srqsb?text=form-type%3D10-K
XBRL tag definitions: https://xbrl.us/xbrl-taxonomy/2024-us-gaap/
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Concept:
    """One financial line item and the tags that might carry it.

    Attributes:
        name: Short column name used throughout the project.
        tags: Candidate XBRL tags, most preferred first.
        kind: ``"duration"`` for flow items measured over a period (revenue, expenses,
            cash flow) or ``"instant"`` for stock items measured at a point in time
            (assets, liabilities). This decides how we filter the raw facts.
        unit: XBRL unit key, almost always ``"USD"``.
        taxonomy: ``"us-gaap"`` for financial statement lines, ``"dei"`` for entity
            metadata such as share count.
        required: If ``False``, a company missing this concept is still usable; the
            column is filled with NaN and the ratios that need it are skipped.
    """

    name: str
    tags: tuple[str, ...]
    kind: str
    unit: str = "USD"
    taxonomy: str = "us-gaap"
    required: bool = True
    note: str = field(default="", compare=False)


# --- Income statement (duration) -------------------------------------------------

REVENUE = Concept(
    name="revenue",
    tags=(
        # "Revenues" is the consolidated top line and is deliberately ranked first.
        # Caterpillar, for instance, reports total revenues under this tag for every
        # year, while its contract-revenue tag excludes the Financial Products segment.
        # Preferring the contract tag would make CAT's revenue appear to fall in the
        # year it started using it, purely from a change in tagging.
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "RevenuesNetOfInterestExpense",
    ),
    kind="duration",
    note="ASC 606 split the old SalesRevenueNet tag into the two contract tags in 2018.",
)

COGS = Concept(
    name="cogs",
    tags=(
        "CostOfGoodsAndServicesSold",
        "CostOfRevenue",
        "CostOfGoodsSold",
        "CostOfServices",
    ),
    kind="duration",
    note=(
        "Variants ending in ExcludingDepreciationDepletionAndAmortization are "
        "deliberately left out. Mixing them with the tags above would make gross "
        "margin jump in the year a company switched, and the Beneish gross margin "
        "index would read that as manipulation. A missing year beats a wrong one."
    ),
)

SGA = Concept(
    name="sga",
    tags=(
        "SellingGeneralAndAdministrativeExpense",
        "GeneralAndAdministrativeExpense",
    ),
    kind="duration",
    required=False,
    note="Some filers split selling from G&A; we take the combined tag when present.",
)

DEPRECIATION = Concept(
    name="depreciation",
    tags=(
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "DepreciationAndAmortization",
        "Depreciation",
    ),
    kind="duration",
    required=False,
)

NET_INCOME = Concept(
    name="net_income",
    tags=("NetIncomeLoss", "ProfitLoss"),
    kind="duration",
)

INCOME_CONTINUING = Concept(
    name="income_continuing",
    tags=(
        "IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest",
        "IncomeLossFromContinuingOperations",
        "NetIncomeLoss",
    ),
    kind="duration",
    required=False,
    note="Beneish uses continuing operations; we fall back to net income when absent.",
)

# --- Cash flow (duration) --------------------------------------------------------

CFO = Concept(
    name="cfo",
    tags=(
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ),
    kind="duration",
)

# --- Balance sheet (instant) -----------------------------------------------------

RECEIVABLES = Concept(
    name="receivables",
    tags=(
        "AccountsReceivableNetCurrent",
        # ASU 2016-13 (the CECL standard) replaced the "net" tag with an
        # "after allowance for credit loss" one from roughly 2020. Companies that
        # adopted it stop reporting the tag above, which is why GE's receivables
        # appeared to end in 2020.
        "AccountsReceivableAfterAllowanceForCreditLossCurrent",
        "ReceivablesNetCurrent",
        "AccountsAndOtherReceivablesNetCurrent",
        "AccountsReceivableGrossCurrent",
    ),
    kind="instant",
)

INVENTORY = Concept(
    name="inventory",
    tags=("InventoryNet", "InventoryGross"),
    kind="instant",
    required=False,
    note="Service and software companies legitimately report no inventory.",
)

CURRENT_ASSETS = Concept(
    name="current_assets",
    tags=("AssetsCurrent",),
    kind="instant",
    required=False,
    note="Banks and insurers use an unclassified balance sheet and omit this.",
)

TOTAL_ASSETS = Concept(
    name="total_assets",
    tags=("Assets",),
    kind="instant",
)

PPE_NET = Concept(
    name="ppe_net",
    tags=(
        "PropertyPlantAndEquipmentNet",
        "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
    ),
    kind="instant",
    required=False,
)

SECURITIES = Concept(
    name="securities",
    tags=(
        # NONCURRENT holdings only. Beneish's asset quality index subtracts current
        # assets, PP&E and securities from total assets; short-term investments are
        # already inside current assets, so including them subtracts the same dollars
        # twice and can drive the "soft asset" share negative. Measured on real S&P
        # data, the current-securities tags produced a negative share in 5.4% of
        # company-years and an index reaching 3,888.
        "LongTermInvestments",
        "MarketableSecuritiesNoncurrent",
        "AvailableForSaleSecuritiesDebtSecuritiesNoncurrent",
        "OtherLongTermInvestments",
    ),
    kind="instant",
    required=False,
    note="Noncurrent securities only; treated as 0 when not reported.",
)

CURRENT_LIABILITIES = Concept(
    name="current_liabilities",
    tags=("LiabilitiesCurrent",),
    kind="instant",
    required=False,
)

LONG_TERM_DEBT = Concept(
    name="long_term_debt",
    tags=(
        "LongTermDebtNoncurrent",
        "LongTermDebt",
        "LongTermDebtAndCapitalLeaseObligations",
    ),
    kind="instant",
    required=False,
)

TOTAL_LIABILITIES = Concept(
    name="total_liabilities",
    tags=("Liabilities",),
    kind="instant",
    required=False,
)

STOCKHOLDERS_EQUITY = Concept(
    name="equity",
    tags=(
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    kind="instant",
    required=False,
)


#: Every concept the pipeline extracts, in a stable order.
CONCEPTS: tuple[Concept, ...] = (
    REVENUE,
    COGS,
    SGA,
    DEPRECIATION,
    NET_INCOME,
    INCOME_CONTINUING,
    CFO,
    RECEIVABLES,
    INVENTORY,
    CURRENT_ASSETS,
    TOTAL_ASSETS,
    PPE_NET,
    SECURITIES,
    CURRENT_LIABILITIES,
    LONG_TERM_DEBT,
    TOTAL_LIABILITIES,
    STOCKHOLDERS_EQUITY,
)

#: Concepts without which a company-year row is unusable.
REQUIRED_NAMES: tuple[str, ...] = tuple(c.name for c in CONCEPTS if c.required)


def by_name(name: str) -> Concept:
    """Look up a concept by its short column name."""
    for concept in CONCEPTS:
        if concept.name == name:
            return concept
    raise KeyError(f"No concept named {name!r}. Known: {[c.name for c in CONCEPTS]}")
