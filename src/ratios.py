"""Earnings-quality signals computed from the tidy annual financials.

Every function here takes a company's years in order and returns one value per year,
with NaN wherever an input is missing. Nothing is imputed silently: a company that does
not report inventory gets NaN for the inventory signals rather than a zero that would
quietly read as "no inventory growth".

The signals fall into three families:

* **Beneish M-Score** -- eight ratios combined with coefficients Messod Beneish fitted
  on known manipulators. Above roughly -1.78 is the conventional screening threshold.
* **Accruals** -- the gap between reported profit and the cash that actually arrived.
  Richard Sloan showed that firms whose earnings are mostly accruals go on to
  underperform, whether or not anyone was cheating.
* **Divergence checks** -- receivables or inventory growing faster than sales, which is
  what channel stuffing and revenue-recognition games look like in the statements.

None of these prove anything. They are screens: they say a company's numbers are
unusual relative to its own past, which has innocent explanations far more often than
not.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Beneish's fitted coefficients, in the order the components are listed below.
BENEISH_COEFFICIENTS: dict[str, float] = {
    "intercept": -4.840,
    "dsri": 0.920,
    "gmi": 0.528,
    "aqi": 0.404,
    "sgi": 0.892,
    "depi": 0.115,
    "sgai": -0.172,
    "tata": 4.679,
    "lvgi": -0.327,
}

#: Conventional screening cutoff. Beneish's own paper discusses both -1.78 and -2.22;
#: the looser threshold catches more manipulators at the cost of more false positives.
M_SCORE_THRESHOLD = -1.78


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division that yields NaN instead of inf when the denominator is 0."""
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def _prior(series: pd.Series) -> pd.Series:
    """The previous fiscal year's value. Assumes rows are sorted by fiscal_year."""
    return series.shift(1)


def beneish_components(df: pd.DataFrame) -> pd.DataFrame:
    """The eight Beneish indices for one company, one row per fiscal year.

    Args:
        df: One company's rows, sorted ascending by ``fiscal_year``, with the concept
            columns produced by :func:`src.parse.annual_frame`.

    Returns:
        A frame indexed like ``df`` with columns ``dsri``, ``gmi``, ``aqi``, ``sgi``,
        ``depi``, ``sgai``, ``tata``, ``lvgi``. The first year of any company is all
        NaN, because every index except TATA compares against the prior year.
    """
    out = pd.DataFrame(index=df.index)
    sales, cogs, assets = df["revenue"], df["cogs"], df["total_assets"]

    # Days Sales in Receivables Index: receivables rising faster than sales is the
    # classic signature of revenue booked before the cash is collectable.
    receivable_intensity = _safe_divide(df["receivables"], sales)
    out["dsri"] = _safe_divide(receivable_intensity, _prior(receivable_intensity))

    # Gross Margin Index: >1 means margins deteriorated, which pressures management.
    gross_margin = _safe_divide(sales - cogs, sales)
    # A ratio of margins is meaningless once either year's margin is zero or
    # negative: the sign flips and the magnitude explodes. Those company-years get
    # NaN rather than a number that looks like evidence.
    gross_margin = gross_margin.where(gross_margin > 0)
    out["gmi"] = _safe_divide(_prior(gross_margin), gross_margin)

    # Asset Quality Index: the share of assets that is neither current nor hard
    # property, i.e. the soft, capitalisable kind. A jump suggests costs are being
    # parked on the balance sheet instead of run through the income statement.
    soft = assets - (
        df["current_assets"].fillna(0)
        + df["ppe_net"].fillna(0)
        + df["securities"].fillna(0)
    )
    asset_quality = _safe_divide(soft, assets)
    # Only meaningful where the company actually reports a classified balance sheet.
    asset_quality = asset_quality.where(df["current_assets"].notna())
    # The share of assets that is "soft" cannot be below 0 or above 1. It can still
    # stray outside that range when a company's components overlap or when a tag
    # switch puts two definitions in the same series, and a near-zero denominator
    # then sends the index into the thousands. Clipping keeps one bad row from
    # dominating the ranking.
    asset_quality = asset_quality.clip(0.0, 1.0)
    out["aqi"] = _safe_divide(asset_quality, _prior(asset_quality))

    # Sales Growth Index: growth is not manipulation, but it correlates with the
    # pressure to keep the streak going.
    out["sgi"] = _safe_divide(sales, _prior(sales))

    # Depreciation Index: >1 means the depreciation rate slowed, which inflates
    # earnings by stretching asset lives.
    dep = df["depreciation"]
    dep_rate = _safe_divide(dep, dep + df["ppe_net"])
    out["depi"] = _safe_divide(_prior(dep_rate), dep_rate)

    # SG&A Index: overheads rising faster than sales signals a deteriorating business.
    sga_intensity = _safe_divide(df["sga"], sales)
    out["sgai"] = _safe_divide(sga_intensity, _prior(sga_intensity))

    # Leverage Index.
    leverage = _safe_divide(
        df["current_liabilities"].fillna(0) + df["long_term_debt"].fillna(0), assets
    )
    leverage = leverage.where(
        df["current_liabilities"].notna() | df["long_term_debt"].notna()
    )
    out["lvgi"] = _safe_divide(leverage, _prior(leverage))

    # Total Accruals to Total Assets: the single heaviest-weighted term. Profit that
    # never became cash.
    income = df["income_continuing"].where(df["income_continuing"].notna(), df["net_income"])
    out["tata"] = _safe_divide(income - df["cfo"], assets)

    return out


def beneish_m_score(
    df: pd.DataFrame, neutral_substitution: bool = False
) -> pd.DataFrame:
    """Combine the eight indices into the M-Score.

    Args:
        df: One company's rows, sorted by ``fiscal_year``.
        neutral_substitution: When True, a missing index is replaced by 1.0, the value
            it takes when nothing changed year over year. This lets companies with
            partial data receive a score, at the cost of understating the components
            that are absent. The count of substitutions is always reported so the
            caller can discard heavily imputed scores.

    Returns:
        Columns ``m_score``, ``m_score_flag`` (above the threshold), and
        ``m_components_imputed``.
    """
    components = beneish_components(df)
    return combine_m_score(components, neutral_substitution=neutral_substitution)


INDEX_NAMES: list[str] = [n for n in BENEISH_COEFFICIENTS if n != "intercept"]

#: Percentile at each tail where the indices are capped before being combined.
WINSOR_LIMIT = 0.01

#: Winsorising a handful of rows just clips them to their own min and max, which
#: is meaningless. Below this many scoreable rows the panel is left alone.
MIN_ROWS_FOR_WINSOR = 100

#: Sales growth outside this band marks a year as structurally odd rather than
#: suspicious: a COVID collapse, a first product launch, a transformative merger.
SANE_SALES_GROWTH = (0.5, 2.0)

#: Total assets growing by more than this in one year almost always means an
#: acquisition. Measured on S&P 500 data: the 100 highest M-Scores had median asset
#: growth of 54.6% against 5.3% for everyone else, and company-years combining a
#: >25% asset jump with receivables outpacing sales by 50 points were flagged 74% of
#: the time against a 3.5% base rate. The model is reading the target's balance
#: sheet, not manipulation.
ASSET_JUMP_THRESHOLD = 0.25


def combine_m_score(
    components: pd.DataFrame, neutral_substitution: bool = False
) -> pd.DataFrame:
    """Apply Beneish's coefficients to a frame of the eight indices."""
    imputed = components[INDEX_NAMES].isna().sum(axis=1)
    working = components[INDEX_NAMES]
    if neutral_substitution:
        # TATA is a level, not a ratio; its neutral value is 0, not 1.
        working = working.fillna(
            {name: (0.0 if name == "tata" else 1.0) for name in INDEX_NAMES}
        )

    score = pd.Series(
        BENEISH_COEFFICIENTS["intercept"], index=components.index, dtype="float64"
    )
    for name in INDEX_NAMES:
        score = score + BENEISH_COEFFICIENTS[name] * working[name]

    result = components.copy()
    result["m_score"] = score
    result["m_score_flag"] = score > M_SCORE_THRESHOLD
    result["m_components_imputed"] = imputed
    result.loc[result["m_score"].isna(), "m_score_flag"] = False
    return result


def winsorize_indices(
    components: pd.DataFrame, limit: float = WINSOR_LIMIT
) -> tuple[pd.DataFrame, pd.Series]:
    """Cap each Beneish index at its own percentiles across the whole panel.

    Beneish's indices are ratios of ratios, so a company whose sales fell 99% in a
    year produces values orders of magnitude from the rest of the distribution. With
    TATA carrying a coefficient of 4.679, a handful of those rows take over the top
    of any ranking: measured on S&P 500 data the asset quality index reached 3,888
    against a 99th percentile of 4.5, and the worst "fraud" the screen found was a
    cruise line during the pandemic.

    Capping at the 1st and 99th percentile keeps those years in the sample, with a
    high but finite score, instead of letting arithmetic decide the ranking. Beneish
    winsorised when fitting the model, so this also brings the inputs closer to the
    data the coefficients were estimated on.

    Args:
        components: Panel of the eight indices, one row per company-year.
        limit: Tail fraction to clip at each end.

    Returns:
        The capped frame, and a count per row of how many of its indices were capped.
    """
    capped = components.copy()
    was_capped = pd.DataFrame(False, index=components.index, columns=INDEX_NAMES)

    scoreable = components[INDEX_NAMES].notna().any(axis=1).sum()
    if scoreable < MIN_ROWS_FOR_WINSOR:
        log.debug("Only %d scoreable rows; skipping winsorisation", scoreable)
        return capped, was_capped.sum(axis=1)

    for name in INDEX_NAMES:
        series = components[name].replace([np.inf, -np.inf], np.nan)
        low, high = series.quantile(limit), series.quantile(1 - limit)
        was_capped[name] = series.notna() & ((series < low) | (series > high))
        capped[name] = series.clip(low, high)

    return capped, was_capped.sum(axis=1)


def quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Mark company-years whose inputs make any ratio model unreliable.

    These are not accusations and not errors. A year in which sales halved or
    tripled breaks the assumption behind every year-over-year index, so the right
    move is to label it and let the reader decide, rather than quietly ranking it
    first.
    """
    ordered = df.sort_values("fiscal_year")
    growth = _safe_divide(ordered["revenue"], _prior(ordered["revenue"]))
    low, high = SANE_SALES_GROWTH

    asset_growth = _safe_divide(
        ordered["total_assets"], _prior(ordered["total_assets"])
    ) - 1

    out = pd.DataFrame(index=ordered.index)
    out["revenue_growth"] = growth - 1
    out["asset_growth"] = asset_growth
    out["extreme_revenue_change"] = growth.notna() & ~growth.between(low, high)
    out["asset_jump"] = asset_growth > ASSET_JUMP_THRESHOLD
    out["loss_making"] = ordered["net_income"] < 0
    out["negative_gross_margin"] = (ordered["revenue"] - ordered["cogs"]) <= 0
    # One column the reader can sort on: this year's inputs are distorted by
    # something structural, so a high score needs checking before it means anything.
    out["review_required"] = (
        out["extreme_revenue_change"] | out["asset_jump"] | out["negative_gross_margin"]
    )
    return out


def accruals_ratio(df: pd.DataFrame) -> pd.Series:
    """Sloan's accruals: (net income - operating cash flow) / average total assets.

    High positive values mean profit is largely bookkeeping rather than cash. Average
    assets, rather than ending assets, keeps an acquisition from distorting the ratio.
    """
    average_assets = (df["total_assets"] + _prior(df["total_assets"])) / 2
    return _safe_divide(df["net_income"] - df["cfo"], average_assets)


def cash_conversion(df: pd.DataFrame) -> pd.Series:
    """Operating cash flow divided by net income.

    Below 1 persistently means reported profit is not turning into cash. Returns NaN
    where net income is negative, since the ratio is not interpretable there.
    """
    positive_income = df["net_income"].where(df["net_income"] > 0)
    return _safe_divide(df["cfo"], positive_income)


def growth_divergence(df: pd.DataFrame, column: str) -> pd.Series:
    """How much faster a balance-sheet item grew than sales, in percentage points.

    Receivables outrunning sales is what channel stuffing looks like; inventory
    outrunning sales is what unsold goods look like before the write-down.
    """
    item_growth = _safe_divide(df[column], _prior(df[column])) - 1
    sales_growth = _safe_divide(df["revenue"], _prior(df["revenue"])) - 1
    return item_growth - sales_growth


def days_sales_outstanding(df: pd.DataFrame) -> pd.Series:
    """Average days to collect a sale. A rising trend is the readable form of DSRI."""
    return _safe_divide(df["receivables"] * 365, df["revenue"])


def score_company(df: pd.DataFrame, neutral_substitution: bool = False) -> pd.DataFrame:
    """Run every signal over one company's history.

    Args:
        df: One company's rows. Sorted by fiscal year internally, so callers need not.
        neutral_substitution: Passed through to :func:`beneish_m_score`.

    Returns:
        The input columns plus every signal, one row per fiscal year.
    """
    ordered = df.sort_values("fiscal_year").reset_index(drop=True)
    signals = beneish_m_score(ordered, neutral_substitution=neutral_substitution)

    signals["accruals_ratio"] = accruals_ratio(ordered)
    signals["cash_conversion"] = cash_conversion(ordered)
    signals["receivables_vs_sales"] = growth_divergence(ordered, "receivables")
    signals["inventory_vs_sales"] = growth_divergence(ordered, "inventory")
    signals["days_sales_outstanding"] = days_sales_outstanding(ordered)
    signals["dso_change"] = signals["days_sales_outstanding"] - _prior(
        signals["days_sales_outstanding"]
    )

    return pd.concat([ordered, signals], axis=1)


def score_all(
    df: pd.DataFrame,
    neutral_substitution: bool = False,
    winsorize: bool = True,
    limit: float = WINSOR_LIMIT,
) -> pd.DataFrame:
    """Score a whole panel of companies, capping the indices across the sample.

    This is the function to use for a real universe. It differs from calling
    :func:`score_company` company by company in one important way: the Beneish
    indices are winsorised against the *panel* distribution before being combined,
    so one company's collapsed denominator cannot dominate the ranking. Winsorising
    needs the cross-section, which a single company does not have.

    Args:
        df: Tidy financials for many companies, with a ``ticker`` column.
        neutral_substitution: Passed through to the M-Score.
        winsorize: Set False to see the raw, uncapped scores.
        limit: Tail fraction clipped at each end.

    Returns:
        Every input column, every signal, the capped indices, and the quality
        flags, with ``m_components_winsorized`` recording how many of a row's eight
        indices were capped.
    """
    if df.empty:
        return df

    scored = pd.concat(
        [
            score_company(group, neutral_substitution=neutral_substitution)
            for _, group in df.groupby("ticker", sort=False)
        ],
        ignore_index=True,
    )

    flags = pd.concat(
        [quality_flags(group) for _, group in scored.groupby("ticker", sort=False)]
    ).sort_index()
    scored = pd.concat([scored, flags], axis=1)

    if not winsorize:
        scored["m_components_winsorized"] = 0
        return scored

    capped, n_capped = winsorize_indices(scored[INDEX_NAMES], limit=limit)
    recombined = combine_m_score(capped, neutral_substitution=neutral_substitution)

    for column in INDEX_NAMES + ["m_score", "m_score_flag", "m_components_imputed"]:
        scored[column] = recombined[column]
    scored["m_components_winsorized"] = n_capped
    return scored
