"""Test the screen against what actually happened to the share price.

A screen nobody has checked against outcomes is a calculator with opinions. This
module joins scored company-years to forward returns and asks whether the signals
separated winners from losers.

Three decisions decide whether the answer means anything:

1. **Entry is after the filing, not after the fiscal year.** A 10-K appears weeks or
   months past the period end. Measuring from the period end credits the strategy
   with information nobody had, which is look-ahead bias and the usual reason a
   backtest looks better than reality. Entry defaults to 90 days after period end.

2. **Returns are measured against a peer group, not in absolute terms.** 2009 was a
   recovery and 2022 was a drawdown; a strategy holding more names in one year than
   another would look skilful purely from timing. Excess return is computed within
   an entry-year cohort, and optionally within sector as well.

3. **Sector neutrality is not optional for this question.** Low-accrual companies in
   this sample skew heavily to technology, which outperformed for most of the
   period. Without differencing sector out, a tech tilt masquerades as an accounting
   signal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Days after the fiscal period end before the filing is assumed public. Large
#: accelerated filers owe their 10-K within 60 days, others within 75 or 90; 90 is
#: the conservative choice because it never assumes earlier knowledge than reality.
DEFAULT_FILING_LAG_DAYS = 90

#: Holding period for the forward return.
DEFAULT_HOLDING_DAYS = 365

#: How far past a target date we will accept a monthly price observation.
PRICE_TOLERANCE_DAYS = 45


@dataclass(frozen=True)
class SpreadResult:
    """The gap between the top and bottom of a signal, and whether it is real."""

    signal: str
    low_mean: float
    high_mean: float
    low_n: int
    high_n: int
    spread: float
    t_stat: float
    p_value: float
    neutral_by: tuple[str, ...]

    def __str__(self) -> str:
        stars = "***" if self.p_value < 0.01 else "**" if self.p_value < 0.05 else ""
        return (
            f"{self.signal:<18} low {self.low_mean * 100:+6.1f}%  "
            f"high {self.high_mean * 100:+6.1f}%  "
            f"spread {self.spread * 100:+6.1f}pp  p={self.p_value:.4f}{stars}"
        )


def attach_prices(
    scored: pd.DataFrame,
    prices: pd.DataFrame,
    filing_lag_days: int = DEFAULT_FILING_LAG_DAYS,
    holding_days: int = DEFAULT_HOLDING_DAYS,
) -> pd.DataFrame:
    """Add entry and exit prices and the forward return to each company-year.

    Args:
        scored: Output of :func:`src.ratios.score_all`, with ``period_end``.
        prices: Long frame of ``ticker``, ``date``, ``close`` (split/dividend adjusted).
        filing_lag_days: Days after period end before the filing is assumed public.
        holding_days: Length of the holding period.

    Returns:
        ``scored`` plus ``entry``, ``exit``, ``price_entry``, ``price_exit`` and
        ``forward_return``. Rows without a price near either date get NaN rather
        than being dropped, so coverage stays visible.
    """
    out = scored.copy().reset_index(drop=True)
    out["period_end"] = pd.to_datetime(out["period_end"])
    out["entry"] = out["period_end"] + pd.Timedelta(days=filing_lag_days)
    out["exit"] = out["entry"] + pd.Timedelta(days=holding_days)

    clean = prices.copy()
    clean["date"] = pd.to_datetime(clean["date"])
    clean = clean.dropna(subset=["close"]).sort_values("date")

    for when, label in (("entry", "price_entry"), ("exit", "price_exit")):
        left = out[["ticker", when]].reset_index().sort_values(when)
        right = clean.rename(columns={"date": when, "close": label}).sort_values(when)
        matched = pd.merge_asof(
            left,
            right,
            on=when,
            by="ticker",
            direction="forward",
            tolerance=pd.Timedelta(days=PRICE_TOLERANCE_DAYS),
        )
        out[label] = matched.set_index("index")[label].reindex(out.index)

    out["forward_return"] = out["price_exit"] / out["price_entry"] - 1
    out["entry_year"] = out["entry"].dt.year
    return out


def add_excess_returns(
    df: pd.DataFrame, neutral_by: tuple[str, ...] = ("entry_year",)
) -> pd.DataFrame:
    """Subtract the mean return of each peer group, leaving the part a signal could explain.

    Args:
        df: Output of :func:`attach_prices`.
        neutral_by: Columns defining the peer group. ``("entry_year",)`` removes the
            market's move that year; adding ``"sector"`` also removes industry drift.
    """
    out = df.copy()
    missing = [c for c in neutral_by if c not in out.columns]
    if missing:
        raise KeyError(f"Cannot neutralise by {missing}; column(s) not in the frame")
    group_mean = out.groupby(list(neutral_by))["forward_return"].transform("mean")
    out["excess_return"] = out["forward_return"] - group_mean
    return out


def spread_test(
    df: pd.DataFrame,
    signal: str,
    neutral_by: tuple[str, ...] = ("entry_year", "sector"),
    tail: float = 0.2,
    min_group: int = 5,
) -> SpreadResult:
    """Compare forward returns of the top and bottom of a signal, within peer groups.

    Ranking happens *inside* each peer group, so a year when every company scored
    badly does not fill the "high" bucket on its own.

    Args:
        df: Output of :func:`attach_prices`.
        signal: Column to rank on, e.g. ``"accruals_ratio"`` or ``"m_score"``.
        neutral_by: Peer-group columns for both ranking and the return benchmark.
        tail: Fraction at each end to compare.
        min_group: Peer groups smaller than this are skipped; a percentile rank
            over three companies is noise.

    Returns:
        A :class:`SpreadResult`. A negative spread means high values of the signal
        underperformed, which is what a working red-flag signal should do.
    """
    from scipy import stats

    usable = df[df["forward_return"].notna() & df[signal].notna()].copy()
    for column in neutral_by:
        usable = usable[usable[column].notna()]
    if usable.empty:
        raise ValueError(f"No rows with both a forward return and {signal!r}")

    keys = list(neutral_by)
    usable["_benchmark"] = usable.groupby(keys)["forward_return"].transform("mean")
    usable["_excess"] = usable["forward_return"] - usable["_benchmark"]
    usable["_rank"] = usable.groupby(keys)[signal].transform(
        lambda s: s.rank(pct=True) if s.notna().sum() >= min_group else np.nan
    )
    usable = usable[usable["_rank"].notna()]

    low = usable.loc[usable["_rank"] <= tail, "_excess"]
    high = usable.loc[usable["_rank"] >= 1 - tail, "_excess"]
    if len(low) < 2 or len(high) < 2:
        raise ValueError(f"Not enough rows in the tails of {signal!r} to compare")

    t_stat, p_value = stats.ttest_ind(high, low, equal_var=False)
    return SpreadResult(
        signal=signal,
        low_mean=float(low.mean()),
        high_mean=float(high.mean()),
        low_n=int(len(low)),
        high_n=int(len(high)),
        spread=float(high.mean() - low.mean()),
        t_stat=float(t_stat),
        p_value=float(p_value),
        neutral_by=tuple(neutral_by),
    )


def decile_table(
    df: pd.DataFrame, signal: str, buckets: int = 10, neutral_by: tuple[str, ...] = ("entry_year",)
) -> pd.DataFrame:
    """Mean and median excess return by signal decile, to show whether it is monotonic.

    A signal that works should slope. One that is flat in the middle and noisy at the
    ends is picking up something other than what it claims.
    """
    usable = df[df["forward_return"].notna() & df[signal].notna()].copy()
    usable = add_excess_returns(usable, neutral_by=neutral_by)
    usable["bucket"] = pd.qcut(usable[signal], buckets, labels=False, duplicates="drop")
    table = usable.groupby("bucket")["excess_return"].agg(["count", "mean", "median"])
    table.index = table.index.astype(int) + 1
    table.index.name = "decile"
    return table


def yearly_spread(
    df: pd.DataFrame, signal: str, quintiles: int = 5
) -> pd.DataFrame:
    """Top-minus-bottom spread year by year, to show the result is not one lucky year.

    Returns one row per entry year with the two portfolio returns and their gap.
    """
    usable = df[df["forward_return"].notna() & df[signal].notna()].copy()
    usable["q"] = usable.groupby("entry_year")[signal].transform(
        lambda s: pd.qcut(s, quintiles, labels=False, duplicates="drop")
    )
    usable = usable[usable["q"].notna()]
    pivot = usable.pivot_table(
        index="entry_year", columns="q", values="forward_return", aggfunc="mean"
    )
    if pivot.empty:
        return pivot
    low_col, high_col = pivot.columns.min(), pivot.columns.max()
    out = pd.DataFrame(
        {
            "low_quintile": pivot[low_col],
            "high_quintile": pivot[high_col],
        }
    )
    out["spread"] = out["high_quintile"] - out["low_quintile"]
    return out.dropna()
