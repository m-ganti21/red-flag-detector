"""Tests for the backtest layer.

The most important test here is the look-ahead one. Every other bug in a backtest
makes the result wrong; look-ahead bias makes it wrong *and* flattering, which is
why it survives review so often.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    DEFAULT_FILING_LAG_DAYS,
    add_excess_returns,
    attach_prices,
    decile_table,
    spread_test,
    yearly_spread,
)


@pytest.fixture
def prices() -> pd.DataFrame:
    """Monthly prices for two tickers, one compounding, one flat."""
    dates = pd.date_range("2018-01-01", "2024-12-01", freq="MS")
    rising = pd.DataFrame(
        {"date": dates, "ticker": "UP", "close": 100 * (1.01 ** np.arange(len(dates)))}
    )
    flat = pd.DataFrame({"date": dates, "ticker": "FLAT", "close": 50.0})
    return pd.concat([rising, flat], ignore_index=True)


@pytest.fixture
def scored() -> pd.DataFrame:
    rows = []
    for ticker in ("UP", "FLAT"):
        for year in (2019, 2020, 2021):
            rows.append(
                {
                    "ticker": ticker,
                    "fiscal_year": year,
                    "period_end": f"{year}-12-31",
                    "m_score": -2.0,
                    "accruals_ratio": 0.05,
                    "sector": "Industrials",
                }
            )
    return pd.DataFrame(rows)


class TestNoLookAhead:
    """Entry must never precede the date the filing became public."""

    def test_entry_is_after_the_period_end(self, scored, prices):
        panel = attach_prices(scored, prices)
        gap = (panel["entry"] - panel["period_end"]).dt.days
        assert (gap == DEFAULT_FILING_LAG_DAYS).all()

    def test_entry_price_comes_from_on_or_after_the_entry_date(self, scored):
        """Price encodes its own month, so we can prove which observation was used."""
        dates = pd.date_range("2018-01-01", "2024-12-01", freq="MS")
        coded = pd.DataFrame(
            {"date": dates, "ticker": "UP", "close": np.arange(len(dates), dtype=float)}
        )
        only_up = scored[scored["ticker"] == "UP"]
        panel = attach_prices(only_up, coded)

        lookup = dict(zip(coded["close"], coded["date"]))
        for _, row in panel.dropna(subset=["price_entry"]).iterrows():
            matched_date = lookup[row["price_entry"]]
            assert matched_date >= row["entry"], (
                f"used a price from {matched_date.date()} for an entry on "
                f"{row['entry'].date()} -- that is look-ahead bias"
            )
            # And it should be the *first* such month, not a later one.
            assert (matched_date - row["entry"]).days < 32

    def test_a_longer_lag_moves_entry_later(self, scored, prices):
        early = attach_prices(scored, prices, filing_lag_days=30)
        late = attach_prices(scored, prices, filing_lag_days=180)
        assert (late["entry"] > early["entry"]).all()

    def test_holding_period_is_respected(self, scored, prices):
        panel = attach_prices(scored, prices, holding_days=730)
        assert ((panel["exit"] - panel["entry"]).dt.days == 730).all()


class TestForwardReturns:
    def test_rising_stock_has_positive_return(self, scored, prices):
        panel = attach_prices(scored, prices)
        up = panel[panel["ticker"] == "UP"]
        assert (up["forward_return"].dropna() > 0).all()

    def test_flat_stock_has_zero_return(self, scored, prices):
        panel = attach_prices(scored, prices)
        flat = panel[panel["ticker"] == "FLAT"]
        assert flat["forward_return"].dropna().abs().max() < 1e-9

    def test_missing_prices_give_nan_not_a_dropped_row(self, scored, prices):
        # A company with no price history at all must still appear, with NaN.
        orphan = scored.copy()
        orphan["ticker"] = "GONE"
        panel = attach_prices(pd.concat([scored, orphan], ignore_index=True), prices)
        assert (panel["ticker"] == "GONE").sum() == len(orphan)
        assert panel.loc[panel["ticker"] == "GONE", "forward_return"].isna().all()


class TestExcessReturns:
    def test_peer_mean_is_removed(self, scored, prices):
        panel = add_excess_returns(attach_prices(scored, prices))
        usable = panel[panel["excess_return"].notna()]
        by_year = usable.groupby("entry_year")["excess_return"].mean()
        assert by_year.abs().max() < 1e-9

    def test_unknown_peer_column_is_rejected(self, scored, prices):
        panel = attach_prices(scored, prices)
        with pytest.raises(KeyError):
            add_excess_returns(panel, neutral_by=("not_a_column",))


def _signal_panel(n: int = 400, seed: int = 3) -> pd.DataFrame:
    """A panel where a known signal genuinely predicts returns, by construction."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    # High signal -> low return, the relationship a red flag should have.
    returns = -0.15 * signal + rng.normal(scale=0.05, size=n)
    return pd.DataFrame(
        {
            "ticker": [f"T{i:04d}" for i in range(n)],
            "entry_year": rng.choice([2019, 2020, 2021], size=n),
            "sector": rng.choice(["Tech", "Health"], size=n),
            "forward_return": returns,
            "planted": signal,
            "noise": rng.normal(size=n),
        }
    )


class TestSpreadTest:
    def test_detects_a_planted_signal_with_the_right_sign(self):
        result = spread_test(_signal_panel(), "planted")
        assert result.spread < 0          # high signal underperformed
        assert result.p_value < 0.01
        assert result.low_n > 0 and result.high_n > 0

    def test_finds_nothing_in_noise(self):
        result = spread_test(_signal_panel(), "noise")
        assert result.p_value > 0.05

    def test_records_how_it_was_neutralised(self):
        result = spread_test(_signal_panel(), "planted", neutral_by=("entry_year",))
        assert result.neutral_by == ("entry_year",)

    def test_rejects_a_signal_with_no_usable_rows(self):
        panel = _signal_panel()
        panel["empty"] = np.nan
        with pytest.raises(ValueError):
            spread_test(panel, "empty")


class TestTables:
    def test_decile_table_slopes_for_a_real_signal(self):
        table = decile_table(_signal_panel(n=600), "planted")
        assert len(table) == 10
        assert table.loc[1, "mean"] > table.loc[10, "mean"]

    def test_yearly_spread_has_one_row_per_year(self):
        panel = _signal_panel()
        yearly = yearly_spread(panel, "planted")
        assert set(yearly.index) <= {2019, 2020, 2021}
        assert (yearly["spread"] < 0).all()      # the planted signal works every year
