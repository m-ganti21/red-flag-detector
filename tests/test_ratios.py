"""Tests for the earnings-quality signals.

The core test works a full Beneish M-Score by hand. Every component is written out
with the arithmetic that produces it, so if a coefficient or a formula is ever mistyped
the test says which index broke rather than just that a number moved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ratios import (
    M_SCORE_THRESHOLD,
    accruals_ratio,
    beneish_components,
    beneish_m_score,
    cash_conversion,
    days_sales_outstanding,
    growth_divergence,
    score_all,
    score_company,
)


@pytest.fixture
def two_years() -> pd.DataFrame:
    """A company whose receivables outran sales and whose cash flow fell.

    Worked by hand in the assertions below; M-Score comes to about -1.437.
    """
    return pd.DataFrame(
        {
            "ticker": ["TEST", "TEST"],
            "fiscal_year": [2022, 2023],
            "revenue": [1000.0, 1200.0],
            "cogs": [600.0, 750.0],
            "receivables": [100.0, 180.0],
            "inventory": [200.0, 260.0],
            "current_assets": [400.0, 500.0],
            "ppe_net": [300.0, 350.0],
            "securities": [0.0, 0.0],
            "total_assets": [1000.0, 1300.0],
            "depreciation": [50.0, 55.0],
            "sga": [100.0, 130.0],
            "current_liabilities": [200.0, 250.0],
            "long_term_debt": [100.0, 150.0],
            "net_income": [120.0, 150.0],
            "income_continuing": [120.0, 150.0],
            "cfo": [90.0, 60.0],
        }
    )


class TestBeneishComponents:
    """Each index checked against arithmetic done by hand."""

    def test_first_year_is_all_nan(self, two_years):
        components = beneish_components(two_years)
        assert components.iloc[0].isna().drop("tata").all()

    def test_dsri(self, two_years):
        # (180/1200) / (100/1000) = 0.15 / 0.10
        assert beneish_components(two_years)["dsri"].iloc[1] == pytest.approx(1.5)

    def test_gmi(self, two_years):
        # prior margin 400/1000 = 0.40; current 450/1200 = 0.375; 0.40/0.375
        assert beneish_components(two_years)["gmi"].iloc[1] == pytest.approx(0.40 / 0.375)

    def test_aqi(self, two_years):
        # soft assets: 1 - (500+350+0)/1300 = 0.3461538; prior 1 - 700/1000 = 0.30
        expected = (1 - 850 / 1300) / (1 - 700 / 1000)
        assert beneish_components(two_years)["aqi"].iloc[1] == pytest.approx(expected)

    def test_sgi(self, two_years):
        assert beneish_components(two_years)["sgi"].iloc[1] == pytest.approx(1.2)

    def test_depi(self, two_years):
        # prior rate 50/350; current 55/405
        expected = (50 / 350) / (55 / 405)
        assert beneish_components(two_years)["depi"].iloc[1] == pytest.approx(expected)

    def test_sgai(self, two_years):
        # (130/1200) / (100/1000)
        assert beneish_components(two_years)["sgai"].iloc[1] == pytest.approx(
            (130 / 1200) / 0.10
        )

    def test_lvgi(self, two_years):
        # (250+150)/1300 over (200+100)/1000
        assert beneish_components(two_years)["lvgi"].iloc[1] == pytest.approx(
            (400 / 1300) / 0.30
        )

    def test_tata_needs_no_prior_year(self, two_years):
        # (income - cfo) / assets, computable in the very first year
        components = beneish_components(two_years)
        assert components["tata"].iloc[0] == pytest.approx((120 - 90) / 1000)
        assert components["tata"].iloc[1] == pytest.approx((150 - 60) / 1300)


class TestMScore:
    def test_matches_hand_calculation(self, two_years):
        scored = beneish_m_score(two_years)
        assert scored["m_score"].iloc[1] == pytest.approx(-1.4371, abs=1e-3)

    def test_flags_above_threshold(self, two_years):
        scored = beneish_m_score(two_years)
        assert scored["m_score"].iloc[1] > M_SCORE_THRESHOLD
        assert bool(scored["m_score_flag"].iloc[1]) is True

    def test_clean_company_is_not_flagged(self, two_years):
        # Flat receivables and cash flow that tracks profit: nothing unusual.
        calm = two_years.copy()
        calm.loc[1, "receivables"] = 120.0
        calm.loc[1, "cfo"] = 185.0
        calm.loc[1, "cogs"] = 720.0
        scored = beneish_m_score(calm)
        assert scored["m_score"].iloc[1] < M_SCORE_THRESHOLD
        assert bool(scored["m_score_flag"].iloc[1]) is False

    def test_missing_input_yields_nan_not_a_flag(self, two_years):
        gappy = two_years.copy()
        gappy.loc[1, "cogs"] = np.nan
        scored = beneish_m_score(gappy)
        assert np.isnan(scored["m_score"].iloc[1])
        assert bool(scored["m_score_flag"].iloc[1]) is False

    def test_imputation_count_is_reported(self, two_years):
        gappy = two_years.copy()
        gappy.loc[:, "sga"] = np.nan
        gappy.loc[:, "depreciation"] = np.nan
        scored = beneish_m_score(gappy)
        assert scored["m_components_imputed"].iloc[1] == 2

    def test_neutral_substitution_produces_a_score(self, two_years):
        gappy = two_years.copy()
        gappy.loc[:, "sga"] = np.nan
        strict = beneish_m_score(gappy)
        lenient = beneish_m_score(gappy, neutral_substitution=True)
        assert np.isnan(strict["m_score"].iloc[1])
        assert not np.isnan(lenient["m_score"].iloc[1])
        assert lenient["m_components_imputed"].iloc[1] == 1


class TestOtherSignals:
    def test_accruals_use_average_assets(self, two_years):
        # (150 - 60) / ((1300 + 1000) / 2)
        assert accruals_ratio(two_years).iloc[1] == pytest.approx(90 / 1150)

    def test_cash_conversion(self, two_years):
        assert cash_conversion(two_years).iloc[1] == pytest.approx(60 / 150)

    def test_cash_conversion_is_nan_when_income_is_negative(self, two_years):
        loss = two_years.copy()
        loss.loc[1, "net_income"] = -50.0
        assert np.isnan(cash_conversion(loss).iloc[1])

    def test_receivables_outran_sales(self, two_years):
        # receivables +80%, sales +20% => 60 points of divergence
        assert growth_divergence(two_years, "receivables").iloc[1] == pytest.approx(0.6)

    def test_days_sales_outstanding(self, two_years):
        assert days_sales_outstanding(two_years).iloc[1] == pytest.approx(180 * 365 / 1200)

    def test_division_by_zero_gives_nan_not_infinity(self, two_years):
        zeroed = two_years.copy()
        zeroed.loc[1, "revenue"] = 0.0
        result = days_sales_outstanding(zeroed)
        assert np.isnan(result.iloc[1])
        assert not np.isinf(result).any()


class TestScoreCompany:
    def test_keeps_input_columns_and_adds_signals(self, two_years):
        scored = score_company(two_years)
        assert "revenue" in scored.columns
        assert "m_score" in scored.columns
        assert "dso_change" in scored.columns
        assert len(scored) == 2

    def test_sorts_by_fiscal_year_regardless_of_input_order(self, two_years):
        shuffled = two_years.iloc[::-1].reset_index(drop=True)
        scored = score_company(shuffled)
        assert list(scored["fiscal_year"]) == [2022, 2023]
        assert scored["m_score"].iloc[1] == pytest.approx(-1.4371, abs=1e-3)

    def test_dso_change(self, two_years):
        scored = score_company(two_years)
        expected = (180 * 365 / 1200) - (100 * 365 / 1000)
        assert scored["dso_change"].iloc[1] == pytest.approx(expected)


class TestScoreAll:
    def test_companies_do_not_bleed_into_each_other(self, two_years):
        other = two_years.copy()
        other["ticker"] = "OTHR"
        other["revenue"] = [5000.0, 5100.0]
        combined = pd.concat([two_years, other], ignore_index=True)
        scored = score_all(combined)
        assert len(scored) == 4
        # TEST's first year must still be NaN; it must not borrow OTHR's prior year.
        first_rows = scored.groupby("ticker").head(1)
        assert first_rows["sgi"].isna().all()

    def test_empty_input(self):
        assert score_all(pd.DataFrame()).empty


class TestGuardsAgainstExplodingRatios:
    """Regression tests for the failure modes found on real S&P 500 data.

    Before these guards the worst M-Score in the panel was 97, belonging to a cruise
    line whose revenue had collapsed during the pandemic. The model was ranking
    volatility, not accounting.
    """

    def test_asset_quality_cannot_leave_zero_to_one(self, two_years):
        # Overlapping components (securities counted inside current assets) used to
        # push the soft-asset share negative, which inverted the index.
        overlapping = two_years.copy()
        overlapping.loc[:, "securities"] = overlapping["current_assets"]
        components = beneish_components(overlapping)
        assert (components["aqi"].dropna() >= 0).all()
        assert np.isfinite(components["aqi"].dropna()).all()

    def test_gross_margin_index_is_nan_when_margin_is_negative(self, two_years):
        loss = two_years.copy()
        loss.loc[1, "cogs"] = loss.loc[1, "revenue"] * 1.2  # selling below cost
        assert np.isnan(beneish_components(loss)["gmi"].iloc[1])

    def test_gross_margin_index_is_nan_when_margin_is_zero(self, two_years):
        breakeven = two_years.copy()
        breakeven.loc[1, "cogs"] = breakeven.loc[1, "revenue"]
        assert np.isnan(beneish_components(breakeven)["gmi"].iloc[1])


def _panel(n_companies: int = 40, years: int = 6, seed: int = 7) -> pd.DataFrame:
    """A synthetic panel big enough, and varied enough, for winsorising to mean something.

    The variation matters. An earlier version of this fixture gave every company
    identical ratios, so the 1st and 99th percentiles of each index landed on the
    same value and floating-point noise decided which rows got clipped. Real panels
    have spread; the fixture needs it too.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_companies):
        revenue = float(rng.uniform(500, 5000))
        growth = rng.uniform(1.02, 1.15)
        margin = rng.uniform(0.25, 0.55)
        receivable_days = rng.uniform(0.08, 0.20)
        asset_intensity = rng.uniform(0.9, 1.4)
        for year in range(2018, 2018 + years):
            revenue *= growth * rng.uniform(0.97, 1.03)
            assets = revenue * asset_intensity * rng.uniform(0.98, 1.02)
            rows.append(
                {
                    "ticker": f"C{i:03d}",
                    "fiscal_year": year,
                    "revenue": revenue,
                    "cogs": revenue * (1 - margin),
                    "receivables": revenue * receivable_days,
                    "inventory": revenue * rng.uniform(0.1, 0.3),
                    "current_assets": assets * 0.35,
                    "ppe_net": assets * 0.3,
                    "securities": 0.0,
                    "total_assets": assets,
                    "depreciation": assets * 0.05,
                    "sga": revenue * rng.uniform(0.08, 0.15),
                    "current_liabilities": assets * 0.2,
                    "long_term_debt": assets * 0.1,
                    "net_income": revenue * margin * 0.3,
                    "income_continuing": revenue * margin * 0.3,
                    "cfo": revenue * margin * 0.25,
                }
            )
    return pd.DataFrame(rows)


class TestWinsorizing:
    def test_caps_an_outlier_without_dropping_it(self):
        panel = _panel()
        # One company's sales collapse, the pattern that produced the 97 score.
        # Revenue and cost fall together so the margin stays positive and the row
        # remains scoreable; receivables stay put, which is what explodes DSRI.
        hit = (panel["ticker"] == "C000") & (panel["fiscal_year"] == 2022)
        panel.loc[hit, ["revenue", "cogs"]] *= 0.01

        raw = score_all(panel, winsorize=False)
        capped = score_all(panel, winsorize=True)
        victim = (capped["ticker"] == "C000") & (capped["fiscal_year"] == 2022)

        assert victim.sum() == 1                              # still in the sample
        assert raw.loc[victim, "m_score"].iloc[0] > 10        # explodes uncapped
        assert capped.loc[victim, "m_score"].iloc[0] < 5      # tamed
        assert capped["m_score"].max() < raw["m_score"].max()
        assert capped.loc[victim, "m_components_winsorized"].iloc[0] > 0

    def test_most_rows_are_untouched(self):
        panel = _panel()
        panel.loc[(panel["ticker"] == "C000") & (panel["fiscal_year"] == 2022), "revenue"] *= 0.01
        capped = score_all(panel)
        # Clipping the 1% tails of eight indices should leave the bulk alone.
        assert (capped["m_components_winsorized"] == 0).mean() > 0.80

    def test_small_panels_are_left_alone(self, two_years):
        # Winsorising four rows would just clip them to their own min and max.
        other = two_years.copy()
        other["ticker"] = "OTHR"
        combined = pd.concat([two_years, other], ignore_index=True)
        scored = score_all(combined)
        assert (scored["m_components_winsorized"] == 0).all()

    def test_winsorize_flag_changes_nothing_else(self):
        panel = _panel()
        a = score_all(panel, winsorize=False)
        b = score_all(panel, winsorize=True)
        assert list(a["ticker"]) == list(b["ticker"])
        assert len(a) == len(b)


class TestQualityFlags:
    def test_acquisition_sized_asset_jump_is_flagged(self, two_years):
        # Assets 1000 -> 1300 is +30%, past the 25% threshold.
        scored = score_all(pd.concat([two_years] * 1, ignore_index=True))
        assert bool(scored["asset_jump"].iloc[1]) is True

    def test_ordinary_growth_is_rarely_flagged(self):
        scored = score_all(_panel())
        steady = scored[scored["fiscal_year"] > 2018]
        # Companies growing 2-15% a year should almost never trip a structural flag.
        assert steady["review_required"].mean() < 0.10

    def test_revenue_collapse_is_flagged(self):
        panel = _panel()
        panel.loc[(panel["ticker"] == "C000") & (panel["fiscal_year"] == 2022), "revenue"] *= 0.01
        scored = score_all(panel)
        hit = scored[(scored["ticker"] == "C000") & (scored["fiscal_year"] == 2022)]
        assert bool(hit["extreme_revenue_change"].iloc[0]) is True
        assert bool(hit["review_required"].iloc[0]) is True

    def test_review_required_is_the_union_of_the_others(self):
        panel = _panel()
        panel.loc[(panel["ticker"] == "C001") & (panel["fiscal_year"] == 2021), "total_assets"] *= 2
        scored = score_all(panel)
        expected = (
            scored["extreme_revenue_change"]
            | scored["asset_jump"]
            | scored["negative_gross_margin"]
        )
        assert scored["review_required"].equals(expected)
