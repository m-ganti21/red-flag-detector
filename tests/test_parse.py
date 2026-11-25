"""Tests for the EDGAR parsing layer, run against a hand-built payload.

These use a synthetic ``companyfacts`` document rather than live SEC data so the suite
runs offline and deterministically. The fixture deliberately contains the three things
that break naive parsers: a non-standard revenue tag, quarterly facts mixed in with
annual ones, and a restated prior year.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from src.concepts import REVENUE, by_name
from src.parse import (
    annual_frame,
    as_first_reported,
    concept_observations,
    fiscal_year_of,
    restatement_report,
)


def usd(entries: list[dict]) -> dict:
    return {"units": {"USD": entries}}


@pytest.fixture
def payload() -> dict:
    """A small company that reports revenue under a legacy tag and restated FY2022."""
    return {
        "cik": 1234567,
        "entityName": "Testco Industries Inc.",
        "facts": {
            "us-gaap": {
                # Legacy tag: the parser should fall through to it because the
                # preferred ASC 606 tags are absent.
                "SalesRevenueNet": usd(
                    [
                        # FY2022 as originally filed in the FY2022 10-K.
                        {
                            "start": "2022-01-01", "end": "2022-12-31", "val": 1_000_000,
                            "form": "10-K", "filed": "2023-02-15", "accn": "0001-23-000001",
                        },
                        # FY2022 again, restated downward in the FY2023 10-K.
                        {
                            "start": "2022-01-01", "end": "2022-12-31", "val": 900_000,
                            "form": "10-K", "filed": "2024-02-15", "accn": "0001-24-000001",
                        },
                        # FY2023.
                        {
                            "start": "2023-01-01", "end": "2023-12-31", "val": 1_300_000,
                            "form": "10-K", "filed": "2024-02-15", "accn": "0001-24-000001",
                        },
                        # A quarter, which must be excluded.
                        {
                            "start": "2023-01-01", "end": "2023-03-31", "val": 300_000,
                            "form": "10-Q", "filed": "2023-05-01", "accn": "0001-23-000002",
                        },
                        # An annual-length period filed on a 10-Q: also excluded.
                        {
                            "start": "2022-10-01", "end": "2023-09-30", "val": 1_200_000,
                            "form": "10-Q", "filed": "2023-11-01", "accn": "0001-23-000003",
                        },
                    ]
                ),
                "CostOfGoodsAndServicesSold": usd(
                    [
                        {"start": "2022-01-01", "end": "2022-12-31", "val": 600_000,
                         "form": "10-K", "filed": "2023-02-15", "accn": "0001-23-000001"},
                        {"start": "2023-01-01", "end": "2023-12-31", "val": 800_000,
                         "form": "10-K", "filed": "2024-02-15", "accn": "0001-24-000001"},
                    ]
                ),
                "NetIncomeLoss": usd(
                    [
                        {"start": "2022-01-01", "end": "2022-12-31", "val": 120_000,
                         "form": "10-K", "filed": "2023-02-15", "accn": "0001-23-000001"},
                        {"start": "2023-01-01", "end": "2023-12-31", "val": 150_000,
                         "form": "10-K", "filed": "2024-02-15", "accn": "0001-24-000001"},
                    ]
                ),
                "NetCashProvidedByUsedInOperatingActivities": usd(
                    [
                        {"start": "2022-01-01", "end": "2022-12-31", "val": 90_000,
                         "form": "10-K", "filed": "2023-02-15", "accn": "0001-23-000001"},
                        {"start": "2023-01-01", "end": "2023-12-31", "val": 60_000,
                         "form": "10-K", "filed": "2024-02-15", "accn": "0001-24-000001"},
                    ]
                ),
                "AccountsReceivableNetCurrent": usd(
                    [
                        {"end": "2022-12-31", "val": 200_000, "form": "10-K",
                         "filed": "2023-02-15", "accn": "0001-23-000001"},
                        {"end": "2023-12-31", "val": 400_000, "form": "10-K",
                         "filed": "2024-02-15", "accn": "0001-24-000001"},
                        # A quarterly balance, which must be excluded.
                        {"end": "2023-03-31", "val": 250_000, "form": "10-Q",
                         "filed": "2023-05-01", "accn": "0001-23-000002"},
                    ]
                ),
                "Assets": usd(
                    [
                        {"end": "2022-12-31", "val": 2_000_000, "form": "10-K",
                         "filed": "2023-02-15", "accn": "0001-23-000001"},
                        {"end": "2023-12-31", "val": 2_400_000, "form": "10-K",
                         "filed": "2024-02-15", "accn": "0001-24-000001"},
                    ]
                ),
            }
        },
    }


class TestFiscalYearLabelling:
    def test_calendar_year_end_keeps_its_year(self):
        assert fiscal_year_of(dt.date(2023, 12, 31)) == 2023

    def test_september_year_end_keeps_its_year(self):
        # Apple's FY2023 ended 30 September 2023.
        assert fiscal_year_of(dt.date(2023, 9, 30)) == 2023

    def test_january_year_end_rolls_back(self):
        # Walmart calls the year ending 31 January 2024 "fiscal 2024", but almost all of
        # it happened in calendar 2023, so we align it with 2023.
        assert fiscal_year_of(dt.date(2024, 1, 31)) == 2023

    def test_june_is_the_cutoff(self):
        assert fiscal_year_of(dt.date(2023, 6, 1)) == 2023
        assert fiscal_year_of(dt.date(2023, 5, 31)) == 2022


class TestConceptObservations:
    def test_falls_back_to_a_legacy_tag(self, payload):
        observed = concept_observations(payload, REVENUE)
        assert not observed.empty
        assert set(observed["tag"]) == {"SalesRevenueNet"}

    def test_excludes_quarterly_and_non_10k_facts(self, payload):
        observed = concept_observations(payload, REVENUE)
        # Three 10-K entries survive; the two 10-Q entries do not.
        assert len(observed) == 3
        assert set(observed["fiscal_year"]) == {2022, 2023}

    def test_excludes_quarterly_balance_sheet_facts(self, payload):
        observed = concept_observations(payload, by_name("receivables"))
        assert sorted(observed["fiscal_year"]) == [2022, 2023]

    def test_missing_concept_returns_empty_frame(self, payload):
        observed = concept_observations(payload, by_name("inventory"))
        assert observed.empty


class TestCrossTagMerging:
    """The Caterpillar bug: a company that re-tags a line mid-decade.

    CAT reported net income under NetIncomeLoss through 2010 and ProfitLoss after.
    An earlier parser took the first tag with any data and stopped, truncating the
    company to four usable years.
    """

    @pytest.fixture
    def retagged(self, payload) -> dict:
        payload["facts"]["us-gaap"]["NetIncomeLoss"] = usd(
            [
                {"start": "2022-01-01", "end": "2022-12-31", "val": 120_000,
                 "form": "10-K", "filed": "2023-02-15", "accn": "0001-23-000001"},
            ]
        )
        payload["facts"]["us-gaap"]["ProfitLoss"] = usd(
            [
                {"start": "2023-01-01", "end": "2023-12-31", "val": 150_000,
                 "form": "10-K", "filed": "2024-02-15", "accn": "0001-24-000001"},
            ]
        )
        return payload

    def test_both_tags_contribute_their_own_years(self, retagged):
        collapsed = as_first_reported(
            concept_observations(retagged, by_name("net_income"))
        )
        assert sorted(collapsed["fiscal_year"]) == [2022, 2023]
        assert set(collapsed["tag"]) == {"NetIncomeLoss", "ProfitLoss"}

    def test_series_is_continuous_in_the_output_frame(self, retagged):
        frame = annual_frame(retagged, min_required=5)
        assert list(frame["fiscal_year"]) == [2022, 2023]
        assert frame["net_income"].notna().all()

    def test_higher_priority_tag_wins_a_contested_year(self, retagged):
        # Both tags now cover 2023; NetIncomeLoss is ranked first, so it should win.
        retagged["facts"]["us-gaap"]["NetIncomeLoss"]["units"]["USD"].append(
            {"start": "2023-01-01", "end": "2023-12-31", "val": 999_000,
             "form": "10-K", "filed": "2024-02-15", "accn": "0001-24-000001"}
        )
        collapsed = as_first_reported(
            concept_observations(retagged, by_name("net_income"))
        )
        fy2023 = collapsed.loc[collapsed["fiscal_year"] == 2023].iloc[0]
        assert fy2023["tag"] == "NetIncomeLoss"
        assert fy2023["value"] == 999_000

    def test_restatements_are_not_invented_across_tags(self, retagged):
        # A value from one tag must never be compared against another tag's value.
        retagged["facts"]["us-gaap"]["ProfitLoss"]["units"]["USD"].append(
            {"start": "2022-01-01", "end": "2022-12-31", "val": 1, "form": "10-K",
             "filed": "2025-02-15", "accn": "0001-25-000001"}
        )
        report = restatement_report(retagged)
        assert "net_income" not in set(report["concept"])


class TestAsFirstReported:
    def test_keeps_the_originally_filed_value(self, payload):
        collapsed = as_first_reported(concept_observations(payload, REVENUE))
        fy2022 = collapsed.loc[collapsed["fiscal_year"] == 2022].iloc[0]
        assert fy2022["value"] == 1_000_000       # as first reported
        assert fy2022["latest_value"] == 900_000  # after the restatement
        assert fy2022["n_filings"] == 2

    def test_one_row_per_fiscal_year(self, payload):
        collapsed = as_first_reported(concept_observations(payload, REVENUE))
        assert collapsed["fiscal_year"].is_unique


class TestAnnualFrame:
    def test_builds_one_row_per_year(self, payload):
        frame = annual_frame(payload, min_required=5)
        assert list(frame["fiscal_year"]) == [2022, 2023]
        assert frame["company"].iloc[0] == "Testco Industries Inc."
        assert frame["cik"].iloc[0] == "0001234567"

    def test_uses_as_reported_revenue(self, payload):
        frame = annual_frame(payload, min_required=5)
        assert frame.loc[frame["fiscal_year"] == 2022, "revenue"].iloc[0] == 1_000_000

    def test_records_which_tag_supplied_each_value(self, payload):
        frame = annual_frame(payload, min_required=5)
        assert frame["tag_revenue"].iloc[0] == "SalesRevenueNet"

    def test_dates_each_row(self, payload):
        frame = annual_frame(payload, min_required=5)
        assert frame["period_end"].iloc[0] == dt.date(2022, 12, 31)

    def test_strict_mode_keeps_years_with_every_required_concept(self, payload):
        # Testco omits inventory and current assets, but both are optional, so strict
        # mode still accepts both years.
        assert len(annual_frame(payload)) == 2

    def test_strict_mode_drops_years_missing_a_required_concept(self, payload):
        # Remove cash flow from operations, which is required, and strict mode should
        # reject every year while a loosened threshold still accepts them.
        del payload["facts"]["us-gaap"]["NetCashProvidedByUsedInOperatingActivities"]
        assert annual_frame(payload).empty
        assert len(annual_frame(payload, min_required=5)) == 2

    def test_empty_payload_returns_empty_frame(self):
        assert annual_frame({"cik": 1, "entityName": "Nobody", "facts": {}}).empty


class TestRestatementReport:
    def test_flags_the_restated_year(self, payload):
        report = restatement_report(payload)
        revenue_rows = report[report["concept"] == "revenue"]
        assert len(revenue_rows) == 1
        row = revenue_rows.iloc[0]
        assert row["fiscal_year"] == 2022
        assert row["first_reported"] == 1_000_000
        assert row["latest_reported"] == 900_000
        assert row["pct_change"] == pytest.approx(-0.10)

    def test_ignores_unrestated_concepts(self, payload):
        report = restatement_report(payload)
        assert "net_income" not in set(report["concept"])


def test_pipeline_output_is_usable_by_pandas(payload):
    """The output should be a plain tidy frame, ready for the scoring layer."""
    frame = annual_frame(payload, min_required=5)
    assert isinstance(frame, pd.DataFrame)
    assert frame["fiscal_year"].dtype.kind == "i"
    assert frame["revenue"].dtype.kind == "f"
