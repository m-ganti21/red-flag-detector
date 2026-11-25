"""Turn raw EDGAR ``companyfacts`` JSON into a tidy table of annual financials.

The raw payload is not a financial statement. It is every fact the company has ever
tagged, from every filing, including restated versions of the same number. Three
decisions turn that into something an analyst can use, and each one is a judgement call
worth defending:

1. **Which period a fact belongs to.** The ``fy`` field in a fact describes the *filing*
   that carried it, not the fact itself, so a 10-K filed in 2024 stamps its 2022
   comparatives with ``fy=2024``. We therefore derive the fiscal year from the fact's own
   ``end`` date: the calendar year it ends in, or the previous year when it ends before
   June. That puts Walmart's year ending January 2024 alongside Apple's year ending
   September 2023, because both cover mostly calendar 2023.

2. **Annual versus quarterly.** Duration facts are kept only when they span 340 to 400
   days and arrived on a 10-K, which excludes quarters, half-years, and stub periods.

3. **Original versus restated.** When the same fiscal year appears in several filings, we
   keep the number **as first reported**. A red-flag detector is meant to see what
   investors saw at the time; using the corrected figures would quietly leak the answer
   into the features. The disagreement between the first and latest versions is kept
   separately in :func:`restatement_report`, because being restated is itself a signal.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Iterable

import pandas as pd

from .concepts import CONCEPTS, REQUIRED_NAMES, Concept

log = logging.getLogger(__name__)

#: Forms that carry audited annual figures.
ANNUAL_FORMS = frozenset({"10-K", "10-K/A"})

#: A fiscal year is allowed to run from 340 to 400 days (52/53-week years, leap years).
MIN_PERIOD_DAYS = 340
MAX_PERIOD_DAYS = 400

#: Fiscal years ending before this month are labelled with the previous calendar year.
FISCAL_YEAR_CUTOFF_MONTH = 6


def fiscal_year_of(end: dt.date) -> int:
    """Label the fiscal year a period ending on ``end`` belongs to.

    Args:
        end: Last day of the reporting period.

    Returns:
        The calendar year that most of the period fell in.

    Example:
        >>> fiscal_year_of(dt.date(2023, 9, 30))   # Apple FY2023
        2023
        >>> fiscal_year_of(dt.date(2024, 1, 31))   # Walmart "fiscal 2024"
        2023
    """
    return end.year if end.month >= FISCAL_YEAR_CUTOFF_MONTH else end.year - 1


def _parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def _usable_observations(
    facts: dict[str, Any], concept: Concept, tag: str
) -> list[dict[str, Any]]:
    """Pull every annual observation of one tag out of the raw payload."""
    taxonomy_block = facts.get("facts", {}).get(concept.taxonomy, {})
    tag_block = taxonomy_block.get(tag)
    if not tag_block:
        return []
    entries = tag_block.get("units", {}).get(concept.unit, [])

    observations: list[dict[str, Any]] = []
    for entry in entries:
        if entry.get("form") not in ANNUAL_FORMS:
            continue
        if entry.get("end") is None or entry.get("val") is None:
            continue
        try:
            end = _parse_date(entry["end"])
            filed = _parse_date(entry["filed"]) if entry.get("filed") else None
        except (ValueError, TypeError):
            continue

        if concept.kind == "duration":
            if not entry.get("start"):
                continue
            try:
                start = _parse_date(entry["start"])
            except ValueError:
                continue
            span = (end - start).days
            if not MIN_PERIOD_DAYS <= span <= MAX_PERIOD_DAYS:
                continue

        observations.append(
            {
                "fiscal_year": fiscal_year_of(end),
                "period_end": end,
                "value": float(entry["val"]),
                "filed": filed or dt.date.max,
                "accession": entry.get("accn", ""),
                "tag": tag,
            }
        )
    return observations


def concept_observations(facts: dict[str, Any], concept: Concept) -> pd.DataFrame:
    """Every annual observation of one concept, across all of its candidate tags.

    Companies re-tag their statements. Caterpillar reported net income under
    ``NetIncomeLoss`` through 2010 and ``ProfitLoss`` afterwards; GE used ``Revenues``
    until ASC 606 and ``RevenueFromContractWithCustomerExcludingAssessedTax`` after.
    An earlier version of this function returned the first tag that held *any* data,
    which silently truncated those companies to the handful of years their first-choice
    tag happened to cover.

    So every candidate tag is collected, each row carrying the tag's position in the
    priority list. :func:`as_first_reported` then resolves one value per fiscal year,
    preferring the highest-priority tag available *for that year*. A company can
    therefore switch tags mid-decade and still produce one continuous series.
    """
    rows: list[dict[str, Any]] = []
    for priority, tag in enumerate(concept.tags):
        for row in _usable_observations(facts, concept, tag):
            rows.append({**row, "tag_priority": priority})
    if not rows:
        return pd.DataFrame(
            columns=[
                "fiscal_year", "period_end", "value", "filed",
                "accession", "tag", "tag_priority",
            ]
        )
    return pd.DataFrame(rows)


def as_first_reported(observations: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per fiscal year, keeping the earliest-filed version.

    Also records how many distinct filings reported the year and what the most recent
    version says, so restatements can be measured later.
    """
    if observations.empty:
        return observations

    ordered = observations.sort_values(
        ["fiscal_year", "tag_priority", "filed", "accession"]
    )

    # Within each fiscal year, keep only rows from the best-ranked tag that covers it.
    # Comparing a first-reported value from one tag against a restated value from
    # another would manufacture restatements that never happened.
    winners = (
        ordered.groupby("fiscal_year")["tag_priority"].min().rename("win_priority")
    )
    ordered = ordered.merge(winners, on="fiscal_year")
    ordered = ordered[ordered["tag_priority"] == ordered["win_priority"]]

    grouped = ordered.groupby("fiscal_year", as_index=False)
    first = grouped.first().rename(columns={"filed": "first_filed"})
    latest = grouped.last()[["fiscal_year", "value"]].rename(
        columns={"value": "latest_value"}
    )
    counts = grouped.size().rename(columns={"size": "n_filings"})

    return first.merge(latest, on="fiscal_year").merge(counts, on="fiscal_year")


def annual_frame(
    facts: dict[str, Any],
    concepts: Iterable[Concept] = CONCEPTS,
    min_required: int | None = None,
) -> pd.DataFrame:
    """Build one tidy row per fiscal year for a single company.

    Args:
        facts: Raw ``companyfacts`` payload from :meth:`EdgarClient.company_facts`.
        concepts: Which concepts to extract. Defaults to all of them.
        min_required: Drop any fiscal year missing more than this many required
            concepts. ``None`` means every required concept must be present.

    Returns:
        A DataFrame indexed by nothing, with columns ``cik``, ``company``,
        ``fiscal_year``, ``period_end``, one column per concept, and a
        ``tag_<concept>`` column recording which XBRL tag supplied each value.
    """
    cik = str(facts.get("cik", "")).zfill(10)
    company = facts.get("entityName", "")

    frames: list[pd.DataFrame] = []
    period_ends: dict[int, dt.date] = {}
    for concept in concepts:
        collapsed = as_first_reported(concept_observations(facts, concept))
        if collapsed.empty:
            continue
        # Balance-sheet and income-statement dates can differ by a day or two; the
        # first one seen for a year is good enough to date the row.
        for year, end in zip(collapsed["fiscal_year"], collapsed["period_end"]):
            period_ends.setdefault(int(year), end)
        piece = collapsed[["fiscal_year", "value", "tag"]].rename(
            columns={"value": concept.name, "tag": f"tag_{concept.name}"}
        )
        frames.append(piece)

    if not frames:
        return pd.DataFrame()

    merged = frames[0]
    for piece in frames[1:]:
        merged = merged.merge(piece, on="fiscal_year", how="outer")

    merged["fiscal_year"] = merged["fiscal_year"].astype(int)
    merged.insert(0, "period_end", merged["fiscal_year"].map(period_ends))
    merged.insert(0, "company", company)
    merged.insert(0, "cik", cik)

    # Any required concept the company never reported becomes an all-null column, so
    # the row filter below treats it as missing rather than raising a KeyError.
    for name in REQUIRED_NAMES:
        if name not in merged.columns:
            merged[name] = pd.NA

    required_present = merged[list(REQUIRED_NAMES)].notna().sum(axis=1)
    threshold = len(REQUIRED_NAMES) if min_required is None else min_required
    merged = merged[required_present >= threshold]

    return merged.sort_values("fiscal_year").reset_index(drop=True)


def restatement_report(facts: dict[str, Any], concepts: Iterable[Concept] = CONCEPTS) -> pd.DataFrame:
    """Where a company's later filings disagree with what it first reported.

    A restatement is not proof of anything on its own, but material downward revisions to
    revenue or earnings are one of the strongest public signals that the original numbers
    were wrong. Returns one row per concept-year where the first and latest reported
    values differ by more than a rounding error.
    """
    rows: list[dict[str, Any]] = []
    for concept in concepts:
        collapsed = as_first_reported(concept_observations(facts, concept))
        if collapsed.empty:
            continue
        for _, row in collapsed.iterrows():
            original, latest = row["value"], row["latest_value"]
            if original in (0, None) or pd.isna(original) or pd.isna(latest):
                continue
            change = (latest - original) / abs(original)
            if abs(change) > 0.005:  # ignore sub-0.5% rounding and reclassification noise
                rows.append(
                    {
                        "cik": str(facts.get("cik", "")).zfill(10),
                        "company": facts.get("entityName", ""),
                        "concept": concept.name,
                        "fiscal_year": int(row["fiscal_year"]),
                        "first_reported": original,
                        "latest_reported": latest,
                        "pct_change": change,
                        "n_filings": int(row["n_filings"]),
                    }
                )
    return pd.DataFrame(rows)
